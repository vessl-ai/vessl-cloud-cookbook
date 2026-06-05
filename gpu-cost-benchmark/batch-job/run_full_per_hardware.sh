#!/usr/bin/env bash
#
# To run on a different hardware target, just change the HW arg: ./run_full_per_hardware.sh H100 (or A100 | B200).
#
# run_full_per_hardware.sh — full training + inference sweep for ONE hardware target.
#
# Runs the LoRA cost-benchmark matrix on an 8-GPU node for a single GPU type and
# writes per-run summary.json / loss_log.jsonl plus inference result JSONs under
# $OUTPUT_BASE/runs/. Cost is reported downstream as a RELATIVE index only
# (A100 = 1.00); no absolute $/GPU-hour values are produced here.
#
# Training cells (matrix):
#   T-{HW}-{QUANT}        default training (no gradient checkpointing, mbsz=auto)
#   T-{HW}-{QUANT}-gc     gradient checkpointing ON ("realistic" production config)
#   T-{HW}-bf16-baseline  bf16 ablation (only on H100/B200 — isolates the FP8 effect;
#                         on A100 the default already IS bf16)
#
# Inference cells (using the default cell's LoRA adapter):
#   I1  LoRA=OFF MTP=OFF   base ceiling
#   I2  LoRA=OFF MTP=ON    base + speculative decoding
#   I3  LoRA=ON  MTP=OFF   LoRA serving overhead
#   I4  LoRA=ON  MTP=ON    MTP x LoRA
# each swept over per-instance concurrency {1, 16, 64, 256}.
#
# QUANT auto-selected by hardware: A100=bf16, H100/B200=fp8.
#
# Args:
#   $1  HARDWARE  (A100 | H100 | B200; default B200)
#
# Env vars:
#   OUTPUT_BASE          Base output dir (object volume). Default: /shared
#   DATASET              Packed dataset path. Default: /shared/datasets/sharegpt_packed_2048
#   WORKDIR              Dir holding train_sft.py / serve_and_bench.sh.
#                        Default: this script's directory.
#   NPROC                GPUs per node for torchrun. Default: 8
#   MASTER_PORT          torchrun master port. Default: 29500
#   MAX_STEPS            Training steps per cell. Default: 220
#   WARMUP_STEPS         Default: 20
#   SKIP_DEFAULT_T=1     skip the default training cell
#   SKIP_GC=1            skip the gradient-checkpoint variant
#   SKIP_BF16_BASELINE=1 skip the bf16 ablation cell (H100/B200 only)
#   SKIP_INFERENCE=1     run only the training cells (alias: ONLY_T=1)
#   HF_HOME              Persistent HF cache. If unset, falls back to
#                        ~/.cache/huggingface (slow first-time model download).
#   WANDB_PROJECT        Optional — if set, training cells log to W&B.
#
# Expected infra:
#   * 8-GPU node (A100 / H100 / B200 SXM, 80 GB class).
#   * Object storage volume mounted at /shared (dataset in, results out);
#     workspace/clone at /root.
#
# Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.

set -euo pipefail

HARDWARE="${1:-B200}"
case "$HARDWARE" in
    A100) PRIMARY_QUANT="bf16"; CAN_FP8=0 ;;
    H100|B200) PRIMARY_QUANT="fp8"; CAN_FP8=1 ;;
    *) echo "ERROR: unknown HARDWARE=$HARDWARE (expected A100|H100|B200)"; exit 1 ;;
esac

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
OUTPUT_BASE="${OUTPUT_BASE:-/shared}"
DATASET="${DATASET:-/shared/datasets/sharegpt_packed_2048}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
MAX_STEPS="${MAX_STEPS:-220}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"

export OUTPUT_BASE
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${HF_HOME:-}" ]]; then
    echo "WARNING: HF_HOME is unset — falling back to ~/.cache/huggingface" >&2
fi

SCRIPT_TRAIN="$WORKDIR/train_sft.py"
SCRIPT_INFER="$WORKDIR/serve_and_bench.sh"
RUNDIR_PRIMARY="${OUTPUT_BASE}/runs/${HARDWARE}_${PRIMARY_QUANT}"
LOGDIR="${LOGDIR:-${WORKDIR}/logs}"
mkdir -p "$LOGDIR" "$OUTPUT_BASE"

if [[ ! -d "$DATASET" ]]; then
    echo "ERROR: packed dataset not found at $DATASET" >&2
    echo "       Build it with your data-prep step (see ../data_prep/) and place" >&2
    echo "       it on the object volume, or set DATASET=<path>." >&2
    exit 1
fi

PYTHON="${PYTHON:-python}"

run_train_cell () {
    local quant="$1" variant="$2" extra_flag="${3:-}"
    local tag="T-${HARDWARE}-${quant}${variant:+-$variant}"
    local stamp; stamp=$(date -u +%Y%m%dT%H%M%S)
    local log="$LOGDIR/sft_${HARDWARE}_${quant}${variant:+_$variant}_${stamp}.log"
    echo "============================================================"
    echo ">>> ${tag}  quant=${quant} variant=${variant:-(none)} flag=${extra_flag}"
    echo ">>> log: $log"
    echo "============================================================"
    set -x
    torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
        "$SCRIPT_TRAIN" \
        --variant ddp \
        --hardware "$HARDWARE" --quant "$quant" --mbsz auto \
        --dataset "$DATASET" \
        --seq-len 2048 --max-steps "$MAX_STEPS" --warmup-steps "$WARMUP_STEPS" \
        $extra_flag \
        2>&1 | tee "$log"
    set +x
}

# 1. Default training
if [[ "${SKIP_DEFAULT_T:-0}" != "1" ]]; then
    run_train_cell "$PRIMARY_QUANT" ""
else
    echo ">>> SKIP_DEFAULT_T=1, skipping default training cell"
fi

# 2. GC variant (gradient checkpointing ON)
if [[ "${SKIP_GC:-0}" != "1" ]]; then
    run_train_cell "$PRIMARY_QUANT" "gc" "--grad-checkpoint"
fi

# 3. bf16 ablation (only if HW supports fp8 — otherwise default IS bf16)
if [[ "$CAN_FP8" == "1" && "${SKIP_BF16_BASELINE:-0}" != "1" ]]; then
    run_train_cell "bf16" "baseline"
fi

# 4. Inference cells (use the default training cell's LoRA adapter)
if [[ "${SKIP_INFERENCE:-${ONLY_T:-0}}" != "1" ]]; then
    LORA_PATH=""
    candidate="${RUNDIR_PRIMARY}/T-${HARDWARE}/checkpoint-final"
    if [[ -d "$candidate" ]]; then
        LORA_PATH="$(realpath "$candidate")"
    else
        echo "WARN: LoRA checkpoint not found at $candidate; running I1/I2 only"
    fi
    stamp=$(date -u +%Y%m%dT%H%M%S)

    run_infer_cell () {
        local cell_id="$1" lora_flag="$2" mtp_flag="$3"
        local cell_out="${RUNDIR_PRIMARY}/${cell_id}"
        local lora_env="" mtp_env="0"
        [[ "$lora_flag" == "ON" ]] && lora_env="$LORA_PATH"
        [[ "$mtp_flag" == "ON" ]] && mtp_env="1"
        local log="$LOGDIR/infer_${HARDWARE}_${cell_id}_${stamp}.log"
        echo "============================================================"
        echo ">>> CELL ${cell_id}  HARDWARE=${HARDWARE} LoRA=${lora_flag} MTP=${mtp_flag}"
        echo ">>> log: $log"
        echo "============================================================"
        HARDWARE="$HARDWARE" \
        MTP_ON="$mtp_env" \
        LORA_PATH="$lora_env" \
        OUT_DIR="$cell_out" \
        TAG_PREFIX="${cell_id}" \
        bash "$SCRIPT_INFER" 2>&1 | tee "$log" \
            || echo "!!! inference cell ${cell_id} failed; continuing"
    }

    run_infer_cell "I1-${HARDWARE}" "OFF" "OFF"
    run_infer_cell "I2-${HARDWARE}" "OFF" "ON"
    if [[ -n "$LORA_PATH" ]]; then
        run_infer_cell "I3-${HARDWARE}" "ON" "OFF"
        run_infer_cell "I4-${HARDWARE}" "ON" "ON"
    else
        echo "(skipping I3/I4: no LORA_PATH available)"
    fi
fi

echo "============================================================"
echo ">>> ALL CELLS DONE for ${HARDWARE}"
echo "============================================================"
ls -la "${OUTPUT_BASE}/runs/${HARDWARE}_"* 2>/dev/null | head -20 || true
