#!/usr/bin/env bash
#
# serve_and_bench.sh — launch vLLM instances, wait for readiness, run the
# benchmark client across them, then tear the servers down.
#
# Spawns NUM_INSTANCES = 8 / TP_SIZE vLLM serve processes on the node's 8 GPUs
# (one process per GPU when TP_SIZE=1; one per GPU-group otherwise), polls
# /health until all are ready, runs bench_client.py at each requested
# concurrency, and SIGKILLs the servers at the end.
#
# Env vars (all optional unless noted):
#   HARDWARE         A100|H100|B200 (default: B200). Picks the default MODEL_ID.
#   MODEL_ID         HF model id to serve. Auto-defaulted by HARDWARE if unset:
#                      A100      -> google/gemma-4-31B-it (bf16)
#                      H100|B200 -> RedHatAI/gemma-4-31B-it-FP8-block
#   TP_SIZE          Tensor-parallel size per instance (default 1).
#                    TP=1 -> 8 single-GPU instances (best aggregate TPS).
#                    TP=2 -> 4 instances on GPU pairs (0+1, 2+3, ...).
#                    NUM_INSTANCES is derived as 8/TP_SIZE and exported for the client.
#   MTP_ON           "1" to enable an MTP speculative-decoding draft (default 0).
#   MTP_GAMMA        num_speculative_tokens for MTP (default 1; raise to 2/3 to
#                    test higher-speculation throughput; requires MTP_ON=1).
#   MTP_DRAFT_MODEL  Draft model id for MTP (default: google/gemma-4-31B-it-assistant).
#   LORA_PATH        Non-empty -> hot-load a LoRA adapter; vLLM alias = "sft-lora".
#                    When set, requests are routed to "sft-lora" automatically.
#   MAX_MODEL_LEN    default 4096
#   MAX_NUM_SEQS     default 64
#   MAX_NUM_BATCHED_TOKENS  default 4096 (MM-capable bf16 models need >= the
#                    per-mm-item token budget even with images zeroed out).
#   ATTN_BACKEND     Attention backend override (default "" = vLLM auto).
#                    e.g. FLASHINFER for FlashInfer decode attention.
#   KV_CACHE_DTYPE   KV cache quantization (default "auto").
#                    Values: auto | fp8 | fp8_e4m3 | fp8_e5m2 | fp8_inc | nvfp4
#   CHUNKED_PREFILL  "1" to enable chunked prefill (pair with a larger
#                    MAX_NUM_BATCHED_TOKENS for throughput gains).
#   BASE_PORT        First API port (default 8000); use 8100 to avoid conflicts.
#   CONCURRENCIES    Space-separated per-instance concurrency sweep
#                    (default "1 16 64 256").
#   REQUEST_MODEL    Override the model name sent in requests (default: auto —
#                    "sft-lora" when LORA_PATH is set, else MODEL_ID).
#   TAG_PREFIX       Tag prefix for result files (default: "I-${HARDWARE}").
#   OUT_DIR          Where bench_client.py writes <tag>.json results.
#                    Default: ${OUTPUT_BASE:-/shared}/runs/${HARDWARE}_inference
#   SUBSET           Prompt subset JSON (default /shared/datasets/bench_subset.json).
#   WORKDIR          Directory containing bench_client.py (default: this script's dir).
#   INDUCTOR_CACHE_BASE  torch.compile cache base. Default: ${WORKDIR}/.cache/torch_inductor
#                    (must be on an exec-allowed mount; /tmp and /dev/shm are
#                    often noexec).
#   HF_HOME          HF cache (default ${HOME}/.cache/huggingface). Point at a
#                    persistent path to avoid re-downloading the model.
#
# Expected infra:
#   * 8-GPU node (A100 / H100 / B200 SXM, 80 GB class).
#   * Object volume mounted at /shared for the prompt subset and results;
#     workspace/clone at /root.
#   * vllm, curl, and python with aiohttp available on PATH.
#
# Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.

set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"

HARDWARE="${HARDWARE:-B200}"

# HW-aware default model
if [[ -z "${MODEL_ID:-}" ]]; then
    case "$HARDWARE" in
        A100) MODEL_ID="google/gemma-4-31B-it" ;;
        H100|B200) MODEL_ID="RedHatAI/gemma-4-31B-it-FP8-block" ;;
        *) echo "ERROR: unknown HARDWARE=$HARDWARE"; exit 1 ;;
    esac
fi

MTP_ON="${MTP_ON:-0}"
MTP_GAMMA="${MTP_GAMMA:-1}"
MTP_DRAFT_MODEL="${MTP_DRAFT_MODEL:-google/gemma-4-31B-it-assistant}"
LORA_PATH="${LORA_PATH:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
TP_SIZE="${TP_SIZE:-1}"
ATTN_BACKEND="${ATTN_BACKEND:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-0}"
BASE_PORT="${BASE_PORT:-8000}"
INDUCTOR_CACHE_BASE="${INDUCTOR_CACHE_BASE:-$WORKDIR/.cache/torch_inductor}"
SUBSET="${SUBSET:-/shared/datasets/bench_subset.json}"
OUTPUT_BASE="${OUTPUT_BASE:-/shared}"
OUT_DIR="${OUT_DIR:-${OUTPUT_BASE}/runs/${HARDWARE}_inference}"
TAG_PREFIX="${TAG_PREFIX:-I-${HARDWARE}}"
read -r -a CONCURRENCIES <<< "${CONCURRENCIES:-1 16 64 256}"

# Derived: number of instances = 8 / TP_SIZE (always 8 GPUs total)
NUM_INSTANCES=$(( 8 / TP_SIZE ))
export NUM_INSTANCES   # inherited by bench_client.py
export BASE_PORT

# Request-time model name: LoRA alias when an adapter is loaded, else base id.
if [[ -n "${REQUEST_MODEL:-}" ]]; then
    : # explicit override
elif [[ -n "$LORA_PATH" ]]; then
    REQUEST_MODEL="sft-lora"
else
    REQUEST_MODEL="$MODEL_ID"
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1

LOGDIR="${LOGDIR:-${WORKDIR}/logs/serve}"
mkdir -p "$LOGDIR" "$INDUCTOR_CACHE_BASE" "$OUT_DIR"

LORA_ARGS=()
if [[ -n "$LORA_PATH" ]]; then
    LORA_ARGS=(
        --enable-lora
        --lora-modules "sft-lora=${LORA_PATH}"
        --max-lora-rank 16
    )
fi

MTP_ARGS=()
if [[ "$MTP_ON" == "1" ]]; then
    MTP_ARGS=(
        --speculative-config
        "{\"method\":\"mtp\",\"model\":\"${MTP_DRAFT_MODEL}\",\"num_speculative_tokens\":${MTP_GAMMA}}"
    )
fi

ATTN_ARGS=()
if [[ -n "$ATTN_BACKEND" ]]; then
    ATTN_ARGS=(--attention-backend "$ATTN_BACKEND")
fi

CACHE_ARGS=()
if [[ "$KV_CACHE_DTYPE" != "auto" ]]; then
    CACHE_ARGS=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

CP_ARGS=()
if [[ "$CHUNKED_PREFILL" == "1" ]]; then
    CP_ARGS=(--enable-chunked-prefill)
fi

shutdown_all () {
    echo ">>> [shutdown] SIGKILL vllm api servers"
    pkill -KILL -f "vllm serve" 2>/dev/null || true
    sleep 5
    echo ">>> [shutdown] killing remaining engine children by PID"
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    for p in $PIDS; do kill -KILL "$p" 2>/dev/null || true; done
    sleep 5
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null || true
}
trap shutdown_all EXIT

echo ">>> serve MODEL=$MODEL_ID MTP=$MTP_ON GAMMA=$MTP_GAMMA TP=$TP_SIZE ATTN=${ATTN_BACKEND:-auto} KV=${KV_CACHE_DTYPE} CP=${CHUNKED_PREFILL} LORA=${LORA_PATH:-OFF} NUM_INSTANCES=$NUM_INSTANCES"

# ── launch instances ─────────────────────────────────────────────────────────
for inst in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + inst))
    VLLM_INTERNAL_PORT=$((50000 + inst * 100))
    GPU_START=$((inst * TP_SIZE))
    GPUS=$(seq -s, "$GPU_START" $((GPU_START + TP_SIZE - 1)))
    INDUCTOR_DIR="$INDUCTOR_CACHE_BASE/inst${inst}"
    mkdir -p "$INDUCTOR_DIR"
    LOG="$LOGDIR/instance_${inst}.log"
    echo ">>> GPU(s) $GPUS -> api $PORT, internal $VLLM_INTERNAL_PORT, log $LOG"
    CUDA_VISIBLE_DEVICES="$GPUS" \
    TORCHINDUCTOR_CACHE_DIR="$INDUCTOR_DIR" \
    VLLM_PORT="$VLLM_INTERNAL_PORT" \
    nohup vllm serve "$MODEL_ID" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{"image":0,"audio":0}' \
        "${MTP_ARGS[@]}" \
        "${LORA_ARGS[@]}" \
        "${ATTN_ARGS[@]}" \
        "${CACHE_ARGS[@]}" \
        "${CP_ARGS[@]}" \
        --port "$PORT" --host 0.0.0.0 --trust-remote-code \
        > "$LOG" 2>&1 &
    echo "    pid=$! port=$PORT gpus=$GPUS"
done

# ── wait for readiness (poll /health, scan logs for fatal lines) ──────────────
PORTS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do PORTS+=($((BASE_PORT + i))); done
DEADLINE=$((SECONDS + 1200))   # 20 min
echo ">>> [wait] deadline=20m NUM_INSTANCES=$NUM_INSTANCES"
while true; do
    ready=0
    for p in "${PORTS[@]}"; do
        if curl -fsS "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
            ready=$((ready + 1))
        fi
    done
    if [[ $ready -eq $NUM_INSTANCES ]]; then
        echo ">>> [wait] all $NUM_INSTANCES ready at t=$SECONDS"
        break
    fi
    scan_logs=()
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        f="$LOGDIR/instance_${i}.log"
        [[ -f "$f" ]] && scan_logs+=("$f")
    done
    if [[ ${#scan_logs[@]} -gt 0 ]]; then
        fatal_out=$(grep -mE5 "RuntimeError|CUDA error|out of memory|ImportError|FATAL" "${scan_logs[@]}" 2>/dev/null || true)
        if [[ -n "$fatal_out" ]]; then
            echo "$fatal_out"
            echo ">>> [wait] FATAL in serve logs (ready=$ready/$NUM_INSTANCES at t=$SECONDS)"
            exit 1
        fi
    fi
    if [[ $SECONDS -gt $DEADLINE ]]; then
        echo ">>> [wait] TIMEOUT after 20m (ready=$ready/$NUM_INSTANCES)"
        exit 1
    fi
    echo "  ready=$ready/$NUM_INSTANCES (t=$SECONDS)"
    sleep 20
done

# ── benchmark across concurrencies ───────────────────────────────────────────
PYTHON="${PYTHON:-python}"
for c in "${CONCURRENCIES[@]}"; do
    TAG=$(printf "%s__c%04d" "$TAG_PREFIX" "$c")
    echo ">>> [bench] ${TAG} (concurrency/instance=$c, model=$REQUEST_MODEL)"
    "$PYTHON" "$WORKDIR/bench_client.py" \
        --model "$REQUEST_MODEL" \
        --concurrency "$c" \
        --tag "$TAG" \
        --subset "$SUBSET" \
        --outdir "$OUT_DIR" \
        || echo "!!! bench failed for ${TAG}, continuing"
done

echo ">>> DONE. results in $OUT_DIR (servers torn down on exit)"
