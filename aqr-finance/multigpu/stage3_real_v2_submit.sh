#!/usr/bin/env bash
# stage3_real_v2_submit.sh — multi-GPU full-weight CPT of Qwen3.5-35B-A3B-Base
# on 8×H100 (single node) via axolotl + FSDP2.
#
# This is the multi-GPU full-weight arm of the aqr-finance recipe. It trains
# EVERY one of the 35 B parameters (no LoRA adapter) — something a single
# 80 GB H100 physically can't do (optimizer + weights alone exceed ~140 GB).
# FSDP2 FULL_SHARD across 8 GPUs is what makes it fit. The trained checkpoint
# is later evaluated with the parent recipe's `eval.py` (same leakage contract).
#
# The job writes its checkpoint to a PERSISTENT object volume so the merged
# checkpoint survives pod termination. The `--object-volume` flag below is a
# HARD prerequisite: without it, the container writes to ephemeral pod storage,
# which is lost when the pod terminates.
#
# Required env vars (set once per session, e.g. in your shell rc):
#   AQR_CACHE_VOLUME    slug of the object volume holding ~/.cache/aqr-finance
#                       (e.g. objvol-...). Create with `vesslctl volume create`.
#   AQR_RESOURCE_SPEC   resource spec slug for 8×H100 SXM single-node.
#                       Find with `vesslctl resource-spec list`.
#
# Optional env vars:
#   AQR_IMAGE           default: axolotlai/axolotl-uv:main-latest
#   AQR_TARGET_TOKENS   default: 1B   (point-in-time FineWeb budget for prep)
#   AQR_POLL_MAX        default: 2640 (30 s × 2640 ≈ 22 h hard polling ceiling)
#
# Cost envelope (measured — see ../multigpu/benchmarks.md):
#   - inline prep (skip-if-exists): ~1 h CPU on first run
#   - ~56k steps × ~1.06 s/step ≈ 18.6 h pure train
#   - mid-run saves + final merge ≈ ~2 h
#   ≈ ~$386 total at an 8×H100 hourly rate. Treat each run as expensive.
#
# Mid-run abort gates (auto, conservative — abort only on signals that almost
# certainly mean the run is broken; everything else prints a sanity snapshot):
#   GATE A (i=30, ~15 min): step 0/1 NaN/Inf/OOM in loss → auto-abort.
#   GATE B (i=180, ~1.5 h): first checkpoint not in logs yet → volume mount
#     likely broken → auto-abort.
#   GATE C (i=960, ~8 h): sanity printout of recent loss/mem only; review by eye.

set -euo pipefail

VOL="${AQR_CACHE_VOLUME:?set AQR_CACHE_VOLUME to your cache volume slug (e.g. objvol-...)}"
SPEC="${AQR_RESOURCE_SPEC:?set AQR_RESOURCE_SPEC to your 8xH100 SXM spec slug (vesslctl resource-spec list)}"
IMAGE="${AQR_IMAGE:-axolotlai/axolotl-uv:main-latest}"
TARGET_TOKENS="${AQR_TARGET_TOKENS:-1B}"
POLL_MAX="${AQR_POLL_MAX:-2640}"
JOB_NAME="aqr-fullweight-real-v2-$(date +%s)"
HERE="$(cd "$(dirname "$0")" && pwd)"

YAML_B64="$(base64 < "$HERE/qwen3-moe-fullweight-cpt.yaml" | tr -d '\n')"
PREP_B64="$(base64 < "$HERE/prepare_text.py" | tr -d '\n')"

JOB_CMD="set -e
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p /workspace && cd /workspace
echo ${YAML_B64} | base64 -d > qwen3-moe-fullweight-cpt.yaml
echo ${PREP_B64} | base64 -d > prepare_text.py
mkdir -p /root/.cache/aqr-finance/text
echo '=== env ==='
date -u
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c 'import torch,transformers,axolotl; print(\"torch\",torch.__version__,\"transformers\",transformers.__version__,\"axolotl\",getattr(axolotl,\"__version__\",\"?\"),\"gpus\",torch.cuda.device_count())'
echo '=== volume inventory ==='
ls -la /root/.cache/aqr-finance/ 2>&1 || true
echo '--- text/ ---'
ls -la /root/.cache/aqr-finance/text/ 2>&1 || true
echo '--- axolotl-fullweight-out/ (stale from prior run?) ---'
ls -la /root/.cache/aqr-finance/axolotl-fullweight-out/ 2>&1 || true
echo '=== end inventory ==='
JSONL=/root/.cache/aqr-finance/text/fineweb-pit-2017-06.jsonl
if [ -f \"\$JSONL\" ]; then
  echo \"prep: SKIP — JSONL exists, size \$(ls -lh \"\$JSONL\" | awk '{print \$5}')\"
else
  echo '=== prep ${TARGET_TOKENS} point-in-time FineWeb JSONL (inline, single-process) ==='
  date -u
  python prepare_text.py --target_tokens ${TARGET_TOKENS}
  date -u
  ls -lh /root/.cache/aqr-finance/text/
fi
echo '=== launch axolotl FSDP2 FULL-WEIGHT REAL RUN on 8 GPUs (volume-mounted) ==='
accelerate launch --num_processes 8 -m axolotl.cli.train qwen3-moe-fullweight-cpt.yaml"

echo "stage3-v2: creating 8xH100 job $JOB_NAME on $SPEC, volume=$VOL"
# NOTE: --object-volume is a HARD prerequisite (see header). Do not remove it.
vesslctl job create \
  -n "$JOB_NAME" \
  -r "$SPEC" \
  -i "$IMAGE" \
  --object-volume "${VOL}:/root/.cache/aqr-finance" \
  --tag aqr-fullweight-real-v2 \
  --cmd "$JOB_CMD"

# Slug discovery
SLUG=""
for _ in $(seq 1 15); do
  SLUG="$(vesslctl job list -o json 2>/dev/null \
    | python3 -c 'import json,sys;d=json.load(sys.stdin);n=sys.argv[1];print(next((j["slug"] for j in d if j["name"]==n),""))' "$JOB_NAME" 2>/dev/null || true)"
  [ -n "$SLUG" ] && break
  sleep 2
done
[ -z "$SLUG" ] && { echo "stage3-v2: failed to resolve slug"; exit 3; }
echo "stage3-v2: slug=$SLUG"
echo "stage3-v2: terminate anytime with: vesslctl job terminate $SLUG"

# Mid-run abort gates via logs grep.
prev=""; st=""; i=0
ABORTED=0
while [ "$i" -lt "$POLL_MAX" ]; do
  st="$(vesslctl job show "$SLUG" -o json 2>/dev/null \
    | python3 -c 'import json,sys;print(json.load(sys.stdin).get("workloadState",""))' 2>/dev/null || true)"
  [ "$st" != "$prev" ] && { echo "[$(date -u +%H:%M:%S)] state: $st (i=$i)"; prev="$st"; }
  case "$st" in succeeded|failed|terminated|cancelled) break ;; esac

  # ============== GATES ==============
  case "$i" in
    30|180|960)
      echo "=== GATE i=$i ($(date -u +%H:%M:%S)) ==="
      LOGS=$(vesslctl job logs --limit 80 "$SLUG" 2>&1 || echo "")

      # GATE A (any i): NaN/Inf/OOM in loss = auto-abort
      if echo "$LOGS" | grep -E -i "loss.*nan|loss.*inf|RuntimeError|CUDA out of memory" | grep -v "running\|skipping\|warmup" | head -3 | grep -q .; then
        echo "GATE FAIL: NaN/Inf/OOM signal detected. Aborting."
        echo "--- suspect lines ---"
        echo "$LOGS" | grep -E -i "loss.*nan|loss.*inf|RuntimeError|CUDA out of memory" | head -5
        vesslctl job terminate "$SLUG" || true
        ABORTED=1; break
      fi

      # GATE B (only at i=180, 1.5h): first save must have appeared
      if [ "$i" = "180" ]; then
        if ! echo "$LOGS" | grep -E -i "saving model checkpoint|checkpoint-[0-9]+|save_strategy" | head -1 | grep -q .; then
          echo "GATE B FAIL: no save activity by 1.5h. Volume mount likely broken. Aborting."
          echo "--- last 20 log lines ---"
          echo "$LOGS" | tail -20
          vesslctl job terminate "$SLUG" || true
          ABORTED=1; break
        fi
        echo "GATE B PASS: save activity present"
      fi

      # GATE C (only at i=960, 8h): sanity printout only
      if [ "$i" = "960" ]; then
        echo "GATE C sanity printout (no auto-abort):"
        echo "$LOGS" | grep -E "loss|max_allocated|epoch" | tail -5
      fi

      echo "=== GATE i=$i PASS ==="
      ;;
  esac

  i=$((i+1)); sleep 30
done

echo "--- stage3-v2 final logs ($SLUG) ---"
vesslctl job logs --limit 1000 "$SLUG" 2>&1 || true
echo "--- end logs (final state: $st, aborted=$ABORTED) ---"
[ "$st" = "succeeded" ] && echo "STAGE 3 v2: PASS (checkpoint should be at /root/.cache/aqr-finance/axolotl-fullweight-out on volume $VOL)" \
                       || echo "STAGE 3 v2: $st — inspect logs above. ABORTED=$ABORTED"
