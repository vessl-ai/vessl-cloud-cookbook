#!/usr/bin/env bash
# volume_inspect_submit.sh — cheap CPU probe to inspect the aqr-finance object
# volume contents. Use it to confirm that a full-weight run actually wrote its
# merged checkpoint to the PERSISTENT volume (not ephemeral pod storage).
#
# This is the diagnostic that catches the "$329 mistake": if a submit script
# forgets the `--object-volume` flag, the run writes to ephemeral storage and
# the checkpoint is gone on pod terminate. This probe mounts the volume and
# lists the paths that *should* contain the merged checkpoint, the baseline
# LoRA adapter, the raw point-in-time text, etc.
#
# Required env vars:
#   AQR_CACHE_VOLUME    slug of the object volume holding ~/.cache/aqr-finance.
#   AQR_CPU_SPEC        a cheap CPU-only resource spec slug, ideally on the same
#                       cluster as the volume. Find with
#                       `vesslctl resource-spec list`.
#
# Optional:
#   AQR_INSPECT_IMAGE   default: alpine:3.20

set -euo pipefail

VOL="${AQR_CACHE_VOLUME:?set AQR_CACHE_VOLUME to your cache volume slug (e.g. objvol-...)}"
SPEC="${AQR_CPU_SPEC:?set AQR_CPU_SPEC to a cheap CPU-only spec slug (vesslctl resource-spec list)}"
IMAGE="${AQR_INSPECT_IMAGE:-alpine:3.20}"
JOB_NAME="aqr-volume-inspect-$(date +%s)"

JOB_CMD=$(cat <<'EOF'
set -e
echo '=== volume root ==='
ls -la /root/.cache/aqr-finance/ 2>/dev/null || echo 'volume root not present'
echo
echo '=== axolotl-fullweight-out (full-weight run expected target) ==='
ls -la /root/.cache/aqr-finance/axolotl-fullweight-out/ 2>/dev/null \
  || echo '!! axolotl-fullweight-out NOT on volume (run wrote ephemeral only?)'
echo
echo '=== merged subdir ==='
ls -la /root/.cache/aqr-finance/axolotl-fullweight-out/merged/ 2>/dev/null \
  || echo '!! merged/ NOT on volume'
echo
echo '=== adapter (baseline LoRA, expected present) ==='
ls -la /root/.cache/aqr-finance/adapter/ 2>/dev/null | head -10 \
  || echo '!! adapter NOT on volume'
echo
echo '=== text (fineweb point-in-time JSONL, expected present from prep run) ==='
ls -la /root/.cache/aqr-finance/text/ 2>/dev/null | head -5 \
  || echo '!! text NOT on volume'
echo
echo '=== checkpoint markers if anywhere on volume ==='
find /root/.cache/aqr-finance -maxdepth 4 -name 'config.json' -o -name 'pytorch_model*' -o -name 'model*.safetensors' 2>/dev/null | head -20
echo '=== done ==='
EOF
)

echo "volume-inspect: creating $JOB_NAME on $SPEC, volume=$VOL"
vesslctl job create \
  -n "$JOB_NAME" \
  -r "$SPEC" \
  -i "$IMAGE" \
  --object-volume "${VOL}:/root/.cache/aqr-finance" \
  --tag aqr-volume-inspect \
  --cmd "$JOB_CMD" >&2

SLUG=""
for _ in $(seq 1 10); do
  SLUG="$(vesslctl job list 2>&1 | grep "$JOB_NAME" | awk '{print $1}' | head -1)"
  [ -n "$SLUG" ] && break
  sleep 2
done
[ -z "$SLUG" ] && { echo "volume-inspect: failed to resolve slug"; exit 1; }
echo "volume-inspect: slug=$SLUG"

prev=""
for i in $(seq 1 30); do
  state=$(vesslctl job show "$SLUG" 2>&1 | grep '^State:' | awk '{print $2}' || true)
  if [ -n "$state" ] && [ "$state" != "$prev" ]; then
    echo "[$(date -u +%H:%M:%S)] state: $state"
    prev="$state"
  fi
  case "$state" in
    succeeded) echo "volume-inspect: SUCCEEDED"; break ;;
    failed|terminated|cancelled) echo "volume-inspect: terminal=$state"; break ;;
  esac
  sleep 15
done

echo "--- logs ---"
vesslctl job logs --limit 200 "$SLUG" 2>&1 || true
echo "--- end logs ---"
echo "volume-inspect: final=$state slug=$SLUG"
