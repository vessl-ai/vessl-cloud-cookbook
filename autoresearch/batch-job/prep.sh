#!/usr/bin/env bash
# prep.sh — one-time data prep job. Populates the cache volume with the
# ClimbMix shards and trained BPE tokenizer that every train.py run reads.
#
# Run this once after creating your AUTORESEARCH_CACHE_VOLUME. After it
# succeeds, every batch-job/submit.sh invocation skips prepare.py because
# the tokenizer file already exists on the volume.
#
# Required env vars:
#   AUTORESEARCH_CACHE_VOLUME   slug of the empty object volume to seed.
#
# Optional env vars (same defaults as submit.sh):
#   AUTORESEARCH_RESOURCE_SPEC, AUTORESEARCH_IMAGE, AUTORESEARCH_REPO_URL,
#   AUTORESEARCH_BRANCH         default: main (the branch to clone for prepare.py)
#
# Usage:
#   bash batch-job/prep.sh

set -euo pipefail

CACHE_VOLUME="${AUTORESEARCH_CACHE_VOLUME:?set AUTORESEARCH_CACHE_VOLUME to your cache volume slug}"
# Tokenizer training is single-threaded CPU-bound work — burning a GPU on it
# is wasteful. Use a CPU-only spec for prep if you have one; otherwise the
# default A100x1 is fine, just a bit overkill for ~5 minutes.
RESOURCE_SPEC="${AUTORESEARCH_RESOURCE_SPEC:-resourcespec-a100x1}"
IMAGE="${AUTORESEARCH_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
REPO_URL="${AUTORESEARCH_REPO_URL:-https://github.com/vessl-ai/vessl-cloud-cookbook.git}"
BRANCH="${AUTORESEARCH_BRANCH:-main}"

JOB_NAME="autoresearch-prep-$(date +%s)"

JOB_CMD=$(cat <<EOF
set -e
apt-get update -qq && apt-get install -y -qq git curl
mkdir -p /workspace && cd /workspace
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" .
cd autoresearch
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
mkdir -p "\$HOME/.cache/autoresearch"
uv run prepare.py
ls -la "\$HOME/.cache/autoresearch"
ls -la "\$HOME/.cache/autoresearch/data" | head -20
ls -la "\$HOME/.cache/autoresearch/tokenizer"
EOF
)

echo "prep.sh: creating job $JOB_NAME on $RESOURCE_SPEC"
vesslctl job create \
  -n "$JOB_NAME" \
  -r "$RESOURCE_SPEC" \
  -i "$IMAGE" \
  --object-volume "${CACHE_VOLUME}:/root/.cache/autoresearch" \
  --tag autoresearch \
  --tag autoresearch-prep \
  --cmd "$JOB_CMD"

SLUG=""
for _ in $(seq 1 10); do
  SLUG="$(vesslctl job list -o json 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); n=sys.argv[1]; print(next((j["slug"] for j in d if j["name"]==n), ""))' "$JOB_NAME" || true)"
  [ -n "$SLUG" ] && break
  sleep 2
done
[ -n "$SLUG" ] || { echo "prep.sh: failed to find job slug for $JOB_NAME" >&2; exit 3; }
echo "prep.sh: job slug $SLUG — streaming logs (this can take ~5-15 min)"

vesslctl job logs -f "$SLUG"

STATE="$(vesslctl job show "$SLUG" -o json 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' || true)"
echo "prep.sh: final state $STATE"
case "$STATE" in
  succeeded) exit 0 ;;
  *) exit 1 ;;
esac
