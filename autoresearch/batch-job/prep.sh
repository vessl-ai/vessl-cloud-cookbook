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
#   AUTORESEARCH_RESOURCE_SPEC  default: resourcespec-a100cpu  (CPU-only is enough)
#   AUTORESEARCH_IMAGE          default: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
#   AUTORESEARCH_REPO_URL       default: https://github.com/vessl-ai/vessl-cloud-cookbook.git
#   AUTORESEARCH_BRANCH         default: main
#
# Usage:
#   bash batch-job/prep.sh

set -euo pipefail

CACHE_VOLUME="${AUTORESEARCH_CACHE_VOLUME:?set AUTORESEARCH_CACHE_VOLUME to your cache volume slug}"
# Tokenizer training is single-threaded CPU work. A CPU-only spec is the
# right default — no point burning a GPU for ~5-10 minutes of idle.
RESOURCE_SPEC="${AUTORESEARCH_RESOURCE_SPEC:-resourcespec-a100cpu}"
IMAGE="${AUTORESEARCH_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
REPO_URL="${AUTORESEARCH_REPO_URL:-https://github.com/vessl-ai/vessl-cloud-cookbook.git}"
BRANCH="${AUTORESEARCH_BRANCH:-main}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_lib.sh
. "$SCRIPT_DIR/_lib.sh"

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

SLUG="$(find_job_slug "$JOB_NAME")" || { echo "prep.sh: failed to find job slug for $JOB_NAME" >&2; exit 3; }
echo "prep.sh: job slug $SLUG — polling state (this can take ~10-15 min)"

WAIT_RC=0
wait_for_job "$SLUG" || WAIT_RC=$?

echo "--- job logs ($SLUG) ---"
dump_job_logs "$SLUG"
echo "--- end job logs ---"

FINAL_STATE="$(job_state "$SLUG")"
echo "prep.sh: final state $FINAL_STATE (wait_rc=$WAIT_RC)"
[ "$FINAL_STATE" = "succeeded" ]
