#!/usr/bin/env bash
# submit.sh — run one autoresearch experiment as a VESSL Cloud batch job.
#
# Called by the autoresearch agent loop in place of `uv run train.py`.
# Pushes the current autoresearch/<tag> branch to origin, submits a vesslctl
# job that clones the cookbook at that branch and runs train.py on a single
# GPU, then streams logs to stdout until the job terminates.
#
# Prereqs:
#   - vesslctl installed and authenticated (org=Lidia, team=Floyd or your own).
#   - Object volume populated by batch-job/prep.sh (data + tokenizer cached).
#   - Current branch matches autoresearch/* and the working tree is clean.
#
# Required env vars (set once per session, e.g. in your shell rc):
#   AUTORESEARCH_CACHE_VOLUME   slug of the object volume holding ~/.cache/autoresearch
#                               (e.g. objvol-abc123).
#
# Optional env vars:
#   AUTORESEARCH_RESOURCE_SPEC  default: resourcespec-a100x1
#   AUTORESEARCH_IMAGE          default: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
#   AUTORESEARCH_REPO_URL       default: https://github.com/vessl-ai/vessl-cloud-cookbook.git
#   AUTORESEARCH_TIMEOUT_S      default: 1200 (kill the wait after this many seconds)
#
# Usage (from the agent loop):
#   bash batch-job/submit.sh > run.log 2>&1
#   grep "^val_bpb:\|^peak_vram_mb:" run.log

set -euo pipefail

CACHE_VOLUME="${AUTORESEARCH_CACHE_VOLUME:?set AUTORESEARCH_CACHE_VOLUME to your cache volume slug (e.g. objvol-...)}"
RESOURCE_SPEC="${AUTORESEARCH_RESOURCE_SPEC:-resourcespec-a100x1}"
IMAGE="${AUTORESEARCH_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
REPO_URL="${AUTORESEARCH_REPO_URL:-https://github.com/vessl-ai/vessl-cloud-cookbook.git}"
TIMEOUT_S="${AUTORESEARCH_TIMEOUT_S:-1200}"

# Resolve repo root so this script works whether called from the recipe dir or batch-job/.
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
case "$BRANCH" in
  autoresearch/*) ;;
  *) echo "submit.sh: must be on an autoresearch/<tag> branch (current: $BRANCH)" >&2; exit 2 ;;
esac
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "submit.sh: working tree is dirty — commit your train.py edits first" >&2
  exit 2
fi

COMMIT="$(git rev-parse --short HEAD)"
TAG="${BRANCH#autoresearch/}"
JOB_NAME="autoresearch-${TAG}-${COMMIT}"

echo "submit.sh: pushing $BRANCH ($COMMIT) to origin"
git push --force-with-lease -u origin "$BRANCH" >&2

# Job command runs inside the container. Idempotent: prepare.py is a no-op if
# data + tokenizer are already cached on the mounted volume.
JOB_CMD=$(cat <<EOF
set -e
apt-get update -qq && apt-get install -y -qq git curl
mkdir -p /workspace && cd /workspace
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" .
cd autoresearch
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
mkdir -p "\$HOME/.cache/autoresearch"
if [ ! -f "\$HOME/.cache/autoresearch/tokenizer/tokenizer.pkl" ]; then
  echo "submit: cache empty, running prepare.py"
  uv run prepare.py
fi
uv run train.py
EOF
)

echo "submit.sh: creating job $JOB_NAME on $RESOURCE_SPEC"
vesslctl job create \
  -n "$JOB_NAME" \
  -r "$RESOURCE_SPEC" \
  -i "$IMAGE" \
  --object-volume "${CACHE_VOLUME}:/root/.cache/autoresearch" \
  --tag autoresearch \
  --tag "tag:${TAG}" \
  --cmd "$JOB_CMD" >&2

# vesslctl job create doesn't return the slug — look it up by name.
SLUG=""
for _ in $(seq 1 10); do
  SLUG="$(vesslctl job list -o json 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); n=sys.argv[1]; print(next((j["slug"] for j in d if j["name"]==n), ""))' "$JOB_NAME" || true)"
  [ -n "$SLUG" ] && break
  sleep 2
done
if [ -z "$SLUG" ]; then
  echo "submit.sh: failed to locate job slug for $JOB_NAME" >&2
  exit 3
fi
echo "submit.sh: job slug $SLUG"

# Follow logs with a hard timeout so a stuck job doesn't block the agent loop.
# `job logs -f` exits when the job reaches a terminal state.
( vesslctl job logs -f "$SLUG" ) &
LOGS_PID=$!
( sleep "$TIMEOUT_S"; kill -TERM $LOGS_PID 2>/dev/null || true ) &
WATCHER_PID=$!
wait "$LOGS_PID" || true
kill "$WATCHER_PID" 2>/dev/null || true

# Final status check — the train.py exit code propagates through the job state.
STATE="$(vesslctl job show "$SLUG" -o json 2>/dev/null \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))' || true)"
echo "submit.sh: final state $STATE"
case "$STATE" in
  succeeded) exit 0 ;;
  *) exit 1 ;;
esac
