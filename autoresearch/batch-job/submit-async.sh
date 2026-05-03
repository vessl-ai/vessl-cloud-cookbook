#!/usr/bin/env bash
# submit-async.sh — submit one experiment job and return immediately.
#
# Like submit.sh, but does NOT wait for the job to finish. Prints just the
# job slug (e.g. job-abc123) to stdout so the caller can collect K slugs
# from K invocations and pass them all to wait-jobs.sh later. Verbose
# progress goes to stderr so it doesn't pollute the captured slug.
#
# Use this for "Mode B: parallel batch" runs where one agent fans out K
# candidate experiments at once. For the classic linear keep-or-revert
# loop, use submit.sh instead — it's the same logic but blocks on the job.
#
# Same prereqs and env vars as submit.sh:
#   AUTORESEARCH_CACHE_VOLUME (required)
#   AUTORESEARCH_RESOURCE_SPEC, AUTORESEARCH_IMAGE, AUTORESEARCH_REPO_URL
#
# Usage:
#   slug_a=$(bash batch-job/submit-async.sh)
#   git checkout autoresearch/<tag>-cand-b
#   slug_b=$(bash batch-job/submit-async.sh)
#   bash batch-job/wait-jobs.sh "$slug_a" "$slug_b"

set -euo pipefail

CACHE_VOLUME="${AUTORESEARCH_CACHE_VOLUME:?set AUTORESEARCH_CACHE_VOLUME to your cache volume slug (e.g. objvol-...)}"
RESOURCE_SPEC="${AUTORESEARCH_RESOURCE_SPEC:-resourcespec-5qp3iq5lcd90}"
IMAGE="${AUTORESEARCH_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
REPO_URL="${AUTORESEARCH_REPO_URL:-https://github.com/vessl-ai/vessl-cloud-cookbook.git}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_lib.sh
. "$SCRIPT_DIR/_lib.sh"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
case "$BRANCH" in
  autoresearch/*) ;;
  *) echo "submit-async.sh: must be on an autoresearch/<tag> branch (current: $BRANCH)" >&2; exit 2 ;;
esac
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "submit-async.sh: working tree is dirty — commit your train.py edits first" >&2
  exit 2
fi

COMMIT="$(git rev-parse --short HEAD)"
TAG="${BRANCH#autoresearch/}"
TAG_SAFE="$(printf '%s' "$TAG" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | sed 's/^-*//;s/-*$//')"
JOB_NAME="autoresearch-${TAG_SAFE}-${COMMIT}"

echo "submit-async.sh: pushing $BRANCH ($COMMIT) to origin" >&2
git push --force-with-lease -u origin "$BRANCH" >&2

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
  echo "submit-async: cache empty, running prepare.py"
  uv run prepare.py
fi
uv run train.py
EOF
)

echo "submit-async.sh: creating job $JOB_NAME on $RESOURCE_SPEC" >&2
vesslctl job create \
  -n "$JOB_NAME" \
  -r "$RESOURCE_SPEC" \
  -i "$IMAGE" \
  --object-volume "${CACHE_VOLUME}:/root/.cache/autoresearch" \
  --tag autoresearch \
  --tag "ar-${TAG_SAFE}" \
  --cmd "$JOB_CMD" >&2

SLUG="$(find_job_slug "$JOB_NAME")" || { echo "submit-async.sh: failed to locate job slug for $JOB_NAME" >&2; exit 3; }
echo "submit-async.sh: submitted $SLUG (branch=$BRANCH commit=$COMMIT)" >&2

# The captured value: just the slug, on a single stdout line.
echo "$SLUG"
