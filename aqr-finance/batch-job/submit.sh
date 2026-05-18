#!/usr/bin/env bash
# submit.sh — run one AQR cookbook experiment as a VESSL Cloud batch job.
#
# Called by the autoresearch agent loop in place of `uv run train.py`.
# Pushes the current aqr-finance/<tag> branch to origin, submits a vesslctl
# job that clones the cookbook at that branch and runs train.py + eval.py
# on 8xH100 SXM single-node, polls until the job reaches a terminal state,
# then dumps the full job log to stdout. Exit code reflects final state.
#
# Prereqs:
#   - vesslctl installed and authenticated. The active org and team
#     (`vesslctl auth login`, `vesslctl config set default_org/default_team`,
#     or `--org` / `--team` / VESSLCTL_ORG / VESSLCTL_TEAM) determine where
#     the job is billed. Run `vesslctl auth status` to see what's resolved.
#   - Object volume populated by batch-job/prep.sh (data shards + manifest).
#   - Current branch matches aqr-finance/* and the working tree is clean.
#
# Required env vars (set once per session, e.g. in your shell rc):
#   AQR_CACHE_VOLUME       slug of the object volume holding ~/.cache/aqr-finance
#                          (e.g. objvol-abc123).
#   AQR_RESOURCE_SPEC      resource spec slug for 8xH100 SXM single-node.
#                          Find with `vesslctl resource-spec list`.
#
# Optional env vars:
#   AQR_IMAGE              default: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
#   AQR_REPO_URL           default: https://github.com/vessl-ai/vessl-cloud-cookbook.git
#   AQR_TIMEOUT_S          default: 36000 (10h — covers a 5-9h baseline + buffer)
#
# Usage (from the agent loop):
#   bash batch-job/submit.sh > run.log 2>&1
#   grep "^r2_leakage_off:\|^r2_leakage_on:\|^peak_vram_mb:" run.log

set -euo pipefail

CACHE_VOLUME="${AQR_CACHE_VOLUME:?set AQR_CACHE_VOLUME to your cache volume slug (e.g. objvol-...)}"
RESOURCE_SPEC="${AQR_RESOURCE_SPEC:?set AQR_RESOURCE_SPEC to your 8xH100 SXM spec slug (vesslctl resource-spec list)}"
IMAGE="${AQR_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel}"
REPO_URL="${AQR_REPO_URL:-https://github.com/vessl-ai/vessl-cloud-cookbook.git}"
TIMEOUT_S="${AQR_TIMEOUT_S:-36000}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_lib.sh
. "$SCRIPT_DIR/_lib.sh"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
case "$BRANCH" in
  aqr-finance/*) ;;
  *) echo "submit.sh: must be on an aqr-finance/<tag> branch (current: $BRANCH)" >&2; exit 2 ;;
esac
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "submit.sh: working tree is dirty — commit your train.py edits first" >&2
  exit 2
fi

COMMIT="$(git rev-parse --short HEAD)"
TAG="${BRANCH#aqr-finance/}"
TAG_SAFE="$(printf '%s' "$TAG" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | sed 's/^-*//;s/-*$//')"
JOB_NAME="aqr-finance-${TAG_SAFE}-${COMMIT}"

echo "submit.sh: pushing $BRANCH ($COMMIT) to origin"
git push --force-with-lease -u origin "$BRANCH" >&2

# Job command runs inside the container. Idempotent: prepare.py is a no-op
# if the manifest matches (data + tokenizer cached on mounted volume).
#
# We rely on the NVIDIA pytorch:25.10-py3 image's pre-installed, NVIDIA-tuned
# torch + triton + CUDA stack — building our own venv would shadow those
# binaries with PyPI wheels (the trap that ate dry-runs 3-6). We only add
# the framework layer (unsloth + trl + transformers v5 + peft + accelerate)
# and the dataset/eval Python deps. Unsloth ships its own Triton kernels so
# we don't pin flash-linear-attention / causal-conv1d directly — Unsloth
# pulls compatible versions transitively.
JOB_CMD=$(cat <<EOF
set -e
apt-get update -qq && apt-get install -y -qq git curl unzip
mkdir -p /workspace && cd /workspace
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" .
cd aqr-finance
# Object-volume HF cache caused "no config file" errors on dry-run 10 — a
# previous OOM'd run left a partial cache on the volume and object storage
# doesn't guarantee atomic writes. Stick with the ephemeral default
# (~/.cache/huggingface) until boot stabilizes; we eat a ~3-min Qwen3.5-35B
# download per cold start until then.
# expandable_segments reduces VRAM fragmentation — with a 74 GB model on an
# 80 GB GPU the headroom is small enough that fragmentation alone can
# trigger OOM mid-step. PyTorch's own OOM hint flags this in our logs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Framework layer on top of NVIDIA-tuned base stack.
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir \
  'unsloth' \
  'unsloth_zoo' \
  'transformers>=5.5' \
  'trl>=0.12' \
  'peft>=0.13' \
  'accelerate>=1.0' \
  'datasets>=3.0' \
  'huggingface_hub>=0.26' \
  'numpy<3.0' \
  'pandas>=2.2' \
  'scikit-learn>=1.5' \
  'matplotlib>=3.10' \
  'pyarrow>=18.0' \
  'kaggle>=1.6' \
  'tqdm>=4.66'

if [ ! -f "\$HOME/.cache/aqr-finance/data/manifest.json" ]; then
  echo "submit: cache empty, running prepare.py"
  python prepare.py
fi

# Single-process train — Unsloth's OSS path is single-GPU optimized; multi-GPU
# DDP comes later once boot is verified. accelerate_config.yaml is retained
# in-tree for the multi-GPU follow-up but unused here.
python train.py
python eval.py
EOF
)

echo "submit.sh: creating job $JOB_NAME on $RESOURCE_SPEC"
vesslctl job create \
  -n "$JOB_NAME" \
  -r "$RESOURCE_SPEC" \
  -i "$IMAGE" \
  --object-volume "${CACHE_VOLUME}:/root/.cache/aqr-finance" \
  --tag aqr-finance \
  --tag "aqr-${TAG_SAFE}" \
  --cmd "$JOB_CMD" >&2

SLUG="$(find_job_slug "$JOB_NAME")" || { echo "submit.sh: failed to locate job slug for $JOB_NAME" >&2; exit 3; }
echo "submit.sh: job slug $SLUG"

# Poll until terminal, with a hard timeout so a stuck job doesn't block the
# agent loop indefinitely. The job continues running on VESSL after timeout —
# kill it manually with `vesslctl job terminate $SLUG` to stop billing.
WAIT_RC=0
( wait_for_job "$SLUG" ) &
WAIT_PID=$!
( sleep "$TIMEOUT_S"; kill -TERM $WAIT_PID 2>/dev/null || true ) &
WATCHER_PID=$!
wait "$WAIT_PID" || WAIT_RC=$?
kill "$WATCHER_PID" 2>/dev/null || true

# Always dump logs — the agent's grep runs against this output.
echo "--- job logs ($SLUG) ---"
dump_job_logs "$SLUG"
echo "--- end job logs ---"

FINAL_STATE="$(job_state "$SLUG")"
echo "submit.sh: final state $FINAL_STATE (wait_rc=$WAIT_RC)"
[ "$FINAL_STATE" = "succeeded" ]
