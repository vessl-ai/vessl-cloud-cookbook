#!/usr/bin/env bash
# prep.sh — one-time data prep job. Streams HuggingFaceFW/fineweb, filters
# to CC dumps <= 2017-W26 (the lookahead-bias cutoff), tokenizes with the
# Qwen3.5 tokenizer, and writes uint32 shards to the cache volume.
#
# Run this once after creating your AQR_CACHE_VOLUME. After it succeeds,
# every batch-job/submit.sh invocation skips prepare.py because the data
# shards and manifest already exist on the volume.
#
# Required env vars:
#   AQR_CACHE_VOLUME       slug of the empty object volume to seed.
#   AQR_RESOURCE_SPEC      resource spec for the prep job (CPU is enough for
#                          tokenization; defaults to an a100cpu spec).
#
# Optional env vars (same defaults as submit.sh):
#   AQR_IMAGE              default: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
#   AQR_REPO_URL           default: https://github.com/vessl-ai/vessl-cloud-cookbook.git
#   AQR_BRANCH             default: main
#   AQR_TARGET_TOKENS      default: 1B (passed to prepare.py)
#
# Usage:
#   bash batch-job/prep.sh

set -euo pipefail

CACHE_VOLUME="${AQR_CACHE_VOLUME:?set AQR_CACHE_VOLUME to your cache volume slug}"
# CPU spec — tokenization is single-threaded I/O-bound work, GPU would idle.
RESOURCE_SPEC="${AQR_RESOURCE_SPEC:-resourcespec-a100cpu}"
IMAGE="${AQR_IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel}"
REPO_URL="${AQR_REPO_URL:-https://github.com/vessl-ai/vessl-cloud-cookbook.git}"
BRANCH="${AQR_BRANCH:-main}"
TARGET_TOKENS="${AQR_TARGET_TOKENS:-1B}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_lib.sh
. "$SCRIPT_DIR/_lib.sh"

# Auto-load Kaggle creds. Two formats supported (kaggle.com switched API
# token systems in 2025, so both shapes are in the wild):
#
#   (a) New API token: ~/.kaggle/access_token contains a single-line
#       "KGAT_..." token. Forwarded into the container as KAGGLE_API_TOKEN.
#       This is what the kaggle.com Settings UI now hands out.
#
#   (b) Legacy creds: ~/.kaggle/kaggle.json contains
#       {"username":"...","key":"..."}. Forwarded as KAGGLE_USERNAME +
#       KAGGLE_KEY. Still works; produced by the "Create Legacy API Key"
#       button if the user clicks it.
#
# Either file is enough — newest matching value wins via env precedence.
if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$HOME/.kaggle/access_token" ]; then
  KAGGLE_API_TOKEN="$(tr -d '[:space:]' < "$HOME/.kaggle/access_token" 2>/dev/null || true)"
  if [ -n "$KAGGLE_API_TOKEN" ]; then
    export KAGGLE_API_TOKEN
    echo "prep.sh: loaded Kaggle API token from ~/.kaggle/access_token (KGAT_...)"
  fi
fi
if [ -z "${KAGGLE_USERNAME:-}" ] && [ -f "$HOME/.kaggle/kaggle.json" ]; then
  KAGGLE_USERNAME="$(python3 -c 'import json,os; print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])' 2>/dev/null || true)"
  KAGGLE_KEY="$(python3 -c 'import json,os; print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["key"])' 2>/dev/null || true)"
  if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    export KAGGLE_USERNAME KAGGLE_KEY
    echo "prep.sh: loaded legacy Kaggle creds from ~/.kaggle/kaggle.json (user: $KAGGLE_USERNAME)"
  fi
fi

JOB_NAME="aqr-finance-prep-$(date +%s)"

JOB_CMD=$(cat <<EOF
set -e
apt-get update -qq && apt-get install -y -qq git curl unzip
mkdir -p /workspace && cd /workspace
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" .
cd aqr-finance
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="\$HOME/.local/bin:\$PATH"
mkdir -p "\$HOME/.cache/aqr-finance"
uv run prepare.py --target_tokens "${TARGET_TOKENS}"

# Optional: Kaggle JPX dataset for eval.py. Skips silently if no creds.
# Container-side: materialise kaggle.json from whichever creds env var
# made it through, so the kaggle CLI finds them.
mkdir -p "\$HOME/.cache/aqr-finance/jpx" "\$HOME/.kaggle"
if [ -f "\$HOME/.cache/aqr-finance/jpx/stock_prices.csv" ]; then
  echo "prep.sh: JPX already cached, skipping"
elif [ -n "\${KAGGLE_API_TOKEN:-}" ]; then
  echo "prep.sh: KAGGLE_API_TOKEN detected, writing ~/.kaggle/access_token"
  printf '%s' "\$KAGGLE_API_TOKEN" > "\$HOME/.kaggle/access_token"
  chmod 600 "\$HOME/.kaggle/access_token"
  cd "\$HOME/.cache/aqr-finance/jpx"
  uv run --with kaggle kaggle competitions download -c jpx-tokyo-stock-exchange-prediction || \
    echo "prep.sh: kaggle download failed (new-token path) — eval.py will need stock_prices.csv pre-staged"
  for zf in *.zip; do [ -e "\$zf" ] && unzip -q "\$zf"; done
  if [ -f train_files/stock_prices.csv ]; then
    mv train_files/stock_prices.csv .
  fi
  cd - > /dev/null
elif [ -n "\${KAGGLE_USERNAME:-}" ] && [ -n "\${KAGGLE_KEY:-}" ]; then
  echo "prep.sh: legacy KAGGLE_USERNAME/KEY detected, writing ~/.kaggle/kaggle.json"
  printf '{"username":"%s","key":"%s"}' "\$KAGGLE_USERNAME" "\$KAGGLE_KEY" > "\$HOME/.kaggle/kaggle.json"
  chmod 600 "\$HOME/.kaggle/kaggle.json"
  cd "\$HOME/.cache/aqr-finance/jpx"
  uv run --with kaggle kaggle competitions download -c jpx-tokyo-stock-exchange-prediction || \
    echo "prep.sh: kaggle download failed (legacy path) — eval.py will need stock_prices.csv pre-staged"
  for zf in *.zip; do [ -e "\$zf" ] && unzip -q "\$zf"; done
  if [ -f train_files/stock_prices.csv ]; then
    mv train_files/stock_prices.csv .
  fi
  cd - > /dev/null
else
  echo "prep.sh: no Kaggle creds — JPX skipped (eval.py will fail loudly unless stock_prices.csv is pre-staged on the volume)"
fi

ls -la "\$HOME/.cache/aqr-finance/data" | head -30
cat "\$HOME/.cache/aqr-finance/data/manifest.json"
ls -la "\$HOME/.cache/aqr-finance/jpx" 2>/dev/null | head -10
EOF
)

echo "prep.sh: creating job $JOB_NAME on $RESOURCE_SPEC (target=${TARGET_TOKENS})"
# Forward Kaggle creds into the container so the JPX block in JOB_CMD can
# download the dataset. vesslctl does NOT propagate local env vars by
# default — they have to be passed explicitly via --env. Empty values are
# harmless: JOB_CMD skips the download if neither var is set.
vesslctl job create \
  -n "$JOB_NAME" \
  -r "$RESOURCE_SPEC" \
  -i "$IMAGE" \
  --object-volume "${CACHE_VOLUME}:/root/.cache/aqr-finance" \
  --env "KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:-}" \
  --env "KAGGLE_USERNAME=${KAGGLE_USERNAME:-}" \
  --env "KAGGLE_KEY=${KAGGLE_KEY:-}" \
  --tag aqr-finance \
  --tag aqr-finance-prep \
  --cmd "$JOB_CMD"

SLUG="$(find_job_slug "$JOB_NAME")" || { echo "prep.sh: failed to find job slug for $JOB_NAME" >&2; exit 3; }
echo "prep.sh: job slug $SLUG — polling state (this can take ~30-60 min for 1B tokens)"

WAIT_RC=0
wait_for_job "$SLUG" || WAIT_RC=$?

echo "--- job logs ($SLUG) ---"
dump_job_logs "$SLUG"
echo "--- end job logs ---"

FINAL_STATE="$(job_state "$SLUG")"
echo "prep.sh: final state $FINAL_STATE (wait_rc=$WAIT_RC)"
[ "$FINAL_STATE" = "succeeded" ]
