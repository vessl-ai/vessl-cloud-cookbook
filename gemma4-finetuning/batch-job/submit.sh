#!/usr/bin/env bash
# submit.sh — reference vesslctl invocation for this recipe.
#
# Prereqs:
#   - vesslctl installed and authenticated (vesslctl version tested: 2026.04.16-01).
#   - An Object volume in your org to mount at /shared.
#
# Usage:
#   VESSL_OBJECT_VOLUME=<your-volume-name> ./submit.sh <mode> <tag>
#     mode: 'generic' or 'vessl' (matches DATASET_MODE in finetune_gemma4.py)
#     tag:  a short run identifier, e.g., 'my-first-run'
#
# Tested 2026-04-16 on A100 SXM.

set -euo pipefail

MODE="${1:?mode required: generic|vessl}"
TAG="${2:?tag required}"
VOLUME="${VESSL_OBJECT_VOLUME:?set VESSL_OBJECT_VOLUME to your object-volume name}"

# 1. Upload script and dataset to the object volume (one-time per volume)
vesslctl volume upload "$VOLUME" finetune_gemma4.py --remote-prefix scripts/
vesslctl volume upload "$VOLUME" ../data/vessl-cloud-qa-dataset.json --remote-prefix datasets/

# 2. Submit the fine-tuning job
vesslctl job create \
  --name "gemma4-${MODE}-${TAG}" \
  --resource-spec resourcespec-a100x1 \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --object-volume "${VOLUME}:/shared" \
  --env DATASET_MODE="${MODE}" \
  --env RUN_TAG="${TAG}" \
  --cmd "pip install unsloth trl transformers datasets && python -u /shared/scripts/finetune_gemma4.py"

echo "Submitted. Tail logs with: vesslctl job logs -f gemma4-${MODE}-${TAG}"
