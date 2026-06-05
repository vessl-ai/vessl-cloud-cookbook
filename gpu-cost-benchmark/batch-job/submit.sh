#!/usr/bin/env bash
# submit.sh — reference vesslctl invocation for the GPU cost benchmark.
#
# Runs the full training + inference sweep for ONE hardware target by
# submitting `run_full_per_hardware.sh <HW>` as a vesslctl batch job.
#
# Prereqs:
#   - vesslctl installed and authenticated.
#   - An Object storage volume in your org to mount at /shared.
#   - An 8-GPU resource spec for the target hardware.
#   - A Hugging Face token (gemma-4-31B-it is gated): export HF_TOKEN.
#
# Usage:
#   RESOURCE_SPEC=resourcespec-a100x8 VESSL_OBJECT_VOLUME=<volume> ./submit.sh <HW> <tag>
#     <HW>:  A100 | H100 | B200
#     <tag>: short run identifier, e.g. 'my-first-run'
#
#   Benchmark a different GPU by changing the HW arg (and RESOURCE_SPEC):
#     RESOURCE_SPEC=resourcespec-h100x8 ... ./submit.sh H100 my-run
#     RESOURCE_SPEC=resourcespec-b200x8 ... ./submit.sh B200 my-run
#
# Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.

set -eo pipefail

HW="${1:?hardware required: A100|H100|B200}"
TAG="${2:?tag required, e.g. my-first-run}"
VOLUME="${VESSL_OBJECT_VOLUME:?set VESSL_OBJECT_VOLUME to your object-volume name}"
RESOURCE_SPEC="${RESOURCE_SPEC:?set RESOURCE_SPEC to your 8-GPU spec, e.g. resourcespec-a100x8}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"   # needs Transformer Engine for B200 te-fp8
HF_TOKEN="${HF_TOKEN:?export HF_TOKEN (gemma-4-31B-it is gated)}"

# 1. Upload scripts + data-prep to the object volume (one-time per volume).
vesslctl volume upload "$VOLUME" train_sft.py             --remote-prefix scripts/
vesslctl volume upload "$VOLUME" serve_and_bench.sh       --remote-prefix scripts/
vesslctl volume upload "$VOLUME" bench_client.py          --remote-prefix scripts/
vesslctl volume upload "$VOLUME" run_full_per_hardware.sh --remote-prefix scripts/
vesslctl volume upload "$VOLUME" ../data_prep/pack_sharegpt.py --remote-prefix scripts/
vesslctl volume upload "$VOLUME" ../requirements.txt          --remote-prefix scripts/

# 2. Submit the per-hardware sweep.
vesslctl job create \
  --name "gpu-cost-${HW}-${TAG}" \
  --resource-spec "${RESOURCE_SPEC}" \
  --image "${IMAGE}" \
  --object-volume "${VOLUME}:/shared" \
  --env HARDWARE="${HW}" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env OUTPUT_BASE="/shared/gpu-cost-${HW}-${TAG}" \
  --cmd "pip install -r /shared/scripts/requirements.txt 2>/dev/null; cd /shared/scripts && bash run_full_per_hardware.sh ${HW}"

echo "Submitted. Tail logs with: vesslctl job logs -f gpu-cost-${HW}-${TAG}"
echo "Results land under the object volume at /shared/gpu-cost-${HW}-${TAG}/"
