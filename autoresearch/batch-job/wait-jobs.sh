#!/usr/bin/env bash
# wait-jobs.sh — wait for N submitted jobs to terminate, then dump a
# per-job summary (state, val_bpb, peak_vram, key training metrics).
#
# Pair with submit-async.sh for "Mode B: parallel batch" runs:
#
#   slug_a=$(... checkout candidate-a ... ; bash batch-job/submit-async.sh)
#   slug_b=$(... checkout candidate-b ... ; bash batch-job/submit-async.sh)
#   slug_c=$(... checkout candidate-c ... ; bash batch-job/submit-async.sh)
#   bash batch-job/wait-jobs.sh "$slug_a" "$slug_b" "$slug_c" > batch.log 2>&1
#
# Output: a header line per slug ("=== <slug> (state) ===") followed by
# the train.py summary block grepped from that job's logs (val_bpb, etc.).
# Exit 0 if all jobs reached `succeeded`, non-zero otherwise — listing the
# failed slugs on stderr.
#
# Optional env:
#   AUTORESEARCH_TIMEOUT_S       hard wall-clock cap, default 1800. Past
#                                this, any still-running jobs are reported
#                                as 'timeout' (they keep running on VESSL —
#                                terminate manually if you don't want them).
#   AUTORESEARCH_POLL_INTERVAL_S poll interval, default 20s.

set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: wait-jobs.sh <slug> [<slug> ...]" >&2
  exit 64
fi

TIMEOUT_S="${AUTORESEARCH_TIMEOUT_S:-1800}"
POLL_INTERVAL_S="${AUTORESEARCH_POLL_INTERVAL_S:-20}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_lib.sh
. "$SCRIPT_DIR/_lib.sh"

slugs=("$@")
declare -A last_state=()
deadline=$(( $(date +%s) + TIMEOUT_S ))

echo "wait-jobs.sh: tracking ${#slugs[@]} job(s): ${slugs[*]}" >&2

while true; do
  pending=0
  for slug in "${slugs[@]}"; do
    state="$(job_state "$slug")"
    [ -z "$state" ] && state="?"
    if [ "${last_state[$slug]:-}" != "$state" ]; then
      echo "[$(date -u +%H:%M:%S)] $slug: $state" >&2
      last_state[$slug]="$state"
    fi
    case "$state" in
      succeeded|failed|terminated|cancelled) ;;
      *) pending=$((pending + 1)) ;;
    esac
  done
  [ "$pending" -eq 0 ] && break
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "wait-jobs.sh: timeout after ${TIMEOUT_S}s with $pending job(s) still running" >&2
    break
  fi
  sleep "$POLL_INTERVAL_S"
done

# Per-slug summary. Pull each job's log once and grep the train.py summary
# block (val_bpb, training_seconds, peak_vram_mb, mfu_percent, etc).
overall_rc=0
failed_slugs=()
for slug in "${slugs[@]}"; do
  state="$(job_state "$slug")"
  [ -z "$state" ] && state="?"
  echo
  echo "=== $slug ($state) ==="
  if [ "$state" = "succeeded" ]; then
    vesslctl job logs --limit 1000 "$slug" 2>&1 \
      | grep -E "val_bpb:|training_seconds:|total_seconds:|peak_vram_mb:|mfu_percent:|num_params_M:|num_steps:|depth:|num_tokens_M:|total_tokens_M:|^---" \
      | tail -20
  else
    # Non-succeeded: dump the tail so the agent can see the trace.
    vesslctl job logs --limit 1000 "$slug" 2>&1 | tail -25
    overall_rc=1
    failed_slugs+=("$slug")
  fi
done

if [ "$overall_rc" -ne 0 ]; then
  echo
  echo "wait-jobs.sh: ${#failed_slugs[@]} job(s) did not succeed: ${failed_slugs[*]}" >&2
fi
exit "$overall_rc"
