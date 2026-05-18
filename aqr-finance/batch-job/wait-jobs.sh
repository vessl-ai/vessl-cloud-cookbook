#!/usr/bin/env bash
# wait-jobs.sh — wait for N submitted jobs to terminate, then dump a
# per-job summary (state, r2_leakage_off, r2_leakage_on, leakage_premium,
# peak_vram_mb, training_seconds).
#
# Pair with submit-async.sh for "Mode B: parallel batch" runs:
#
#   slug_a=$(... checkout candidate-a ... ; bash batch-job/submit-async.sh)
#   slug_b=$(... checkout candidate-b ... ; bash batch-job/submit-async.sh)
#   slug_c=$(... checkout candidate-c ... ; bash batch-job/submit-async.sh)
#   bash batch-job/wait-jobs.sh "$slug_a" "$slug_b" "$slug_c" > batch.log 2>&1
#
# Output: a header line per slug ("=== <slug> (state) ===") followed by
# the train.py + eval.py summary blocks grepped from that job's logs.
# Exit 0 if all jobs reached `succeeded`, non-zero otherwise — listing
# the failed slugs on stderr.
#
# Optional env:
#   AQR_TIMEOUT_S            hard wall-clock cap, default 36000 (10h). Past
#                            this, any still-running jobs are reported as
#                            'timeout' (they keep running on VESSL — terminate
#                            manually if you don't want them).
#   AQR_POLL_INTERVAL_S      poll interval, default 60s.

set -uo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: wait-jobs.sh <slug> [<slug> ...]" >&2
  exit 64
fi

TIMEOUT_S="${AQR_TIMEOUT_S:-36000}"
POLL_INTERVAL_S="${AQR_POLL_INTERVAL_S:-60}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_lib.sh
. "$SCRIPT_DIR/_lib.sh"

slugs=("$@")
n_slugs=${#slugs[@]}
last_states=()
for i in $(seq 0 $((n_slugs - 1))); do
  last_states[i]=""
done
deadline=$(( $(date +%s) + TIMEOUT_S ))

echo "wait-jobs.sh: tracking ${n_slugs} job(s): ${slugs[*]}" >&2

while true; do
  pending=0
  for i in $(seq 0 $((n_slugs - 1))); do
    slug="${slugs[i]}"
    state="$(job_state "$slug")"
    [ -z "$state" ] && state="?"
    if [ "${last_states[i]}" != "$state" ]; then
      echo "[$(date -u +%H:%M:%S)] $slug: $state" >&2
      last_states[i]="$state"
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

# Per-slug summary. Pull each job's log once and grep the train.py + eval.py
# summary blocks (r2_leakage_off, r2_leakage_on, leakage_premium, etc).
overall_rc=0
failed_slugs=()
for slug in "${slugs[@]}"; do
  state="$(job_state "$slug")"
  [ -z "$state" ] && state="?"
  echo
  echo "=== $slug ($state) ==="
  if [ "$state" = "succeeded" ]; then
    vesslctl job logs --limit 1000 "$slug" 2>&1 \
      | grep -E "^r2_leakage_off:|^r2_leakage_on:|^leakage_premium:|^base_r2_leakage_off:|^base_r2_leakage_on:|^base_leakage_premium:|^premium_reduction:|^val_loss_final:|^val_loss_first30min:|^training_seconds:|^total_seconds:|^peak_vram_mb:|^num_trainable_M:|^num_trainable_pct:|^num_train_tokens_M:|^---" \
      | tail -40
  else
    # Non-succeeded: dump the tail so the agent can see the trace.
    vesslctl job logs --limit 1000 "$slug" 2>&1 | tail -40
    overall_rc=1
    failed_slugs+=("$slug")
  fi
done

if [ "$overall_rc" -ne 0 ]; then
  echo
  echo "wait-jobs.sh: ${#failed_slugs[@]} job(s) did not succeed: ${failed_slugs[*]}" >&2
fi
exit "$overall_rc"
