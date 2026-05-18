# _lib.sh — shared helpers for prep.sh, submit.sh, submit-async.sh, wait-jobs.sh.
# Sourced; not directly executable.

# Find the slug of a job by its name. Retries because there's a brief delay
# between `job create` and the job appearing in `job list`.
find_job_slug() {
  local name="$1"
  local slug=""
  for _ in $(seq 1 10); do
    slug="$(vesslctl job list -o json 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); n=sys.argv[1]; print(next((j["slug"] for j in d if j["name"]==n), ""))' "$name" 2>/dev/null || true)"
    [ -n "$slug" ] && { echo "$slug"; return 0; }
    sleep 2
  done
  return 1
}

# Get the current state of a job. Empty string on lookup failure.
job_state() {
  vesslctl job show "$1" -o json 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("workloadState",""))' 2>/dev/null \
    || true
}

# Wait until the job reaches a terminal state. Echos "state: <s>" lines on
# every state change so the caller's stdout has a coarse progress trail.
# Returns 0 if the terminal state is `succeeded`, 1 otherwise.
wait_for_job() {
  local slug="$1"
  local prev=""
  local state=""
  while true; do
    state="$(job_state "$slug")"
    if [ -n "$state" ] && [ "$state" != "$prev" ]; then
      echo "[$(date -u +%H:%M:%S)] state: $state"
      prev="$state"
    fi
    case "$state" in
      succeeded) return 0 ;;
      failed|terminated|cancelled) return 1 ;;
    esac
    sleep 30
  done
}

# Dump the full job log. The vesslctl API caps --limit at 1000. AQR training
# runs are longer than autoresearch's 5-min experiments (5-9 h on 8xH100),
# so the bulk of the log is loss-curve lines from train.py. Use \r-overwritten
# progress lines and a `^---` summary block at the end so a 1000-line dump
# always captures the summary regardless of run length.
dump_job_logs() {
  vesslctl job logs --limit 1000 "$1" 2>&1 || true
}
