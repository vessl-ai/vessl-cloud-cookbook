# autoresearch (VESSL Cloud edition)

This is an experiment to have the LLM do its own research. Same idea as
[karpathy/autoresearch](https://github.com/karpathy/autoresearch), but every
training run executes as a single VESSL Cloud batch job instead of running on
a local GPU. You edit code locally, push, the job clones the cookbook at your
commit and runs `train.py`, you read back the result.

## Setup

To set up a new experiment, work with the user to:

1. **Confirm the environment**: `vesslctl auth status` should show you logged
   in to the org/team you want experiments billed to. The env var
   `AUTORESEARCH_CACHE_VOLUME` must be set to the slug of the object volume
   that holds `~/.cache/autoresearch` (data shards + tokenizer). Check with:
   ```
   echo "$AUTORESEARCH_CACHE_VOLUME"
   vesslctl volume show "$AUTORESEARCH_CACHE_VOLUME"
   ```
   If the volume is empty (no `tokenizer/tokenizer.pkl` shown by
   `vesslctl volume ls "$AUTORESEARCH_CACHE_VOLUME"`), tell the user to run
   `bash batch-job/prep.sh` once before continuing — it submits a one-time
   job that downloads the data shards and trains the BPE tokenizer into the
   volume. This takes ~5-15 minutes.
2. **Agree on a run tag**: propose a tag based on today's date and an
   optional theme that hints at what this run is exploring
   (e.g. `mar5-opt`, `mar5-arch`). The branch `autoresearch/<tag>` must not
   already exist on origin — this is a fresh run. Check with
   `git ls-remote --heads origin autoresearch/<tag>`. If the user is running
   multiple agents in parallel, each agent must have a unique tag — the
   tag is what keeps the branches and job names from colliding.
3. **Create the branch**: from `main` of `vessl-cloud-cookbook`,
   `git checkout -b autoresearch/<tag>`.
4. **Read the in-scope files**:
   - `README.md` — recipe context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader,
     evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer,
     training loop.
   - `batch-job/submit.sh` — the runner. Don't modify unless you have to.
5. **Initialize results.tsv**: Create `results.tsv` in the recipe directory
   with just the header row. The baseline will be recorded after the first
   run. Do not commit this file.
6. **Confirm and go**: confirm setup looks good, then kick off
   experimentation.

## Experimentation

Each experiment runs as a single VESSL job on a single GPU. The training
script runs for a **fixed time budget of 5 minutes** (wall clock training
time, excluding startup/compilation, defined in `prepare.py`). End-to-end the
job takes ~7-12 minutes once you include image pull, repo clone, `uv sync`,
torch.compile, and the final eval.

You launch one experiment with:

```
bash batch-job/submit.sh > run.log 2>&1
```

The script will:

- Push your `autoresearch/<tag>` branch to origin (force-with-lease).
- Submit a `vesslctl job create` that clones the cookbook at that branch and
  runs `train.py` with the cache volume mounted at `~/.cache/autoresearch`.
- Poll `vesslctl job show` until the job reaches a terminal state, then
  dump the job's full log to stdout (which `> run.log` captures locally).
- Exit 0 if the job's final state is `succeeded`, non-zero otherwise.

**What you CAN do:**

- Modify `train.py`. Everything is fair game: model architecture, optimizer,
  hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation,
  data loading, tokenizer, and training constants (time budget, sequence
  length, etc).
- Install new packages or add dependencies. You can only use what's already
  in `pyproject.toml`. (Adding a dep would require a new `uv.lock` and would
  rebuild the venv on every job, costing a minute per experiment.)
- Modify the evaluation harness (`evaluate_bpb` in `prepare.py`).
- Modify `batch-job/submit.sh` or `batch-job/prep.sh` for experimental
  reasons. They are the runner, not the experiment.

**The goal is simple: get the lowest val_bpb.** Since the time budget is
fixed, you don't worry about training time — it's always 5 minutes inside
the container. Everything else is fair game.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful
val_bpb gains, but it should not blow up dramatically. The default GPU
target is H100 SXM ×1 (80 GB), so OOM at ~70+ GB peak.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. Conversely, removing
something and getting equal or better results is a great outcome — that's a
simplification win. A 0.001 val_bpb improvement that adds 20 lines of hacky
code? Probably not worth it. A 0.001 val_bpb improvement from deleting code?
Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the
baseline, so you will run `train.py` as is.

## Output format

Once `train.py` finishes inside the job it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

`submit.sh` dumps the job's full log to stdout once the job terminates, so
this summary lands in your local `run.log`. Extract the key metrics with:

```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

Note: numbers depend on the GPU. The recipe targets H100 SXM ×1 by default,
which approximately matches karpathy's reference numbers.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024)
   — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch `autoresearch/<tag>` of
`vessl-cloud-cookbook`. There are two equally valid loop shapes — pick one
upfront and tell the user which you're using; you can switch between runs
but don't switch mid-run, since the branch hygiene is different.

### Mode A: Linear (karpathy-style accept/reject)

Best for: depth-first iteration where each idea builds on the last,
debugging a single direction, or the very first run where you don't have
enough signal to fan out.

LOOP FOREVER:

1. Look at the git state: the current branch/commit you're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. `git commit` (locally — `submit.sh` will push for you).
4. Run the experiment: `bash batch-job/submit.sh > run.log 2>&1`. This
   blocks until the job is done. Do NOT use `tee` or let the streamed step
   lines flood your context.
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`.
6. If the grep output is empty, the run crashed. Run `tail -n 80 run.log`
   to read the trace and attempt a fix. The trace may be a Python error
   from `train.py` or a VESSL error (image pull, OOM kill, scheduling
   failure) — they look different. If you can't get things to work after a
   few attempts, give up.
7. Record the results in `results.tsv` (do not commit this file).
8. If `val_bpb` improved (lower), "advance" the branch — keep the commit.
9. If `val_bpb` is equal or worse, `git reset --hard HEAD~1` to discard
   the change. (`submit.sh` uses force-with-lease on push, so this is safe
   — the next push will overwrite the discarded commit on origin.)

### Mode B: Batch (parallel fan-out, pick best of K)

Best for: breadth-first sweeps (try LR ∈ {0.02, 0.04, 0.06, 0.08} at once),
when you have multiple plausible-but-untried ideas and want to compare
them apples-to-apples, or just to use the cloud GPU pool while you can.
Round duration is ~10 min (one job-wall-time) instead of ~10 min × K, so
the throughput multiplier vs. Mode A is roughly K.

Each round:

1. Look at the git state. Your "main" branch is still `autoresearch/<tag>`
   — that's the current best. Note its HEAD as `BASE`.
2. Generate K candidate experiments (typically 3-5; more is fine but
   read `vesslctl billing` first). Each candidate is one tweak to
   `train.py`. For each candidate `i` in 1..K:
   - `git checkout -b autoresearch/<tag>-r<round>-<i> BASE`
   - Edit `train.py` for candidate `i`.
   - `git commit -m "round <round> cand <i>: <one-line description>"`
   - `slug_<i>=$(bash batch-job/submit-async.sh)` — captures the job slug.
3. Once all K are submitted, wait for them together:
   `bash batch-job/wait-jobs.sh "$slug_1" "$slug_2" ... "$slug_K" > round.log 2>&1`
   This blocks until all K jobs reach a terminal state, then prints one
   summary block per slug.
4. Parse `round.log`: extract `val_bpb` per slug, map back to the
   candidate branch (the slug's job name has the form
   `autoresearch-<tag>-r<round>-<i>-<commit>`).
5. Pick the winner: lowest `val_bpb` that beats `BASE`'s previously
   recorded `val_bpb`. Crashes count as a loss.
6. Advance the main branch:
   - `git checkout autoresearch/<tag>`
   - `git reset --hard autoresearch/<tag>-r<round>-<winner>`
   - (or, if no candidate beat `BASE`, leave `autoresearch/<tag>` alone
     and try a different direction next round.)
7. Record all K candidates in `results.tsv` — one row per candidate, with
   `keep` for the winner and `discard` for the rest, plus `crash` for any
   that didn't finish.
8. Optional cleanup: `git branch -D autoresearch/<tag>-r<round>-<i>` for
   the losers, and `git push origin --delete autoresearch/<tag>-r<round>-<i>`
   to also remove them from origin. The winner's branch can stay around
   for git history.
9. Increment `round` and go to step 1.

In Mode B, the "branch reset" semantics are at the round level instead of
the per-experiment level. The main branch only ever advances when a
candidate beats it. Losers are discarded as branches, not as commits on a
shared trunk, so there's no `force-with-lease` needed for cleanup.

### Mode flexibility

The two modes can coexist in one tag's history if needed (e.g. start with
Mode B for a coarse sweep, switch to Mode A for fine-tuning the best
candidate), but try not to weave them within a single round — it makes
results.tsv hard to read.

**Timeout**: `submit.sh` enforces `AUTORESEARCH_TIMEOUT_S` (default 1800s).
If the job hasn't finished by then, the script kills its polling loop; the
job continues running in the background — you treat it as a failure
(discard and revert), and optionally `vesslctl job terminate <slug>` to
stop billing.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment:
if it's something dumb and easy to fix (e.g. a typo, a missing import), fix
it and re-run. If the idea itself is fundamentally broken, just skip it,
log "crash" as the status in the tsv, and move on.

**VESSL-specific failure modes** to be aware of:

- **Image pull / scheduling delays**: occasionally the queue is slow. Don't
  treat slow start as a crash — wait for the timeout.
- **Volume mount issues**: if the cache volume isn't mounted correctly,
  `prepare.py` will start re-downloading data inside the job. The 5-minute
  training budget will then never be reached and the job will time out.
  Check `vesslctl volume show "$AUTORESEARCH_CACHE_VOLUME"` if you see this.
- **Spec unavailability**: if the resource spec is at low availability, the
  job may sit in `queued` for a long time. Pick a different spec via
  `AUTORESEARCH_RESOURCE_SPEC`.

**NEVER STOP**: Once the experiment loop has begun (after the initial
setup), do NOT pause to ask the human if you should continue. Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human might
be asleep, or gone from a computer and expects you to continue working
*indefinitely* until you are manually stopped. You are autonomous. If you
run out of ideas, think harder — read papers referenced in the code, re-read
the in-scope files for new angles, try combining previous near-misses, try
more radical architectural changes. The loop runs until the human interrupts
you, period.

As an example use case, a user might leave you running while they sleep. If
each experiment takes ~10 minutes end-to-end on VESSL (including job startup
overhead) you can run ~6/hour, for a total of ~50 over the duration of the
average human sleep. The user wakes up to experimental results, all
completed by you while they slept!
