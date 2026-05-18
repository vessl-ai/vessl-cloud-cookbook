# aqr-finance (VESSL Cloud edition)

This recipe runs LoRA continued pretraining on `Qwen/Qwen3.5-35B-A3B-Base`
against a chronologically-filtered FineWeb slice (CC dumps `<= 2017-06-30`),
then evaluates the trained adapter on Kaggle JPX with two splits to surface
the leakage premium. Every training run executes as a single VESSL Cloud
batch job on 8xH100 SXM single-node. You edit code locally, push, the job
clones the cookbook at your commit and runs `train.py` + `eval.py`, you
read back the result.

Same loop semantics as the autoresearch cookbook — the agent edits
`train.py`, submits a job, reads back the metrics, keeps or reverts. The
unit of "an experiment" is just longer (~6-10 h instead of 5 min).

## Setup

To set up a new experiment, work with the user to:

1. **Confirm the environment**: `vesslctl auth status` should show you
   logged in to the org/team you want experiments billed to. The env vars
   `AQR_CACHE_VOLUME` and `AQR_RESOURCE_SPEC` must be set:
   ```
   echo "$AQR_CACHE_VOLUME"   # object volume slug holding ~/.cache/aqr-finance
   echo "$AQR_RESOURCE_SPEC"  # 8xH100 SXM single-node spec
   vesslctl volume show "$AQR_CACHE_VOLUME"
   ```
   If the volume is empty (no `data/manifest.json` shown by
   `vesslctl volume ls "$AQR_CACHE_VOLUME"`), tell the user to run
   `bash batch-job/prep.sh` once before continuing — it submits a one-time
   job that streams FineWeb, filters by CC dump date, tokenizes with the
   Qwen tokenizer, and writes the shards to the cache volume. It also
   downloads the Kaggle JPX dataset if `KAGGLE_USERNAME` + `KAGGLE_KEY` are
   set as job env vars. Prep takes ~30-60 minutes.
2. **Agree on a run tag**: propose a tag based on today's date and an
   optional theme that hints at what this run is exploring (e.g.
   `may18-lr`, `may18-target`, `may18-rank`). The branch
   `aqr-finance/<tag>` must not already exist on origin. Check with
   `git ls-remote --heads origin aqr-finance/<tag>`. If the user is running
   multiple agents in parallel, each must have a unique tag.
3. **Create the branch**: from `main` of `vessl-cloud-cookbook`,
   `git checkout -b aqr-finance/<tag>`.
4. **Read the in-scope files**:
   - `README.md` — recipe context.
   - `prepare.py` — FineWeb cutoff + tokenization. Read-only.
   - `eval.py` — leakage on/off measurement. Read-only.
   - `train.py` — the file you modify. LoRA config, optimizer, scheduler,
     training loop.
   - `accelerate_config.yaml` — FSDP config. Don't modify unless OOM
     forces a wrap-policy or sharding-strategy change.
   - `batch-job/submit.sh` — the runner. Don't modify.
5. **Initialize results.tsv**: Create `results.tsv` in the recipe directory
   with the header row described in "Logging results". Do not commit it.
6. **Confirm and go**: confirm setup looks good, then kick off
   experimentation.

## Experimentation

Each experiment runs as one VESSL job on 8xH100 SXM single-node. The
training script runs for a **fixed step budget of `TOTAL_STEPS`** (default
3815 at 1 B tokens / 64 global batch / 4096 seq len). End-to-end the job
takes ~6-10 hours once you include image pull, repo clone, `uv sync`,
torch.compile, `prepare.py` cache hit, train, and final eval.

You launch one experiment with:

```
bash batch-job/submit.sh > run.log 2>&1
```

The script will:

- Push your `aqr-finance/<tag>` branch to origin (force-with-lease).
- Submit a `vesslctl job create` that clones the cookbook at that branch,
  runs `accelerate launch ... train.py` followed by `eval.py`, with the
  cache volume mounted at `~/.cache/aqr-finance`.
- Poll `vesslctl job show` until the job reaches a terminal state, then
  dump the job's full log to stdout (which `> run.log` captures locally).
- Exit 0 if the job's final state is `succeeded`, non-zero otherwise.

**What you CAN modify**:

- `train.py`. LoRA rank, alpha, dropout, target modules (but read the
  WARNING below), learning rate, schedule shape (cosine / linear / wsd),
  warmup, weight decay, optimizer betas, batch size, grad accum, total
  steps, sequence length, gradient clipping.

**What you CANNOT modify**:

- `prepare.py`. The CC-MAIN-2017-W26 cutoff is the experiment's lookahead
  guarantee. Changing it invalidates the leakage premium measurement.
- `eval.py`. The chronological train/test split and Ridge regression
  define the leakage premium. Changing them invalidates cross-run
  comparisons.
- `batch-job/*.sh` for experimental reasons. They are the runner, not
  the experiment.
- `accelerate_config.yaml` unless you have a memory or sharding reason.

**WARNING — LoRA target modules**: The default target list:

```python
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

is required because Qwen3.5-35B-A3B-Base is a 40-layer hybrid (30 Gated
DeltaNet + 10 Gated Attention). With only `q/k/v/o`, the 30 DeltaNet layers
stay frozen, trainable drops to ~0.02 %, and the loss curve goes NaN
within the first few hundred steps. If you prune the target list, expect
the NaN watchdog to fire. Reference:
<https://github.com/shanemmattner/qwen-rft-pipeline#deltanet-lora-target-reference>

**Success metric**: lower `val_loss_final` is better in train.py terms,
but the cookbook's real metric is **`premium_reduction`** from `eval.py`:
the gap between `base_leakage_premium` (base model alone) and the adapter's
`leakage_premium`. If continued PT on the <= 2017-06 slice removed
lookahead bias, `premium_reduction > 0` is the quantitative proof.
`r2_leakage_off` (adapter, chronological split, higher = better
out-of-sample) is the secondary headline.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. A small improvement
from deleting code is a great outcome — keep it. A 0.001 `r2_leakage_off`
improvement that adds 20 lines of hacky LoRA target massaging? Probably
not worth it. A 0.001 improvement from deleting a misguided regularization
term? Definitely keep.

**NaN watchdog**: `train.py` aborts with exit code 42 if any loss is NaN
within the first `NAN_WATCHDOG_MINUTES` (30 min default) of the run. The
job's logs end with the abort line. Treat exit 42 as "this LoRA / lr / FSDP
config diverged" — record `crash` in `results.tsv`, revert the commit, and
move on. Cost loss is < $20 on 8xH100 at $4/GPU/h.

**The first run**: Your very first run should always be to establish the
baseline. Run `train.py` and `eval.py` unmodified.

## Output format

Once `train.py` + `eval.py` finish inside the job, the log contains two
summary blocks. Extract them with:

```
grep "^r2_leakage_off:\|^r2_leakage_on:\|^leakage_premium:\|^val_loss_final:\|^peak_vram_mb:\|^training_seconds:" run.log
```

Example:

```
--- train.py summary ---
val_loss_final:      2.4127
val_loss_first30min: 2.6189
training_seconds:    24310.3
total_seconds:       25840.7
peak_vram_mb:        72340.1
num_trainable_M:     19.1
num_trainable_pct:   0.0552
num_train_tokens_M:  1000.0
---
--- eval.py summary ---
r2_leakage_off:        0.0123        # adapter, chronological split
r2_leakage_on:         0.0418        # adapter, random 5-fold CV
leakage_premium:       0.0295        # adapter (r2_on - r2_off)
base_r2_leakage_off:   0.0089        # base only, chronological
base_r2_leakage_on:    0.0612        # base only, random 5-fold CV
base_leakage_premium:  0.0523        # base (r2_on - r2_off)
premium_reduction:     0.0228        # base_premium - adapter_premium >> 0 = good
n_test_samples:        1421
n_total_samples:       6000
n_unique_stocks:       200
dates_per_stock:       30
eval_seconds:          1820.5
---
```

Note: numbers depend on the GPU spec, FSDP wrap policy, and the cache
volume's data manifest. The recipe targets 8xH100 SXM single-node by
default.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated).

The TSV has a header row and these columns:

```
commit	val_loss_final	r2_leakage_off	r2_leakage_on	status	description
```

1. git commit hash (short, 7 chars)
2. `val_loss_final` from train.py (e.g. 2.4127) — use 9.9999 for crashes
3. `r2_leakage_off` from eval.py (e.g. 0.0123) — use -9.9999 for crashes
4. `r2_leakage_on` from eval.py (e.g. 0.0418) — use -9.9999 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_loss_final	r2_leakage_off	r2_leakage_on	status	description
a1b2c3d	2.4127	0.0123	0.0418	keep	baseline
b2c3d4e	2.3014	0.0156	0.0421	keep	bump LoRA rank 16 -> 32
c3d4e5f	2.5810	0.0089	0.0410	discard	prune LoRA targets to attention-only
d4e5f6g	9.9999	-9.9999	-9.9999	crash	double LoRA rank to 64 (NaN at step 240)
```

## The experiment loop

Same two modes as the autoresearch cookbook.

### Mode A: Linear (karpathy-style accept/reject)

Best for: depth-first iteration where each idea builds on the last,
debugging a single direction, or the very first run.

LOOP FOREVER:

1. Look at the git state: the current branch/commit you're on.
2. Tune `train.py` with an experimental idea.
3. `git commit` locally — `submit.sh` will push for you.
4. Run the experiment: `bash batch-job/submit.sh > run.log 2>&1`. This
   blocks for 6-10 hours.
5. Read out the results:
   `grep "^r2_leakage_off:\|^val_loss_final:" run.log`.
6. If grep is empty, the run crashed. `tail -n 80 run.log` to read the
   trace. NaN exit 42 = LoRA / lr / FSDP config; OOM = wrap policy or
   batch; VESSL scheduling error = retry the job.
7. Record the results in `results.tsv` (do not commit this file).
8. If `premium_reduction` improved (or `r2_leakage_off` if the base
   number didn't move), "advance" the branch — keep the commit.
9. If neither moved or both got worse, `git reset --hard HEAD~1` to
   discard the change.

### Mode B: Batch (parallel fan-out, pick best of K)

Best for: sweeps over hyperparameters (e.g. LR ∈ {1e-5, 2e-5, 4e-5},
LoRA rank ∈ {8, 16, 32}). Round duration is ~6-10 h instead of ~6-10 h × K.

Each round:

1. Look at the git state. Note current `aqr-finance/<tag>` HEAD as `BASE`.
2. Generate K candidate experiments (typically 3-4 given the per-run cost).
   For each `i` in 1..K:
   - `git checkout -b aqr-finance/<tag>-r<round>-<i> BASE`
   - Edit `train.py` for candidate `i`.
   - `git commit -m "round <round> cand <i>: <one-line description>"`
   - `slug_<i>=$(bash batch-job/submit-async.sh)`
3. Once all K are submitted, wait for them together:
   `bash batch-job/wait-jobs.sh "$slug_1" "$slug_2" ... > round.log 2>&1`
4. Parse `round.log`: extract `r2_leakage_off` per slug.
5. Pick the winner: highest `r2_leakage_off` that beats BASE. Crashes lose.
6. Advance the main branch to the winner, or leave alone if no candidate
   beat BASE.
7. Record all K candidates in `results.tsv`.
8. Optional cleanup of loser branches.

### Mode flexibility

The two modes can coexist in one tag's history (start with Mode B for a
coarse sweep, switch to Mode A for fine-tuning the best candidate), but
don't weave them within a single round.

**Timeout**: `submit.sh` enforces `AQR_TIMEOUT_S` (default 36000s / 10 h).
If the job hasn't finished by then, the script kills its polling loop;
the job continues running on VESSL — treat it as a failure (discard and
revert), and optionally `vesslctl job terminate <slug>` to stop billing.

**Crashes**: If a run crashes (NaN exit 42, OOM, etc.), use your judgment.
NaN within 30 min on the default target list usually means lr is too high
or LoRA rank is too aggressive — revert and lower. OOM with peak_vram_mb
> 75 GB / GPU usually means PER_DEVICE_BATCH too high or seq_len too long;
also try `fsdp_min_num_params` reduction.

**VESSL-specific failure modes**:

- **Image pull / scheduling delays**: occasionally the queue is slow.
  Don't treat slow start as a crash — wait for the timeout.
- **Volume mount issues**: if the cache volume isn't mounted correctly,
  `prepare.py` will start re-downloading FineWeb inside the job and the
  training step budget will never be reached. Check
  `vesslctl volume show "$AQR_CACHE_VOLUME"` if you see this.
- **Spec unavailability**: if `AQR_RESOURCE_SPEC` is at low availability,
  the job may sit in `queued` for a long time. Pick a different 8xH100
  spec slug.

**NEVER STOP**: Once the experiment loop has begun (after the initial
setup), do NOT pause to ask the human if you should continue. Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human
might be asleep, or gone from a computer and expects you to continue
working *indefinitely* until you are manually stopped. You are autonomous.
If you run out of ideas, think harder — read prepare.py / eval.py to
understand what signal you're trying to surface, try different LoRA target
subsets (always keeping the DeltaNet `in_proj_*` / `out_proj` group), try
different optimizers (Lion, paged AdamW), try shorter or longer schedules
within the same token budget.

Because each experiment on this recipe takes 6-10 h instead of 5 min, you
might only get 1-3 experiments per overnight session in Mode A. Use Mode B
with K=3-4 to multiply per overnight slot.
