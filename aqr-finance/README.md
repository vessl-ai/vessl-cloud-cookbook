# AQR-style point-in-time financial LLM on VESSL Cloud

LoRA continued pretraining on `Qwen/Qwen3.5-35B-A3B-Base` (MoE, 35 B total /
3 B active) against a **chronologically-filtered FineWeb slice** (Common
Crawl dumps published `<= 2017-06-30`), then a downstream eval on Kaggle JPX
that measures the **leakage premium** — the R² gap between a chronological
test split (the LLM has never seen the test years) and a random-shuffle
split (model capacity ceiling).

The recipe runs as one VESSL Cloud batch job on 8xH100 SXM single-node,
in the same shape as the [autoresearch](../autoresearch/) cookbook: edit
`train.py` locally, the runner submits the job, you read back the metrics.

| GPU | Cost / experiment | Wall time per experiment | Peak VRAM | Trainable params |
|-----|------------------:|-------------------------:|----------:|-----------------:|
| H100 SXM 80 GB × 8 | ~$110-$170 | ~6-10 h (5-9 h train + ~1 h startup/eval) | ~70-75 GB / GPU | ~19 M (0.055 %) |

Numbers are pre-dry-run estimates. Real values land in
[`results.tsv`](./results.tsv) after the baseline run.

> **Cookbook narrative**: the full motivation (why point-in-time matters,
> what the leakage premium tells you, how Bryan Kelly's AQR paper informs
> the cutoff choice) lives on Notion. This README is the developer-facing
> README — it explains *how to run* the recipe.

## What this recipe is

The AQR paper [Kelly, Malamud, Pedersen 2025](https://www.aqr.com/Insights/Research/Working-Paper/Financial-Machine-Learning) measures
*leakage premium*: open-web LLMs cheat at financial prediction because their
pretraining corpus encodes future returns. The fix: pretrain on a corpus
hard-clipped to dates the LLM cannot possibly know future returns from.
Then evaluate downstream on later years.

This recipe is the smallest reproducible version of that idea:

- **`prepare.py` streams FineWeb** and keeps only CC dumps `<= 2017-W26`
  (~2017-06-26 to 2017-06-30). Tokenizes with the Qwen3.5 tokenizer.
  Writes uint32 shards to the cache volume.
- **`train.py` does LoRA continued PT** on `Qwen/Qwen3.5-35B-A3B-Base` —
  the OSS pure-Base release before Qwen3.6's post-trained unified lineup.
  LoRA target list covers both Gated DeltaNet (30 layers) and Gated
  Attention (10 layers) plus the MoE expert MLPs. Without the DeltaNet
  `in_proj_*` / `out_proj` targets, 75 % of the model stays frozen and
  the loss diverges to NaN within a few hundred steps.
- **`eval.py` measures the leakage premium** on Kaggle JPX Tokyo Stock
  Exchange Prediction: compute one LLM embedding per stock, fit Ridge on
  (LLM emb || log_close) → next-day return rank, report R² on a
  chronological split (>= 2021) and on random 5-fold CV.
- **`batch-job/submit.sh`** wraps `vesslctl` so the agent never touches
  the CLI directly. Force-with-lease pushes the experiment branch to
  origin, the container clones it, runs `accelerate launch train.py`
  followed by `eval.py`.

You analyze results in `results.tsv`.

## Prerequisites

- A VESSL Cloud account with credits.
- `vesslctl` installed and authenticated (`vesslctl auth status`).
- A VESSL org and team active on `vesslctl`. The interactive `vesslctl auth
  login` flow prompts you to pick both, and `vesslctl auth status` shows
  the resolved context. To change them later:
  ```bash
  vesslctl config set default_org  <your-org>
  vesslctl config set default_team <your-team>
  vesslctl auth status   # confirm
  ```
- An object volume to hold the data cache (~5-10 GB for 1 B tokens + JPX).
  Create one once:
  ```bash
  vesslctl volume create \
    --name aqr-finance-cache \
    --storage <your-object-storage-slug> \
    --teams <your-team>
  vesslctl volume list   # grab the new volume's slug
  export AQR_CACHE_VOLUME=objvol-...
  ```
- An 8xH100 SXM single-node resource spec. Find with
  `vesslctl resource-spec list` and `vesslctl cluster list`, then
  `export AQR_RESOURCE_SPEC=resourcespec-...`.
- A coding agent that can run shell commands locally (Claude Code,
  Codex, Cursor, etc.).
- (Optional but recommended for `eval.py`) Kaggle creds —
  `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars, or `~/.kaggle/kaggle.json`
  pre-staged on the cache volume — to download the JPX dataset during
  `prep.sh`. If absent, `eval.py` will refuse to run unless
  `stock_prices.csv` is pre-staged at `~/.cache/aqr-finance/jpx/`.

## Two ways to run

### Path A — Drive the run yourself (sanity check)

```bash
# 1. One-time data prep (streams FineWeb, ~30-60 min CPU job).
bash batch-job/prep.sh

# 2. Cut a branch, run the baseline once.
git checkout -b aqr-finance/sanity-check
bash batch-job/submit.sh > run.log 2>&1
grep "^r2_leakage_off:\|^val_loss_final:\|^peak_vram_mb:" run.log
```

If `r2_leakage_off` and `r2_leakage_on` show up, the recipe is wired
correctly.

### Path B — Hand it to the agent (overnight loop)

```bash
# In your coding agent (Claude Code, Codex, etc.), with this directory open:
> Have a look at program.md and let's kick off a new experiment.
> Let's do the setup first.
```

The agent reads `program.md`, walks the setup checklist (cuts a branch,
verifies the cache volume), then enters the experiment loop. Every
iteration is one `bash batch-job/submit.sh` call. You wake up to a
populated `results.tsv` and an `aqr-finance/<tag>` branch on
`vessl-cloud-cookbook` with one commit per kept experiment.

Because one experiment is 6-10 h (vs autoresearch's 5 min), expect
1-3 Mode A experiments per overnight slot. Use Mode B (K=3-4) for sweeps.

## How `submit.sh` works

```
agent edits train.py → git commit
                     ↓
              bash batch-job/submit.sh
                     ↓
   git push origin aqr-finance/<tag>   (force-with-lease)
                     ↓
   vesslctl job create --object-volume CACHE:/root/.cache/aqr-finance
                     ↓
   container: clone cookbook @ branch, uv sync,
              accelerate launch --config_file accelerate_config.yaml train.py,
              uv run eval.py
                     ↓
   poll vesslctl job show until terminal state
                     ↓
   vesslctl job logs --limit 1000 → captured to local run.log
                     ↓
   exit 0 if job succeeded, non-zero otherwise
```

The cache volume holds `~/.cache/aqr-finance` between jobs, so
`prepare.py` runs only once (in `prep.sh`) and every subsequent
`submit.sh` invocation skips it. The trained LoRA adapter from each
experiment overwrites `~/.cache/aqr-finance/adapter/`.

## Configuration

`prep.sh`, `submit.sh`, and `submit-async.sh` read these env vars:

| Var | Required | Default |
|---|---|---|
| `AQR_CACHE_VOLUME` | yes | — |
| `AQR_RESOURCE_SPEC` | yes for submit.sh | — (set to your 8xH100 SXM spec) |
| `AQR_IMAGE` | no | `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel` |
| `AQR_REPO_URL` | no | `https://github.com/vessl-ai/vessl-cloud-cookbook.git` |
| `AQR_BRANCH` | no | `main` (prep.sh only) |
| `AQR_TARGET_TOKENS` | no | `1B` (prep.sh only) |
| `AQR_TIMEOUT_S` | no | `36000` (submit.sh, wait-jobs.sh) |
| `AQR_POLL_INTERVAL_S` | no | `60` (wait-jobs.sh) |

To run the prep job on a CPU spec (the default `resourcespec-a100cpu`
is fine), set `AQR_RESOURCE_SPEC` separately for prep vs submit.

## LoRA target list — why it has 10 entries instead of 4

`Qwen/Qwen3.5-35B-A3B-Base` is a 40-layer **hybrid attention** model:

- 30 layers of **Gated DeltaNet** (linear attention, projection modules
  named `in_proj_qkv`, `in_proj_z`, `out_proj`).
- 10 layers of **Gated Attention** (standard softmax attention,
  projections `q_proj`, `k_proj`, `v_proj`, `o_proj`).
- Every layer has a MoE expert block with `gate_proj`, `up_proj`,
  `down_proj`.

The "standard" LoRA target list `[q_proj, k_proj, v_proj, o_proj]` only
hits the 10 attention layers (25 % of the model). The 30 DeltaNet layers
stay frozen — trainable params drop to ~0.02 %, and the loss curve goes
NaN within a few hundred steps. This is documented in
[shanemmattner/qwen-rft-pipeline](https://github.com/shanemmattner/qwen-rft-pipeline#deltanet-lora-target-reference)
which provides the verified target list:

```python
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",            # Gated Attention
    "in_proj_qkv", "in_proj_z", "out_proj",            # Gated DeltaNet
    "gate_proj", "up_proj", "down_proj",               # MoE expert MLP
]
```

This brings trainable params to ~19 M (~0.055 %, ~250 MB), and continued
PT trains stably. The first 30 minutes of any run is the NaN watchdog
window — if you see `train.py: NaN train loss at step N — ABORT (exit 42)`,
your edits probably broke this invariant.

## Known limitations

- **Per-experiment cost is real.** Each run is ~$110-$170 on 8xH100 SXM
  at typical cloud rates. A single overnight Mode A session is 1-3
  experiments; Mode B with K=3 is one round. Treat each experiment as
  expensive and plan accordingly.
- **No autoresearch "5-min budget" knob here.** The autoresearch cookbook
  caps each run at 5 min of training, which lets the agent run 50+
  experiments per night on one GPU. This recipe is 5-9 h per run. The
  loop shape is the same, but the cadence is slower.
- **Kaggle creds must be staged for `eval.py`.** If `prep.sh` skipped the
  JPX download (no creds), `eval.py` will refuse to run. Set
  `KAGGLE_USERNAME` + `KAGGLE_KEY` in your local shell **before** running
  `prep.sh` — the script forwards them into the container via
  `vesslctl job create --env`. Alternatively, place `stock_prices.csv`
  at `~/.cache/aqr-finance/jpx/` on the cache volume manually.
- **Eval baseline is a first cut, not a production signal.** `eval.py`
  encodes one LLM embedding per *stock identifier* (constant across dates
  of that stock), which from the regression's point of view is effectively
  a stock fixed effect. The numbers are useful for "does the adapter shift
  R^2 vs the base model" sanity checks, but the AQR-style date-conditional
  signal needs a richer prompt (latest news headline, recent price moves)
  that we'll layer in a follow-up.
- **`Qwen3.5-35B-A3B-Base` requires `trust_remote_code=True`.** The
  hybrid DeltaNet + Gated Attention architecture isn't (yet) upstream in
  `transformers`. Pin the model card revision in production.
- **FSDP + PEFT save semantics.** `accelerate_config.yaml` uses
  `fsdp_state_dict_type: SHARDED_STATE_DICT`. The PEFT `save_pretrained`
  call inside `train.py` writes the adapter via the accelerator's save
  wrapper, but `eval.py` reads it on a single GPU. If you see
  shape-mismatch errors loading the adapter, switch to
  `FULL_STATE_DICT` in the config (slower save, more memory at save time).
- **Branch hygiene.** Each experiment runs entirely on an
  `aqr-finance/<tag>` branch and force-pushes to origin. Do not run two
  agents on the same tag concurrently — the second one will clobber the
  first's commits.
- **Cost is unbounded by default.** A runaway loop = real spend at this
  cost-per-run. Set a daily-cap routine on `vesslctl billing show`.

## Further reading

- Upstream Qwen3.5 model card: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base>
- Bryan Kelly et al. (AQR), "Financial Machine Learning": <https://www.aqr.com/Insights/Research/Working-Paper/Financial-Machine-Learning>
- Kaggle JPX competition: <https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction>
- DeltaNet LoRA target reference: <https://github.com/shanemmattner/qwen-rft-pipeline#deltanet-lora-target-reference>
- VESSL Cloud docs: <https://docs.cloud.vessl.ai>
- Sibling [`autoresearch`](../autoresearch/) cookbook — same loop shape,
  shorter experiments.
