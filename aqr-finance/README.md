# AQR-style point-in-time financial LLM on VESSL Cloud

LoRA continued pretraining on `Qwen/Qwen3.5-35B-A3B-Base` (MoE, 35 B total /
3 B active) against a **chronologically-filtered FineWeb slice** (Common
Crawl dumps published `<= 2017-06-30`), then a downstream eval on Kaggle JPX
that measures the **leakage premium** — the R² gap between a chronological
test split (the LLM has never seen the test years) and a stock-disjoint
GroupKFold split that isolates temporal leakage.

The recipe runs as **one VESSL Cloud batch job on a single H100 80 GB**,
driven end-to-end by `vesslctl`: edit `train.py` locally, `submit.sh`
pushes the branch and calls `vesslctl job create`, then you read the
metrics back from `vesslctl job logs`. The whole recipe is one `vesslctl`
batch job — there is no agent loop and no sweep; you drive the single run
yourself.

This is the **single-H100 LoRA** arm. A companion **multi-GPU full-weight**
arm (8×H100 FSDP2, every one of the 35 B parameters trained) lives in
[`multigpu/`](./multigpu/) and shares the same base model and the same
`eval.py` leakage contract.

| GPU | Cost | Wall time | Peak VRAM | Trainable params |
|-----|-----:|----------:|----------:|-----------------:|
| H100 SXM 80 GB × 1 | ~$60 (~$56 train + ~$4 eval) | ~23 h train + ~2 h eval | ~74 GB | ~945 M (~2.6 %) |

Measured 2026-05-20/21 on VESSL Cloud (single H100): 23 h 23 m train, final
loss 2.26; 1 h 49 m strict eval over 1000 stocks. Headline leakage premium
0.22 base / 0.14 adapter (both 95% CIs exclude zero); `premium_reduction`
0.08 with a 95% CI of [−0.04, 0.32] that crosses zero — leakage is real and
measurable, but one continued-PT pass does not significantly remove it. Full
table in [`benchmarks.md`](./benchmarks.md).

> **Cookbook narrative**: the full motivation (why point-in-time matters,
> what the leakage premium tells you, how the Kelly et al. point-in-time
> paper informs the cutoff choice) lives on the VESSL blog. This README is
> the developer-facing README — it explains *how to run* the recipe.

## What this recipe is

The point-in-time paper [Kelly, Malamud, Schwab & Xu, "Scaling Point-in-Time
Language Models," SSRN Working Paper No. 6681860](https://ssrn.com/abstract=6681860)
shows why open-web LLMs flatter themselves at financial prediction: their
pretraining corpus already encodes future returns, so a naive split leaks
look-ahead information. The remedy is to pretrain (or continue-pretrain) on a
corpus hard-clipped to dates from which the model cannot know future returns,
then evaluate downstream on later years.

This recipe is **inspired by** that work, not a replication of it. Kelly et
al. train a 4 B decoder from scratch on ~1 T tokens; here we run *continued*
pretraining of an existing 35 B base on ~1 B tokens. The point is not to
reproduce their model — it is to give you a small, single-H100 harness that
**measures the leakage premium** so you can see the effect for yourself.

**What "measure" means, and what this recipe does NOT promise.** The
deliverable here is the *measurement methodology*, not alpha and not a
leakage fix. Concretely (see [`benchmarks.md`](./benchmarks.md)): the leakage
premium is real and statistically significant for both the base model and the
continued-PT adapter (both 95% CIs exclude zero), but a single 1 B-token
continued-PT pass does **not** significantly reduce it (the
`premium_reduction` CI crosses zero). Treat this recipe as a measuring
instrument you can rerun and extend — not as a recipe that removes
look-ahead bias.

> **Measurement precision note:** the GroupKFold evaluation isolates
> stock identity but leaves dates mixed across folds, so the measured
> premium contains both temporal leakage and evaluation-construction noise.
> See [Known limitations](#known-limitations) for the open refinement lever
> and the companion [full-weight multigpu arm](./multigpu/) for robustness
> checks that confirm the headline finding.

The smallest reproducible version of the idea breaks into four scripts:

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
  Exchange Prediction: build a date-conditional LLM embedding per
  (stock, date), fit Ridge on (LLM emb || numeric features) → next-day
  return, and report R² on a chronological split (train <= 2020, test
  >= 2021) vs a **stock-disjoint GroupKFold** split. The gap is the
  leakage premium; a clustered bootstrap over stocks puts a 95% CI on it.
- **`batch-job/submit.sh`** wraps `vesslctl` so you don't touch the CLI
  flags by hand. It pushes the experiment branch to origin, the container
  clones it, runs `python train.py` (single process — the 35 B base + LoRA
  fits one H100) followed by `python eval.py`. You read the leakage-premium
  numbers back from `vesslctl job logs` (or the captured `run.log`).

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
- A single H100 80 GB resource spec (e.g. resourcespec-ch100x1). Find with
  `vesslctl resource-spec list` and `vesslctl cluster list`, then
  `export AQR_RESOURCE_SPEC=resourcespec-...`.
- A local shell with `git` and `bash` (the `batch-job/*.sh` scripts run from
  your machine and shell out to `vesslctl`).
- (Optional but recommended for `eval.py`) Kaggle creds —
  `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars, or `~/.kaggle/kaggle.json`
  pre-staged on the cache volume — to download the JPX dataset during
  `prep.sh`. If absent, `eval.py` will refuse to run unless
  `stock_prices.csv` is pre-staged at `~/.cache/aqr-finance/jpx/`.

## How to run

The recipe is one `vesslctl` batch job that you drive yourself — there is no
agent loop and no sweep.

```bash
# 1. One-time data prep (streams FineWeb, ~30-60 min CPU job).
bash batch-job/prep.sh

# 2. Cut an experiment branch, run the baseline once.
git checkout -b aqr-finance/my-run
bash batch-job/submit.sh > run.log 2>&1
grep "^r2_leakage_off:\|^r2_leakage_on:\|^val_loss_final:\|^peak_vram_mb:" run.log
```

`submit.sh` pushes your `aqr-finance/<tag>` branch to origin, the container
clones it at that branch, runs `train.py` then `eval.py`, and the script
polls until the job reaches a terminal state and dumps the full job log to
`run.log`. If `r2_leakage_off` and `r2_leakage_on` show up, the recipe is
wired correctly. To try a variant, edit `train.py`, commit, and rerun
`submit.sh` on a fresh tag.

## How `submit.sh` works

```
edit train.py → git commit
                     ↓
              bash batch-job/submit.sh
                     ↓
   git push origin aqr-finance/<tag>
                     ↓
   vesslctl job create --object-volume CACHE:/root/.cache/aqr-finance
                     ↓
   container: clone cookbook @ branch, pip install deps,
              python train.py  (single process — fits one H100),
              python eval.py
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
run overwrites `~/.cache/aqr-finance/adapter/`.

> **No bundled dataset, no notebook.** Unlike the template's
> `notebook/` + `data/DATASET_CARD.md` layout, this recipe **streams** its
> inputs at runtime — the FineWeb slice and the Kaggle JPX prices are
> downloaded inside the job — so there is nothing to bundle and no notebook
> walk-through. This is an intentional divergence from the template, noted
> here so reviewers don't look for files that don't exist.

## Configuration

`prep.sh`, `submit.sh`, and `submit-async.sh` read these env vars:

| Var | Required | Default |
|---|---|---|
| `AQR_CACHE_VOLUME` | yes | — |
| `AQR_RESOURCE_SPEC` | yes for submit.sh | — (set to your single-H100 spec, e.g. resourcespec-ch100x1) |
| `AQR_IMAGE` | no | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| `AQR_REPO_URL` | no | `https://github.com/vessl-ai/vessl-cloud-cookbook.git` |
| `AQR_BRANCH` | no | `main` (prep.sh only) |
| `AQR_TARGET_TOKENS` | no | `1B` (prep.sh only) |
| `AQR_TIMEOUT_S` | no | `36000` (submit.sh, wait-jobs.sh) |
| `AQR_POLL_INTERVAL_S` | no | `60` (wait-jobs.sh) |

To run the prep job on a CPU spec (the default `resourcespec-a100cpu`
is fine), set `AQR_RESOURCE_SPEC` separately for prep vs submit.

## LoRA target list — why it has 12 entries instead of 4

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
    "lm_head", "embed_tokens",                          # full vocab, smaller embedding LR (CPT)
]
```

The ten rank-16 LoRA modules are only ~19 M params, but `lm_head` + `embed_tokens`
train as full matrices (Unsloth's CPT pattern, smaller embedding LR), taking the
real run to ~945 M trainable (~2.6 %) and a 5.83 GB adapter. Continued PT trains stably. The first 30 minutes of any run is the NaN watchdog
window — if you see `train.py: NaN train loss at step N — ABORT (exit 42)`,
your edits probably broke this invariant.

## Known limitations

- **Per-run cost is real.** A full run is ~$56 on a single H100 SXM at
  typical cloud rates (~23 h train) plus a ~$4 strict eval (~$60 total).
  At ~23 h a run is effectively a full-day slot — one run at a time, not a
  sweep. Treat each run as expensive and plan accordingly.
- **Measurement, not removal.** This recipe measures the leakage premium; it
  does not promise to remove it.
- **GroupKFold is a proxy, not full temporal isolation.** The stock-disjoint
  split prevents the evaluation from being gamed by memorizing stock
  identities, but the training folds still contain data from multiple time
  periods — dates can mix. The measured premium therefore captures both real
  temporal leakage and noise from the evaluation construction itself. Robustness
  checks (embargoed split boundary, walk-forward) were run on the full-weight
  arm and confirm the premium is not a split-edge artifact — see the
  [multigpu benchmarks](./multigpu/benchmarks.md). The open refinement lever is
  replacing the leaky-side GroupKFold with a purged time-series CV that does not
  mix dates; that is the most direct way to tighten the CI. The honest result is that one 1 B-token
  continued-PT pass does not significantly cut the premium (see
  [`benchmarks.md`](./benchmarks.md)). Don't read the adapter's lower point
  estimate as a fix.
- **Kaggle creds must be staged for `eval.py`.** If `prep.sh` skipped the
  JPX download (no creds), `eval.py` will refuse to run. Set
  `KAGGLE_USERNAME` + `KAGGLE_KEY` in your local shell **before** running
  `prep.sh` — the script forwards them into the container via
  `vesslctl job create --env`. Alternatively, place `stock_prices.csv`
  at `~/.cache/aqr-finance/jpx/` on the cache volume manually.
- **Eval uses date-conditional prompts + base-vs-adapter A/B.** `eval.py`
  builds one prompt per (stock, date) pair embedding the date plus recent
  5-day / 30-day log returns and 20-day volatility, then compares the
  chronological vs **stock-disjoint GroupKFold** R^2 gap (leakage premium)
  for BOTH the base model AND the LoRA-trained adapter. GroupKFold (not a
  plain shuffle KFold) keeps each stock out of both train and test, so the
  premium reflects temporal leakage rather than a memorized stock fixed
  effect. The publishable signal is `premium_reduction = base_premium -
  adapter_premium`, reported with a clustered-bootstrap 95% CI (resample
  stocks). Sample size = `AQR_MAX_STOCKS` (default 1000) × `DATES_PER_STOCK`
  (default 30); raise MAX_STOCKS to tighten the CI.
- **`Qwen3.5-35B-A3B-Base` requires `trust_remote_code=True`.** The
  hybrid DeltaNet + Gated Attention architecture isn't (yet) upstream in
  `transformers`. Pin the model card revision in production.
- **Single-process training, single-GPU adapter.** The recipe runs
  `python train.py` on one H100 (multi-GPU DDP/FSDP of the 35 B base OOM'd
  — see above), so the LoRA adapter is a standard single-GPU PEFT
  `save_pretrained` that `eval.py` loads directly. `accelerate_config.yaml`
  is retained in-tree for reference but is unused by the single-process path.
- **Branch hygiene.** Each run lives on its own `aqr-finance/<tag>` branch
  and pushes to origin. Use a fresh tag per run so reruns don't clobber a
  previous run's commits.
- **Cost is unbounded by default.** Each run is real spend at this
  cost-per-run. Set a daily-cap routine on `vesslctl billing show`.

## Further reading

- Companion [`multigpu/`](./multigpu/) recipe — the multi-GPU full-weight
  (8×H100 FSDP2) arm, same base model and same `eval.py` leakage contract.
- Kelly, Malamud, Schwab & Xu, "Scaling Point-in-Time Language Models,"
  SSRN Working Paper No. 6681860: <https://ssrn.com/abstract=6681860>
- He, Lv, Manela & Wu, "Chronologically Consistent Large Language Models"
  (ChronoBERT / ChronoGPT), arXiv:2502.21206: <https://arxiv.org/abs/2502.21206>
- Upstream Qwen3.5 model card: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base>
- Kaggle JPX competition: <https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction>
- DeltaNet LoRA target reference: <https://github.com/shanemmattner/qwen-rft-pipeline#deltanet-lora-target-reference>
- VESSL Cloud docs: <https://docs.cloud.vessl.ai>
- Sibling [`autoresearch`](../autoresearch/) cookbook — a different VESSL
  Cloud recipe (an agent-driven experiment sweep). Not how you run *this*
  recipe; linked only for readers interested in the cheap-sweep pattern.
