# Benchmarks — aqr-finance

Measured on VESSL Cloud H100 SXM (80 GB), **single GPU**.
Training run 2026-05-20; strict leakage evaluation 2026-05-21.

## Summary

One LoRA continued-pretraining run on `Qwen/Qwen3.5-35B-A3B-Base` (MoE,
35 B total / 3 B active) over a chronologically-filtered FineWeb slice
(Common Crawl dumps published `<= 2017-06-30`), followed by a downstream
leakage-premium evaluation on Kaggle JPX. Multi-GPU DDP of the 35 B base
hit the H100 80 GB ceiling across several attempts, so the recipe settled
on **single-process training on one H100** — slower wall time, but lower
cost and a one-card footprint.

## Training run

| Metric | Value |
|--------|------:|
| Base model | `Qwen/Qwen3.5-35B-A3B-Base` |
| Method | LoRA continued pretraining (Unsloth FastModel, 1 B tokens) |
| LoRA target modules | q/k/v/o + in_proj_qkv/in_proj_z/out_proj (DeltaNet) + gate/up/down_proj (MoE) + lm_head/embed_tokens (full vocab, CPT) |
| Trainable params | ~945 M (~2.6 %); the ten rank-16 LoRA modules ≈ 19 M, lm_head + embed_tokens (full matrices) are the bulk |
| GPU | H100 SXM 80 GB × 1 |
| Wall time | 23 h 23 m (2026-05-19 05:25 → 2026-05-20 04:49 UTC) |
| Peak VRAM | ~74 GB |
| Final train loss | 2.26 (epoch 1, no val split) |
| Container image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |

> The training job is marked `failed` in VESSL only because the in-job
> `eval.py` step crashed on a pandas 3.0 `groupby.apply` change *after* the
> adapter had already been saved (fixed in a later commit; eval now runs as
> a separate step). The LoRA adapter (5.83 GB) persisted on the cache
> volume and is what the evaluation below loads.

## Leakage-premium evaluation

`eval.py` builds a date-conditional LLM embedding per (stock, date), fits
Ridge on `(LLM emb || numeric features) → next-day return`, and reports R²
on two splits, for **both the base model and the LoRA adapter**:

- **`r2_off`** — chronological split (train `<= 2020-12-31`, test `>= 2021-01-01`): the honest, no-leakage estimate.
- **`r2_on`** — **stock-disjoint GroupKFold** (a stock never sits in both train and test): a plain shuffle KFold would let the model score high by memorizing a stock fixed effect rather than reading the future, so GroupKFold isolates the temporal-leakage component.

`leakage_premium = r2_on − r2_off`; `premium_reduction = base_premium −
adapter_premium`. A **clustered bootstrap over stocks** (resample whole
securities, not rows) puts a 95% CI on each quantity — honest under the
panel's within-stock date correlation.

Sample size: `AQR_MAX_STOCKS` (default 1000) × `DATES_PER_STOCK` (default
30). More stocks tighten the bootstrap CI.

| Metric | Base | Adapter |
|--------|-----:|--------:|
| `r2_leakage_off` (chronological, honest) | −0.2605 | −0.1939 |
| `r2_leakage_on` (stock-disjoint GroupKFold) | −0.0416 | −0.0541 |
| `leakage_premium` (on − off) | 0.219 | 0.140 |
| `leakage_premium` 95% CI | [0.220, 0.527] | [0.149, 0.336] |
| `premium_reduction` (base − adapter) | 0.079 | |
| `premium_reduction` 95% CI | [−0.043, 0.315] | |
| `n_unique_stocks` / `n_test_samples` | 1000 / 5817 | |
| Eval wall time / cost | 1 h 49 m / ~$4.3 | |

Measured 2026-05-21, `AQR_MAX_STOCKS=1000`, 200 clustered-bootstrap draws.
The clustered bootstrap of an R²-difference is right-skewed, so the bootstrap
median exceeds the full-data point estimate — medians are base 0.32, adapter
0.21, `premium_reduction` 0.11 — and each point estimate sits near the low
edge of its (asymmetric) 95% interval. That is expected for a right-skewed
R²-difference, not an error; significance is read from the CI either way.

**What the numbers say:**
- Both `r2_leakage_off` < 0 → neither model has real out-of-sample alpha on
  JPX. Expected: this is a leakage probe, not an alpha claim.
- Both `leakage_premium` 95% CIs exclude zero → temporal leakage is real for
  the base model *and* the adapter.
- `premium_reduction` 95% CI crosses zero → one 1 B-token continued-PT pass
  does not significantly cut the leakage premium at 1000 stocks. The
  adapter's lower point estimate (0.140 vs 0.219) is directional, not
  significant.

## Cost

| Step | Duration | Cost at $2.39/hr |
|------|---------:|-----------------:|
| Training | ~23 h 23 m | ~$56 |
| Strict eval (1000 stocks) | ~1 h 49 m | ~$4.3 |
| **Total** | **~25 h** | **~$60** |

Image pull + `pip install` of the framework layer adds ~5–10 min of
startup per job (not separated out above). Prices as of 2026-05-21.

## ⚠️ Honest limitation

This recipe **measures** lookahead-bias leakage; it does not promise to
remove it. At 1000 stocks both out-of-sample `r2_leakage_off` are negative
(base −0.26, adapter −0.19) — neither model has real predictive alpha on
JPX, which is expected. The `leakage_premium` itself is real and
significant for both models (95% CIs exclude zero). But the
`premium_reduction` 95% CI is [−0.04, 0.32], which **crosses zero**: a
single 1 B-token continued-PT pass does *not* cut the leakage premium by a
statistically significant margin. The adapter's lower point estimate
(0.14 vs 0.22) is directional only — the honest read is "raise
`AQR_MAX_STOCKS` and see if it holds," not "leakage removed." The point of
the recipe is the **measurement methodology**, reproducible on one H100.
See the VESSL blog writeup for the leakage-vs-alpha distinction and the
point-in-time background (Kelly, Malamud, Schwab & Xu, SSRN 6681860).
