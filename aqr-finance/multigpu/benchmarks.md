# Benchmarks — aqr-finance / multigpu (full-weight, 8×H100 FSDP2)

Measured on VESSL Cloud H100 SXM (80 GB), **8 GPUs, single node**, axolotl
`0.16.2.dev0` + FSDP2.

## Summary

One **full-weight** continued-pretraining run on
`Qwen/Qwen3.5-35B-A3B-Base` (MoE, 35 B total / 3 B active) over a
chronologically-filtered FineWeb slice (Common Crawl dumps published
`<= 2017-06-30`), training **all 35 B parameters** via FSDP2 FULL_SHARD across
8×H100. Evaluated with the parent recipe's `eval.py` (same leakage contract) so
the full-weight checkpoint can be A/B'd against the single-H100 LoRA adapter.

## Training run

| Metric | Value |
|--------|------:|
| Base model | `Qwen/Qwen3.5-35B-A3B-Base` |
| Method | Full-weight continued PT (axolotl + FSDP2, ~1 B tokens) |
| Trained params | 35 B (all) |
| GPU | H100 SXM 80 GB × 8 (single node) |
| Steps / epochs | 56,430 steps / epoch 1.0 |
| `train_runtime` | 18 h 36 m |
| Final train loss | 2.182 |
| Peak VRAM | ~51 GB / GPU (activation checkpointing OFF) |
| axolotl version | `0.16.2.dev0` |
| Container image | `axolotlai/axolotl-uv:main-latest` |

The merged checkpoint is produced by a **separate single-GPU offline merge
job** — gathering 35 B shards inside the distributed job hits the NCCL watchdog
timeout (see the recipe README). The training job saves
`state_dict_type: SHARDED_STATE_DICT` to the persistent volume.

## Leakage-premium evaluation

Same `eval.py` as the parent recipe: date-conditional LLM embedding per
(stock, date), Ridge on `(LLM emb || numeric features) → next-day return`, R²
on a chronological split (`r2_leakage_off`, the honest estimate) vs a
**stock-disjoint GroupKFold** split (`r2_leakage_on`), for **both the base
model and the full-weight checkpoint**. A **clustered bootstrap over stocks**
(resample whole securities) puts a 95% CI on each quantity.

Eval config: 1000 stocks, `n_test = 5817`, 200 stock-clustered bootstrap draws.

| Metric | Base | Full-weight |
|--------|-----:|------------:|
| `r2_leakage_off` (chronological, honest) | −0.1936 | −0.1577 |
| `r2_leakage_on` (stock-disjoint GroupKFold) | −0.0562 | −0.0678 |
| `leakage_premium` median [95% CI] | 0.20 [0.13, 0.33] | 0.13 [0.07, 0.27] |
| `premium_reduction` (base − full-weight) median [95% CI] | 0.07 [−0.06, 0.22] | |

`premium_reduction`'s 95% CI **crosses zero** → the reduction is **not
statistically significant**.

## Cost

| Step | GPU | Rate | Duration | Cost |
|------|-----|-----:|---------:|-----:|
| Training | 8×H100 SXM | $19.12/hr | ~18 h 36 m | ~$378 |
| Offline merge | single H100 | $2.39/hr | — | ~$1.5 |
| Eval | single H100 | $2.39/hr | — | ~$7 |
| **Total** | | | | **~$386** |

One-time data prep (point-in-time FineWeb JSONL) is ~1 h on CPU (~$19),
amortized across reruns because the JSONL persists on the volume.

> **⚠️ The $329 mistake.** A first attempt at this run forgot the
> `--object-volume` flag. The job wrote its checkpoint to **ephemeral pod
> storage**, which was lost on pod terminate — ~$329 of compute gone with
> nothing to evaluate. The `--object-volume` mount is a hard prerequisite; the
> submit script keeps it and `volume_inspect_submit.sh` exists to confirm the
> checkpoint actually landed on the persistent volume.

Prices as of the measured run.

## ⚠️ Honest limitation

- **No alpha.** Both `r2_leakage_off` are negative (base −0.1936, full-weight
  −0.1577) — neither model has real out-of-sample predictive power on JPX.
  Expected: this is a leakage probe, not an alpha claim.
- **Leakage is real.** Both `leakage_premium` 95% CIs exclude zero → temporal
  leakage is present for the base model *and* the full-weight checkpoint.
- **One full-weight pass does not significantly reduce the premium.** The
  `premium_reduction` CI is [−0.06, 0.22], which crosses zero. The full-weight
  point estimate (0.07) is directional only.
- **Training every parameter did NOT beat the LoRA arm.** Going from ~2.6 %
  trainable (LoRA) to 100 % (full-weight) did not produce a significant
  reduction in measured leakage. The lever on this signal is the **eval
  protocol**, not the trainable-parameter count. The deliverable is the
  measurement methodology, reproducible on 8×H100.
