# autoresearch benchmarks

Measured numbers from real VESSL Cloud runs. Update this file after each
end-to-end smoke test on a new spec or image.

## Headline (baseline `train.py`, no agent edits)

| Date | GPU | Cluster | Image | Train time | Job wall time | Peak VRAM | val_bpb | Cost |
|------|-----|---------|-------|-----------:|--------------:|----------:|--------:|-----:|
| 2026-05-03 | A100 SXM ×1 | betelgeuse-na | pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel | 5m1s | 8m46s | 44.0 GB | 1.109645 | ~$0.23 |

Numbers above are from a single run of the unmodified upstream `train.py`
submitted via `bash batch-job/submit.sh`. The "Train time" is the 5-minute
training budget enforced by `prepare.py`; "Job wall time" includes image
pull, repo clone, `uv sync`, torch.compile, and the final `evaluate_bpb` —
this is what you actually pay for at $1.55/hr.

For reference, karpathy's H100 numbers (from his README): val_bpb ≈ 0.997900,
total_tokens ≈ 500 M, MFU ≈ 39.8%. The A100 here sees ~200 M tokens at MFU
≈ 15.7%, so the headline gap (1.11 vs 1.00 val_bpb) is roughly what you'd
expect from a GPU running ~2.5× fewer tokens in the same 5-minute budget.

Run details (job `job-no6r43nrqlcr`):
```
val_bpb:          1.109645
training_seconds: 300.7
total_seconds:    381.1   # train.py-internal; excludes apt/uv overhead
peak_vram_mb:     45012.5
mfu_percent:      15.71
total_tokens_M:   200.8
num_steps:        383
num_params_M:     50.3
depth:            8
```

## One-time prep cost

| Date | GPU | Cluster | Wall time | Cache size after |
|------|-----|---------|----------:|-----------------:|
| 2026-05-03 | CPU only (`resourcespec-a100cpu`) | betelgeuse-na | 5m4s | ~1.0 GB (10 train shards × ~92 MB + val shard + tokenizer) |

`prep.sh` downloads ~10 ClimbMix shards and trains the BPE tokenizer into
`AUTORESEARCH_CACHE_VOLUME`. You pay this once per cache volume. CPU spec
is the right default — the work is single-threaded BPE training, a GPU
would idle.

> **Pricing note**: `vesslctl billing show` and `vesslctl job create`
> output an internal credit rate that does not match public/customer
> pricing. Refer to the [VESSL Cloud pricing page](https://vessl.ai/pricing)
> for current rates. The $1.55/hr figure used above is for A100 SXM ×1.

Run details (job `job-dgwb4asbl100`).

## Per-experiment overhead breakdown

Helpful for predicting how many experiments fit in an N-hour overnight run.
Numbers from the 2026-05-03 smoke run on A100 SXM ×1 (job `job-no6r43nrqlcr`).

| Phase | Approx. time |
|-------|-------------:|
| `vesslctl job create` → container start | 5 s |
| `apt-get install git curl` + `curl ... uv install` | ~50 s |
| `uv sync` (torch 2.9.1 cu128 reinstall) | ~55 s |
| `prepare.py` skip + `train.py` startup + torch.compile | ~70 s |
| **Training** (fixed budget) | **5 min** |
| `evaluate_bpb` | ~40 s |
| **Total job wall** | **~8m46s** |

Net throughput: ~6.8 experiments/hour, vs ~12/hour on a dedicated local GPU
where the startup phases don't repeat. Over an 8-hour overnight run, expect
~50 completed experiments.

## Repro

```bash
export AUTORESEARCH_CACHE_VOLUME=objvol-...   # your seeded cache volume
git checkout -b autoresearch/bench-baseline
bash batch-job/submit.sh > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:\|^total_seconds:\|^training_seconds:" run.log
```

`vesslctl job show <slug>` reports the exact start/end timestamps used to
fill in "Job wall time" and "Cost" above.
