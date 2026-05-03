# autoresearch benchmarks

Measured numbers from real VESSL Cloud runs.

## Headline (baseline `train.py`, no agent edits)

| Date | GPU | Cluster | Image | Train time | Job wall time | Peak VRAM | val_bpb | Cost |
|------|-----|---------|-------|-----------:|--------------:|----------:|--------:|-----:|
| 2026-05-03 | H100 SXM ×1 | deneb-kr | pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel | 5m0s | 8m21s | 44.0 GB | 1.010748 | ~$0.33 |

Numbers above are from a single run of the unmodified upstream `train.py`
submitted via `bash batch-job/submit.sh`. The "Train time" is the 5-minute
training budget enforced by `prepare.py`; "Job wall time" includes image
pull, repo clone, `uv sync`, torch.compile, and the final `evaluate_bpb` —
this is what you actually pay for at $2.39/hr.

For reference, karpathy's H100 numbers (from his README): val_bpb ≈ 0.997900,
total_tokens ≈ 500 M, MFU ≈ 39.8%. The H100 here sees ~396 M tokens at MFU
≈ 31.5%, so the headline numbers are within ~1.3% of karpathy's reference —
likely down to driver / hardware revision differences.

Run details (job `job-l3i6bc25mrgt`):
```
val_bpb:          1.010748
training_seconds: 300.3
total_seconds:    356.9   # train.py-internal; excludes apt/uv overhead
peak_vram_mb:     45060.2
mfu_percent:      31.47
total_tokens_M:   396.4
num_steps:        756
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
would idle. Object storage is cross-cluster, so the prep job's cluster
doesn't have to match where you'll run training.

> **Pricing note**: `vesslctl billing show` and `vesslctl job create`
> output an internal credit rate that does not match public/customer
> pricing. Refer to the [VESSL Cloud pricing page](https://vessl.ai/pricing)
> for current rates. The $2.39/hr figure used above is for H100 SXM ×1.

## Per-experiment overhead breakdown

Helpful for predicting how many experiments fit in an N-hour overnight run.
Numbers from the 2026-05-03 baseline run on H100 SXM ×1 (job `job-l3i6bc25mrgt`).

| Phase | Approx. time |
|-------|-------------:|
| `vesslctl job create` → container start | 5 s |
| `apt-get install git curl` + `curl ... uv install` | ~50 s |
| `uv sync` (torch 2.9.1 cu128 reinstall) | ~55 s |
| `prepare.py` skip + `train.py` startup + torch.compile | ~70 s |
| **Training** (fixed budget) | **5 min** |
| `evaluate_bpb` | ~40 s |
| **Total job wall** | **~8m21s** |

Net throughput: ~7.2 experiments/hour, vs ~12/hour on a dedicated local GPU
where the startup phases don't repeat. Over an 8-hour overnight run, expect
~50 completed experiments. With Mode B fan-out (K parallel jobs/round), 4×
that.

## Repro

```bash
export AUTORESEARCH_CACHE_VOLUME=objvol-...   # your seeded cache volume
git checkout -b autoresearch/bench-baseline
bash batch-job/submit.sh > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:\|^total_seconds:\|^training_seconds:" run.log
```

`vesslctl job show <slug>` reports the exact start/end timestamps used to
fill in "Job wall time" and "Cost" above.
