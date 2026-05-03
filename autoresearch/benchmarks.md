# autoresearch benchmarks

Measured numbers from real VESSL Cloud runs. Update this file after each
end-to-end smoke test on a new spec or image.

## Headline (baseline `train.py`, no agent edits)

| Date | GPU | Cluster | Image | Train time | Job wall time | Peak VRAM | val_bpb | Cost |
|------|-----|---------|-------|-----------:|--------------:|----------:|--------:|-----:|
| TODO | A100 SXM ×1 | betelgeuse-na | pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel | TODO | TODO | TODO | TODO | TODO |

Numbers above are from a single run of the unmodified upstream `train.py`
submitted via `bash batch-job/submit.sh`. The "Train time" is the 5-minute
training budget enforced by `prepare.py`; "Job wall time" includes image
pull, repo clone, `uv sync`, torch.compile, and the final `evaluate_bpb` —
this is what you actually pay for.

## One-time prep cost

| Date | GPU | Cluster | Wall time | Cache size after | Cost |
|------|-----|---------|----------:|-----------------:|-----:|
| TODO | A100 SXM ×1 | betelgeuse-na | TODO | TODO | TODO |

`prep.sh` downloads ~10 ClimbMix shards and trains the BPE tokenizer into
`AUTORESEARCH_CACHE_VOLUME`. You pay this once per cache volume.

## Per-experiment overhead breakdown

Helpful for predicting how many experiments fit in an N-hour overnight run.

| Phase | Approx. time (A100 SXM ×1) |
|-------|---------------------------:|
| `vesslctl job create` → container start | TODO |
| `apt-get install git curl` | TODO |
| `git clone --depth 1` cookbook | TODO |
| `curl ... uv install` | TODO |
| `uv sync` (torch 2.9.1 cu128 reinstall) | TODO |
| `train.py` startup + torch.compile | TODO |
| **Training** (fixed budget) | **5 min** |
| `evaluate_bpb` | TODO |

## Repro

```bash
export AUTORESEARCH_CACHE_VOLUME=objvol-...   # your seeded cache volume
git checkout -b autoresearch/bench-baseline
bash batch-job/submit.sh > run.log 2>&1
grep "^val_bpb:\|^peak_vram_mb:\|^total_seconds:\|^training_seconds:" run.log
```

`vesslctl job show <slug>` reports the exact start/end timestamps used to
fill in "Job wall time" and "Cost" above.
