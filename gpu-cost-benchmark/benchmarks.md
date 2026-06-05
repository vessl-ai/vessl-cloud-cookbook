# Benchmarks — gpu-cost-benchmark

Measured on VESSL Cloud, 8-GPU nodes (A100 SXM 80 GB × 8, H100 SXM 80 GB × 8, B200 × 8), 2026-05.
Model: `google/gemma-4-31B-it` (training) · `RedHatAI/gemma-4-31B-it-FP8-block` (inference, H100/B200).

## Summary

LoRA fine-tuning and vLLM inference of a 31B model on three NVIDIA generations, normalized to **cost per token**. The headline: once each platform is tuned, cost-per-token lands within ~5% — so price-per-token does not decide the GPU. Speed and serving capacity do, and there the newer hardware leads, especially across multiple GPUs.

**All cost numbers are relative indices (A100 best config = 1.00 / 100). No absolute \$/GPU-hour figures are published here** — convert GPU-hours to your spend using VESSL Cloud pricing.

## Training (LoRA SFT, 220 steps, seq 2048, 8 GPUs)

| Hardware | Best config | Throughput (tok/s) | Peak VRAM/GPU | GPU-hours | Relative cost/token |
|---|---|--:|--:|--:|--:|
| A100 | FSDP, no grad-checkpointing | 4,746 | 56.6 GB | 3.62 | **1.00** |
| A100 | DDP + grad-checkpointing | 3,565 | 60.1 GB | 4.51 | 1.33 |
| H100 | FSDP + GC, micro-batch 4 | 9,559 | 52.5 GB | 3.42 | 1.04 |
| H100 | bf16 DDP + GC | 9,021 | 60.1 GB | 1.80 | 1.10 |
| B200 | bf16 + `torch.compile` | 17,634 | 61.0 GB | 1.23 | 1.05 |
| B200 | te-fp8 + `torch.compile` | 17,539 | 61.0 GB | 0.97 | 1.05 |
| B200 | naive fp8 (avoid) | 6,832 | 136.8 GB | 2.36 | 4.3 |

- Hardware throughput ratio (bf16 baseline): **B200/A100 ≈ 3.98×**, H100/A100 ≈ 2.53×.
- **FSDP-noGC > DDP+GC**: sharding parameters frees enough memory to disable gradient checkpointing → A100 +33% (4,746 vs 3,565).
- **FP8 only via Transformer Engine + compile**: naive FP8 is 0.48× bf16 throughput at 2.2× VRAM on B200; H100 fp8 training is also slower than bf16. `te-fp8 + torch.compile` gives the lowest wall-clock of any config.

## Inference (vLLM, 8 instances TP=1, ShareGPT 1,000 prompts)

| Hardware | Base, no MTP | + MTP γ=1 | Peak (best config) | Relative cost/token (peak) |
|---|--:|--:|--:|--:|
| A100 | 1,774 | 2,591 | 2,591 | 100 |
| H100 | 11,600 | 16,665 | 22,062 (γ=2 + FP8 KV) | 24.7 |
| B200 | 18,142 | 27,812 | 55,066 (γ=2 + FP8 KV, seqs 1024) | 18.3 |

- Hardware throughput ratio (base, no MTP): **B200/A100 ≈ 10.2×**, H100/A100 ≈ 6.5×. Wider than training because inference is memory-bandwidth bound — but see Known limitations: A100 runs bf16 while H100/B200 run FP8, and A100's 80 GB caps concurrency.
- **MTP (speculative decoding)**: γ=1 +53% (accept 82.8%), γ=2 +77% vs base (accept 73.4%), γ=3 +90% (accept 65.2%). Practical optimum **γ=2**.
- **LoRA + MTP**: acceptance gap between base and LoRA ≤ **0.3 pp** on all hardware → ship together.
- **FP8 KV cache**: +44–57% on top of γ=2; pushes the saturation point right. B200 LoRA + γ=2 + FP8 KV (seqs 512) = **40,734 tok/s**, exceeding the un-optimized base.

## Fine-tuning stability (200 steps, after 20-step warmup)

| Hardware | Loss spikes (>3σ) | Grad-norm max | Grad anomalies | Final loss |
|---|--:|--:|--:|--:|
| A100 | 3 | 1,650.6 | 8 | 1.19 |
| H100 | 2 | 222.2 | 6 | 1.11 |
| B200 | 3 | 233.0 | 5 | 1.14 |

All three converge to a similar final loss; A100's warmup grad-norm spike is an order of magnitude higher but normalizes after ~step 30. No run diverged.

## Cost

Relative cost-per-token index only (A100 best config = 1.00 for training; A100 peak = 100 for inference). Absolute \$/GPU-hour is intentionally omitted — multiply the measured **GPU-hours** (training table) by VESSL Cloud's published per-GPU-hour price for your tier to get your spend.

Reproduce the relative index from the bundled CSVs: `python analysis/compute_cost.py` (reads `results/training.csv` + `results/inference.csv`, writes `results/cost_summary.csv`).

## Known limitations

See [README.md](./README.md#known-limitations) for the full list. In short: A100-vs-newer inference mixes precision (bf16 vs FP8) and memory capacity with raw speed; the acceptance-rate result is LoRA-specific; cost is relative only; and a full three-hardware reproduction is a large multi-GPU spend (the bundled `results/` exist so you don't have to re-run).
