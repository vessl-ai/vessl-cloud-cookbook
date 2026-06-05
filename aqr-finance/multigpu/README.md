# Multi-GPU full-weight continued-PT (8×H100 FSDP2) — aqr-finance

Multi-GPU **full-weight** continued pretraining of `Qwen/Qwen3.5-35B-A3B-Base`
on 8×H100 with axolotl + FSDP2 — every one of the 35 B parameters is trained,
not just a LoRA adapter. This is the multi-GPU arm of the
[parent LoRA recipe](../README.md); it shares the **same base model** and the
**same `eval.py` leakage contract**, so you can A/B the full-weight checkpoint
against the single-H100 LoRA adapter on an identical measurement.

| GPU | Cost | Wall time | Peak VRAM | Trained params |
|-----|-----:|----------:|----------:|---------------:|
| H100 SXM 80 GB × 8 | ~$386 | ~18 h 36 m train | ~51 GB/GPU (AC off) | 35 B (all) |

Numbers measured on VESSL Cloud (8×H100 single node) — full breakdown in
[benchmarks.md](./benchmarks.md).

> **Why a separate arm?** A single 80 GB H100 physically can't full-weight
> train a 35 B model — optimizer state + weights + grads alone exceed ~140 GB.
> FSDP2 FULL_SHARD across 8 GPUs shards those tensors so the run fits at
> ~51 GB/GPU. The single-H100 LoRA arm trains ~2.6 % of the parameters; this
> arm trains 100 %.

## Prerequisites

- A **VESSL Cloud** account with credits.
- **vesslctl** installed and authenticated — see the [vesslctl docs](https://docs.cloud.vessl.ai/). The active org/team determine where the job is billed (`vesslctl auth status`).
- An **8×H100 SXM single-node** resource spec. Find with `vesslctl resource-spec list`, then `export AQR_RESOURCE_SPEC=resourcespec-...`.
- A **persistent object volume** (~150 GB) for the data cache + the full-weight checkpoint. Create once with `vesslctl volume create`, then `export AQR_CACHE_VOLUME=objvol-...`.
- **The `--object-volume` mount is a HARD prerequisite.** The submit script mounts it at `/root/.cache/aqr-finance`. Without it the container writes the checkpoint to ephemeral pod storage, which is **lost on pod terminate**. Do not remove the flag.
- **Hugging Face read access** to `Qwen/Qwen3.5-35B-A3B-Base` (the base is downloaded inside the job).

## How to run

The full-weight run is one `vesslctl` batch job. It inline-preps the
point-in-time FineWeb slice (skip-if-exists on the volume), then launches
axolotl FSDP2 across 8 GPUs:

```bash
export AQR_CACHE_VOLUME=objvol-...      # your persistent volume slug
export AQR_RESOURCE_SPEC=resourcespec-... # your 8×H100 SXM spec slug
bash multigpu/stage3_real_v2_submit.sh
```

The script submits the job, polls until terminal, and enforces conservative
mid-run abort gates (NaN/OOM at step 0/1, no checkpoint by 1.5 h, an 8 h
sanity printout). To confirm a checkpoint actually landed on the persistent
volume (not ephemeral storage), run the cheap CPU probe:

```bash
export AQR_CPU_SPEC=resourcespec-...    # a cheap CPU-only spec slug
bash multigpu/volume_inspect_submit.sh
```

The full FSDP2 config lives in [`qwen3-moe-fullweight-cpt.yaml`](./qwen3-moe-fullweight-cpt.yaml);
the point-in-time data prep is [`prepare_text.py`](./prepare_text.py) (mirrors
the parent recipe's chronological cutoff but emits raw-text JSONL for axolotl's
`type: completion` tokenizer instead of pre-tokenized shards).

## Key FSDP2 config invariants

These are the non-obvious settings in `qwen3-moe-fullweight-cpt.yaml` that
make a 35 B full-weight run fit and train on 8×H100. Change them at your peril:

- **`gradient_accumulation_steps: 1`.** With GA > 1, Accelerate/Trainer puts
  FSDP into `no_sync()` for the first GA−1 microbatches, which accumulates
  **unsharded full-model gradients** (~70 GB bf16 for 35 B) per GPU and blows
  the 80 GB budget at the first backward. GA = 1 keeps gradients sharded.
- **`optimizer: adamw_torch_fused`.** 8-bit Adam (bitsandbytes) fails on FSDP2
  DTensor with a "mixed torch.Tensor and DTensor" error
  ([bitsandbytes #1633](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1633)).
  `adamw_torch_fused` is PyTorch-native and FSDP2 first-class.
- **`activation_checkpointing: false`.** AC = True fit comfortably (~42 GB/GPU)
  but pays a recompute tax. Turning it off costs extra activation residency
  (~51 GB/GPU peak) and buys ~+32 % throughput on H100 — worth it given the
  headroom.
- **`transformer_layer_cls_to_wrap: Qwen3_5MoeDecoderLayer`.** The single
  decoder-layer class FSDP2 wraps (one class @ `model.layers.N`), covering the
  10 Gated-Attention + 30 Gated-DeltaNet layers + the MoE expert MLPs.

## Offline merge (separate single-GPU job)

Consolidating the FSDP shards into a merged checkpoint **inside the distributed
job** hits the NCCL watchdog timeout — the rank-0 gather of a 35 B model takes
long enough that the other ranks' collectives time out. Run the merge as a
**separate single-GPU job** that reads the sharded checkpoint off the volume
and writes the merged weights back. (The training job here uses
`state_dict_type: SHARDED_STATE_DICT` precisely so the in-job save doesn't try
to gather 35 B to one host.)

## Evaluation

Use the parent recipe's [`eval.py`](../eval.py) unchanged — the leakage
contract (chronological vs stock-disjoint GroupKFold R², clustered-bootstrap
CI) is identical for both arms. Point it at the merged full-weight checkpoint
instead of the LoRA adapter and compare the `leakage_premium` /
`premium_reduction` numbers head-to-head. The measured A/B is in
[benchmarks.md](./benchmarks.md).

## Known limitations

- **Single-node only.** This arm is FSDP2 FULL_SHARD on one 8×GPU node. No
  tensor parallelism (TP) or pipeline parallelism (PP) — for >8 GPUs or
  multi-node you'd need a different parallelism strategy.
- **Measurement, not alpha.** Same as the parent recipe: this measures the
  leakage premium, it does not produce a profitable model. Both out-of-sample
  R² are negative.
- **Training every parameter did not beat the LoRA arm.** The full-weight
  `premium_reduction` CI still crosses zero — the lever on the measured signal
  is the **eval protocol**, not the trainable-parameter count. See
  [benchmarks.md](./benchmarks.md).
- **H100 / FSDP2 specific.** The memory accounting and config invariants above
  are tuned for 8×H100 80 GB. B200 / Rubin-class hardware is out of scope —
  the sharding math and the AC-off headroom would need re-measuring.

## Further reading

- VESSL blog: the full engineering trail of this run (probes, OOM ladder, the
  no_sync / DTensor fixes, the offline-merge NCCL timeout).
- Parent [LoRA recipe](../README.md) — the single-H100 arm with the same eval.
- Kelly, Malamud, Schwab & Xu, "Scaling Point-in-Time Language Models," SSRN
  Working Paper No. 6681860: <https://ssrn.com/abstract=6681860>
- He, Lv, Manela & Wu, "Chronologically Consistent Large Language Models"
  (ChronoBERT / ChronoGPT), arXiv:2502.21206: <https://arxiv.org/abs/2502.21206>
- [axolotl documentation](https://docs.axolotl.ai/) — multi-GPU FSDP2 config.
- VESSL Cloud docs: <https://docs.cloud.vessl.ai>
