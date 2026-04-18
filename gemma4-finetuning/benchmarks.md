# Benchmarks — gemma4-finetuning

Measured on VESSL Cloud A100 SXM (80 GB), single GPU, 2026-04-16.

## Summary

Two successful fine-tuning runs, two different training directions (generic instruction-following vs VESSL domain QA), both on identical infrastructure. Results below come from runs `v3-A` (generic) and `v2-B` (VESSL-domain) of the batch-job script in this recipe.

## Run statistics

| Metric | Generic run (3,000 samples) | VESSL-domain run (36 samples) |
|--------|-----------------------------:|------------------------------:|
| Dataset | FineTome-100k subset | VESSL Cloud QA (this recipe) |
| LoRA rank | 8 | 32 |
| Epochs / max_steps | 60 steps (~1 epoch) | 20 epochs |
| Learning rate | 2e-4 (linear) | 5e-4 (cosine) |
| Training runtime | 196.7 s | 548.7 s |
| Final loss | 4.0601 | 0.6114 |
| Peak VRAM | 12.44 GB | 11.08 GB |
| Total wall time (script) | 748.8 s (~12.5 min) | 990.8 s (~16.5 min) |
| GPU | A100 SXM 80 GB | A100 SXM 80 GB |
| Container image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | Same |

## Three-way inference comparison

Same prompts, three models: **base Gemma 4 E4B** / **generic-trained** / **VESSL-domain-trained**.

### "How do I pause a VESSL Cloud workspace to save cost?" (English)

- **Base model**: "I do not have specific, real-time documentation..."
- **Generic-trained**: "I do not have specific instructions on how to pause a VESSL Cloud workspace..." — nearly identical refusal.
- **VESSL-domain-trained**: "To pause a VESSL Cloud workspace, use the **Pause** function within the Cloud Console. CPU usage stops immediately. Memory usage stops immediately. The disk volume remains active..."

Domain training flips a refusal into a confident, mechanism-aware response. This is the cleanest before/after demonstration.

### "클러스터 스토리지와 오브젝트 스토리지 차이가 뭔가요?" (Korean)

- **Base / generic-trained**: generic framing ("파일 시스템을 확장한 블록 스토리지").
- **VESSL-domain-trained**: uses the VESSL narrative ("클러스터 환경에 최적화되어 같은 클러스터 내에서 빠르게 접근… 팀원들끼리 빠르게 공유하는 개인 작업 공간").

Generic training does not shift domain-specific reasoning. Only domain-specific data does.

### "VESSL Cloud에서 GPU 가격이 어떻게 되나요?" (Korean) — honest caveat

- **Base / generic-trained**: refuse ("공식 문서 참조").
- **VESSL-domain-trained**: confidently answers with **fabricated** prices ("L40S 8GB $0.40/hr", "16GB $0.50/hr") that do not match real VESSL pricing (A100 SXM is $1.55/hr as of 2026-04-16).

36 samples with aggressive LoRA config teach the *shape* of an answer but not specific numeric facts. The fine-tuned model hallucinates specific claims.

### "Explain LoRA fine-tuning in two sentences." (control)

All three models produce comparable, correct explanations. General technical knowledge is preserved through fine-tuning — LoRA only adds adapters, the base weights stay intact.

### "What is the capital of France?" (control)

All three: "The capital of France is **Paris**." No catastrophic forgetting.

## Cost

| Run | Duration (script wall time) | Cost at $1.55/hr |
|-----|----------------------------:|-------------------:|
| Generic | ~12.5 min | ~$0.32 |
| VESSL-domain | ~16.5 min | ~$0.43 |

Image pull + pip install accounts for ~5–7 min of each full job wall time (not included above). For repeated production use, prepare a pre-built image with `unsloth`/`trl` baked in to save that time.

Prices as of 2026-04-16.

## ⚠️ Honest limitation

See `data/DATASET_CARD.md`. This recipe is educational. The fine-tuned model produces factually incorrect specific claims (fabricated prices). For production domain adaptation, use 5,000+ curated samples and evaluate on a held-out test split.
