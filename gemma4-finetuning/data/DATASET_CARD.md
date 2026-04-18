# VESSL Cloud QA Dataset

**File:** `vessl-cloud-qa-dataset.json`
**License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
**Size:** 36 question-answer pairs
**Language:** English + Korean (mixed)
**Last updated:** 2026-04-16

## Source

Hand-authored question-answer pairs derived from VESSL Cloud's user-facing documentation. Intended to teach a language model the structure of VESSL-specific concepts (Cluster storage, Object storage, workspace lifecycle, vesslctl commands).

## Format

JSON array of conversation records. Each record follows the ShareGPT-style `conversations` schema:

```json
{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}
```

The recipe code maps `from/value` → `role/content` before applying the Gemma 4 chat template. See `../notebook/gemma4-finetuning.ipynb` for the conversion.

## Intended use

- Demonstrating domain-specific LoRA fine-tuning on a small dataset.
- Illustrating how 36 samples with aggressive LoRA configuration shift model behaviour on VESSL-specific topics (storage types, workspace lifecycle, vesslctl workflows).

## ⚠️ Known limitations

This dataset is too small to teach the model specific factual claims reliably. In the runs documented in `../benchmarks.md`, the fine-tuned model **fabricated specific GPU prices** (e.g., "L40S 8GB $0.40/hr", "16GB $0.50/hr") that have no basis in the dataset or in reality. The model learns the *shape* of a VESSL-style answer — not the facts.

**Do not deploy a model fine-tuned on this dataset alone in production.** For production domain adaptation, expand the dataset to at least 5,000 curated samples and validate outputs with a held-out test split and human review.

## Attribution

When you redistribute or derive from this dataset, attribute VESSL AI Inc. (`https://vessl.ai`) per CC-BY-4.0 terms.
