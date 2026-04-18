# VESSL Cloud Cookbook

Runnable recipes for fine-tuning, training, and deploying models on VESSL Cloud.

Each top-level folder is a self-contained recipe. Clone the repo (or just the folder you need), follow the recipe's README, and you'll have a working end-to-end run on VESSL Cloud.

## Recipe catalog

| Recipe | Task | GPU | Approx. cost | Approx. time |
|--------|------|-----|-------------:|-------------:|
| [gemma4-finetuning](./gemma4-finetuning) | LoRA fine-tune Gemma 4 E4B on a small domain QA dataset | A100 SXM 80 GB × 1 | ~$0.43 | ~16 min |

Prices as of 2026-04-16; see each recipe's `benchmarks.md` for details.

## Quickstart

1. [Sign up for VESSL Cloud](https://cloud.vessl.ai/~/signup) if you don't have an account.
2. Pick a recipe folder and open its `README.md`.
3. Follow either Path A (notebook in a workspace) or Path B (vesslctl batch job).

## How recipes are organized

- `notebook/<recipe>.ipynb` — interactive workspace walk-through. Good for a first run.
- `batch-job/<recipe>.py` + `submit.sh` — reproducible vesslctl invocation. Good for automation.
- `data/` + `DATASET_CARD.md` — bundled dataset with provenance and license.
- `benchmarks.md` — measured time, VRAM, cost, loss on VESSL Cloud.

## Prerequisites

- A VESSL Cloud account with credits.
- [vesslctl](https://docs.vessl.ai/) installed and authenticated (required for Path B).
- A Hugging Face access token for gated models (Gemma 4 is gated).

## Contributing

Issues and new-recipe proposals are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

- **Code:** Apache-2.0 (see [LICENSE](./LICENSE)).
- **Datasets:** CC-BY-4.0 (declared per recipe in `data/DATASET_CARD.md`).

## Links

- [VESSL Cloud](https://cloud.vessl.ai)
- [VESSL Cloud docs](https://docs.vessl.ai)
- [VESSL blog](https://blog.vessl.ai)
