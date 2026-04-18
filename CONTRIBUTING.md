# Contributing to VESSL Cloud Cookbook

We welcome new recipes.

## Propose a recipe

1. Open a GitHub issue describing the recipe you'd like to add: the task, the model, the target GPU tier, and the expected cost/time on VESSL Cloud.
2. Wait for maintainer feedback before starting the work — a quick alignment prevents wasted effort.

## Recipe structure

Each recipe lives in its own top-level folder and is self-contained:

- `README.md` — what the recipe does, how to run it, expected results, known limitations.
- `requirements.txt` — Python dependencies.
- `notebook/<recipe>.ipynb` — interactive workspace walk-through.
- `batch-job/<recipe>.py` + `submit.sh` — reproducible vesslctl batch-job path.
- `data/<dataset>` + `DATASET_CARD.md` — bundled dataset with provenance and license.
- `benchmarks.md` — measured wall time, VRAM, loss, and cost on VESSL Cloud.

See `gemma4-finetuning/` as the reference implementation.

## What "self-contained" means

A reader should be able to clone the repo, open a workspace on VESSL Cloud, and run the recipe without reading any other document. In practice:

- **Measured numbers, not estimates.** Record wall time, VRAM, and cost in `benchmarks.md` from a real run, with the date and the GPU you ran on.
- **Pin the tools you relied on.** Container image, `vesslctl --version`, and — once you have a known-good `pip freeze` — exact package versions.
- **State the limitations.** If the recipe is educational and the resulting model shouldn't be shipped (fabrication, small dataset, etc.), say so plainly in the recipe README, `benchmarks.md`, and `DATASET_CARD.md`. Honesty is cheaper than surprised users.
- **No PII in datasets.** Scan bundled data for personal information before committing.
- **Clear notebook outputs before committing** (`jupyter nbconvert --clear-output --inplace <notebook>`). Cached outputs leak environment paths and hostnames.

## License

Code: Apache-2.0. Datasets: CC-BY-4.0 (declared per recipe in `data/DATASET_CARD.md`). By contributing you agree your contribution is licensed accordingly.
