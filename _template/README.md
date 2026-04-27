<!--
If you copied this from `_template/`, fill in the TODOs and delete this
comment block. The headings `## LoRA hyperparameters`, `## Custom data`,
`## Evaluation`, `## Load adapter`, `## Known limitations` are stable
anchors used by external docs and other recipes' cross-references —
keep them where applicable, or note divergence in this README.

The reference implementation is `gemma4-finetuning/`; mirror its
section structure unless your recipe genuinely differs.
-->

# <Recipe title>

<TODO: one-sentence description of what this recipe does and why someone would run it on VESSL Cloud.>

| GPU | Cost | Wall time | Peak VRAM | Final loss |
|-----|-----:|----------:|----------:|-----------:|
| <GPU model> | ~$<X> | ~<X> min | <X> GB | <X> |

Numbers measured <YYYY-MM-DD> on VESSL Cloud — full breakdown in [benchmarks.md](./benchmarks.md).

## Two ways to run

- **Path A — Interactive notebook** (`notebook/<recipe>.ipynb`): step through the workflow in a VESSL Cloud workspace.
- **Path B — vesslctl batch job** (`batch-job/<recipe>.py` + `submit.sh`): fire-and-forget submission.

## Prerequisites

- A VESSL Cloud account.
- vesslctl installed and authenticated (Path B only).
- <TODO: recipe-specific prereqs — HF token, dataset access, model gating, etc.>

## Path A: Interactive notebook

1. Create a workspace on VESSL Cloud:
   - Resource spec: <TODO: GPU>.
   - Image: <TODO: container image>.
   - Cluster storage at `/root`, Object storage at `/shared`.
2. In the JupyterLab terminal:
   ```bash
   cd /root
   git clone https://github.com/vessl-ai/vessl-cloud-cookbook.git
   cd vessl-cloud-cookbook/<recipe-folder>
   pip install -r requirements.txt
   ```
3. Open `notebook/<recipe>.ipynb` and **Run All Cells**.

## <Add domain-specific sections here>

<TODO: e.g., for fine-tuning recipes, document hyperparameters under
`## LoRA hyperparameters` and `## Training parameters`. For evaluation
recipes, use `## Evaluation`. Match the structure of `gemma4-finetuning`
where applicable.>

## Path B: vesslctl batch job

```bash
cd <recipe-folder>/batch-job
chmod +x submit.sh
VESSL_OBJECT_VOLUME=<your-volume-name> ./submit.sh <args>
```

`submit.sh` uploads assets to your Object volume and submits a `vesslctl job create`. Tail logs with `vesslctl job logs -f <job-name>`. Download results with `vesslctl volume download`.

## Expected results

<TODO: mirror the headline numbers from `benchmarks.md` for skim-ability.>

## Custom data

<TODO: dataset format spec + how to swap in user's own data. Optional but recommended for recipes where users will bring their own data.>

## Evaluation

<TODO: held-out test split snippet or other evaluation pattern. Optional.>

## Load adapter

<TODO: how to load the produced artefact (LoRA adapter, checkpoint, etc.) from another workspace or production environment. Optional but useful when the artefact is portable.>

## Known limitations

<TODO: honest assessment of where this recipe falls short. Be specific. Examples: small dataset → fabrication, single-GPU only, untested across container image versions, etc.>

## Further reading

- [VESSL Cloud docs](https://docs.vessl.ai)
- <TODO: links to model card, paper, or related blog post>
