<!--
Heading anchors below are referenced by external docs (VESSL Cloud
Mintlify tutorial): #lora-hyperparameters, #custom-data, #evaluation,
#load-adapter, #known-limitations. Rename with care.
-->

# Gemma 4 LoRA fine-tuning on VESSL Cloud

LoRA fine-tune Gemma 4 E4B on a small (36-sample) domain QA dataset using Unsloth's QLoRA workflow, producing an adapter that answers VESSL-specific questions while keeping the base model's general knowledge intact.

| GPU | Cost | Wall time | Peak VRAM | Final loss |
|-----|-----:|----------:|----------:|-----------:|
| A100 SXM 80 GB × 1 | ~$0.43 | ~16 min | 11.08 GB | 0.6114 |

Numbers measured 2026-04-16 on VESSL Cloud — full breakdown in [benchmarks.md](./benchmarks.md).

## Two ways to run

- **Path A — Interactive notebook** (`notebook/gemma4-finetuning.ipynb`): step through the fine-tune in a VESSL Cloud workspace. Best for a first run and for iterating on hyperparameters.
- **Path B — vesslctl batch job** (`batch-job/finetune_gemma4.py` + `submit.sh`): fire-and-forget training submitted via `vesslctl job create`. Best for automation and reproducibility.

Both paths use the same model, dataset, and LoRA configuration.

## Prerequisites

- A **VESSL Cloud** account — [sign up here](https://cloud.vessl.ai/~/signup) if you don't have one.
- A **Hugging Face** access token. Gemma 4 itself is gated, but this recipe uses the `unsloth/gemma-4-E4B-it` mirror which does not require a token. Keep a token handy if you later switch to the official Google mirror.
- **vesslctl** installed and authenticated (Path B only) — see the [vesslctl docs](https://docs.vessl.ai/).
- **Python ≥ 3.10** (matches the container image used below).

## Path A: Interactive notebook

1. Create a workspace on VESSL Cloud with:
   - Resource spec: **A100 SXM 80 GB × 1**.
   - Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.
   - Cluster storage mounted at `/root` (fast local cache).
   - Object storage mounted at `/shared` (persistent outputs — the adapter lives here after training).
2. In the workspace's JupyterLab terminal, clone this repo and open the notebook:
   ```bash
   cd /root
   git clone https://github.com/vessl-ai/vessl-cloud-cookbook.git
   cd vessl-cloud-cookbook/gemma4-finetuning
   pip install -r requirements.txt
   ```
3. Open `notebook/gemma4-finetuning.ipynb` in JupyterLab and **Run All Cells**.
4. When done, the adapter is saved to `/shared/gemma4-vessl-expert/final/`. Teammates who mount the same Object volume can load it with `PeftModel.from_pretrained` (see [Load adapter](#load-adapter)).

The notebook runs end-to-end on a single A100 SXM in about 15 minutes plus image pull time. Stop the workspace when you're done — Object storage persists across stops.

## LoRA hyperparameters

These flags control the adapter's capacity and which parts of the model it modifies. The values shown are what this recipe uses for the VESSL-domain run (`DATASET_MODE=vessl`); the generic run (`DATASET_MODE=generic`) uses `r=8, lora_alpha=8`.

| Parameter | Value (vessl) | Value (generic) | What it does |
|-----------|--------------:|----------------:|--------------|
| `r` | 32 | 8 | LoRA rank — capacity of the adapter. Higher = can memorise more, but more parameters to train and store. Small domain datasets benefit from larger `r`. |
| `lora_alpha` | 32 | 8 | LoRA scaling factor. A common practice is `alpha == r`, which we follow here. |
| `lora_dropout` | 0 | 0 | Dropout on LoRA layers. `0` is the Unsloth default; raise to `0.05–0.1` if you see overfitting. |
| `bias` | `"none"` | `"none"` | Whether to train bias terms. `"none"` is fastest and usually sufficient. |
| `finetune_language_layers` | `True` | `True` | Train the LLM transformer blocks (what you almost always want). |
| `finetune_attention_modules` | `True` | `True` | Apply LoRA to Q/K/V/O projections in attention layers. |
| `finetune_mlp_modules` | `True` | `True` | Apply LoRA to the MLP (feed-forward) projections. |
| `finetune_vision_layers` | `False` | `False` | Gemma 4 E4B has a vision tower — leave off for text-only fine-tuning. |
| `random_state` | 3407 | 3407 | Seed for LoRA init + shuffling. Any integer works; keep fixed for reproducibility. |
| `use_gradient_checkpointing` | `"unsloth"` (in notebook) | — | Unsloth's memory-efficient checkpointing. Saves VRAM with negligible speed cost. |

Full call is in `batch-job/finetune_gemma4.py` under `FastModel.get_peft_model(...)`.

## Training parameters

Also from `batch-job/finetune_gemma4.py` — `SFTConfig`. Values shown for the VESSL-domain run.

| Parameter | Value (vessl) | Value (generic) | Note |
|-----------|--------------:|----------------:|------|
| `per_device_train_batch_size` | 1 | 1 | Batch size per GPU. Gemma 4 E4B at 4-bit with 2048 max_seq_length fits at 1 on a single A100. |
| `gradient_accumulation_steps` | 2 | 4 | Effective batch = 2 or 4. Bigger grad accumulation smooths noisy gradients on tiny datasets. |
| `warmup_steps` | 10 | 5 | Warm-up steps before LR hits peak. |
| `num_train_epochs` | 20 | — | 20 epochs × 36 samples ≈ 720 steps — enough for the adapter to memorise the domain. |
| `max_steps` | — | 60 | Generic run caps at 60 steps (~1 epoch on 3,000 samples). |
| `learning_rate` | 5e-4 | 2e-4 | Higher LR for vessl mode compensates for the small dataset. |
| `lr_scheduler_type` | `"cosine"` | `"linear"` | Cosine smooths convergence on the longer vessl run. |
| `weight_decay` | 0.01 | 0.001 | Standard regularisation. |
| `optim` | `"adamw_8bit"` | `"adamw_8bit"` | 8-bit AdamW keeps optimiser memory low. |
| `seed` | 3407 | 3407 | Fixed for reproducibility. |

## Path B: vesslctl batch job

For automated or headless runs, use `batch-job/submit.sh`:

```bash
cd gemma4-finetuning/batch-job
chmod +x submit.sh
VESSL_OBJECT_VOLUME=<your-object-volume-name> ./submit.sh vessl my-first-run
```

`submit.sh` does three things:

1. Uploads `finetune_gemma4.py` and the dataset to your Object volume under `scripts/` and `datasets/`.
2. Calls `vesslctl job create` with the PyTorch + CUDA 12.4 image, mounting the volume at `/shared`, and passing `DATASET_MODE` + `RUN_TAG` as environment variables.
3. Prints the command to tail logs.

Tail logs with:

```bash
vesslctl job logs -f gemma4-vessl-my-first-run
```

When the job finishes, download the output adapter:

```bash
vesslctl volume download <your-object-volume-name> ./outputs \
  --remote-prefix gemma4-vessl-my-first-run/
```

You'll find `final/` (the LoRA adapter), `metrics.json` (structured run metrics), and any checkpoints under the downloaded folder.

## Expected results

Mirror of [benchmarks.md](./benchmarks.md) for skim-ability — a successful VESSL-domain run produces:

- **Training wall time**: ~9 min actual training (`548.7 s`) inside a ~16 min total script wall time.
- **Final loss**: `0.6114` (from `4.0+` at start).
- **Peak VRAM**: 11.08 GB on an 80 GB A100 — plenty of headroom.
- **Behavioural change**: the fine-tuned model confidently answers VESSL-specific questions the base model refuses (e.g., workspace pause mechanics, storage trade-offs).
- **Honest caveat**: the model fabricates specific numbers (prices, VRAM specs). See [Known limitations](#known-limitations).

## Custom data

To fine-tune on your own QA data, produce a JSON file in the same shape as `data/vessl-cloud-qa-dataset.json`:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What does your product do?"},
      {"from": "gpt", "value": "It does X and Y..."}
    ]
  },
  {
    "conversations": [
      {"from": "human", "value": "..."},
      {"from": "gpt", "value": "..."}
    ]
  }
]
```

Then either (a) replace `data/vessl-cloud-qa-dataset.json` in place and re-run the notebook, or (b) upload your dataset to an Object volume and point the batch-job script at it via `DATASET_PATH=/shared/datasets/your-dataset.json` when you submit the job.

Dataset size guidance:

- **< 100 samples** — you'll get behavioural shifts (answer style, tone) but the model will hallucinate specific facts. This recipe's dataset sits here.
- **1,000–5,000 samples** — specific facts start sticking, but add a held-out test split to catch regressions.
- **> 5,000 samples** — production territory. Also consider DPO/RLHF as a follow-up step for response preference.

## Evaluation

For proper domain-adaptation evaluation, hold out ~10–20% of your dataset as a test split. Compare the base model and the fine-tuned adapter on identical test prompts and eyeball the deltas.

```python
import json, random
from transformers import TextStreamer

random.seed(42)
with open("../data/vessl-cloud-qa-dataset.json") as f:
    data = json.load(f)
random.shuffle(data)
split = int(len(data) * 0.8)
train_data, test_data = data[:split], data[split:]

# Train on `train_data` (replace the notebook's dataset load with this split).
# After training, evaluate:
for sample in test_data:
    prompt = sample["conversations"][0]["value"]
    expected = sample["conversations"][1]["value"]
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to("cuda")
    print("PROMPT:", prompt)
    print("EXPECTED:", expected[:120], "...")
    print("ACTUAL: ", end="")
    _ = model.generate(**inputs, max_new_tokens=256,
                       streamer=TextStreamer(tokenizer, skip_prompt=True))
    print("\n---")
```

For scoring, pair this with a fact-checking pass (either a human reviewer or an LLM-as-judge) and log scores over multiple training runs to catch regressions.

## Load adapter

From another workspace, another teammate's machine, or a production serving environment — mount the same Object volume (or copy the `final/` directory down) and load the adapter on top of the base model:

```python
from peft import PeftModel
from unsloth import FastModel

# 1. Load the same base model used during training.
base_model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
    dtype=None,
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)

# 2. Attach the saved LoRA adapter.
model = PeftModel.from_pretrained(
    base_model,
    "/shared/gemma4-vessl-expert/final",  # path to the saved adapter
)

# 3. Inference.
messages = [{"role": "user", "content": [{"type": "text", "text": "How do I pause a VESSL Cloud workspace?"}]}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

The adapter file is small (tens of MB) — copy it freely. The base model stays untouched, so you can attach and detach adapters per request if you're serving multiple domain experts from one GPU.

## Known limitations

- **Fabricated facts**: 36 samples teach the *shape* of a VESSL-style answer, not specific numeric claims. The fine-tuned model confidently produces incorrect prices and specs. See `benchmarks.md` Q3 and `data/DATASET_CARD.md`.
- **Not production-ready**: do not deploy a model fine-tuned on this dataset alone. For production domain adaptation, expand to at least 5,000 curated samples and validate with a held-out test split and human review.
- **Reproducibility bounds**: results depend on the exact Unsloth / trl / transformers / datasets versions installed at training time. Pin versions in `requirements.txt` if you need bit-for-bit reproducibility across runs.

## Further reading

- VESSL blog: *How we fine-tuned Gemma 4 in 16 minutes on VESSL Cloud* — [coming soon]
- VESSL docs tutorial: Fine-tuning Gemma 4 on VESSL Cloud — [coming soon]
- [Unsloth documentation](https://docs.unsloth.ai/)
- [PEFT library](https://huggingface.co/docs/peft)
