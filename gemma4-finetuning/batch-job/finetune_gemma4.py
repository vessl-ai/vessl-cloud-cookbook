"""
Gemma 4 LoRA fine-tuning — vesslctl batch-job script.

Trains a LoRA adapter on top of Gemma 4 E4B using the Unsloth QLoRA workflow,
saves the adapter to /shared (Object storage), and writes a metrics JSON
summarising the run.

Usage: submit via `vesslctl job create` with an Object volume mounted at
/shared. See `submit.sh` in the same directory for a reference invocation.

Environment variables:
  DATASET_MODE    'generic' (FineTome-100k subset) | 'vessl' (VESSL QA dataset)
                  Required. Controls which training data is used.
  OUTPUT_BASE     Base directory for outputs. Default: /shared
  RUN_TAG         Short identifier appended to output paths.
                  Default: unix epoch timestamp.
  DATASET_PATH    Path to the VESSL QA JSON (vessl mode only).
                  Default: /shared/datasets/vessl-cloud-qa-dataset.json

Outputs (written to $OUTPUT_BASE/gemma4-$MODE-$RUN_TAG/):
  final/          Saved LoRA adapter + tokenizer.
  metrics.json    Structured metrics + before/after inference samples.
  run.log         Captures stdout/stderr when submitted via `tee` in job cmd.

Tested: A100 SXM 80 GB on pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel,
        2026-04-16.

Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
MODE = os.environ.get("DATASET_MODE", "generic").lower()
if MODE not in ("generic", "vessl"):
    print(f"[FATAL] DATASET_MODE must be 'generic' or 'vessl', got {MODE!r}", flush=True)
    sys.exit(1)

OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/shared")
RUN_TAG = os.environ.get("RUN_TAG", str(int(time.time())))
VESSL_DATASET_PATH = os.environ.get(
    "DATASET_PATH", "/shared/datasets/vessl-cloud-qa-dataset.json"
)

OUTPUT_DIR = Path(OUTPUT_BASE) / f"gemma4-{MODE}-{RUN_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR = OUTPUT_DIR / "final"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# Shared test prompts — same across both modes → produces a
# side-by-side "generic vs vessl-trained" comparison table.
TEST_PROMPTS = [
    # Domain-specific VESSL prompts (expected: only vessl-trained model answers correctly)
    "What are the GPU prices on VESSL Cloud?",
    "What's the difference between Cluster storage and Object storage?",
    "How do I pause a VESSL Cloud workspace to save cost?",
    # Generic prompts (expected: both modes answer reasonably)
    "Explain LoRA fine-tuning in two sentences.",
    "What is the capital of France?",
]


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------
metrics = {
    "mode": MODE,
    "run_tag": RUN_TAG,
    "started_at": datetime.utcnow().isoformat() + "Z",
    "output_dir": str(OUTPUT_DIR),
    "stages": {},
    "env": {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hostname": os.environ.get("HOSTNAME", ""),
    },
}

_stage_start = {}


def log(msg):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def banner(title):
    bar = "=" * 68
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)


def stage_start(name):
    _stage_start[name] = time.time()
    banner(f"STAGE: {name}")


def stage_end(name):
    dur = time.time() - _stage_start[name]
    metrics["stages"][name] = round(dur, 2)
    log(f"STAGE DONE: {name} — {dur:.1f}s")


def gpu_snapshot(label=""):
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
        ).decode().strip()
        log(f"GPU[{label}]: {out}")
        return out
    except Exception as exc:
        log(f"GPU[{label}]: nvidia-smi failed — {exc}")
        return None


def save_metrics():
    with open(METRICS_PATH, "w") as fp:
        json.dump(metrics, fp, indent=2, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Stage 1: environment report
# ---------------------------------------------------------------------------
stage_start("environment")
log(f"MODE={MODE}  RUN_TAG={RUN_TAG}  OUTPUT_DIR={OUTPUT_DIR}")
log(f"Python {sys.version.split()[0]}")
gpu_snapshot("startup")
try:
    import torch

    metrics["torch_version"] = torch.__version__
    metrics["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        metrics["gpu_name"] = torch.cuda.get_device_name(0)
    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  gpu={metrics.get('gpu_name')}")
except Exception as exc:
    log(f"torch import failed: {exc}")
    raise
stage_end("environment")
save_metrics()


# ---------------------------------------------------------------------------
# Stage 2: model load (unsloth mirror — no HF token needed)
# ---------------------------------------------------------------------------
stage_start("model_load")
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it",
    dtype=None,
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)
metrics["model"] = "unsloth/gemma-4-E4B-it"
gpu_snapshot("after_model_load")
stage_end("model_load")
save_metrics()


# ---------------------------------------------------------------------------
# Stage 3: attach LoRA adapters
# ---------------------------------------------------------------------------
stage_start("lora_attach")
# Mode-dependent LoRA capacity:
# - generic: small dataset fine-tune is about style, r=8 is standard unsloth default
# - vessl: need enough parameters to memorize 36 domain facts, bump to r=32
if MODE == "vessl":
    lora_r, lora_alpha = 32, 32
else:
    lora_r, lora_alpha = 8, 8

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=0,
    bias="none",
    random_state=3407,
)
metrics["lora"] = {"r": lora_r, "alpha": lora_alpha, "dropout": 0}
stage_end("lora_attach")
save_metrics()


# ---------------------------------------------------------------------------
# Stage 4: dataset
# ---------------------------------------------------------------------------
stage_start("dataset")
from datasets import Dataset, load_dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats

tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")


def _format_conv(conversations):
    """Convert a conversation to gemma-4 chat-formatted text.

    Handles both schemas seen in our datasets:
    - ShareGPT (raw VESSL JSON): [{"from": "human|gpt", "value": str}]
    - Standardized/OpenAI (FineTome after standardize_data_formats):
      [{"role": "user|assistant|model", "content": str | list[dict]}]

    Returns None if the conversation is malformed for Gemma (system messages
    that can't be represented, non-alternating user/model, empty, etc.).
    Callers should filter out None entries after mapping.
    """
    messages = []
    for turn in conversations:
        if "from" in turn:
            src = turn["from"]
            if src == "human":
                role = "user"
            elif src in ("gpt", "assistant", "model"):
                role = "model"
            else:
                # Skip system/function/tool turns — Gemma can't represent them
                continue
            content_text = turn.get("value", "")
        elif "role" in turn:
            turn_role = turn["role"]
            if turn_role == "user":
                role = "user"
            elif turn_role in ("assistant", "model"):
                role = "model"
            else:
                continue  # skip system/tool/function
            content = turn.get("content", "")
            if isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                content_text = " ".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            else:
                content_text = str(content)
        else:
            return None  # malformed turn

        if not content_text.strip():
            continue

        messages.append(
            {"role": role, "content": [{"type": "text", "text": content_text}]}
        )

    # Gemma requires user/model alternation starting with user
    if len(messages) < 2 or messages[0]["role"] != "user":
        return None
    for i in range(1, len(messages)):
        if messages[i]["role"] == messages[i - 1]["role"]:
            return None

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None
    if text.startswith("<bos>"):
        text = text[5:]
    return text


if MODE == "generic":
    raw = load_dataset("mlabonne/FineTome-100k", split="train[:3000]")
    raw = standardize_data_formats(raw)
    raw_count = len(raw)

    def _format_batch(examples):
        return {"text": [_format_conv(c) for c in examples["conversations"]]}

    mapped = raw.map(_format_batch, batched=True)
    # Drop malformed/non-alternating conversations (None) and empty strings
    dataset = mapped.filter(lambda ex: ex["text"] is not None and len(ex["text"]) > 0)
    filtered_count = raw_count - len(dataset)
    log(f"Generic dataset: kept {len(dataset)}/{raw_count} samples ({filtered_count} filtered as malformed)")
    metrics["dataset"] = {
        "source": "mlabonne/FineTome-100k",
        "split": "train[:3000]",
        "raw_size": raw_count,
        "size": len(dataset),
        "filtered_out": filtered_count,
    }
else:  # vessl
    with open(VESSL_DATASET_PATH, "r") as fp:
        qa_data = json.load(fp)
    raw_count = len(qa_data)
    texts = [_format_conv(item["conversations"]) for item in qa_data]
    texts = [t for t in texts if t is not None and len(t) > 0]
    filtered_count = raw_count - len(texts)
    log(f"VESSL dataset: kept {len(texts)}/{raw_count} samples ({filtered_count} filtered as malformed)")
    dataset = Dataset.from_dict({"text": texts})
    metrics["dataset"] = {
        "source": VESSL_DATASET_PATH,
        "raw_size": raw_count,
        "size": len(dataset),
        "filtered_out": filtered_count,
    }

log(f"Dataset ready: {len(dataset)} samples")
stage_end("dataset")
save_metrics()


# ---------------------------------------------------------------------------
# Stage 5: baseline inference (BEFORE training — frozen for blog comparison)
# ---------------------------------------------------------------------------
def run_inference(phase_label):
    from transformers import TextStreamer  # noqa: F401

    responses = []
    for i, prompt in enumerate(TEST_PROMPTS):
        log(f"INFER[{phase_label}] Q{i+1}: {prompt}")
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                use_cache=True,
                do_sample=False,
                temperature=1.0,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        log(f"INFER[{phase_label}] A{i+1}: {response[:200]}{'...' if len(response) > 200 else ''}")
        responses.append({"prompt": prompt, "response": response})
    return responses


stage_start("inference_baseline")
metrics["inference_baseline"] = run_inference("baseline")
stage_end("inference_baseline")
save_metrics()


# ---------------------------------------------------------------------------
# Stage 6: training
# ---------------------------------------------------------------------------
stage_start("training")
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

if MODE == "generic":
    cfg = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        output_dir=str(OUTPUT_DIR / "checkpoints"),
    )
    metrics["training_config"] = {
        "max_steps": 60,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "grad_accum": 4,
        "scheduler": "linear",
    }
else:
    # VESSL mode — aggressive config for factual-knowledge injection on a small
    # dataset (36 samples). Each fact needs many repetitions for LoRA to memorize.
    cfg = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        num_train_epochs=20,
        learning_rate=5e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
        output_dir=str(OUTPUT_DIR / "checkpoints"),
    )
    metrics["training_config"] = {
        "num_train_epochs": 20,
        "learning_rate": 5e-4,
        "batch_size": 1,
        "grad_accum": 2,
        "scheduler": "cosine",
        "warmup_steps": 10,
    }

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=cfg,
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

gpu_snapshot("before_training")
train_out = trainer.train()
gpu_snapshot("after_training")

metrics["training_result"] = {
    "train_runtime_sec": round(train_out.metrics.get("train_runtime", 0), 2),
    "train_samples_per_second": round(train_out.metrics.get("train_samples_per_second", 0), 4),
    "train_steps_per_second": round(train_out.metrics.get("train_steps_per_second", 0), 4),
    "train_loss": round(train_out.metrics.get("train_loss", 0), 4),
    "total_flos": train_out.metrics.get("total_flos"),
}

try:
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    metrics["peak_vram_gb"] = round(peak_vram_gb, 2)
    log(f"Peak VRAM: {peak_vram_gb:.2f} GB / 80 GB")
except Exception as exc:
    log(f"peak VRAM query failed: {exc}")

stage_end("training")
save_metrics()


# ---------------------------------------------------------------------------
# Stage 7: post-training inference (same prompts → blog comparison!)
# ---------------------------------------------------------------------------
stage_start("inference_posttraining")
metrics["inference_posttraining"] = run_inference("post-train")
stage_end("inference_posttraining")
save_metrics()


# ---------------------------------------------------------------------------
# Stage 8: save model
# ---------------------------------------------------------------------------
stage_start("save")
FINAL_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(FINAL_DIR))
tokenizer.save_pretrained(str(FINAL_DIR))
log(f"Model + tokenizer saved to {FINAL_DIR}")
metrics["final_output_path"] = str(FINAL_DIR)
stage_end("save")

metrics["finished_at"] = datetime.utcnow().isoformat() + "Z"
metrics["total_wall_sec"] = round(
    sum(metrics["stages"].values()), 2
)
save_metrics()


# ---------------------------------------------------------------------------
# Final summary — screenshot-friendly table for blog
# ---------------------------------------------------------------------------
banner(f"FINAL SUMMARY — MODE={MODE}  RUN={RUN_TAG}")
print(f"Model           : {metrics['model']}", flush=True)
print(f"GPU             : {metrics.get('gpu_name', 'N/A')}", flush=True)
print(f"Dataset         : {metrics['dataset']['source']} ({metrics['dataset']['size']} samples)", flush=True)
print(f"Peak VRAM       : {metrics.get('peak_vram_gb', 'N/A')} GB / 80 GB", flush=True)
print(f"Training runtime: {metrics['training_result']['train_runtime_sec']} s", flush=True)
print(f"Final loss      : {metrics['training_result']['train_loss']}", flush=True)
print(f"Samples/sec     : {metrics['training_result']['train_samples_per_second']}", flush=True)
print(f"Output dir      : {OUTPUT_DIR}", flush=True)
print(f"Metrics JSON    : {METRICS_PATH}", flush=True)
print("\nStage timings:", flush=True)
for name, dur in metrics["stages"].items():
    print(f"  {name:<25} {dur:>8.1f}s", flush=True)
print(f"\nTotal wall time : {metrics['total_wall_sec']} s", flush=True)
banner("DONE")
