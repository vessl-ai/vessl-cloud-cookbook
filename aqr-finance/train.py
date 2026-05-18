"""train.py — Unsloth-based LoRA continued pretraining on Qwen3.5-35B-A3B-Base.

Reads uint32 token shards written by prepare.py (1 B tokens of FineWeb filtered
to CC-MAIN <= 2017-W26) and runs LoRA continued pretraining with the hybrid
Gated DeltaNet + Gated Attention target list documented in
[shanemmattner/qwen-rft-pipeline](https://github.com/shanemmattner/qwen-rft-pipeline).

We pivoted from raw accelerate+FSDP+PEFT to Unsloth + TRL after 7 dry-runs
revealed that the bleeding-edge Qwen3.5 ecosystem (transformers 5.x + custom
DeltaNet kernels + new triton autotune signatures + PEFT FSDP compat) is not
stable enough to compose from scratch — every working reference we found
(NVIDIA DGX Spark forum, Unsloth official Qwen3.5 guide, HF unsloth org)
ran through Unsloth's pre-validated stack. See README "Stack" section for
the full reasoning. The hybrid LoRA target list — the actual cookbook
contribution — survives the pivot unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from datasets import IterableDataset as HFIterableDataset
from transformers import TrainerCallback
from unsloth import FastModel, UnslothTrainer, UnslothTrainingArguments

# ---------- config ----------

BASE_MODEL = "unsloth/Qwen3.5-35B-A3B-Base"  # Unsloth mirror — identical weights, pre-patched config.

CACHE_DIR = Path(os.environ.get("AQR_CACHE_DIR", str(Path.home() / ".cache" / "aqr-finance")))
DATA_DIR = CACHE_DIR / "data"
ADAPTER_DIR = CACHE_DIR / "adapter"
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 4096
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 8
# Steps × tokens-per-step ≈ target tokens. With 4096 seq, 1 micro × 8 accum × 8 GPU = 256 K tokens/step.
# 1 B / 256 K ≈ 3815 steps. For single-GPU dry-run, drop num_processes to 1 in accelerate config and
# the global batch becomes 32 K tokens/step; scale TOTAL_STEPS accordingly via AQR_MAX_STEPS env.
TOTAL_STEPS = int(os.environ.get("AQR_MAX_STEPS", "3815"))

LEARNING_RATE = 2e-5
# Embeddings + lm_head get a smaller LR during CPT (Unsloth pattern — keeps the pre-trained
# vocabulary geometry intact while letting attention/MLP shift toward the new domain).
EMBEDDING_LR = 2e-6
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

LORA_R = 16
LORA_ALPHA = 32
# Unsloth's MoE expert LoRA wrapper (ParamWrapper) doesn't support dropout — see
# peft/tuners/lora/layer.py:2142. With 256 experts × 2 expert proj targets, this
# is a hard constraint, not a regression we can live with. Setting to 0 also
# matches Unsloth's "fast patching" path (which only kicks in at dropout=0).
LORA_DROPOUT = 0.0
# Hybrid LoRA target list — see plan v3.3-4 and README "Why these LoRA targets".
# Qwen3.5-35B-A3B-Base has 30 Gated DeltaNet layers + 10 Gated Attention layers + MoE experts.
# Targeting only q/k/v/o (the conventional attention list) hits 10 of 40 layers and produces
# trainable ≈ 0.02% of params → NaN loss in practice. Adding in_proj_qkv/in_proj_z/out_proj
# (DeltaNet) and gate/up/down_proj (MoE) brings trainable to ≈ 0.055% and stabilizes training.
# For continued pretraining specifically, we also expose lm_head and embed_tokens at
# embedding_learning_rate so the model can shift its output distribution toward the filtered
# slice without overwhelming the frozen base.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "lm_head", "embed_tokens",
]

NAN_WATCHDOG_MINUTES = 30
NAN_EXIT_CODE = 42

LOG_INTERVAL = 20
EVAL_INTERVAL = 100
SAVE_INTERVAL = 500


# ---------- dataset ----------

def token_stream(shard_paths: list[str], seq_len: int) -> Iterator[dict]:
    """Yield {"input_ids": int64 tensor[seq_len]} from uint32 shard files.

    Pre-tokenized — prepare.py already ran tokenizer.encode over the FineWeb
    slice. We just reshape into (n_seq, seq_len) and emit each row. SFTTrainer
    accepts this format under skip_prepare_dataset=True.
    """
    for path in shard_paths:
        data = np.fromfile(path, dtype=np.uint32)
        n = (len(data) // seq_len) * seq_len
        if n == 0:
            continue
        data = data[:n].reshape(-1, seq_len)
        for row in data:
            yield {"input_ids": torch.tensor(row.astype(np.int64))}


# ---------- NaN watchdog ----------

class NaNWatchdog(TrainerCallback):
    """Hard-abort if loss goes NaN during the first NAN_WATCHDOG_MINUTES.

    The first ~30 min is the window where LoRA-target misconfiguration manifests —
    if the target list misses the actual decoder layer names, trainable params collapse
    and loss diverges. After that window we trust the run.
    """

    def __init__(self, minutes: int, exit_code: int) -> None:
        self.start = time.time()
        self.deadline = self.start + minutes * 60
        self.exit_code = exit_code

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if logs and "loss" in logs and time.time() < self.deadline:
            loss = logs["loss"]
            if loss != loss:  # NaN check
                print(
                    f"train.py: NaN loss detected within {NAN_WATCHDOG_MINUTES}-min "
                    f"watchdog window (step={state.global_step}) — aborting.",
                    flush=True,
                )
                sys.stdout.flush()
                os._exit(self.exit_code)


# ---------- main ----------

def main() -> None:
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"train.py: manifest not found at {manifest_path} — run prepare.py first "
            f"(or mount the cache volume populated by batch-job/prep.sh)."
        )
    manifest = json.loads(manifest_path.read_text())
    train_paths = [str(DATA_DIR / s) for s in manifest["train_shards"]]
    val_paths = [str(DATA_DIR / s) for s in manifest["val_shards"]]
    print(
        f"train.py: manifest hit — {len(train_paths)} train shards, "
        f"{len(val_paths)} val shards, {manifest['train_tokens']/1e6:.0f} M train tokens",
        flush=True,
    )

    train_ds = HFIterableDataset.from_generator(
        token_stream,
        gen_kwargs={"shard_paths": train_paths, "seq_len": MAX_SEQ_LEN},
    )
    val_ds = HFIterableDataset.from_generator(
        token_stream,
        gen_kwargs={"shard_paths": val_paths, "seq_len": MAX_SEQ_LEN},
    )

    print(f"train.py: loading base {BASE_MODEL} (bf16, no quant)", flush=True)
    t_load = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        dtype=torch.bfloat16,
    )
    print(f"train.py: base loaded in {time.time() - t_load:.0f}s", flush=True)

    # PEFT wrap. Unsloth's get_peft_model forwards target_modules straight through to PEFT,
    # so the hybrid list lands on the actual hybrid decoder modules.
    model = FastModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's offload-aware variant
        random_state=42,
    )

    # Sanity-check that PEFT actually found the target modules — silent fall-through
    # (trainable ≈ 0) is the failure mode we burned the previous dry-runs on.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable_pct = 100.0 * n_trainable / n_total
    print(
        f"train.py: trainable = {n_trainable / 1e6:.1f} M / "
        f"{n_total / 1e9:.2f} B = {n_trainable_pct:.4f}%",
        flush=True,
    )
    if n_trainable_pct < 0.01:
        raise RuntimeError(
            f"LoRA trainable params {n_trainable_pct:.4f}% < 0.01% — target list "
            f"likely doesn't match this revision's module names. Inspect "
            f"`model.named_modules()` and update LORA_TARGET_MODULES."
        )

    args = UnslothTrainingArguments(
        output_dir=str(ADAPTER_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        max_steps=TOTAL_STEPS,
        learning_rate=LEARNING_RATE,
        embedding_learning_rate=EMBEDDING_LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=LOG_INTERVAL,
        eval_steps=EVAL_INTERVAL,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=SAVE_INTERVAL,
        save_total_limit=2,
        optim="adamw_torch",  # adamw_8bit breaks on CUDA 13.2 (DGX Spark forum confirmed).
        report_to="none",
        seed=42,
        dataloader_num_workers=0,  # forum: avoids multiprocessing fork deadlock with IterableDataset
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field=None,  # pre-tokenized — no text column
        packing=False,
        callbacks=[NaNWatchdog(NAN_WATCHDOG_MINUTES, NAN_EXIT_CODE)],
    )

    print("train.py: starting train loop", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(str(ADAPTER_DIR))

    last_log = trainer.state.log_history[-1] if trainer.state.log_history else {}
    summary = {
        "base_model": BASE_MODEL,
        "training_seconds": round(elapsed, 1),
        "train_loss_final": last_log.get("train_loss") or last_log.get("loss"),
        "val_loss_final": last_log.get("eval_loss"),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            if torch.cuda.is_available() else None,
        "num_trainable_M": round(n_trainable / 1e6, 2),
        "num_trainable_pct": round(n_trainable_pct, 4),
        "num_train_tokens_M": round(manifest["train_tokens"] / 1e6, 1),
        "max_steps": TOTAL_STEPS,
    }
    print("--- summary ---", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
