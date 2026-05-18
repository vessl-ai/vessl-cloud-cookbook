"""
train.py — LoRA continued pretraining on Qwen3.5-35B-A3B-Base.

What this script does:
  1. Load Qwen3.5-35B-A3B-Base. The base is a 40-layer hybrid:
       30 layers of Gated DeltaNet (linear attention) +
       10 layers of Gated Attention (standard softmax attention),
     MoE with ~35 B total / ~3 B active parameters per token.
  2. Attach LoRA adapters to BOTH attention paths plus MoE expert MLPs:
       q_proj, k_proj, v_proj, o_proj                # Gated Attention (10 layers)
       in_proj_qkv, in_proj_z, out_proj              # Gated DeltaNet  (30 layers)
       gate_proj, up_proj, down_proj                 # MoE expert MLP  (all layers)
     This pattern is required: with only q/k/v/o, 75 % of the model (the
     DeltaNet layers) stays frozen, trainable drops to ~0.02 %, and continued
     PT diverges to NaN loss within a few hundred steps. Adding the DeltaNet
     in_proj/out_proj plus the MoE expert MLPs brings trainable to ~0.055 %
     (~19 M params) and the loss curve stabilises. Reference:
     https://github.com/shanemmattner/qwen-rft-pipeline#deltanet-lora-target-reference
  3. Stream uint32 token shards from ~/.cache/aqr-finance/data/ that
     prepare.py wrote. Pack into MAX_SEQ_LEN sequences. AdamW + cosine LR.
  4. Log val_loss every EVAL_INTERVAL steps. The NaN watchdog aborts the
     run with exit code 42 if any loss is NaN — that is the cheap early
     signal that the LoRA target list (or the base model load) is wrong.
  5. Save the LoRA adapter to ~/.cache/aqr-finance/adapter/ at the end.

Usage (inside the VESSL container, via batch-job/submit.sh):
    uv run accelerate launch --config_file accelerate_config.yaml train.py

This script is the file the autoresearch agent edits. See program.md.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Config — agent-editable region.
# ---------------------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen3.5-35B-A3B-Base"
CACHE_DIR = Path(os.environ.get("AQR_CACHE_DIR", str(Path.home() / ".cache" / "aqr-finance")))
DATA_DIR = CACHE_DIR / "data"
ADAPTER_DIR = CACHE_DIR / "adapter"

# Sequence length + per-device batch. 8 GPUs * 1 micro * 8 accum = 64 global.
MAX_SEQ_LEN = 4096
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 8

# LR + schedule.
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01

# Training horizon. 1 B tokens / (64 * 4096) ≈ 3815 steps.
TOTAL_STEPS = 3815
# EVAL_INTERVAL is also the val-NaN watchdog cadence — keep it small enough
# that the first eval fires within the NAN_WATCHDOG_MINUTES window (~30 min).
# At ~8-12 s/step on 8xH100, 100 steps ≈ 13-20 min → first eval inside the
# window. (Per-step train-loss NaN is checked every step regardless.)
EVAL_INTERVAL = 100
LOG_INTERVAL = 20
EVAL_BATCHES = 64
EVAL_BATCHES_FINAL = 128

# LoRA — hybrid attention targets. DO NOT prune without checking the
# DeltaNet reference: pruning back to q/k/v/o reproduces the 0.02 %-trainable
# NaN failure described in the docstring above.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]

# NaN watchdog — abort if any train or val loss is NaN within the first
# 30 minutes (~$20 of compute on 8xH100 at $4/GPU/h).
NAN_WATCHDOG_MINUTES = 30
NAN_EXIT_CODE = 42

# ---------------------------------------------------------------------------
# Dataloader — stream packed sequences from prepare.py uint32 shards.
# ---------------------------------------------------------------------------


class ShardedTokenDataset(IterableDataset):
    """Yields int64 tensors of shape (seq_len,) from uint32 shard files.

    Shards are partitioned across (distributed rank, dataloader worker) so no
    sequence is yielded twice. With num_workers=0 (our default) only the
    distributed rank split applies; bumping num_workers also splits per
    worker so duplicates don't reappear.
    """

    def __init__(self, shard_paths, seq_len):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.seq_len = seq_len

    def __iter__(self):
        # Distributed rank split.
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        # Dataloader worker split (only matters when num_workers > 0).
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        total_splits = world_size * num_workers
        my_split = rank * num_workers + worker_id

        shards = [s for i, s in enumerate(self.shard_paths) if i % total_splits == my_split]
        if not shards:
            return

        # Loop forever — train.py caps by step count, not epoch.
        while True:
            for path in shards:
                data = np.fromfile(path, dtype=np.uint32)
                n = (len(data) // self.seq_len) * self.seq_len
                if n == 0:
                    continue
                data = data[:n].reshape(-1, self.seq_len)
                for row in data:
                    yield torch.from_numpy(row.astype(np.int64))


def build_dataloaders(manifest):
    train_paths = [DATA_DIR / s for s in manifest["train_shards"]]
    val_paths = [DATA_DIR / s for s in manifest["val_shards"]]

    train_ds = ShardedTokenDataset(train_paths, MAX_SEQ_LEN)
    val_ds = ShardedTokenDataset(val_paths, MAX_SEQ_LEN)

    # num_workers=0: I/O is `np.fromfile` (sub-ms) so worker overhead is not
    # worth the rank-vs-worker split bookkeeping risk. If you bump
    # num_workers later, ShardedTokenDataset already handles the split.
    train_loader = DataLoader(
        train_ds, batch_size=PER_DEVICE_BATCH, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=PER_DEVICE_BATCH, num_workers=0, pin_memory=True
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Eval — fixed-K batches, returns mean loss.
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, val_loader, device, n_batches):
    model.eval()
    total = 0.0
    n = 0
    it = iter(val_loader)
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = batch.to(device, non_blocking=True)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        total += out.loss.item()
        n += 1
    model.train()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    is_main = accelerator.is_main_process

    if is_main:
        print(
            f"train.py: world_size={accelerator.num_processes}, "
            f"device={accelerator.device}, dtype=bf16 (via accelerate config)",
            flush=True,
        )

    # Manifest.
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run `uv run prepare.py` first."
        )
    manifest = json.loads(manifest_path.read_text())
    if is_main:
        print(
            f"train.py: manifest = {manifest['train_tokens']:,} train / "
            f"{manifest['val_tokens']:,} val tokens",
            flush=True,
        )

    # Tokenizer (we never tokenize at train time — prepare.py already did —
    # but we need eos_token_id for any future padding).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Base model.
    if is_main:
        print(f"train.py: loading base model {BASE_MODEL}", flush=True)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if is_main:
        print(f"train.py: base loaded in {time.time() - t_load:.0f}s", flush=True)

    # LoRA attach.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # PEFT creates LoRA A/B in fp32 by default; base is bf16. FSDP refuses
    # to flatten a wrap unit containing mixed dtypes ("Must flatten tensors
    # with uniform dtype"). Cast trainable params to bf16 to match. LoRA
    # updates have small magnitude relative to base, so the fp32 precision
    # margin isn't load-bearing at our scale.
    for _name, _param in model.named_parameters():
        if _param.requires_grad:
            _param.data = _param.data.to(torch.bfloat16)
    # Required for gradient checkpointing on PEFT-wrapped models (otherwise
    # the input embeddings don't propagate grad to LoRA params).
    model.enable_input_require_grads()
    # Under PEFT+FSDP, gradient_checkpointing must target the unwrapped base
    # model — calling it on the PeftModel wrapper silently no-ops because the
    # FSDP-wrapped decoder layers live one level down (model.base_model.model).
    # Without this, forward activations for all 40 hybrid decoder layers
    # accumulate and OOM at first batch.
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model.base_model.model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_enable()

    # Sanity-check that PEFT actually found the target modules. PEFT will
    # silently fall through (trainable ≈ 0) if a target name doesn't match
    # any module — that's the cheap signal the hybrid LoRA target list has
    # drifted from the actual module names in this revision of the base
    # model. Crash early instead of burning a 5-9 h run.
    if is_main:
        all_module_suffixes = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
        missing = [t for t in LORA_TARGET_MODULES if t not in all_module_suffixes]
        if missing:
            raise RuntimeError(
                f"LoRA target modules not found in {BASE_MODEL}: {missing}. "
                f"This usually means the base model's internal naming has "
                f"changed; inspect `model.named_modules()` and update "
                f"LORA_TARGET_MODULES. See README for the hybrid attention "
                f"context."
            )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable_pct = 100.0 * n_trainable / n_total
    if is_main:
        print(
            f"train.py: trainable = {n_trainable / 1e6:.1f}M / "
            f"{n_total / 1e9:.2f}B = {n_trainable_pct:.4f}%",
            flush=True,
        )

    # Dataloaders.
    train_loader, val_loader = build_dataloaders(manifest)

    # Optimizer + cosine schedule.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    num_warmup_steps = int(TOTAL_STEPS * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=TOTAL_STEPS,
    )

    model, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(
        model, optimizer, train_loader, scheduler, val_loader
    )

    # Train loop.
    global_batch = PER_DEVICE_BATCH * accelerator.num_processes * GRAD_ACCUM_STEPS
    if is_main:
        print(
            f"train.py: starting training — {TOTAL_STEPS} steps, "
            f"global batch = {global_batch}, "
            f"warmup = {num_warmup_steps} steps",
            flush=True,
        )

    t_start = time.time()
    nan_deadline = t_start + NAN_WATCHDOG_MINUTES * 60
    val_loss_first30min = None
    step = 0
    loss_acc = 0.0
    loss_acc_n = 0
    train_iter = iter(train_loader)

    while step < TOTAL_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        with accelerator.accumulate(model):
            batch = batch.to(accelerator.device, non_blocking=True)
            out = model(input_ids=batch, labels=batch, use_cache=False)
            loss = out.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            step += 1
            loss_val = loss.item()
            loss_acc += loss_val
            loss_acc_n += 1

            if math.isnan(loss_val):
                if is_main:
                    print(
                        f"\ntrain.py: NaN train loss at step {step} "
                        f"(t={time.time() - t_start:.0f}s) — ABORT (exit {NAN_EXIT_CODE})",
                        flush=True,
                    )
                sys.exit(NAN_EXIT_CODE)

            if step % LOG_INTERVAL == 0 and is_main:
                avg = loss_acc / loss_acc_n
                lr_now = scheduler.get_last_lr()[0]
                dt = time.time() - t_start
                print(
                    f"\rstep {step:05d}/{TOTAL_STEPS} | "
                    f"loss: {avg:.4f} | lr: {lr_now:.2e} | "
                    f"{dt:.0f}s elapsed",
                    end="",
                    flush=True,
                )
                loss_acc = 0.0
                loss_acc_n = 0

            if step % EVAL_INTERVAL == 0 or step == TOTAL_STEPS:
                vl = evaluate(model, val_loader, accelerator.device, EVAL_BATCHES)
                if is_main:
                    print(
                        f"\ntrain.py: step {step} val_loss = {vl:.4f}", flush=True
                    )
                if math.isnan(vl):
                    if is_main:
                        print(
                            f"train.py: NaN val_loss at step {step} — "
                            f"ABORT (exit {NAN_EXIT_CODE})",
                            flush=True,
                        )
                    sys.exit(NAN_EXIT_CODE)
                if val_loss_first30min is None and time.time() >= nan_deadline:
                    val_loss_first30min = vl

    total_training_time = time.time() - t_start

    # Final eval (more batches for stability).
    val_loss_final = evaluate(
        model, val_loader, accelerator.device, EVAL_BATCHES_FINAL
    )
    if val_loss_first30min is None:
        # Run completed before the 30-min watchdog (e.g. small target tokens
        # during a dry-run sanity pass).
        val_loss_first30min = val_loss_final

    # Save adapter.
    accelerator.wait_for_everyone()
    if is_main:
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(
        str(ADAPTER_DIR),
        save_function=accelerator.save,
        is_main_process=is_main,
    )

    # Summary block (the keys here must match wait-jobs.sh grep pattern).
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    if is_main:
        print("\n--- train.py summary ---", flush=True)
        print(f"val_loss_final:      {val_loss_final:.4f}", flush=True)
        print(f"val_loss_first30min: {val_loss_first30min:.4f}", flush=True)
        print(f"training_seconds:    {total_training_time:.1f}", flush=True)
        print(f"total_seconds:       {time.time() - t_start:.1f}", flush=True)
        print(f"peak_vram_mb:        {peak_vram_mb:.1f}", flush=True)
        print(f"num_trainable_M:     {n_trainable / 1e6:.1f}", flush=True)
        print(f"num_trainable_pct:   {n_trainable_pct:.4f}", flush=True)
        print(
            f"num_train_tokens_M:  {manifest['train_tokens'] / 1e6:.1f}",
            flush=True,
        )
        print("---", flush=True)


if __name__ == "__main__":
    main()
