"""
LoRA SFT training cell for the GPU cost benchmark — one script, four parallelism variants.

Runs a single supervised fine-tuning measurement cell on an 8-GPU node and
writes a rich per-step `loss_log.jsonl` plus an aggregate `summary.json`
(throughput, MFU, VRAM, step-time stats, loss stability, NVML power/util).
Those JSON files are what the analysis stage turns into the cost index.

This merges three originally-separate scripts into one, selectable via
`--variant` (or the VARIANT env var):

  ddp        DistributedDataParallel — model replicated per GPU, autotunes
             micro-batch size (probes 1->2->4->8, keeps the largest that fit).
  fsdp       FSDP full_shard (HF Trainer + auto_wrap of the transformer block).
             Relieves per-GPU parameter pressure; micro-batch fixed (no autotune).
  fsdp-nogc  Same as fsdp but with gradient checkpointing explicitly OFF — used
             to measure the activation-memory ceiling (often OOMs at seq=2048).
  tp         FSDP2 + 2D DeviceMesh (dp x shard) hybrid shard via the composable
             `fully_shard` API with a custom training loop. --tp-size sets the
             shard-group size (2|4|8); all-gather is limited to that group.

Common to all variants — the real LoRA SFT config:
  * LoRA r=16, alpha=32, dropout=0, applied to the 7 target modules
    (q/k/v/o_proj + gate/up/down_proj) of the language-model decoder layers.
  * seq_len 2048, ShareGPT-style pre-packed dataset (datasets.load_from_disk).
  * bf16 base compute. An FP8 path exists (--quant fp8) via torchao float8 on
    Hopper/Blackwell, and a Transformer-Engine FP8 path (--quant te-fp8) is
    wired for Blackwell when transformer_engine is installed. A100 is bf16-only.
  * Cosine LR schedule, global batch = 16 sequences (grad-accum derived).

Args (argparse; many also read from env for batch-job convenience):
  --variant       ddp | fsdp | fsdp-nogc | tp   (env: VARIANT, default ddp)
  --hardware      A100 | H100 | B200            (env: HARDWARE, required)
  --quant         bf16 | fp8 | te-fp8           (env: QUANT, default bf16)
                  fp8/te-fp8 invalid on A100; fsdp/tp variants are bf16-only.
  --dataset       Path to a load_from_disk packed dataset
                  (env: DATASET, default /shared/datasets/sharegpt_packed_2048)
  --out-dir       Output dir (env: OUTPUT_DIR; default derived under
                  $OUTPUT_BASE/runs/{HW}_{QUANT}[-variant]/T-{HW}[-variant])
  --tp-size       2|4|8, tp variant only (env: TP_SIZE, default 2)
  --mbsz          micro-batch; "auto" only for ddp (env: MBSZ)
  --seq-len 2048  --max-steps 220  --warmup-steps 20  --seed 42
  --grad-checkpoint   enable gradient checkpointing (ddp/fsdp)
  --torch-compile     enable torch.compile (inductor)
  --no-save-checkpoint  skip saving the LoRA adapter (fsdp/tp)

Environment variables:
  HARDWARE, QUANT, VARIANT, DATASET, OUTPUT_DIR, TP_SIZE, MBSZ
                  See args above; CLI flags override env.
  OUTPUT_BASE     Base dir for derived output paths. Default: /shared
  MODEL_ID        Base model HF id. Default: google/gemma-4-31B-it
  BASE_PARAMS     Base param count for MFU calc. Default matches gemma-4-31B-it.
  WANDB_PROJECT, WANDB_ENTITY (optional)
                  If WANDB_PROJECT is set, runs log to Weights & Biases.
                  Unset (default) => no external logging. No entity/project is
                  baked in; bring your own.

Launch (8-GPU node, run under torchrun):
  torchrun --nproc_per_node=8 --master_port=29500 train_sft.py \
      --variant ddp --hardware H100 --quant fp8 --mbsz auto \
      --dataset /shared/datasets/sharegpt_packed_2048

Expected infra:
  * 8x GPU node (A100 / H100 / B200 SXM, 80 GB class).
  * Object storage volume mounted at /shared holding the packed dataset and receiving
    run outputs; workspace/clone at /root.
  * Base model weights are pulled from the Hugging Face Hub (set HF_HOME to a
    persistent cache to avoid re-downloading the full checkpoint each run).

Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

# torchao float8 is only needed for the --quant fp8 path; tolerate its absence.
try:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    HAS_TORCHAO_FP8 = True
except ImportError:
    HAS_TORCHAO_FP8 = False

# Transformer Engine FP8 (--quant te-fp8) — Blackwell-class only, optional.
try:
    import transformer_engine.pytorch as te_pytorch
    from transformer_engine.common.recipe import DelayedScaling, Format as TEFormat
    HAS_TE = True
except ImportError:
    HAS_TE = False
    te_pytorch = None


# ---------------------------------------------------------------------------
# Model / LoRA constants
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-31B-it")
# Total base-model parameter count, used for the theoretical-FLOPs / MFU calc.
# Override via BASE_PARAMS if you swap MODEL_ID.
BASE_PARAMS = int(os.environ.get("BASE_PARAMS", 31_284_138_800))
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

# Anchor LoRA to language_model decoder layers — the vision tower has the same
# proj suffixes but on a non-nn.Linear class PEFT can't wrap, so it's excluded.
LORA_TARGETS = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)

# Vendor peak dense TFLOPS (BF16 / FP8). Used only for MFU reporting.
HW_PEAK_TFLOPS = {
    ("A100", "bf16"): 312.0,
    ("H100", "bf16"): 989.0,
    ("H100", "fp8"): 1979.0,
    ("H100", "te-fp8"): 1979.0,
    ("B200", "bf16"): 2250.0,
    ("B200", "fp8"): 4500.0,
    ("B200", "te-fp8"): 4500.0,
}

GLOBAL_BATCH_SEQ = 16  # target global batch in sequences (grad-accum derived)

# Transformer block class name — used by the FSDP/FSDP2 auto-wrap policy.
GEMMA_LAYER_CLS = os.environ.get("TRANSFORMER_LAYER_CLS", "Gemma4TextDecoderLayer")


# ---------------------------------------------------------------------------
# distributed helpers
# ---------------------------------------------------------------------------
def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def fmt_gib(b: int) -> float:
    return b / (1024 ** 3)


# ---------------------------------------------------------------------------
# Transformer Engine FP8 helpers (--quant te-fp8)
# ---------------------------------------------------------------------------
class TEFp8Trainer(SFTTrainer):
    """SFTTrainer that wraps each training_step with te.fp8_autocast."""

    def __init__(self, *args, fp8_recipe=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._te_fp8_recipe = fp8_recipe

    def training_step(self, model, inputs, num_items_in_batch=None):
        with te_pytorch.fp8_autocast(enabled=True, fp8_recipe=self._te_fp8_recipe):
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            return super().training_step(model, inputs)


def apply_te_fp8(model: torch.nn.Module, rank: int) -> torch.nn.Module:
    """Replace PEFT LoRA base_layer nn.Linear -> te.Linear for FP8 compute.

    Skips LoRA adapter weights (lora_A/lora_B), lm_head, embed_tokens, and
    vision modules so only the frozen base-model linears run in FP8.
    """
    import torch.nn as nn
    SKIP = ("lm_head", "embed_tokens", "lora_A", "lora_B",
            "multi_modal_projector", "vision_tower")
    replaced = 0
    for fqn, mod in model.named_modules():
        if any(s in fqn for s in SKIP):
            continue
        if hasattr(mod, "base_layer") and isinstance(mod.base_layer, nn.Linear):
            orig = mod.base_layer
            te_lin = te_pytorch.Linear(
                orig.in_features, orig.out_features,
                bias=orig.bias is not None,
                params_dtype=torch.bfloat16,
            )
            with torch.no_grad():
                te_lin.weight.copy_(orig.weight)
                if orig.bias is not None:
                    te_lin.bias.copy_(orig.bias)
            # te.Linear defaults requires_grad=True, but base weights stay frozen.
            te_lin.weight.requires_grad_(orig.weight.requires_grad)
            if orig.bias is not None:
                te_lin.bias.requires_grad_(orig.bias.requires_grad)
            mod.base_layer = te_lin
            replaced += 1
    if rank == 0:
        print(f"[te-fp8] replaced {replaced} nn.Linear -> te.Linear (base_layer only)", flush=True)
    return model


# ---------------------------------------------------------------------------
# NVML background sampler (rank 0): power / util / mem / temp
# ---------------------------------------------------------------------------
class NVMLSampler:
    """Background thread sampling GPU power/util/mem-util/temp via NVML.

    Runs on rank 0, aggregates across all visible GPUs every `interval_s`.
    """

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self._latest: dict[str, Any] = {}
        self._handles = []
        self._enabled = False
        self._history: list[dict[str, Any]] = []
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
            self._enabled = True
        except Exception as e:
            print(f"[nvml] disabled — {type(e).__name__}: {e}", flush=True)

    def start(self):
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                samples = {"power_w": [], "sm_util_pct": [], "mem_util_pct": [],
                           "temp_c": [], "mem_used_gib": []}
                for h in self._handles:
                    samples["power_w"].append(self.pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                    u = self.pynvml.nvmlDeviceGetUtilizationRates(h)
                    samples["sm_util_pct"].append(u.gpu)
                    samples["mem_util_pct"].append(u.memory)
                    samples["temp_c"].append(
                        self.pynvml.nvmlDeviceGetTemperature(h, self.pynvml.NVML_TEMPERATURE_GPU))
                    mem = self.pynvml.nvmlDeviceGetMemoryInfo(h)
                    samples["mem_used_gib"].append(mem.used / (1024 ** 3))
                ts = time.time()
                agg = {
                    "ts": ts,
                    "power_w_mean": st.mean(samples["power_w"]),
                    "power_w_max": max(samples["power_w"]),
                    "power_w_total": sum(samples["power_w"]),
                    "sm_util_pct_mean": st.mean(samples["sm_util_pct"]),
                    "sm_util_pct_max": max(samples["sm_util_pct"]),
                    "mem_util_pct_mean": st.mean(samples["mem_util_pct"]),
                    "temp_c_max": max(samples["temp_c"]),
                    "mem_used_gib_mean": st.mean(samples["mem_used_gib"]),
                    "per_gpu_power_w": samples["power_w"],
                    "per_gpu_sm_util": samples["sm_util_pct"],
                }
                self._latest = agg
                self._history.append(agg)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def latest(self) -> dict:
        return dict(self._latest)

    def summary(self) -> dict:
        if not self._history:
            return {}
        powers = [s["power_w_mean"] for s in self._history]
        sm_utils = [s["sm_util_pct_mean"] for s in self._history]
        n_gpus = max(1, len(self._handles))
        return {
            "nvml_samples": len(self._history),
            "avg_power_w_per_gpu": st.mean(powers),
            "avg_power_w_node": st.mean(powers) * n_gpus,
            "peak_power_w_per_gpu": max(s["power_w_max"] for s in self._history),
            "avg_sm_util_pct": st.mean(sm_utils),
            "p50_sm_util_pct": sorted(sm_utils)[len(sm_utils) // 2],
            "avg_mem_util_pct": st.mean([s["mem_util_pct_mean"] for s in self._history]),
            "peak_temp_c": max(s["temp_c_max"] for s in self._history),
        }

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# FLOPs / per-step logging
# ---------------------------------------------------------------------------
def theoretical_step_flops(num_base_params: int, tokens_per_step: int) -> int:
    """Standard 6 x params x tokens (Chinchilla) forward+backward approximation."""
    return 6 * num_base_params * tokens_per_step


class StepLogger(TrainerCallback):
    """Per-step JSONL row on rank 0: loss/ppl, grad-norm, throughput, MFU, memory.

    Also works for the custom (non-HF-Trainer) tp loop via `log_step`.
    """

    def __init__(self, out_path: Path, seq_len: int, weight_baseline_bytes: int,
                 hw_peak_tflops: float, tokens_per_step: int,
                 nvml: "NVMLSampler | None" = None,
                 model: "torch.nn.Module | None" = None,
                 spike_window: int = 32, spike_sigma: float = 3.0):
        self.out_path = out_path
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.out_path, "w", buffering=1)
        self.weight_baseline_bytes = weight_baseline_bytes
        self.seq_len = seq_len
        self.hw_peak_tflops = hw_peak_tflops
        self.tokens_per_step = tokens_per_step
        self.nvml = nvml
        self.model = model
        self.rows: list[dict] = []
        self.t_prev = None
        self.mem_before_step = 0
        self.loss_history: list[float] = []
        self.spike_window = spike_window
        self.spike_sigma = spike_sigma
        self.spike_count = 0
        self.grad_anomaly_count = 0
        self.step_flops = theoretical_step_flops(BASE_PARAMS, tokens_per_step)

    def on_step_begin(self, args, state, control, **kwargs):
        self.t_prev = time.perf_counter()
        torch.cuda.synchronize()
        self.mem_before_step = torch.cuda.memory_allocated()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        dt = time.perf_counter() - self.t_prev
        self._pending_dt = dt

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or state.global_step == 0:
            return
        if "loss" not in logs:
            return
        dt = float(getattr(self, "_pending_dt", float("nan")))

        # on_log fires only on rank 0 in HF Trainer, so we cannot all_gather
        # per-rank step times here (other ranks would never join -> NCCL timeout).
        rank_times = [dt]

        mem_alloc_end = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()
        mem_peak_step = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        activation_delta = mem_peak_step - self.weight_baseline_bytes

        loss = float(logs.get("loss", float("nan")))
        grad_norm = float(logs.get("grad_norm", float("nan")))
        ppl = math.exp(min(loss, 20.0)) if loss == loss else float("nan")

        # Rolling loss median & spike detection
        self.loss_history.append(loss)
        if len(self.loss_history) > self.spike_window:
            window = self.loss_history[-self.spike_window:]
            med = st.median(window)
            sd = st.stdev(window) if len(window) > 1 else 0.0
            dev_pct = (loss - med) / max(abs(med), 1e-6) * 100
            spike_now = sd > 0 and abs(loss - med) > self.spike_sigma * sd
            if spike_now:
                self.spike_count += 1
        else:
            med = loss
            dev_pct = 0.0
            spike_now = False

        if grad_norm != grad_norm or grad_norm > 100 or math.isinf(grad_norm):
            self.grad_anomaly_count += 1

        tok_per_step = self.tokens_per_step
        tps_per_gpu = (tok_per_step / dt / world_size()) if dt > 0 else float("nan")
        achieved_tflops_per_gpu = (self.step_flops / dt / world_size() / 1e12) if dt > 0 else float("nan")
        mfu_pct = (achieved_tflops_per_gpu / self.hw_peak_tflops * 100) if self.hw_peak_tflops > 0 else float("nan")

        nvml_data = self.nvml.latest() if self.nvml else {}

        # LoRA grad norm split: attention (q/k/v/o) vs MLP (gate/up/down)
        attn_keys = ("q_proj", "k_proj", "v_proj", "o_proj")
        mlp_keys = ("gate_proj", "up_proj", "down_proj")
        attn_grads, mlp_grads = [], []
        if self.model is not None:
            for n, p in self.model.named_parameters():
                if p.grad is None or not p.requires_grad or "lora_" not in n:
                    continue
                if any(k in n for k in attn_keys):
                    attn_grads.append(p.grad.detach().norm().item())
                elif any(k in n for k in mlp_keys):
                    mlp_grads.append(p.grad.detach().norm().item())

        rank_min = min(rank_times)
        rank_max = max(rank_times)
        rank_std = float(torch.tensor(rank_times).std().item()) if len(rank_times) > 1 else 0.0
        rank_mean = sum(rank_times) / len(rank_times)

        row = {
            "step": state.global_step,
            "loss": loss,
            "perplexity": ppl,
            "loss_running_median_32": med,
            "loss_deviation_pct": dev_pct,
            "loss_spike_now": int(spike_now),
            "loss_spike_count_so_far": self.spike_count,
            "grad_norm": grad_norm,
            "grad_norm_lora_attn_mean": st.mean(attn_grads) if attn_grads else None,
            "grad_norm_lora_mlp_mean": st.mean(mlp_grads) if mlp_grads else None,
            "grad_anomaly_count_so_far": self.grad_anomaly_count,
            "lr": float(logs.get("learning_rate", float("nan"))),
            "step_time_sec": dt,
            "step_time_per_rank_min": rank_min,
            "step_time_per_rank_max": rank_max,
            "step_time_per_rank_mean": rank_mean,
            "step_time_per_rank_std": rank_std,
            "ddp_straggler_pct": (rank_max - rank_min) / max(rank_min, 1e-9) * 100,
            "tokens_per_sec_aggregate": tok_per_step / dt if dt > 0 else float("nan"),
            "tokens_per_sec_per_gpu": tps_per_gpu,
            "achieved_tflops_per_gpu": achieved_tflops_per_gpu,
            "mfu_pct": mfu_pct,
            "mem_alloc_step_end_gib": fmt_gib(mem_alloc_end),
            "mem_peak_step_gib": fmt_gib(mem_peak_step),
            "mem_reserved_gib": fmt_gib(mem_reserved),
            "activation_estimate_gib": fmt_gib(activation_delta),
            "gpu_power_w_mean": nvml_data.get("power_w_mean"),
            "gpu_power_w_node_total": nvml_data.get("power_w_total"),
            "gpu_sm_util_pct_mean": nvml_data.get("sm_util_pct_mean"),
            "gpu_mem_util_pct_mean": nvml_data.get("mem_util_pct_mean"),
            "gpu_temp_c_max": nvml_data.get("temp_c_max"),
        }
        self.rows.append(row)
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def log_step(self, step: int, loss: float, step_time: float,
                 lr: float = 0.0, grad_norm: float = float("nan")):
        """Direct-call variant for custom training loops (no HF Trainer state)."""
        self._pending_dt = step_time

        class _State:
            global_step = step

        self.on_log(None, _State(), None,
                    logs={"loss": loss, "learning_rate": lr, "grad_norm": grad_norm})

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# summary helpers (shared across variants)
# ---------------------------------------------------------------------------
def _pct(arr, q):
    if not arr:
        return None
    s = sorted(arr)
    i = max(0, min(len(s) - 1, int(round((q / 100) * (len(s) - 1)))))
    return s[i]


def write_summary(out_dir: Path, base: dict, logger_cb: StepLogger,
                  nvml_summary: dict, args, effective_gbs: int, grad_accum: int,
                  weight_baseline_bytes: int, peak_vram: float, wall_sec: float,
                  ws: int):
    """Assemble + write summary.json from collected per-step rows."""
    rows = logger_cb.rows if logger_cb else []
    trim_head = args.warmup_steps + max(1, (args.max_steps - args.warmup_steps) // 10)
    trim_tail = max(1, (args.max_steps - args.warmup_steps) // 10)
    measured = rows[trim_head: len(rows) - trim_tail] if len(rows) > trim_head + trim_tail else rows

    step_times = [r["step_time_sec"] for r in measured]
    tps_list = [r["tokens_per_sec_aggregate"] for r in measured]
    ddp_strag = [r["ddp_straggler_pct"] for r in measured]
    per_rank_max = [r["step_time_per_rank_max"] for r in measured]
    per_rank_min = [r["step_time_per_rank_min"] for r in measured]
    mfu_list = [r["mfu_pct"] for r in measured if r.get("mfu_pct") == r.get("mfu_pct")]
    tflops_list = [r["achieved_tflops_per_gpu"] for r in measured
                   if r.get("achieved_tflops_per_gpu") == r.get("achieved_tflops_per_gpu")]
    mean_st = st.mean(step_times) if step_times else float("nan")
    std_st = st.stdev(step_times) if len(step_times) > 1 else 0.0
    mean_tps = st.mean(tps_list) if tps_list else float("nan")

    all_losses = [r["loss"] for r in rows if r.get("loss") == r.get("loss")]
    loss_after_warmup = [r["loss"] for r in rows if r["step"] > args.warmup_steps]
    peak_tflops_hw = HW_PEAK_TFLOPS.get((args.hardware, args.quant), 0.0)

    summary = dict(base)
    summary.update({
        "model_id": MODEL_ID,
        "lora_rank": LORA_R, "lora_alpha": LORA_ALPHA, "lora_targets_regex": LORA_TARGETS,
        "base_params": BASE_PARAMS,
        "seq_len": args.seq_len,
        "micro_batch": args.mbsz, "grad_accum": grad_accum,
        "global_batch_sequences": effective_gbs,
        "global_batch_tokens": effective_gbs * args.seq_len,
        "warmup_steps": args.warmup_steps,
        "total_steps": args.max_steps,
        "measured_window_steps": len(measured),
        "trim_head": trim_head, "trim_tail": trim_tail,
        "mean_step_sec": mean_st, "std_step_sec": std_st,
        "step_time_p50_sec": _pct(step_times, 50),
        "step_time_p90_sec": _pct(step_times, 90),
        "step_time_p99_sec": _pct(step_times, 99),
        "step_time_cv_pct": (std_st / mean_st * 100) if mean_st > 0 else None,
        "tokens_per_sec_aggregate": mean_tps,
        "tokens_per_sec_per_gpu": mean_tps / ws if ws else float("nan"),
        "tps_p90_aggregate": _pct(tps_list, 90),
        "tps_p10_aggregate": _pct(tps_list, 10),
        "total_tokens_processed": effective_gbs * args.seq_len * args.max_steps,
        "theoretical_step_pflops": logger_cb.step_flops / 1e15 if logger_cb else None,
        "achieved_tflops_per_gpu_mean": st.mean(tflops_list) if tflops_list else None,
        "achieved_tflops_per_gpu_p90": _pct(tflops_list, 90),
        "hw_peak_tflops": peak_tflops_hw,
        "mfu_pct_mean": st.mean(mfu_list) if mfu_list else None,
        "mfu_pct_p90": _pct(mfu_list, 90),
        "peak_vram_gib_rank0": peak_vram,
        "weight_baseline_vram_gib": fmt_gib(weight_baseline_bytes),
        "activation_estimate_gib_max": max([r["activation_estimate_gib"] for r in measured], default=None),
        "activation_estimate_gib_mean": st.mean([r["activation_estimate_gib"] for r in measured]) if measured else None,
        "ddp_straggler_pct_p50": _pct(ddp_strag, 50),
        "ddp_straggler_pct_p99": _pct(ddp_strag, 99),
        "ddp_straggler_pct_max": max(ddp_strag) if ddp_strag else None,
        "per_rank_step_time_max_mean": st.mean(per_rank_max) if per_rank_max else None,
        "per_rank_step_time_min_mean": st.mean(per_rank_min) if per_rank_min else None,
        "loss_first": all_losses[0] if all_losses else None,
        "loss_last": all_losses[-1] if all_losses else None,
        "loss_min_after_warmup": min(loss_after_warmup) if loss_after_warmup else None,
        "loss_median_after_warmup": st.median(loss_after_warmup) if loss_after_warmup else None,
        "loss_spike_count_total": logger_cb.spike_count if logger_cb else 0,
        "grad_anomaly_count": logger_cb.grad_anomaly_count if logger_cb else 0,
        **{f"nvml_{k}": v for k, v in nvml_summary.items()},
        "world_size": ws,
        "wallclock_sec": wall_sec,
        # Relative-cost inputs only: GPU-hours consumed (no absolute $ values).
        "gpu_hours_consumed": ws * wall_sec / 3600.0,
        "grad_checkpointing": args.grad_checkpoint,
        "torch_compile": args.torch_compile,
        "seed": args.seed,
    })
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sft] summary -> {out_dir / 'summary.json'}", flush=True)

    # Optional wandb summary push (only if a project was configured).
    if os.environ.get("WANDB_PROJECT"):
        try:
            import wandb
            if wandb.run is not None:
                for k, v in summary.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        wandb.run.summary[f"final/{k}"] = v
        except Exception as e:
            print(f"[sft] wandb summary push skipped: {e}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def build_base_with_lora(args, rank: int, move_to_cuda: bool, local_rank: int):
    """Load base model (bf16) -> enable_input_require_grads -> attach LoRA.

    `move_to_cuda` is True for DDP (replicate per GPU) and False for FSDP/FSDP2
    (sharding handles placement during wrap).
    """
    if rank == 0:
        print(f"[sft] loading {MODEL_ID} (bf16)…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if rank == 0:
            print("[sft] gradient_checkpointing ENABLED (use_reentrant=False)", flush=True)

    model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # torchao FP8 conversion AFTER PEFT (skip lora_A/lora_B + lm_head/embed_tokens)
    if args.quant == "fp8":
        if not HAS_TORCHAO_FP8:
            sys.exit("ERROR: torchao.float8 not available — cannot use --quant fp8")
        if rank == 0:
            print("[sft] convert_to_float8_training (tensorwise)…", flush=True)
        fp8_cfg = Float8LinearConfig.from_recipe_name("tensorwise")

        def fp8_filter(mod, fqn: str) -> bool:
            return all(skip not in fqn for skip in ("lm_head", "embed_tokens", "lora_A", "lora_B"))

        convert_to_float8_training(model, config=fp8_cfg, module_filter_fn=fp8_filter)
        n_fp8 = sum(1 for m in model.modules() if "Float8Linear" in type(m).__name__)
        if rank == 0:
            print(f"[sft] FP8 modules: {n_fp8}", flush=True)

    if move_to_cuda:
        model = model.cuda(local_rank)
    return model


# ---------------------------------------------------------------------------
# DDP micro-batch autotune
# ---------------------------------------------------------------------------
def autotune_mbsz(args, rank: int, local_rank: int, candidates=(1, 2, 4, 8)):
    """Probe each mbsz with a single fwd+bwd+optstep on the actual model.

    Returns (chosen_mbsz, attempt_log) — chosen = largest that fit.
    """
    attempts = []
    chosen = None
    device = f"cuda:{local_rank}"
    for mbsz in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ok, peak_gib, err = True, None, None
        try:
            if rank == 0:
                print(f"[autotune] probing mbsz={mbsz}…", flush=True)
            ids = torch.randint(0, 100000, (mbsz, args.seq_len), device=device, dtype=torch.long)
            labels = ids.clone()
            out = args._autotune_model(input_ids=ids, labels=labels)
            out.loss.backward()
            args._autotune_optimizer.step()
            args._autotune_optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak_gib = fmt_gib(torch.cuda.max_memory_allocated())
            chosen = mbsz
        except torch.cuda.OutOfMemoryError:
            ok, err = False, "OOM"
            torch.cuda.empty_cache()
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {str(e)[:120]}"
            torch.cuda.empty_cache()
        attempts.append({"mbsz": mbsz, "fit": ok, "peak_vram_gib": peak_gib, "error": err})
        if rank == 0:
            print(f"[autotune]   mbsz={mbsz} fit={ok} peak={peak_gib} err={err}", flush=True)
        if not ok:
            break
    if chosen is None:
        if rank == 0:
            print("[autotune] no mbsz fit! falling back to 1.", flush=True)
        chosen = 1
    return chosen, attempts


# ---------------------------------------------------------------------------
# wandb (optional; bring-your-own project/entity, no internal defaults)
# ---------------------------------------------------------------------------
def maybe_setup_wandb(args, run_name: str, group_suffix: str, rank: int):
    if rank != 0:
        return
    if not os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_DISABLED"] = "true"
        return
    os.environ.setdefault("WANDB_RESUME", "allow")
    os.environ["WANDB_RUN_GROUP"] = f"{args.hardware}-{args.quant}{group_suffix}"
    os.environ["WANDB_NAME"] = run_name
    os.environ["WANDB_RUN_ID"] = run_name


def report_to_list() -> list:
    return ["wandb"] if os.environ.get("WANDB_PROJECT") else []


# ===========================================================================
# Variant: DDP (baseline) + FSDP + FSDP-nogc  (all HF-Trainer based)
# ===========================================================================
def run_trainer_variant(args, rank: int, local_rank: int):
    ws = world_size()
    is_fsdp = args.variant in ("fsdp", "fsdp-nogc")
    suffix = f"-{args.variant}" if args.variant != "ddp" else ""
    run_name = f"T-{args.hardware}-{args.quant}{suffix}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        os.environ.get("OUTPUT_BASE", "/shared")
    ) / "runs" / f"{args.hardware}_{args.quant}{suffix}" / f"T-{args.hardware}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_setup_wandb(args, run_name, "-fsdp" if is_fsdp else "", rank)

    if rank == 0:
        print(f"[sft] variant={args.variant} run_name={run_name} out_dir={out_dir}", flush=True)
        print(f"[sft] hardware={args.hardware} quant={args.quant} world_size={ws} "
              f"seq_len={args.seq_len} grad_checkpoint={args.grad_checkpoint}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = load_from_disk(args.dataset)
    if rank == 0:
        print(f"[sft] dataset: {len(ds)} packs x {args.seq_len} tokens", flush=True)

    # DDP replicates to GPU now; FSDP defers placement to auto_wrap during prepare.
    model = build_base_with_lora(args, rank, move_to_cuda=(not is_fsdp), local_rank=local_rank)

    if args.quant == "te-fp8":
        model = apply_te_fp8(model, rank)

    # ---- micro-batch ----
    autotune_log: list = []
    if not is_fsdp and args.mbsz == "auto":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        probe_optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        args._autotune_model = model
        args._autotune_optimizer = probe_optim
        chosen_mbsz, autotune_log = autotune_mbsz(args, rank, local_rank)
        del probe_optim
        for p in model.parameters():
            p.grad = None
        torch.cuda.empty_cache()
        if dist.is_initialized():
            t = torch.tensor([chosen_mbsz], device=f"cuda:{local_rank}")
            dist.broadcast(t, 0)
            chosen_mbsz = int(t.item())
        args.mbsz = chosen_mbsz
    else:
        args.mbsz = int(args.mbsz) if args.mbsz != "auto" else 1
    if rank == 0:
        print(f"[sft] mbsz={args.mbsz} (autotune: {autotune_log})", flush=True)

    grad_accum = max(1, GLOBAL_BATCH_SEQ // (args.mbsz * ws))
    effective_gbs = args.mbsz * grad_accum * ws
    if rank == 0:
        print(f"[sft] grad_accum={grad_accum} global_batch_seq={effective_gbs}", flush=True)

    if args.torch_compile:
        torch.set_float32_matmul_precision("high")

    sft_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.mbsz,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        adam_beta1=0.9, adam_beta2=0.95, weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=args.grad_checkpoint,
        bf16=True, fp16=False,
        logging_strategy="steps", logging_steps=1, logging_first_step=False,
        save_strategy="no",
        report_to=report_to_list(),
        run_name=run_name,
        dataloader_num_workers=2,
        max_length=args.seq_len,
        packing=False,
        remove_unused_columns=False,
        dataset_text_field=None,
        seed=args.seed,
        torch_compile=args.torch_compile,
        torch_compile_backend="inductor",
    )

    fsdp_config = None
    if is_fsdp:
        # PEFT's fsdp_auto_wrap_policy reads this env var to find the wrap class.
        os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = GEMMA_LAYER_CLS
        fsdp_config = {
            "transformer_layer_cls_to_wrap": [GEMMA_LAYER_CLS],
            "use_orig_params": True,
            "limit_all_gathers": True,
            "forward_prefetch": True,
            "sync_module_states": True,
            "backward_prefetch": "BACKWARD_PRE",
        }
        sft_kwargs.update(fsdp="full_shard auto_wrap", fsdp_config=fsdp_config)
    else:
        sft_kwargs.update(ddp_find_unused_parameters=False)

    train_args = SFTConfig(**sft_kwargs)

    # weight baseline before activations (DDP path; FSDP updates post-wrap below)
    weight_baseline_bytes = 0
    if not is_fsdp:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        weight_baseline_bytes = torch.cuda.memory_allocated()

    callbacks, nvml, logger_cb = [], None, None
    if rank == 0:
        nvml = NVMLSampler(interval_s=2.0)
        nvml.start()
        peak_tflops = HW_PEAK_TFLOPS.get((args.hardware, args.quant), 0.0)
        logger_cb = StepLogger(
            out_dir / "loss_log.jsonl", args.seq_len, weight_baseline_bytes,
            hw_peak_tflops=peak_tflops,
            tokens_per_step=effective_gbs * args.seq_len,
            nvml=nvml, model=model,
        )
        callbacks.append(logger_cb)

    if args.quant == "te-fp8":
        recipe = DelayedScaling(margin=0, fp8_format=TEFormat.HYBRID,
                                amax_history_len=16, amax_compute_algo="max")
        trainer = TEFp8Trainer(model=model, args=train_args, train_dataset=ds,
                               processing_class=tok, callbacks=callbacks, fp8_recipe=recipe)
    else:
        trainer = SFTTrainer(model=model, args=train_args, train_dataset=ds,
                             processing_class=tok, callbacks=callbacks)

    # FSDP: record the true post-wrap baseline VRAM.
    if is_fsdp:
        torch.cuda.synchronize()
        weight_baseline_bytes = torch.cuda.memory_allocated()
        if rank == 0 and logger_cb is not None:
            logger_cb.weight_baseline_bytes = weight_baseline_bytes

    if rank == 0:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        print(f"[sft] trainable_params={trainable:,} / total={total:,} "
              f"({trainable / total * 100:.3f}%)", flush=True)
        print(f"[sft] weight_baseline_vram={fmt_gib(weight_baseline_bytes):.2f} GiB", flush=True)

    t0 = time.time()
    if rank == 0:
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    t1 = time.time()

    if rank == 0:
        nvml_summary = nvml.summary() if nvml else {}
        if nvml:
            nvml.stop()
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # FSDP shards trainable params across ranks; scale to a global count.
        trainable = int(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))
        base = {
            "run_name": run_name,
            "hardware": args.hardware, "quant": args.quant, "variant": args.variant,
            "trainable_params": trainable * (ws if is_fsdp else 1),
        }
        if is_fsdp:
            base["parallelism"] = "FSDP-full_shard"
            base["fsdp_transformer_layer_cls_to_wrap"] = GEMMA_LAYER_CLS
            base["fsdp_config"] = fsdp_config
        else:
            base["parallelism"] = "DDP"
        base["mbsz_autotune_log"] = autotune_log

        write_summary(out_dir, base, logger_cb, nvml_summary, args, effective_gbs,
                      grad_accum, weight_baseline_bytes, peak_vram, t1 - t0, ws)

        # FSDP save_model can hang on the state_dict gather; allow skipping it.
        if args.no_save_checkpoint:
            print("[sft] --no-save-checkpoint set: skipping save_model()", flush=True)
        else:
            trainer.save_model(str(out_dir / "checkpoint-final"))
            print(f"[sft] LoRA adapter -> {out_dir / 'checkpoint-final'}", flush=True)
        logger_cb.close()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ===========================================================================
# Variant: TP (FSDP2 + DeviceMesh, custom training loop)
# ===========================================================================
def run_tp_variant(args, rank: int, local_rank: int):
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    from transformers.optimization import get_cosine_schedule_with_warmup

    ws = world_size()
    assert ws % args.tp_size == 0, f"world_size={ws} not divisible by tp-size={args.tp_size}"
    dp_size = ws // args.tp_size

    variant_tag = args.variant if args.variant not in ("tp", "") else f"fsdp2-tp{args.tp_size}"
    run_name = f"T-{args.hardware}-{args.quant}-{variant_tag}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        os.environ.get("OUTPUT_BASE", "/shared")
    ) / "runs" / f"{args.hardware}_{args.quant}-{variant_tag}" / f"T-{args.hardware}-{variant_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_setup_wandb(args, run_name, "-tp", rank)

    if rank == 0:
        print(f"[sft-tp] run_name={run_name} out_dir={out_dir}", flush=True)
        print(f"[sft-tp] world_size={ws} dp={dp_size} shard(tp)={args.tp_size} mbsz={args.mbsz}", flush=True)

    mesh = init_device_mesh("cuda", (dp_size, args.tp_size), mesh_dim_names=["dp", "shard"])
    shard_mesh = mesh["shard"]
    dp_group = mesh["dp"].get_group()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds_all = load_from_disk(args.dataset)
    if rank == 0:
        print(f"[sft-tp] dataset: {len(ds_all)} packs x {args.seq_len} tokens", flush=True)

    # Build base + LoRA (no .cuda(); FSDP2 places shards), then wrap with FSDP2.
    model = build_base_with_lora(args, rank, move_to_cuda=False, local_rank=local_rank)

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in [m for _, m in model.named_modules() if isinstance(m, Gemma4TextDecoderLayer)]:
        fully_shard(layer, mesh=shard_mesh, mp_policy=mp)
    fully_shard(model, mesh=shard_mesh, mp_policy=mp)
    model.train()

    torch.cuda.synchronize()
    weight_baseline_bytes = torch.cuda.memory_allocated()
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[sft-tp] trainable (per-shard)={trainable:,} / total={total:,}", flush=True)
        print(f"[sft-tp] weight_baseline={fmt_gib(weight_baseline_bytes):.2f} GiB", flush=True)

    args.mbsz = int(args.mbsz) if args.mbsz != "auto" else 4
    grad_accum = max(1, GLOBAL_BATCH_SEQ // (args.mbsz * ws))
    effective_gbs = args.mbsz * grad_accum * ws
    tokens_per_step = effective_gbs * args.seq_len
    if rank == 0:
        print(f"[sft-tp] grad_accum={grad_accum} global_batch_seq={effective_gbs}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0,
    )
    total_steps = args.max_steps * grad_accum
    warmup = args.warmup_steps * grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    dp_rank = mesh["dp"].get_local_rank()
    indices = list(range(dp_rank, len(ds_all), dp_size))
    ds_local = ds_all.select(indices)

    def collate(batch):
        ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

    loader = torch.utils.data.DataLoader(
        ds_local, batch_size=args.mbsz, shuffle=True, collate_fn=collate,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    data_iter = iter(loader)

    nvml, logger_cb = None, None
    if rank == 0:
        nvml = NVMLSampler(interval_s=2.0)
        nvml.start()
        peak_tflops = HW_PEAK_TFLOPS.get((args.hardware, args.quant), 0.0)
        logger_cb = StepLogger(
            out_dir / "loss_log.jsonl", args.seq_len, weight_baseline_bytes,
            hw_peak_tflops=peak_tflops, tokens_per_step=tokens_per_step,
            nvml=nvml, model=model,
        )
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    step = micro_step = 0
    optimizer.zero_grad()
    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / grad_accum
        loss.backward()
        micro_step += 1

        if micro_step % grad_accum == 0:
            step_t0 = time.time()
            # manual grad sync across dp groups (FSDP2 handles intra-shard sync)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad, group=dp_group, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            step_t1 = time.time()
            if rank == 0 and logger_cb is not None:
                cur_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 0.0
                logger_cb.log_step(step, float(loss.item() * grad_accum), step_t1 - step_t0, lr=cur_lr)
                if step % 10 == 0:
                    print(f"  step={step}/{args.max_steps} loss={loss.item() * grad_accum:.4f} "
                          f"tps={tokens_per_step / (step_t1 - step_t0):.0f}", flush=True)
    t1 = time.time()

    if rank == 0:
        nvml_summary = nvml.summary() if nvml else {}
        if nvml:
            nvml.stop()
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        # FSDP2 shards within tp_size GPUs -> scale per-shard trainable by tp_size.
        trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad) * args.tp_size)
        base = {
            "run_name": run_name,
            "hardware": args.hardware, "quant": args.quant, "variant": variant_tag,
            "parallelism": f"FSDP2-shard{args.tp_size}xDP{dp_size}",
            "tp_size": args.tp_size, "dp_size": dp_size,
            "trainable_params": trainable,
            "mbsz_autotune_log": [],
        }
        write_summary(out_dir, base, logger_cb, nvml_summary, args, effective_gbs,
                      grad_accum, weight_baseline_bytes, peak_vram, t1 - t0, ws)

        if not args.no_save_checkpoint:
            model.save_pretrained(str(out_dir / "checkpoint-final"))
            print(f"[sft-tp] LoRA adapter -> {out_dir / 'checkpoint-final'}", flush=True)
        else:
            print("[sft-tp] --no-save-checkpoint: skipping", flush=True)
        if logger_cb:
            logger_cb.close()

    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=os.environ.get("VARIANT", "ddp"),
                    choices=["ddp", "fsdp", "fsdp-nogc", "tp"],
                    help="parallelism strategy (env: VARIANT)")
    ap.add_argument("--hardware", default=os.environ.get("HARDWARE"),
                    choices=["A100", "H100", "B200"], help="env: HARDWARE")
    ap.add_argument("--quant", default=os.environ.get("QUANT", "bf16"),
                    choices=["bf16", "fp8", "te-fp8"], help="env: QUANT")
    ap.add_argument("--dataset",
                    default=os.environ.get("DATASET", "/shared/datasets/sharegpt_packed_2048"),
                    help="load_from_disk packed dataset path (env: DATASET)")
    ap.add_argument("--out-dir", default=os.environ.get("OUTPUT_DIR"),
                    help="output dir (env: OUTPUT_DIR; default derived under $OUTPUT_BASE/runs/...)")
    ap.add_argument("--tp-size", type=int, default=int(os.environ.get("TP_SIZE", 2)),
                    choices=[2, 4, 8], help="tp variant shard-group size (env: TP_SIZE)")
    ap.add_argument("--mbsz", default=os.environ.get("MBSZ", "auto"),
                    help='per-device micro batch; "auto" (ddp only) probes 1->2->4->8')
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="enable gradient checkpointing (ddp/fsdp)")
    ap.add_argument("--torch-compile", action="store_true", help="enable torch.compile (inductor)")
    ap.add_argument("--no-save-checkpoint", action="store_true",
                    help="skip saving the LoRA adapter (fsdp/tp)")
    args = ap.parse_args()

    if not args.hardware:
        sys.exit("ERROR: --hardware (or HARDWARE env) required: A100|H100|B200")

    # fsdp-nogc is the FSDP path with gradient checkpointing explicitly OFF.
    if args.variant == "fsdp-nogc":
        args.grad_checkpoint = False

    # Validity gates.
    if args.hardware == "A100" and args.quant in ("fp8", "te-fp8"):
        sys.exit("ERROR: A100 has no native FP8 — use --quant bf16")
    if args.quant == "te-fp8" and not HAS_TE:
        sys.exit("ERROR: transformer_engine not installed — cannot use --quant te-fp8")
    if args.variant in ("fsdp", "fsdp-nogc", "tp") and args.quant != "bf16":
        sys.exit(f"ERROR: variant {args.variant} is bf16-only (FP8+shard+PEFT untested)")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)

    if rank == 0:
        print(f"[sft] python {sys.version.split()[0]} torch {torch.__version__} "
              f"variant={args.variant}", flush=True)

    if args.variant == "tp":
        run_tp_variant(args, rank, local_rank)
    else:
        run_trainer_variant(args, rank, local_rank)


if __name__ == "__main__":
    main()
