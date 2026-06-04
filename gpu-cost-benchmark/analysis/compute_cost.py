"""Compute a RELATIVE cost index per Mtok from the benchmark result CSVs.

This produces ONLY a relative cost index (A100 = 100 baseline); no absolute
dollar figures are computed or written. The per-hardware index reflects the
relative price of a GPU-hour normalised so that A100 = 100. Tune the indices
for your own provider via the COST_INDEX_{A100,H100,B200} env vars.

Cost index defaults:
  COST_INDEX_A100 = 100
  COST_INDEX_H100 = 210
  COST_INDEX_B200 = 389

Inputs (read from ``<recipe>/results/``):
  training.csv    — per-config training throughput
  inference.csv   — per-cell / per-concurrency inference throughput

Output (written to ``<recipe>/results/``):
  cost_summary.csv  — relative cost index per Mtok (training + inference peak)

Usage:
  python analysis/compute_cost.py

Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TRAIN_CSV = RESULTS / "training.csv"
INFER_CSV = RESULTS / "inference.csv"
COST_CSV = RESULTS / "cost_summary.csv"

DEFAULT_INDEX = {"A100": 100.0, "H100": 210.0, "B200": 389.0}


def cost_index(hw: str) -> float:
    return float(os.environ.get(f"COST_INDEX_{hw}", DEFAULT_INDEX[hw]))


def gpu_hours_per_mtok(tps_aggregate: float, world_size: int = 8) -> float:
    """8 GPUs running at `tps_aggregate` tok/s for 1M tokens."""
    seconds_per_mtok = 1_000_000.0 / tps_aggregate
    return world_size * seconds_per_mtok / 3600.0


def main():
    rows_train, rows_infer = [], []
    if TRAIN_CSV.exists():
        with open(TRAIN_CSV) as f:
            rows_train = list(csv.DictReader(f))
    if INFER_CSV.exists():
        with open(INFER_CSV) as f:
            rows_infer = list(csv.DictReader(f))

    # ---- training cost ----
    out_train = []
    a100_train_gpu_h_per_mtok = None
    for r in rows_train:
        hw = r["hardware"]
        tps = float(r.get("tokens_per_sec_aggregate") or 0)
        if tps <= 0:
            continue
        ws = int(r.get("world_size") or 8)
        gpu_h = gpu_hours_per_mtok(tps, ws)
        out_train.append({
            "hardware": hw, "scope": "training",
            "tps_aggregate": tps, "world_size": ws,
            "gpu_hours_per_mtok": gpu_h,
            "cost_index_hw": cost_index(hw),
            "cost_idx_per_mtok": gpu_h * cost_index(hw),
        })
    # baseline = A100 training
    for r in out_train:
        if r["hardware"] == "A100":
            a100_train_gpu_h_per_mtok = r["gpu_hours_per_mtok"] * 100.0
            break
    for r in out_train:
        if a100_train_gpu_h_per_mtok is not None:
            r["relative_index_vs_A100"] = round(r["cost_idx_per_mtok"] / a100_train_gpu_h_per_mtok * 100, 2)
        else:
            r["relative_index_vs_A100"] = ""

    # ---- inference cost — peak across cells/conc per hardware ----
    by_hw = {}
    for r in rows_infer:
        hw = r["hardware"]
        try:
            tps = float(r.get("throughput_out_tok_per_s") or 0)
        except ValueError:
            tps = 0
        if tps <= 0:
            continue
        by_hw.setdefault(hw, []).append((tps, r))
    out_infer = []
    a100_infer_gpu_h_per_mtok = None
    for hw, rows in by_hw.items():
        peak_tps, peak_row = max(rows, key=lambda x: x[0])
        gpu_h = gpu_hours_per_mtok(peak_tps, 8)
        out_infer.append({
            "hardware": hw, "scope": "inference_peak",
            "cell_id": peak_row["cell_id"], "concurrency": peak_row["concurrency_per_instance"],
            "tps_aggregate": peak_tps, "world_size": 8,
            "gpu_hours_per_mtok": gpu_h,
            "cost_index_hw": cost_index(hw),
            "cost_idx_per_mtok": gpu_h * cost_index(hw),
        })
    for r in out_infer:
        if r["hardware"] == "A100":
            a100_infer_gpu_h_per_mtok = r["cost_idx_per_mtok"]
            break
    for r in out_infer:
        if a100_infer_gpu_h_per_mtok is not None:
            r["relative_index_vs_A100"] = round(r["cost_idx_per_mtok"] / a100_infer_gpu_h_per_mtok * 100, 2)
        else:
            r["relative_index_vs_A100"] = ""

    # write relative-index CSV (no absolute currency)
    public_cols = ["hardware", "scope", "tps_aggregate", "world_size",
                   "gpu_hours_per_mtok", "cost_index_hw", "cost_idx_per_mtok",
                   "relative_index_vs_A100", "cell_id", "concurrency"]
    with open(COST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=public_cols)
        w.writeheader()
        for r in out_train + out_infer:
            w.writerow({c: r.get(c, "") for c in public_cols})
    print(f"[cost] wrote {COST_CSV}")

    # human-readable summary
    print("\n[cost] training")
    for r in out_train:
        print(f"  {r['hardware']:5s}  TPS={r['tps_aggregate']:.1f}  "
              f"index/Mtok = {r['cost_idx_per_mtok']:.2f}  "
              f"(rel A100=100 -> {r.get('relative_index_vs_A100','-')})")
    print("\n[cost] inference (peak across cells)")
    for r in out_infer:
        print(f"  {r['hardware']:5s}  TPS={r['tps_aggregate']:.1f}  "
              f"cell={r['cell_id']} c={r['concurrency']}  "
              f"index/Mtok = {r['cost_idx_per_mtok']:.2f}  "
              f"(rel A100=100 -> {r.get('relative_index_vs_A100','-')})")


if __name__ == "__main__":
    main()
