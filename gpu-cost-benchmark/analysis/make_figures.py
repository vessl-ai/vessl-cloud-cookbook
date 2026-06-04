"""Generate benchmark summary figures for the GPU cost benchmark recipe.

Reads the measured results CSVs (relative to this script) and renders a set of
English-labeled figures using a RELATIVE cost index (A100 = baseline). No
absolute dollar values are computed or shown.

Inputs (read from ``<recipe>/results/``):
  training.csv             — per-config training throughput / VRAM / GPU-hours
  inference.csv            — per-cell / per-concurrency inference throughput
  stability_metrics.json   — per-hardware training stability metrics (fig11)

Outputs (written to ``<recipe>/images/``):
  fig1_training_tps.png        — Training TPS per HW x config (grouped bars)
  fig2_training_cost.png       — Training cost index per Mtok (sorted bar)
  fig3_inference_tps.png       — Inference TPS cross-HW x cell type
  fig4_mtp_effect.png          — MTP gamma speedup (B200 + H100)
  fig6_vram_vs_tps.png         — VRAM vs TPS scatter (training efficiency)
  fig7_kvs_effect.png          — FP8 KV cache optimization (B200 + H100)
  fig8_inference_cost.png      — Inference cost index per Mtok (sorted)
  fig9_concurrency_sweep.png   — TPS vs concurrency sweep (B200 + H100 kvs)
  fig10_lora_overhead.png      — LoRA overhead per HW x MTP config
  fig11_stability_heatmap.png  — Training stability heatmap (3 HW x 5 metrics)

Usage:
  python analysis/make_figures.py

Copyright 2026 VESSL AI Inc. Licensed under Apache-2.0.
"""
import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_OUT = ROOT / "images"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# Colour palette (consistent across all figures)
HW_COLOR = {"A100": "#4878CF", "H100": "#E87D2C", "B200": "#6ABF6A"}
HW_ORDER = ["A100", "H100", "B200"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# Load data
tr = pd.read_csv(RESULTS / "training.csv")
inf = pd.read_csv(RESULTS / "inference.csv")


# Helper
def save(fig, name):
    path = FIG_OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {path.relative_to(ROOT)}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — Training TPS comparison (key configs per HW)
# ════════════════════════════════════════════════════════════════════════════
def fig1_training_tps():
    # Select representative rows per HW (best DDP, best FSDP, baseline for context)
    rows = [
        # (label, HW, TPS, style)
        ("A100\nDDP+GC",        "A100", 3565,  "solid"),
        ("A100\nFSDP-noGC",     "A100", 4746,  "solid"),
        ("A100\nFSDP+GC",       "A100", 3139,  "light"),
        ("H100\nbf16 DDP+GC",   "H100", 9021,  "solid"),
        ("H100\nFSDP mb4",      "H100", 9559,  "solid"),
        ("H100\nFSDP mb2",      "H100", 8572,  "light"),
        ("B200\nbf16 baseline", "B200", 14201, "light"),
        ("B200\nbf16+compile",  "B200", 17634, "solid"),
        ("B200\nte-fp8-compile","B200", 17539, "solid"),
    ]
    labels = [r[0] for r in rows]
    tps    = [r[2] for r in rows]
    hws    = [r[1] for r in rows]
    alphas = [1.0 if r[3] == "solid" else 0.55 for r in rows]
    colors = [HW_COLOR[hw] for hw in hws]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(labels)), tps, color=colors,
                   alpha=1.0, edgecolor="white", linewidth=0.5)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    # value labels
    for bar, val in zip(bars, tps):
        ax.text(val + 100, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", ha="left", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Aggregate Tokens / sec  (8-GPU node)")
    ax.set_title("Training Throughput — Gemma-4 31B LoRA SFT  (A100 x H100 x B200, 8 GPUs)")
    ax.set_xlim(0, max(tps) * 1.15)

    # legend
    patches = [mpatches.Patch(color=HW_COLOR[hw], label=hw) for hw in HW_ORDER]
    ax.legend(handles=patches, loc="upper right")

    # dividers between HW groups
    for x in [2.5, 5.5]:
        ax.axhline(x, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig1_training_tps.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — Training cost index per Mtok (sorted, all configs)
# ════════════════════════════════════════════════════════════════════════════
def fig2_training_cost():
    cost_rows = [
        ("A100 FSDP-noGC",       "A100",  46.83),
        ("H100 FSDP mb4",        "H100",  48.82),
        ("B200 bf16+compile",    "B200",  49.01),
        ("B200 te-fp8-compile",  "B200",  49.29),
        ("B200 fp8+compile",     "B200",  49.72),
        ("H100 FSDP mb2",        "H100",  54.44),
        ("H100 bf16 DDP+GC",     "H100",  51.73),
        ("A100 FSDP-GC mb2",     "A100",  55.72),
        ("A100 FSDP-GC mb4",     "A100",  54.96),
        ("B200 bf16 baseline",   "B200",  60.87),
        ("A100 DDP+GC",          "A100",  62.33),
        ("H100 fp8 DDP+GC",      "H100", 125.07),
        ("B200 fp8 baseline",    "B200", 126.53),
        ("B200 fp8+GC",          "B200", 168.48),
    ]
    cost_rows.sort(key=lambda x: x[2])
    labels = [r[0] for r in cost_rows]
    costs  = [r[2] for r in cost_rows]
    hws    = [r[1] for r in cost_rows]
    colors = [HW_COLOR[hw] for hw in hws]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(labels)), costs, color=colors,
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, costs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", ha="left", fontsize=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Cost Index per Mtok  (A100 DDP = 100; lower = cheaper)")
    ax.set_title("Training Cost Efficiency — Relative Index per Mtok")
    ax.set_xlim(0, max(costs) * 1.12)

    # reference line at A100 DDP baseline = 62.33
    ax.axvline(62.33, color="#4878CF", linewidth=1.2, linestyle="--", alpha=0.7,
               label="A100 DDP+GC (62.3)")
    ax.legend(fontsize=8)

    patches = [mpatches.Patch(color=HW_COLOR[hw], label=hw) for hw in HW_ORDER]
    ax.legend(handles=patches + [
        mpatches.Patch(facecolor="none", edgecolor="#4878CF",
                       linestyle="--", label="A100 DDP baseline")],
              loc="lower right", fontsize=8)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_training_cost.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — Inference TPS cross-HW (I1/I2/I3/I4/I5 per HW)
# ════════════════════════════════════════════════════════════════════════════
def fig3_inference_tps():
    # peak TPS per (HW, cell_id) from inference.csv
    # mtp_on: 0=no, 1=yes; lora_on: 0=no, 1=yes
    cells_wanted = ["I1", "I2", "I3", "I4", "I5"]

    # group by HW, cell_id -> take max throughput_out_tok_per_s
    peak = (
        inf[inf["cell_id"].isin(cells_wanted)]
        .groupby(["hardware", "cell_id"])["throughput_out_tok_per_s"]
        .max()
        .reset_index()
        .rename(columns={"throughput_out_tok_per_s": "tps"})
    )

    # cell display labels
    cell_labels = {
        "I1": "I1\nbase, no MTP",
        "I2": "I2\nbase, MTP g=1",
        "I3": "I3\nLoRA, no MTP",
        "I4": "I4\nLoRA, MTP g=1",
        "I5": "I5\nbase, MTP g=2",
    }

    fig, ax = plt.subplots(figsize=(9, 4.5))

    n_cells = len(cells_wanted)
    n_hw    = len(HW_ORDER)
    bar_w   = 0.22
    x       = np.arange(n_cells)

    for i, hw in enumerate(HW_ORDER):
        hw_data = peak[peak["hardware"] == hw].set_index("cell_id")
        vals = [hw_data.loc[c, "tps"] if c in hw_data.index else 0 for c in cells_wanted]
        offset = (i - 1) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=hw, color=HW_COLOR[hw],
                      edgecolor="white", linewidth=0.4)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val,
                        f"{val/1000:.0f}k" if val >= 1000 else f"{val:.0f}",
                        ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([cell_labels[c] for c in cells_wanted], fontsize=8.5)
    ax.set_ylabel("Aggregate Output Tokens / sec  (8-GPU node)")
    ax.set_title("Inference Throughput — Gemma-4 31B  (FP8-block, vLLM)")
    ax.legend(title="Hardware")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_inference_tps.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — MTP gamma effect: TPS speedup vs gamma for B200 and H100
# ════════════════════════════════════════════════════════════════════════════
def fig4_mtp_effect():
    data = {
        "B200 base":  [(0, 18142), (1, 27812), (2, 32198), (3, 34553)],
        "B200 LoRA":  [(0, 13773), (1, 19018), (2, 21880)],
        "H100 base":  [(0, 11600), (1, 16665), (2, 18235)],
        "H100 LoRA":  [(0,  9390), (1, 12462), (2, 13941)],
        "A100 base":  [(0,  1774), (1,  2591)],
        "A100 LoRA":  [(0,  1497), (1,  2376)],
    }
    styles = {
        "B200 base":  dict(color=HW_COLOR["B200"], linestyle="-",  marker="o", lw=2),
        "B200 LoRA":  dict(color=HW_COLOR["B200"], linestyle="--", marker="s", lw=1.5),
        "H100 base":  dict(color=HW_COLOR["H100"], linestyle="-",  marker="o", lw=2),
        "H100 LoRA":  dict(color=HW_COLOR["H100"], linestyle="--", marker="s", lw=1.5),
        "A100 base":  dict(color=HW_COLOR["A100"], linestyle="-",  marker="o", lw=2),
        "A100 LoRA":  dict(color=HW_COLOR["A100"], linestyle="--", marker="s", lw=1.5),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: absolute TPS
    for name, pts in data.items():
        gammas, tps_vals = zip(*pts)
        ax1.plot(gammas, tps_vals, label=name, **styles[name])
        ax1.scatter(gammas, tps_vals, color=styles[name]["color"],
                    marker=styles[name]["marker"], s=40, zorder=5)
    ax1.set_xlabel("MTP draft tokens (gamma)")
    ax1.set_ylabel("Aggregate TPS (8-GPU node)")
    ax1.set_title("MTP gamma Effect on TPS")
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(["g=0\n(no MTP)", "g=1", "g=2", "g=3"])
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: relative speedup vs gamma=0 (normalised per series)
    for name, pts in data.items():
        base_tps = dict(pts)[0]
        gammas, tps_vals = zip(*pts)
        speedup = [t / base_tps for t in tps_vals]
        ax2.plot(gammas, speedup, label=name, **styles[name])
        ax2.scatter(gammas, speedup, color=styles[name]["color"],
                    marker=styles[name]["marker"], s=40, zorder=5)
    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("MTP draft tokens (gamma)")
    ax2.set_ylabel("Speedup vs no-MTP (gamma=0)")
    ax2.set_title("MTP Relative Speedup")
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(["g=0\n(no MTP)", "g=1", "g=2", "g=3"])
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("MTP Speculative Decoding Effect  (Gemma-4 31B, ShareGPT workload)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "fig4_mtp_effect.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — VRAM vs TPS scatter (training efficiency)
# ════════════════════════════════════════════════════════════════════════════
def fig6_vram_tps():
    rows = [
        # (label, hw, TPS, VRAM, parallelism)
        ("DDP+GC",           "A100", 3565,  60.1, "DDP"),
        ("FSDP+GC mb1",      "A100", 3139,  20.7, "FSDP"),
        ("FSDP+GC mb2",      "A100", 3986,  29.8, "FSDP"),
        ("FSDP+GC mb4",      "A100", 4043,  48.4, "FSDP"),
        ("FSDP-noGC",        "A100", 4746,  56.6, "FSDP"),
        ("FSDP+compile+GC",  "A100", 3301,  17.7, "FSDP"),
        ("fp8 DDP+GC",       "H100", 3731,  60.1, "DDP"),
        ("bf16 DDP+GC",      "H100", 9021,  60.1, "DDP"),
        ("FSDP+GC mb1",      "H100", 5204,   7.6, "FSDP"),
        ("FSDP+GC mb2",      "H100", 8572,   7.6, "FSDP"),
        ("FSDP+GC mb4",      "H100", 9559,  52.5, "FSDP"),
        ("fp8 default",      "B200", 6832, 136.8, "DDP"),
        ("bf16 baseline",    "B200",14201,  61.0, "DDP"),
        ("bf16+compile",     "B200",17634,  61.0, "DDP"),
        ("te-fp8-compile",   "B200",17539,  61.0, "DDP"),
    ]

    marker_map = {"DDP": "o", "FSDP": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for hw in HW_ORDER:
        hw_rows = [r for r in rows if r[1] == hw]
        for r in hw_rows:
            label, _, tps, vram, par = r
            ax.scatter(vram, tps, color=HW_COLOR[hw],
                       marker=marker_map[par], s=70, zorder=4,
                       edgecolors="white", linewidths=0.5)
            # annotate key points
            if any(k in label for k in ["FSDP-noGC", "bf16+compile",
                                         "te-fp8", "FSDP+GC mb4", "DDP+GC"]):
                ax.annotate(label, (vram, tps),
                            textcoords="offset points", xytext=(6, 3),
                            fontsize=7.5, color=HW_COLOR[hw])

    # legend: HW patches + marker legend
    hw_patches = [mpatches.Patch(color=HW_COLOR[hw], label=hw) for hw in HW_ORDER]
    ddp_marker = plt.Line2D([0], [0], marker="o", color="gray",
                             markerfacecolor="gray", markersize=7, label="DDP", linestyle="")
    fsdp_marker = plt.Line2D([0], [0], marker="^", color="gray",
                              markerfacecolor="gray", markersize=7, label="FSDP", linestyle="")
    ax.legend(handles=hw_patches + [ddp_marker, fsdp_marker],
              loc="upper left", fontsize=8)

    ax.set_xlabel("Peak VRAM / rank  (GiB)")
    ax.set_ylabel("Aggregate TPS  (8-GPU node)")
    ax.set_title("Training Efficiency: VRAM vs TPS  — FSDP efficiency frontier")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig6_vram_vs_tps.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — FP8 KV cache optimization effect (B200 + H100 comparison)
# ════════════════════════════════════════════════════════════════════════════
def fig7_kvs_effect():
    # seqs (KV cache budget) -> TPS for base+gamma=2 and LoRA+gamma=2
    b200_base = [(64, 32198), (256, 46475), (512, 52759), (1024, 55066)]
    b200_lora = [(64, 21880), (256, 32891), (512, 40734), (1024, 40783)]
    h100_base = [(64, 18235), (256, 22062), (512, 21484)]   # kvs512 < kvs256 on H100
    h100_lora = [(64, 13941), (256, 17764)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    def plot_series(ax, data_pairs, hw_color, hw_label, linestyle):
        xs, ys = zip(*data_pairs)
        ax.plot(range(len(xs)), ys, color=hw_color, linestyle=linestyle,
                marker="o", lw=2, label=hw_label)
        ax.scatter(range(len(xs)), ys, color=hw_color, marker="o", s=50, zorder=5)
        for i, (_, y) in enumerate(data_pairs):
            ax.annotate(f"{y/1000:.0f}k", (i, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color=hw_color)

    seqs_labels_b200 = ["baseline\n(seqs=64)", "kvs256", "kvs512", "kvs1024"]
    seqs_labels_h100 = ["baseline\n(seqs=64)", "kvs256", "kvs512"]

    # Left panel: base model (I5)
    ax1.set_title("Base model  (MTP gamma=2)")
    plot_series(ax1, b200_base, HW_COLOR["B200"], "B200", "-")
    # H100 x-positions: reuse B200 scale but only 3 points
    for i, (_, y) in enumerate(h100_base):
        ax1.scatter(i, y, color=HW_COLOR["H100"], marker="^", s=60, zorder=5)
        ax1.annotate(f"{y/1000:.0f}k", (i, y), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=HW_COLOR["H100"])
    ax1.plot(range(len(h100_base)), [v for _, v in h100_base],
             color=HW_COLOR["H100"], linestyle="--", marker="^", lw=2, label="H100")
    ax1.set_xticks(range(len(seqs_labels_b200)))
    ax1.set_xticklabels(seqs_labels_b200, fontsize=8.5)
    ax1.set_ylabel("Aggregate TPS (8-GPU node)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right panel: LoRA model (I6)
    ax2.set_title("LoRA model  (MTP gamma=2)")
    plot_series(ax2, b200_lora, HW_COLOR["B200"], "B200", "-")
    for i, (_, y) in enumerate(h100_lora):
        ax2.scatter(i, y, color=HW_COLOR["H100"], marker="^", s=60, zorder=5)
        ax2.annotate(f"{y/1000:.0f}k", (i, y), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color=HW_COLOR["H100"])
    ax2.plot(range(len(h100_lora)), [v for _, v in h100_lora],
             color=HW_COLOR["H100"], linestyle="--", marker="^", lw=2, label="H100")
    ax2.set_xticks(range(len(seqs_labels_b200)))
    ax2.set_xticklabels(seqs_labels_b200, fontsize=8.5)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("FP8 KV Cache Optimization Effect  (B200 x H100, gamma=2 MTP)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "fig7_kvs_effect.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 8 — Inference cost index per Mtok (all configs, sorted)
# ════════════════════════════════════════════════════════════════════════════
def fig8_inference_cost():
    # GPU-h/Mtok = 8 / (TPS x 3600) x 1e6;  cost_idx = gpu_h x hw_index
    hw_idx = {"A100": 100, "H100": 210, "B200": 389}

    def ci(tps, hw):
        return 8 / (tps * 3600) * 1e6 * hw_idx[hw]

    rows = [
        # (label, hw, TPS, is_lora)
        ("B200 I5-kvs1024\nbase+g=2", "B200", 55066, False),
        ("B200 I5-kvs512\nbase+g=2",  "B200", 52759, False),
        ("B200 I6-kvs512\nLoRA+g=2",  "B200", 40734, True),
        ("B200 I5-kvs256\nbase+g=2",  "B200", 46475, False),
        ("B200 I6-kvs256\nLoRA+g=2",  "B200", 32891, True),
        ("H100 I5-kvs256\nbase+g=2",  "H100", 22062, False),
        ("H100 I6-kvs256\nLoRA+g=2",  "H100", 17764, True),
        ("B200 I5\nbase+g=2",          "B200", 32198, False),
        ("H100 I5\nbase+g=2",          "H100", 18235, False),
        ("H100 I2\nbase+g=1",          "H100", 16665, False),
        ("B200 I2\nbase+g=1",          "B200", 27812, False),
        ("B200 I6\nLoRA+g=2",          "B200", 21880, True),
        ("H100 I6\nLoRA+g=2",          "H100", 13941, True),
        ("H100 I4\nLoRA+g=1",          "H100", 12462, True),
        ("B200 I4\nLoRA+g=1",          "B200", 19018, True),
        ("H100 I1\nbase",              "H100", 11600, False),
        ("B200 I1\nbase",              "B200", 18142, False),
        ("H100 I3\nLoRA",              "H100",  9390, True),
        ("B200 I3\nLoRA",              "B200", 13773, True),
        ("A100 I2\nbase+g=1",          "A100",  2591, False),
        ("A100 I4\nLoRA+g=1",          "A100",  2376, True),
        ("A100 I1\nbase",              "A100",  1774, False),
        ("A100 I3\nLoRA",              "A100",  1497, True),
    ]
    rows.sort(key=lambda r: ci(r[2], r[1]))

    labels   = [r[0] for r in rows]
    costs    = [ci(r[2], r[1]) for r in rows]
    hws      = [r[1] for r in rows]
    is_lora  = [r[3] for r in rows]
    colors   = [HW_COLOR[hw] for hw in hws]
    alphas   = [0.55 if l else 1.0 for l in is_lora]
    hatches  = ["////" if l else "" for l in is_lora]

    fig, ax = plt.subplots(figsize=(8.5, 8))
    bars = ax.barh(range(len(labels)), costs, color=colors,
                   edgecolor="white", linewidth=0.4)
    for bar, alpha, hatch in zip(bars, alphas, hatches):
        bar.set_alpha(alpha)
        bar.set_hatch(hatch)

    for bar, val in zip(bars, costs):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", ha="left", fontsize=7.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Cost Index per Mtok  (A100 I2 = 100; lower = cheaper)")
    ax.set_title("Inference Cost Efficiency — Relative Index per Mtok")
    ax.set_xlim(0, max(costs) * 1.12)

    # reference at A100 I2 baseline
    a100_i2 = ci(2591, "A100")
    ax.axvline(a100_i2, color="#4878CF", linewidth=1.2, linestyle="--", alpha=0.6)
    ax.text(a100_i2 + 0.5, len(labels) - 0.5, f"A100 I2\n({a100_i2:.1f})",
            color="#4878CF", fontsize=8, va="top")

    hw_patches = [mpatches.Patch(color=HW_COLOR[hw], label=hw) for hw in HW_ORDER]
    lora_patch = mpatches.Patch(facecolor="gray", hatch="////", edgecolor="white",
                                alpha=0.55, label="LoRA (hatched)")
    ax.legend(handles=hw_patches + [lora_patch], loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig8_inference_cost.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 9 — TPS vs concurrency sweep (B200 + H100, kvs variants)
# ════════════════════════════════════════════════════════════════════════════
def fig9_concurrency_sweep():
    def get_variant(row):
        rp = str(row.get("result_path", ""))
        for v in ["kvs1024", "kvs512", "kvs256"]:
            if v in rp: return v
        return "baseline"

    inf["variant"] = inf.apply(get_variant, axis=1)

    # Series to plot: (label, hw, cell_id, variant, linestyle, marker, color, alpha)
    series = [
        ("B200 I1 (no MTP)",         "B200", "I1", "baseline", "-",  "o", HW_COLOR["B200"], 0.45),
        ("B200 I5 baseline",          "B200", "I5", "baseline", "-",  "o", HW_COLOR["B200"], 1.0),
        ("B200 I5 kvs256",            "B200", "I5", "kvs256",   "--", "s", HW_COLOR["B200"], 1.0),
        ("B200 I5 kvs512",            "B200", "I5", "kvs512",   ":",  "^", HW_COLOR["B200"], 1.0),
        ("B200 I5 kvs1024",           "B200", "I5", "kvs1024",  "-.", "D", HW_COLOR["B200"], 1.0),
        ("H100 I5 baseline",          "H100", "I5", "baseline", "-",  "o", HW_COLOR["H100"], 1.0),
        ("H100 I5 kvs256",            "H100", "I5", "kvs256",   "--", "s", HW_COLOR["H100"], 1.0),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, use_log in [(ax1, False), (ax2, True)]:
        for label, hw, cell, var, ls, mk, color, alpha in series:
            df = inf[(inf["hardware"]==hw) & (inf["cell_id"]==cell) &
                     (inf["variant"]==var)].copy()
            if df.empty: continue
            df = df.groupby("node_concurrency")["throughput_out_tok_per_s"].max().reset_index()
            df = df.sort_values("node_concurrency")
            xs = df["node_concurrency"].values
            ys = df["throughput_out_tok_per_s"].values
            ax.plot(xs, ys, label=label, color=color, linestyle=ls,
                    marker=mk, lw=2, alpha=alpha, markersize=5)

        ax.set_xlabel("Node Concurrency  (requests x 8 instances)")
        ax.set_ylabel("Aggregate TPS")
        ax.set_title("TPS vs Concurrency — linear" if not use_log else "TPS vs Concurrency — log x")
        if use_log:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    ax1.legend(fontsize=8, loc="upper left")
    fig.suptitle("Inference Concurrency Sweep  (B200 + H100, base model MTP gamma=2)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "fig9_concurrency_sweep.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 10 — LoRA overhead per HW x MTP config
# ════════════════════════════════════════════════════════════════════════════
def fig10_lora_overhead():
    # (HW, base_cell, lora_cell, gamma_label, base_tps, lora_tps)
    data = [
        # no MTP
        ("A100", "I1", "I3", "no MTP",  1774,  1497),
        ("H100", "I1", "I3", "no MTP", 11600,  9390),
        ("B200", "I1", "I3", "no MTP", 18142, 13773),
        # gamma=1
        ("A100", "I2", "I4", "MTP g=1",  2591,  2376),
        ("H100", "I2", "I4", "MTP g=1", 16665, 12462),
        ("B200", "I2", "I4", "MTP g=1", 27812, 19018),
        # gamma=2
        ("H100", "I5", "I6", "MTP g=2", 18235, 13941),
        ("B200", "I5", "I6", "MTP g=2", 32198, 21880),
    ]

    configs = ["no MTP", "MTP g=1", "MTP g=2"]
    hws_with_gamma2 = ["H100", "B200"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)

    for ax_idx, cfg in enumerate(configs):
        ax = axes[ax_idx]
        cfg_data = [(hw, base_tps, lora_tps)
                    for hw, _, _, g, base_tps, lora_tps in data if g == cfg]
        if not cfg_data: ax.set_visible(False); continue

        hw_labels = [r[0] for r in cfg_data]
        base_vals = [r[1] for r in cfg_data]
        lora_vals = [r[2] for r in cfg_data]
        x = np.arange(len(hw_labels))
        w = 0.35

        bars_base = ax.bar(x - w/2, base_vals, w, label="Base",
                           color=[HW_COLOR[hw] for hw in hw_labels],
                           edgecolor="white", linewidth=0.4)
        bars_lora = ax.bar(x + w/2, lora_vals, w, label="LoRA",
                           color=[HW_COLOR[hw] for hw in hw_labels],
                           edgecolor="white", linewidth=0.4, alpha=0.55, hatch="////")

        # LoRA gap annotations
        for i, (base, lora) in enumerate(zip(base_vals, lora_vals)):
            gap_pct = (lora - base) / base * 100
            ax.annotate(f"{gap_pct:+.0f}%",
                        xy=(x[i] + w/2, lora),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8.5,
                        color="red" if gap_pct < 0 else "green", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(hw_labels)
        ax.set_title(cfg)
        ax.set_ylabel("Aggregate TPS" if ax_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # shared legend
    base_patch = mpatches.Patch(facecolor="gray", label="Base model")
    lora_patch = mpatches.Patch(facecolor="gray", alpha=0.55, hatch="////",
                                edgecolor="white", label="LoRA model")
    fig.legend(handles=[base_patch, lora_patch], loc="upper right",
               fontsize=9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("LoRA Serving Overhead vs Base Model  (per HW, per MTP config)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig10_lora_overhead.png")


# ════════════════════════════════════════════════════════════════════════════
# Fig 11 — Training stability heatmap (3 HW x 5 metrics)
# ════════════════════════════════════════════════════════════════════════════
def fig11_stability_heatmap():
    import json as _json

    with open(RESULTS / "stability_metrics.json") as f:
        sm = _json.load(f)

    hws = ["A100", "H100", "B200"]
    metrics = [
        ("loss_spikes_gt_3sigma", "Loss spikes\n(>3 sigma)", False),
        ("grad_norm_anomalies",   "Grad anomalies\n(>100 / NaN)", False),
        ("grad_norm_max",         "Grad norm max", False),
        ("loss_first",            "Loss first\n(post-warmup)", False),
        ("loss_last",             "Loss last\n(step 220)", True),
    ]

    raw = np.array([[sm[hw][m] for m, _, _ in metrics] for hw in hws], dtype=float)
    labels = [[f"{sm[hw][m]:.1f}" if isinstance(sm[hw][m], float) else str(sm[hw][m])
               for m, _, _ in metrics] for hw in hws]

    # Normalise 0->1 (0=good, 1=bad) so greener = better
    norm = np.zeros_like(raw)
    for j, (_, _, lower_is_better_already_noted) in enumerate(metrics):
        col = raw[:, j]
        mn, mx = col.min(), col.max()
        if mx == mn:
            norm[:, j] = 0.5
        elif lower_is_better_already_noted:
            # lower is better -> invert (higher raw value = less red)
            norm[:, j] = 1 - (col - mn) / (mx - mn)
        else:
            norm[:, j] = (col - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_yticks(range(len(hws)))
    ax.set_yticklabels(hws, fontsize=10)

    for i in range(len(hws)):
        for j in range(len(metrics)):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="white" if norm[i, j] > 0.6 else "black")

    ax.set_title("Training Stability Metrics  (red = more concern, green = better)")
    ax.set_xlabel("Metric  (colour relative within column — not absolute)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["best", "mid", "worst"])
    fig.tight_layout()
    save(fig, "fig11_stability_heatmap.png")


if __name__ == "__main__":
    print("Generating benchmark figures...")
    fig1_training_tps()
    fig2_training_cost()
    fig3_inference_tps()
    fig4_mtp_effect()
    fig6_vram_tps()
    fig7_kvs_effect()
    fig8_inference_cost()
    fig9_concurrency_sweep()
    fig10_lora_overhead()
    fig11_stability_heatmap()
    print("Done.")
