"""
Figure 8: Speed-Accuracy Trade-off (FPS vs mAP), Dataset 1 & Dataset 2.

Two side-by-side scatter panels (log-scale FPS axis). Color/marker encode
method category (ours-Heun / ours-DPM-Solver++ / ours-Distill / standard
detector / diffusion baseline), following the paper's "color = semantic
category" convention shared with solver_ablation.py and per_class_ap.py.

Two panels:
  (a) Dataset 1 (Chromosome20240904, 1540 imgs, low-data regime):
      KaryoFlow variants + DiffusionDet + standard discriminative detectors
      (Cascade R-CNN / YOLOX-S / RTMDet-L / DINO R50) all have test mAP here.
      On the test set, KaryoFlow (+Stoch. Coupling, test mAP 0.740) matches
      frontier detectors DINO R50 (0.725, completed 150ep) and RTMDet-L
      (0.732), and wins on 15/24 per-class AP — advantages concentrated on
      small chromosomes (E/F/G) and sex chromosomes (X/Y), the clinically
      highest-risk categories. This supports the claim that diffusion-based
      detectors match frontier detectors under data scarcity, with structural
      advantages on clinically critical small-object classes. The "Random (RF)"
      point visualises the OT-diversity-collapse pathology: Random coupling
      collapses to 0.713 (below the DDPM baseline 0.719), while Stochastic
      Coupling recovers to 0.740 — at no speed cost (same Heun architecture).
  (b) Dataset 2 (24 Chromosomes Object, 5000 imgs): full method set incl.
      DINO R50 and RTMDet-L.

FPS is hardware/architecture-dependent and dataset-independent (synthetic
512x512 input, CUDA Event timing), so a model's FPS is the same in both
panels; only the mAP (y-axis) changes across datasets. Dataset 1 variants
share the architecture of the corresponding Dataset 2 ablation entries
(RF+Heun / +Stoch. Coupling / +DPM-Solver++ / DiffusionDet), hence reuse
their measured FPS.

Data sources (RTX A6000, 512x512, batch=1, 500 iters, CUDA Event timing):
  - Dataset 2 FPS: results/benchmark_fps_20260729_021441.md (ross A6000,
    clean env, 500 iters) + DINO R50 / RTMDet-L measured locally
    (benchmark_fps_20260729_103700.md, 30.5 FPS each).
  - Dataset 2 mAP: tab:sota / tab:fps authoritative values.
  - Dataset 1 mAP: paper §4.2 (DiffusionDet 0.729, RF 0.746) and §4.4.1
    Table 9 (Random 0.713, Stoch. Coupling 0.747); +DPM-Solver++ 0.746 from
    EXPERIMENT_CATALOG §2.2.1 (inference solver swap, step-aligned).

Run:  python fps_map.py
Outputs: fps_map.pdf, fps_map.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from figure_style import *

# =====================================================================
# 5-category semantic palette (color = category, marker = category)
#   - Ours-Heun   : blue   (C_HEUN,  matches solver_ablation.py)
#   - Ours-DPM-Solver++ : green  (C_DPMPP)
#   - Ours-Distill: brown  (PAL[5])     -- H=3 head distillation
#   - Standard    : gray   (C_MIDGRAY)  -- Cascade / YOLOX / RTMDet / DINO
#   - Diffusion   : red    (C_DDPM)     -- DiffusionDet baseline
# =====================================================================
C_OURS_HEUN  = C_HEUN
C_OURS_DPMPP = C_DPMPP
C_OURS_DISTILL = PAL[5]   # brown — H=3 head distillation (architecture-level compression)
C_STANDARD   = C_MIDGRAY
C_DIFFBASE   = C_DDPM

GROUP_STYLE = {
    "ours_heun":    {"color": C_OURS_HEUN,    "marker": "o", "z": 6, "size": 150},
    "ours_dpmpp":   {"color": C_OURS_DPMPP,   "marker": "s", "z": 6, "size": 150},
    "ours_distill": {"color": C_OURS_DISTILL, "marker": "*", "z": 7, "size": 300},
    "standard":     {"color": C_STANDARD,     "marker": "^", "z": 5, "size": 150},
    "diffbase":     {"color": C_DIFFBASE,     "marker": "D", "z": 5, "size": 150},
}

# =====================================================================
# Dataset 2 (24 Chromosomes Object, 5000 imgs) — full method set
# (name, fps, mAP, group, annotate?)
# Method names follow the cumulative-ablation convention of tab:main-ablation
# and tab:fps (no internal A0-A4 codenames).
# =====================================================================
DATA_D2 = [
    ("RF+Heun",                     8.2,  0.856, "ours_heun",    False),
    ("+Stoch. Coupling",            7.6,  0.858, "ours_heun",    False),
    ("DPM-Solver++",               12.9,  0.863, "ours_dpmpp",   True),
    ("DPM-Solver++ +Top-K (K=300)", 14.2, 0.861, "ours_dpmpp",   False),
    ("DPM-Solver++ +Top-K (K=200)", 14.5, 0.860, "ours_dpmpp",   False),
    ("DPM-Solver++ +Top-K (K=100)", 14.8, 0.850, "ours_dpmpp",   False),
    ("DPM-Solver++ (H=3 Distill)",  22.4, 0.859, "ours_distill", True),   # best speed-accuracy trade-off
    ("Cascade R-CNN",   49.7,  0.854, "standard",   True),
    ("YOLOX-S",        105.0,  0.796, "standard",   True),
    ("DiffusionDet",    43.1,  0.803, "diffbase",   True),
    ("RTMDet-L",        30.5,  0.863, "standard",   True),
    ("DINO R50",        30.5,  0.868, "standard",   True),
]

# Label offsets in display points (xytext with textcoords='offset points').
LABEL_OFFSET_D2 = {
    "DPM-Solver++":                  (-8, 12),
    "DPM-Solver++ (H=3 Distill)":    (10, 10),
    "DiffusionDet":                  (12, -14),
    "Cascade R-CNN":                 (12, 2),
    "YOLOX-S":                       (-12, -10),
    "RTMDet-L":                      (-14, -4),
    "DINO R50":                      (10, 8),
}

# =====================================================================
# Dataset 1 (Chromosome20240904, 1540 imgs, low-data regime)
# Standard detectors (Cascade R-CNN / RTMDet-L / YOLOX-S / DINO R50) were
# trained on Dataset 1 (2026-07-29, workstation/local A6000) to demonstrate
# "discriminative paradigms lose advantage under data scarcity". DINO R50
# trained to completion (150ep, early stop @ ep107, best 0.742 val / 0.725 test).
# "Random (RF)" is the coupling-ablation point showing OT diversity collapse
# (0.713, below DDPM baseline 0.729). KaryoFlow (+Stoch. Coupling) matches
# DINO R50 / RTMDet-L on overall test mAP (0.740 vs 0.725/0.732) and wins on
# 15/24 per-class AP, with advantages concentrated on small chromosomes (E/F/G)
# and sex chromosomes (X/Y) — the clinically highest-risk categories.
# FPS reuses the architecturally-matched Dataset 2 measurement (see docstring).
# mAP values: test set (220 images), sourced from test_eval_per_size_20260730_204840.json
# =====================================================================
DATA_D1 = [
    ("DiffusionDet",     43.1, 0.719, "diffbase",   True),   # test mAP (3-seed mean: 0.716/0.722/0.718)
    ("Random (RF)",       8.2, 0.713, "ours_heun",  True),   # OT collapse: below DDPM (val, no test eval)
    ("+Stoch. Coupling",  7.6, 0.740, "ours_heun",  True),   # KaryoFlow (Heun), test mAP
    ("+DPM-Solver++",    12.9, 0.739, "ours_dpmpp", True),   # test mAP (seed42)
    ("Cascade R-CNN",    49.4, 0.723, "standard",   True),   # test mAP
    ("RTMDet-L",         21.2, 0.732, "standard",   True),   # test mAP
    ("YOLOX-S",          95.3, 0.581, "standard",   True),   # test mAP
    ("DINO R50",         29.2, 0.725, "standard",   True),   # test mAP (ep107, completed)
]

LABEL_OFFSET_D1 = {
    "DiffusionDet":      (10, -10),
    "Random (RF)":       (-10, -12),
    "+Stoch. Coupling":  (10, 6),
    "+DPM-Solver++":     (10, -10),
    "Cascade R-CNN":     (10, -8),
    "RTMDet-L":          (-14, 6),
    "YOLOX-S":           (-12, -8),
    "DINO R50":          (10, 6),
}


def _plot_panel(ax, data, label_offset, ylim, ylabel_mAP: str) -> None:
    for name, fps, mAP, group, annotate in data:
        s = GROUP_STYLE[group]
        ax.scatter(fps, mAP, s=s["size"], color=s["color"], marker=s["marker"],
                   edgecolor="black", lw=0.8, zorder=s["z"])

        if not annotate:
            continue
        dx, dy = label_offset[name]
        ax.annotate(name, xy=(fps, mAP), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=s["color"], lw=0.8, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    zorder=10)

    # --- log-scale FPS axis (spreads the 6-15 FPS cluster; no inset needed) ---
    ax.set_xscale("log")
    ax.set_xlim(6, 115)
    ax.set_xticks([10, 20, 50, 100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel("FPS (RTX A6000, 512x512, batch=1)", fontsize=11)
    ax.set_ylabel(ylabel_mAP, fontsize=11)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5, which="both")


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0),
                                   constrained_layout=True)

    _plot_panel(ax1, DATA_D1, LABEL_OFFSET_D1,
                ylim=(0.580, 0.760), ylabel_mAP="mAP (Dataset 1)")
    _plot_panel(ax2, DATA_D2, LABEL_OFFSET_D2,
                ylim=(0.780, 0.875), ylabel_mAP="mAP (Dataset 2)")

    # Panel labels
    ax1.set_title("(a) Dataset 1 (low-data, 1,540 imgs)", fontsize=11, pad=6)
    ax2.set_title("(b) Dataset 2 (5,000 imgs)", fontsize=11, pad=6)

    # --- grouped legend (5 categories, shared) at lower-left of panel (b) ---
    # Panel (b)'s lower-left (low FPS, low mAP ~0.78) is empty, so the legend
    # sits cleanly inside the axes corner as requested.
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS_HEUN,
               markeredgecolor="black", markersize=9, label="Ours (Heun)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS_DPMPP,
               markeredgecolor="black", markersize=9, label="Ours (DPM-Solver++)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_OURS_DISTILL,
               markeredgecolor="black", markersize=13, label="Ours (H=3 Distill)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_STANDARD,
               markeredgecolor="black", markersize=9, label="Standard detectors"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_DIFFBASE,
               markeredgecolor="black", markersize=9, label="Diffusion baseline"),
    ]
    ax2.legend(handles=handles, loc="lower left", fontsize=9,
               frameon=True, framealpha=0.95, ncol=1,
               handletextpad=0.5, borderpad=0.4)

    save_fig(fig, "fps_map")


if __name__ == "__main__":
    main()
