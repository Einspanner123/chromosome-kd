"""
Figure 6: Speed-Accuracy Trade-off (FPS vs mAP).

Scatter plot with log-scale FPS axis. Color/marker encode method category
(ours-Heun / ours-DPM-Solver++ / standard detector / diffusion baseline), following
the paper's "color = semantic category" convention shared with
solver_ablation.py and per_class_ap.py.

Method set = Table 10 (10 models): RF+Heun / +Stoch. Coupling / DPM-Solver++ /
DPM-Solver++ +Top-K x3 / DPM-Solver++ (H=3 Distill) + Cascade R-CNN + YOLOX-S +
DiffusionDet. RTMDet-L and DINO-R50 are compared on accuracy only in tab:sota
(no FPS column) and are intentionally excluded from the speed story here;
they will be added after their FPS is benchmarked.

Data sources (RTX A6000, 512x512, batch=1):
  - 9 measured points: results/benchmark_fps_*.md (verified measurements).
  - H=3 Distill (~22 FPS): estimated from head computation ratio (H=3 vs H=6
    measured 0.57x) applied to DPM-Solver++ 75.03 ms; full measurement pending.
  - DiffusionDet mAP = 0.803 (tab:fps / tab:sota authoritative value).

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
#   - Standard    : gray   (C_MIDGRAY)  -- Cascade / YOLOX
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

# (name, fps, mAP, group, annotate?)
# annotate=True -> label the point on the plot (key comparison points only;
# RF+Heun / +Stoch. Coupling / Top-K K=300/K=100 are identified by the legend
# rather than labelled). Method names follow the cumulative-ablation
# convention of tab:main-ablation and tab:fps (no internal A0-A4 codenames).
DATA = [
    ("RF+Heun",                   8.0,  0.856, "ours_heun",  False),
    ("+Stoch. Coupling",          7.8,  0.858, "ours_heun",  False),
    ("DPM-Solver++",             13.3,  0.863, "ours_dpmpp", True),
    ("DPM-Solver++ +Top-K (K=300)", 14.0,  0.861, "ours_dpmpp", False),
    ("DPM-Solver++ +Top-K (K=200)", 14.2,  0.860, "ours_dpmpp", False),
    ("DPM-Solver++ +Top-K (K=100)", 14.3,  0.850, "ours_dpmpp", False),
    ("DPM-Solver++ (H=3 Distill)",  22.0,  0.860, "ours_distill", True),   # best speed-accuracy trade-off
    ("Cascade R-CNN",   48.4,  0.854, "standard",   True),
    ("YOLOX-S",         98.5,  0.796, "standard",   True),
    ("DiffusionDet",    41.0,  0.803, "diffbase",   True),
]

# Label offsets in display points (xytext with textcoords='offset points').
LABEL_OFFSET = {
    "DPM-Solver++":                  (-8, 12),
    "DPM-Solver++ (H=3 Distill)":    (10, 10),
    "DiffusionDet":                  (12, -14),
    "Cascade R-CNN":                 (12, 2),
    "YOLOX-S":                       (-12, -10),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)

    for name, fps, mAP, group, annotate in DATA:
        s = GROUP_STYLE[group]
        ax.scatter(fps, mAP, s=s["size"], color=s["color"], marker=s["marker"],
                   edgecolor="black", lw=0.8, zorder=s["z"])

        if not annotate:
            continue
        dx, dy = LABEL_OFFSET[name]
        ax.annotate(name, xy=(fps, mAP), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=s["color"], lw=0.8, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    zorder=10)

    # --- log-scale FPS axis (spreads the 6-15 FPS cluster; no inset needed) ---
    ax.set_xscale("log")
    ax.set_xlim(6, 110)
    ax.set_xticks([10, 20, 50, 100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel("FPS (RTX A6000, 512x512, batch=1)", fontsize=12)
    ax.set_ylabel("mAP (Dataset 2)", fontsize=12)
    ax.set_ylim(0.780, 0.875)
    ax.tick_params(labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5, which="both")

    # --- grouped legend (4 categories, not one entry per method) ---
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS_HEUN,
               markeredgecolor="black", markersize=10, label="Ours (Heun)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS_DPMPP,
               markeredgecolor="black", markersize=10, label="Ours (DPM-Solver++)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_OURS_DISTILL,
               markeredgecolor="black", markersize=14, label="Ours (H=3 Distill)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_STANDARD,
               markeredgecolor="black", markersize=10, label="Standard detectors"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_DIFFBASE,
               markeredgecolor="black", markersize=10, label="Diffusion baseline"),
    ]
    ax.legend(handles=handles, loc="center left", fontsize=10,
              frameon=True, framealpha=0.95,
              bbox_to_anchor=(1.02, 0.5))

    save_fig(fig, "fps_map")


if __name__ == "__main__":
    main()
