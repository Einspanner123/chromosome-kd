"""
Figure 7: Speed-Accuracy Trade-off (FPS vs mAP).

Scatter plot with log-scale FPS axis. Color/marker encode method category
(ours-Heun / ours-DPM++ / standard detector / diffusion baseline), following
the paper's "color = semantic category" convention shared with
solver_ablation.py and per_class_ap.py.

Data sources (RTX A6000, 512x512, batch=1):
  - Ours (A1/A2/A3/TopK), Cascade, YOLOX, DiffusionDet, RTMDet-L:
    results/benchmark_fps_*.md (verified measurements).
  - DINO-R50: mAP = 0.868 (from tab:sota); FPS = 6.5 is an UNVERIFIED
    placeholder (DINO was never benchmarked) -- plotted as a hollow marker
    with a dagger, to be replaced once the GPU benchmark is run.

Run:  python fps_map.py
Outputs: fps_map.pdf, fps_map.png
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from figure_style import *

# =====================================================================
# 4-category semantic palette (color = category, marker = category)
#   - Ours-Heun   : blue   (C_HEUN,  matches solver_ablation.py)
#   - Ours-DPM++  : green  (C_DPMPP)
#   - Standard    : gray   (C_MIDGRAY)  -- DINO/RTMDet/Cascade/YOLOX
#   - Diffusion   : red    (C_DDPM)     -- DiffusionDet baseline
# =====================================================================
C_OURS_HEUN  = C_HEUN
C_OURS_DPMPP = C_DPMPP
C_STANDARD   = C_MIDGRAY
C_DIFFBASE   = C_DDPM

GROUP_STYLE = {
    "ours_heun":  {"color": C_OURS_HEUN,  "marker": "o", "z": 6},
    "ours_dpmpp": {"color": C_OURS_DPMPP, "marker": "s", "z": 6},
    "standard":   {"color": C_STANDARD,   "marker": "^", "z": 5},
    "diffbase":   {"color": C_DIFFBASE,   "marker": "D", "z": 5},
}

# (name, fps, mAP, group, annotate?)
# annotate=True -> label the point on the plot (key comparison points only;
# A1/A2/TopK variants are identified by the legend rather than labelled).
DATA = [
    ("A1 Heun",        8.0,  0.856, "ours_heun",  False),
    ("A2 +StochOT",    7.8,  0.858, "ours_heun",  False),
    ("A3 DPM++",      13.3,  0.863, "ours_dpmpp", True),
    ("A3 +IO3 K300",  14.0,  0.861, "ours_dpmpp", False),
    ("A3 +IO3 K200",  14.2,  0.860, "ours_dpmpp", False),
    ("A3 +IO3 K100",  14.3,  0.850, "ours_dpmpp", False),
    ("Cascade R-CNN", 48.4,  0.854, "standard",   True),
    ("YOLOX-S",       98.5,  0.796, "standard",   True),
    ("DiffusionDet",  41.0,  0.803, "diffbase",   True),
    ("RTMDet-L",      30.3,  0.863, "standard",   True),
    # † DINO-R50 FPS = 6.5 is UNVERIFIED (DINO never benchmarked); hollow
    #   marker flags it. Replace with the measured value after GPU benchmark.
    ("DINO-R50",       6.5,  0.868, "standard",   True),
]

# Label offsets in display points (xytext with textcoords='offset points').
LABEL_OFFSET = {
    "DINO-R50":      (12, 6),
    "A3 DPM++":      (-8, 12),
    "RTMDet-L":      (10, 8),
    "DiffusionDet":  (12, -14),
    "Cascade R-CNN": (12, 2),
    "YOLOX-S":       (-12, -10),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)

    for name, fps, mAP, group, annotate in DATA:
        s = GROUP_STYLE[group]
        # DINO-R50: hollow marker to flag its unverified FPS.
        if name == "DINO-R50":
            ax.scatter(fps, mAP, s=150, facecolor="white",
                       edgecolor=s["color"], marker=s["marker"],
                       linewidths=1.6, zorder=s["z"])
        else:
            ax.scatter(fps, mAP, s=150, color=s["color"], marker=s["marker"],
                       edgecolor="black", lw=0.8, zorder=s["z"])

        if not annotate:
            continue
        label = f"{name}\u2020" if name == "DINO-R50" else name
        dx, dy = LABEL_OFFSET[name]
        ax.annotate(label, xy=(fps, mAP), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=9, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=s["color"], lw=0.8, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    zorder=10)

    # --- log-scale FPS axis (spreads the 6-15 FPS cluster; no inset needed) ---
    ax.set_xscale("log")
    ax.set_xlim(5, 110)
    ax.set_xticks([10, 20, 50, 100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.set_xlabel("FPS (RTX A6000, 512x512, batch=1)", fontsize=12)
    ax.set_ylabel("mAP (24obj)", fontsize=12)
    ax.set_ylim(0.780, 0.875)
    ax.tick_params(labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5, which="both")

    # --- grouped legend (4 categories, not one entry per method) ---
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS_HEUN,
               markeredgecolor="black", markersize=10, label="Ours (Heun)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS_DPMPP,
               markeredgecolor="black", markersize=10, label="Ours (DPM++)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_STANDARD,
               markeredgecolor="black", markersize=10, label="Standard detectors"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_DIFFBASE,
               markeredgecolor="black", markersize=10, label="Diffusion baseline"),
    ]
    ax.legend(handles=handles, loc="center left", fontsize=10,
              frameon=True, framealpha=0.95,
              bbox_to_anchor=(1.02, 0.5),
              title="\u2020 DINO-R50 FPS unverified (pending benchmark)",
              title_fontsize=8)

    save_fig(fig, "fps_map")


if __name__ == "__main__":
    main()
