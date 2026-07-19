"""
Figure 7: Speed-Accuracy Trade-off (FPS vs mAP).

Clean scatter plot — no zoomed inset, no overlapping labels.
- Our RF variants: circles (Heun-based) / squares (DPM++ variants)
- Baselines: triangles (paper-reported FPS from Table 7)
- RTMDet-L FPS from our own benchmark (same hardware/setup)
- Direct text labels with minimal leader lines

All FPS from RTX A6000, 512x512, batch=1, 200 images (paper) or 50 runs (ours).

Run:  python fps_map.py
Outputs: fps_map.pdf, fps_map.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_style import *

# Colors
C_OURS_BASE = C_EULER       # orange — Heun-based
C_OURS_FAST = C_DPMPP       # green  — DPM-Solver++
C_BASELINE  = C_DDPM        # red    — non-RF baselines (paper-reported FPS)
C_RTM       = "#009E73"     # green  — RTMDet-L (our benchmark)

# (name, fps, mAP, group)
DATA = [
    # Our variants
    ("A1 Heun",       8.0,   0.856, "base"),
    ("A2 +StochOT",   7.8,   0.858, "base"),
    ("A3 DPM++",     13.3,   0.863, "fast"),
    ("A3 +IO3 K300",  14.0,   0.861, "fast"),
    ("A3 +IO3 K200",  14.2,   0.860, "fast"),
    ("A3 +IO3 K100",  14.3,   0.850, "fast"),
    # Baselines (paper Table 7)
    ("Cascade R-CNN",  48.4,  0.854, "baseline"),
    ("YOLOX-S",        98.5,  0.796, "baseline"),
    ("DiffusionDet",   41.0,  0.787, "baseline"),
    # Additional (benchmarked)
    ("RTMDet-L",       12.1,  0.863, "rtmdet"),
]

GROUP_STYLE = {
    "base":     {"color": C_OURS_BASE, "marker": "o", "z": 6},
    "fast":     {"color": C_OURS_FAST, "marker": "s", "z": 6},
    "baseline": {"color": C_BASELINE,  "marker": "^", "z": 5},
    "rtmdet":   {"color": C_RTM,       "marker": "D", "z": 5},
}

# Manual label placements to avoid overlap
LABEL_POS = {
    "A1 Heun":       (8.0,   0.856,  -8,  12,  "right"),
    "A2 +StochOT":   (7.8,   0.858, -10, -14,  "right"),
    "A3 DPM++":      (13.3,  0.863,  14, -12,  "left"),
    "A3 +IO3 K300":  (14.0,  0.861,  10,  10,  "left"),
    "A3 +IO3 K200":  (14.2,  0.860,  18, -4,   "left"),
    "A3 +IO3 K100":  (14.3,  0.850, -14,  14,  "right"),
    "Cascade R-CNN": (48.4,  0.854,  10,  14,  "left"),
    "YOLOX-S":       (98.5,  0.796, -10,  10,  "right"),
    "DiffusionDet":  (41.0,  0.787,  10, -16,  "left"),
    "RTMDet-L":      (12.1,  0.863, -12, -14,  "right"),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Draw points
    for name, fps, mAP, group in DATA:
        s = GROUP_STYLE[group]
        ax.scatter(fps, mAP, s=70, color=s["color"], marker=s["marker"],
                   edgecolor="k", lw=0.6, zorder=s["z"])

    # Draw labels with leader lines
    for name, _, _, _ in DATA:
        fps, mAP, dx, dy, ha = LABEL_POS[name]
        # Shorten display name for labels
        display = name
        if name.startswith("A1"): display = "A1"
        elif name.startswith("A2"): display = "A2"
        elif name.startswith("A3") and "IO3" not in name: display = "A3"
        elif "IO3 K300" in name: display = "K300"
        elif "IO3 K200" in name: display = "K200"
        elif "IO3 K100" in name: display = "K100"
        elif "Cascade" in name: display = "Cascade R-CNN"
        elif "YOLOX" in name: display = "YOLOX-S"

        # Leader line
        ax.annotate("", xy=(fps + dx * 0.15, mAP + dy * 0.15),
                    xytext=(fps, mAP),
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
                    zorder=2)
        # Label
        ax.text(fps + dx, mAP + dy, display, fontsize=7.5,
                ha=ha, va="center", color="black", zorder=7)

    # Axes
    ax.set_xlabel("FPS (RTX A6000, 512×512)", fontsize=9)
    ax.set_ylabel("mAP (24obj)", fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_ylim(0.775, 0.875)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5)

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS_BASE,
               markeredgecolor="k", markersize=8, label="Ours (Heun)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS_FAST,
               markeredgecolor="k", markersize=8, label="Ours (DPM++)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_BASELINE,
               markeredgecolor="k", markersize=8, label="Baseline (paper)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_RTM,
               markeredgecolor="k", markersize=8, label="RTMDet-L"),
    ]
    ax.legend(handles=handles, loc="lower right",
              frameon=True, framealpha=0.95, fontsize=7.5)

    save_fig(fig, "fps_map")


if __name__ == "__main__":
    main()
