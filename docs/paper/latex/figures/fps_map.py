"""
Figure 7: Speed-Accuracy Trade-off (FPS vs mAP).

Clean scatter plot with clear labels and no overlap.
- Our RF variants: circles (Heun-based) / squares (DPM++ variants)
- Baselines: triangles (paper-reported FPS)
- RTMDet-L: distinct color (teal)

All FPS from RTX A6000, 512x512, batch=1, 200 images.

Run:  python fps_map.py
Outputs: fps_map.pdf, fps_map.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_style import *

C_OURS_BASE = C_EULER
C_OURS_FAST = C_DPMPP
C_BASELINE  = C_DDPM
C_RTM       = "#196f7b"

DATA = [
    ("A1 Heun",       8.0,   0.856, "base"),
    ("A2 +StochOT",   7.8,   0.858, "base"),
    ("A3 DPM++",     13.3,   0.863, "fast"),
    ("A3 +IO3 K300",  14.0,   0.861, "fast"),
    ("A3 +IO3 K200",  14.2,   0.860, "fast"),
    ("A3 +IO3 K100",  14.3,   0.850, "fast"),
    ("Cascade R-CNN", 48.4,  0.854, "baseline"),
    ("YOLOX-S",       98.5,   0.796, "baseline"),
    ("DiffusionDet",  41.0,   0.787, "baseline"),
    ("RTMDet-L",      12.1,   0.863, "rtmdet"),
]

GROUP_STYLE = {
    "base":     {"color": C_OURS_BASE, "marker": "o", "z": 6},
    "fast":     {"color": C_OURS_FAST, "marker": "s", "z": 6},
    "baseline": {"color": C_BASELINE,  "marker": "^", "z": 5},
    "rtmdet":   {"color": C_RTM,       "marker": "D", "z": 5},
}

LABEL_CONFIG = {
    "A1 Heun":       {"display": "A1",       "xy": (8.0, 0.856),  "xytext": (1.5, 0.850),  "ha": "right", "color": C_OURS_BASE},
    "A2 +StochOT":   {"display": "A2",       "xy": (7.8, 0.858),  "xytext": (14.5, 0.840), "ha": "left", "color": C_OURS_BASE},
    "A3 DPM++":      {"display": "A3",       "xy": (13.3, 0.863), "xytext": (6.5, 0.873),  "ha": "right", "color": C_OURS_FAST},
    "A3 +IO3 K300":  {"display": "K300",     "xy": (14.0, 0.861), "xytext": (19.5, 0.868), "ha": "left", "color": C_OURS_FAST},
    "A3 +IO3 K200":  {"display": "K200",     "xy": (14.2, 0.860), "xytext": (19.5, 0.856), "ha": "left", "color": C_OURS_FAST},
    "A3 +IO3 K100":  {"display": "K100",     "xy": (14.3, 0.850), "xytext": (19.5, 0.843), "ha": "left", "color": C_OURS_FAST},
    "Cascade R-CNN": {"display": "Cascade",  "xy": (48.4, 0.854), "xytext": (60.0, 0.860), "ha": "left", "color": C_BASELINE},
    "YOLOX-S":       {"display": "YOLOX-S",  "xy": (98.5, 0.796), "xytext": (82.0, 0.798), "ha": "right", "color": C_BASELINE},
    "DiffusionDet":  {"display": "DiffDet",  "xy": (41.0, 0.787), "xytext": (28.0, 0.780), "ha": "right", "color": C_BASELINE},
    "RTMDet-L":      {"display": "RTMDet-L", "xy": (12.1, 0.863), "xytext": (5.5, 0.863),  "ha": "right", "color": C_RTM},
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)

    for name, fps, mAP, group in DATA:
        s = GROUP_STYLE[group]
        ax.scatter(fps, mAP, s=100, color=s["color"], marker=s["marker"],
                   edgecolor="black", lw=0.8, zorder=s["z"])

    for name in LABEL_CONFIG:
        cfg = LABEL_CONFIG[name]
        ax.annotate(
            cfg["display"],
            xy=cfg["xy"],
            xytext=cfg["xytext"],
            fontsize=8,
            ha=cfg["ha"],
            va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cfg["color"], lw=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6),
            zorder=7,
        )

    ax.set_xlabel("FPS (RTX A6000, 512x512)", fontsize=10)
    ax.set_ylabel("mAP (24obj)", fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_ylim(0.775, 0.875)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS_BASE,
               markeredgecolor="black", markersize=9, label="Ours (Heun)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS_FAST,
               markeredgecolor="black", markersize=9, label="Ours (DPM++)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_BASELINE,
               markeredgecolor="black", markersize=9, label="Baseline (paper)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_RTM,
               markeredgecolor="black", markersize=9, label="RTMDet-L"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              frameon=True, framealpha=0.95)

    save_fig(fig, "fps_map")


if __name__ == "__main__":
    main()
