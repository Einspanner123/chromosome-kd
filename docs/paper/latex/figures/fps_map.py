"""
Figure 7: Speed-Accuracy Trade-off (FPS vs mAP).

Scatter plot with large zoomed inset for crowded top-left area.
Main plot shows full range, inset shows crowded region with labels.

All FPS from RTX A6000, 512x512, batch=1, 200 images.

Run:  python fps_map.py
Outputs: fps_map.pdf, fps_map.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from figure_style import *

C_OURS_BASE = C_EULER
C_OURS_FAST = C_DPMPP
C_BASELINE = C_DDPM
C_RTM = "#196f7b"
C_DINO = "#8E44AD"
C_INSET = "#7F8C8D"

DATA = [
    ("A1 Heun",       8.0,   0.856, "base"),
    ("A2 +StochOT",   7.8,   0.858, "base"),
    ("A3 DPM++",     13.3,   0.863, "fast"),
    ("A3 +IO3 K300",  14.0,   0.861, "fast"),
    ("A3 +IO3 K200",  14.2,   0.860, "fast"),
    ("A3 +IO3 K100",  14.3,   0.850, "fast"),
    ("Cascade R-CNN", 48.4,  0.854, "baseline"),
    ("YOLOX-S",       98.5,  0.796, "baseline"),
    ("DiffusionDet",  41.0,  0.787, "baseline"),
    ("RTMDet-L",      12.1,  0.863, "rtmdet"),
    ("DINO-R50",       6.5,  0.869, "dino"),
]

CROWDED_MODELS = {"DINO-R50", "RTMDet-L", "A1 Heun", "A2 +StochOT",
                  "A3 DPM++", "A3 +IO3 K300", "A3 +IO3 K200", "A3 +IO3 K100"}

GROUP_STYLE = {
    "base": {"color": C_OURS_BASE, "marker": "o", "z": 6},
    "fast": {"color": C_OURS_FAST, "marker": "s", "z": 6},
    "baseline": {"color": C_BASELINE, "marker": "^", "z": 5},
    "rtmdet": {"color": C_RTM, "marker": "D", "z": 5},
    "dino": {"color": C_DINO, "marker": "p", "z": 5},
}

INSET_LABELS = {
    "DINO-R50": {"display": "DINO-R50", "xytext": (-35, 35), "rad": 0.3},
    "RTMDet-L": {"display": "RTMDet-L", "xytext": (20, -25), "rad": -0.2},
    "A1 Heun": {"display": "A1", "xytext": (-40, -15), "rad": 0.2},
    "A2 +StochOT": {"display": "A2", "xytext": (-30, 40), "rad": -0.3},
    "A3 DPM++": {"display": "A3", "xytext": (25, 30), "rad": 0.25},
    "A3 +IO3 K300": {"display": "K300", "xytext": (40, -30), "rad": -0.3},
    "A3 +IO3 K200": {"display": "K200", "xytext": (45, 5), "rad": 0.2},
    "A3 +IO3 K100": {"display": "K100", "xytext": (50, -10), "rad": -0.25},
}


def draw_main_labels(ax):
    for name, fps, mAP, group in DATA:
        if name in CROWDED_MODELS:
            continue

        s = GROUP_STYLE[group]
        color = s["color"]

        if name == "Cascade R-CNN":
            ax.annotate("Cascade", xy=(fps, mAP), xytext=(fps + 6, mAP),
                       fontsize=9, color="black", va='center',
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                       arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
                       zorder=7)
        elif name == "YOLOX-S":
            ax.annotate("YOLOX-S", xy=(fps, mAP), xytext=(fps - 8, mAP - 0.004),
                       fontsize=9, color="black",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                       arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
                       zorder=7)
        elif name == "DiffusionDet":
            ax.annotate("DiffDet", xy=(fps, mAP), xytext=(fps + 2, mAP - 0.004),
                       fontsize=9, color="black",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8),
                       arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
                       zorder=7)


def setup_inset(ax_main):
    inset_ax = ax_main.inset_axes([0.05, 0.45, 0.40, 0.48])

    inset_ax.set_xlim(4, 18)
    inset_ax.set_ylim(0.845, 0.872)

    inset_ax.set_xlabel("FPS", fontsize=9)
    inset_ax.set_ylabel("mAP", fontsize=9)
    inset_ax.tick_params(labelsize=8)
    inset_ax.grid(True, ls=":", lw=0.3, alpha=0.5)

    for name, fps, mAP, group in DATA:
        if name not in CROWDED_MODELS:
            continue
        s = GROUP_STYLE[group]
        inset_ax.scatter(fps, mAP, s=120, color=s["color"], marker=s["marker"],
                        edgecolor="black", lw=0.8, zorder=s["z"])

    for name, config in INSET_LABELS.items():
        model_data = next((d for d in DATA if d[0] == name), None)
        if not model_data:
            continue

        fps, mAP, group = model_data[1], model_data[2], model_data[3]
        color = GROUP_STYLE[group]["color"]
        rad = config.get("rad", 0.2)

        inset_ax.annotate(config["display"], xy=(fps, mAP),
                         xytext=config["xytext"],
                         textcoords='offset points',
                         fontsize=8, color="black",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.8, alpha=0.95),
                         arrowprops=dict(arrowstyle="-", color="gray", lw=0.5,
                                       connectionstyle=f"arc3,rad={rad}"),
                         zorder=10)

    for spine in inset_ax.spines.values():
        spine.set_color(C_INSET)
        spine.set_linewidth(1.2)
        spine.set_linestyle('--')

    return inset_ax


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)

    for name, fps, mAP, group in DATA:
        s = GROUP_STYLE[group]
        ax.scatter(fps, mAP, s=150, color=s["color"], marker=s["marker"],
                  edgecolor="black", lw=0.8, zorder=s["z"])

    draw_main_labels(ax)

    setup_inset(ax)

    ax.set_xlabel("FPS (RTX A6000, 512x512)", fontsize=12)
    ax.set_ylabel("mAP (24obj)", fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_ylim(0.780, 0.875)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS_BASE,
               markeredgecolor="black", markersize=10, label="Ours (Heun)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS_FAST,
               markeredgecolor="black", markersize=10, label="Ours (DPM++)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_BASELINE,
               markeredgecolor="black", markersize=10, label="Baseline (paper)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_RTM,
               markeredgecolor="black", markersize=10, label="RTMDet-L"),
        Line2D([0], [0], marker="p", color="w", markerfacecolor=C_DINO,
               markeredgecolor="black", markersize=10, label="DINO-R50"),
    ]
    ax.legend(handles=handles, loc="center right", fontsize=10,
              frameon=True, framealpha=0.95,
              bbox_to_anchor=(1.02, 0.5))

    save_fig(fig, "fps_map")


if __name__ == "__main__":
    main()
