"""
Figure 2: Solver Disentanglement Ablation (bar chart).

Bars colored by solver type, hatched by step count.
Annotation: 94% gain from RF paradigm vs 6% from solver/step.

Data (A1 checkpoint, 24obj val, seed 42):
  Heun    4-step  0.856
  Euler   4-step  0.855
  DPM++   4-step  0.855
  Euler   1-step  0.851
  DPM++   1-step  0.851

Run:  python solver_ablation.py
Outputs: solver_ablation.pdf, solver_ablation.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import *

# Data
CONFIGS = [
    ("Heun",  4, 0.856),
    ("Euler", 4, 0.855),
    ("DPM++", 4, 0.855),
    ("Euler", 1, 0.851),
    ("DPM++", 1, 0.851),
]
SOLVER_COLORS = {"Heun": C_HEUN, "Euler": C_EULER, "DPM++": C_DPMPP}


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # Short labels: no embedded newline
    names = [f"{s} {st}-step" for s, st, _ in CONFIGS]
    maps = [m for _, _, m in CONFIGS]
    colors = [SOLVER_COLORS[s] for s, _, _ in CONFIGS]
    hatches = ["//", "//", "//", "..", ".."]

    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, maps, width=0.55, color=colors, edgecolor="black", lw=0.7)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    # Value labels above bars
    for xi, m in zip(x, maps):
        ax.text(xi, m + 0.0008, f"{m:.3f}", ha="center", va="bottom",
                fontsize=9, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("mAP (24obj val)", fontsize=9)
    ax.set_ylim(0.845, 0.866)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=8)

    # Subtle y-axis break marker
    d = 0.00035
    ax.plot([-0.04, -0.01], [0.847 - d, 0.847 + d],
            color="black", lw=0.8, clip_on=False,
            transform=ax.get_yaxis_transform())
    ax.plot([-0.04, -0.01], [0.847 + d*1.5, 0.847 + d*3.5],
            color="black", lw=0.8, clip_on=False,
            transform=ax.get_yaxis_transform())

    # Legends: solver (left) + steps (right), below axis
    solver_handles = [
        Patch(facecolor=C_HEUN,  edgecolor="black", label="Heun"),
        Patch(facecolor=C_EULER, edgecolor="black", label="Euler"),
        Patch(facecolor=C_DPMPP, edgecolor="black", label="DPM++"),
    ]
    step_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="4-step"),
        Patch(facecolor="white", edgecolor="black", hatch="..",  label="1-step"),
    ]

    leg_solver = ax.legend(
        handles=solver_handles, loc="upper left",
        bbox_to_anchor=(0.0, -0.20),
        fontsize=8, frameon=False, ncol=3,
        handletextpad=0.4, handlelength=1.2,
        columnspacing=1.0, borderpad=0.3,
        title="Solver", title_fontsize=8,
    )
    ax.add_artist(leg_solver)

    ax.legend(
        handles=step_handles, loc="upper left",
        bbox_to_anchor=(0.55, -0.20),
        fontsize=8, frameon=False, ncol=2,
        handletextpad=0.4, handlelength=1.2,
        columnspacing=0.8, borderpad=0.3,
        title="Steps", title_fontsize=8,
    )

    # Bottom annotation — simpler
    fig.text(
        0.5, 0.035,
        "$\\Delta$4-step vs 1-step = 0.004 mAP (6% of RF gain)",
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="0.7", lw=0.5),
    )

    plt.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.28)
    save_fig(fig, "solver_ablation")


if __name__ == "__main__":
    main()
