"""
Figure 2: Solver Disentanglement Ablation — seaborn barplot.

Data (A1 checkpoint, 24obj val, seed 42):
  Heun    4-step  0.856
  Euler   4-step  0.855
  DPM++   4-step  0.855
  Euler   1-step  0.851
  DPM++   1-step  0.851

Uses sns.barplot() with hue for solver type and hatch pattern for step count.

Run:  python solver_ablation.py
Outputs: solver_ablation.pdf, solver_ablation.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import *

# Build tidy DataFrame
df = pd.DataFrame([
    {"solver": "Heun",  "steps": 4, "mAP": 0.856, "label": "Heun\n4-step"},
    {"solver": "Euler", "steps": 4, "mAP": 0.855, "label": "Euler\n4-step"},
    {"solver": "DPM++", "steps": 4, "mAP": 0.855, "label": "DPM++\n4-step"},
    {"solver": "Euler", "steps": 1, "mAP": 0.851, "label": "Euler\n1-step"},
    {"solver": "DPM++", "steps": 1, "mAP": 0.851, "label": "DPM++\n1-step"},
])

SOLVER_PAL = {"Heun": C_HEUN, "Euler": C_EULER, "DPM++": C_DPMPP}
STEP_HATCH = {4: "//", 1: ".."}


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # Seaborn barplot with solver as hue
    bars = sns.barplot(
        data=df, x="label", y="mAP", hue="solver",
        palette=SOLVER_PAL, edgecolor="black", linewidth=0.7,
        saturation=1, dodge=False, ax=ax, legend=False,
    )

    # Apply hatches by step count (seaborn doesn't support hatch via hue)
    for bar, (_, row) in zip(bars.patches, df.iterrows()):
        bar.set_hatch(STEP_HATCH[row["steps"]])

    # Value labels above bars
    for bar, (_, row) in zip(bars.patches, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                f"{row['mAP']:.3f}", ha="center", va="bottom",
                fontsize=9, weight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("mAP (24obj val)", fontsize=9)
    ax.set_ylim(0.845, 0.866)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=8)

    # Y-axis break marker
    d = 0.00035
    ax.plot([-0.04, -0.01], [0.847 - d, 0.847 + d],
            color="black", lw=0.8, clip_on=False,
            transform=ax.get_yaxis_transform())
    ax.plot([-0.04, -0.01], [0.847 + d * 1.5, 0.847 + d * 3.5],
            color="black", lw=0.8, clip_on=False,
            transform=ax.get_yaxis_transform())

    # Dual legends: solver (colors) + steps (hatches)
    solver_handles = [
        Patch(facecolor=SOLVER_PAL[s], edgecolor="black", label=s)
        for s in ["Heun", "Euler", "DPM++"]
    ]
    step_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=STEP_HATCH[s],
              label=f"{s}-step")
        for s in [4, 1]
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

    # Bottom annotation
    fig.text(
        0.5, 0.035,
        "$\Delta$4-step vs 1-step = 0.004 mAP (6% of RF gain)",
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="0.7", lw=0.5),
    )

    plt.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.28)
    save_fig(fig, "solver_ablation")


if __name__ == "__main__":
    main()
