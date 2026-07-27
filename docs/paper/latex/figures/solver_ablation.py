"""
Figure 2: Solver Disentanglement Ablation - seaborn barplot.

Data (RF+Heun checkpoint, Dataset 2 val, seed 42):
  Heun    4-step  0.856
  Euler   4-step  0.855
  DPM++   4-step  0.855
  Euler   1-step  0.851
  DPM++   1-step  0.851

Uses sns.barplot() with hatching by step count.

Run:  python solver_ablation.py
Outputs: solver_ablation.pdf, solver_ablation.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import *

df = pd.DataFrame([
    {"solver": "Heun", "steps": 4, "mAP": 0.856, "label": "Heun\n4-step"},
    {"solver": "Euler", "steps": 4, "mAP": 0.855, "label": "Euler\n4-step"},
    {"solver": "DPM++", "steps": 4, "mAP": 0.855, "label": "DPM++\n4-step"},
    {"solver": "Euler", "steps": 1, "mAP": 0.851, "label": "Euler\n1-step"},
    {"solver": "DPM++", "steps": 1, "mAP": 0.851, "label": "DPM++\n1-step"},
])

SOLVER_PAL = {"Heun": C_HEUN, "Euler": C_EULER, "DPM++": C_DPMPP}
STEP_HATCH = {4: "//", 1: ".."}


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.2), constrained_layout=True)

    bars = sns.barplot(
        data=df, x="label", y="mAP", hue="solver",
        palette=SOLVER_PAL, edgecolor="black", linewidth=0.7,
        saturation=1, dodge=False, ax=ax, legend=False,
    )

    for bar, (_, row) in zip(bars.patches, df.iterrows()):
        bar.set_hatch(STEP_HATCH[row["steps"]])

    for bar, (_, row) in zip(bars.patches, df.iterrows()):
        y_pos = bar.get_height() + 0.0003
        if row['mAP'] == 0.851:
            y_pos += 0.0012
            va = 'bottom'
        else:
            va = 'bottom'
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{row['mAP']:.3f}", ha="center", va=va,
                fontsize=9, weight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("mAP (Dataset 2 val)", fontsize=11)
    ax.set_ylim(0.848, 0.860)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.002))
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)

    d = 0.00015
    ax.plot([-0.04, -0.01], [0.848 - d, 0.848 + d],
            color="black", lw=0.8, clip_on=False,
            transform=ax.get_yaxis_transform())
    ax.plot([-0.04, -0.01], [0.848 + d * 2.5, 0.848 + d * 4],
            color="black", lw=0.8, clip_on=False,
            transform=ax.get_yaxis_transform())

    solver_handles = [
        Patch(facecolor=SOLVER_PAL[s], edgecolor="black", label=s)
        for s in ["Heun", "Euler", "DPM++"]
    ]
    step_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=STEP_HATCH[s],
              label=f"{s}-step")
        for s in [4, 1]
    ]

    all_handles = solver_handles + step_handles
    all_labels = [h.get_label() for h in all_handles]

    ax.legend(handles=all_handles, labels=all_labels,
              loc="upper center", ncol=5, fontsize=9,
              frameon=True, framealpha=0.95,
              columnspacing=1.2, handletextpad=0.4,
              title="Colors = Solver  |  Hatch = Steps", title_fontsize=8)

    save_fig(fig, "solver_ablation")


if __name__ == "__main__":
    main()
