"""Figure 2: Solver Disentanglement Ablation.

Bar chart comparing 5 configurations (Solver x Steps) on the A1 checkpoint
(RF+Heun trained) on the 24obj validation set (seed 42).

Data (from paper Sec 3.2.3):
  Heun         4-step   0.856
  Euler        4-step   0.855
  DPM++        4-step   0.855
  Euler        1-step   0.851
  DPM++        1-step   0.851

Annotations:
  - Color by solver type.
  - Hatching by step count.
  - Annotation: 94% gain attributable to RF training paradigm vs 6% solver/step.

Run:  python solver_ablation.py
Outputs:
  solver_ablation.pdf
  solver_ablation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

# colorblind palette
PAL = sns.color_palette("colorblind")
C_HEUN = PAL[0]   # blue
C_EULER = PAL[1]  # orange
C_DPMPP = PAL[2]  # green

HERE = Path(__file__).resolve().parent

# Data
CONFIGS = [
    ("Heun", 4, 0.856),
    ("Euler", 4, 0.855),
    ("DPM++", 4, 0.855),
    ("Euler", 1, 0.851),
    ("DPM++", 1, 0.851),
]
SOLVER_COLORS = {"Heun": C_HEUN, "Euler": C_EULER, "DPM++": C_DPMPP}


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    names = [f"{s}\n{st}-step" for s, st, _ in CONFIGS]
    maps = [m for _, _, m in CONFIGS]
    colors = [SOLVER_COLORS[s] for s, _, _ in CONFIGS]
    hatches = ["//", "//", "//", "..", ".."]

    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, maps, width=0.6, color=colors, edgecolor="black", lw=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    for xi, m in zip(x, maps):
        ax.text(xi, m + 0.00015, f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("mAP (24obj val)")
    ax.set_ylim(0.845, 0.858)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)

    # y-axis break symbol (//)
    break_y = 0.847
    d = 0.00035
    ax.plot([-0.06, -0.02], [break_y - d, break_y + d],
            color="black", lw=0.9, clip_on=False, transform=ax.get_yaxis_transform())
    ax.plot([-0.06, -0.02], [break_y + d * 0.8, break_y + d * 2.8],
            color="black", lw=0.9, clip_on=False, transform=ax.get_yaxis_transform())

    from matplotlib.patches import Patch

    solver_handles = [
        Patch(facecolor=C_HEUN, edgecolor="black", label="Heun"),
        Patch(facecolor=C_EULER, edgecolor="black", label="Euler"),
        Patch(facecolor=C_DPMPP, edgecolor="black", label="DPM++"),
    ]
    step_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="4-step"),
        Patch(facecolor="white", edgecolor="black", hatch="..", label="1-step"),
    ]

    leg_solver = ax.legend(handles=solver_handles, loc="upper right",
                           fontsize=7, frameon=True, framealpha=0.9,
                           ncol=1, handletextpad=0.4, borderpad=0.3,
                           handlelength=1.2, labelspacing=0.3,
                           title="Solver", title_fontsize=7,
                           bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(leg_solver)

    leg_step = ax.legend(handles=step_handles, loc="upper right",
                         fontsize=7, frameon=True, framealpha=0.9,
                         ncol=1, handletextpad=0.4, borderpad=0.3,
                         handlelength=1.2, labelspacing=0.3,
                         title="Steps", title_fontsize=7,
                         bbox_to_anchor=(0.68, 1.0))

    fig.text(
        0.5, 0.01,
        r"RF paradigm: 94\% ($+0.077$)  vs  solver/step: 6\% ($+0.005$)",
        ha="center", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="0.7", lw=0.5),
    )

    plt.tight_layout()

    out_pdf = HERE / "solver_ablation.pdf"
    out_png = HERE / "solver_ablation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
