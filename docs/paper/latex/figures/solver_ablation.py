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
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
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
    fig, ax = plt.subplots(figsize=(5.2, 3.9))

    names = [f"{s}\n{st}-step" for s, st, _ in CONFIGS]
    maps = [m for _, _, m in CONFIGS]
    colors = [SOLVER_COLORS[s] for s, _, _ in CONFIGS]
    hatches = ["//", "//", "//", "..", ".."]

    x = np.arange(len(CONFIGS))
    bars = ax.bar(x, maps, width=0.62, color=colors, edgecolor="black", lw=0.7)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    for xi, m in zip(x, maps):
        ax.text(xi, m + 0.0002, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8.5, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("mAP (24obj val)", fontsize=10)
    ax.set_ylim(0.845, 0.862)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=9)

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

    # Legends placed BELOW the axes (avoids overlap with tall broken-axis bars)
    leg_solver = ax.legend(handles=solver_handles, loc="upper left",
                           bbox_to_anchor=(0.0, -0.18),
                           fontsize=8.5, frameon=False, ncol=3,
                           handletextpad=0.4, handlelength=1.3,
                           columnspacing=1.1, borderpad=0.3,
                           title="Solver", title_fontsize=8.5)
    ax.add_artist(leg_solver)

    leg_step = ax.legend(handles=step_handles, loc="upper left",
                         bbox_to_anchor=(0.58, -0.18),
                         fontsize=8.5, frameon=False, ncol=2,
                         handletextpad=0.4, handlelength=1.3,
                         columnspacing=0.9, borderpad=0.3,
                         title="Steps", title_fontsize=8.5)

    fig.text(
        0.5, 0.045,
        r"4-step vs 1-step: $\Delta$0.004 mAP $\Rightarrow$ solver/step contributes 6\% of total RF gain",
        ha="center", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="0.7", lw=0.5),
    )

    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.30)

    out_pdf = HERE / "solver_ablation.pdf"
    out_png = HERE / "solver_ablation.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
