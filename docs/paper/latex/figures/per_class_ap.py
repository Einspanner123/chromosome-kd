"""Figure 5: Per-Class AP Analysis.

Horizontal bar chart showing 24 classes' AP from paper Sec 4.3.1,
sorted by chromosome size group (Large A-C, Medium D-E, Small F-G, Sex X/Y).
Annotates size-dependent degradation.

Data source: AAAI_INTEGRATED_DRAFT.md, Section 4.3.1 per-class AP table.

Run:  python per_class_ap.py
Outputs:
  per_class_ap.pdf
  per_class_ap.png
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
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

PAL = sns.color_palette("colorblind")
C_LARGE = PAL[0]
C_MED = PAL[1]
C_SMALL = PAL[2]
C_SEX = PAL[3]
C_OVERALL = "0.3"

HERE = Path(__file__).resolve().parent

DATA = [
    ("Large", "A1", 0.913), ("Large", "A2", 0.907), ("Large", "A3", 0.905),
    ("Large", "B4", 0.905), ("Large", "B5", 0.908),
    ("Large", "C6", 0.900), ("Large", "C7", 0.896), ("Large", "C8", 0.883),
    ("Large", "C9", 0.880), ("Large", "C10", 0.877), ("Large", "C11", 0.871),
    ("Large", "C12", 0.890),
    ("Medium", "D13", 0.857), ("Medium", "D14", 0.856), ("Medium", "D15", 0.845),
    ("Medium", "E16", 0.854), ("Medium", "E17", 0.842), ("Medium", "E18", 0.834),
    ("Small", "F19", 0.821), ("Small", "F20", 0.818),
    ("Small", "G21", 0.789), ("Small", "G22", 0.790),
    ("Sex", "X", 0.885), ("Sex", "Y", 0.776),
]
GROUP_COLOR = {"Large": C_LARGE, "Medium": C_MED, "Small": C_SMALL, "Sex": C_SEX}


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 5.2), constrained_layout=True)

    data = list(reversed(DATA))
    classes = [d[1] for d in data]
    aps = [d[2] for d in data]
    colors = [GROUP_COLOR[d[0]] for d in data]
    groups = [d[0] for d in data]

    y = np.arange(len(classes))
    bars = ax.barh(y, aps, height=0.6, color=colors, edgecolor="black", lw=0.4)

    group_bounds = {}
    last_group = None
    start = 0
    for i, g in enumerate(groups):
        if g != last_group and last_group is not None:
            group_bounds[last_group] = (start, i - 1)
            start = i
        last_group = g
    if last_group is not None:
        group_bounds[last_group] = (start, len(groups) - 1)

    for yi, m in zip(y, aps):
        ax.text(m - 0.003, yi, f"{m:.3f}", va="center", ha="right",
                fontsize=6.5, color="white", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=7.5)
    ax.set_xlabel("AP")
    ax.set_xlim(0.75, 0.93)
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)

    overall_mean = float(np.mean(aps))
    ax.axvline(overall_mean, color=C_OVERALL, lw=1.0, ls="--", alpha=0.8)
    ax.text(0.752, len(classes) - 0.8,
            f"mean = {overall_mean:.3f}",
            fontsize=7, color=C_OVERALL, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_OVERALL,
                      lw=0.5, alpha=0.9))

    group_order = ["Large", "Medium", "Small", "Sex"]
    for i in range(len(group_order) - 1):
        g_top = group_order[i]
        g_bottom = group_order[i + 1]
        if g_top in group_bounds and g_bottom in group_bounds:
            top_bottom = group_bounds[g_top][0]
            bottom_top = group_bounds[g_bottom][1]
            sep_y = (top_bottom + bottom_top) / 2
            ax.axhline(sep_y, color="0.7", lw=0.6, ls="--", alpha=0.7)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_LARGE, edgecolor="black", label="Large (A-C)"),
        Patch(facecolor=C_MED, edgecolor="black", label="Medium (D-E)"),
        Patch(facecolor=C_SMALL, edgecolor="black", label="Small (F-G)"),
        Patch(facecolor=C_SEX, edgecolor="black", label="Sex (X, Y)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=7,
              framealpha=0.95)

    out_pdf = HERE / "per_class_ap.pdf"
    out_png = HERE / "per_class_ap.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
