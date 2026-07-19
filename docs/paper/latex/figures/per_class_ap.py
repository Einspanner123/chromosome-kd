"""
Figure 6: Per-Class AP Analysis — seaborn horizontal barplot with hue.

24 classes grouped by chromosome size (Large A-C, Medium D-E, Small F-G, Sex X/Y).
Uses sns.barplot() for automatic hue-based coloring and category ordering.

Run:  python per_class_ap.py
Outputs: per_class_ap.pdf, per_class_ap.png
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import *

# Tidy data: (group, class, AP)
DATA = [
    ("Large", "A1", 0.913), ("Large", "A2", 0.907), ("Large", "A3", 0.905),
    ("Large", "B4", 0.905), ("Large", "B5", 0.908),
    ("Large", "C6", 0.900), ("Large", "C7", 0.896), ("Large", "C8", 0.883),
    ("Large", "C9", 0.880), ("Large", "C10", 0.877), ("Large", "C11", 0.871),
    ("Large", "C12", 0.890),
    ("Medium", "D13", 0.857), ("Medium", "D14", 0.856),
    ("Medium", "D15", 0.845),
    ("Medium", "E16", 0.854), ("Medium", "E17", 0.842),
    ("Medium", "E18", 0.834),
    ("Small", "F19", 0.821), ("Small", "F20", 0.818),
    ("Small", "G21", 0.789), ("Small", "G22", 0.790),
    ("Sex", "X", 0.885), ("Sex", "Y", 0.776),
]

GROUP_COLOR = {"Large": C_LARGE, "Medium": C_MED, "Small": C_SMALL, "Sex": C_SEX}
GROUP_ORDER = ["Large", "Medium", "Small", "Sex"]


def _bar_text_color(rgb) -> str:
    """White on dark bars, black on light bars."""
    if isinstance(rgb, str):
        return "white"
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "white" if lum < 0.5 else "black"


def main() -> None:
    df = pd.DataFrame(DATA, columns=["group", "class", "ap"])
    # Reverse so Large appears at top
    cat_order = df["class"].tolist()[::-1]

    fig, ax = plt.subplots(figsize=(4.2, 5.2), constrained_layout=True)

    sns.barplot(
        data=df, y="class", x="ap", hue="group",
        palette=GROUP_COLOR, order=cat_order, hue_order=GROUP_ORDER,
        dodge=False, edgecolor="black", linewidth=0.4,
        saturation=1, ax=ax,
    )

    # Value labels inside bars — one per patch, text color adaptive
    for i, (_, row) in enumerate(df.iloc[::-1].iterrows()):
        bar = ax.patches[i]
        txt_color = _bar_text_color(GROUP_COLOR[row["group"]])
        ax.text(row["ap"] - 0.004, bar.get_y() + bar.get_height() / 2,
                f"{row['ap']:.3f}", va="center", ha="right",
                fontsize=8, color=txt_color, fontweight="bold")

    ax.set_xlabel("AP", fontsize=9)
    ax.set_ylabel("")
    ax.set_xlim(0.75, 0.93)
    ax.set_axisbelow(True)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
    ax.legend().remove()  # remove seaborn auto-legend; we'll add custom

    # Overall mean line
    overall_mean = df["ap"].mean()
    ax.axvline(overall_mean, color=C_OVERALL, lw=1.0, ls="--", alpha=0.8)
    ax.text(overall_mean + 0.002, len(df) - 0.8,
            f"mean={overall_mean:.3f}",
            fontsize=7.5, color=C_OVERALL, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_OVERALL,
                      lw=0.5, alpha=0.9))

    # Group separator lines (between group boundaries)
    sep_y = lambda g: (cat_order.index(df[df["group"] == g]["class"].iloc[-1]) +
                       cat_order.index(df[df["group"] == g]["class"].iloc[0])) / 2
    # Actually simpler: compute from reversed index positions
    reverted_positions = {row["class"]: i for i, (_, row)
                          in enumerate(df.iloc[::-1].iterrows())}
    for i in range(len(GROUP_ORDER) - 1):
        g_top = GROUP_ORDER[i]
        g_bot = GROUP_ORDER[i + 1]
        top_last = df[df["group"] == g_top]["class"].iloc[-1]
        bot_first = df[df["group"] == g_bot]["class"].iloc[0]
        if top_last in reverted_positions and bot_first in reverted_positions:
            sy = (reverted_positions[top_last] + reverted_positions[bot_first]) / 2
            ax.axhline(sy, color="0.7", lw=0.6, ls="--", alpha=0.7)

    # Custom legend
    handles = [
        Patch(facecolor=GROUP_COLOR[g], edgecolor="black",
              label=f"{g} ({chr(65+i)}–{chr(68+i) if i < 2 else 'G' if i == 2 else 'Y'})")
        for i, g in enumerate(GROUP_ORDER)
    ]
    # Fix the label formatting
    handles = [
        Patch(facecolor=C_LARGE, edgecolor="black", label="Large (A–C)"),
        Patch(facecolor=C_MED, edgecolor="black", label="Medium (D–E)"),
        Patch(facecolor=C_SMALL, edgecolor="black", label="Small (F–G)"),
        Patch(facecolor=C_SEX, edgecolor="black", label="Sex (X, Y)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              fontsize=8, framealpha=0.95)

    save_fig(fig, "per_class_ap")


if __name__ == "__main__":
    main()
