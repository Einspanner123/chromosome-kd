"""
Figure 6: Per-Class AP Analysis.

Horizontal bar chart: 24 classes grouped by chromosome size (Large A-C,
Medium D-E, Small F-G, Sex X/Y). Text color adapts to bar brightness
for readability.

Data source: main.tex / AAAI_INTEGRATED_DRAFT Section 4.3.1.

Run:  python per_class_ap.py
Outputs: per_class_ap.pdf, per_class_ap.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import *

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

# Human-readable group names for separator labels
GROUP_LABEL = {
    "Large": "Large (A–C)",
    "Medium": "Medium (D–E)",
    "Small": "Small (F–G)",
    "Sex": "Sex (X, Y)",
}


def _bar_text_color(rgb_tuple) -> str:
    """Choose white or black text depending on bar luminance."""
    if isinstance(rgb_tuple, str):
        return "white"  # fallback
    lum = 0.299 * rgb_tuple[0] + 0.587 * rgb_tuple[1] + 0.114 * rgb_tuple[2]
    return "white" if lum < 0.5 else "black"


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 5.2), constrained_layout=True)

    # Reverse so Large appears at top
    data = list(reversed(DATA))
    classes = [d[1] for d in data]
    aps = [d[2] for d in data]
    colors = [GROUP_COLOR[d[0]] for d in data]
    groups = [d[0] for d in data]

    y = np.arange(len(classes))
    bars = ax.barh(y, aps, height=0.6, color=colors, edgecolor="black", lw=0.4)

    # Value labels inside bars — adapt text color, 8pt minimum
    for bar, m, col in zip(bars, aps, colors):
        txt_color = _bar_text_color(col)
        ax.text(m - 0.004, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}", va="center", ha="right",
                fontsize=8, color=txt_color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("AP", fontsize=9)
    ax.set_xlim(0.75, 0.93)
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.set_axisbelow(True)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)

    # Overall mean line
    overall_mean = float(np.mean(aps))
    ax.axvline(overall_mean, color=C_OVERALL, lw=1.0, ls="--", alpha=0.8)
    # Mean label — concise
    ax.text(overall_mean + 0.002, len(classes) - 0.8,
            f"mean={overall_mean:.3f}",
            fontsize=7.5, color=C_OVERALL, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_OVERALL,
                      lw=0.5, alpha=0.9))

    # Group separator lines with group labels
    group_order = ["Large", "Medium", "Small", "Sex"]
    for i in range(len(group_order) - 1):
        g_top = group_order[i]
        g_bottom = group_order[i + 1]
        if g_top in [d[0] for d in reversed(DATA) if True] and \
           g_bottom in [d[0] for d in reversed(DATA) if True]:
            # Find the boundary between last of g_top and first of g_bottom
            reversed_data = list(reversed(DATA))
            top_last_idx = None
            bottom_first_idx = None
            for j, (g, c, val) in enumerate(reversed_data):
                if g == g_top:
                    top_last_idx = j
            for j, (g, c, val) in enumerate(reversed_data):
                if g == g_bottom:
                    bottom_first_idx = j
                    break
            if top_last_idx is not None and bottom_first_idx is not None:
                sep_y = (y[top_last_idx] + y[bottom_first_idx]) / 2
                ax.axhline(sep_y, color="0.7", lw=0.6, ls="--", alpha=0.7)

    # Legend
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
