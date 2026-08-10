"""Figure 1: generation--interaction--decision overview.

This is a conceptual diagram. It contains no empirical metric, does not claim
that RF or DPM-Solver++ are original algorithms, and includes only components
retained in the evidence-supported model definition.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from figure_style_v2 import (
    C_FOUNDATION,
    C_LIGHT,
    C_LINE,
    C_LQCR,
    C_LQCR_LIGHT,
    C_MUTED,
    C_OUTPUT,
    C_RF,
    C_RF_LIGHT,
    C_TEXT,
    arrow,
    configure_style,
    panel_title,
    rounded_box,
    save_vector_figure,
)


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_generation(ax: plt.Axes) -> None:
    setup_axis(ax)
    panel_title(ax, "a", "Generate candidates")

    # Noise boxes: same visual vocabulary as boxes, not abstract point clouds.
    noise_specs = [
        (0.06, 0.18, 0.22, 0.16, -7),
        (0.05, 0.44, 0.18, 0.23, 5),
        (0.14, 0.70, 0.20, 0.12, -4),
    ]
    for x, y, w, h, angle in noise_specs:
        rect = Rectangle(
            (x, y), w, h, angle=angle,
            facecolor="none", edgecolor=C_LINE,
            linewidth=0.8, linestyle=(0, (2, 2)), zorder=2,
        )
        ax.add_patch(rect)

    ax.text(0.05, 0.90, r"noisy proposals  $x_1$", color=C_MUTED,
            fontsize=6.6, ha="left")
    ax.text(0.95, 0.90, r"terminal boxes  $\hat{x}_0$", color=C_RF,
            fontsize=6.2, ha="right")

    # RF transports each proposal toward the terminal set.  Curves are kept
    # nearly straight to communicate the path choice without overstating a
    # theorem about learned trajectories.
    starts = np.array([[0.22, 0.28], [0.22, 0.55], [0.28, 0.76]])
    ends = np.array([[0.79, 0.26], [0.76, 0.54], [0.80, 0.76]])
    for start, end in zip(starts, ends):
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color=C_RF, linewidth=1.35, zorder=3,
        )
        for frac in (0.25, 0.50, 0.75):
            point = (1 - frac) * start + frac * end
            ax.scatter(*point, s=10, facecolor="white", edgecolor=C_RF,
                       linewidth=0.7, zorder=4)

    target_specs = [
        (0.76, 0.18, 0.18, 0.16),
        (0.73, 0.43, 0.20, 0.22),
        (0.77, 0.69, 0.17, 0.13),
    ]
    for x, y, w, h in target_specs:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=C_RF_LIGHT,
                               edgecolor=C_RF, linewidth=1.0, zorder=4))

    rounded_box(ax, (0.20, 0.03), 0.29, 0.11, "RF path",
                facecolor=C_RF_LIGHT, edgecolor=C_RF,
                textcolor=C_RF, fontsize=7.0, weight="bold")
    rounded_box(ax, (0.52, 0.03), 0.43, 0.11, "DPM-Solver++\n4 NFE",
                facecolor="white", edgecolor=C_FOUNDATION,
                textcolor=C_FOUNDATION, fontsize=6.0)


def panel_interaction(ax: plt.Axes) -> None:
    setup_axis(ax)
    panel_title(ax, "b", "Refine the proposal set")

    ax.text(0.04, 0.86, "coarse localization", color=C_MUTED,
            fontsize=6.7, ha="left")
    ax.text(0.58, 0.86, "late proposal refinement", color=C_RF,
            fontsize=6.7, ha="left", weight="bold")

    head_y, head_w, head_h = 0.54, 0.115, 0.20
    xs = np.linspace(0.035, 0.845, 6)
    for idx, x in enumerate(xs):
        late = idx >= 3
        rounded_box(
            ax, (x, head_y), head_w, head_h,
            "" if late else f"H{idx + 1}",
            facecolor=C_RF_LIGHT if late else C_LIGHT,
            edgecolor=C_RF if late else C_LINE,
            textcolor=C_RF if late else C_FOUNDATION,
            fontsize=7.0, weight="bold" if late else "normal",
        )
        if late:
            ax.text(x + head_w / 2, head_y + 0.155, f"H{idx + 1}",
                    ha="center", va="center", color=C_RF,
                    fontsize=6.6, weight="bold", zorder=6)
        if idx < 5:
            arrow(ax, (x + head_w + 0.012, head_y + head_h / 2),
                  (xs[idx + 1] - 0.012, head_y + head_h / 2),
                  color=C_LINE, linewidth=0.75)

    # Proposal icons indicate repeated box refinement without introducing an
    # unvalidated auxiliary module into the final method diagram.
    for center_x in xs[3:]:
        ax.add_patch(Rectangle((center_x + 0.025, 0.575), 0.042, 0.044,
                               fill=False, edgecolor=C_RF, linewidth=0.65,
                               zorder=5))
        ax.add_patch(Rectangle((center_x + 0.046, 0.590), 0.042, 0.044,
                               fill=False, edgecolor=C_RF, linewidth=0.65,
                               zorder=5))

    rounded_box(ax, (0.04, 0.06), 0.91, 0.12,
                "appearance + RoI features",
                facecolor="white", edgecolor=C_FOUNDATION,
                textcolor=C_FOUNDATION, fontsize=6.7)


def score_bar(ax: plt.Axes, y: float, label: str, value: float,
              color: str, rank: int) -> None:
    ax.text(0.04, y, label, ha="left", va="center", fontsize=6.7,
            color=C_TEXT)
    ax.add_patch(Rectangle((0.20, y - 0.025), 0.46, 0.05,
                           facecolor=C_LIGHT, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((0.20, y - 0.025), 0.46 * value, 0.05,
                           facecolor=color, edgecolor="none", zorder=2))
    ax.text(0.69, y, f"#{rank}", ha="left", va="center",
            fontsize=6.7, weight="bold", color=color)


def panel_decision(ax: plt.Axes) -> None:
    setup_axis(ax)
    panel_title(ax, "c", "Calibrate the final ranking")

    ax.text(0.04, 0.88, "classification-only", color=C_MUTED,
            fontsize=6.7, weight="bold")
    score_bar(ax, 0.78, "A", 0.92, C_FOUNDATION, 1)
    score_bar(ax, 0.68, "B", 0.84, C_FOUNDATION, 2)
    score_bar(ax, 0.58, "C", 0.76, C_FOUNDATION, 3)

    arrow(ax, (0.45, 0.51), (0.45, 0.42), color=C_LQCR, linewidth=1.0)
    rounded_box(
        ax, (0.18, 0.31), 0.54, 0.11, r"LQCR   $s=p\,q^2$",
        facecolor=C_LQCR_LIGHT, edgecolor=C_LQCR,
        textcolor=C_LQCR, fontsize=7.2, weight="bold",
    )

    ax.text(0.04, 0.24, "localization-aware", color=C_LQCR,
            fontsize=6.7, weight="bold")
    # B is promoted because its localization quality is higher.
    score_bar(ax, 0.15, "B", 0.78, C_LQCR, 1)
    score_bar(ax, 0.07, "A", 0.57, C_OUTPUT, 2)



def main() -> None:
    configure_style()
    fig = plt.figure(figsize=(7.2, 3.25), facecolor="white")
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=(1.0, 1.50, 1.05),
        left=0.025, right=0.985, bottom=0.08, top=0.91,
        wspace=0.12,
    )
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    panel_generation(axes[0])
    panel_interaction(axes[1])
    panel_decision(axes[2])

    # Thin separators give the three-stage logic structure without enclosing
    # every panel in a heavy box.
    for x in (0.305, 0.705):
        fig.add_artist(plt.Line2D([x, x], [0.10, 0.89], transform=fig.transFigure,
                                  color="#D0D5DD", linewidth=0.6))

    save_vector_figure(fig, "fig01_system_overview")


if __name__ == "__main__":
    main()
