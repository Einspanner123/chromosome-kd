"""
Figure 1: Method Overview - Rectified Flow for Chromosome Detection.

Three-panel overview with clearer layout:
  (a) RF straight-line vs DDPM curved trajectory.
  (b) AdaLN-Zero time conditioning block (redesigned with more space).
  (c) Coupling comparison: Random vs Stochastic OT (Sinkhorn).

Run:  python method_overview.py
Outputs: method_overview.pdf, method_overview.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_style import *


def panel_trajectories(ax: plt.Axes) -> None:
    x1 = np.array([0.0, 0.0])
    x0 = np.array([1.0, 1.0])
    t = np.linspace(0.0, 1.0, 300)

    rf = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    ax.plot(rf[:, 0], rf[:, 1], color=C_RF, lw=2.0, label="RF (straight)")

    base = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    perp = np.array([-0.6, 0.6])
    perp = perp / np.linalg.norm(perp)
    bend = 0.28 * np.sin(np.pi * t) ** 1.5
    ddpm = base + bend[:, None] * perp
    ax.plot(ddpm[:, 0], ddpm[:, 1], color=C_DDPM, lw=2.0, ls="--",
            label="DDPM (curved)")

    ax.scatter(*x1, s=90, color=C_SOURCE, zorder=6, edgecolor="k", lw=1.0)
    ax.scatter(*x0, s=90, color=C_GT, zorder=6, edgecolor="k", lw=1.0)

    step_ts = [0.0, 0.33, 0.67, 1.0]
    for ts in step_ts:
        p = (1.0 - ts) * x0 + ts * x1
        ax.scatter(*p, s=25, color=C_RF, zorder=5, marker="o",
                   edgecolor="white", lw=0.8)

    ax.text(-0.16, -0.14, r"$\mathbf{x}_1$ (noise)", fontsize=9,
            ha="left", va="top", color=C_SOURCE)
    ax.text(1.10, 1.08, r"$\mathbf{x}_0$ (GT box)", fontsize=9,
            ha="left", va="bottom", color="black")

    ax.annotate("", xy=(0.88, -0.22), xytext=(0.12, -0.22),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(0.5, -0.32, "decreasing $t$  (few-step)",
            fontsize=8, ha="center", color="black")

    ax.set_xlim(-0.18, 1.22)
    ax.set_ylim(-0.42, 1.24)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Trajectories", **PANEL_LABEL_KW)
    hide_spines(ax)

    handles = [
        Line2D([0], [0], color=C_RF, lw=2, label="RF (straight)"),
        Line2D([0], [0], color=C_DDPM, lw=2, ls="--", label="DDPM (curved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SOURCE,
               markeredgecolor="k", markersize=6, label=r"$\mathbf{x}_1$ noise"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GT,
               markeredgecolor="k", markersize=6, label=r"$\mathbf{x}_0$ GT"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
              handletextpad=0.4)


def panel_adaln(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) AdaLN-Zero", **PANEL_LABEL_KW)
    hide_spines(ax)

    def line(x1, y1, x2, y2, color=C_DARKGRAY, lw=0.9):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2)

    x_start = 0.5
    x_mid1 = 2.5
    x_mid2 = 4.5
    x_mid3 = 6.5
    x_end = 8.5

    y_time = 6.0
    y_mlp = 5.0
    y_bus = 4.2
    y_modulation = 3.2
    y_feature_in = 1.5
    y_out = 0.5

    box(ax, x_start, y_time, 1.2, 0.8, r"$t$", fc=C_LIGHTGRAY, fontsize=9, weight="bold")
    box(ax, x_mid1, y_time, 1.8, 0.8, r"$\phi(t)$", fc=C_ADALN, ec=C_ADALN, text_color="white", fontsize=9, weight="bold")
    box(ax, x_mid2, y_time, 1.5, 0.8, "MLP", fc=C_LIGHTGRAY, fontsize=9, weight="bold")

    arrow(ax, x_start + 1.2, y_time + 0.4, x_mid1, y_time + 0.4)
    arrow(ax, x_mid1 + 1.8, y_time + 0.4, x_mid2, y_time + 0.4)

    line(x_mid2 + 0.75, y_time, x_mid2 + 0.75, y_bus)
    line(x_start + 0.6, y_bus, x_mid3 + 1.0, y_bus)

    gamma_x = x_start + 0.5
    beta_x = x_mid1 + 0.9
    alpha_x = x_mid3

    arrow(ax, gamma_x, y_bus, gamma_x, y_modulation + 0.8)
    arrow(ax, beta_x, y_bus, beta_x, y_modulation + 0.8)
    arrow(ax, alpha_x, y_bus, alpha_x, y_modulation + 0.8)

    box(ax, gamma_x - 0.6, y_modulation, 1.2, 0.8, r"$\gamma$", fc="#e8f5ee", ec=C_ADALN, fontsize=9, weight="bold")
    box(ax, beta_x - 0.6, y_modulation, 1.2, 0.8, r"$\beta$", fc="#e8f5ee", ec=C_ADALN, fontsize=9, weight="bold")
    box(ax, alpha_x - 0.6, y_modulation, 1.2, 0.8, r"$\alpha$", fc="#f0f0f0", ec=C_MIDGRAY, fontsize=9, weight="bold")

    ax.text(gamma_x, y_modulation - 0.2, "(scale)", fontsize=7, ha="center", color=C_ADALN)
    ax.text(beta_x, y_modulation - 0.2, "(shift)", fontsize=7, ha="center", color=C_ADALN)
    ax.text(alpha_x, y_modulation - 0.2, "(gate)", fontsize=7, ha="center", color=C_MIDGRAY)

    ax.text(x_start + 0.5, y_feature_in + 0.4, r"$\mathbf{h}$", fontsize=10, ha="center", color=C_DARKGRAY, weight="bold")
    box(ax, x_mid1 - 0.3, y_feature_in, 2.2, 0.9, r"$\mathbf{h} \odot \gamma + \beta$", fc="#ffffff", fontsize=9, weight="bold")
    arrow(ax, x_start + 0.7, y_feature_in + 0.45, x_mid1 - 0.3, y_feature_in + 0.45, color=C_MIDGRAY)

    arrow(ax, gamma_x, y_modulation, gamma_x, y_feature_in + 0.9, color=C_ADALN, lw=0.9)
    arrow(ax, beta_x, y_modulation, beta_x, y_feature_in + 0.9, color=C_ADALN, lw=0.9)

    ax.text(x_mid1 + 0.8, y_feature_in + 0.4, r"$\mathbf{h}'$", fontsize=10, ha="center", color=C_DARKGRAY, weight="bold")
    box(ax, x_mid2, y_feature_in, 2.0, 0.9, r"$\alpha \cdot \mathbf{h}'$", fc="#ffffff", fontsize=9, weight="bold")
    arrow(ax, x_mid1 + 0.8, y_feature_in + 0.45, x_mid2, y_feature_in + 0.45, color=C_MIDGRAY)

    arrow(ax, alpha_x, y_modulation, alpha_x, y_feature_in + 0.9, color=C_MIDGRAY, lw=0.9)

    ax.text(x_end, y_feature_in + 0.4, "out", fontsize=10, ha="center", color=C_DARKGRAY, weight="bold")
    arrow(ax, x_mid2 + 2.0, y_feature_in + 0.45, x_end, y_feature_in + 0.45, color=C_MIDGRAY)

    annotation_box(ax, 0.5, 0.05, r"$zero$-$init$ $\Rightarrow$ identity at $t{=}0$",
                   fontsize=8.5, ha="left", va="bottom", fc="#f7f7f7", ec="#cccccc")


def panel_ot(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) Coupling strategies", **PANEL_LABEL_KW)
    hide_spines(ax)

    rng = np.random.default_rng(7)
    n = 6

    left_center = 2.5
    noise_x_l = left_center - 1.4
    gt_x_l = left_center + 1.4
    noise_y = np.linspace(1.4, 4.9, n)
    gt_y = np.linspace(0.9, 4.4, n) + rng.normal(0, 0.1, size=n)

    ax.text(left_center, 5.6, "Random", fontsize=10, ha="center",
            weight="bold", color=C_RAND)

    perm = rng.permutation(n)
    for i in range(n):
        ax.plot([noise_x_l, gt_x_l], [noise_y[i], gt_y[perm[i]]],
                color=C_RAND, lw=0.9, alpha=0.75, zorder=2)

    for y in noise_y:
        ax.scatter(noise_x_l, y, s=70, color=C_SOURCE, edgecolor="k",
                   lw=0.7, zorder=4, marker="o")
    for y in gt_y:
        ax.scatter(gt_x_l, y, s=70, color=C_GT, edgecolor="k",
                   lw=0.7, zorder=4, marker="s")

    ax.text(noise_x_l, 0.85, "noise", fontsize=9, ha="center",
            color=C_SOURCE, weight="bold")
    ax.text(gt_x_l, 0.35, "GT", fontsize=9, ha="center",
            color=C_GT, weight="bold")
    ax.text(left_center, 0.08, r"$H(V|X_t)=\log K$", fontsize=10,
            ha="center", color=C_RAND, weight="bold")

    right_center = 7.5
    noise_x_r = right_center - 1.4
    gt_x_r = right_center + 1.4

    ax.text(right_center, 5.6, "Stochastic OT", fontsize=10, ha="center",
            weight="bold", color=C_OT)

    ot_perm = np.arange(n)
    ot_perm[1], ot_perm[2] = ot_perm[2], ot_perm[1]
    ot_perm[5] = 4
    ot_perm[4] = 5

    for i in range(n):
        ax.plot([noise_x_r, gt_x_r], [noise_y[i], gt_y[ot_perm[i]]],
                color=C_OT, lw=0.9, alpha=0.75, zorder=2)

    for y in noise_y:
        ax.scatter(noise_x_r, y, s=70, color=C_SOURCE, edgecolor="k",
                   lw=0.7, zorder=4, marker="o")
    for y in gt_y:
        ax.scatter(gt_x_r, y, s=70, color=C_GT, edgecolor="k",
                   lw=0.7, zorder=4, marker="s")

    ax.text(noise_x_r, 0.85, "noise", fontsize=9, ha="center",
            color=C_SOURCE, weight="bold")
    ax.text(gt_x_r, 0.35, "GT", fontsize=9, ha="center",
            color=C_GT, weight="bold")
    ax.text(right_center, 0.08, r"$0 < H(V|X_t) < \log K$",
            fontsize=10, ha="center", color=C_OT, weight="bold")


def main() -> None:
    fig = plt.figure(figsize=(9.0, 3.5), constrained_layout=True)

    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.2], wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_trajectories(ax_a)
    panel_adaln(ax_b)
    panel_ot(ax_c)

    save_fig(fig, "method_overview")


if __name__ == "__main__":
    main()
