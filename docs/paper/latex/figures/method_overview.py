"""
Figure 1: Method Overview — Rectified Flow for Chromosome Detection.

Three-panel overview:
  (a) RF straight-line vs DDPM curved trajectory.
  (b) AdaLN-Zero time conditioning block.
  (c) Coupling comparison: Random vs Stochastic OT (Sinkhorn).

Run:  python method_overview.py
Outputs: method_overview.pdf, method_overview.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figure_style import *


# =====================================================================
# Panel (a): Trajectories — RF straight vs DDPM curved
# =====================================================================
def panel_trajectories(ax: plt.Axes) -> None:
    x1 = np.array([0.0, 0.0])   # noise
    x0 = np.array([1.0, 1.0])   # GT
    t = np.linspace(0.0, 1.0, 300)

    # RF straight-line path
    rf = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    ax.plot(rf[:, 0], rf[:, 1], color=C_RF, lw=2.0, label="RF (straight)")

    # DDPM curved path — smooth sine bend
    base = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    perp = np.array([-0.6, 0.6])
    perp = perp / np.linalg.norm(perp)
    bend = 0.28 * np.sin(np.pi * t) ** 1.5
    ddpm = base + bend[:, None] * perp
    ax.plot(ddpm[:, 0], ddpm[:, 1], color=C_DDPM, lw=2.0, ls="--",
            label="DDPM (curved)")

    # Endpoints
    ax.scatter(*x1, s=90, color=C_SOURCE, zorder=6, edgecolor="k", lw=1.0)
    ax.scatter(*x0, s=90, color=C_GT, zorder=6, edgecolor="k", lw=1.0)

    # RF step nodes
    step_ts = [0.0, 0.33, 0.67, 1.0]
    for ts in step_ts:
        p = (1.0 - ts) * x0 + ts * x1
        ax.scatter(*p, s=22, color=C_RF, zorder=5, marker="o",
                   edgecolor="white", lw=0.8)

    # Endpoint labels (offset to avoid overlap with markers)
    ax.text(-0.16, -0.14, r"$\mathbf{x}_1$ (noise)", fontsize=9,
            ha="left", va="top", color=C_SOURCE)
    ax.text(1.10, 1.08, r"$\mathbf{x}_0$ (GT box)", fontsize=9,
            ha="left", va="bottom", color="black")

    # Time arrow (inference direction: noise → GT ≡ decreasing t)
    ax.annotate("", xy=(0.88, -0.22), xytext=(0.12, -0.22),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(0.5, -0.32, "decreasing $t$  (few-step inference)",
            fontsize=8, ha="center", color="black")

    ax.set_xlim(-0.18, 1.22)
    ax.set_ylim(-0.42, 1.24)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Trajectories", loc="left", fontsize=10, weight="bold",
                 pad=3)
    hide_spines(ax)

    # Compact legend
    handles = [
        Line2D([0], [0], color=C_RF, lw=2, label="RF (straight)"),
        Line2D([0], [0], color=C_DDPM, lw=2, ls="--", label="DDPM (curved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SOURCE,
               markeredgecolor="k", markersize=6, label="$\\mathbf{x}_1$ noise"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GT,
               markeredgecolor="k", markersize=6, label="$\\mathbf{x}_0$ GT"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8,
              handletextpad=0.4)


# =====================================================================
# Panel (b): AdaLN-Zero block
# =====================================================================
def panel_adaln(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) AdaLN-Zero", loc="left", fontsize=10, weight="bold",
                 pad=3)
    hide_spines(ax)

    def line_2pt(x1, y1, x2, y2, color=C_DARKGRAY, lw=0.9):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2)

    # === TOP row: time embedding ===
    y_top = 5.5; h_top = 0.8
    box(ax, 0.3, y_top, 1.0, h_top, r"$t$", fc=C_LIGHTGRAY, fontsize=9)
    box(ax, 1.7, y_top, 2.2, h_top, r"sin emb $\phi(t)$",
        fc=C_ADALN, ec=C_ADALN, text_color="white", fontsize=9)
    arrow(ax, 1.3, y_top + h_top/2, 1.7, y_top + h_top/2)
    box(ax, 4.3, y_top, 1.4, h_top, "MLP", fc=C_LIGHTGRAY, fontsize=9)
    arrow(ax, 3.9, y_top + h_top/2, 4.3, y_top + h_top/2)

    mlp_cx = 5.0
    mlp_bottom_y = y_top

    # === MIDDLE: modulation branches ===
    y_branch = 3.8; h_branch = 0.7
    mod_cx = 2.9
    gate_cx = 6.75
    gx = mod_cx - 0.7
    bx = mod_cx + 0.7
    alx = gate_cx

    # Junction bar: MLP → horizontal bus → branches
    jun_y = mlp_bottom_y - 0.5
    branch_top = y_branch + h_branch
    line_2pt(mlp_cx, mlp_bottom_y, mlp_cx, jun_y)
    line_2pt(gx, jun_y, alx, jun_y)
    arrow(ax, gx, jun_y, gx, branch_top, lw=0.9)
    arrow(ax, bx, jun_y, bx, branch_top, lw=0.9)
    arrow(ax, alx, jun_y, alx, branch_top, lw=0.9)

    box(ax, gx - 0.6, y_branch, 1.2, h_branch, r"$\gamma$ (scale)",
        fc="#e8f5ee", ec=C_ADALN, fontsize=8)
    box(ax, bx - 0.6, y_branch, 1.2, h_branch, r"$\beta$ (shift)",
        fc="#e8f5ee", ec=C_ADALN, fontsize=8)
    box(ax, alx - 0.6, y_branch, 1.2, h_branch, r"$\alpha$ (gate)",
        fc="#f0f0f0", ec=C_MIDGRAY, fontsize=8)

    # === BOTTOM row: feature path ===
    y_feat = 1.6; h_feat = 0.9
    feat_top = y_feat + h_feat

    ax.text(0.6, y_feat + h_feat/2, r"$\mathbf{h}$",
            fontsize=10, ha="center", color=C_DARKGRAY, weight="bold")

    box(ax, 1.5, y_feat, 2.8, h_feat, r"$\mathbf{h} \odot \gamma + \beta$",
        fc="#ffffff", fontsize=9)
    arrow(ax, 1.0, y_feat + h_feat/2, 1.5, y_feat + h_feat/2,
          color=C_MIDGRAY, lw=1.0)

    arrow(ax, gx, y_branch, gx, feat_top, color=C_ADALN, lw=0.9)
    arrow(ax, bx, y_branch, bx, feat_top, color=C_ADALN, lw=0.9)

    ax.text(4.7, y_feat + h_feat/2, r"$\mathbf{h}'$",
            fontsize=10, ha="center", color=C_DARKGRAY, weight="bold")
    arrow(ax, 4.3, y_feat + h_feat/2, 4.5, y_feat + h_feat/2,
          color=C_MIDGRAY, lw=1.0)

    box(ax, 5.5, y_feat, 2.5, h_feat, r"$\alpha \cdot \mathbf{h}'$",
        fc="#ffffff", fontsize=9)
    arrow(ax, 4.9, y_feat + h_feat/2, 5.5, y_feat + h_feat/2,
          color=C_MIDGRAY, lw=1.0)

    arrow(ax, alx, y_branch, alx, feat_top, color=C_MIDGRAY, lw=0.9)

    ax.text(8.7, y_feat + h_feat/2, "out",
            fontsize=10, ha="center", color=C_DARKGRAY, weight="bold")
    arrow(ax, 8.0, y_feat + h_feat/2, 8.3, y_feat + h_feat/2,
          color=C_MIDGRAY, lw=1.0)

    # Zero-init note at bottom
    annotation_box(ax, 0.5, 0.08,
                   r"zero-init $\Rightarrow$ identity at $t{=}0$",
                   fontsize=7.5, ha="center", va="bottom",
                   fc="#f7f7f7", ec="#cccccc")


# =====================================================================
# Panel (c): Coupling — Random vs Stochastic OT
# =====================================================================
def panel_ot(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) Coupling strategies", loc="left", fontsize=10,
                 weight="bold", pad=3)
    hide_spines(ax)

    rng = np.random.default_rng(7)
    n = 6

    # --- Left: Random coupling ---
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

    # --- Right: Stochastic OT ---
    right_center = 7.5
    noise_x_r = right_center - 1.4
    gt_x_r = right_center + 1.4

    ax.text(right_center, 5.6, "Stochastic OT", fontsize=10, ha="center",
            weight="bold", color=C_OT)

    # Sinkhorn: mostly nearest-neighbor with small stochasticity
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


# =====================================================================
# Main
# =====================================================================
def main() -> None:
    fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)

    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_trajectories(ax_a)
    panel_adaln(ax_b)
    panel_ot(ax_c)

    save_fig(fig, "method_overview")


if __name__ == "__main__":
    main()
