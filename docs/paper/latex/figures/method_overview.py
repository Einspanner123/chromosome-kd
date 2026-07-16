"""Figure 1: Method Overview.

Three-panel overview of the RF chromosome detection framework:
  (a) DDPM curved trajectory vs RF straight-line trajectory.
  (b) AdaLN-Zero time conditioning schematic.
  (c) OT coupling strategy: Random vs Sinkhorn stochastic assignment.

Run:  python method_overview.py
Outputs:
  method_overview.pdf  (vector)
  method_overview.png  (300 dpi preview)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------------------
# Style
# -------------------------------------------------------------------------------------
import seaborn as sns

sns.set_style("white")
sns.set_palette("colorblind")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
    }
)

# colorblind palette (avoid red-green)
C_DDPM = "#D55E00"   # vermillion
C_RF = "#0072B2"     # blue
C_ADALN = "#009E73"  # green
C_OT = "#CC79A7"     # pink
C_RAND = "#E69F00"   # orange
C_NOISE = "#56B4E9"  # sky blue
C_GT = "#000000"     # black
C_DARKGRAY = "#333333"
C_MIDGRAY = "#666666"
C_LIGHTGRAY = "#E8E8E8"

HERE = Path(__file__).resolve().parent


# -------------------------------------------------------------------------------------
# Panel (a): Trajectories
# -------------------------------------------------------------------------------------
def panel_trajectories(ax: plt.Axes) -> None:
    # 2D illustration: x_1 (noise) at (0, 0), x_0 (GT) at (1, 1)
    x1 = np.array([0.0, 0.0])
    x0 = np.array([1.0, 1.0])
    t = np.linspace(0.0, 1.0, 300)

    # RF straight-line path
    rf = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    ax.plot(rf[:, 0], rf[:, 1], color=C_RF, lw=2.0, label="RF (straight)")

    # DDPM curved trajectory - natural smooth curve
    # DDPM forward process adds noise gradually, creating a curved path in 2D projection
    base = (1.0 - t)[:, None] * x0 + t[:, None] * x1
    # perpendicular direction to the straight path
    perp = np.array([-0.6, 0.6])
    perp = perp / np.linalg.norm(perp)
    
    # Smooth, natural-looking bend using sine^1.5 for rounder curve
    bend = 0.28 * np.sin(np.pi * t) ** 1.5
    ddpm = base + bend[:, None] * perp
    ax.plot(ddpm[:, 0], ddpm[:, 1], color=C_DDPM, lw=2.0, ls="--", label="DDPM (curved)")

    # endpoints - larger and more distinct
    ax.scatter(*x1, s=90, c=C_NOISE, zorder=6, edgecolor="k", lw=1.0, label="$x_1$ noise")
    ax.scatter(*x0, s=90, c=C_GT, zorder=6, edgecolor="k", lw=1.0, label="$x_0$ GT box")

    # RF step nodes - fewer, clearer
    step_ts = [0.0, 0.33, 0.67, 1.0]
    for ts in step_ts:
        p = (1.0 - ts) * x0 + ts * x1
        ax.scatter(*p, s=22, c=C_RF, zorder=5, marker="o", edgecolor="white", lw=0.8)

    # endpoint labels
    ax.text(-0.06, -0.06, "$x_1$ (noise)", fontsize=7.5, ha="left", va="top")
    ax.text(1.04, 1.04, "$x_0$ (GT)", fontsize=7.5, ha="left", va="bottom")

    # time arrow - black solid, prominent
    # Arrow: noise (x1, left) → GT (x0, right) = decreasing t (inference direction)
    ax.annotate("", xy=(0.88, -0.22), xytext=(0.12, -0.22),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(0.5, -0.30, r"decreasing $t$  (few-step)", fontsize=7.5, ha="center",
            color="black")

    ax.set_xlim(-0.18, 1.22)
    ax.set_ylim(-0.42, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Trajectories: RF vs DDPM", fontsize=10, weight="bold")
    
    # custom legend
    handles = [
        Line2D([0], [0], color=C_RF, lw=2, label="RF (straight)"),
        Line2D([0], [0], color=C_DDPM, lw=2, ls="--", label="DDPM (curved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NOISE,
               markeredgecolor="k", markersize=7, label="$x_1$ noise"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GT,
               markeredgecolor="k", markersize=7, label="$x_0$ GT"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)


# -------------------------------------------------------------------------------------
# Panel (b): AdaLN-Zero block
# -------------------------------------------------------------------------------------
def panel_adaln(ax: plt.Axes) -> None:
    # Classic T-shape layout:
    #   TOP (horizontal, left->right):  t -> sin emb -> MLP
    #   MLP drops down to 3 parallel branches:  gamma | beta | alpha
    #   BOTTOM (horizontal, left->right):  h -> Modulate -> Gate -> output
    #   gamma, beta feed into Modulate; alpha feeds into Gate
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) AdaLN-Zero Time Conditioning", fontsize=10, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    def box(x, y, w, h, text, fc="#ffffff", ec=C_DARKGRAY, text_color="black", fontsize=8):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", 
                              fc=fc, ec=ec, lw=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", 
                fontsize=fontsize, color=text_color)

    def arrow(x1, y1, x2, y2, color=C_DARKGRAY, lw=1.0, mutation_scale=12):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", mutation_scale=mutation_scale,
                                    color=color, lw=lw), zorder=3)

    # === TOP: time path (horizontal) ===
    y_top = 5.0
    h_top = 0.8
    
    # t
    box(0.3, y_top, 1.0, h_top, r"$t$", fc=C_LIGHTGRAY)
    # sin emb
    box(1.7, y_top, 2.2, h_top, r"sin emb $\phi(t)$", 
        fc=C_ADALN, ec=C_ADALN, text_color="white")
    arrow(1.3, y_top + h_top/2, 1.7, y_top + h_top/2)
    # MLP
    box(4.3, y_top, 1.4, h_top, "MLP", fc=C_LIGHTGRAY)
    arrow(3.9, y_top + h_top/2, 4.3, y_top + h_top/2)

    # MLP splits into 3 vertical branches: gamma, beta, alpha
    mlp_bottom_y = y_top
    # Modulate box center x = 1.5 + 2.8/2 = 2.9
    # Gate box center x = 5.5 + 2.5/2 = 6.75
    # gamma and beta both feed into Modulate, so place them above Modulate
    mod_cx = 1.5 + 2.8 / 2   # 2.9
    gate_cx = 5.5 + 2.5 / 2  # 6.75

    # gamma (scale) — left half above Modulate
    gx = mod_cx - 0.7
    # beta (shift) — right half above Modulate
    bx = mod_cx + 0.7
    # alpha (gate) — above Gate
    alx = gate_cx

    # gamma (scale)
    box(gx - 0.7, 3.9, 1.4, 0.7, r"$\gamma$ (scale)", fc="#ffffff", fontsize=7.5)
    # beta (shift)
    box(bx - 0.7, 3.9, 1.4, 0.7, r"$\beta$ (shift)", fc="#ffffff", fontsize=7.5)
    # alpha (gate) - zero init
    box(alx - 0.7, 3.9, 1.4, 0.7, r"$\alpha$ (gate)", fc="#f0f0f0", fontsize=7.5)

    # Vertical arrows from MLP down to each of the 3
    mlp_cx = 5.0
    # horizontal junction line
    ax.plot([gx, alx], [mlp_bottom_y - 0.3, mlp_bottom_y - 0.3], 
            color=C_DARKGRAY, lw=0.9, zorder=2)
    # vertical drop from MLP
    arrow(mlp_cx, mlp_bottom_y, mlp_cx, mlp_bottom_y - 0.3, lw=0.9)
    # vertical drops to gamma, beta, alpha
    arrow(gx, mlp_bottom_y - 0.3, gx, 4.6, lw=0.9)
    arrow(bx, mlp_bottom_y - 0.3, bx, 4.6, lw=0.9)
    arrow(alx, mlp_bottom_y - 0.3, alx, 4.6, lw=0.9)

    # === BOTTOM: feature path (horizontal) ===
    y_feat = 1.4
    h_feat = 0.9
    
    # feature h input
    ax.text(0.6, y_feat + h_feat/2, r"$h$", 
            fontsize=9, ha="center", color=C_DARKGRAY, weight="bold")
    
    # Modulate box: h * gamma + beta
    box(1.5, y_feat, 2.8, h_feat, r"$h \odot \gamma + \beta$", fc="#ffffff")
    arrow(1.0, y_feat + h_feat/2, 1.5, y_feat + h_feat/2, color=C_MIDGRAY, lw=1.0)
    
    # gamma -> modulate (from above, into top of Modulate box)
    arrow(gx, 3.9, gx, y_feat + h_feat, color=C_DARKGRAY, lw=0.8)
    # beta -> modulate (from above, into top of Modulate box)
    arrow(bx, 3.9, bx, y_feat + h_feat, color=C_DARKGRAY, lw=0.8)

    # h' 
    ax.text(4.7, y_feat + h_feat/2, r"$h'$", 
            fontsize=9, ha="center", color=C_DARKGRAY, weight="bold")
    arrow(4.3, y_feat + h_feat/2, 4.5, y_feat + h_feat/2, color=C_MIDGRAY, lw=1.0)

    # Gate box: alpha * h'
    box(5.5, y_feat, 2.5, h_feat, r"$\alpha \cdot h'$", fc="#ffffff")
    arrow(4.9, y_feat + h_feat/2, 5.5, y_feat + h_feat/2, color=C_MIDGRAY, lw=1.0)
    
    # alpha -> gate (from above, into top of Gate box)
    arrow(alx, 3.9, alx, y_feat + h_feat, color=C_DARKGRAY, lw=0.8)

    # output
    ax.text(8.9, y_feat + h_feat/2, r"out", 
            fontsize=9, ha="center", color=C_DARKGRAY, weight="bold")
    arrow(8.0, y_feat + h_feat/2, 8.5, y_feat + h_feat/2, color=C_MIDGRAY, lw=1.0)

    # zero-init note at bottom, dark gray
    ax.text(5.0, 0.4, r"zero-init $\Rightarrow$ identity at $t{=}0$",
            fontsize=7.5, ha="center", color=C_DARKGRAY, style="italic")


# -------------------------------------------------------------------------------------
# Panel (c): OT coupling
# -------------------------------------------------------------------------------------
def panel_ot(ax: plt.Axes) -> None:
    # Two-column bipartite graphs: Random vs Stochastic OT
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(c) Coupling: Random vs Stochastic OT", fontsize=10, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)

    rng = np.random.default_rng(7)

    n = 6  # fewer nodes for clarity

    # --- Left sub-panel: Random coupling ---
    left_center = 2.5
    noise_x_l = left_center - 1.4
    gt_x_l = left_center + 1.4
    
    noise_y = np.linspace(1.2, 4.8, n)
    gt_y = np.linspace(1.2, 4.8, n) + rng.normal(0, 0.1, size=n)

    ax.text(left_center, 5.5, "Random", fontsize=9, ha="center", weight="bold", color=C_RAND)
    
    # Draw random coupling lines (many crossings = high entropy)
    perm = rng.permutation(n)
    for i in range(n):
        ax.plot([noise_x_l, gt_x_l], [noise_y[i], gt_y[perm[i]]],
                color=C_RAND, lw=0.8, alpha=0.75, zorder=2)

    # Draw nodes - noise are circles (light), GT are squares (dark)
    for y in noise_y:
        ax.scatter(noise_x_l, y, s=60, c=C_NOISE, edgecolor="k", lw=0.7, 
                   zorder=4, marker="o")
    for y in gt_y:
        ax.scatter(gt_x_l, y, s=60, c=C_GT, edgecolor="k", lw=0.7, 
                   zorder=4, marker="s")

    # Labels
    ax.text(noise_x_l, 0.7, "noise", fontsize=7.5, ha="center", color=C_NOISE, weight="bold")
    ax.text(gt_x_l, 0.7, "GT", fontsize=7.5, ha="center", color=C_GT, weight="bold")
    
    # Entropy formula - larger font at bottom
    ax.text(left_center, 0.15, r"$H(V|X_t)=\log K$", fontsize=9, ha="center", 
            color=C_RAND, weight="bold")

    # --- Right sub-panel: Stochastic OT (Sinkhorn) ---
    right_center = 7.5
    noise_x_r = right_center - 1.4
    gt_x_r = right_center + 1.4

    ax.text(right_center, 5.5, "Stochastic OT", fontsize=9, ha="center", 
            weight="bold", color=C_OT)

    # Sinkhorn transport: mostly nearest-neighbor, with 1-2 longer jumps
    ot_perm = np.arange(n)
    # One swap of neighbors (small stochasticity)
    ot_perm[1], ot_perm[2] = ot_perm[2], ot_perm[1]
    # One slightly longer jump
    ot_perm[5] = 4
    ot_perm[4] = 5
    
    for i in range(n):
        ax.plot([noise_x_r, gt_x_r], [noise_y[i], gt_y[ot_perm[i]]],
                color=C_OT, lw=0.8, alpha=0.75, zorder=2)

    # Draw nodes
    for y in noise_y:
        ax.scatter(noise_x_r, y, s=60, c=C_NOISE, edgecolor="k", lw=0.7, 
                   zorder=4, marker="o")
    for y in gt_y:
        ax.scatter(gt_x_r, y, s=60, c=C_GT, edgecolor="k", lw=0.7, 
                   zorder=4, marker="s")

    # Labels
    ax.text(noise_x_r, 0.7, "noise", fontsize=7.5, ha="center", color=C_NOISE, weight="bold")
    ax.text(gt_x_r, 0.7, "GT", fontsize=7.5, ha="center", color=C_GT, weight="bold")
    
    # Entropy formula - larger font at bottom
    ax.text(right_center, 0.15, r"$0 < H(V|X_t) < \log K$", fontsize=9, ha="center", 
            color=C_OT, weight="bold")


# -------------------------------------------------------------------------------------
# Main figure
# -------------------------------------------------------------------------------------
def main() -> None:
    fig = plt.figure(figsize=(7.0, 2.8), constrained_layout=True)

    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_trajectories(ax_a)
    panel_adaln(ax_b)
    panel_ot(ax_c)

    out_pdf = HERE / "method_overview.pdf"
    out_png = HERE / "method_overview.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
