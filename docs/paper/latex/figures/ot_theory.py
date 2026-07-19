"""
Figure 3: OT Diversity Collapse Theory.

Two panels:
  (a) Voronoi partitioning with K=6 GT boxes in 2D projection.
      OT (nearest-neighbor) and random assignments from noise to GT.
  (b) Empirical validation: theoretical log K vs empirical ΔH.

Run:  python ot_theory.py
Outputs: ot_theory.pdf, ot_theory.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi as SciVoronoi
from scipy.spatial import voronoi_plot_2d

from figure_style import *

# GT box positions (2D projection)
GT_POINTS = np.array([
    [0.8, 1.6],
    [2.2, 0.8],
    [3.5, 2.0],
    [1.5, 2.8],
    [4.8, 1.4],
    [5.5, 2.8],
])

# Softer tint colors for Voronoi cells
CELL_FILLS = [
    (0.93, 0.90, 0.98),
    (0.92, 0.97, 0.92),
    (0.97, 0.95, 0.90),
    (0.90, 0.95, 0.99),
    (0.98, 0.92, 0.92),
    (0.95, 0.93, 0.90),
]


def _clip_voronoi_polygon(vertices, xlim, ylim):
    poly = np.asarray(vertices)
    poly[:, 0] = np.clip(poly[:, 0], xlim[0], xlim[1])
    poly[:, 1] = np.clip(poly[:, 1], ylim[0], ylim[1])
    return poly


def panel_voronoi(ax: plt.Axes) -> None:
    rng = np.random.default_rng(42)
    noise = rng.uniform(0.3, 5.7, size=(8, 2))

    xlim = (-0.3, 6.3)
    ylim = (-0.3, 3.8)

    vor = SciVoronoi(GT_POINTS)

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        vertices = vor.vertices[region]
        poly = _clip_voronoi_polygon(vertices, xlim, ylim)
        ax.fill(poly[:, 0], poly[:, 1],
                color=CELL_FILLS[i % len(CELL_FILLS)], alpha=0.8, zorder=0)

    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                    line_colors=C_VOR if False else "#999999",
                    line_width=0.8, line_alpha=0.7)

    # Re-plot Voronoi boundaries with proper color
    # (voronoi_plot_2d uses its own line_colors; we manually draw)
    for simplex in vor.ridge_vertices:
        if -1 in simplex:
            continue
        v = vor.vertices[simplex]
        ax.plot(v[:, 0], v[:, 1], color="#999999", lw=0.8, alpha=0.7, zorder=1)

    # GT boxes (squares)
    ax.scatter(GT_POINTS[:, 0], GT_POINTS[:, 1], s=70, color=C_GT,
               marker="s", edgecolor="k", lw=0.7, zorder=5,
               label="GT boxes ($b_k$)")

    # Noise samples (circles)
    ax.scatter(noise[:, 0], noise[:, 1], s=40, color=C_SOURCE,
               edgecolor="k", lw=0.5, zorder=5, label="noise $z_i$")

    # OT assignments (solid, full set)
    nn_idx = np.argmin(
        np.linalg.norm(noise[:, None, :] - GT_POINTS[None, :, :], axis=2),
        axis=1,
    )
    for z, k in zip(noise, nn_idx):
        ax.plot([z[0], GT_POINTS[k, 0]], [z[1], GT_POINTS[k, 1]],
                color=C_OT, lw=1.0, alpha=0.85, zorder=3)

    # Random assignments (dashed, subset for clarity)
    rand_perm = rng.permutation(len(GT_POINTS))
    rand_perm = np.resize(rand_perm, len(noise))
    n_rand_show = 4
    rand_indices = rng.choice(len(noise), size=n_rand_show, replace=False)
    for idx in rand_indices:
        z = noise[idx]
        k = rand_perm[idx]
        ax.plot([z[0], GT_POINTS[k, 0]], [z[1], GT_POINTS[k, 1]],
                color=C_RAND, lw=0.9, alpha=0.75, ls="--", zorder=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("(a) Voronoi partitioning", loc="left", fontsize=10,
                 weight="bold", pad=3)
    hide_spines(ax)

    handles = [
        Line2D([0], [0], color=C_OT, lw=1.4, label="OT assignment"),
        Line2D([0], [0], color=C_RAND, lw=1.4, ls="--",
               label="Random assignment"),
        Line2D([0], [0], color="#999999", lw=1.0, label="Voronoi cells"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=True,
                       fontsize=8, framealpha=0.9, edgecolor="#cccccc",
                       borderpad=0.5, handletextpad=0.5)
    legend.get_frame().set_facecolor("white")


def panel_dh(ax: plt.Axes) -> None:
    K_mean = 46.6
    theory = np.log(K_mean)       # 3.8427
    empirical = 3.8415
    rel_err = abs(theory - empirical) / theory * 100

    bar_labels = [r"$\log K$ (theory)", r"$\Delta H$ (empirical)"]
    values = [theory, empirical]
    colors_panel = [C_RF, C_OT]

    x = np.arange(2)
    bars = ax.bar(x, values, width=0.45, color=colors_panel,
                  edgecolor="black", lw=0.7)

    # Inside-bar labels: use contrasting text color
    for bar, v, col in zip(bars, values, colors_panel):
        # Determine text color based on bar brightness
        if isinstance(col, tuple) and len(col) >= 3:
            brightness = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        else:
            brightness = 0.5  # default
        txt_color = "white" if brightness < 0.5 else "black"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", ha="center", va="center",
                fontsize=9, color=txt_color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel(r"Conditional entropy reduction $\Delta H$", fontsize=9)
    ax.set_ylim(0.0, 4.7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_title("(b) Theory vs. empirical", loc="left", fontsize=10,
                 weight="bold", pad=3)
    ax.tick_params(axis="y", labelsize=8)

    # Arrow connecting the two bars
    y_top = max(theory, empirical) + 0.35
    ax.annotate("", xy=(0, theory + 0.1), xytext=(1, empirical + 0.1),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0,
                                connectionstyle="arc3,rad=0"))
    ax.text(0.5, y_top,
            f"rel. err. {rel_err:.2f}%",
            ha="center", va="bottom", fontsize=8, color=C_DARKGRAY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.95))


def main() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1.2, 1]},
        constrained_layout=True,
    )
    panel_voronoi(axes[0])
    panel_dh(axes[1])

    save_fig(fig, "ot_theory")


if __name__ == "__main__":
    main()
