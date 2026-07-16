"""Figure 3: OT Diversity Collapse Theory.

Two panels:
  (a) Voronoi partitioning illustration. 2D projection with K=6 GT boxes,
      showing OT (nearest-neighbor / Voronoi) vs Random assignments between
      noise samples and GT boxes.
  (b) Empirical validation of Proposition 1 (Delta H = log K).
      Theoretical log(46.6) = 3.8427 vs empirical 3.8415 (relative error 0.03%).

Run:  python ot_theory.py
Outputs:
  ot_theory.pdf
  ot_theory.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from scipy.spatial import Voronoi as SciVoronoi
from scipy.spatial import voronoi_plot_2d

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)

PAL = sns.color_palette("colorblind")
C_OT = PAL[3]
C_RAND = PAL[1]
C_NOISE = PAL[0]
C_GT = "#222222"
C_VOR = "#9e9e9e"

CELL_FILL_COLORS = [
    (0.95, 0.93, 0.97),
    (0.94, 0.97, 0.94),
    (0.97, 0.96, 0.92),
    (0.93, 0.96, 0.98),
    (0.98, 0.94, 0.94),
    (0.96, 0.95, 0.93),
]

HERE = Path(__file__).resolve().parent

GT_POINTS = np.array(
    [
        [0.8, 1.6],
        [2.2, 0.8],
        [3.5, 2.0],
        [1.5, 2.8],
        [4.8, 1.4],
        [5.5, 2.8],
    ]
)


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
        color = CELL_FILL_COLORS[i % len(CELL_FILL_COLORS)]
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.8, zorder=0)

    voronoi_plot_2d(
        vor, ax=ax, show_points=False, show_vertices=False,
        line_colors=C_VOR, line_width=0.8, line_alpha=0.7,
    )

    ax.scatter(GT_POINTS[:, 0], GT_POINTS[:, 1], s=55, color=C_GT,
               marker="s", edgecolor="k", lw=0.6, zorder=5, label="GT boxes ($b_k$)")
    ax.scatter(noise[:, 0], noise[:, 1], s=30, color=C_NOISE,
               edgecolor="k", lw=0.4, zorder=5, label="noise $z_i$")

    nn_idx = np.argmin(
        np.linalg.norm(noise[:, None, :] - GT_POINTS[None, :, :], axis=2),
        axis=1,
    )
    for z, k in zip(noise, nn_idx):
        ax.plot([z[0], GT_POINTS[k, 0]], [z[1], GT_POINTS[k, 1]],
                color=C_OT, lw=0.9, alpha=0.8, zorder=3)

    rand_perm = rng.permutation(len(GT_POINTS))
    rand_perm = np.resize(rand_perm, len(noise))
    n_rand_show = 4
    rand_indices = rng.choice(len(noise), size=n_rand_show, replace=False)
    for idx in rand_indices:
        z = noise[idx]
        k = rand_perm[idx]
        ax.plot([z[0], GT_POINTS[k, 0]], [z[1], GT_POINTS[k, 1]],
                color=C_RAND, lw=0.8, alpha=0.7, ls="--", zorder=2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("(a) Voronoi partitioning", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=C_OT, lw=1.2, label="OT assignment"),
        plt.Line2D([0], [0], color=C_RAND, lw=1.2, ls="--", label="Random assignment"),
        plt.Line2D([0], [0], color=C_VOR, lw=0.8, label="Voronoi cells"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=True,
                       fontsize=6.5, framealpha=0.7, edgecolor="#cccccc",
                       borderpad=0.5, handletextpad=0.5)
    legend.get_frame().set_facecolor("white")


def panel_dh(ax: plt.Axes) -> None:
    K_mean = 46.6
    theory = np.log(K_mean)
    empirical = 3.8415
    rel_err = abs(theory - empirical) / theory * 100

    bar_labels = [r"$\log K$ (theory)", r"$\Delta H$ (empirical)"]
    values = [theory, empirical]
    colors = [PAL[0], PAL[2]]

    x = np.arange(2)
    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor="black", lw=0.6)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
                f"{v:.4f}", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel(r"Conditional entropy reduction $\Delta H$", fontsize=9)
    ax.set_ylim(0.0, 4.6)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_title(r"(b) $\Delta H$ validation", fontsize=10)

    y_top = max(theory, empirical) + 0.25
    ax.annotate(
        "",
        xy=(0, theory + 0.08), xytext=(1, empirical + 0.08),
        arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8,
                        connectionstyle="arc3,rad=0"),
    )
    ax.text(0.5, y_top,
            f"rel. err. {rel_err:.2f}%",
            ha="center", va="bottom", fontsize=7.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1.2, 1]},
                             constrained_layout=True)
    panel_voronoi(axes[0])
    panel_dh(axes[1])

    out_pdf = HERE / "ot_theory.pdf"
    out_png = HERE / "ot_theory.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
