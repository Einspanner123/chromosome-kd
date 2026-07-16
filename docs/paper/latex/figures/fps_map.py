"""Figure 6: Speed-Accuracy Trade-off (FPS vs mAP).

Scatter plot of mAP vs FPS for the 24obj benchmark.

Ours (A1-A3, IO3 variants) colored separately from baselines
(Cascade R-CNN, YOLOX-S, DiffusionDet). All FPS measurements are from
the same hardware (RTX A6000, 512x512, batch=1).

Run:  python fps_map.py
Outputs:
  fps_map.pdf
  fps_map.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

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

C_OURS_BASE = PAL[1]
C_OURS_FAST = PAL[2]
C_BASELINE = PAL[3]
C_PARETO = PAL[0]

HERE = Path(__file__).resolve().parent

DATA = [
    ("A1 RF+Heun",        8.0,   0.856, "Ours-base"),
    ("A2 +StochOT",       7.8,   0.858, "Ours-base"),
    ("A3 DPM-Solver++",  13.3,   0.863, "Ours-fast"),
    ("A3 +IO3 K=300",    14.0,   0.861, "Ours-fast"),
    ("A3 +IO3 K=200",    14.2,   0.860, "Ours-fast"),
    ("A3 +IO3 K=100",    14.3,   0.850, "Ours-fast"),
    ("Cascade R-CNN",    48.4,   0.854, "Baseline"),
    ("YOLOX-S",          98.5,   0.796, "Baseline"),
    ("DiffusionDet",     41.0,   0.787, "Baseline"),
]
TYPE_COLOR = {
    "Ours-base": C_OURS_BASE,
    "Ours-fast": C_OURS_FAST,
    "Baseline": C_BASELINE,
}
TYPE_MARKER = {
    "Ours-base": "o",
    "Ours-fast": "s",
    "Baseline": "^",
}


def compute_pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute Pareto front (maximize both x=FPS and y=mAP)."""
    sorted_pts = sorted(points, key=lambda p: (-p[0], p[1]))
    front: list[tuple[float, float]] = []
    max_y = -float("inf")
    for x, y in sorted_pts:
        if y > max_y:
            front.append((x, y))
            max_y = y
    front.reverse()
    return front


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)

    ours_data = [(n, f, m, t) for n, f, m, t in DATA if t in ("Ours-base", "Ours-fast")]
    baseline_data = [(n, f, m, t) for n, f, m, t in DATA if t == "Baseline"]

    for name, fps, mAP, mtype in DATA:
        ax.scatter(
            fps, mAP, s=65,
            color=TYPE_COLOR[mtype], marker=TYPE_MARKER[mtype],
            edgecolor="k", lw=0.6, zorder=5,
        )

    for name, fps, mAP, mtype in baseline_data:
        if "YOLOX" in name:
            offset = (-6, 5)
            ha = "right"
        elif "Cascade" in name:
            offset = (6, 5)
            ha = "left"
        else:
            offset = (6, -9)
            ha = "left"
        ax.annotate(
            name, xy=(fps, mAP), xytext=offset, textcoords="offset points",
            fontsize=7, ha=ha,
        )

    ours_pts = [(f, m) for _, f, m, t in DATA if t in ("Ours-base", "Ours-fast")]
    pareto_pts = compute_pareto_front(ours_pts)
    if len(pareto_pts) >= 2:
        pf_x = [p[0] for p in pareto_pts]
        pf_y = [p[1] for p in pareto_pts]
        ax.plot(pf_x, pf_y, color=C_PARETO, lw=1.2, ls="--", zorder=3, alpha=0.7)

    ax.text(
        0.98, 0.96,
        "Ours: Pareto front",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color=C_PARETO, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_PARETO,
                  lw=0.6, alpha=0.9),
    )

    ax.set_xlabel("FPS (RTX A6000, 512×512)")
    ax.set_ylabel("mAP (24obj)")
    ax.set_xlim(0, 105)
    ax.set_ylim(0.77, 0.875)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_axisbelow(True)
    ax.grid(ls=":", lw=0.5, alpha=0.5)

    ax_inset = zoomed_inset_axes(ax, zoom=1.7, loc="upper left", borderpad=1.5)

    for name, fps, mAP, mtype in ours_data:
        ax_inset.scatter(
            fps, mAP, s=45,
            color=TYPE_COLOR[mtype], marker=TYPE_MARKER[mtype],
            edgecolor="k", lw=0.6, zorder=5,
        )

    if len(pareto_pts) >= 2:
        pf_x = [p[0] for p in pareto_pts]
        pf_y = [p[1] for p in pareto_pts]
        ax_inset.plot(pf_x, pf_y, color=C_PARETO, lw=1.2, ls="--", zorder=3)

    left_labels = {"A1 RF+Heun": "A1", "A2 +StochOT": "A2"}
    right_labels = {"A3 DPM-Solver++": "A3", "A3 +IO3 K=300": "K300",
                    "A3 +IO3 K=200": "K200", "A3 +IO3 K=100": "K100"}

    left_data = [(n, f, m, t) for n, f, m, t in ours_data if n in left_labels]
    right_data = [(n, f, m, t) for n, f, m, t in ours_data if n in right_labels]
    right_data.sort(key=lambda x: -x[2])

    for i, (name, fps, mAP, mtype) in enumerate(left_data):
        short = left_labels[name]
        y_offset = 8 + i * (-18)
        ax_inset.annotate(
            short, xy=(fps, mAP),
            xytext=(-6, y_offset), textcoords="offset points",
            fontsize=6, ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
        )

    for i, (name, fps, mAP, mtype) in enumerate(right_data):
        short = right_labels[name]
        y_offset = 10 - i * 14
        ax_inset.annotate(
            short, xy=(fps, mAP),
            xytext=(6, y_offset), textcoords="offset points",
            fontsize=6, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
        )

    ax_inset.set_xlim(5.5, 17.5)
    ax_inset.set_ylim(0.840, 0.872)
    ax_inset.set_xticks([8, 12, 16])
    ax_inset.set_yticks([0.845, 0.855, 0.865])
    ax_inset.tick_params(labelsize=6)
    ax_inset.grid(ls=":", lw=0.5, alpha=0.5)
    ax_inset.set_axisbelow(True)

    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8, ls="--")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OURS_BASE,
               markeredgecolor="k", markersize=7, label="Ours-base"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_OURS_FAST,
               markeredgecolor="k", markersize=7, label="Ours-fast"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_BASELINE,
               markeredgecolor="k", markersize=7, label="Baseline"),
        Line2D([0], [0], color=C_PARETO, lw=1.2, ls="--", label="Pareto front"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=7,
    )

    out_pdf = HERE / "fps_map.pdf"
    out_png = HERE / "fps_map.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
