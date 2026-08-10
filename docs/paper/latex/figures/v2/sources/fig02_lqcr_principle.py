"""Figure 2: LQCR theory, causal boundary, and measured effect."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style_v2 import (
    C_FOUNDATION, C_LIGHT, C_LINE, C_LQCR, C_LQCR_LIGHT, C_MUTED,
    C_RF, C_TEXT, arrow, configure_style, panel_title, rounded_box,
    save_vector_figure,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> dict:
    with (DATA / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def clean(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_principle(ax: plt.Axes) -> None:
    clean(ax); panel_title(ax, "a", "Probability-ranking principle")
    rounded_box(ax, (0.04, 0.69), 0.28, 0.13, "class posterior\n$p=P(C=1\\mid F)$",
                facecolor=C_LIGHT, edgecolor=C_FOUNDATION,
                textcolor=C_FOUNDATION, fontsize=6.5)
    rounded_box(ax, (0.38, 0.69), 0.54, 0.13,
                "localization survival\n$q_\\tau=P(U\\geq\\tau\\mid C=1,F)$",
                facecolor=C_LQCR_LIGHT, edgecolor=C_LQCR,
                textcolor=C_LQCR, fontsize=6.4)
    ax.text(0.35, 0.755, r"$\times$", fontsize=11, ha="center", va="center")
    arrow(ax, (0.50, 0.66), (0.50, 0.56), color=C_LQCR)
    rounded_box(ax, (0.15, 0.39), 0.70, 0.14,
                r"$P(C=1, U\geq\tau\mid F)=p\,q_\tau$",
                facecolor="white", edgecolor=C_LQCR,
                textcolor=C_TEXT, fontsize=7.3, weight="bold")
    ax.text(0.50, 0.28, "COCO averages multiple IoU thresholds",
            ha="center", va="center", fontsize=6.4, color=C_MUTED)
    rounded_box(ax, (0.05, 0.08), 0.90, 0.12, r"low-cost surrogate:  $s=p\,q^2$",
                facecolor=C_LQCR_LIGHT, edgecolor=C_LQCR,
                textcolor=C_LQCR, fontsize=6.5, weight="bold")


def panel_isolation(ax: plt.Axes) -> None:
    clean(ax); panel_title(ax, "b", "Strict final-only causal isolation")
    rounded_box(ax, (0.04, 0.68), 0.28, 0.13, "RF solver",
                facecolor="white", edgecolor=C_RF, textcolor=C_RF,
                fontsize=6.8, weight="bold")
    rounded_box(ax, (0.38, 0.68), 0.28, 0.13, "box renewal",
                facecolor="white", edgecolor=C_FOUNDATION,
                textcolor=C_FOUNDATION, fontsize=6.8)
    rounded_box(ax, (0.72, 0.68), 0.24, 0.13, "Top-K",
                facecolor="white", edgecolor=C_FOUNDATION,
                textcolor=C_FOUNDATION, fontsize=6.8)
    arrow(ax, (0.32, 0.745), (0.38, 0.745), color=C_LINE)
    arrow(ax, (0.66, 0.745), (0.72, 0.745), color=C_LINE)
    ax.text(0.50, 0.58, "raw class score only", ha="center",
            fontsize=6.5, color=C_MUTED)
    rounded_box(ax, (0.03, 0.34), 0.41, 0.13, "last proposal feature",
                facecolor=C_LIGHT, edgecolor=C_FOUNDATION,
                textcolor=C_FOUNDATION, fontsize=6.1)
    rounded_box(ax, (0.55, 0.34), 0.40, 0.13, r"quality head  $q$",
                facecolor=C_LQCR_LIGHT, edgecolor=C_LQCR,
                textcolor=C_LQCR, fontsize=6.7, weight="bold")
    arrow(ax, (0.44, 0.405), (0.55, 0.405), color=C_LQCR)
    arrow(ax, (0.75, 0.33), (0.75, 0.22), color=C_LQCR)
    rounded_box(ax, (0.18, 0.05), 0.77, 0.14,
                r"emitted detections ranked by $p q^2$",
                facecolor="white", edgecolor=C_LQCR,
                textcolor=C_LQCR, fontsize=6.0, weight="bold")
    ax.text(0.03, 0.25, "boxes and classes remain fixed", ha="left",
            va="center", fontsize=5.9, color=C_MUTED)


def panel_effect(ax: plt.Axes, base: dict, lqcr: dict) -> None:
    panel_title(ax, "c", "Measured ranking gain (Dataset 2)")
    thresholds = np.array([0.50, 0.75, 0.85, 0.90, 0.95])
    base_ap = base["baseline"]["AP_by_iou"]
    lqcr_ap = lqcr["baseline"]["AP_by_iou"]
    delta = np.array([lqcr_ap[f"{t:.2f}"] - base_ap[f"{t:.2f}"] for t in thresholds])
    ax.axhline(0, color=C_LINE, linewidth=0.7)
    ax.plot(thresholds, 100 * delta, color=C_LQCR, marker="o",
            markersize=3.8, linewidth=1.4)
    ax.fill_between(thresholds, 0, 100 * delta, color=C_LQCR_LIGHT, alpha=0.9)
    for x, y in zip(thresholds, 100 * delta):
        ax.text(x, y + (0.18 if y >= 0 else -0.22), f"{y:+.2f}",
                ha="center", va="bottom" if y >= 0 else "top",
                fontsize=6.0, color=C_LQCR)
    map_delta = 100 * (lqcr["baseline"]["mAP"] - base["baseline"]["mAP"])
    ax.text(0.03, 0.92, f"mAP  {map_delta:+.2f} points", transform=ax.transAxes,
            fontsize=6.8, weight="bold", color=C_LQCR)
    ax.text(0.03, 0.84, "same boxes/classes; ranking only", transform=ax.transAxes,
            fontsize=6.1, color=C_MUTED)
    ax.set_xlabel("IoU threshold"); ax.set_ylabel(r"$\Delta$AP (points)")
    ax.set_xticks(thresholds, [f"{t:.2f}" for t in thresholds])
    ax.set_ylim(-0.7, max(3.8, 100 * delta.max() + 0.6))
    ax.grid(axis="y", color=C_LIGHT, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.3)


def main() -> None:
    configure_style()
    base = load("source_precision_a4_seed42.json")
    lqcr = load("source_lqcr_final_only_seed42.json")
    fig = plt.figure(figsize=(7.2, 2.75), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.12, 1.0),
                          left=0.025, right=0.985, bottom=0.17, top=0.90,
                          wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_principle(axes[0]); panel_isolation(axes[1])
    panel_effect(axes[2], base, lqcr)
    save_vector_figure(fig, "fig02_lqcr_principle")


if __name__ == "__main__":
    main()
