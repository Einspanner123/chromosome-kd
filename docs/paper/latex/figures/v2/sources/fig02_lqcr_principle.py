"""Figure 2: localization-aware ranking as an effect-first empirical figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from figure_style_v2 import (
    C_FOUNDATION, C_LIGHT, C_LINE, C_LQCR, C_LQCR_LIGHT, C_MUTED, C_RF,
    C_TEXT, configure_style, save_vector_figure,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> dict:
    with (DATA / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def ap_series(record: dict) -> tuple[np.ndarray, np.ndarray]:
    values = record["baseline"]["AP_by_iou"]
    thresholds = np.array(sorted(float(key) for key in values))
    return thresholds, np.array([values[f"{value:.2f}"] for value in thresholds])


def panel_curves(ax: plt.Axes, base: dict, lqcr: dict) -> None:
    thresholds, base_ap = ap_series(base)
    _, lqcr_ap = ap_series(lqcr)
    ax.axvspan(0.85, 0.955, color=C_LQCR_LIGHT, alpha=0.72, lw=0)
    ax.plot(thresholds, base_ap, color=C_FOUNDATION, lw=1.45,
            marker="o", ms=2.8, label="class score  $p$")
    ax.plot(thresholds, lqcr_ap, color=C_LQCR, lw=1.65,
            marker="o", ms=2.8, label="quality-aware  $p q^2$")
    # Keep the empirical curves unobstructed. The shaded band communicates the
    # strict-localization regime; exact gains are reported in panel (b).
    ax.text(0.902, 0.975, "strict-IoU regime", color=C_LQCR,
            fontsize=6.0, ha="center", va="top", weight="bold")
    ax.set_xlim(0.49, 0.96); ax.set_ylim(0.15, 1.02)
    ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    ax.set_xlabel("evaluation IoU threshold")
    ax.set_ylabel("AP")
    ax.grid(axis="y", color=C_LIGHT, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.1, loc="lower left")
    ax.tick_params(labelsize=6.2)


def panel_delta(ax: plt.Axes, base: dict, lqcr: dict) -> None:
    thresholds, base_ap = ap_series(base)
    _, lqcr_ap = ap_series(lqcr)
    delta = 100 * (lqcr_ap - base_ap)
    # This panel visualizes positive ranking gains.  Preserve every positive
    # exported delta, however small; omit non-positive values rather than
    # drawing zero-valued markers along the axis.
    positive = delta > 0
    shown_x = thresholds[positive]
    shown_delta = delta[positive]
    ax.vlines(shown_x, 0, shown_delta, colors=C_LQCR, lw=1.5)
    ax.scatter(shown_x, shown_delta, c=C_LQCR, s=18, zorder=3,
               edgecolors="white", linewidths=0.35)
    for x, y in zip(shown_x, shown_delta):
        # Do not over-emphasize sub-0.05-point numerical differences.  Their
        # markers remain visible and the exact exports remain the data source.
        if x in (0.75, 0.85, 0.90, 0.95):
            ax.text(x, y + (0.16 if y >= 0 else -0.18), f"{y:+.2f}",
                    ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=5.7, color=C_LQCR if y > 0 else C_MUTED)
    map_delta = 100 * (lqcr["baseline"]["mAP"] - base["baseline"]["mAP"])
    ax.text(0.03, 0.88, f"mean across thresholds  +{map_delta:.2f} points", transform=ax.transAxes,
            color=C_LQCR, fontsize=7.0, weight="bold")
    ax.text(0.03, 0.79, "same checkpoint, boxes and classes", transform=ax.transAxes,
            color=C_MUTED, fontsize=5.9)
    ax.set_xlim(0.49, 0.96); ax.set_ylim(0.0, max(3.75, delta.max() + 0.42))
    ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    ax.set_xlabel("evaluation IoU threshold")
    ax.set_ylabel(r"$\Delta$AP (percentage points)")
    ax.grid(axis="y", color=C_LIGHT, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.2)


def panel_checksum(ax: plt.Axes) -> None:
    """Show the final-only intervention on one fixed candidate set."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Three fixed candidate thumbnails. A and B localize the same chromosome
    # with different tightness; separating the thumbnails keeps the comparison
    # legible at journal column scale.
    ax.add_patch(Rectangle((0.04, 0.49), 0.92, 0.43, facecolor="#F8FAFC",
                           edgecolor=C_LINE, linewidth=0.7))
    ax.text(0.06, 0.875, "schematic: fixed candidates and class scores", fontsize=5.8,
            color=C_MUTED, weight="bold", va="top")
    candidates = [
        ("A", 0.08, C_FOUNDATION, r"$p=.98,\ q=.62$", 0.025),
        ("B", 0.38, C_LQCR, r"$p=.96,\ q=.91$", 0.045),
        ("C", 0.68, C_RF, r"$p=.95,\ q=.89$", 0.045),
    ]
    for name, x, color, score, inset in candidates:
        ax.add_patch(Rectangle((x, 0.575), 0.23, 0.235, facecolor="white",
                               edgecolor=C_LIGHT, linewidth=0.6))
        ax.plot([x + 0.095, x + 0.135, x + 0.105, x + 0.155],
                [0.60, 0.75, 0.68, 0.79], color="#8D989B", lw=4.0,
                solid_capstyle="round", alpha=0.78)
        ax.plot([x + 0.14, x + 0.09, x + 0.16, x + 0.11],
                [0.60, 0.74, 0.68, 0.79], color="#8D989B", lw=3.4,
                solid_capstyle="round", alpha=0.78)
        ax.add_patch(Rectangle((x + inset, 0.59), 0.23 - 2 * inset, 0.20,
                               fill=False, edgecolor=color, linewidth=1.15))
        ax.text(x + 0.012, 0.795, name, color=color, fontsize=6.1,
                weight="bold", va="top")
        ax.text(x + 0.115, 0.535, score, color=color, fontsize=5.0,
                ha="center", va="center")

    ax.text(0.05, 0.395, "class-only", fontsize=5.8, color=C_FOUNDATION,
            weight="bold", va="center")
    ax.text(0.05, 0.205, "LQCR", fontsize=5.8, color=C_LQCR,
            weight="bold", va="center")
    ax.text(0.05, 0.345, r"rank by $p$", fontsize=5.2, color=C_MUTED)
    ax.text(0.05, 0.155, r"rank by $pq^2$", fontsize=5.2, color=C_MUTED)

    def rank_row(y: float, order: list[str], colors: dict[str, str]) -> None:
        for idx, name in enumerate(order):
            x = 0.34 + idx * 0.18
            selected = idx < 2
            patch = FancyBboxPatch((x, y), 0.13, 0.10,
                                   boxstyle="round,pad=0.006,rounding_size=0.012",
                                   facecolor=colors[name] if selected else "white",
                                   edgecolor=colors[name], linewidth=0.9)
            ax.add_patch(patch)
            ax.text(x + 0.065, y + 0.05, name, ha="center", va="center",
                    fontsize=6.3, weight="bold",
                    color="white" if selected else colors[name])
        ax.text(0.90, y + 0.05, "top-2", fontsize=5.2, color=C_MUTED,
                ha="right", va="center")

    colors = {"A": C_FOUNDATION, "B": C_LQCR, "C": C_RF}
    rank_row(0.325, ["A", "B", "C"], colors)
    rank_row(0.135, ["B", "C", "A"], colors)
    ax.text(0.50, 0.035, "coordinates unchanged; retained subset corrected",
            fontsize=5.45, color=C_TEXT, ha="center", va="center")


def main() -> None:
    configure_style()
    evidence = load("source_lqcr_test_ap_curve.json")
    base = evidence["beta0"]
    lqcr = evidence["beta2"]
    fig = plt.figure(figsize=(7.2, 2.55), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.05, 1.02),
                          left=0.055, right=0.985, bottom=0.20, top=0.90,
                          wspace=0.30)
    axes = [fig.add_subplot(gs[0, index]) for index in range(3)]
    panel_curves(axes[0], base, lqcr)
    panel_delta(axes[1], base, lqcr)
    panel_checksum(axes[2])
    for label, ax in zip(("a", "b", "c"), axes):
        bounds = ax.get_position()
        fig.text(bounds.x0, 0.925, f"({label})", ha="left", va="bottom",
                 fontsize=8.6, weight="bold", color=C_TEXT)
    save_vector_figure(fig, "fig02_lqcr_principle")


if __name__ == "__main__":
    main()
