"""Figure 2: localization-aware ranking as an effect-first empirical figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style_v2 import (
    C_FOUNDATION, C_LIGHT, C_LINE, C_LQCR, C_LQCR_LIGHT, C_MUTED,
    C_TEXT, configure_style, panel_title, save_vector_figure,
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
    panel_title(ax, "a", "Where the ranking gain appears")
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
    panel_title(ax, "b", "Fixed-box effect across thresholds")
    thresholds, base_ap = ap_series(base)
    _, lqcr_ap = ap_series(lqcr)
    delta = 100 * (lqcr_ap - base_ap)
    colors = [C_LQCR if value > 0 else C_MUTED for value in delta]
    ax.axhline(0, color=C_LINE, lw=0.8)
    ax.vlines(thresholds, 0, delta, colors=colors, lw=1.5)
    ax.scatter(thresholds, delta, c=colors, s=18, zorder=3,
               edgecolors="white", linewidths=0.35)
    for x, y in zip(thresholds, delta):
        if x in (0.50, 0.75, 0.85, 0.90, 0.95):
            ax.text(x, y + (0.16 if y >= 0 else -0.18), f"{y:+.2f}",
                    ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=5.7, color=C_LQCR if y > 0 else C_MUTED)
    map_delta = 100 * (lqcr["baseline"]["mAP"] - base["baseline"]["mAP"])
    ax.text(0.03, 0.88, f"mAP  +{map_delta:.2f} points", transform=ax.transAxes,
            color=C_LQCR, fontsize=7.0, weight="bold")
    ax.text(0.03, 0.79, "same checkpoint, boxes and classes", transform=ax.transAxes,
            color=C_MUTED, fontsize=5.9)
    ax.set_xlim(0.49, 0.96); ax.set_ylim(-0.45, max(3.75, delta.max() + 0.42))
    ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    ax.set_xlabel("evaluation IoU threshold")
    ax.set_ylabel(r"$\Delta$AP (points)")
    ax.grid(axis="y", color=C_LIGHT, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.2)


def panel_checksum(ax: plt.Axes) -> None:
    panel_title(ax, "c", "Controlled intervention")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rows = [
        ("RF trajectory", "fixed"),
        ("proposal survival", "fixed"),
        ("box coordinates", "fixed"),
        ("class logits", "fixed"),
    ]
    y_values = np.linspace(0.77, 0.47, len(rows))
    ax.text(0.06, 0.89, "BEFORE FINAL RANKING", fontsize=5.8,
            color=C_MUTED, weight="bold")
    for idx, ((name, state), y) in enumerate(zip(rows, y_values)):
        ax.plot([0.07, 0.10], [y, y], color=C_FOUNDATION, lw=2.6,
                solid_capstyle="round")
        ax.text(0.14, y, name, va="center", fontsize=6.4,
                color=C_TEXT)
        ax.text(0.93, y, state, va="center", ha="right", fontsize=6.3,
                color=C_FOUNDATION, weight="bold")
        ax.plot([0.07, 0.93], [y - 0.048, y - 0.048], color=C_LIGHT, lw=0.55)

    # The intervention is a single decision-stage change, not a new trajectory.
    ax.text(0.06, 0.35, "ONLY CHANGED AT OUTPUT", fontsize=5.8,
            color=C_LQCR, weight="bold")
    ax.text(0.07, 0.245, "ranking score", va="center", fontsize=6.5,
            color=C_TEXT, weight="bold")
    ax.text(0.53, 0.245, r"$p$", va="center", ha="center",
            fontsize=8.0, color=C_FOUNDATION)
    ax.annotate("", xy=(0.75, 0.245), xytext=(0.61, 0.245),
                arrowprops=dict(arrowstyle="-|>", color=C_LQCR,
                                lw=1.0, mutation_scale=8))
    ax.text(0.91, 0.245, r"$p q^2$", va="center", ha="right",
            fontsize=8.0, color=C_LQCR, weight="bold")

    ax.text(0.50, 0.085,
            "590 shared tensors identical  |  5 quality-head tensors added",
            ha="center", va="center", fontsize=5.8, color=C_MUTED,
            bbox=dict(boxstyle="round,pad=0.30", fc=C_LIGHT,
                      ec="none", alpha=1.0))


def main() -> None:
    configure_style()
    base = load("source_precision_a4_seed42.json")
    lqcr = load("source_lqcr_final_only_seed42.json")
    fig = plt.figure(figsize=(7.2, 2.55), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.05, 1.02),
                          left=0.055, right=0.985, bottom=0.20, top=0.90,
                          wspace=0.30)
    panel_curves(fig.add_subplot(gs[0, 0]), base, lqcr)
    panel_delta(fig.add_subplot(gs[0, 1]), base, lqcr)
    panel_checksum(fig.add_subplot(gs[0, 2]))
    save_vector_figure(fig, "fig02_lqcr_principle")


if __name__ == "__main__":
    main()
