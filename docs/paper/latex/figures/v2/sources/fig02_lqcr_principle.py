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
    ax.text(0.902, 0.36, "strict localization", color=C_LQCR,
            fontsize=6.0, ha="center")
    ax.annotate("+3.25 AP", xy=(0.95, lqcr_ap[-1]), xytext=(0.912, 0.41),
                fontsize=6.3, color=C_LQCR, ha="center",
                arrowprops=dict(arrowstyle="-", color=C_LQCR, lw=0.8))
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
    panel_title(ax, "c", "Causal checksum")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    rows = [
        ("RF trajectory", "unchanged"),
        ("proposal renewal", "unchanged"),
        ("box coordinates", "unchanged"),
        ("predicted classes", "unchanged"),
        ("final ordering", r"$p \rightarrow p q^2$"),
    ]
    y_values = np.linspace(0.75, 0.27, len(rows))
    ax.text(0.06, 0.88, "intervention audit", fontsize=6.1,
            color=C_MUTED, weight="bold")
    for idx, ((name, state), y) in enumerate(zip(rows, y_values)):
        changed = idx == len(rows) - 1
        color = C_LQCR if changed else C_FOUNDATION
        ax.plot([0.07, 0.10], [y, y], color=color, lw=2.6,
                solid_capstyle="round")
        ax.text(0.14, y, name, va="center", fontsize=6.4,
                color=C_TEXT, weight="bold" if changed else "normal")
        ax.text(0.93, y, state, va="center", ha="right", fontsize=6.3,
                color=color, weight="bold" if changed else "normal")
        if idx < len(rows) - 1:
            ax.plot([0.07, 0.93], [y - 0.058, y - 0.058], color=C_LIGHT, lw=0.55)
    ax.text(0.50, 0.09,
            r"$P(C{=}1,U{\geq}\tau\mid F)=p\,q_\tau$"
            "   $\Rightarrow$   cross-threshold surrogate $s=pq^2$",
            ha="center", va="center", fontsize=6.5, color=C_LQCR,
            bbox=dict(boxstyle="round,pad=0.35", fc=C_LQCR_LIGHT,
                      ec="none", alpha=0.88))


def main() -> None:
    configure_style()
    base = load("source_precision_a4_seed42.json")
    lqcr = load("source_lqcr_final_only_seed42.json")
    fig = plt.figure(figsize=(7.2, 2.55), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.05, 0.95),
                          left=0.055, right=0.985, bottom=0.20, top=0.90,
                          wspace=0.28)
    panel_curves(fig.add_subplot(gs[0, 0]), base, lqcr)
    panel_delta(fig.add_subplot(gs[0, 1]), base, lqcr)
    panel_checksum(fig.add_subplot(gs[0, 2]))
    save_vector_figure(fig, "fig02_lqcr_principle")


if __name__ == "__main__":
    main()
