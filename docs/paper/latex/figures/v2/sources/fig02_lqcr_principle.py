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
    ax.text(0.03, 0.88, f"mAP  +{map_delta:.2f} points", transform=ax.transAxes,
            color=C_LQCR, fontsize=7.0, weight="bold")
    ax.text(0.03, 0.79, "same checkpoint, boxes and classes", transform=ax.transAxes,
            color=C_MUTED, fontsize=5.9)
    ax.set_xlim(0.49, 0.96); ax.set_ylim(0.0, max(3.75, delta.max() + 0.42))
    ax.set_xticks([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    ax.set_xlabel("evaluation IoU threshold")
    ax.set_ylabel(r"$\Delta$AP (points)")
    ax.grid(axis="y", color=C_LIGHT, lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.2)


def panel_checksum(ax: plt.Axes) -> None:
    """State the intervention as an aligned operator comparison.

    A compact mathematical contrast is more falsifiable than a pipeline-style
    checklist: it separates held-fixed detector variables from the intervened
    score and from the downstream ordering that is allowed to change.
    """
    panel_title(ax, "c", "Output-only ranking intervention")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Shared detector output (not two independently generated proposal sets).
    ax.text(0.04, 0.855, "HELD FIXED", fontsize=5.7, color=C_MUTED,
            weight="bold")
    ax.text(0.04, 0.765,
            r"$\mathcal{C}=\{(b_i,p_i,q_i)\}_{i=1}^{N}$",
            fontsize=8.0, color=C_TEXT, va="center")
    ax.text(0.04, 0.675,
            r"$x_{0:T},\; b_i,\; p_i$ identical",
            fontsize=6.3, color=C_FOUNDATION, va="center")
    ax.plot([0.04, 0.96], [0.605, 0.605], color=C_LINE, lw=0.75)

    # Aligned counterfactual score definitions.  The central vertical rule is
    # deliberately table-like rather than a decorative flow arrow.
    ax.text(0.04, 0.525, "RANKING OPERATOR", fontsize=5.7, color=C_MUTED,
            weight="bold")
    ax.text(0.30, 0.425, "baseline", ha="center", fontsize=6.1,
            color=C_FOUNDATION, weight="bold")
    ax.text(0.75, 0.425, "LQCR", ha="center", fontsize=6.1,
            color=C_LQCR, weight="bold")
    ax.plot([0.515, 0.515], [0.235, 0.465], color=C_LIGHT, lw=0.8)
    ax.text(0.30, 0.325, r"$s_i=p_i$", ha="center", fontsize=8.2,
            color=C_FOUNDATION)
    ax.text(0.75, 0.325, r"$s_i=p_iq_i^2$", ha="center", fontsize=8.2,
            color=C_LQCR, weight="bold")
    ax.text(0.30, 0.225, r"$\pi_0=\operatorname{argsort}_i\,p_i$",
            ha="center", fontsize=6.7, color=C_TEXT)
    ax.text(0.75, 0.225, r"$\pi_1=\operatorname{argsort}_i\,p_iq_i^2$",
            ha="center", fontsize=6.7, color=C_TEXT)

    ax.plot([0.04, 0.96], [0.155, 0.155], color=C_LINE, lw=0.75)
    ax.text(0.04, 0.095, "CONSEQUENCE", fontsize=5.7, color=C_MUTED,
            weight="bold", va="center")
    ax.text(0.96, 0.035,
            r"different $\pi$ may change the retained subset",
            fontsize=6.15, color=C_TEXT, ha="right", va="center")


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
