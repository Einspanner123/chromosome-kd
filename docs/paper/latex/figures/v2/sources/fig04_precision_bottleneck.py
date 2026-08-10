"""Figure 4: empirical localization and ranking bottleneck diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style_v2 import (
    C_LIGHT, C_LQCR, C_MUTED, C_RF, configure_style, panel_title,
    save_vector_figure,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> dict:
    with (DATA / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def panel_oracles(ax: plt.Axes, d1: dict, d2: dict) -> None:
    panel_title(ax, "a", "Oracle headroom")
    labels = ["class", "GT-IoU rank", "perfect boxes"]
    keys = ["classification", "quality_beta_2", "localization"]
    y = np.arange(len(labels)); width = 0.34
    for offset, data, color, name in [
        (-width / 2, d1, C_RF, "Dataset 1"),
        (width / 2, d2, C_LQCR, "Dataset 2"),
    ]:
        gains = [100 * (data["oracles"][k]["mAP"] - data["baseline"]["mAP"])
                 for k in keys]
        ax.barh(y + offset, gains, height=width * 0.78, color=color,
                label=name, alpha=0.90)
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.set_xlabel(r"oracle $\Delta$mAP (points)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.grid(axis="x", color=C_LIGHT, linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=6.3, length=0)


def panel_threshold(ax: plt.Axes, d1: dict, d2: dict) -> None:
    panel_title(ax, "b", "Strict-IoU AP")
    for data, color, name in [(d1, C_RF, "Dataset 1"),
                              (d2, C_LQCR, "Dataset 2")]:
        points = sorted((float(k), v) for k, v in data["baseline"]["AP_by_iou"].items())
        x, y = zip(*points)
        ax.plot(x, y, color=color, marker="o", markersize=2.7,
                linewidth=1.3, label=name)
    ax.set_xlabel("IoU threshold"); ax.set_ylabel("AP"); ax.set_ylim(0, 1.03)
    ax.set_xticks([0.50, 0.70, 0.90, 0.95])
    ax.grid(color=C_LIGHT, linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.3)


def panel_strata(ax: plt.Axes, d1: dict, d2: dict) -> None:
    panel_title(ax, "c", "Recall at IoU 0.90")
    labels = ["small", "medium", "large", "isolated", "near", "overlap"]
    keys = ["small", "medium", "large", "isolated_<0.01",
            "near_0.01-0.20", "overlap_>=0.20"]
    x = np.arange(len(labels)); width = 0.36
    v1 = [d1["stratified_recall"][k]["detection_recall@0.90"] for k in keys]
    v2 = [d2["stratified_recall"][k]["detection_recall@0.90"] for k in keys]
    ax.bar(x - width / 2, v1, width, color=C_RF, label="Dataset 1")
    ax.bar(x + width / 2, v2, width, color=C_LQCR, label="Dataset 2")
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel(r"recall @ IoU $\geq 0.90$"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color=C_LIGHT, linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.2)


def main() -> None:
    configure_style()
    d1 = load("source_precision_chr2024_seed42_100.json")
    d2 = load("source_precision_a4_seed42.json")
    fig = plt.figure(figsize=(7.2, 2.75), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.08, 1.0, 1.15),
                          left=0.055, right=0.985, bottom=0.23, top=0.90,
                          wspace=0.34)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_oracles(axes[0], d1, d2); panel_threshold(axes[1], d1, d2)
    panel_strata(axes[2], d1, d2)
    save_vector_figure(fig, "fig04_precision_bottleneck")


if __name__ == "__main__":
    main()
