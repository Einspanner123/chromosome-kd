"""Front-page cross-dataset speed--accuracy evidence figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from figure_style_v2 import (
    C_FOUNDATION, C_LIGHT, C_LQCR, C_MUTED, C_OUTPUT, C_RF, C_TEXT,
    configure_style, panel_title, save_vector_figure,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "source_speed_accuracy_cross_dataset.json"


def draw_panel(ax: plt.Axes, rows: list[dict], metric: str, panel: str,
               ylim: tuple[float, float], offsets: dict[str, tuple[int, int]]) -> None:
    styles = {
        "ours_lqcr": (C_LQCR, "*", 60),
        "ours": (C_RF, "o", 28),
        "standard": (C_FOUNDATION, "^", 25),
        "diffusion": (C_OUTPUT, "D", 26),
    }
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        color, marker, size = styles[row["family"]]
        ax.scatter(row["fps"], value, color=color, marker=marker, s=size,
                   edgecolor="white", linewidth=0.45, zorder=3)
        dx, dy = offsets[row["id"]]
        weight = "bold" if row["family"] == "ours_lqcr" else "normal"
        ax.annotate(row["label"], (row["fps"], value), xytext=(dx, dy),
                    textcoords="offset points", fontsize=5.1, color=color,
                    weight=weight, ha="left" if dx >= 0 else "right")
    ax.set_xscale("log")
    ax.set_xlim(10, 115)
    ax.set_ylim(*ylim)
    ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
    ax.set_ylabel("mAP", fontsize=6.6)
    ax.grid(color=C_LIGHT, linewidth=0.6, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=5.7)
    panel_title(ax, panel, "")


def main() -> None:
    configure_style()
    with DATA.open(encoding="utf-8") as stream:
        rows = json.load(stream)["rows"]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.15), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.965, hspace=0.26)
    d1_offsets = {
        "lqcr": (5, 7), "k500": (5, -11), "dino": (-4, -10),
        "rtmdet": (5, 5), "cascade": (5, 2), "diffusiondet": (-4, -10),
        "yolox": (-5, 4), "k200": (0, 0), "k100": (0, 0),
    }
    d2_offsets = {
        "lqcr": (5, 8), "k500": (5, -9), "k200": (22, 1), "k100": (22, -8),
        "dino": (-4, 5), "rtmdet": (5, -9), "cascade": (5, 2),
        "diffusiondet": (-4, 5), "yolox": (-5, -10),
    }
    draw_panel(axes[0], rows, "d1_mAP", "a", (0.56, 0.765), d1_offsets)
    draw_panel(axes[1], rows, "d2_mAP", "b", (0.775, 0.878), d2_offsets)
    axes[0].text(0.97, 0.90, "Dataset 1", transform=axes[0].transAxes,
                 fontsize=6.2, weight="bold", color=C_TEXT, ha="right")
    axes[1].text(0.97, 0.90, "Dataset 2", transform=axes[1].transAxes,
                 fontsize=6.2, weight="bold", color=C_TEXT, ha="right")
    axes[1].set_xlabel("model-forward FPS (RTX A6000, 512 x 512, batch 1)",
                       fontsize=6.2)
    save_vector_figure(fig, "fig00_speed_accuracy")


if __name__ == "__main__":
    main()
