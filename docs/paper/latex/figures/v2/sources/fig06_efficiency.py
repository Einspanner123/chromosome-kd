"""Figure 6: auditable RTX A6000 latency and speed--accuracy trade-off."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style_v2 import (
    C_FOUNDATION, C_LIGHT, C_LQCR, C_MUTED, C_OUTPUT, C_RF,
    configure_style, panel_title, save_vector_figure,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> list[dict]:
    with (DATA / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def by_name(rows: list[dict]) -> dict[str, dict]:
    return {row["name"]: row for row in rows}


def panel_latency(ax: plt.Axes, rows: dict[str, dict]) -> None:
    panel_title(ax, "a", "Where the latency is removed")
    names = ["a1", "a4", "a4_io3_k200"]
    labels = ["Heun\n7 NFE", "DPM++\n4 NFE", "DPM++ + Top-K\n4 NFE, K=200"]
    x = np.arange(len(names))
    backbone = np.array([rows[name]["backbone_neck_ms"] for name in names])
    head = np.array([rows[name]["head_ms"] for name in names])
    ax.bar(x, backbone, color=C_FOUNDATION, width=0.58, label="backbone + neck")
    ax.bar(x, head, bottom=backbone, color=C_RF, width=0.58, label="iterative head")
    for xpos, name in zip(x, names):
        total = rows[name]["total_ms"]
        ax.text(xpos, total + 3.2, f"{total:.1f} ms\n{rows[name]['fps']:.1f} FPS",
                ha="center", va="bottom", fontsize=6.1, color=C_MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylabel("latency (ms / image)")
    ax.set_ylim(0, max((rows[name]["total_ms"] for name in names)) * 1.25)
    ax.grid(axis="y", color=C_LIGHT, linewidth=0.7)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.3)


def panel_frontier(ax: plt.Axes, rows: dict[str, dict]) -> None:
    panel_title(ax, "b", "Speed--accuracy trade-off")
    ours = ["a4", "a4_io3_k300", "a4_io3_k200", "a4_io3_k100"]
    standards = ["cascade_rcnn", "yolox_s", "dino_r50", "rtmdet_l"]
    diffusion = ["diffusiondet"]
    calibrated = ["lqcr"]

    for names, color, marker, label in [
        (ours, C_RF, "o", "RF + DPM++ variants"),
        (standards, C_FOUNDATION, "^", "standard detectors"),
        (diffusion, C_OUTPUT, "D", "DiffusionDet"),
        (calibrated, C_LQCR, "*", "KaryoFlow + LQCR"),
    ]:
        ax.scatter([rows[n]["fps"] for n in names],
                   [rows[n]["known_mAP"] for n in names],
                   color=color, marker=marker, s=58 if label == "KaryoFlow + LQCR" else 35, edgecolor="white",
                   linewidth=0.6, label=label, zorder=4)

    # Connect only the controlled Top-K sweep from the same A4 checkpoint.
    ax.plot([rows[n]["fps"] for n in ours],
            [rows[n]["known_mAP"] for n in ours],
            color=C_RF, linewidth=0.9, alpha=0.75, zorder=2)

    short = {
        "a4": "K=500", "a4_io3_k300": "K=300", "a4_io3_k200": "K=200",
        "a4_io3_k100": "K=100", "cascade_rcnn": "Cascade R-CNN",
        "yolox_s": "YOLOX-S", "dino_r50": "DINO R50",
        "rtmdet_l": "RTMDet-L",
        "diffusiondet": "DiffusionDet",
        "lqcr": "KaryoFlow + LQCR",
    }
    offsets = {
        "a4": (-12, -12), "a4_io3_k300": (4, 8), "a4_io3_k200": (5, -10),
        "a4_io3_k100": (5, -11), "cascade_rcnn": (4, -10),
        "yolox_s": (-23, 7), "dino_r50": (4, 7),
        "rtmdet_l": (4, -10), "diffusiondet": (4, 7),
        "lqcr": (6, 9),
    }
    for name in ours + standards + diffusion:
        ax.annotate(short[name], (rows[name]["fps"], rows[name]["known_mAP"]),
                    xytext=offsets[name], textcoords="offset points",
                    fontsize=5.7, color=C_MUTED)
    ax.set_xscale("log")
    ax.set_xlim(10, 120)
    ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
    ax.set_ylim(0.780, 0.875)
    ax.set_xlabel("model-forward FPS (RTX A6000, 512 x 512, batch 1)")
    ax.set_ylabel("validation mAP")
    ax.grid(color=C_LIGHT, linewidth=0.7, which="both")
    ax.legend(frameon=False, fontsize=6.1, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.3)


def main() -> None:
    configure_style()
    rows = load("source_fps_a6000_karyoflow.json") + load("source_fps_a6000_dino.json")
    indexed = by_name(rows)
    fig = plt.figure(figsize=(7.2, 2.85), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.22),
                          left=0.075, right=0.985, bottom=0.23, top=0.88,
                          wspace=0.25)
    panel_latency(fig.add_subplot(gs[0, 0]), indexed)
    panel_frontier(fig.add_subplot(gs[0, 1]), indexed)
    save_vector_figure(fig, "fig06_efficiency")


if __name__ == "__main__":
    main()
