"""Figure 7: cross-dataset scale shift and small-object performance.

The former selected-case mosaic was not an accuracy estimator.  This rebuild
uses all training annotations for the scale distribution and complete-split
COCO AP_S exports for within-dataset detector comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_style_v2 import (
    C_FOUNDATION, C_LIGHT, C_LINE, C_MUTED, C_RF, C_RF_LIGHT, C_TEXT,
    configure_style, panel_title, save_vector_figure,
)


ROOT = Path(__file__).resolve().parents[6]
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "source_small_object_cross_dataset.json"
ANNOTATIONS = {
    "Dataset 1": ROOT / "data/Chromosome20240904_NoAug_NoResize_coco/train/_annotations.coco.json",
    "Dataset 2": ROOT / "data/24_chromosomes_object/coco/train/_annotations.coco.json",
}


def relative_areas(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        coco = json.load(stream)
    images = {int(item["id"]): item for item in coco["images"]}
    values = []
    for ann in coco["annotations"]:
        image = images[int(ann["image_id"])]
        _, _, width, height = ann["bbox"]
        values.append(100.0 * width * height / (image["width"] * image["height"]))
    return np.asarray(values, dtype=float)


def draw_metric_panel(ax: plt.Axes, records: list[dict], title: str,
                      panel: str, note: str) -> None:
    methods = [item["method"] for item in records]
    values = np.asarray([item["AP_S"] for item in records], dtype=float)
    y = np.arange(len(records))[::-1]
    lower = min(values) - 0.035
    upper = max(values) + 0.025

    ax.axvspan(lower, upper, color=C_LIGHT, alpha=0.45, zorder=0)
    for yi, value, method, record in zip(y, values, methods, records):
        ours = method.startswith("KaryoFlow")
        color = C_RF if ours else C_FOUNDATION
        ax.plot([lower, value], [yi, yi], color=color, linewidth=2.4 if ours else 1.4,
                alpha=0.95 if ours else 0.65, zorder=2)
        ax.scatter(value, yi, s=34 if ours else 24, color=color,
                   edgecolor="white", linewidth=0.6, zorder=3)
        if "std" in record:
            ax.errorbar(value, yi, xerr=float(record["std"]), fmt="none",
                        ecolor=color, elinewidth=1.0, capsize=2.2, zorder=2)
        ax.text(value + 0.004, yi, f"{value:.4f}", va="center", ha="left",
                fontsize=6.4, color=color, weight="bold" if ours else "normal")

    ax.set_yticks(y, [m.replace("KaryoFlow ", "KaryoFlow\n") for m in methods])
    ax.set_xlim(lower, upper + 0.018)
    ax.set_xlabel(r"COCO $AP_S$")
    ax.grid(axis="x", color=C_LINE, linewidth=0.45, alpha=0.45)
    ax.tick_params(axis="y", length=0, labelsize=6.4)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    panel_title(ax, panel, title)
    ax.text(0.0, -0.36, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=5.9, color=C_MUTED)


def main() -> None:
    configure_style()
    with DATA_FILE.open(encoding="utf-8") as stream:
        evidence = json.load(stream)

    areas = {name: relative_areas(path) for name, path in ANNOTATIONS.items()}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.70),
                             gridspec_kw={"width_ratios": [1.16, 1.0, 1.0]})
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.34, top=0.86, wspace=0.39)

    ax = axes[0]
    colors = {"Dataset 1": C_RF, "Dataset 2": C_FOUNDATION}
    for name in ("Dataset 1", "Dataset 2"):
        vals = np.sort(areas[name])
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, color=colors[name], linewidth=1.8, label=name)
        median = float(np.median(vals))
        ax.scatter([median], [0.5], s=28, color=colors[name], edgecolor="white",
                   linewidth=0.6, zorder=4)
        ax.text(median, 0.44 if name == "Dataset 1" else 0.56,
                f"median {median:.2f}%", color=colors[name], fontsize=6.2,
                ha="center", va="center", weight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.045, 6.0)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Relative box area (%) - log scale")
    ax.set_ylabel("Empirical CDF")
    ax.grid(color=C_LINE, linewidth=0.45, alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    panel_title(ax, "a", "A 2.3x object-scale shift")
    ax.text(0.0, -0.36,
            f"All training boxes: D1 n={len(areas['Dataset 1']):,}; "
            f"D2 n={len(areas['Dataset 2']):,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.9, color=C_MUTED)

    draw_metric_panel(
        axes[1], evidence["dataset_1_test"], "Small-heavy Dataset 1", "b",
        "220 test images; LQCR is 0.5147 ± 0.0090 over 3 seeds.\nQuality ranking adds +0.0057 $AP_S$.",
    )
    draw_metric_panel(
        axes[2], evidence["dataset_2_validation"], "Larger-scale Dataset 2", "c",
        "500 validation images; LQCR is the fixed seed-42 result.\nIt improves AP_S but remains below DINO R50.",
    )

    save_vector_figure(fig, "fig07_qualitative")


if __name__ == "__main__":
    main()
