"""Figure 7: cross-dataset scale shift and small-object performance.

The former selected-case mosaic was not an accuracy estimator.  This rebuild
uses all held-out test annotations for the scale distribution and complete-split
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
    "Dataset 1": ROOT / "data/Chromosome20240904_NoAug_NoResize_coco/test/_annotations.coco.json",
    "Dataset 2": ROOT / "data/24_chromosomes_object/coco/test/_annotations.coco.json",
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


def draw_scale_response(ax: plt.Axes, evidence: dict) -> None:
    panel_title(ax, "b", "")
    d1 = evidence["scale_response"]["dataset_1"]
    d2 = evidence["scale_response"]["dataset_2"]
    cohorts = [("Dataset 1", d1, 1.0), ("Dataset 2", d2, 0.0)]
    for label, row, y in cohorts:
        x0, x1 = row["karyoflow_AP_S"], row["lqcr_AP_S"]
        ax.plot([x0, x1], [y, y], color=C_LINE, lw=1.5, zorder=1)
        ax.errorbar(x0, y, xerr=row["karyoflow_AP_S_std"], fmt="o",
                    color=C_FOUNDATION, mfc="white", mec=C_FOUNDATION,
                    ms=5.2, capsize=2.2, lw=1.0, zorder=3)
        ax.errorbar(x1, y, xerr=row["lqcr_AP_S_std"], fmt="o",
                    color=C_RF, mfc=C_RF, mec="white", mew=0.45,
                    ms=6.0, capsize=2.2, lw=1.0, zorder=4)
        ax.text(0.472, y + 0.20, label, fontsize=6.5, color=C_TEXT,
                weight="bold", ha="left")
        ax.text(0.472, y + 0.04,
                f"small share {100*row['small_instance_share']:.1f}%   "
                f"$\\Delta AP_S$ {row['lqcr_delta_AP_S']:+.4f}",
                fontsize=5.8, color=C_RF if row['lqcr_delta_AP_S'] > 0 else C_MUTED,
                ha="left")
        comparator = row["best_comparator_AP_S"]
        ax.scatter([comparator], [y - 0.19], marker="D", s=18,
                   color=C_MUTED, edgecolor="white", linewidth=0.4, zorder=4)
        ha = "left" if comparator < 0.58 else "right"
        dx = 0.003 if ha == "left" else -0.003
        ax.text(comparator + dx, y - 0.19,
                f"{row['best_comparator']}: {comparator:.4f}",
                fontsize=5.5, color=C_MUTED, ha=ha, va="center")
    ax.set_xlim(0.47, 0.625); ax.set_ylim(-0.42, 1.42)
    ax.set_yticks([])
    ax.set_xlabel(r"COCO $AP_S$ (mean $\pm$ training-run SD)")
    ax.grid(axis="x", color=C_LIGHT, lw=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.0)
    ax.scatter([], [], marker="o", facecolor="white", edgecolor=C_FOUNDATION,
               label="KaryoFlow")
    ax.scatter([], [], marker="o", color=C_RF, edgecolor="white",
               label="KaryoFlow+LQCR")
    ax.scatter([], [], marker="D", color=C_MUTED, edgecolor="white",
               label="best comparator")
    ax.legend(loc="upper right", frameon=False, fontsize=5.5, ncol=3,
              handletextpad=0.20, columnspacing=0.6)
    ax.text(0.0, -0.25,
            r"Within-cohort comparison; Dataset 2 does not show an $AP_S$ gain.",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.7,
            color=C_MUTED)


def main() -> None:
    configure_style()
    with DATA_FILE.open(encoding="utf-8") as stream:
        evidence = json.load(stream)

    areas = {name: relative_areas(path) for name, path in ANNOTATIONS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.40),
                             gridspec_kw={"width_ratios": [1.0, 1.55]})
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.86, wspace=0.28)

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
    ax.text(0.0, -0.27,
            f"All annotated boxes: D1 n={len(areas['Dataset 1']):,}; "
            f"D2 n={len(areas['Dataset 2']):,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.9, color=C_MUTED)

    draw_scale_response(axes[1], evidence)

    save_vector_figure(fig, "fig07_qualitative")


if __name__ == "__main__":
    main()
