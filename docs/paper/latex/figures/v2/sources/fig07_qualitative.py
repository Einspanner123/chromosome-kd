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


def draw_scale_response_table(ax: plt.Axes, evidence: dict) -> None:
    ax.axis("off")
    panel_title(ax, "b", "")
    d1 = evidence["scale_response"]["dataset_1"]
    d2 = evidence["scale_response"]["dataset_2"]
    cell_text = [
        ["Small-instance share", f"{100*d1['small_instance_share']:.1f}%", f"{100*d2['small_instance_share']:.1f}%"],
        [r"KaryoFlow $AP_S$", f"{d1['karyoflow_AP_S']:.4f}", f"{d2['karyoflow_AP_S']:.4f}"],
        [r"KaryoFlow+LQCR $AP_S$", f"{d1['lqcr_AP_S']:.4f} ± {d1['lqcr_AP_S_std']:.4f}", f"{d2['lqcr_AP_S']:.4f}"],
        [r"LQCR $\Delta AP_S$", f"{d1['lqcr_delta_AP_S']:+.4f}", f"{d2['lqcr_delta_AP_S']:+.4f}"],
        ["Best comparator", d1["best_comparator"], d2["best_comparator"]],
        [r"Gap to comparator in $AP_S$", f"{d1['gap_to_best_comparator']:+.4f}", f"{d2['gap_to_best_comparator']:+.4f}"],
        ["Overall mAP rank", "1st", "1st"],
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Measure", "Dataset 1", "Dataset 2"],
        colLoc="left",
        cellLoc="left",
        colWidths=[0.46, 0.27, 0.27],
        bbox=[0.0, 0.09, 1.0, 0.77],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.0)
        cell.PAD = 0.08
        if row == 0:
            cell.set_facecolor(C_LIGHT)
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(C_TEXT)
        elif row == 3:
            cell.set_facecolor(C_RF_LIGHT)
            cell.get_text().set_color(C_RF)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#F7F8FA")
            cell.get_text().set_color(C_TEXT)
    ax.text(0.0, 0.015,
            r"$AP_S$ is interpreted within each cohort; Dataset 1 LQCR is the 3-seed mean.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.7, color=C_MUTED)


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
            f"All training boxes: D1 n={len(areas['Dataset 1']):,}; "
            f"D2 n={len(areas['Dataset 2']):,}",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.9, color=C_MUTED)

    draw_scale_response_table(axes[1], evidence)

    save_vector_figure(fig, "fig07_qualitative")


if __name__ == "__main__":
    main()
