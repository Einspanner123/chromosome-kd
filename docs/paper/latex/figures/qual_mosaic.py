"""
Figure 8: Qualitative Detection Comparison — our wins vs multiple baselines.

Each row: one case, comparing our method vs a different baseline.
Only show our winning boxes. Baseline column: blank + miss text.

Row 1: Y chromo   → Ours vs DiffusionDet
Row 2: Dense      → Ours vs Cascade R-CNN
Row 3: Y chromo   → Ours vs YOLOX-S

Each: [Ours: blue box + label]  [Baseline: blank + "missed" text]  [GT: black outline]

Run:  python qual_mosaic.py
Outputs: qual_mosaic.pdf, qual_mosaic.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from figure_style import *

DATA_ROOT = HERE.parent.parent.parent.parent / "data"
JEPG_DIR = DATA_ROOT / "24_chromosomes_object" / "JEPG"
ANN_FILE = DATA_ROOT / "24_chromosomes_object" / "coco" / "valid" / "_annotations.coco.json"
SOTA_FILE = HERE.parent.parent.parent.parent / "experiments" / "analysis" / "baseline_vs_sota_cache" / "24obj_SOTA_seed42_preds.json"

CAT_SHORT = {1:"A1",2:"A2",3:"A3",4:"B4",5:"B5",6:"C6",7:"C7",8:"C8",9:"C9",
             10:"C10",11:"C11",12:"C12",13:"D13",14:"D14",15:"D15",16:"E16",
             17:"E17",18:"E18",19:"F19",20:"F20",21:"G21",22:"G22",23:"X",24:"Y"}

PRED_DIR = HERE / "baseline_preds"


def load_preds(model_name, iid):
    if model_name == "diffusiondet":
        # Full validation set predictions, filter by image_id
        p = PRED_DIR / "diffusiondet_full.json"
        if p.exists():
            all_preds = json.load(open(p))
            return [p for p in all_preds if p["image_id"] == iid]
        return []
    p = PRED_DIR / f"{model_name}_{iid}.json"
    if p.exists():
        return json.load(open(p))
    return []


def zoom_region(gt_anns, img_shape, margin=35):
    if not gt_anns:
        return 0, img_shape[1], img_shape[0], 0
    xs = [a["bbox"][0] for a in gt_anns]
    ys = [a["bbox"][1] for a in gt_anns]
    ws = [a["bbox"][2] for a in gt_anns]
    hs = [a["bbox"][3] for a in gt_anns]
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(img_shape[1], max(x + w for x, w in zip(xs, ws)) + margin)
    y2 = min(img_shape[0], max(y + h for y, h in zip(ys, hs)) + margin)
    return x1, x2, y2, y1


def main() -> None:
    gt = json.load(open(ANN_FILE))
    sota_preds = json.load(open(SOTA_FILE))

    sota_bi = {}
    for p in sota_preds:
        sota_bi.setdefault(p["image_id"], []).append(p)
    gt_bi = {}
    for ann in gt["annotations"]:
        gt_bi.setdefault(ann["image_id"], []).append(ann)
    img_map = {img["id"]: img["file_name"] for img in gt["images"]}

    # Three rows: each compares our method against a different baseline
    rows = [
        {   # Row 1: Ours vs DiffusionDet — Y chromosome
            "iid": 118,
            "baseline": "diffusiondet",  # from existing cache
            "baseline_label": "DiffusionDet",
            "target_cats": [24],
            "miss_note": "Y missed",
            "label": "Y chromosome (rare class)",
        },
        {   # Row 2: Ours vs Cascade R-CNN — dense scene, F-group
            "iid": 278,
            "baseline": "cascade_rcnn",
            "baseline_label": "Cascade R-CNN",
            "target_cats": [19, 20, 21, 22],  # F19–G22 small objects
            "miss_note": "F/G mostly missed",
            "label": "Small objects (F–G group)",
        },
        {   # Row 3: Ours vs RTMDet-L — Y chromosome cross-validation
            "iid": 432,
            "baseline": "rtmdet_l",
            "baseline_label": "RTMDet-L",
            "target_cats": [24],
            "miss_note": "Y missed",
            "label": "Y chromosome (cross-image)",
        },
    ]

    # Baseline colors
    BASELINE_COLORS = {
        "diffusiondet": C_DDPM,
        "cascade_rcnn": "#D55E00",
        "rtmdet_l": "#009E73",
    }

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(5.5, 5.0),
        gridspec_kw={"wspace": 0.08, "hspace": 0.18},
    )

    # Column headers (on row 0 only)
    for j, lbl in enumerate(["Ours (A3 DPM++)", "Baseline"]):
        axes[0, j].set_title(lbl, fontsize=9, weight="bold", pad=5)

    for row_idx, row_def in enumerate(rows):
        iid = row_def["iid"]
        image = np.array(Image.open(JEPG_DIR / img_map[iid]).convert("RGB"))
        tc = row_def["target_cats"]
        baseline_name = row_def["baseline"]
        bl_color = BASELINE_COLORS[baseline_name]

        # Our wins
        our_wins = [p for p in sota_bi.get(iid, [])
                    if p["category_id"] in tc and p["score"] > 0.3]
        # Baseline predictions
        bl_preds = load_preds(baseline_name, iid)
        bl_wins = [p for p in bl_preds
                   if p["category_id"] in tc and p["score"] > 0.3]
        # GT targets
        gt_targets = [a for a in gt_bi.get(iid, [])
                      if a["category_id"] in tc]

        # Zoom region
        zx1, zx2, zy2, zy1 = zoom_region(
            gt_targets if gt_targets else our_wins, image.shape[:2]
        )

        # ---- Column 0: Our method — only winning boxes ----
        ax_our = axes[row_idx, 0]
        ax_our.imshow(image)
        ax_our.set_xticks([]); ax_our.set_yticks([])
        for spine in ax_our.spines.values():
            spine.set_visible(False)
        ax_our.set_xlim(zx1, zx2); ax_our.set_ylim(zy2, zy1)

        for p in our_wins:
            x, y, w, h = p["bbox"]
            cid = p["category_id"]
            s = p["score"]
            name = CAT_SHORT.get(cid, "?")
            rect = mpatches.Rectangle(
                (x, y), w, h, linewidth=2.5, edgecolor=C_RF,
                facecolor="none", alpha=0.9, zorder=5)
            ax_our.add_patch(rect)
            ax_our.text(x, y - 4, f"{name} {s:.2f}", fontsize=8,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc=C_RF,
                                  ec="none", alpha=0.85),
                        ha="left", va="bottom", zorder=6)

        # Row label
        ax_our.text(0.03, 0.97, row_def["label"],
                    transform=ax_our.transAxes, fontsize=7.5,
                    color="white", ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black",
                              ec="none", alpha=0.65))

        # ---- Column 1: Baseline — blank where we win, with note ----
        ax_bl = axes[row_idx, 1]
        ax_bl.imshow(image)
        ax_bl.set_xticks([]); ax_bl.set_yticks([])
        for spine in ax_bl.spines.values():
            spine.set_visible(False)
        ax_bl.set_xlim(zx1, zx2); ax_bl.set_ylim(zy2, zy1)

        # Draw baseline's non-target detections (context only, faint)
        for p in bl_preds:
            if p["category_id"] in tc or p["score"] < 0.3:
                continue
            x, y, w, h = p["bbox"]
            rect = mpatches.Rectangle(
                (x, y), w, h, linewidth=0.5, edgecolor=bl_color,
                facecolor="none", alpha=0.25, zorder=2)
            ax_bl.add_patch(rect)

        # Miss annotation
        if gt_targets:
            mx = sum(a["bbox"][0] + a["bbox"][2] / 2
                     for a in gt_targets) / len(gt_targets)
            my = sum(a["bbox"][1] + a["bbox"][3] / 2
                     for a in gt_targets) / len(gt_targets)
            ax_bl.text(mx, my, row_def["miss_note"], fontsize=8,
                       color="#E63946", weight="bold",
                       ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                 ec="#E63946", lw=0.8, alpha=0.9),
                       zorder=10)

        # Baseline name in top-left of col 1
        ax_bl.text(0.03, 0.97, row_def["baseline_label"],
                   transform=ax_bl.transAxes, fontsize=8,
                   color=bl_color, weight="bold", ha="left", va="top",
                   bbox=dict(boxstyle="round,pad=0.15", fc="white",
                             ec=bl_color, lw=0.6, alpha=0.8))

    # Legend
    handles = [
        mpatches.Patch(color=C_RF, label="Ours (A3 DPM++)"),
        mpatches.Patch(color="#E63946", label="Missed by baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.005))

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04)
    save_fig(fig, "qual_mosaic")


if __name__ == "__main__":
    main()
