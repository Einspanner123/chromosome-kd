"""
Figure 8: Qualitative Detection Comparison - 6 rows of results.

Extended mosaic showing our method (A3 DPM++) with various cases.
Baseline predictions loaded from available files or marked as unavailable.

Cases (selected for diverse challenges):
  Row 1: Y chromosome (rare class) - Ours vs DiffusionDet
  Row 2: F/G group (small objects) - Ours vs Cascade R-CNN  
  Row 3: Single Y (isolated) - Ours vs RTMDet-L
  Row 4: D chromosomes (medium) - Ours vs DiffusionDet
  Row 5: X chromosome (edge case) - Ours vs Cascade R-CNN
  Row 6: G21/G22 (very small) - Ours vs RTMDet-L

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
        p = PRED_DIR / "diffusiondet_full.json"
        if p.exists():
            try:
                all_preds = json.load(open(p))
                return [pred for pred in all_preds if pred["image_id"] == iid]
            except:
                return []
        return []
    p = PRED_DIR / f"{model_name}_{iid}.json"
    if p.exists():
        try:
            return json.load(open(p))
        except:
            return []
    return []


def zoom_region(gt_anns, img_shape, margin=50):
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


def draw_case_row(ax_our, ax_bl, row_def, image, gt_bi, sota_bi):
    iid = row_def["iid"]
    tc = row_def["target_cats"]
    bl_name = row_def["baseline"]
    bl_color = row_def["baseline_color"]

    our_preds = [p for p in sota_bi.get(iid, [])
                 if p["category_id"] in tc and p["score"] > 0.3]
    bl_preds = load_preds(bl_name, iid)
    gt_targets = [a for a in gt_bi.get(iid, [])
                  if a["category_id"] in tc]

    margin = 80 if len(gt_targets) > 2 else 50
    zoom_x1, zoom_x2, zoom_y2, zoom_y1 = zoom_region(
        gt_targets if gt_targets else our_preds, image.shape[:2], margin=margin
    )

    for ax in [ax_our, ax_bl]:
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(zoom_x1, zoom_x2)
        ax.set_ylim(zoom_y2, zoom_y1)

    our_preds_sorted = sorted(our_preds, key=lambda p: p["score"], reverse=True)
    max_labels = 3 if len(our_preds_sorted) > 3 else len(our_preds_sorted)
    our_preds_filtered = our_preds_sorted[:max_labels]
    
    label_positions = []
    for idx, p in enumerate(our_preds_filtered):
        x, y, w, h = p["bbox"]
        rect = mpatches.Rectangle(
            (x, y), w, h, linewidth=2.5, edgecolor=C_RF,
            facecolor="none", alpha=0.95, zorder=5)
        ax_our.add_patch(rect)

        label = f"{CAT_SHORT.get(p['category_id'], '?')} {p['score']:.2f}"
        
        if h > 18 and w > 30:
            ax_our.text(x + w/2, y + h/2, label,
                       fontsize=7, color="white", ha="center", va="center",
                       fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.12", fc=C_RF,
                                 ec="none", alpha=0.9), zorder=7)
            label_positions.append((x + w/2, y + h/2, 'inside'))
        else:
            lx = x + w/2
            ly = y - 25
            
            overlap = True
            attempts = 0
            min_dist = 40
            while overlap and attempts < 25:
                overlap = False
                for prev_lx, prev_ly, prev_type in label_positions:
                    if prev_type == 'outside':
                        dist = np.sqrt((lx - prev_lx)**2 + (ly - prev_ly)**2)
                        if dist < min_dist:
                            overlap = True
                            break
                if overlap:
                    ly -= 25
                    if attempts % 4 == 3:
                        lx += 25 if idx % 2 == 0 else -25
                    attempts += 1
            
            label_positions.append((lx, ly, 'outside'))
            
            ax_our.annotate(label, xy=(x + w/2, y), xytext=(lx, ly),
                          fontsize=6.5, color=C_RF, ha="center", va="bottom",
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                    ec=C_RF, lw=0.8, alpha=0.95),
                          arrowprops=dict(arrowstyle="-", color=C_RF, lw=0.8,
                                        shrinkA=0, shrinkB=4),
                          zorder=7)
    
    if len(our_preds_sorted) > max_labels:
        ax_our.text(0.95, 0.02, f"+{len(our_preds_sorted) - max_labels} more",
                   transform=ax_our.transAxes, fontsize=6.5,
                   color="#888888", ha="right", va="bottom",
                   style="italic",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white",
                            ec="#CCCCCC", lw=0.5, alpha=0.85))

    for p in bl_preds:
        x, y, w, h = p["bbox"]
        rect = mpatches.Rectangle(
            (x, y), w, h, linewidth=0.8, edgecolor=bl_color,
            facecolor="none", alpha=0.3, zorder=2)
        ax_bl.add_patch(rect)

    if gt_targets and not bl_preds:
        ax_bl.text(0.5, 0.5, "Baseline\npreds\nunavailable",
                  transform=ax_bl.transAxes, fontsize=8,
                  color="#999999", ha="center", va="center",
                  style="italic")
    elif gt_targets:
        mx = sum(a["bbox"][0] + a["bbox"][2] / 2
                 for a in gt_targets) / len(gt_targets)
        my = sum(a["bbox"][1] + a["bbox"][3] / 2
                 for a in gt_targets) / len(gt_targets)
        
        baseline_targets = [p for p in bl_preds if p["category_id"] in tc]
        if not baseline_targets:
            ax_bl.text(mx, my, row_def.get("miss_note", "GT present"), fontsize=8.5,
                      color="#E63946", weight="bold",
                      ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                ec="#E63946", lw=0.8, alpha=0.95),
                      zorder=10)


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

    cases = [
        {
            "iid": 1, "baseline": "diffusiondet",
            "baseline_label": "DiffusionDet", "baseline_color": C_DDPM,
            "target_cats": [24], "miss_note": "Y missed",
            "label": "Y (rare)",
        },
        {
            "iid": 2, "baseline": "cascade_rcnn",
            "baseline_label": "Cascade R-CNN", "baseline_color": "#D55E00",
            "target_cats": [19, 20, 21, 22], "miss_note": "F/G missed",
            "label": "F/G (small)",
        },
        {
            "iid": 3, "baseline": "rtmdet_l",
            "baseline_label": "RTMDet-L", "baseline_color": "#196f7b",
            "target_cats": [24], "miss_note": "Y missed",
            "label": "Y (single)",
        },
        {
            "iid": 4, "baseline": "diffusiondet",
            "baseline_label": "DiffusionDet", "baseline_color": C_DDPM,
            "target_cats": [13, 14], "miss_note": "D missed",
            "label": "D (multi)",
        },
        {
            "iid": 5, "baseline": "cascade_rcnn",
            "baseline_label": "Cascade R-CNN", "baseline_color": "#D55E00",
            "target_cats": [23], "miss_note": "X missed",
            "label": "X (edge)",
        },
        {
            "iid": 7, "baseline": "rtmdet_l",
            "baseline_label": "RTMDet-L", "baseline_color": "#196f7b",
            "target_cats": [21, 22], "miss_note": "G missed",
            "label": "G21/G22",
        },
    ]

    valid_cases = []
    for case in cases:
        iid = case["iid"]
        if iid in img_map and iid in sota_bi:
            img_name = img_map[iid]
            img_path = JEPG_DIR / img_name
            if img_path.exists():
                valid_cases.append(case)

    if len(valid_cases) < len(cases):
        print(f"Found {len(valid_cases)}/{len(cases)} valid cases")
        if len(valid_cases) < 3:
            print("Warning: Using fallback cases")
            for case in cases[:3]:
                valid_cases.append(case)

    n_rows = len(valid_cases)

    fig = plt.figure(figsize=(7.0, 12.5))

    gs = fig.add_gridspec(n_rows, 2, hspace=0.12, wspace=0.06,
                          height_ratios=[1] * n_rows)

    for row_idx, row_def in enumerate(valid_cases):
        ax_our = fig.add_subplot(gs[row_idx, 0])
        ax_bl = fig.add_subplot(gs[row_idx, 1])

        iid = row_def["iid"]
        img_name = img_map[iid] if iid in img_map else f"{iid}.jpg"
        img_path = JEPG_DIR / img_name

        if img_path.exists():
            image = np.array(Image.open(img_path).convert("RGB"))
        else:
            image = np.random.randint(200, 255, (100, 100, 3), dtype=np.uint8)
            print(f"Warning: Image not found: {img_path}")

        draw_case_row(ax_our, ax_bl, row_def, image, gt_bi, sota_bi)

        if row_idx == 0:
            ax_our.set_title("Ours (A3 DPM++)", fontsize=10, weight="bold", pad=4)
            ax_bl.set_title("Baseline", fontsize=10, weight="bold", pad=4)

        ax_our.text(0.03, 0.97, f"{row_idx + 1}",
                   transform=ax_our.transAxes, fontsize=8.5,
                   color="white", ha="left", va="top",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", fc=C_RF,
                             ec="none", alpha=0.8))

        ax_our.text(0.97, 0.97, row_def["label"],
                   transform=ax_our.transAxes, fontsize=8,
                   color=C_RF, ha="right", va="top",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.15", fc="white",
                             ec=C_RF, lw=0.6, alpha=0.9))

        ax_bl.text(0.97, 0.97, row_def["baseline_label"],
                  transform=ax_bl.transAxes, fontsize=8,
                  color=row_def["baseline_color"], ha="right", va="top",
                  fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.15", fc="white",
                            ec=row_def["baseline_color"], lw=0.6, alpha=0.9))

    handles = [
        mpatches.Patch(color=C_RF, label="Ours (detected)"),
        mpatches.Patch(color="#E63946", label="Baseline (missed)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Qualitative Results", fontsize=12, weight="bold",
                 y=1.005)

    save_fig(fig, "qual_mosaic")


if __name__ == "__main__":
    main()
