"""
Figure 8: Qualitative Detection Comparison - 3x3 Grid.

Nine-patch mosaic: 3 rows (cases) x 3 columns (GT / Ours / Baseline).
Each patch is cropped to target region and resized to uniform 300x300 px.

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

CROP_SIZE = 300
TARGET_SIZE = 300

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


def get_crop_region(anns, img_shape, margin_ratio=0.1):
    if not anns:
        cx, cy = img_shape[1] // 2, img_shape[0] // 2
        half = min(img_shape[0], img_shape[1]) // 2
        return cx - half, cx + half, cy - half, cy + half
    
    xs = [a["bbox"][0] for a in anns]
    ys = [a["bbox"][1] for a in anns]
    ws = [a["bbox"][2] for a in anns]
    hs = [a["bbox"][3] for a in anns]
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x + w for x, w in zip(xs, ws))
    y_max = max(y + h for y, h in zip(ys, hs))
    
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    crop_size = max(crop_w, crop_h)
    
    margin = int(crop_size * margin_ratio)
    
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    half = crop_size / 2 + margin
    
    x1 = max(0, int(cx - half))
    x2 = min(img_shape[1], int(cx + half))
    y1 = max(0, int(cy - half))
    y2 = min(img_shape[0], int(cy + half))
    
    return x1, x2, y1, y2


def crop_and_resize(image, crop_region, target_size=TARGET_SIZE):
    x1, x2, y1, y2 = crop_region
    cropped = image[y1:y2, x1:x2]
    
    pil_img = Image.fromarray(cropped)
    resized = pil_img.resize((target_size, target_size), Image.LANCZOS)
    
    return np.array(resized)


def transform_bbox(bbox, crop_region, target_size=TARGET_SIZE):
    x, y, w, h = bbox
    x1, x2, y1, y2 = crop_region
    
    scale_x = target_size / (x2 - x1)
    scale_y = target_size / (y2 - y1)
    
    new_x = (x - x1) * scale_x
    new_y = (y - y1) * scale_y
    new_w = w * scale_x
    new_h = h * scale_y
    
    return new_x, new_y, new_w, new_h


def draw_patch(ax, image, bboxes, colors, labels=None):
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        color = colors[idx] if idx < len(colors) else C_RF
        
        rect = mpatches.Rectangle(
            (x, y), w, h, 
            linewidth=2.5, edgecolor=color,
            facecolor="none", alpha=0.9, zorder=5
        )
        ax.add_patch(rect)
        
        if labels and idx < len(labels):
            label = labels[idx]
            if h > 20 and w > 30:
                ax.text(x + w/2, y + h/2, label,
                       fontsize=7, color="white", ha="center", va="center",
                       fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.1", fc=color,
                                 ec="none", alpha=0.85), zorder=7)
            elif h > 15 and w > 20:
                ax.text(x + w/2, y, label,
                       fontsize=6, color=color, ha="center", va="bottom",
                       fontweight="bold", zorder=7)


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
            "target_cats": [24], "label": "Y (rare)",
        },
        {
            "iid": 2, "baseline": "cascade_rcnn",
            "baseline_label": "Cascade R-CNN", "baseline_color": "#D55E00",
            "target_cats": [19, 20, 21, 22], "label": "F/G (small)",
        },
        {
            "iid": 7, "baseline": "rtmdet_l",
            "baseline_label": "RTMDet-L", "baseline_color": "#196f7b",
            "target_cats": [21, 22], "label": "G21/G22",
        },
    ]

    n_cases = len(cases)
    n_cols = 3

    fig, axes = plt.subplots(n_cases, n_cols, figsize=(9, 9))
    
    col_titles = ["Ground Truth", "Ours (A3 DPM++)", "Baseline"]
    col_colors = ["#333333", C_RF, "#666666"]
    
    for col_idx, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", 
                                   color=color, pad=8)

    for row_idx, row_def in enumerate(cases):
        iid = row_def["iid"]
        tc = row_def["target_cats"]
        bl_name = row_def["baseline"]
        bl_color = row_def["baseline_color"]

        img_name = img_map.get(iid, f"{iid}.jpg")
        img_path = JEPG_DIR / img_name

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        
        gt_anns = gt_bi.get(iid, [])
        gt_targets = [a for a in gt_anns if a["category_id"] in tc]
        our_preds = [p for p in sota_bi.get(iid, [])
                     if p["category_id"] in tc and p["score"] > 0.3]
        bl_preds = load_preds(bl_name, iid)
        bl_targets = [p for p in bl_preds if p["category_id"] in tc]

        crop_source = gt_targets if gt_targets else our_preds
        crop_region = get_crop_region(crop_source, image.shape[:2])
        image_resized = crop_and_resize(image, crop_region)

        gt_bboxes = []
        gt_labels = []
        gt_colors = []
        for ann in gt_targets:
            bbox = transform_bbox(ann["bbox"], crop_region)
            gt_bboxes.append(bbox)
            gt_labels.append(CAT_SHORT.get(ann["category_id"], "?"))
            gt_colors.append("#2ECC71")

        our_bboxes = []
        our_labels = []
        our_colors = []
        our_preds_sorted = sorted(our_preds, key=lambda p: p["score"], reverse=True)[:3]
        for pred in our_preds_sorted:
            bbox = transform_bbox(pred["bbox"], crop_region)
            our_bboxes.append(bbox)
            our_labels.append(f"{CAT_SHORT.get(pred['category_id'], '?')} {pred['score']:.2f}")
            our_colors.append(C_RF)

        bl_bboxes = []
        bl_labels = []
        bl_colors = []
        for pred in bl_targets[:3]:
            bbox = transform_bbox(pred["bbox"], crop_region)
            bl_bboxes.append(bbox)
            bl_labels.append(CAT_SHORT.get(pred["category_id"], "?"))
            bl_colors.append(bl_color)

        draw_patch(axes[row_idx, 0], image_resized, gt_bboxes, gt_colors, gt_labels)
        draw_patch(axes[row_idx, 1], image_resized, our_bboxes, our_colors, our_labels)
        
        if bl_bboxes:
            draw_patch(axes[row_idx, 2], image_resized, bl_bboxes, bl_colors, bl_labels)
        else:
            axes[row_idx, 2].imshow(image_resized)
            axes[row_idx, 2].set_xticks([])
            axes[row_idx, 2].set_yticks([])
            for spine in axes[row_idx, 2].spines.values():
                spine.set_visible(False)
            axes[row_idx, 2].text(0.5, 0.5, "Missed",
                                 transform=axes[row_idx, 2].transAxes,
                                 fontsize=12, fontweight="bold",
                                 color="#E63946", ha="center", va="center",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                          ec="#E63946", lw=1.5, alpha=0.95))

        axes[row_idx, 0].text(0.05, 0.05, f"{row_idx + 1}. {row_def['label']}",
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=9, fontweight="bold",
                             color="#333333", ha="left", va="bottom",
                             bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec="#333333", lw=0.8, alpha=0.9))

    fig.subplots_adjust(wspace=0.05, hspace=0.15, 
                        left=0.05, right=0.95, 
                        top=0.92, bottom=0.08)

    handles = [
        mpatches.Patch(color="#2ECC71", label="Ground Truth"),
        mpatches.Patch(color=C_RF, label="Ours (detected)"),
        mpatches.Patch(color="#E63946", label="Baseline (missed)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Qualitative Comparison: Ours vs Baselines", 
                fontsize=14, fontweight="bold", y=0.98)

    save_fig(fig, "qual_mosaic")


if __name__ == "__main__":
    main()
