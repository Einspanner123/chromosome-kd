"""
Figure 8: Qualitative Detection Comparison - 3x3 Grid per Model.

Each model gets its own 3x3 mosaic (9 patches, seamless).
3 models total: Ground Truth | Ours (A3 DPM++) | Best Baseline.

Run:  python qual_mosaic.py
Outputs: qual_mosaic.pdf, qual_mosaic.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from figure_style import *

DATA_ROOT = HERE.parent.parent.parent.parent / "data"
JEPG_DIR = DATA_ROOT / "24_chromosomes_object" / "JEPG"
ANN_FILE = DATA_ROOT / "24_chromosomes_object" / "coco" / "valid" / "_annotations.coco.json"
SOTA_FILE = HERE.parent.parent.parent.parent / "experiments" / "analysis" / "baseline_vs_sota_cache" / "24obj_SOTA_seed42_preds.json"

PATCH_SIZE = 300
GRID_SIZE = 3

CAT_SHORT = {1:"A1",2:"A2",3:"A3",4:"B4",5:"B5",6:"C6",7:"C7",8:"C8",9:"C9",
             10:"C10",11:"C11",12:"C12",13:"D13",14:"D14",15:"D15",16:"E16",
             17:"E17",18:"E18",19:"F19",20:"F20",21:"G21",22:"G22",23:"X",24:"Y"}

COLORS = {
    'gt': '#2ECC71',
    'ours': '#1E90FF',
    'baseline': '#FF6347',
    'missed': '#E63946',
    'text_bg': 'white',
    'text_fg': 'black',
}


def load_preds(model_name, iid):
    from pathlib import Path
    PRED_DIR = HERE / "baseline_preds"
    
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


def get_crop_region(anns, img_shape, margin_ratio=0.15):
    if not anns:
        cx, cy = img_shape[1] // 2, img_shape[0] // 2
        half = min(img_shape[0], img_shape[1]) // 2
        return cx - half, cx + half, cy - half, cy + half
    
    xs = [a["bbox"][0] for a in anns]
    ys = [a["bbox"][1] for a in anns]
    ws = [a["bbox"][2] for a in anns]
    hs = [a["bbox"][3] for a in anns]
    
    x_min, x_max = min(xs), max(x + w for x, w in zip(xs, ws))
    y_min, y_max = min(ys), max(y + h for y, h in zip(ys, hs))
    
    crop_w, crop_h = x_max - x_min, y_max - y_min
    crop_size = max(crop_w, crop_h)
    margin = int(crop_size * margin_ratio)
    
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = crop_size / 2 + margin
    
    x1 = max(0, int(cx - half))
    x2 = min(img_shape[1], int(cx + half))
    y1 = max(0, int(cy - half))
    y2 = min(img_shape[0], int(cy + half))
    
    return x1, x2, y1, y2


def transform_bbox(bbox, crop_region, target_size=PATCH_SIZE):
    x, y, w, h = bbox
    x1, x2, y1, y2 = crop_region
    
    scale_x = target_size / (x2 - x1)
    scale_y = target_size / (y2 - y1)
    
    return (
        (x - x1) * scale_x,
        (y - y1) * scale_y,
        w * scale_x,
        h * scale_y
    )


def draw_boxes_on_patch(image_pil, bboxes, labels, color, show_label=True):
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for bbox, label in zip(bboxes, labels):
        x, y, w, h = bbox
        
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        if show_label and label:
            text_bbox = draw.textbbox((x, y), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            draw.rectangle(
                [x, y - text_h - 6, x + text_w + 4, y],
                fill=color
            )
            draw.text((x + 2, y - text_h - 4), label, fill='white', font=font)
    
    return image_pil


def draw_missed_badge(image_pil, color, case_num=None):
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    cx = image_pil.width // 2
    cy = image_pil.height // 2
    
    if case_num is not None:
        text = str(case_num)
        text_bbox = draw.textbbox((0, 0), text, font=font_large)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        pad = 8
        draw.rectangle(
            [cx - text_w // 2 - pad, cy - text_h // 2 - pad,
             cx + text_w // 2 + pad, cy + text_h // 2 + pad],
            fill=color
        )
        draw.text((cx - text_w // 2, cy - text_h // 2), text, fill='white', font=font_large)
    else:
        text = "MISSED"
        text_bbox = draw.textbbox((0, 0), text, font=font_small)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        pad = 8
        draw.rectangle(
            [cx - text_w // 2 - pad, cy - text_h // 2 - pad,
             cx + text_w // 2 + pad, cy + text_h // 2 + pad],
            fill='white', outline=color, width=3
        )
        draw.text((cx - text_w // 2, cy - text_h // 2), text, fill=color, font=font_small)
    
    return image_pil


def draw_case_number(image_pil, case_num):
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    text = str(case_num)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    pad = 4
    draw.rectangle(
        [pad, pad, pad + text_w + pad * 2, pad + text_h + pad * 2],
        fill='white', outline='#333333', width=1
    )
    draw.text((pad * 2, pad + 1), text, fill='#333333', font=font)
    
    return image_pil


def generate_patch(image, crop_region, bboxes, labels, color, has_detections=True, case_num=None):
    x1, x2, y1, y2 = crop_region
    cropped = image.crop((x1, y1, x2, y2))
    resized = cropped.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)
    
    if bboxes and has_detections:
        transformed_bboxes = [transform_bbox(bbox, crop_region) for bbox in bboxes]
        resized = draw_boxes_on_patch(resized, transformed_bboxes, labels, color)
    elif not has_detections:
        resized = draw_missed_badge(resized, COLORS['missed'])
    
    if case_num is not None:
        resized = draw_case_number(resized, case_num)
    
    return np.array(resized)


def compose_grid(patches):
    rows = []
    for i in range(0, len(patches), GRID_SIZE):
        row = np.hstack(patches[i:i + GRID_SIZE])
        rows.append(row)
    return np.vstack(rows)


def main() -> None:
    gt_data = json.load(open(ANN_FILE))
    sota_preds = json.load(open(SOTA_FILE))

    sota_bi = {}
    for p in sota_preds:
        sota_bi.setdefault(p["image_id"], []).append(p)
    
    gt_bi = {}
    for ann in gt_data["annotations"]:
        gt_bi.setdefault(ann["image_id"], []).append(ann)
    
    img_map = {img["id"]: img["file_name"] for img in gt_data["images"]}

    cases = [
        {"iid": 1, "cats": [24], "label": "Y (rare)"},
        {"iid": 2, "cats": [19, 20, 21, 22], "label": "F/G (small)"},
        {"iid": 3, "cats": [24], "label": "Y (multi)"},
        {"iid": 4, "cats": [13, 14], "label": "D (medium)"},
        {"iid": 5, "cats": [23], "label": "X (single)"},
        {"iid": 6, "cats": [1, 2], "label": "A1/A2"},
        {"iid": 7, "cats": [21, 22], "label": "G21/G22"},
        {"iid": 8, "cats": [1], "label": "A1 (large)"},
        {"iid": 9, "cats": [16], "label": "E16"},
    ]

    baseline_name = "diffusiondet"

    gt_patches = []
    ours_patches = []
    baseline_patches = []

    for case_idx, case in enumerate(cases):
        iid = case["iid"]
        target_cats = case["cats"]
        case_num = case_idx + 1

        if iid not in img_map:
            print(f"Skipping image {iid}: not found in img_map")
            continue

        img_path = JEPG_DIR / img_map[iid]
        if not img_path.exists():
            print(f"Skipping image {iid}: file not found at {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")

        gt_anns = gt_bi.get(iid, [])
        gt_targets = [a for a in gt_anns if a["category_id"] in target_cats]
        
        ours_preds = [p for p in sota_bi.get(iid, [])
                     if p["category_id"] in target_cats and p["score"] > 0.3]
        
        baseline_preds = load_preds(baseline_name, iid)
        baseline_targets = [p for p in baseline_preds 
                           if p["category_id"] in target_cats and p["score"] > 0.3]

        crop_source = gt_targets if gt_targets else ours_preds
        crop_region = get_crop_region(crop_source, image.size[::-1])

        gt_bboxes = [ann["bbox"] for ann in gt_targets]
        gt_labels = [CAT_SHORT.get(ann["category_id"], "?") for ann in gt_targets]
        
        ours_preds_sorted = sorted(ours_preds, key=lambda p: p["score"], reverse=True)[:3]
        ours_bboxes = [pred["bbox"] for pred in ours_preds_sorted]
        ours_labels = [f"{CAT_SHORT.get(pred['category_id'], '?')} {pred['score']:.2f}" 
                       for pred in ours_preds_sorted]
        
        baseline_sorted = sorted(baseline_targets, key=lambda p: p["score"], reverse=True)[:3]
        baseline_bboxes = [pred["bbox"] for pred in baseline_sorted]
        baseline_labels = [CAT_SHORT.get(pred["category_id"], "?") for pred in baseline_sorted]

        gt_patch = generate_patch(image, crop_region, gt_bboxes, gt_labels, COLORS['gt'], case_num=case_num)
        ours_patch = generate_patch(image, crop_region, ours_bboxes, ours_labels, 
                                     COLORS['ours'], has_detections=len(ours_preds) > 0, case_num=case_num)
        baseline_patch = generate_patch(image, crop_region, baseline_bboxes, baseline_labels,
                                         COLORS['baseline'], has_detections=len(baseline_targets) > 0, case_num=case_num)

        gt_patches.append(gt_patch)
        ours_patches.append(ours_patch)
        baseline_patches.append(baseline_patch)

    if not gt_patches:
        print("Error: No valid patches generated!")
        return

    print(f"Generated {len(gt_patches)} patches per model")

    gt_grid = compose_grid(gt_patches)
    ours_grid = compose_grid(ours_patches)
    baseline_grid = compose_grid(baseline_patches)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    titles = ["Ground Truth", "Ours (A3 DPM++)", "Baseline (DiffusionDet)"]
    colors = ['#2ECC71', '#1E90FF', '#FF6347']
    grids = [gt_grid, ours_grid, baseline_grid]

    for idx, (ax, title, color, grid) in enumerate(zip(axes, titles, colors, grids)):
        ax.imshow(grid)
        ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=15)
        ax.axis('off')

    fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.88, bottom=0.05)

    fig.suptitle("Qualitative Comparison: 9 Chromosome Cases", fontsize=16, fontweight='bold', y=0.95)

    save_fig(fig, "qual_mosaic")

    print("Done! Files saved: qual_mosaic.pdf, qual_mosaic.png")


if __name__ == "__main__":
    main()
