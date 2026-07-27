"""
Figure 8: Qualitative Detection Comparison - 3x3 Grid per Model.

Each model gets its own 3x3 mosaic (9 patches, seamless).
5 models total: Ground Truth | Ours | DiffusionDet | RTMDet | DINO-R50.

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
BASELINE_DIR = HERE.parent.parent.parent.parent / "experiments" / "analysis" / "baseline_inference_24obj_cache"

PATCH_SIZE = 400
GRID_SIZE = 3

CAT_SHORT = {1:"A1",2:"A2",3:"A3",4:"B4",5:"B5",6:"C6",7:"C7",8:"C8",9:"C9",
             10:"C10",11:"C11",12:"C12",13:"D13",14:"D14",15:"D15",16:"E16",
             17:"E17",18:"E18",19:"F19",20:"F20",21:"G21",22:"G22",23:"X",24:"Y"}

MODELS = [
    {"name": "gt", "title": "Ground Truth", "color": "#2ECC71"},
    {"name": "ours", "title": "KaryoFlow (DPM-Solver++)", "color": "#1E90FF"},
    {"name": "diffusiondet", "title": "DiffusionDet", "color": "#FF6347"},
    {"name": "rtmdet", "title": "RTMDet-L", "color": "#9B59B6"},
    {"name": "dino", "title": "DINO-R50", "color": "#F39C12"},
]


def load_model_preds(model_name):
    if model_name == "ours":
        with open(SOTA_FILE) as f:
            preds = json.load(f)
        result = {}
        for p in preds:
            result.setdefault(p["image_id"], []).append(p)
        return result
    
    if model_name == "diffusiondet":
        pred_file = SOTA_FILE.parent / "24obj_DiffusionDet_seed42_preds.json"
        if pred_file.exists():
            with open(pred_file) as f:
                all_preds = json.load(f)
            result = {}
            for p in all_preds:
                result.setdefault(p["image_id"], []).append(p)
            return result
        return {}
    
    if model_name == "rtmdet":
        p_file = BASELINE_DIR / "RTMDet_L_seed42_preds.json"
        if p_file.exists():
            with open(p_file) as f:
                data = json.load(f)
            preds = data.get("predictions", []) if isinstance(data, dict) else data
            result = {}
            for p in preds:
                result.setdefault(p["image_id"], []).append(p)
            return result
        return {}
    
    if model_name == "dino":
        p_file = BASELINE_DIR / "DINO_R50_seed42_preds.json"
        if p_file.exists():
            with open(p_file) as f:
                preds = json.load(f)
            result = {}
            for p in preds:
                result.setdefault(p["image_id"], []).append(p)
            return result
        return {}
    
    return {}


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
    img_w, img_h = image_pil.size
    
    base_pad = max(1, int(PATCH_SIZE * 0.004))
    border_w = max(1, int(PATCH_SIZE * 0.002))
    box_line_w = max(2, int(PATCH_SIZE * 0.006))
    
    try:
        font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                       max(10, int(PATCH_SIZE * 0.025)))
    except:
        font_tiny = ImageFont.load_default()
    
    placed_labels = []
    
    for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        x, y, w, h = bbox
        
        box_font_size = max(int(min(w, h) * 0.2), 10)
        try:
            box_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                          box_font_size)
        except:
            box_font = font_tiny
        
        draw.rectangle([x, y, x + w, y + h], outline=color, width=box_line_w)
        
        if not show_label or not label:
            continue
        
        is_large_box = w > PATCH_SIZE * 0.1 and h > PATCH_SIZE * 0.08
        
        text_bbox = draw.textbbox((0, 0), label, font=box_font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        if is_large_box:
            avail_w = w - 2 * base_pad
            if text_w > avail_w:
                scale = avail_w / text_w
                new_size = max(int(box_font_size * scale), 8)
                try:
                    box_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", new_size)
                    text_bbox = draw.textbbox((0, 0), label, font=box_font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                except:
                    pass
            
            lx = x + (w - text_w) / 2
            ly = y + (h - text_h) / 2
            
            draw.rectangle([lx - base_pad, ly - base_pad, lx + text_w + base_pad, ly + text_h + base_pad], fill=color)
            draw.text((lx, ly), label, fill='white', font=box_font)
            placed_labels.append([lx - base_pad, ly - base_pad, lx + text_w + base_pad, ly + text_h + base_pad])
        else:
            margin = max(2, int(PATCH_SIZE * 0.006))
            positions = []
            
            above_y = y - text_h - margin - base_pad * 2
            above_x = x + (w - text_w) / 2
            above_x = max(margin, min(img_w - text_w - margin, above_x))
            if above_y >= margin:
                positions.append((above_x, above_y))
            
            below_y = y + h + margin
            below_x = x + (w - text_w) / 2
            below_x = max(margin, min(img_w - text_w - margin, below_x))
            if below_y + text_h + base_pad * 2 <= img_h - margin:
                positions.append((below_x, below_y))
            
            right_x = x + w + margin
            right_y = y + (h - text_h) / 2 - base_pad
            if right_x + text_w + base_pad * 2 <= img_w - margin:
                positions.append((right_x, right_y))
            
            left_x = x - text_w - margin - base_pad * 2
            left_y = y + (h - text_h) / 2 - base_pad
            if left_x >= margin:
                positions.append((left_x, left_y))
            
            if not positions:
                lx = x + (w - text_w) / 2
                ly = y + h + margin
                lx = max(margin, min(img_w - text_w - margin, lx))
                ly = max(margin, min(img_h - text_h - margin, ly))
                positions.append((lx, ly))
            
            for lx, ly in positions:
                label_rect = [lx - base_pad, ly, lx + text_w + base_pad, ly + text_h + base_pad * 2]
                label_rect[0] = max(0, label_rect[0])
                label_rect[1] = max(0, label_rect[1])
                label_rect[2] = min(img_w, label_rect[2])
                label_rect[3] = min(img_h, label_rect[3])
                
                overlap = False
                for pl in placed_labels:
                    if (label_rect[0] < pl[2] and label_rect[2] > pl[0] and
                        label_rect[1] < pl[3] and label_rect[3] > pl[1]):
                        overlap = True
                        break
                
                if not overlap:
                    draw.rectangle(label_rect, fill='white', outline=color, width=border_w)
                    draw.text((lx, ly + base_pad), label, fill=color, font=box_font)
                    placed_labels.append(label_rect)
                    break
    
    return image_pil


def draw_missed_badge(image_pil, color, case_num=None):
    draw = ImageDraw.Draw(image_pil)
    
    big_font_size = max(16, int(PATCH_SIZE * 0.05))
    small_font_size = max(10, int(PATCH_SIZE * 0.03))
    
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", big_font_size)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", small_font_size)
    except:
        font_big = ImageFont.load_default()
        font_sm = font_big
    
    cx = image_pil.width // 2
    cy = image_pil.height // 2
    
    if case_num is not None:
        text = str(case_num)
        tb = draw.textbbox((0, 0), text, font=font_big)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(2, int(PATCH_SIZE * 0.015))
        draw.rectangle([cx - tw // 2 - pad, cy - th // 2 - pad, cx + tw // 2 + pad, cy + th // 2 + pad], fill=color)
        draw.text((cx - tw // 2, cy - th // 2), text, fill='white', font=font_big)
    else:
        text = "MISSED"
        tb = draw.textbbox((0, 0), text, font=font_sm)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(2, int(PATCH_SIZE * 0.015))
        draw.rectangle([cx - tw // 2 - pad, cy - th // 2 - pad, cx + tw // 2 + pad, cy + th // 2 + pad],
                      fill='white', outline=color, width=max(2, int(PATCH_SIZE * 0.004)))
        draw.text((cx - tw // 2, cy - th // 2), text, fill=color, font=font_sm)
    
    return image_pil


def draw_case_number(image_pil, case_num):
    draw = ImageDraw.Draw(image_pil)
    
    font_size = max(12, int(PATCH_SIZE * 0.035))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text = str(case_num)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    
    pad = max(1, int(PATCH_SIZE * 0.008))
    draw.rectangle([pad, pad, pad + tw + pad * 2, pad + th + pad * 2], fill='white', outline='#333333', width=1)
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
        resized = draw_missed_badge(resized, "#E63946")
    
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
    
    all_model_preds = {}
    for model in MODELS:
        if model["name"] != "gt":
            all_model_preds[model["name"]] = load_model_preds(model["name"])
    
    gt_bi = {}
    for ann in gt_data["annotations"]:
        gt_bi.setdefault(ann["image_id"], []).append(ann)
    
    img_map = {img["id"]: img["file_name"] for img in gt_data["images"]}

    cases = [
        {"iid": 1, "cats": [24]},
        {"iid": 2, "cats": [19, 20, 21, 22]},
        {"iid": 3, "cats": [24]},
        {"iid": 4, "cats": [13, 14]},
        {"iid": 5, "cats": [23]},
        {"iid": 6, "cats": [1, 2]},
        {"iid": 7, "cats": [21, 22]},
        {"iid": 8, "cats": [1]},
        {"iid": 9, "cats": [16]},
    ]

    model_patches_dict = {model["name"]: [] for model in MODELS}

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
        
        all_model_targets = {}
        for model_name, preds_bi in all_model_preds.items():
            preds = preds_bi.get(iid, [])
            all_model_targets[model_name] = [p for p in preds if p["score"] > 0.3]

        crop_source = gt_targets if gt_targets else (
            all_model_targets.get("ours", []) or 
            all_model_targets.get("diffusiondet", [])
        )
        crop_region = get_crop_region(crop_source, image.size[::-1])

        gt_bboxes = [ann["bbox"] for ann in gt_targets]
        gt_labels = [CAT_SHORT.get(ann["category_id"], "?") for ann in gt_targets]
        gt_patch = generate_patch(image, crop_region, gt_bboxes, gt_labels, 
                                 MODELS[0]["color"], case_num=case_num)
        model_patches_dict["gt"].append(gt_patch)

        for model in MODELS[1:]:
            model_name = model["name"]
            model_color = model["color"]
            
            model_targets = all_model_targets.get(model_name, [])
            
            model_targets_sorted = sorted(model_targets, key=lambda p: p["score"], reverse=True)[:3]
            model_bboxes = [p["bbox"] for p in model_targets_sorted]
            model_labels = [CAT_SHORT.get(p["category_id"], "?") for p in model_targets_sorted]
            
            has_detections = len(model_targets) > 0
            
            patch = generate_patch(image, crop_region, model_bboxes, model_labels, 
                                   model_color, has_detections=has_detections, case_num=case_num)
            model_patches_dict[model_name].append(patch)

    first_model = MODELS[0]["name"]
    if not model_patches_dict[first_model]:
        print("Error: No valid patches generated!")
        return

    print(f"Generated {len(model_patches_dict[first_model])} patches per model")

    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, model in enumerate(MODELS):
        model_name = model["name"]
        model_title = model["title"]
        model_color = model["color"]
        
        grid = compose_grid(model_patches_dict[model_name])
        
        axes[idx].imshow(grid)
        axes[idx].set_title(model_title, fontsize=12, fontweight='bold', color=model_color, pad=10)
        axes[idx].axis('off')

    fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.04)
    fig.suptitle("Qualitative Comparison: 9 Chromosome Cases", fontsize=14, fontweight='bold', y=0.93)

    save_fig(fig, "qual_mosaic")

    print("Done! Files saved: qual_mosaic.pdf, qual_mosaic.png")


if __name__ == "__main__":
    main()
