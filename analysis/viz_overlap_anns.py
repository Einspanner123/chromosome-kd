#!/usr/bin/env python3
"""
Find images where chromosomes overlap/cross, and visualize the annotation
pattern (full box on upper chromosome vs split boxes on lower chromosome).

Heuristic to find overlap cases:
  - Pairs of bboxes with significant IoU > 0.05 OR
  - Two boxes from the same chromosome category, aligned (head + tail pattern)
    with a "gap" between them and a third box overlapping the gap region

Usage:
    python viz_overlap_anns.py --dataset 24obj
    python viz_overlap_anns.py --dataset autokary

Output: analysis/{dataset}/overlap_anns/overlap_{split}_img{id}_{idx}.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from datasets import get_dataset_path, get_output_dir, get_splits, add_dataset_arg


def load_split(data_dir, split):
    with open(data_dir / split / '_annotations.coco.json') as f:
        return json.load(f)


def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def find_overlap_candidates(coco, top_k=10):
    """Score each image by overlap density; return top_k image_ids."""
    img_to_anns = {}
    for a in coco['annotations']:
        img_to_anns.setdefault(a['image_id'], []).append(a)

    scored = []
    for img_id, anns in img_to_anns.items():
        if len(anns) < 10:
            continue
        n_overlap = 0
        n_overlap_same_cat = 0
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if iou(anns[i]['bbox'], anns[j]['bbox']) > 0.05:
                    n_overlap += 1
                    if anns[i]['category_id'] == anns[j]['category_id']:
                        n_overlap_same_cat += 1
        score = n_overlap + n_overlap_same_cat * 2
        scored.append((score, img_id, n_overlap, n_overlap_same_cat, len(anns)))
    scored.sort(reverse=True)
    return scored[:top_k]


def find_split_pattern(anns):
    """Find candidate 'split annotation' pattern:
    Two boxes from same category with a small gap, AND a third box (different
    category) overlapping the gap region (suggesting the upper chromosome
    occludes the middle of the lower one)."""
    candidates = []
    by_cat = {}
    for a in anns:
        by_cat.setdefault(a['category_id'], []).append(a)

    for cat, cat_anns in by_cat.items():
        if len(cat_anns) < 2:
            continue
        for i in range(len(cat_anns)):
            for j in range(i + 1, len(cat_anns)):
                a1, a2 = cat_anns[i], cat_anns[j]
                x1, y1, w1, h1 = a1['bbox']
                x2, y2, w2, h2 = a2['bbox']
                cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
                cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
                # vertical gap
                if abs(cx1 - cx2) < max(w1, w2) * 0.5 and cy2 > cy1 + h1:
                    gap = (cy2 - (cy1 + h1 / 2)) - h2 / 2
                    if 0 < gap < max(h1, h2) * 1.5:
                        gap_y_min = y1 + h1
                        gap_y_max = y2
                        gap_x_min = min(x1, x2)
                        gap_x_max = max(x1 + w1, x2 + w2)
                        gap_bbox = (gap_x_min, gap_y_min,
                                    gap_x_max - gap_x_min, gap_y_max - gap_y_min)
                        overlapper = None
                        for a in anns:
                            if a['category_id'] == cat:
                                continue
                            if iou(a['bbox'], gap_bbox) > 0.1:
                                overlapper = a
                                break
                        if overlapper:
                            candidates.append({
                                'type': 'vertical',
                                'lower_part1': a1,
                                'lower_part2': a2,
                                'upper': overlapper,
                                'gap_bbox': gap_bbox,
                            })
                # horizontal alignment
                if abs(cy1 - cy2) < max(h1, h2) * 0.5 and cx2 > cx1 + w1:
                    gap = (cx2 - (cx1 + w1 / 2)) - w2 / 2
                    if 0 < gap < max(w1, w2) * 1.5:
                        gap_x_min = x1 + w1
                        gap_x_max = x2
                        gap_y_min = min(y1, y2)
                        gap_y_max = max(y1 + h1, y2 + h2)
                        gap_bbox = (gap_x_min, gap_y_min,
                                    gap_x_max - gap_x_min, gap_y_max - gap_y_min)
                        overlapper = None
                        for a in anns:
                            if a['category_id'] == cat:
                                continue
                            if iou(a['bbox'], gap_bbox) > 0.1:
                                overlapper = a
                                break
                        if overlapper:
                            candidates.append({
                                'type': 'horizontal',
                                'lower_part1': a1,
                                'lower_part2': a2,
                                'upper': overlapper,
                                'gap_bbox': gap_bbox,
                            })
    return candidates


def viz_overlap(data_dir, out_dir, coco, img_map, cat_map, img_id, split,
                candidate, idx):
    """Visualize a candidate split-annotation pattern."""
    im_meta = img_map[img_id]
    img_path = data_dir / split / im_meta['file_name']

    fig, ax = plt.subplots(figsize=(10, 10))
    img = Image.open(img_path)
    ax.imshow(img)

    for a in coco['annotations']:
        if a['image_id'] != img_id:
            continue
        x, y, w, h = a['bbox']
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=0.5, edgecolor='#999999',
            facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        ax.text(x, y - 2, cat_map[a['category_id']],
                fontsize=5, color='#666666', alpha=0.7)

    p1 = candidate['lower_part1']
    p2 = candidate['lower_part2']
    upper = candidate['upper']
    gap = candidate['gap_bbox']

    for ann, label in [(p1, 'part1'), (p2, 'part2')]:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2.5, edgecolor='blue',
            facecolor='blue', alpha=0.2)
        ax.add_patch(rect)
        rect_b = patches.Rectangle(
            (x, y), w, h, linewidth=2.5, edgecolor='blue', facecolor='none')
        ax.add_patch(rect_b)
        ax.text(x + w / 2, y + h / 2, f'{cat_map[ann["category_id"]]}\n{label}',
                fontsize=9, color='blue', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='blue', alpha=0.85))

    x, y, w, h = upper['bbox']
    rect = patches.Rectangle(
        (x, y), w, h, linewidth=2.5, edgecolor='red',
        facecolor='red', alpha=0.2)
    ax.add_patch(rect)
    rect_b = patches.Rectangle(
        (x, y), w, h, linewidth=2.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect_b)
    ax.text(x + w / 2, y + h / 2, f'{cat_map[upper["category_id"]]}\nupper',
            fontsize=9, color='red', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='red', alpha=0.85))

    gx, gy, gw, gh = gap
    if gw > 0 and gh > 0:
        rect = patches.Rectangle(
            (gx, gy), gw, gh, linewidth=1.5, edgecolor='orange',
            facecolor='orange', alpha=0.15, linestyle='--')
        ax.add_patch(rect)
        ax.text(gx + gw / 2, gy + gh / 2, 'gap\n(occluded)',
                fontsize=8, color='darkorange', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='darkorange', alpha=0.8))

    title = (f'[{split}] {im_meta["file_name"]}\n'
             f'Pattern: {candidate["type"]} — '
             f'lower={cat_map[p1["category_id"]]} split into 2 boxes, '
             f'upper={cat_map[upper["category_id"]]} is 1 full box\n'
             f'gap={gw:.0f}×{gh:.0f}px, IoU(p1,gap)={iou(p1["bbox"], gap):.2f}, '
             f'IoU(upper,gap)={iou(upper["bbox"], gap):.2f}')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)

    out_name = f'overlap_{split}_img{img_id}_{idx}.png'
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument('--top-k', type=int, default=15,
                        help='top-K overlap candidate images per split (default: 15)')
    args = parser.parse_args()

    data_dir = get_dataset_path(args.dataset)
    out_dir = get_output_dir(args.dataset, 'overlap_anns')
    splits = get_splits(args.dataset)

    print('=' * 78)
    print('  Searching for split-annotation patterns on overlapping chromosomes')
    print(f'  Dataset: {args.dataset}')
    print(f'  Data:    {data_dir}')
    print(f'  Output:  {out_dir}')
    print('=' * 78)

    total_saved = 0
    for split in splits:
        coco = load_split(data_dir, split)
        img_map = {im['id']: im for im in coco['images']}
        cat_map = {c['id']: c['name'] for c in coco['categories']}

        top_imgs = find_overlap_candidates(coco, top_k=args.top_k)
        print(f'\n[{split}] top overlap candidates:')
        for s, img_id, n_ov, n_ov_same, n_ann in top_imgs[:5]:
            print(f'  img_id={img_id}, score={s}, n_overlap={n_ov}, '
                  f'n_overlap_same_cat={n_ov_same}, n_anns={n_ann}')

        n_split = 0
        for score, img_id, _, _, _ in top_imgs:
            anns = [a for a in coco['annotations'] if a['image_id'] == img_id]
            candidates = find_split_pattern(anns)
            if not candidates:
                continue
            for i, cand in enumerate(candidates[:2]):
                out = viz_overlap(data_dir, out_dir, coco, img_map, cat_map,
                                  img_id, split, cand, i)
                print(f'  saved: {out}')
                total_saved += 1
                n_split += 1
            if n_split >= 4:
                break

    print()
    print('=' * 78)
    print(f'  Total overlap examples visualized: {total_saved}')
    print(f'  Output dir: {out_dir}')
    print('=' * 78)


if __name__ == '__main__':
    main()
