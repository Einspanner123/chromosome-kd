#!/usr/bin/env python3
"""
Detect and quantify 'split annotation' patterns in a dataset.

Split pattern: one chromosome split into 2 bboxes (head + tail) due to
occlusion by another chromosome.

Detection criteria (must satisfy ALL):
  1. Two boxes (a1, a2) of same category_id in same image
  2. Their long axes are collinear (vertical or horizontal alignment)
  3. The gap between them is small relative to the larger box's long side
  4. A third box (different category) covers most of the gap region (the occluder)
  5. The two same-cat boxes do NOT overlap with each other (gap > 0)
  6. After excluding duplicates of (a1, a2), each is counted once

Usage:
    python detect_split_anns.py --dataset 24obj
    python detect_split_anns.py --dataset autokary

Outputs (analysis/{dataset}/split_detection/):
  - Statistics: # of split patterns per split, % of images affected
  - Sample visualization (top-K candidates)
  - Suggested merge targets: list of (ann_id_1, ann_id_2, proposed_merged_bbox)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from datasets import get_dataset_path, get_output_dir, get_splits, add_dataset_arg

# Tunable thresholds
GAP_RATIO_MAX = 1.5          # gap must be < 1.5× max long side of (a1, a2)
ALIGN_TOL_RATIO = 0.6        # centers' offset on short axis < 0.6× max short side
OCCLUDER_IOU_MIN = 0.15      # occluder bbox must overlap gap region with IoU >= 0.15
OCCLUDER_COVERAGE_MIN = 0.3  # occluder must cover >=30% of gap area


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


def coverage(small, big):
    """How much of 'big' is covered by 'small' (intersection / big area)."""
    x1, y1, w1, h1 = small
    x2, y2, w2, h2 = big
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    big_area = w2 * h2
    return inter / big_area if big_area > 0 else 0


def detect_split_pattern(anns):
    """For each pair of same-category bboxes, check if they form a split pattern."""
    by_cat = {}
    for a in anns:
        by_cat.setdefault(a['category_id'], []).append(a)

    candidates = []
    used_pairs = set()

    for cat, cat_anns in by_cat.items():
        if len(cat_anns) < 2:
            continue
        for i in range(len(cat_anns)):
            for j in range(i + 1, len(cat_anns)):
                a1, a2 = cat_anns[i], cat_anns[j]
                pair_key = (a1['id'], a2['id'])
                if pair_key in used_pairs:
                    continue

                x1, y1, w1, h1 = a1['bbox']
                x2, y2, w2, h2 = a2['bbox']

                inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                if inter_x > 0 and inter_y > 0:
                    continue  # they overlap, not a split pattern

                gap_bbox = None
                max_long_side = 0

                # Vertical case: a1 above a2
                if y1 + h1 <= y2:
                    gap_y = y2 - (y1 + h1)
                    cx1, cx2 = x1 + w1 / 2, x2 + w2 / 2
                    x_offset = abs(cx1 - cx2)
                    max_short_side = max(min(w1, h1), min(w2, h2))
                    if x_offset > max_short_side * ALIGN_TOL_RATIO:
                        continue
                    gap_x_min = min(x1, x2)
                    gap_x_max = max(x1 + w1, x2 + w2)
                    gap_bbox = (gap_x_min, y1 + h1, gap_x_max - gap_x_min, gap_y)
                    max_long_side = max(h1, h2)
                    if gap_y > max_long_side * GAP_RATIO_MAX:
                        continue
                elif y2 + h2 <= y1:  # a2 above a1
                    gap_y = y1 - (y2 + h2)
                    cx1, cx2 = x1 + w1 / 2, x2 + w2 / 2
                    x_offset = abs(cx1 - cx2)
                    max_short_side = max(min(w1, h1), min(w2, h2))
                    if x_offset > max_short_side * ALIGN_TOL_RATIO:
                        continue
                    gap_x_min = min(x1, x2)
                    gap_x_max = max(x1 + w1, x2 + w2)
                    gap_bbox = (gap_x_min, y2 + h2, gap_x_max - gap_x_min, gap_y)
                    max_long_side = max(h1, h2)
                    if gap_y > max_long_side * GAP_RATIO_MAX:
                        continue
                elif x1 + w1 <= x2:  # Horizontal: a1 left of a2
                    gap_x = x2 - (x1 + w1)
                    cy1, cy2 = y1 + h1 / 2, y2 + h2 / 2
                    y_offset = abs(cy1 - cy2)
                    max_short_side = max(min(w1, h1), min(w2, h2))
                    if y_offset > max_short_side * ALIGN_TOL_RATIO:
                        continue
                    gap_y_min = min(y1, y2)
                    gap_y_max = max(y1 + h1, y2 + h2)
                    gap_bbox = (x1 + w1, gap_y_min, gap_x, gap_y_max - gap_y_min)
                    max_long_side = max(w1, w2)
                    if gap_x > max_long_side * GAP_RATIO_MAX:
                        continue
                elif x2 + w2 <= x1:
                    gap_x = x1 - (x2 + w2)
                    cy1, cy2 = y1 + h1 / 2, y2 + h2 / 2
                    y_offset = abs(cy1 - cy2)
                    max_short_side = max(min(w1, h1), min(w2, h2))
                    if y_offset > max_short_side * ALIGN_TOL_RATIO:
                        continue
                    gap_y_min = min(y1, y2)
                    gap_y_max = max(y1 + h1, y2 + h2)
                    gap_bbox = (x2 + w2, gap_y_min, gap_x, gap_y_max - gap_y_min)
                    max_long_side = max(w1, w2)
                    if gap_x > max_long_side * GAP_RATIO_MAX:
                        continue
                else:
                    continue

                gap_area = gap_bbox[2] * gap_bbox[3]
                if gap_area <= 0:
                    continue

                # Look for occluder
                occluder = None
                for a in anns:
                    if a['category_id'] == cat:
                        continue
                    if a['id'] == a1['id'] or a['id'] == a2['id']:
                        continue
                    if (coverage(a['bbox'], gap_bbox) >= OCCLUDER_COVERAGE_MIN
                            and iou(a['bbox'], gap_bbox) >= OCCLUDER_IOU_MIN):
                        occluder = a
                        break
                if occluder is None:
                    continue

                mx = min(x1, x2)
                my = min(y1, y2)
                mw = max(x1 + w1, x2 + w2) - mx
                mh = max(y1 + h1, y2 + h2) - my

                candidates.append({
                    'category_id': cat,
                    'ann1': a1,
                    'ann2': a2,
                    'occluder': occluder,
                    'gap_bbox': gap_bbox,
                    'merged_bbox': (mx, my, mw, mh),
                })
                used_pairs.add(pair_key)

    return candidates


def viz_candidate(data_dir, out_dir, coco, img_map, cat_map, cand, split, idx):
    """Visualize a detected split pattern with proposed merge."""
    a1 = cand['ann1']
    a2 = cand['ann2']
    occluder = cand['occluder']
    merged = cand['merged_bbox']

    im_meta = img_map[a1['image_id']]
    img_path = data_dir / split / im_meta['file_name']

    fig, ax = plt.subplots(figsize=(10, 10))
    img = Image.open(img_path)
    ax.imshow(img)

    for a in coco['annotations']:
        if a['image_id'] != a1['image_id']:
            continue
        if a['id'] in (a1['id'], a2['id'], occluder['id']):
            continue
        x, y, w, h = a['bbox']
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=0.4, edgecolor='#999999',
            facecolor='none', alpha=0.4)
        ax.add_patch(rect)
        ax.text(x, y - 2, cat_map[a['category_id']],
                fontsize=5, color='#666666', alpha=0.7)

    for ann, label in [(a1, 'part1'), (a2, 'part2')]:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='blue',
            facecolor='blue', alpha=0.15)
        ax.add_patch(rect)
        rect_b = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect_b)
        ax.text(x + w / 2, y + h / 2, f'{cat_map[ann["category_id"]]}\n{label}',
                fontsize=8, color='blue', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='blue', alpha=0.85))

    x, y, w, h = occluder['bbox']
    rect = patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor='red',
        facecolor='red', alpha=0.15)
    ax.add_patch(rect)
    rect_b = patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect_b)
    ax.text(x + w / 2, y + h / 2, f'{cat_map[occluder["category_id"]]}\noccluder',
            fontsize=8, color='red', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='red', alpha=0.85))

    gx, gy, gw, gh = cand['gap_bbox']
    rect = patches.Rectangle(
        (gx, gy), gw, gh, linewidth=1.2, edgecolor='orange',
        facecolor='orange', alpha=0.1, linestyle='--')
    ax.add_patch(rect)

    mx, my, mw, mh = merged
    rect = patches.Rectangle(
        (mx, my), mw, mh, linewidth=2.5, edgecolor='green',
        facecolor='green', alpha=0.05, linestyle='--')
    ax.add_patch(rect)
    rect_b = patches.Rectangle(
        (mx, my), mw, mh, linewidth=2.5, edgecolor='green',
        facecolor='none', linestyle='--')
    ax.add_patch(rect_b)
    ax.text(mx, my - 4, f'PROPOSED MERGE: {cat_map[cand["category_id"]]}',
            fontsize=9, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='green', alpha=0.9))

    title = (f'[{split}] {im_meta["file_name"]}  '
             f'cat={cat_map[cand["category_id"]]}  '
             f'ann_ids=({a1["id"]},{a2["id"]})  occluder={occluder["id"]}\n'
             f'part1=({a1["bbox"][0]:.0f},{a1["bbox"][1]:.0f},'
             f'{a1["bbox"][2]:.0f},{a1["bbox"][3]:.0f})  '
             f'part2=({a2["bbox"][0]:.0f},{a2["bbox"][1]:.0f},'
             f'{a2["bbox"][2]:.0f},{a2["bbox"][3]:.0f})\n'
             f'merged=({mx:.0f},{my:.0f},{mw:.0f},{mh:.0f})  '
             f'area={mw*mh:.0f}')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(alpha=0.2)

    out_name = f'split_{split}_img{a1["image_id"]}_cat{cand["category_id"]}_{idx}.png'
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument('--top-k', type=int, default=8,
                        help='top-K visualizations per split (default: 8)')
    args = parser.parse_args()

    data_dir = get_dataset_path(args.dataset)
    out_dir = get_output_dir(args.dataset, 'split_detection')
    splits = get_splits(args.dataset)

    print('=' * 78)
    print('  Detect split-annotation patterns (with occluder verification)')
    print(f'  Dataset: {args.dataset}')
    print(f'  Data:    {data_dir}')
    print(f'  Output:  {out_dir}')
    print('=' * 78)

    all_candidates_per_split = {}
    for split in splits:
        coco = load_split(data_dir, split)
        img_map = {im['id']: im for im in coco['images']}
        cat_map = {c['id']: c['name'] for c in coco['categories']}

        img_to_anns = {}
        for a in coco['annotations']:
            img_to_anns.setdefault(a['image_id'], []).append(a)

        all_candidates = []
        for img_id, anns in img_to_anns.items():
            cands = detect_split_pattern(anns)
            for c in cands:
                c['img_id'] = img_id
                all_candidates.append(c)

        all_candidates_per_split[split] = (all_candidates, coco, img_map, cat_map)
        print(f'\n[{split}]')
        print(f'  total images: {len(img_to_anns)}')
        print(f'  detected split patterns: {len(all_candidates)}')
        n_imgs_affected = len(set(c['img_id'] for c in all_candidates))
        print(f'  images affected: {n_imgs_affected} '
              f'({n_imgs_affected/len(img_to_anns)*100:.1f}%)')

        cat_counter = Counter(c['category_id'] for c in all_candidates)
        print(f'  top categories with split patterns:')
        for cat_id, n in cat_counter.most_common(5):
            print(f'    {cat_map[cat_id]:>4s}: {n}')

    for split, (cands, coco, img_map, cat_map) in all_candidates_per_split.items():
        print(f'\n[{split}] saving top {args.top_k} visualizations...')
        cands_sorted = sorted(cands, key=lambda c: c['gap_bbox'][2] * c['gap_bbox'][3])
        for i, c in enumerate(cands_sorted[:args.top_k]):
            out = viz_candidate(data_dir, out_dir, coco, img_map, cat_map,
                                c, split, i)
            print(f'  saved: {out}')

    print('\nWriting merge_proposals.json...')
    proposals = []
    for split, (cands, coco, img_map, cat_map) in all_candidates_per_split.items():
        for c in cands:
            proposals.append({
                'split': split,
                'image_id': c['img_id'],
                'category_id': c['category_id'],
                'category_name': cat_map[c['category_id']],
                'ann1_id': c['ann1']['id'],
                'ann2_id': c['ann2']['id'],
                'ann1_bbox': c['ann1']['bbox'],
                'ann2_bbox': c['ann2']['bbox'],
                'occluder_id': c['occluder']['id'],
                'occluder_bbox': c['occluder']['bbox'],
                'merged_bbox': list(c['merged_bbox']),
            })
    with open(out_dir / 'merge_proposals.json', 'w') as f:
        json.dump(proposals, f, indent=2)
    print(f'  wrote {len(proposals)} merge proposals')


if __name__ == '__main__':
    main()
