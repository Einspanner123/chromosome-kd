#!/usr/bin/env python3
"""
Inspect annotation pattern under overlap for a given dataset.
Strategy: visualize top-overlap images with ALL bboxes drawn,
let the user visually judge whether the annotation is amodal (1 box per
chromosome) or visible-only (split boxes on occluded chromosomes).

Usage:
    python viz_overlap.py --dataset 24obj
    python viz_overlap.py --dataset autokary

Output: analysis/{dataset}/overlap/overlap_{split}_img{id}.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
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


def find_top_overlap_imgs(coco, top_k=8):
    """Rank images by total IoU-overlapping bbox pairs."""
    img_to_anns = {}
    for a in coco['annotations']:
        img_to_anns.setdefault(a['image_id'], []).append(a)

    scored = []
    for img_id, anns in img_to_anns.items():
        if len(anns) < 10:
            continue
        n_overlap = 0
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if iou(anns[i]['bbox'], anns[j]['bbox']) > 0.1:
                    n_overlap += 1
        scored.append((n_overlap, img_id, len(anns)))
    scored.sort(reverse=True)
    return scored[:top_k]


def viz_image(data_dir, out_dir, coco, img_map, cat_map, img_id, split, idx, total):
    """Draw ALL annotations on this image, color-coded by category."""
    im_meta = img_map[img_id]
    img_path = data_dir / split / im_meta['file_name']

    fig, ax = plt.subplots(figsize=(12, 12))
    img = Image.open(img_path)
    ax.imshow(img)

    cmap = plt.get_cmap('tab20')

    img_anns = [a for a in coco['annotations'] if a['image_id'] == img_id]
    for a in img_anns:
        x, y, w, h = a['bbox']
        cat_id = a['category_id']
        color = cmap(cat_id % 20)
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.2, edgecolor=color,
            facecolor=color, alpha=0.15)
        ax.add_patch(rect)
        rect_b = patches.Rectangle(
            (x, y), w, h, linewidth=1.2, edgecolor=color, facecolor='none')
        ax.add_patch(rect_b)
        ax.text(x, y - 2, cat_map[cat_id], fontsize=6,
                color=color, alpha=0.9,
                bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                          edgecolor=color, alpha=0.7, linewidth=0.5))

    cat_counts = {}
    for a in img_anns:
        cat_counts[a['category_id']] = cat_counts.get(a['category_id'], 0) + 1
    multi_count_cats = {cat_map[k]: v for k, v in cat_counts.items() if v > 2}

    title = (f'[{split}] {im_meta["file_name"]}  '
             f'img_id={img_id}  [{idx+1}/{total}]\n'
             f'n_anns={len(img_anns)}, '
             f'cats_with_>2_boxes: {multi_count_cats if multi_count_cats else "none"}')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)

    out_name = f'overlap_{split}_img{img_id}.png'
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    parser.add_argument('--top-k', type=int, default=6,
                        help='top-K overlap images per split (default: 6)')
    args = parser.parse_args()

    data_dir = get_dataset_path(args.dataset)
    out_dir = get_output_dir(args.dataset, 'overlap')
    splits = get_splits(args.dataset)

    print('=' * 78)
    print(f'  Inspecting overlap annotation pattern for {args.dataset}')
    print(f'  Data:  {data_dir}')
    print(f'  Output: {out_dir}')
    print('=' * 78)

    total_saved = 0
    for split in splits:
        coco = load_split(data_dir, split)
        img_map = {im['id']: im for im in coco['images']}
        cat_map = {c['id']: c['name'] for c in coco['categories']}

        top = find_top_overlap_imgs(coco, top_k=args.top_k)
        print(f'\n[{split}] top overlap candidates:')
        for n_ov, img_id, n_ann in top[:5]:
            print(f'  img_id={img_id}, n_overlap_pairs={n_ov}, n_anns={n_ann}')

        for idx, (n_ov, img_id, n_ann) in enumerate(top):
            out = viz_image(data_dir, out_dir, coco, img_map, cat_map,
                            img_id, split, idx, len(top))
            print(f'  saved: {out}')
            total_saved += 1

    print()
    print('=' * 78)
    print(f'  Total images: {total_saved}')
    print(f'  Output dir: {out_dir}')
    print('=' * 78)


if __name__ == '__main__':
    main()
