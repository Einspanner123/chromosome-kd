#!/usr/bin/env python3
"""
Visualize problematic annotations in a dataset.

Problem categories:
  1. Tiny bbox (area < 100 px²) — likely format conversion artifacts
  2. Duplicate annotation (same image_id + bbox + category_id)
  3. Full-image bbox (bbox covers > 90% of image area)

Usage:
    python viz_problematic_anns.py --dataset 24obj
    python viz_problematic_anns.py --dataset autokary

Output: analysis/{dataset}/problematic/{type}_{split}_img{id}.png
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

TINY_AREA = 100        # bboxes with area below this are considered tiny / suspicious
FULL_IMAGE_RATIO = 0.9  # bbox covers > 90% of image


def load_split(data_dir, split):
    with open(data_dir / split / '_annotations.coco.json') as f:
        return json.load(f)


def find_problems(coco, split):
    """Return dict of problem_type -> list of annotations."""
    img_map = {im['id']: im for im in coco['images']}
    cat_map = {c['id']: c['name'] for c in coco['categories']}

    problems = {
        'tiny': [],
        'duplicate': [],
        'full_image': [],
    }

    seen_keys = {}
    for a in coco['annotations']:
        key = (a['image_id'], tuple(a['bbox']), a['category_id'])
        if key in seen_keys:
            problems['duplicate'].append(seen_keys[key])
            problems['duplicate'].append(a)
        else:
            seen_keys[key] = a

    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        area = w * h
        if area < TINY_AREA:
            problems['tiny'].append(a)
        im = img_map[a['image_id']]
        img_area = im['width'] * im['height']
        if img_area > 0 and area / img_area > FULL_IMAGE_RATIO:
            problems['full_image'].append(a)

    return problems, img_map, cat_map


def viz_one(data_dir, out_dir, coco, img_map, cat_map, ann, split,
            problem_type, idx, total):
    """Draw the image with the problematic annotation highlighted."""
    im_meta = img_map[ann['image_id']]
    img_path = data_dir / split / im_meta['file_name']

    fig, ax = plt.subplots(figsize=(8, 8))
    img = Image.open(img_path)
    ax.imshow(img)

    for other in coco['annotations']:
        if other['image_id'] != ann['image_id']:
            continue
        ox, oy, ow, oh = other['bbox']
        is_target = (other['id'] == ann['id'])
        if is_target:
            continue
        rect = patches.Rectangle(
            (ox, oy), ow, oh,
            linewidth=0.5, edgecolor='#888888', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        ax.text(ox, oy - 2, cat_map[other['category_id']],
                fontsize=5, color='#666666', alpha=0.7)

    x, y, w, h = ann['bbox']
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=2.5, edgecolor='red', facecolor='red', alpha=0.25)
    ax.add_patch(rect)
    rect_border = patches.Rectangle(
        (x, y), w, h,
        linewidth=2.5, edgecolor='red', facecolor='none')
    ax.add_patch(rect_border)

    if problem_type == 'tiny':
        cx, cy = x + w / 2, y + h / 2
        zoom_w = max(w * 6, 80)
        zoom_h = max(h * 6, 80)
        zx0 = max(0, cx - zoom_w / 2)
        zy0 = max(0, cy - zoom_h / 2)
        zx1 = min(im_meta['width'], cx + zoom_w / 2)
        zy1 = min(im_meta['height'], cy + zoom_h / 2)
        inset = ax.inset_axes([0.62, 0.62, 0.35, 0.35])
        inset.imshow(img)
        inset.set_xlim(zx0, zx1)
        inset.set_ylim(zy1, zy0)
        inset.set_title(f'Zoom ({w:.0f}×{h:.0f}px, area={w*h:.0f})',
                        fontsize=9, color='red')
        inset.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='red',
            facecolor='red', alpha=0.4))
        for spine in inset.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(1.5)

    title = (f'[{problem_type.upper()}] {split}/{im_meta["file_name"]}\n'
             f'ann_id={ann["id"]}  cat={cat_map[ann["category_id"]]}  '
             f'bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f})  area={w*h:.0f}  '
             f'img={im_meta["width"]}×{im_meta["height"]}\n'
             f'[{idx+1}/{total}]')
    ax.set_title(title, fontsize=9, fontweight='bold', color='red')
    ax.set_xlabel(f'width={im_meta["width"]}', fontsize=8)
    ax.set_ylabel(f'height={im_meta["height"]}', fontsize=8)
    ax.grid(alpha=0.2)

    out_name = f'{problem_type}_{split}_img{ann["image_id"]}_ann{ann["id"]}.png'
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_name


def viz_duplicates_together(data_dir, out_dir, coco, img_map, cat_map, anns, split):
    """For duplicates: draw the same image once with both duplicate boxes highlighted."""
    if not anns:
        return None
    base_ann = anns[0]
    im_meta = img_map[base_ann['image_id']]
    img_path = data_dir / split / im_meta['file_name']

    fig, ax = plt.subplots(figsize=(8, 8))
    img = Image.open(img_path)
    ax.imshow(img)

    for other in coco['annotations']:
        if other['image_id'] != base_ann['image_id']:
            continue
        if any(o['id'] == other['id'] for o in anns):
            continue
        ox, oy, ow, oh = other['bbox']
        rect = patches.Rectangle(
            (ox, oy), ow, oh,
            linewidth=0.5, edgecolor='#888888', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        ax.text(ox, oy - 2, cat_map[other['category_id']],
                fontsize=5, color='#666666', alpha=0.7)

    for i, ann in enumerate(anns):
        x, y, w, h = ann['bbox']
        offset = i * 2
        rect = patches.Rectangle(
            (x + offset, y + offset), w, h,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.2)
        ax.add_patch(rect)
        rect_border = patches.Rectangle(
            (x + offset, y + offset), w, h,
            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect_border)
        ax.annotate(f'ann_id={ann["id"]}',
                    xy=(x + w / 2, y + h / 2),
                    fontsize=8, color='red', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', edgecolor='red', alpha=0.8))

    x, y, w, h = base_ann['bbox']
    title = (f'[DUPLICATE] {split}/{im_meta["file_name"]}\n'
             f'{len(anns)} identical annotations on same bbox  '
             f'cat={cat_map[base_ann["category_id"]]}  '
             f'bbox=({x:.0f},{y:.0f},{w:.0f},{h:.0f})  area={w*h:.0f}\n'
             f'ann_ids: {[a["id"] for a in anns]}')
    ax.set_title(title, fontsize=9, fontweight='bold', color='red')
    ax.grid(alpha=0.2)
    out_name = f'duplicate_{split}_img{base_ann["image_id"]}_combined.png'
    fig.tight_layout()
    fig.savefig(out_dir / out_name, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    args = parser.parse_args()

    data_dir = get_dataset_path(args.dataset)
    out_dir = get_output_dir(args.dataset, 'problematic')
    splits = get_splits(args.dataset)

    print('=' * 78)
    print(f'  Visualizing problematic annotations in {args.dataset}')
    print(f'  Data:  {data_dir}')
    print(f'  Output: {out_dir}')
    print('=' * 78)

    total_per_type = {'tiny': 0, 'duplicate': 0, 'full_image': 0}
    saved = []

    for split in splits:
        coco = load_split(data_dir, split)
        problems, img_map, cat_map = find_problems(coco, split)

        print(f'\n[{split}] problems found:')
        for k, v in problems.items():
            print(f'  {k}: {len(v)}')
            total_per_type[k] += len(v)

        for problem_type, anns in problems.items():
            if problem_type == 'duplicate':
                groups = {}
                for a in anns:
                    key = (a['image_id'], tuple(a['bbox']), a['category_id'])
                    groups.setdefault(key, []).append(a)
                for key, group_anns in groups.items():
                    out = viz_duplicates_together(
                        data_dir, out_dir, coco, img_map, cat_map,
                        group_anns, split)
                    if out:
                        saved.append(out)
                        print(f'  saved: {out}')
                continue

            for i, ann in enumerate(anns):
                out = viz_one(data_dir, out_dir, coco, img_map, cat_map,
                              ann, split, problem_type, i, len(anns))
                saved.append(out)

    print()
    print('=' * 78)
    print(f'  Total problems visualized: {len(saved)}')
    print(f'    tiny bbox:       {total_per_type["tiny"]}')
    print(f'    duplicate:       {total_per_type["duplicate"]} (grouped)')
    print(f'    full-image bbox: {total_per_type["full_image"]}')
    print(f'  Output dir: {out_dir}')
    print('=' * 78)


if __name__ == '__main__':
    main()
