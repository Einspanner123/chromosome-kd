#!/usr/bin/env python3
"""Visualize a chromosome detection dataset — amodal annotation quality.

Outputs (under analysis/{dataset}/dataset/):
  1. samples/     — 9 sample images (3 per split) with all bboxes drawn
  2. class_dist.png — per-class annotation count bar chart across splits
  3. bbox_stats.png — bbox size distribution + per-image ann count histogram
  4. overlap_examples/ — top-6 overlap images (verify amodal annotation)
  5. summary.txt  — text summary of dataset statistics

Usage:
    python viz_dataset.py --dataset 24obj
    python viz_dataset.py --dataset autokary
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from datasets import get_dataset_path, get_output_dir, get_splits, add_dataset_arg

# Biological names for visualization (id -> label)
# Applies to 24-class chromosome datasets (1-22 autosomes, 23=X, 24=Y)
BIO_NAMES = {
    1: 'A1', 2: 'A2', 3: 'A3', 4: 'B4', 5: 'B5',
    6: 'C6', 7: 'C7', 8: 'C8', 9: 'C9', 10: 'C10', 11: 'C11', 12: 'C12',
    13: 'D13', 14: 'D14', 15: 'D15',
    16: 'E16', 17: 'E17', 18: 'E18',
    19: 'F19', 20: 'F20',
    21: 'G21', 22: 'G22',
    23: 'X', 24: 'Y',
}

PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
    (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
    (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
    (199, 100, 0), (72, 0, 118),
]
PALETTE_NORM = [(r / 255, g / 255, b / 255) for r, g, b in PALETTE]


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


def viz_sample(data_dir, out_dir, coco, img_map, img_id, split, idx):
    """Draw all annotations on a sample image."""
    im_meta = img_map[img_id]
    img_path = data_dir / split / im_meta['file_name']

    fig, ax = plt.subplots(figsize=(12, 10))
    img = Image.open(img_path)
    ax.imshow(img)

    img_anns = [a for a in coco['annotations'] if a['image_id'] == img_id]
    for a in img_anns:
        x, y, w, h = a['bbox']
        cat_id = a['category_id']
        color = PALETTE_NORM[(cat_id - 1) % 24]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.0, edgecolor=color,
            facecolor=color, alpha=0.15)
        ax.add_patch(rect)
        rect_b = patches.Rectangle(
            (x, y), w, h, linewidth=1.0, edgecolor=color, facecolor='none')
        ax.add_patch(rect_b)
        ax.text(x, y - 3, BIO_NAMES.get(cat_id, str(cat_id)),
                fontsize=5, color=color, alpha=0.9,
                bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                          edgecolor=color, alpha=0.7, linewidth=0.5))

    title = (f'[{split}] {im_meta["file_name"][:40]}...  '
             f'img_id={img_id}  n_anns={len(img_anns)}  '
             f'size={im_meta["width"]}x{im_meta["height"]}')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_axis_off()

    out_name = f'sample_{split}_{idx}_img{img_id}.png'
    fig.tight_layout()
    fig.savefig(out_dir / 'samples' / out_name, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_class_dist(out_dir, all_data, splits):
    """Per-class annotation count bar chart across splits."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(24)
    width = 0.27
    for i, split in enumerate(splits):
        coco, _, _ = all_data[split]
        cat_counter = Counter(a['category_id'] for a in coco['annotations'])
        counts = [cat_counter.get(c, 0) for c in range(1, 25)]
        ax.bar(x + i * width, counts, width, label=split)
    ax.set_xticks(x + width)
    ax.set_xticklabels([BIO_NAMES[c] for c in range(1, 25)], fontsize=8)
    ax.set_ylabel('Annotation count')
    ax.set_title('Per-class annotation count')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'class_dist.png', dpi=120)
    plt.close(fig)


def plot_bbox_stats(out_dir, all_data, splits):
    """Bbox size distribution + per-image ann count histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    all_w, all_h = [], []
    for split in splits:
        coco, _, _ = all_data[split]
        for a in coco['annotations']:
            all_w.append(a['bbox'][2])
            all_h.append(a['bbox'][3])
    axes[0].hist(all_w, bins=80, alpha=0.6, label='width')
    axes[0].hist(all_h, bins=80, alpha=0.6, label='height')
    axes[0].set_xlabel('px')
    axes[0].set_ylabel('count')
    axes[0].set_title(f'Bbox size distribution\n'
                      f'(w: mean={np.mean(all_w):.0f}, '
                      f'h: mean={np.mean(all_h):.0f})')
    axes[0].legend()
    axes[0].set_xlim(0, max(np.percentile(all_w, 99), np.percentile(all_h, 99)))

    for split in splits:
        coco, _, _ = all_data[split]
        img_to_n = Counter(a['image_id'] for a in coco['annotations'])
        axes[1].hist(list(img_to_n.values()), bins=60, alpha=0.6, label=split)
    axes[1].set_xlabel('anns per image')
    axes[1].set_ylabel('image count')
    axes[1].set_title('Per-image annotation count')
    axes[1].legend()
    axes[1].axvline(46, color='red', linestyle='--', alpha=0.5,
                    label='normal karyotype (46)')

    all_areas = [w * h for w, h in zip(all_w, all_h)]
    axes[2].hist(np.log10(all_areas), bins=80, color='green', alpha=0.6)
    axes[2].set_xlabel('log10(bbox area px²)')
    axes[2].set_ylabel('count')
    axes[2].set_title(f'Bbox area distribution\n'
                      f'(median={np.median(all_areas):.0f} px²)')

    fig.tight_layout()
    fig.savefig(out_dir / 'bbox_stats.png', dpi=120)
    plt.close(fig)


def find_top_overlap_imgs(coco, top_k=6):
    """Rank images by total IoU-overlapping bbox pairs."""
    img_to_anns = defaultdict(list)
    for a in coco['annotations']:
        img_to_anns[a['image_id']].append(a)
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


def viz_overlap(data_dir, out_dir, coco, img_map, img_id, split, idx, total):
    """Draw all annotations on a high-overlap image (verify amodal quality)."""
    im_meta = img_map[img_id]
    img_path = data_dir / split / im_meta['file_name']

    fig, ax = plt.subplots(figsize=(12, 12))
    img = Image.open(img_path)
    ax.imshow(img)

    img_anns = [a for a in coco['annotations'] if a['image_id'] == img_id]
    for a in img_anns:
        x, y, w, h = a['bbox']
        cat_id = a['category_id']
        color = PALETTE_NORM[(cat_id - 1) % 24]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.0, edgecolor=color,
            facecolor=color, alpha=0.12)
        ax.add_patch(rect)
        rect_b = patches.Rectangle(
            (x, y), w, h, linewidth=1.0, edgecolor=color, facecolor='none')
        ax.add_patch(rect_b)
        ax.text(x, y - 2, BIO_NAMES.get(cat_id, str(cat_id)),
                fontsize=5, color=color, alpha=0.9,
                bbox=dict(boxstyle='square,pad=0.1', facecolor='white',
                          edgecolor=color, alpha=0.7, linewidth=0.5))

    title = (f'[{split}] img_id={img_id}  [{idx+1}/{total}]\n'
             f'n_anns={len(img_anns)}, '
             f'size={im_meta["width"]}x{im_meta["height"]}')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)

    out_name = f'overlap_{split}_img{img_id}.png'
    fig.tight_layout()
    fig.savefig(out_dir / 'overlap_examples' / out_name, dpi=120,
                bbox_inches='tight')
    plt.close(fig)


def write_summary(out_dir, all_data, splits, dataset_alias):
    """Write text summary of dataset statistics."""
    lines = ['=' * 70, f'  Dataset Summary: {dataset_alias}', '=' * 70, '']
    total_imgs = 0
    total_anns = 0
    for split in splits:
        coco, _, _ = all_data[split]
        n_imgs = len(coco['images'])
        n_anns = len(coco['annotations'])
        total_imgs += n_imgs
        total_anns += n_anns
        img_to_n = Counter(a['image_id'] for a in coco['annotations'])
        counts = sorted(img_to_n.values())
        lines.append(f'[{split}]')
        lines.append(f'  images: {n_imgs}')
        lines.append(f'  annotations: {n_anns}')
        lines.append(f'  anns/image: min={counts[0]}, max={counts[-1]}, '
                     f'median={counts[len(counts)//2]}, '
                     f'mean={sum(counts)/len(counts):.1f}')
        cat_counter = Counter(a['category_id'] for a in coco['annotations'])
        lines.append(f'  category counts:')
        for c in range(1, 25):
            lines.append(f'    {BIO_NAMES[c]:>3s} (id={c:2d}): {cat_counter.get(c, 0):>5d}')
        lines.append('')

    lines.append(f'TOTAL: {total_imgs} images, {total_anns} annotations')
    lines.append(f'Avg anns/image: {total_anns/total_imgs:.1f}')

    lines.append('')
    lines.append('Amodal quality check:')
    for split in splits:
        coco, _, _ = all_data[split]
        img_to_cats = defaultdict(Counter)
        for a in coco['annotations']:
            img_to_cats[a['image_id']][a['category_id']] += 1
        total = 0
        gt2 = 0
        for img_id, cc in img_to_cats.items():
            for cat_id, n in cc.items():
                total += 1
                if n > 2:
                    gt2 += 1
        lines.append(f'  [{split}] per-img per-cat >2 bboxes: '
                     f'{gt2}/{total} ({gt2/total*100:.2f}%)')

    text = '\n'.join(lines) + '\n'
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_arg(parser)
    args = parser.parse_args()

    data_dir = get_dataset_path(args.dataset)
    out_dir = get_output_dir(args.dataset, 'dataset')
    splits = get_splits(args.dataset)

    # Create subdirs
    (out_dir / 'samples').mkdir(exist_ok=True)
    (out_dir / 'overlap_examples').mkdir(exist_ok=True)

    print('=' * 70)
    print(f'  Visualizing dataset: {args.dataset}')
    print(f'  Data:   {data_dir}')
    print(f'  Output: {out_dir}')
    print('=' * 70)

    all_data = {}
    for split in splits:
        coco = load_split(data_dir, split)
        img_map = {im['id']: im for im in coco['images']}
        all_data[split] = (coco, img_map, None)
        print(f'  [{split}] images={len(coco["images"])}, '
              f'anns={len(coco["annotations"])}')

    # 1. Sample visualizations (3 per split)
    print('\n[samples] drawing 3 samples per split...')
    for split in splits:
        coco, img_map, _ = all_data[split]
        img_to_n = Counter(a['image_id'] for a in coco['annotations'])
        candidates = sorted(
            [img_id for img_id in img_map if img_id in img_to_n],
            key=lambda i: abs(img_to_n[i] - 46)
        )
        for idx, img_id in enumerate(candidates[:3]):
            viz_sample(data_dir, out_dir, coco, img_map, img_id, split, idx)

    # 2. Class distribution chart
    print('[class_dist] plotting per-class distribution...')
    plot_class_dist(out_dir, all_data, splits)

    # 3. Bbox statistics
    print('[bbox_stats] plotting bbox statistics...')
    plot_bbox_stats(out_dir, all_data, splits)

    # 4. Overlap examples (verify amodal)
    print('[overlap_examples] drawing top-6 overlap images from train...')
    if 'train' in all_data:
        coco, img_map, _ = all_data['train']
        top = find_top_overlap_imgs(coco, top_k=6)
        for idx, (n_ov, img_id, n_ann) in enumerate(top):
            viz_overlap(data_dir, out_dir, coco, img_map, img_id,
                        'train', idx, len(top))

    # 5. Summary
    print('\n[summary] writing summary...')
    write_summary(out_dir, all_data, splits, args.dataset)

    print(f'\nDone. All outputs in: {out_dir}')


if __name__ == '__main__':
    main()
