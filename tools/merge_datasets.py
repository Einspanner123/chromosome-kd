#!/usr/bin/env python3
"""合并 Original + 24obj 两个 COCO 数据集为一个，避免 ID 冲突。"""
import json
import os
import shutil
from pathlib import Path

DATA_ROOT = Path('/media/ross/8TB/linkst/chromo/chromosome-kd/data')
OUT_DIR = DATA_ROOT / 'merged_24obj_original'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ['train', 'valid']

datasets = {
    'original': {
        'root': DATA_ROOT / 'Chromosome20240904_NoAug_NoResize_coco',
    },
    '24obj': {
        'root': DATA_ROOT / '24_chromosomes_object/coco',
    },
}

for split in SPLITS:
    merged = {
        'info': {'description': f'Merged Original + 24obj ({split})', 'version': '1.0'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': None,
    }

    img_id_offset = 0
    ann_id_offset = 0
    img_dir_map = {}

    for ds_name, ds_info in datasets.items():
        ann_path = ds_info['root'] / split / '_annotations.coco.json'
        if not ann_path.exists():
            print(f"[SKIP] {ds_name}/{split} not found")
            continue

        print(f"[LOAD] {ds_name}/{split}")
        with open(ann_path) as f:
            data = json.load(f)

        # Copy categories from first dataset
        if merged['categories'] is None:
            merged['categories'] = data['categories']

        # Remap image IDs
        for img in data['images']:
            new_id = img['id'] + img_id_offset
            img['id'] = new_id
            merged['images'].append(img)

        # Remap annotation IDs + image_id references
        for ann in data['annotations']:
            ann['id'] += ann_id_offset
            ann['image_id'] += img_id_offset
            merged['annotations'].append(ann)

        img_id_offset += max(img['id'] for img in data['images']) + 1
        ann_id_offset = max(a['id'] for a in merged['annotations']) + 1

        # Copy image files (symlink to save space)
        src_img_dir = ds_info['root'] / split
        dst_img_dir = OUT_DIR / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        for img in data['images']:
            src = src_img_dir / img['file_name']
            dst = dst_img_dir / img['file_name']
            if not dst.exists() and src.exists():
                dst.symlink_to(src)

    # Write merged annotation
    out_path = OUT_DIR / split / '_annotations.coco.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(merged, f)

    print(f"[DONE] {split}: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
    print(f"       -> {out_path}")

print("\n合并完成！")
print(f"数据根目录: {OUT_DIR}")
