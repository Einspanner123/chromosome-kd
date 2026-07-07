#!/usr/bin/env python3
"""
Data quality comparison across all registered datasets.

Focuses on quality dimensions (not size/distribution which is covered by
dataset_comparison.py):
  - Annotation integrity (degenerate, OOB, duplicates, tiny, missing refs)
  - Image integrity (duplicate file_names, cross-split leakage, image_id collisions)
  - COCO format compliance
  - Split quality (balance, integrity)
  - Cross-dataset category consistency
  - Actual image file existence on disk

Datasets are defined in datasets.py (DATASETS dict).
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from datasets import DATASETS, get_dataset_path, get_splits

TINY_AREA = 100   # bboxes with area below this are considered "tiny" / suspicious
OOB_TOL = 1.0     # tolerance (px) for out-of-bounds check


# ── Loader ──────────────────────────────────────────────────────────
def load_split(data_dir: Path, split: str):
    """Load one split's COCO JSON."""
    p = data_dir / split / '_annotations.coco.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_dataset(alias: str):
    """Load all splits for a dataset alias. Returns dict[split] = coco_dict."""
    data_dir = get_dataset_path(alias)
    splits_list = get_splits(alias)
    splits = {}
    for sp in splits_list:
        d = load_split(data_dir, sp)
        if d is None:
            print(f'  [WARN] {data_dir}/{sp}/_annotations.coco.json missing')
            continue
        splits[sp] = d
    return splits


# ── Quality checks ──────────────────────────────────────────────────
def check_annotation_integrity(coco, split):
    """Per-split annotation integrity."""
    img_map = {im['id']: im for im in coco['images']}
    cat_ids = {c['id'] for c in coco['categories']}

    n_ann = len(coco['annotations'])
    n_img = len(coco['images'])
    img_with_ann = {a['image_id'] for a in coco['annotations']}

    issues = {
        'split': split,
        'n_images': n_img,
        'n_annotations': n_ann,
        'n_categories': len(coco['categories']),
        # ── degenerate bbox (w<=0 or h<=0) ──
        'degenerate_bbox': 0,
        # ── bbox out of image bounds ──
        'oob_bbox': 0,
        # ── tiny bbox (area < TINY_AREA) ──
        'tiny_bbox': 0,
        # ── annotations referencing missing image_id ──
        'dangling_image_ref': 0,
        # ── annotations referencing missing category_id ──
        'dangling_category_ref': 0,
        # ── duplicate annotations (same image_id + bbox + category) ──
        'duplicate_annotations': 0,
        # ── images with no annotations ──
        'images_without_annotations': 0,
        # ── annotation id uniqueness ──
        'duplicate_ann_ids': 0,
        # ── negative bbox coords ──
        'negative_coords': 0,
        # ── bbox area field mismatch (bbox w*h != area) ──
        'area_field_mismatch': 0,
        # ── iscrowd field present ──
        'iscrowd_field_present': 0,
        'iscrowd_1_count': 0,
        # ── segmentation field present ──
        'segmentation_field_present': 0,
    }

    # image_id set
    img_ids = set(img_map.keys())
    images_with_ann = set()
    seen_ann_ids = set()
    seen_ann_keys = Counter()  # (image_id, bbox_tuple, cat_id)

    for a in coco['annotations']:
        # ann id uniqueness
        if a['id'] in seen_ann_ids:
            issues['duplicate_ann_ids'] += 1
        seen_ann_ids.add(a['id'])

        # image / category reference
        if a['image_id'] not in img_ids:
            issues['dangling_image_ref'] += 1
            continue
        if a['category_id'] not in cat_ids:
            issues['dangling_category_ref'] += 1
            continue
        images_with_ann.add(a['image_id'])

        x, y, w, h = a['bbox']
        # degenerate
        if w <= 0 or h <= 0:
            issues['degenerate_bbox'] += 1
        # negative coords
        if x < 0 or y < 0:
            issues['negative_coords'] += 1
        # OOB
        im = img_map[a['image_id']]
        if x + w > im['width'] + OOB_TOL or y + h > im['height'] + OOB_TOL:
            issues['oob_bbox'] += 1
        # tiny
        if w * h < TINY_AREA:
            issues['tiny_bbox'] += 1
        # area mismatch
        if 'area' in a and abs(a['area'] - w * h) > 1.0:
            issues['area_field_mismatch'] += 1
        # iscrowd
        if 'iscrowd' in a:
            issues['iscrowd_field_present'] += 1
            if a.get('iscrowd', 0) == 1:
                issues['iscrowd_1_count'] += 1
        # segmentation
        if 'segmentation' in a:
            issues['segmentation_field_present'] += 1
        # duplicate detection
        key = (a['image_id'], tuple(a['bbox']), a['category_id'])
        seen_ann_keys[key] += 1

    issues['duplicate_annotations'] = sum(1 for k, v in seen_ann_keys.items() if v > 1)
    issues['images_without_annotations'] = len(img_ids - images_with_ann)
    return issues


def check_image_integrity(coco, split, data_dir):
    """Per-split image integrity, including on-disk file existence check."""
    issues = {
        'split': split,
        'n_images': len(coco['images']),
        'duplicate_file_names': 0,
        'duplicate_image_ids': 0,
        'missing_image_files': 0,    # checked against disk
        'extreme_width_range': None,
        'extreme_height_range': None,
        'images_checked_on_disk': 0,
    }
    file_names = [im['file_name'] for im in coco['images']]
    image_ids = [im['id'] for im in coco['images']]
    issues['duplicate_file_names'] = len(file_names) - len(set(file_names))
    issues['duplicate_image_ids'] = len(image_ids) - len(set(image_ids))

    ws = [im['width'] for im in coco['images']]
    hs = [im['height'] for im in coco['images']]
    issues['extreme_width_range'] = (min(ws), max(ws))
    issues['extreme_height_range'] = (min(hs), max(hs))

    # On-disk existence check (sample up to 200 to keep it fast)
    img_dir = data_dir / split
    sample = coco['images'][:200] if len(coco['images']) > 200 else coco['images']
    missing = 0
    checked = 0
    for im in sample:
        fp = img_dir / im['file_name']
        checked += 1
        if not fp.exists():
            missing += 1
    issues['missing_image_files'] = missing
    issues['images_checked_on_disk'] = checked
    return issues


def check_cross_split_integrity(splits, data_dir):
    """Cross-split issues: same file_name in train AND test (data leakage),
    same image_id reused across splits."""
    issues = {
        'cross_split_file_name_overlap': 0,
        'cross_split_image_id_overlap': 0,
        'split_file_counts': {},
        'split_ann_counts': {},
    }
    file_per_split = {}
    id_per_split = {}
    for sp, coco in splits.items():
        files = {im['file_name'] for im in coco['images']}
        ids = {im['id'] for im in coco['images']}
        file_per_split[sp] = files
        id_per_split[sp] = ids
        issues['split_file_counts'][sp] = len(files)
        issues['split_ann_counts'][sp] = len(coco['annotations'])

    # pairwise overlap
    sp_list = list(splits.keys())
    overlap_files = set()
    for i in range(len(sp_list)):
        for j in range(i + 1, len(sp_list)):
            overlap_files |= file_per_split[sp_list[i]] & file_per_split[sp_list[j]]
    issues['cross_split_file_name_overlap'] = len(overlap_files)

    overlap_ids = set()
    for i in range(len(sp_list)):
        for j in range(i + 1, len(sp_list)):
            overlap_ids |= id_per_split[sp_list[i]] & id_per_split[sp_list[j]]
    issues['cross_split_image_id_overlap'] = len(overlap_ids)
    return issues, overlap_files


def file_hash(path, algo='md5', chunk=65536):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_cross_split_image_content_overlap(splits, data_dir, max_check=200):
    """Hash a sample of images in train, then check if any test/valid image
    has identical hash (likely duplicated content)."""
    results = {
        'train_sampled': 0,
        'test_valid_checked': 0,
        'content_duplicates_train_vs_test_valid': 0,
        'duplicate_pairs': [],
    }
    if 'train' not in splits:
        return results
    train_hashes = {}
    train_imgs = splits['train']['images']
    sample = train_imgs[:max_check] if len(train_imgs) > max_check else train_imgs
    for im in sample:
        fp = data_dir / 'train' / im['file_name']
        if not fp.exists():
            continue
        h = file_hash(fp)
        train_hashes[h] = im['file_name']
        results['train_sampled'] += 1

    for sp in ['test', 'valid']:
        if sp not in splits:
            continue
        for im in splits[sp]['images']:
            fp = data_dir / sp / im['file_name']
            if not fp.exists():
                continue
            results['test_valid_checked'] += 1
            h = file_hash(fp)
            if h in train_hashes:
                results['content_duplicates_train_vs_test_valid'] += 1
                if len(results['duplicate_pairs']) < 10:
                    results['duplicate_pairs'].append(
                        (train_hashes[h], sp, im['file_name']))
    return results


def check_category_consistency(splits_ds1, splits_ds2):
    """Are the 24 categories identical between the two datasets?"""
    cats1 = {c['id']: c['name'] for c in splits_ds1['train']['categories']}
    cats2 = {c['id']: c['name'] for c in splits_ds2['train']['categories']}
    return {
        'ds1_categories': cats1,
        'ds2_categories': cats2,
        'same_id_name_pairs': cats1 == cats2,
        'same_name_set': set(cats1.values()) == set(cats2.values()),
        'ds1_only': {k: v for k, v in cats1.items() if k not in cats2 or cats2[k] != v},
        'ds2_only': {k: v for k, v in cats2.items() if k not in cats1 or cats1[k] != v},
    }


def check_split_class_balance(splits):
    """Per-split class distribution — should be similar across splits."""
    out = {}
    for sp, coco in splits.items():
        cat_counts = Counter(a['category_id'] for a in coco['annotations'])
        out[sp] = dict(cat_counts)
    return out


def check_bbox_vs_image_area_ratio(splits):
    """Are there suspicious bboxes that cover the entire image (likely
    full-image mis-annotation) or bboxes that are > image area?"""
    out = {}
    for sp, coco in splits.items():
        img_map = {im['id']: im for im in coco['images']}
        ratios = []
        n_full_image = 0  # bbox area > 90% of image area
        n_exceed_image = 0  # bbox area > image area (impossible if compliant)
        for a in coco['annotations']:
            im = img_map.get(a['image_id'])
            if im is None:
                continue
            img_area = im['width'] * im['height']
            x, y, w, h = a['bbox']
            bbox_area = w * h
            if img_area == 0:
                continue
            r = bbox_area / img_area
            ratios.append(r)
            if r > 0.9:
                n_full_image += 1
            if r > 1.0:
                n_exceed_image += 1
        out[sp] = {
            'mean_ratio': float(np.mean(ratios)) if ratios else 0,
            'median_ratio': float(np.median(ratios)) if ratios else 0,
            'p95_ratio': float(np.percentile(ratios, 95)) if ratios else 0,
            'p99_ratio': float(np.percentile(ratios, 99)) if ratios else 0,
            'max_ratio': float(max(ratios)) if ratios else 0,
            'n_full_image_bbox_gt_90pct': n_full_image,
            'n_bbox_exceeds_image_area': n_exceed_image,
        }
    return out


# ── Main ────────────────────────────────────────────────────────────
def main():
    print('=' * 78)
    print('  Data Quality Comparison')
    print('=' * 78)

    all_data = {}
    for ds_name in DATASETS:
        ds_path = get_dataset_path(ds_name)
        print(f'\nLoading {ds_name} from {ds_path}...')
        splits = load_dataset(ds_name)
        all_data[ds_name] = (splits, ds_path)
        for sp, coco in splits.items():
            print(f'  {sp}: {len(coco["images"])} images, {len(coco["annotations"])} anns')

    # ── Per-split annotation integrity ──
    print('\n' + '=' * 78)
    print('  [1] Annotation Integrity (per split)')
    print('=' * 78)
    rows = []
    for ds_name, (splits, ds_path) in all_data.items():
        for sp, coco in splits.items():
            r = check_annotation_integrity(coco, sp)
            r['dataset'] = ds_name
            rows.append(r)
    df_ann = pd.DataFrame(rows)
    cols = ['dataset', 'split', 'n_images', 'n_annotations',
            'degenerate_bbox', 'oob_bbox', 'tiny_bbox',
            'dangling_image_ref', 'dangling_category_ref',
            'duplicate_annotations', 'images_without_annotations',
            'duplicate_ann_ids', 'negative_coords', 'area_field_mismatch',
            'iscrowd_field_present', 'iscrowd_1_count',
            'segmentation_field_present']
    print(df_ann[cols].to_string(index=False))

    # ── Per-split image integrity ──
    print('\n' + '=' * 78)
    print('  [2] Image Integrity (per split, includes on-disk check)')
    print('=' * 78)
    rows = []
    for ds_name, (splits, ds_path) in all_data.items():
        for sp, coco in splits.items():
            r = check_image_integrity(coco, sp, ds_path)
            r['dataset'] = ds_name
            rows.append(r)
    df_img = pd.DataFrame(rows)
    cols = ['dataset', 'split', 'n_images', 'duplicate_file_names',
            'duplicate_image_ids', 'missing_image_files',
            'images_checked_on_disk', 'extreme_width_range',
            'extreme_height_range']
    print(df_img[cols].to_string(index=False))

    # ── Cross-split integrity ──
    print('\n' + '=' * 78)
    print('  [3] Cross-Split Integrity (data leakage check)')
    print('=' * 78)
    for ds_name, (splits, ds_path) in all_data.items():
        r, overlap = check_cross_split_integrity(splits, ds_path)
        print(f'\n[{ds_name}]')
        for k, v in r.items():
            print(f'  {k}: {v}')
        if overlap:
            print(f'  sample overlapping files: {list(overlap)[:5]}')

    # ── Cross-split image content overlap (hash-based) ──
    print('\n' + '=' * 78)
    print('  [4] Cross-Split Image Content Overlap (md5 hash, sampled)')
    print('=' * 78)
    for ds_name, (splits, ds_path) in all_data.items():
        r = check_cross_split_image_content_overlap(splits, ds_path, max_check=200)
        print(f'\n[{ds_name}]')
        for k, v in r.items():
            if k == 'duplicate_pairs':
                if v:
                    print(f'  duplicate_pairs (first 10):')
                    for p in v:
                        print(f'    train={p[0]}  <->  {p[1]}/{p[2]}')
            else:
                print(f'  {k}: {v}')

    # ── Category consistency between datasets ──
    print('\n' + '=' * 78)
    print('  [5] Category Consistency between datasets')
    print('=' * 78)
    ds1_name, ds2_name = list(all_data.keys())
    r = check_category_consistency(all_data[ds1_name][0], all_data[ds2_name][0])
    print(f'  same id->name mapping: {r["same_id_name_pairs"]}')
    print(f'  same name set: {r["same_name_set"]}')
    if r['ds1_only']:
        print(f'  {ds1_name} only: {r["ds1_only"]}')
    if r['ds2_only']:
        print(f'  {ds2_name} only: {r["ds2_only"]}')

    # ── BBox vs Image area ratio ──
    print('\n' + '=' * 78)
    print('  [6] BBox / Image Area Ratio (suspicious full-image annotations)')
    print('=' * 78)
    for ds_name, (splits, ds_path) in all_data.items():
        print(f'\n[{ds_name}]')
        r = check_bbox_vs_image_area_ratio(splits)
        for sp, st in r.items():
            print(f'  [{sp}] mean={st["mean_ratio"]:.3f} median={st["median_ratio"]:.3f} '
                  f'p95={st["p95_ratio"]:.3f} p99={st["p99_ratio"]:.3f} max={st["max_ratio"]:.3f}')
            print(f'         bbox>90% img: {st["n_full_image_bbox_gt_90pct"]}  '
                  f'bbox>img: {st["n_bbox_exceeds_image_area"]}')

    # ── Per-split class balance ──
    print('\n' + '=' * 78)
    print('  [7] Per-Split Class Balance (top-3 + bottom-3 classes)')
    print('=' * 78)
    for ds_name, (splits, ds_path) in all_data.items():
        print(f'\n[{ds_name}]')
        bal = check_split_class_balance(splits)
        # Get the union of category ids
        all_cats = sorted(set().union(*[set(b.keys()) for b in bal.values()]))
        cat_map = {c['id']: c['name'] for c in splits['train']['categories']}
        for sp in bal:
            counts = bal[sp]
            sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
            top3 = sorted_counts[:3]
            bot3 = sorted_counts[-3:]
            total = sum(counts.values())
            print(f'  [{sp}] total={total}, '
                  f'top3: {[(cat_map.get(c,c), n) for c, n in top3]}, '
                  f'bot3: {[(cat_map.get(c,c), n) for c, n in bot3]}')
        # compute split imbalance: train_per_class / test_per_class ratio
        if 'train' in bal and 'test' in bal:
            ratios = []
            for c in all_cats:
                tr = bal['train'].get(c, 0)
                te = bal['test'].get(c, 0)
                if te > 0 and tr > 0:
                    ratios.append(tr / te)
            if ratios:
                print(f'  train/test per-class ratio: '
                      f'min={min(ratios):.2f} max={max(ratios):.2f} '
                      f'mean={np.mean(ratios):.2f} std={np.std(ratios):.2f}')

    # ── Summary ──
    print('\n' + '=' * 78)
    print('  SUMMARY: Aggregated quality metrics')
    print('=' * 78)
    summary_rows = []
    for ds_name, (splits, ds_path) in all_data.items():
        agg = {
            'dataset': ds_name,
            'total_images': sum(len(c['images']) for c in splits.values()),
            'total_annotations': sum(len(c['annotations']) for c in splits.values()),
        }
        # aggregate annotation issues
        total_deg = total_oob = total_tiny = total_dup = 0
        total_dangling_img = total_dangling_cat = 0
        total_imgs_no_ann = total_neg = total_area_mm = 0
        for sp, coco in splits.items():
            r = check_annotation_integrity(coco, sp)
            total_deg += r['degenerate_bbox']
            total_oob += r['oob_bbox']
            total_tiny += r['tiny_bbox']
            total_dup += r['duplicate_annotations']
            total_dangling_img += r['dangling_image_ref']
            total_dangling_cat += r['dangling_category_ref']
            total_imgs_no_ann += r['images_without_annotations']
            total_neg += r['negative_coords']
            total_area_mm += r['area_field_mismatch']
        agg.update({
            'degenerate_bbox_total': total_deg,
            'oob_bbox_total': total_oob,
            'tiny_bbox_total(<100px2)': total_tiny,
            'duplicate_annotations_total': total_dup,
            'dangling_image_ref_total': total_dangling_img,
            'dangling_category_ref_total': total_dangling_cat,
            'images_without_annotations_total': total_imgs_no_ann,
            'negative_coords_total': total_neg,
            'area_field_mismatch_total': total_area_mm,
        })
        # cross-split
        cs, _ = check_cross_split_integrity(splits, ds_path)
        agg['cross_split_filename_overlap'] = cs['cross_split_file_name_overlap']
        agg['cross_split_imageid_overlap'] = cs['cross_split_image_id_overlap']
        # content hash check (sampled 200 train vs all test/valid)
        h = check_cross_split_image_content_overlap(splits, ds_path, max_check=200)
        agg['content_dup_(sampled)'] = h['content_duplicates_train_vs_test_valid']
        summary_rows.append(agg)
    df_sum = pd.DataFrame(summary_rows)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_sum.to_string(index=False))


if __name__ == '__main__':
    main()
