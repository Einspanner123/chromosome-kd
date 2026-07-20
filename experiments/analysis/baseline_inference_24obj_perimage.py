#!/usr/bin/env python3
"""Compute per-image AP + paired Wilcoxon tests for RF vs baselines on Dataset 2.

Reuses cached predictions from baseline_inference_24obj_cache/ and
baseline_vs_sota_cache/, then computes per-image AP@[IoU=0.5:0.95] using a
correct per-image COCOeval approach (the original baseline_inference_24obj.py
had a bug where mini-COCO returned 0 for every image).

Method:
  - Build ONE coco_dt from each model's predictions.
  - For each image, restrict coco_eval.params.imgIds to [img_id] and run
    evaluate+accumulate. This gives the correct single-image AP.
  - Then run paired Wilcoxon signed-rank tests on per-image AP arrays.

Inputs (caches):
  - experiments/analysis/baseline_inference_24obj_cache/RTMDet_L_seed42_preds.json
  - experiments/analysis/baseline_inference_24obj_cache/Cascade_R_CNN_seed42_preds.json
  - experiments/analysis/baseline_vs_sota_cache/24obj_SOTA_seed42_preds.json  (RF)
  - experiments/analysis/baseline_vs_sota_cache/24obj_DiffusionDet_seed42_preds.json

Output:
  - experiments/analysis/baseline_inference_24obj_perimage_results.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO = Path(__file__).resolve().parents[2]

ANN_FILE = str(
    REPO / 'data/24_chromosomes_object/coco/valid/_annotations.coco.json'
)

CACHES = {
    'RF (LDMDet StochOT)': REPO / 'experiments/analysis/baseline_vs_sota_cache/24obj_SOTA_seed42_preds.json',
    'RTMDet-L': REPO / 'experiments/analysis/baseline_inference_24obj_cache/RTMDet_L_ep85_seed42_preds.json',
    'Cascade R-CNN': REPO / 'experiments/analysis/baseline_inference_24obj_cache/Cascade_R_CNN_seed42_preds.json',
    'DINO R50': REPO / 'experiments/analysis/baseline_inference_24obj_cache/DINO_R50_seed42_preds.json',
    'DiffusionDet': REPO / 'experiments/analysis/baseline_vs_sota_cache/24obj_DiffusionDet_seed42_preds.json',
}

OUTPUT_JSON = (
    REPO / 'experiments/analysis/baseline_inference_24obj_perimage_results.json'
)


def load_preds(cache_path: Path) -> list[dict]:
    """Load predictions list from cache file (handle both old/new formats)."""
    with open(cache_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get('predictions', [])


def per_image_ap(coco_gt: COCO, predictions: list[dict]) -> dict[int, dict]:
    """Compute per-image AP@[IoU=0.5:0.95] + AP50 + n_gt + n_pred.

    For each image, builds a mini-COCO containing only that image's GT and
    predictions, then computes AP manually from the precision array.

    NOTE: pycocotools' `accumulate()` returns empty `stats` when only one image
    is present (it requires >= 2 images for stable stats). We therefore read
    `eval['precision']` directly and compute AP from it.
    """
    import copy

    if not predictions:
        return {img_id: {'ap': 0.0, 'ap50': 0.0, 'n_gt': 0, 'n_pred': 0}
                for img_id in coco_gt.imgs}

    # Group predictions by image
    preds_by_img: dict[int, list[dict]] = {}
    for p in predictions:
        preds_by_img.setdefault(p['image_id'], []).append(p)

    cat_ids = list(coco_gt.cats.keys())
    per_image: dict[int, dict] = {}

    for img_id in coco_gt.imgs:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)
        n_gt = len(anns)
        img_preds = preds_by_img.get(img_id, [])
        n_pred = len(img_preds)

        if n_gt == 0 or n_pred == 0:
            per_image[img_id] = {
                'ap': 0.0 if n_gt > 0 else float('nan'),
                'ap50': 0.0 if n_gt > 0 else float('nan'),
                'n_gt': n_gt,
                'n_pred': n_pred,
            }
            continue

        # Build mini-COCO with only this image's GT + preds
        img_dict = coco_gt.imgs[img_id]
        mini_coco = COCO()
        mini_coco.dataset = {
            'images': [img_dict],
            'annotations': copy.deepcopy(anns),
            'categories': coco_gt.dataset['categories'],
        }
        # Reset annotation IDs to be unique (avoid mini-COCO key conflicts)
        for i, a in enumerate(mini_coco.dataset['annotations']):
            a['id'] = i + 1
        mini_coco.createIndex()

        try:
            mini_dt = mini_coco.loadRes(img_preds)
            ev = COCOeval(mini_coco, mini_dt, 'bbox')
            ev.evaluate()
            ev.accumulate()

            # Manual AP computation from precision array
            # precision shape: [T=10, R=101, K=24, A=4, M=3]
            precision = ev.eval['precision']
            # AP @[IoU=0.5:0.95, area=all, maxDets=100] = mean over T, K
            ap_per_cat = []
            ap50_per_cat = []
            for k in range(precision.shape[2]):
                ap_k = precision[:, :, k, 0, -1]  # all IoU, all recall, cat k, area=all, maxDet=100
                ap_k = ap_k[ap_k > -1]
                if len(ap_k) > 0:
                    ap_per_cat.append(float(ap_k.mean()))
                else:
                    ap_per_cat.append(0.0)

                # AP50 = precision at IoU=0.5 (index 0)
                ap50_k = precision[0, :, k, 0, -1]
                ap50_k = ap50_k[ap50_k > -1]
                if len(ap50_k) > 0:
                    ap50_per_cat.append(float(ap50_k.mean()))
                else:
                    ap50_per_cat.append(0.0)

            ap = float(np.mean(ap_per_cat))
            ap50 = float(np.mean(ap50_per_cat))
        except Exception as e:
            print(f'  [WARN] img {img_id} eval failed: {e}')
            ap, ap50 = 0.0, 0.0

        per_image[img_id] = {
            'ap': ap,
            'ap50': ap50,
            'n_gt': n_gt,
            'n_pred': n_pred,
        }

    return per_image


def wilcoxon_paired(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p, mean_delta)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    diff = a_arr - b_arr
    mean_delta = float(np.mean(diff))
    if len(diff) < 5 or np.all(diff == 0):
        return (float('nan'), float('nan'), mean_delta)
    try:
        stat, p = wilcoxon(a_arr, b_arr, alternative='two-sided')
        return (float(stat), float(p), mean_delta)
    except ValueError:
        return (float('nan'), float('nan'), mean_delta)


def ttest_paired(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Paired t-test. Returns (statistic, p, mean_delta)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    diff = a_arr - b_arr
    mean_delta = float(np.mean(diff))
    if len(diff) < 2:
        return (float('nan'), float('nan'), mean_delta)
    try:
        stat, p = ttest_rel(a_arr, b_arr)
        return (float(stat), float(p), mean_delta)
    except Exception:
        return (float('nan'), float('nan'), mean_delta)


def main():
    print('=' * 100)
    print('Per-Image AP + Paired Wilcoxon Tests: RF vs Baselines on Dataset 2 (24obj)')
    print('=' * 100)
    print(f'Annotation: {ANN_FILE}')
    print()

    coco_gt = COCO(ANN_FILE)
    print(f'COCO GT: {len(coco_gt.imgs)} imgs, {len(coco_gt.cats)} cats')
    print()

    # Load all predictions
    print('Loading predictions from caches...')
    all_preds: dict[str, list[dict]] = {}
    for label, path in CACHES.items():
        if not path.exists():
            print(f'  [MISS] {label}: {path}')
            continue
        preds = load_preds(path)
        all_preds[label] = preds
        print(f'  {label}: {len(preds)} preds from {path.name}')
    print()

    # Compute per-image AP for each model
    print('Computing per-image AP for each model (this may take ~5 min/model)...')
    per_image_all: dict[str, dict[int, dict]] = {}
    for label, preds in all_preds.items():
        print(f'>>> {label}...')
        pi = per_image_ap(coco_gt, preds)
        per_image_all[label] = pi
        # Summary
        aps = [v['ap'] for v in pi.values()
               if v['ap'] == v['ap']]  # filter NaN
        print(f'    {len(aps)} valid images (after NaN filter)')
        if aps:
            print(f'    mean per-image AP: {np.mean(aps):.4f}, '
                  f'std: {np.std(aps):.4f}, '
                  f'min: {np.min(aps):.4f}, max: {np.max(aps):.4f}')
            print(f'    zero-AP images: {sum(1 for a in aps if a == 0)}')
        print()

    # Aggregate mAP summary (sanity check)
    print('=' * 100)
    print('Aggregate Metrics Summary (full val set)')
    print('=' * 100)
    print(f'{"Model":<24} {"mAP":>8} {"AP50":>8} {"AP75":>8} {"AP_S":>8} {"AP_M":>8} {"AP_L":>8}')
    print('-' * 80)
    for label, preds in all_preds.items():
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        s = ev.stats
        print(f'{label:<24} {s[0]:>8.4f} {s[1]:>8.4f} {s[2]:>8.4f} '
              f'{s[3]:>8.4f} {s[4]:>8.4f} {s[5]:>8.4f}')
    print()

    # Paired Wilcoxon tests on per-image AP
    print('=' * 100)
    print('Paired Wilcoxon Signed-Rank Tests on Per-Image AP@[IoU=0.5:0.95]')
    print('=' * 100)

    ref_label = 'RF (LDMDet StochOT)'
    if ref_label not in per_image_all:
        print(f'  [ERROR] Reference {ref_label} not found')
        return

    ref_per_img = per_image_all[ref_label]
    img_ids = sorted(ref_per_img.keys())

    print(f'{"Comparison":<50} {"N pairs":>8} {"Δ mean":>10} '
          f'{"Wilcoxon W":>12} {"p (Wilcoxon)":>14} {"p (t-test)":>12} {"Sig":>8}')
    print('-' * 120)

    test_results: list[dict] = []
    for label in ['RTMDet-L', 'Cascade R-CNN', 'DINO R50', 'DiffusionDet']:
        if label not in per_image_all:
            continue
        other_per_img = per_image_all[label]
        # Build aligned arrays, filter NaN
        pairs = []
        for i in img_ids:
            a = ref_per_img.get(i, {}).get('ap', float('nan'))
            b = other_per_img.get(i, {}).get('ap', float('nan'))
            if a == a and b == b:  # both not NaN
                pairs.append((a, b))
        a_vals = [p[0] for p in pairs]
        b_vals = [p[1] for p in pairs]
        w_stat, w_p, mean_delta = wilcoxon_paired(a_vals, b_vals)
        t_stat, t_p, _ = ttest_paired(a_vals, b_vals)
        sig = ('***' if w_p < 0.001 else
               '**' if w_p < 0.01 else
               '*' if w_p < 0.05 else 'n.s.')
        print(f'{ref_label} vs {label:<30} {len(pairs):>8} {mean_delta:>+10.4f} '
              f'{w_stat:>12.1f} {w_p:>14.4e} {t_p:>12.4e} {sig:>8}')
        test_results.append({
            'comparison': f'{ref_label} vs {label}',
            'n_pairs': len(pairs),
            'mean_delta': mean_delta,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_p_value': w_p,
            'ttest_statistic': t_stat,
            'ttest_p_value': t_p,
            'significance': sig,
        })
    print()
    print('Note: positive Δ means RF has higher per-image AP than the baseline.')

    # Also test per-image AP50
    print()
    print('=' * 100)
    print('Paired Wilcoxon Signed-Rank Tests on Per-Image AP50 (IoU=0.5)')
    print('=' * 100)
    print(f'{"Comparison":<50} {"N pairs":>8} {"Δ mean":>10} '
          f'{"Wilcoxon W":>12} {"p (Wilcoxon)":>14} {"Sig":>8}')
    print('-' * 100)
    for label in ['RTMDet-L', 'Cascade R-CNN', 'DINO R50', 'DiffusionDet']:
        if label not in per_image_all:
            continue
        other_per_img = per_image_all[label]
        pairs = []
        for i in img_ids:
            a = ref_per_img.get(i, {}).get('ap50', float('nan'))
            b = other_per_img.get(i, {}).get('ap50', float('nan'))
            if a == a and b == b:
                pairs.append((a, b))
        a_vals = [p[0] for p in pairs]
        b_vals = [p[1] for p in pairs]
        w_stat, w_p, mean_delta = wilcoxon_paired(a_vals, b_vals)
        sig = ('***' if w_p < 0.001 else
               '**' if w_p < 0.01 else
               '*' if w_p < 0.05 else 'n.s.')
        print(f'{ref_label} vs {label:<30} {len(pairs):>8} {mean_delta:>+10.4f} '
              f'{w_stat:>12.1f} {w_p:>14.4e} {sig:>8}')

    # Save final JSON
    output = {
        'experiment': 'baseline_inference_24obj_perimage_wilcoxon',
        'description': (
            'Per-image AP and paired Wilcoxon/t-test for RF vs RTMDet-L / '
            'Cascade R-CNN / DINO R50 / DiffusionDet on Dataset 2 (24obj) val. '
            'DINO R50 best@ep102 checkpoint was recovered from workstation on '
            '2026-07-20 (originally thought lost).'
        ),
        'ann_file': ANN_FILE,
        'dataset': '24_chromosomes_object/coco/valid (500 imgs)',
        'models': {
            label: {
                'cache_path': str(path),
                'num_predictions': len(all_preds.get(label, [])),
                'per_image_count': len(per_image_all.get(label, {})),
                'mean_per_image_ap': float(np.mean([
                    v['ap'] for v in per_image_all[label].values()
                    if v['ap'] == v['ap']
                ])) if label in per_image_all else None,
            }
            for label, path in CACHES.items()
        },
        'paired_tests': test_results,
        'note_dino_r50': (
            'DINO R50 best@ep102 checkpoint (340MB) was recovered from '
            'workstation (work_dirs/dino_r50/) on 2026-07-20 after originally '
            'being marked lost. Paper-reported mAP=0.868 (actual 0.869 per '
            'metrics.json). Test now included.'
        ),
        'note_rtmdet_l': (
            'RTMDet-L ep85 (actual best, mAP=0.8630 per train.log, '
            'independently re-inferred mAP=0.8626). Prior runs used ep86 '
            '(mAP=0.8610) under the false assumption that ep85.pth was '
            'lost; ep85.pth actually exists (637 MB). Paper-reported '
            '0.869 in Table 6 is WRONG — training never reached 0.869. '
            'Test now uses ep85 (true best) checkpoint (2026-07-20 fix).'
        ),
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to: {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
