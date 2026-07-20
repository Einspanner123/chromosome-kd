#!/usr/bin/env python3
"""Baseline (RTMDet-L / Cascade R-CNN) inference on Dataset 2 (24obj) val.

Generates per-image COCO DT predictions for paired Wilcoxon tests against
the RF (LDMDet StochOT) baseline. Reuses the cache format of
baseline_vs_sota.py so existing per-image statistics tooling continues to
work.

Background:
  - The user's paper_draft_CN.md §1.2 originally claimed "RF vs DINO R50/
    RTMDet-L on small-object AP_S paired Wilcoxon test showed no significant
    difference (Table 8)". This is a FACTUAL ERROR — Table 8 only covers
    RF's own variants (A3 vs A2, etc.), not cross-method comparisons.
  - To rectify this, we re-run RTMDet-L and Cascade R-CNN on Dataset 2 val
    (500 imgs) and produce per-image AP for paired tests.
  - DINO R50 checkpoint has been LOST (only metrics.json remains in
    ldmdet-experiment/sota/baselines/dino_r50/, no .pth file). Without a
    checkpoint we cannot run DINO R50 inference; we therefore limit the
    cross-method Wilcoxon test to RTMDet-L and Cascade R-CNN.

Inputs:
  - RTMDet-L checkpoint: work_dirs/baselines/rtmdet_l_24obj/epoch_86.pth
    (best is ep85 mAP=0.863, but only ep86 mAP=0.861 is retained; the
    paper-reported 0.869 came from ep116 which is also lost)
  - Cascade R-CNN: work_dirs/baselines/cascade_rcnn_r50/best_coco_bbox_mAP_epoch_72.pth
  - LDMDet StochOT (RF) predictions already exist at:
      experiments/analysis/baseline_vs_sota_cache/24obj_SOTA_seed42_preds.json
  - DiffusionDet predictions already exist at:
      experiments/analysis/baseline_vs_sota_cache/24obj_DiffusionDet_seed42_preds.json

Output:
  - experiments/analysis/baseline_inference_24obj_cache/
      RTMDet_L_seed42_preds.json
      Cascade_R_CNN_seed42_preds.json
  - experiments/analysis/baseline_inference_24obj_results.json
"""

from __future__ import annotations
import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

from mmengine.config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Ensure REPO root is on sys.path so custom_imports resolves
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Trigger mmdet's registry registration. As with per_class_ap_coupling_3seed.py,
# this mmdet fork registers transforms only under the mmdet::transform child
# scope, so we must re-register them under mmengine::transform for
# DATASETS.build() to find them.
import mmdet.datasets  # noqa: F401
import mmdet.datasets.transforms  # noqa: F401
from mmengine.registry import TRANSFORMS as _MMENGINE_TRANSFORMS
from mmdet.registry import TRANSFORMS as _MMDET_TRANSFORMS
for _name, _module in list(_MMDET_TRANSFORMS._module_dict.items()):
    if _name not in _MMENGINE_TRANSFORMS._module_dict:
        _MMENGINE_TRANSFORMS.register_module(
            name=_name, module=_module, force=True
        )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dataset 2 (24_chromosomes_object, val 500 imgs)
ANN_FILE = str(
    REPO / 'data/24_chromosomes_object/coco/valid/_annotations.coco.json'
)

# Baselines to evaluate (label, config_path, checkpoint_path)
MODELS: list[tuple[str, str, str]] = [
    (
        'RTMDet-L',
        'experiments/configs/baselines/benchmark_24obj/rtmdet_l.py',
        'work_dirs/baselines/rtmdet_l_24obj/epoch_86.pth',  # ep85 best lost
    ),
    (
        'Cascade R-CNN',
        'experiments/configs/baselines/benchmark_24obj/cascade_rcnn_r50.py',
        'work_dirs/baselines/cascade_rcnn_r50/best_coco_bbox_mAP_epoch_72.pth',
    ),
]

# Pre-existing RF + DiffusionDet predictions (for paired tests)
RF_PREDS_PATH = (
    REPO / 'experiments/analysis/baseline_vs_sota_cache/24obj_SOTA_seed42_preds.json'
)
DIFF_PREDS_PATH = (
    REPO / 'experiments/analysis/baseline_vs_sota_cache/24obj_DiffusionDet_seed42_preds.json'
)

OUTPUT_DIR = REPO / 'experiments/analysis/baseline_inference_24obj_cache'
OUTPUT_JSON = REPO / 'experiments/analysis/baseline_inference_24obj_results.json'

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for i, arg in enumerate(sys.argv):
    if arg == '--device' and i + 1 < len(sys.argv):
        DEVICE = sys.argv[i + 1]
        break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def _run_inference(cfg: Config, ckpt_path: Path, device: str) -> list[dict]:
    """Run inference on val set, return COCO DT-format predictions."""
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    val_dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)

    model = init_detector(cfg, str(ckpt_path), device=device)
    model.eval()

    predictions: list[dict] = []
    n_imgs = len(dataset)
    with torch.no_grad():
        for i in range(n_imgs):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            out = model.test_step(data)
            for r in (out if isinstance(out, list) else [out]):
                pi = r.pred_instances
                if pi is None or len(pi.bboxes) == 0:
                    continue
                img_id = getattr(r, 'img_id', i)
                for j in range(len(pi.bboxes)):
                    bbox = pi.bboxes[j].cpu().tolist()
                    bbox_xywh = [
                        bbox[0],
                        bbox[1],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                    ]
                    predictions.append({
                        'image_id': img_id,
                        'category_id': int(pi.labels[j].cpu().item()) + 1,
                        'bbox': bbox_xywh,
                        'score': float(pi.scores[j].cpu().item()),
                    })

            if (i + 1) % 50 == 0:
                print(f'    inference: {i + 1}/{n_imgs}')

    del model
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass
    return predictions


def _compute_metrics(coco_gt: COCO, predictions: list[dict]) -> dict:
    """Compute aggregate + per-class AP from COCO DT predictions."""
    if not predictions:
        aggregate = {k: 0.0 for k in
                     ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l']}
        return {'aggregate': aggregate, 'per_class_ap': {}, 'per_class_ap50': {}}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    aggregate = {
        'mAP': float(stats[0]),
        'AP50': float(stats[1]),
        'AP75': float(stats[2]),
        'AP_s': float(stats[3]),
        'AP_m': float(stats[4]),
        'AP_l': float(stats[5]),
    }

    precision = coco_eval.eval['precision']
    cat_ids = coco_eval.params.catIds
    per_class_ap = {}
    per_class_ap50 = {}
    for k, cat_id in enumerate(cat_ids):
        cat_name = coco_gt.cats[cat_id]['name']
        ap_cat = precision[:, :, k, 0, -1]
        ap_cat = ap_cat[ap_cat > -1]
        per_class_ap[cat_name] = (
            float(ap_cat.mean()) if len(ap_cat) > 0 else 0.0
        )
        ap50_cat = precision[0, :, k, 0, -1]
        ap50_cat = ap50_cat[ap50_cat > -1]
        per_class_ap50[cat_name] = (
            float(ap50_cat.mean()) if len(ap50_cat) > 0 else 0.0
        )

    return {
        'aggregate': aggregate,
        'per_class_ap': per_class_ap,
        'per_class_ap50': per_class_ap50,
    }


def _per_image_ap(coco_gt: COCO, predictions: list[dict],
                  cat_ids: list[int]) -> dict[int, dict[str, float]]:
    """Compute per-image AP @ IoU=0.5:0.95 and IoU=0.5 for each cat_id.

    Returns {img_id: {'ap': float, 'ap50': float, 'n_gt': int, 'n_pred': int}}
    """
    # Group predictions by image
    preds_by_img: dict[int, list[dict]] = {}
    for p in predictions:
        preds_by_img.setdefault(p['image_id'], []).append(p)

    per_image: dict[int, dict[str, float]] = {}
    for img_id in coco_gt.imgs:
        img_preds = preds_by_img.get(img_id, [])
        ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco_gt.loadAnns(ann_ids)
        n_gt = len(anns)

        if n_gt == 0 and not img_preds:
            per_image[img_id] = {'ap': float('nan'), 'ap50': float('nan'),
                                 'n_gt': 0, 'n_pred': 0}
            continue

        # Build a mini COCO with only this image
        img_dict = coco_gt.imgs[img_id]
        anns_dict = {a['id']: a for a in anns}
        mini_coco = COCO()
        mini_coco.dataset = {
            'images': [img_dict],
            'annotations': list(anns_dict.values()),
            'categories': coco_gt.dataset['categories'],
        }
        mini_coco.createIndex()

        if not img_preds:
            per_image[img_id] = {'ap': 0.0, 'ap50': 0.0,
                                 'n_gt': n_gt, 'n_pred': 0}
            continue

        try:
            mini_dt = mini_coco.loadRes(img_preds)
            mini_eval = COCOeval(mini_coco, mini_dt, 'bbox')
            mini_eval.params.imgIds = [img_id]
            mini_eval.evaluate()
            mini_eval.accumulate()
            ap = float(mini_eval.stats[0])  # AP @ 0.5:0.95
            ap50 = float(mini_eval.stats[1])  # AP50
            per_image[img_id] = {
                'ap': ap, 'ap50': ap50,
                'n_gt': n_gt, 'n_pred': len(img_preds),
            }
        except Exception:
            per_image[img_id] = {'ap': 0.0, 'ap50': 0.0,
                                 'n_gt': n_gt, 'n_pred': len(img_preds)}

    return per_image


def _wilcoxon_paired(a: list[float], b: list[float]) -> tuple[float, float]:
    a_arr = np.array(a)
    b_arr = np.array(b)
    diff = a_arr - b_arr
    if len(diff) < 5 or np.all(diff == 0):
        return (float('nan'), float('nan'))
    try:
        stat, p = wilcoxon(a_arr, b_arr, alternative='two-sided')
        return (float(stat), float(p))
    except ValueError:
        return (float('nan'), float('nan'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--models', nargs='+',
        default=['RTMDet-L', 'Cascade R-CNN'],
        help='Subset of baselines to run')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip models whose JSON cache exists')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print('Baseline Inference: RTMDet-L + Cascade R-CNN on Dataset 2 (24obj) val')
    print('=' * 100)
    print(f'Device: {DEVICE}')
    print(f'Annotation: {ANN_FILE}')
    print(f'Output cache: {OUTPUT_DIR}')
    print(f'Final JSON:  {OUTPUT_JSON}')
    print()

    coco_gt = COCO(ANN_FILE)
    print(f'COCO GT: {len(coco_gt.imgs)} imgs, {len(coco_gt.cats)} cats')
    cat_ids = list(coco_gt.cats.keys())
    print()

    # ---- Run inference for requested baselines ----
    results: dict[str, dict] = {}
    cache_paths: dict[str, Path] = {}

    for label, config_rel, ckpt_rel in MODELS:
        if label not in args.models:
            continue
        config_path = _resolve(config_rel)
        ckpt_path = _resolve(ckpt_rel)
        cache_path = OUTPUT_DIR / f'{label.replace(" ", "_").replace("-", "_")}_seed42_preds.json'

        print('-' * 100)
        print(f'>>> {label}')
        print(f'    cfg:  {config_path}')
        print(f'    ckpt: {ckpt_path}')
        print(f'    cache: {cache_path}')

        if not config_path.exists():
            print(f'  [SKIP] Config not found')
            continue
        if not ckpt_path.exists():
            print(f'  [SKIP] Checkpoint not found')
            continue
        if args.skip_existing and cache_path.exists():
            print(f'  [SKIP] Cache exists, loading')
            with open(cache_path) as f:
                cache_data = json.load(f)
            results[label] = cache_data
            cache_paths[label] = cache_path
            continue

        cfg = Config.fromfile(str(config_path))
        predictions = _run_inference(cfg, ckpt_path, DEVICE)
        metrics = _compute_metrics(coco_gt, predictions)
        per_image = _per_image_ap(coco_gt, predictions, cat_ids)

        cache_data = {
            'model': label,
            'config': str(config_path),
            'checkpoint': str(ckpt_path),
            'ann_file': ANN_FILE,
            'num_predictions': len(predictions),
            'metrics': metrics,
            'per_image': per_image,
            'predictions': predictions,
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f'  [OK] cached {len(predictions)} preds, {len(per_image)} imgs')
        print(f'  mAP={metrics["aggregate"]["mAP"]:.4f}  '
              f'AP50={metrics["aggregate"]["AP50"]:.4f}  '
              f'AP_S={metrics["aggregate"]["AP_s"]:.4f}  '
              f'AP_M={metrics["aggregate"]["AP_m"]:.4f}  '
              f'AP_L={metrics["aggregate"]["AP_l"]:.4f}')

        results[label] = cache_data
        cache_paths[label] = cache_path
        print()

    # ---- Load RF (LDMDet StochOT) and DiffusionDet predictions ----
    print('=' * 100)
    print('Loading pre-existing RF + DiffusionDet predictions for paired tests')
    print('=' * 100)

    if RF_PREDS_PATH.exists():
        with open(RF_PREDS_PATH) as f:
            rf_cache = json.load(f)
        # RF cache is just a flat list of preds (older format)
        if isinstance(rf_cache, list):
            rf_preds = rf_cache
            rf_metrics = _compute_metrics(coco_gt, rf_preds)
            rf_per_image = _per_image_ap(coco_gt, rf_preds, cat_ids)
        else:
            rf_preds = rf_cache.get('predictions', [])
            rf_metrics = rf_cache.get('metrics', _compute_metrics(coco_gt, rf_preds))
            rf_per_image = rf_cache.get('per_image', _per_image_ap(coco_gt, rf_preds, cat_ids))
        results['RF (LDMDet StochOT)'] = {
            'predictions': rf_preds, 'metrics': rf_metrics,
            'per_image': rf_per_image,
            'config': 'pre-existing cache (24obj_SOTA_seed42_preds.json)',
            'checkpoint': 'work_dirs/reproduce_0751_stochot_eps5_v2/...',
        }
        print(f'  RF: {len(rf_preds)} preds, mAP={rf_metrics["aggregate"]["mAP"]:.4f}')
    else:
        print(f'  [MISS] RF cache not found: {RF_PREDS_PATH}')

    if DIFF_PREDS_PATH.exists():
        with open(DIFF_PREDS_PATH) as f:
            diff_cache = json.load(f)
        if isinstance(diff_cache, list):
            diff_preds = diff_cache
            diff_metrics = _compute_metrics(coco_gt, diff_preds)
            diff_per_image = _per_image_ap(coco_gt, diff_preds, cat_ids)
        else:
            diff_preds = diff_cache.get('predictions', [])
            diff_metrics = diff_cache.get('metrics', _compute_metrics(coco_gt, diff_preds))
            diff_per_image = diff_cache.get('per_image', _per_image_ap(coco_gt, diff_preds, cat_ids))
        results['DiffusionDet'] = {
            'predictions': diff_preds, 'metrics': diff_metrics,
            'per_image': diff_per_image,
            'config': 'pre-existing cache (24obj_DiffusionDet_seed42_preds.json)',
            'checkpoint': 'work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth',
        }
        print(f'  DiffusionDet: {len(diff_preds)} preds, mAP={diff_metrics["aggregate"]["mAP"]:.4f}')
    else:
        print(f'  [MISS] DiffusionDet cache not found: {DIFF_PREDS_PATH}')
    print()

    # ---- Aggregate metrics summary ----
    print('=' * 100)
    print('Aggregate Metrics Summary (Dataset 2 / 24obj val, 500 imgs)')
    print('=' * 100)
    print(f'{"Model":<24} {"mAP":>8} {"AP50":>8} {"AP75":>8} {"AP_S":>8} {"AP_M":>8} {"AP_L":>8}')
    print('-' * 80)
    for label in ['RF (LDMDet StochOT)', 'RTMDet-L', 'Cascade R-CNN', 'DiffusionDet']:
        if label in results and 'metrics' in results[label]:
            agg = results[label]['metrics']['aggregate']
            print(f'{label:<24} {agg["mAP"]:>8.4f} {agg["AP50"]:>8.4f} '
                  f'{agg["AP75"]:>8.4f} {agg["AP_s"]:>8.4f} '
                  f'{agg["AP_m"]:>8.4f} {agg["AP_l"]:>8.4f}')
    print()

    # ---- Paired Wilcoxon tests on per-image AP ----
    print('=' * 100)
    print('Paired Wilcoxon Signed-Rank Tests on Per-Image AP')
    print('=' * 100)
    print('Method: for each image, compute AP@[IoU=0.5:0.95] using only that')
    print('image\'s GT and predictions. Then pair across models.')
    print()

    ref_label = 'RF (LDMDet StochOT)'
    if ref_label not in results:
        print(f'  [ERROR] Reference {ref_label} not found, cannot run tests')
    else:
        ref_per_img = results[ref_label]['per_image']
        # Convert per_image dict to ordered arrays aligned by img_id
        img_ids = sorted(ref_per_img.keys())
        ref_ap = [ref_per_img[i]['ap'] for i in img_ids]

        print(f'{"Comparison":<40} {"Δ mean":>10} {"Wilcoxon W":>12} {"p-value":>12} {"Sig":>8}')
        print('-' * 90)
        for label in ['RTMDet-L', 'Cascade R-CNN', 'DiffusionDet']:
            if label not in results:
                continue
            other_per_img = results[label]['per_image']
            other_ap = [other_per_img.get(i, {}).get('ap', float('nan'))
                        for i in img_ids]
            # Filter out NaN pairs
            pairs = [(a, b) for a, b in zip(ref_ap, other_ap)
                     if not (np.isnan(a) or np.isnan(b))]
            if len(pairs) < 5:
                print(f'{ref_label} vs {label:<25} {"N/A":>10} {"N/A":>12} {"N/A":>12} {"N/A":>8}')
                continue
            a_vals = [p[0] for p in pairs]
            b_vals = [p[1] for p in pairs]
            mean_delta = float(np.mean(np.array(a_vals) - np.array(b_vals)))
            stat, p = _wilcoxon_paired(a_vals, b_vals)
            sig = ('***' if p < 0.001 else
                   '**' if p < 0.01 else
                   '*' if p < 0.05 else 'n.s.')
            print(f'{ref_label} vs {label:<25} {mean_delta:>+10.4f} {stat:>12.1f} {p:>12.4e} {sig:>8}')
        print()
        print('Note: positive Δ means RF (LDMDet StochOT) has higher per-image AP.')

    # ---- Save final JSON ----
    output_data = {
        'experiment': 'baseline_inference_24obj_wilcoxon',
        'description': (
            'RTMDet-L + Cascade R-CNN inference on Dataset 2 (24obj) val + '
            'paired Wilcoxon tests vs RF (LDMDet StochOT). DINO R50 omitted '
            'because its checkpoint was lost (only metrics.json retained).'
        ),
        'ann_file': ANN_FILE,
        'dataset': '24_chromosomes_object/coco/valid (500 imgs)',
        'device': DEVICE,
        'note': (
            'DINO R50 best epoch (102) checkpoint (325MB) was archived with '
            'config + metrics only; no .pth file was preserved. Cannot run '
            'inference without re-training. Cross-method Wilcoxon test '
            'therefore limited to RTMDet-L, Cascade R-CNN, DiffusionDet.'
        ),
        'models': {
            label: {
                'config': data.get('config', ''),
                'checkpoint': data.get('checkpoint', ''),
                'num_predictions': data.get('num_predictions', len(data.get('predictions', []))),
                'metrics': data.get('metrics', {}),
                'per_image_count': len(data.get('per_image', {})),
            }
            for label, data in results.items()
        },
        'cache_dir': str(OUTPUT_DIR),
        'reference_caches': {
            'RF (LDMDet StochOT)': str(RF_PREDS_PATH),
            'DiffusionDet': str(DIFF_PREDS_PATH),
        },
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\nResults saved to: {OUTPUT_JSON}')
    print(f'Per-model caches in: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
