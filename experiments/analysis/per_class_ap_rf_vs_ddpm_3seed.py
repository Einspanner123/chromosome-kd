#!/usr/bin/env python3
"""Per-class AP analysis: RF+Heun (RF+Heun) vs DDPM (DDPM baseline) × seeds (Dataset 2 / 24obj).

Runs inference on DDPM baseline (DDPM baseline) and RF+Heun (RF+Heun) checkpoints and produces:
  - per-class AP @ IoU=0.5:0.95 (and AP@0.5, AP@0.75) for each (method, seed)
  - mean ± std per method across seeds
  - paired Wilcoxon signed-rank tests on per-class AP
  - JSON dump with raw + aggregate results
  - focus on small classes (Y, G22, F19, F20) vs large classes (RF+Heun, +AdaLN-Zero, +Stoch. Coupling)

Background:
  - DDPM baseline / RF+Heun training-log analysis showed RF+Heun significantly outperforms
    DDPM baseline: bbox_mAP Δ=+0.091, bbox_mAP_75 Δ=+0.052 (2 seeds: 123, 789).
    Small classes (Y/G22/F19/F20) had mean Δ=+0.071 vs large classes
    (RF+Heun/+AdaLN-Zero/+Stoch. Coupling) +0.056. This script re-runs inference with strict COCO
    AP@0.5:0.95 to verify the per-class improvement pattern.
  - seed 42 DDPM baseline / RF+Heun checkpoints are on the ross server and may not be locally
    available; default seeds are [123, 789]. Pass --seeds 42 123 789 to
    include seed 42 if checkpoints have been synced.
  - DDPM baseline (a0_baseline_24obj): DDPM 1-step Euler, no RF, no AdaLN, no StochOT.
  - RF+Heun (a1_rf_heun_24obj): +RF +Heun 4-step, no AdaLN, no StochOT.

Inference-side note:
  - No special config modification is needed. Each method loads its own
    config (a0_baseline_24obj_multiseed.py / a1_rf_heun_24obj_multiseed.py)
    which already specifies the correct sampler and timesteps.
"""

from __future__ import annotations
import json
import os
import sys
import tempfile
import argparse
from pathlib import Path

# Ensure REPO root is on sys.path so custom_imports
# ('experiments.mmdet_bridge.*') resolves when Config.fromfile loads temp file
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from scipy.stats import wilcoxon

from mmengine.config import Config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Explicitly trigger mmdet's full registry registration (transforms like
# PackDetInputs). Without this, DATASETS.build() may fail with
# "PackDetInputs is not in the mmengine::transform registry" because
# mmengine.config.Config loads before mmdet has a chance to register.
import mmdet.datasets  # noqa: F401  (registration side effect)
import mmdet.datasets.transforms  # noqa: F401

# Bugfix: in this mmdet fork, transforms registered under the `mmdet::transform`
# child scope are NOT auto-discovered when mmengine's BaseDataset builds a
# pipeline via the root `mmengine::transform` registry. Manually re-register
# the det transforms under the mmengine root scope so DATASETS.build() finds them.
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

# Dataset 2 (24 Chromosomes Object, 24 classes, standard C-group order)
ANN_FILE = str(
    REPO / 'data/24_chromosomes_object/coco/valid/_annotations.coco.json'
)

CLASS_NAMES = [
    'A1', 'A2', 'A3',
    'B4', 'B5',
    'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
    'D13', 'D14', 'D15',
    'E16', 'E17', 'E18',
    'F19', 'F20',
    'G21', 'G22',
    'X', 'Y',
]

GROUP_LABELS = [
    'A (1-3)', 'A (1-3)', 'A (1-3)',
    'B (4-5)', 'B (4-5)',
    'C (6-12)', 'C (6-12)', 'C (6-12)', 'C (6-12)', 'C (6-12)', 'C (6-12)', 'C (6-12)',
    'D (13-15)', 'D (13-15)', 'D (13-15)',
    'E (16-18)', 'E (16-18)', 'E (16-18)',
    'F (19-20)', 'F (19-20)',
    'G (21-22)', 'G (21-22)',
    'X',
    'Y',
]

SMALL_CLASSES = ['Y', 'G22', 'F19', 'F20']
LARGE_CLASSES = ['A1', 'A2', 'A3']

# 2 methods: (display_name, method_key)
METHODS = [
    ('DDPM baseline (DDPM)', 'a0'),
    ('RF+Heun (RF+Heun)', 'a1'),
]

# Default seeds: seed 42 DDPM baseline/RF+Heun checkpoints are on ross server.
# Locally available: seeds 123, 789.
DEFAULT_SEEDS = [123, 789]

# DDPM baseline (DDPM baseline) checkpoint + config paths per seed.
A0_PATHS = {
    123: (
        'work_dirs/multi_seed/a0_baseline_24obj_multiseed_20260718_003700/seed_123/a0_baseline_24obj_multiseed.py',
        'work_dirs/multi_seed/a0_baseline_24obj_multiseed_20260718_003700/seed_123/best_coco_bbox_mAP_epoch_10.pth',
    ),
    789: (
        'work_dirs/multi_seed/a0_baseline_24obj_multiseed_20260718_003700/seed_789/a0_baseline_24obj_multiseed.py',
        'work_dirs/multi_seed/a0_baseline_24obj_multiseed_20260718_003700/seed_789/best_coco_bbox_mAP_epoch_12.pth',
    ),
    # seed 42 is on ross server; add path here if synced locally.
    # 42: (
    #     'work_dirs/multi_seed/a0_baseline_24obj_multiseed_*/seed_42/a0_baseline_24obj_multiseed.py',
    #     'work_dirs/multi_seed/a0_baseline_24obj_multiseed_*/seed_42/best_coco_bbox_mAP_epoch_*.pth',
    # ),
}

# RF+Heun (RF+Heun) checkpoint + config paths per seed.
A1_PATHS = {
    123: (
        'work_dirs/multi_seed/a1_rf_heun_24obj_multiseed_20260718_044730/seed_123/a1_rf_heun_24obj_multiseed.py',
        'work_dirs/multi_seed/a1_rf_heun_24obj_multiseed_20260718_044730/seed_123/best_coco_bbox_mAP_epoch_65.pth',
    ),
    789: (
        'work_dirs/multi_seed/a1_rf_heun_24obj_multiseed_20260718_044730/seed_789/a1_rf_heun_24obj_multiseed.py',
        'work_dirs/multi_seed/a1_rf_heun_24obj_multiseed_20260718_044730/seed_789/best_coco_bbox_mAP_epoch_37.pth',
    ),
    # seed 42 is on ross server; add path here if synced locally.
    # 42: (
    #     'work_dirs/multi_seed/a1_rf_heun_24obj_multiseed_*/seed_42/a1_rf_heun_24obj_multiseed.py',
    #     'work_dirs/multi_seed/a1_rf_heun_24obj_multiseed_*/seed_42/best_coco_bbox_mAP_epoch_*.pth',
    # ),
}

METHOD_PATHS = {'a0': A0_PATHS, 'a1': A1_PATHS}

OUTPUT_DIR = REPO / 'experiments/analysis/per_class_ap_rf_vs_ddpm_3seed_cache'
OUTPUT_JSON = REPO / 'experiments/analysis/per_class_ap_rf_vs_ddpm_3seed_results.json'

# Parse --device from command line, e.g. --device cuda:1
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for i, arg in enumerate(sys.argv):
    if arg == '--device' and i + 1 < len(sys.argv):
        DEVICE = sys.argv[i + 1]
        break


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _load_config_safe(config_path: Path) -> Config:
    """Load config with custom_imports handling.

    DDPM baseline/RF+Heun multi_seed configs already use 'experiments.mmdet_bridge.*' imports
    (verified 2026-07-20), so no string replace is needed. We still go through
    a temp file to allow optional future overrides without mutating source.
    """
    with open(config_path, 'r') as f:
        config_text = f.read()

    # Fallback: legacy configs referencing projects.LDMDet (kept for safety)
    if 'projects.LDMDet' in config_text:
        config_text = config_text.replace(
            'projects.LDMDet.model',
            'experiments.mmdet_bridge.registry',
        ).replace(
            'projects.LDMDet.hooks',
            'experiments.mmdet_bridge.hooks',
        ).replace(
            "'experiments.mmdet_bridge.registry',",
            "'experiments.mmdet_bridge.registry',\n"
            "        'experiments.mmdet_bridge.detector',\n"
            "        'experiments.mmdet_bridge.transforms',",
        )
        print(f'  [FIX] Replaced custom_imports: projects.LDMDet → experiments.mmdet_bridge')

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', dir=str(REPO), delete=False
    ) as tmp:
        tmp.write(config_text)
        tmp_path = tmp.name

    try:
        cfg = Config.fromfile(tmp_path)
    finally:
        os.unlink(tmp_path)
    return cfg


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
    """Compute aggregate + per-class AP from COCO DT predictions.

    Returns per-class AP @ IoU=0.5:0.95, AP@0.5, and AP@0.75.
    """
    if not predictions:
        aggregate = {k: 0.0 for k in
                     ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l']}
        per_class_ap = {n: 0.0 for n in CLASS_NAMES}
        per_class_ap50 = {n: 0.0 for n in CLASS_NAMES}
        per_class_ap75 = {n: 0.0 for n in CLASS_NAMES}
        return {'aggregate': aggregate,
                'per_class_ap': per_class_ap,
                'per_class_ap50': per_class_ap50,
                'per_class_ap75': per_class_ap75}

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

    # per-class AP @ IoU=0.5:0.95, IoU=0.5, IoU=0.75
    # precision shape: [T, R, K, A, M], T=10 IoU thresholds (0.5..0.95)
    # index 0 = IoU=0.5, index 5 = IoU=0.75
    precision = coco_eval.eval['precision']
    cat_ids = coco_eval.params.catIds
    per_class_ap = {}
    per_class_ap50 = {}
    per_class_ap75 = {}

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

        ap75_cat = precision[5, :, k, 0, -1]
        ap75_cat = ap75_cat[ap75_cat > -1]
        per_class_ap75[cat_name] = (
            float(ap75_cat.mean()) if len(ap75_cat) > 0 else 0.0
        )

    return {
        'aggregate': aggregate,
        'per_class_ap': per_class_ap,
        'per_class_ap50': per_class_ap50,
        'per_class_ap75': per_class_ap75,
    }


def _wilcoxon_paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test on two per-class AP arrays.

    Returns (statistic, p_value). If all differences are zero or arrays too
    small, returns (nan, nan).
    """
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
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--methods', nargs='+',
        default=['DDPM baseline (DDPM)', 'RF+Heun (RF+Heun)'],
        help='Subset of methods to run (default: both)')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
        help=f'Seeds to run (default: {DEFAULT_SEEDS}; seed 42 may be on ross)')
    parser.add_argument(
        '--device', type=str, default=DEVICE,
        help=f'Device for inference (default: {DEVICE})')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip (method, seed) pairs whose JSON cache exists')
    args = parser.parse_args()
    DEVICE = args.device

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print('Per-Class AP Analysis: RF+Heun (RF+Heun) vs DDPM (DDPM baseline) × Seeds (Dataset 2 / 24obj)')
    print('=' * 100)
    print(f'Device: {DEVICE}')
    print(f'Annotation: {ANN_FILE}')
    print(f'Output cache: {OUTPUT_DIR}')
    print(f'Final JSON:  {OUTPUT_JSON}')
    print(f'Methods: {args.methods}')
    print(f'Seeds:   {args.seeds}')
    print()

    # Load COCO GT once
    coco_gt = COCO(ANN_FILE)
    print(f'COCO GT: {len(coco_gt.imgs)} imgs, {len(coco_gt.cats)} cats')
    print()

    # Build run plan: (method_name, method_key, seed, config_path, ckpt_path, cache_path)
    plan: list[tuple[str, str, int, Path, Path, Path]] = []
    for method_name, method_key in METHODS:
        if method_name not in args.methods:
            continue
        paths_dict = METHOD_PATHS[method_key]
        for seed in args.seeds:
            if seed not in args.seeds:
                continue
            if seed not in paths_dict:
                print(f'[MISS] {method_name} seed{seed}: no path configured (seed 42 may be on ross)')
                continue
            config_rel, ckpt_rel = paths_dict[seed]
            config_path = _resolve(config_rel)
            ckpt_path = _resolve(ckpt_rel)
            cache_path = OUTPUT_DIR / f'{method_name.replace(" ", "_").replace("(", "").replace(")", "")}_seed{seed}.json'
            if not ckpt_path.exists():
                print(f'[MISS] {method_name} seed{seed}: ckpt not found {ckpt_path}')
                continue
            if not config_path.exists():
                print(f'[MISS] {method_name} seed{seed}: config not found {config_path}')
                continue
            if args.skip_existing and cache_path.exists():
                print(f'[SKIP] {method_name} seed{seed}: cache exists')
                continue
            plan.append((method_name, method_key, seed, config_path, ckpt_path, cache_path))

    if not plan:
        print('Nothing to run. (Use without --skip-existing to force re-run)')
        return

    print(f'Will run {len(plan)} inference jobs:')
    for mn, _, sd, cfg, ckpt, _ in plan:
        print(f'  - {mn} seed{sd}')
        print(f'      cfg={cfg.name}')
        print(f'      ckpt={ckpt.name}')
    print()

    # ---- Run inference + compute metrics ----
    results: dict[tuple[str, int], dict] = {}

    for method_name, method_key, seed, config_path, ckpt_path, cache_path in plan:
        print('-' * 100)
        print(f'>>> {method_name}  seed={seed}')
        print(f'    cfg:  {config_path}')
        print(f'    ckpt: {ckpt_path}')
        print(f'    cache: {cache_path}')

        cfg = _load_config_safe(config_path)

        predictions = _run_inference(cfg, ckpt_path, DEVICE)
        metrics = _compute_metrics(coco_gt, predictions)

        cache_data = {
            'method': method_name,
            'method_key': method_key,
            'seed': seed,
            'config': str(config_path),
            'checkpoint': str(ckpt_path),
            'ann_file': ANN_FILE,
            'num_predictions': len(predictions),
            'metrics': metrics,
            'predictions': predictions,
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print(f'    [OK] cached {len(predictions)} preds')
        print(f'    mAP={metrics["aggregate"]["mAP"]:.4f}  '
              f'AP50={metrics["aggregate"]["AP50"]:.4f}  '
              f'AP75={metrics["aggregate"]["AP75"]:.4f}  '
              f'AP_S={metrics["aggregate"]["AP_s"]:.4f}  '
              f'AP_M={metrics["aggregate"]["AP_m"]:.4f}  '
              f'AP_L={metrics["aggregate"]["AP_l"]:.4f}')

        results[(method_name, seed)] = metrics
        print()

    # ---- Also load any skipped caches so we have the full picture ----
    for method_name, _ in METHODS:
        if method_name not in args.methods:
            continue
        for seed in args.seeds:
            key = (method_name, seed)
            if key in results:
                continue
            cache_path = OUTPUT_DIR / f'{method_name.replace(" ", "_").replace("(", "").replace(")", "")}_seed{seed}.json'
            if cache_path.exists():
                with open(cache_path) as f:
                    cache_data = json.load(f)
                results[key] = cache_data['metrics']
                print(f'[LOAD] {method_name} seed{seed} from cache')

    # ---- Summary table: per-class AP across (method × seed) ----
    method_order = [m[0] for m in METHODS if m[0] in args.methods]
    print()
    print('=' * 110)
    print('Per-Class AP @ IoU=0.5:0.95 (mean over seeds)')
    print('=' * 110)
    header = f'{"Class":>5} {"Group":<14}'
    for m in method_order:
        header += f' {m:>26} mean±std'
    header += f' {"Δ(RF+Heun-DDPM baseline)":>14}'
    print(header)
    print('-' * 110)

    summary_per_class: dict[str, dict[str, dict]] = {}
    delta_per_class: dict[str, float] = {}
    for cls_name in CLASS_NAMES:
        row = f'{cls_name:>5} {GROUP_LABELS[CLASS_NAMES.index(cls_name)]:<14}'
        summary_per_class[cls_name] = {}
        means = {}
        for m in method_order:
            vals = []
            for seed in args.seeds:
                key = (m, seed)
                if key in results:
                    v = results[key]['per_class_ap'].get(cls_name, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
                summary_per_class[cls_name][m] = {
                    'mean': mean, 'std': std,
                    'values': vals,
                }
                means[m] = mean
            else:
                row += f' {"N/A":>22}'
        if len(means) == 2:
            delta = means['RF+Heun (RF+Heun)'] - means['DDPM baseline (DDPM)']
            row += f' {delta:>+14.4f}'
            delta_per_class[cls_name] = delta
        print(row)

    # ---- Aggregate row ----
    print('-' * 110)
    agg_metrics = ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l']
    for agg in agg_metrics:
        row = f'{agg:>5} {"":<14}'
        means = {}
        for m in method_order:
            vals = []
            for seed in args.seeds:
                key = (m, seed)
                if key in results:
                    v = results[key]['aggregate'].get(agg, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
                means[m] = mean
            else:
                row += f' {"N/A":>22}'
        if len(means) == 2:
            delta = means['RF+Heun (RF+Heun)'] - means['DDPM baseline (DDPM)']
            row += f' {delta:>+14.4f}'
        print(row)
    print()

    # ---- Small vs large class focus ----
    print('=' * 110)
    print('SMALL vs LARGE CLASS ANALYSIS (Δ = RF+Heun (RF+Heun) - DDPM baseline (DDPM))')
    print('=' * 110)
    print(f'{"Group":<12} {"Classes":<28} {"Mean Δ":>10} {"Min Δ":>10} {"Max Δ":>10}')
    print('-' * 80)
    for group_name, group_cls in [('Small', SMALL_CLASSES), ('Large', LARGE_CLASSES)]:
        deltas = [delta_per_class[c] for c in group_cls if c in delta_per_class]
        if deltas:
            print(f'{group_name:<12} {", ".join(group_cls):<28} '
                  f'{np.mean(deltas):>+10.4f} {min(deltas):>+10.4f} {max(deltas):>+10.4f}')
    print()

    # ---- Per-class AP@0.75 table (for cross-validation with training logs) ----
    print('=' * 110)
    print('Per-Class AP @ IoU=0.75 (cross-validation with training-log mAP_75)')
    print('=' * 110)
    header = f'{"Class":>5} {"Group":<14}'
    for m in method_order:
        header += f' {m:>26} mean±std'
    header += f' {"Δ(RF+Heun-DDPM baseline)":>14}'
    print(header)
    print('-' * 110)
    for cls_name in CLASS_NAMES:
        row = f'{cls_name:>5} {GROUP_LABELS[CLASS_NAMES.index(cls_name)]:<14}'
        means = {}
        for m in method_order:
            vals = []
            for seed in args.seeds:
                key = (m, seed)
                if key in results:
                    v = results[key]['per_class_ap75'].get(cls_name, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
                means[m] = mean
            else:
                row += f' {"N/A":>22}'
        if len(means) == 2:
            delta = means['RF+Heun (RF+Heun)'] - means['DDPM baseline (DDPM)']
            row += f' {delta:>+14.4f}'
        print(row)
    print()

    # ---- Paired Wilcoxon test: RF+Heun vs DDPM baseline ----
    print('=' * 110)
    print('Paired Wilcoxon Signed-Rank Test: RF+Heun (RF+Heun) vs DDPM baseline (DDPM)')
    print('(paired by seed × class; n = n_seeds × 24)')
    print('=' * 110)
    a1_vals = []
    a0_vals = []
    for seed in args.seeds:
        a1_key = ('RF+Heun (RF+Heun)', seed)
        a0_key = ('DDPM baseline (DDPM)', seed)
        if a1_key in results and a0_key in results:
            a1_ap = results[a1_key]['per_class_ap']
            a0_ap = results[a0_key]['per_class_ap']
            for cls_name in CLASS_NAMES:
                a1_vals.append(a1_ap.get(cls_name, 0.0))
                a0_vals.append(a0_ap.get(cls_name, 0.0))
    if len(a1_vals) >= 5:
        mean_delta = float(np.mean(np.array(a1_vals) - np.array(a0_vals)))
        stat, p = _wilcoxon_paired(a1_vals, a0_vals)
        sig = ('***' if p < 0.001 else
               '**' if p < 0.01 else
               '*' if p < 0.05 else
               'n.s.')
        print(f'  All classes:  Δ mean={mean_delta:+.4f}  W={stat:.1f}  p={p:.4e}  {sig}')

    # Small classes only
    a1_small = []
    a0_small = []
    for seed in args.seeds:
        a1_key = ('RF+Heun (RF+Heun)', seed)
        a0_key = ('DDPM baseline (DDPM)', seed)
        if a1_key in results and a0_key in results:
            a1_ap = results[a1_key]['per_class_ap']
            a0_ap = results[a0_key]['per_class_ap']
            for cls_name in SMALL_CLASSES:
                a1_small.append(a1_ap.get(cls_name, 0.0))
                a0_small.append(a0_ap.get(cls_name, 0.0))
    if len(a1_small) >= 5:
        mean_delta = float(np.mean(np.array(a1_small) - np.array(a0_small)))
        stat, p = _wilcoxon_paired(a1_small, a0_small)
        sig = ('***' if p < 0.001 else
               '**' if p < 0.01 else
               '*' if p < 0.05 else
               'n.s.')
        print(f'  Small classes (Y/G22/F19/F20): Δ mean={mean_delta:+.4f}  W={stat:.1f}  p={p:.4e}  {sig}')

    # Large classes only
    a1_large = []
    a0_large = []
    for seed in args.seeds:
        a1_key = ('RF+Heun (RF+Heun)', seed)
        a0_key = ('DDPM baseline (DDPM)', seed)
        if a1_key in results and a0_key in results:
            a1_ap = results[a1_key]['per_class_ap']
            a0_ap = results[a0_key]['per_class_ap']
            for cls_name in LARGE_CLASSES:
                a1_large.append(a1_ap.get(cls_name, 0.0))
                a0_large.append(a0_ap.get(cls_name, 0.0))
    if len(a1_large) >= 5:
        mean_delta = float(np.mean(np.array(a1_large) - np.array(a0_large)))
        stat, p = _wilcoxon_paired(a1_large, a0_large)
        sig = ('***' if p < 0.001 else
               '**' if p < 0.01 else
               '*' if p < 0.05 else
               'n.s.')
        print(f'  Large classes (RF+Heun/+AdaLN-Zero/+Stoch. Coupling):      Δ mean={mean_delta:+.4f}  W={stat:.1f}  p={p:.4e}  {sig}')
    print()

    # ---- Save final JSON ----
    output_data = {
        'experiment': 'rf_vs_ddpm_per_class_ap_3seed',
        'description': 'Per-class AP for RF+Heun (RF+Heun) vs DDPM baseline (DDPM) × seeds on Dataset 2 (24obj)',
        'ann_file': ANN_FILE,
        'dataset': '24_chromosomes_object (valid)',
        'class_names': CLASS_NAMES,
        'group_labels': GROUP_LABELS,
        'small_classes': SMALL_CLASSES,
        'large_classes': LARGE_CLASSES,
        'seeds': args.seeds,
        'methods': method_order,
        'method_paths': {
            'a0': {str(k): {'config': v[0], 'checkpoint': v[1]} for k, v in A0_PATHS.items()},
            'a1': {str(k): {'config': v[0], 'checkpoint': v[1]} for k, v in A1_PATHS.items()},
        },
        'note': (
            'DDPM baseline (a0_baseline_24obj): DDPM 1-step Euler, no RF, no AdaLN, no StochOT. '
            'RF+Heun (a1_rf_heun_24obj): +RF +Heun 4-step, no AdaLN, no StochOT. '
            'No special config modification is needed; each method loads its own '
            'config which already specifies the correct sampler and timesteps. '
            'seed 42 DDPM baseline/RF+Heun checkpoints are on ross server; default seeds [123, 789].'
        ),
        'results': {
            f'{m}__seed{seed}': {
                'method': m,
                'seed': seed,
                'aggregate': results[(m, seed)]['aggregate'],
                'per_class_ap': results[(m, seed)]['per_class_ap'],
                'per_class_ap50': results[(m, seed)]['per_class_ap50'],
                'per_class_ap75': results[(m, seed)]['per_class_ap75'],
            }
            for m in method_order
            for seed in args.seeds
            if (m, seed) in results
        },
        'summary_per_class': summary_per_class,
        'delta_per_class_a1_minus_a0': delta_per_class,
        'cache_dir': str(OUTPUT_DIR),
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\nResults saved to: {OUTPUT_JSON}')
    print(f'Per-run caches in: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
