#!/usr/bin/env python3
"""Per-class AP analysis: box_renewal ON vs OFF × 3 seeds (Dataset 2 / 24obj).

Runs inference on A3 (a4_dpm_pp_24obj) checkpoints with box_renewal toggled
on/off and produces:
  - per-class AP @ IoU=0.5:0.95 (and AP@0.5, AP@0.75) for each (condition, seed)
  - mean ± std per condition across 3 seeds
  - paired Wilcoxon signed-rank tests on per-class AP
  - JSON dump with raw + aggregate results
  - focus on small classes (Y, G22, F19, F20) vs large classes (A1, A2, A3)

Background:
  - D3 log-based analysis (r1_eta_str_a3_seed{42,123,789}{,_noRenewal}.log)
    showed aggregate bbox_mAP_75 Δ(on-off)=0.0000, but small classes had
    mean Δ=-0.0024 vs large classes +0.0008. This script re-runs inference
    with strict COCO AP@0.5:0.95 (not the mmengine classwise metric) to
    verify whether box_renewal significantly affects small-class AP.
  - A3 checkpoints (a4_dpm_pp_24obj, DPM-Solver++ 4-step) are the same
    checkpoints used in the D3 log analysis.
  - seed 42 checkpoint lives in work_dirs/a4_dpm_pp_24obj/ (not multi_seed/).

Inference-side note:
  - box_renewal is an inference-time mechanism (head.py line 708): after
    sampling, raw boxes are refined by re-encoding the predicted bbox and
    re-decoding through the box decoder. Disabling it (box_renewal=False)
    returns the raw sampler output without refinement.
  - Unlike OT coupling (training-only), box_renewal DOES affect inference,
    so we toggle it directly on the loaded config.
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

# 2 conditions: (name, box_renewal_enabled)
CONDITIONS = [
    ('Renewal ON', True),
    ('Renewal OFF', False),
]

SEEDS = [42, 123, 789]

# A3 (a4_dpm_pp_24obj) checkpoint + config paths per seed.
# seed 42 lives in work_dirs/a4_dpm_pp_24obj/ (not multi_seed/);
# seeds 123/789 live in work_dirs/multi_seed/a4_dpm_pp_24obj/seed_{N}/.
A3_PATHS = {
    42: (
        'work_dirs/a4_dpm_pp_24obj/a4_dpm_pp_24obj.py',
        'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
    ),
    123: (
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/a4_dpm_pp_24obj_multiseed.py',
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth',
    ),
    789: (
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/a4_dpm_pp_24obj_multiseed.py',
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth',
    ),
}

OUTPUT_DIR = REPO / 'experiments/analysis/per_class_ap_renewal_3seed_cache'
OUTPUT_JSON = REPO / 'experiments/analysis/per_class_ap_renewal_3seed_results.json'

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

    A3 multi_seed configs already use 'experiments.mmdet_bridge.*' imports
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


def _set_box_renewal(cfg: Config, enabled: bool) -> Config:
    """Toggle box_renewal on the bbox_head.

    box_renewal is an inference-time mechanism (head.py line 708): after
    sampling, raw boxes are refined by re-encoding the predicted bbox and
    re-decoding through the box decoder. Setting False returns raw sampler
    output without refinement.
    """
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'bbox_head'):
        cfg.model.bbox_head.box_renewal = enabled
        print(f'  [SET] box_renewal = {enabled}')
    else:
        print(f'  [WARN] cfg.model.bbox_head not found; box_renewal not set')
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
        '--conditions', nargs='+',
        default=['Renewal ON', 'Renewal OFF'],
        help='Subset of conditions to run (default: both)')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[42, 123, 789],
        help='Seeds to run (default: 42 123 789)')
    parser.add_argument(
        '--device', type=str, default=DEVICE,
        help=f'Device for inference (default: {DEVICE})')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip (condition, seed) pairs whose JSON cache exists')
    args = parser.parse_args()
    DEVICE = args.device

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print('Per-Class AP Analysis: box_renewal ON vs OFF × 3 Seeds (Dataset 2 / 24obj)')
    print('=' * 100)
    print(f'Device: {DEVICE}')
    print(f'Annotation: {ANN_FILE}')
    print(f'Output cache: {OUTPUT_DIR}')
    print(f'Final JSON:  {OUTPUT_JSON}')
    print(f'Conditions: {args.conditions}')
    print(f'Seeds:      {args.seeds}')
    print()

    # Load COCO GT once
    coco_gt = COCO(ANN_FILE)
    print(f'COCO GT: {len(coco_gt.imgs)} imgs, {len(coco_gt.cats)} cats')
    print()

    # Build run plan: (condition_name, box_renewal_enabled, seed, config_path, ckpt_path, cache_path)
    plan: list[tuple[str, bool, int, Path, Path, Path]] = []
    for cond_name, cond_enabled in CONDITIONS:
        if cond_name not in args.conditions:
            continue
        for seed in args.seeds:
            if seed not in args.seeds:
                continue
            if seed not in A3_PATHS:
                print(f'[MISS] {cond_name} seed{seed}: no A3 checkpoint path configured')
                continue
            config_rel, ckpt_rel = A3_PATHS[seed]
            config_path = _resolve(config_rel)
            ckpt_path = _resolve(ckpt_rel)
            cache_path = OUTPUT_DIR / f'{cond_name.replace(" ", "_")}_seed{seed}.json'
            if not ckpt_path.exists():
                print(f'[MISS] {cond_name} seed{seed}: ckpt not found {ckpt_path}')
                continue
            if not config_path.exists():
                print(f'[MISS] {cond_name} seed{seed}: config not found {config_path}')
                continue
            if args.skip_existing and cache_path.exists():
                print(f'[SKIP] {cond_name} seed{seed}: cache exists')
                continue
            plan.append((cond_name, cond_enabled, seed, config_path, ckpt_path, cache_path))

    if not plan:
        print('Nothing to run. (Use without --skip-existing to force re-run)')
        return

    print(f'Will run {len(plan)} inference jobs:')
    for cn, _, sd, cfg, ckpt, _ in plan:
        print(f'  - {cn} seed{sd}')
        print(f'      cfg={cfg.name}')
        print(f'      ckpt={ckpt.name}')
    print()

    # ---- Run inference + compute metrics ----
    results: dict[tuple[str, int], dict] = {}

    for cond_name, cond_enabled, seed, config_path, ckpt_path, cache_path in plan:
        print('-' * 100)
        print(f'>>> {cond_name}  seed={seed}  (box_renewal={cond_enabled})')
        print(f'    cfg:  {config_path}')
        print(f'    ckpt: {ckpt_path}')
        print(f'    cache: {cache_path}')

        cfg = _load_config_safe(config_path)
        cfg = _set_box_renewal(cfg, cond_enabled)

        predictions = _run_inference(cfg, ckpt_path, DEVICE)
        metrics = _compute_metrics(coco_gt, predictions)

        cache_data = {
            'condition': cond_name,
            'box_renewal': cond_enabled,
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

        results[(cond_name, seed)] = metrics
        print()

    # ---- Also load any skipped caches so we have the full picture ----
    for cond_name, _ in CONDITIONS:
        if cond_name not in args.conditions:
            continue
        for seed in args.seeds:
            key = (cond_name, seed)
            if key in results:
                continue
            cache_path = OUTPUT_DIR / f'{cond_name.replace(" ", "_")}_seed{seed}.json'
            if cache_path.exists():
                with open(cache_path) as f:
                    cache_data = json.load(f)
                results[key] = cache_data['metrics']
                print(f'[LOAD] {cond_name} seed{seed} from cache')

    # ---- Summary table: per-class AP across (condition × seed) ----
    cond_order = [c[0] for c in CONDITIONS if c[0] in args.conditions]
    print()
    print('=' * 110)
    print('Per-Class AP @ IoU=0.5:0.95 (mean over seeds)')
    print('=' * 110)
    header = f'{"Class":>5} {"Group":<14}'
    for c in cond_order:
        header += f' {c:>26} mean±std'
    header += f' {"Δ(ON-OFF)":>14}'
    print(header)
    print('-' * 110)

    summary_per_class: dict[str, dict[str, dict]] = {}
    delta_per_class: dict[str, float] = {}
    for cls_name in CLASS_NAMES:
        row = f'{cls_name:>5} {GROUP_LABELS[CLASS_NAMES.index(cls_name)]:<14}'
        summary_per_class[cls_name] = {}
        means = {}
        for c in cond_order:
            vals = []
            for seed in args.seeds:
                key = (c, seed)
                if key in results:
                    v = results[key]['per_class_ap'].get(cls_name, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
                summary_per_class[cls_name][c] = {
                    'mean': mean, 'std': std,
                    'values': vals,
                }
                means[c] = mean
            else:
                row += f' {"N/A":>22}'
        if len(means) == 2:
            delta = means['Renewal ON'] - means['Renewal OFF']
            row += f' {delta:>+14.4f}'
            delta_per_class[cls_name] = delta
        print(row)

    # ---- Aggregate row ----
    print('-' * 110)
    agg_metrics = ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l']
    for agg in agg_metrics:
        row = f'{agg:>5} {"":<14}'
        means = {}
        for c in cond_order:
            vals = []
            for seed in args.seeds:
                key = (c, seed)
                if key in results:
                    v = results[key]['aggregate'].get(agg, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
                means[c] = mean
            else:
                row += f' {"N/A":>22}'
        if len(means) == 2:
            delta = means['Renewal ON'] - means['Renewal OFF']
            row += f' {delta:>+14.4f}'
        print(row)
    print()

    # ---- Small vs large class focus ----
    print('=' * 110)
    print('SMALL vs LARGE CLASS ANALYSIS (Δ = Renewal ON - Renewal OFF)')
    print('=' * 110)
    print(f'{"Group":<12} {"Classes":<28} {"Mean Δ":>10} {"Min Δ":>10} {"Max Δ":>10}')
    print('-' * 80)
    for group_name, group_cls in [('Small', SMALL_CLASSES), ('Large', LARGE_CLASSES)]:
        deltas = [delta_per_class[c] for c in group_cls if c in delta_per_class]
        if deltas:
            print(f'{group_name:<12} {", ".join(group_cls):<28} '
                  f'{np.mean(deltas):>+10.4f} {min(deltas):>+10.4f} {max(deltas):>+10.4f}')
    print()

    # ---- Per-class AP@0.75 table (for cross-validation with D3 logs) ----
    print('=' * 110)
    print('Per-Class AP @ IoU=0.75 (cross-validation with D3 log-based mAP_75)')
    print('=' * 110)
    header = f'{"Class":>5} {"Group":<14}'
    for c in cond_order:
        header += f' {c:>26} mean±std'
    header += f' {"Δ(ON-OFF)":>14}'
    print(header)
    print('-' * 110)
    for cls_name in CLASS_NAMES:
        row = f'{cls_name:>5} {GROUP_LABELS[CLASS_NAMES.index(cls_name)]:<14}'
        means = {}
        for c in cond_order:
            vals = []
            for seed in args.seeds:
                key = (c, seed)
                if key in results:
                    v = results[key]['per_class_ap75'].get(cls_name, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
                means[c] = mean
            else:
                row += f' {"N/A":>22}'
        if len(means) == 2:
            delta = means['Renewal ON'] - means['Renewal OFF']
            row += f' {delta:>+14.4f}'
        print(row)
    print()

    # ---- Paired Wilcoxon test: Renewal ON vs OFF ----
    print('=' * 110)
    print('Paired Wilcoxon Signed-Rank Test: Renewal ON vs OFF')
    print('(paired by seed × class; n = n_seeds × 24)')
    print('=' * 110)
    on_vals = []
    off_vals = []
    for seed in args.seeds:
        on_key = ('Renewal ON', seed)
        off_key = ('Renewal OFF', seed)
        if on_key in results and off_key in results:
            on_ap = results[on_key]['per_class_ap']
            off_ap = results[off_key]['per_class_ap']
            for cls_name in CLASS_NAMES:
                on_vals.append(on_ap.get(cls_name, 0.0))
                off_vals.append(off_ap.get(cls_name, 0.0))
    if len(on_vals) >= 5:
        mean_delta = float(np.mean(np.array(on_vals) - np.array(off_vals)))
        stat, p = _wilcoxon_paired(on_vals, off_vals)
        sig = ('***' if p < 0.001 else
               '**' if p < 0.01 else
               '*' if p < 0.05 else
               'n.s.')
        print(f'  All classes:  Δ mean={mean_delta:+.4f}  W={stat:.1f}  p={p:.4e}  {sig}')

    # Small classes only
    on_small = []
    off_small = []
    for seed in args.seeds:
        on_key = ('Renewal ON', seed)
        off_key = ('Renewal OFF', seed)
        if on_key in results and off_key in results:
            on_ap = results[on_key]['per_class_ap']
            off_ap = results[off_key]['per_class_ap']
            for cls_name in SMALL_CLASSES:
                on_small.append(on_ap.get(cls_name, 0.0))
                off_small.append(off_ap.get(cls_name, 0.0))
    if len(on_small) >= 5:
        mean_delta = float(np.mean(np.array(on_small) - np.array(off_small)))
        stat, p = _wilcoxon_paired(on_small, off_small)
        sig = ('***' if p < 0.001 else
               '**' if p < 0.01 else
               '*' if p < 0.05 else
               'n.s.')
        print(f'  Small classes (Y/G22/F19/F20): Δ mean={mean_delta:+.4f}  W={stat:.1f}  p={p:.4e}  {sig}')

    # Large classes only
    on_large = []
    off_large = []
    for seed in args.seeds:
        on_key = ('Renewal ON', seed)
        off_key = ('Renewal OFF', seed)
        if on_key in results and off_key in results:
            on_ap = results[on_key]['per_class_ap']
            off_ap = results[off_key]['per_class_ap']
            for cls_name in LARGE_CLASSES:
                on_large.append(on_ap.get(cls_name, 0.0))
                off_large.append(off_ap.get(cls_name, 0.0))
    if len(on_large) >= 5:
        mean_delta = float(np.mean(np.array(on_large) - np.array(off_large)))
        stat, p = _wilcoxon_paired(on_large, off_large)
        sig = ('***' if p < 0.001 else
               '**' if p < 0.01 else
               '*' if p < 0.05 else
               'n.s.')
        print(f'  Large classes (A1/A2/A3):      Δ mean={mean_delta:+.4f}  W={stat:.1f}  p={p:.4e}  {sig}')
    print()

    # ---- Save final JSON ----
    output_data = {
        'experiment': 'box_renewal_per_class_ap_3seed',
        'description': 'Per-class AP for box_renewal ON vs OFF × 3 seeds on Dataset 2 (24obj)',
        'ann_file': ANN_FILE,
        'dataset': '24_chromosomes_object (valid)',
        'class_names': CLASS_NAMES,
        'group_labels': GROUP_LABELS,
        'small_classes': SMALL_CLASSES,
        'large_classes': LARGE_CLASSES,
        'seeds': args.seeds,
        'conditions': cond_order,
        'a3_paths': {str(k): {'config': v[0], 'checkpoint': v[1]} for k, v in A3_PATHS.items()},
        'note': (
            'box_renewal is an inference-time mechanism (head.py line 708). '
            'Setting box_renewal=False returns raw sampler output without '
            'refinement. Both conditions use the SAME A3 checkpoints; only the '
            'inference config differs.'
        ),
        'results': {
            f'{c}__seed{seed}': {
                'condition': c,
                'seed': seed,
                'aggregate': results[(c, seed)]['aggregate'],
                'per_class_ap': results[(c, seed)]['per_class_ap'],
                'per_class_ap50': results[(c, seed)]['per_class_ap50'],
                'per_class_ap75': results[(c, seed)]['per_class_ap75'],
            }
            for c in cond_order
            for seed in args.seeds
            if (c, seed) in results
        },
        'summary_per_class': summary_per_class,
        'delta_per_class_on_minus_off': delta_per_class,
        'cache_dir': str(OUTPUT_DIR),
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\nResults saved to: {OUTPUT_JSON}')
    print(f'Per-run caches in: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
