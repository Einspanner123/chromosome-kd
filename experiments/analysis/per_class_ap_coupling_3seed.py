#!/usr/bin/env python3
"""Per-class AP analysis: coupling strategies × 3 seeds (Dataset 1).

Runs inference on 9 checkpoints (3 coupling strategies × 3 seeds) trained on
Dataset 1 (Chromosome20240904_NoAug_NoResize_coco, 1540 imgs) and produces:
  - per-class AP @ IoU=0.5:0.95 for each (strategy, seed)
  - mean ± std per strategy (especially Y chromosome)
  - paired Wilcoxon signed-rank tests on per-class AP
  - JSON dump with raw + aggregate results

Background:
  - The user is rewriting paper_draft_CN.md §1.2 and needs to verify whether
    "Stochastic Coupling mitigates Y chromosome training instability" can be
    supported by experimental data.
  - All 9 checkpoints live under work_dirs/multi_seed/{hard_ot,rf_heun_adaln,
    stochot_eps5_old}/seed_{42,123,789}/.
  - Reference scalars.json values (Table C.2):
      Hard OT:    0.705 / 0.703 / 0.707  (mean 0.705, std 0.002)
      Random:     0.713 / 0.718 / 0.708  (mean 0.713, std 0.004)
      Stoch:      0.745 / 0.745 / 0.750  (mean 0.747, std 0.001)

Inference-side note:
  Coupling (ot_coupling / ot_flow / hard_ot) is a *training-time* strategy
  that affects how noise is paired with GT boxes; at inference time the
  noise-to-box assignment is always random. So we disable OT coupling on all
  3 configs to ensure a fair apples-to-apples comparison (this mirrors the
  treatment in baseline_vs_sota.py).
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

OUTPUT_DIR_REF = REPO  # used by _resolve below

# Dataset 1 (Chromosome20240904, 1540 imgs)
ANN_FILE = str(
    REPO / 'data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json'
)

CLASS_NAMES = [
    'A1', 'A2', 'A3',
    'B4', 'B5',
    'C10', 'C11', 'C12', 'C6', 'C7', 'C8', 'C9',
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

# 9 checkpoints: (strategy, seed, config_rel, ckpt_rel)
STRATEGIES = [
    ('Hard OT', 'work_dirs/multi_seed/hard_ot', 'hard_ot.py'),
    ('Random (AdaLN)', 'work_dirs/multi_seed/rf_heun_adaln', 'rf_heun_adaln.py'),
    ('Stochastic Coupling', 'work_dirs/multi_seed/stochot_eps5_old', 'stochot_eps5_old_multiseed.py'),
]
SEEDS = [42, 123, 789]

OUTPUT_DIR = REPO / 'experiments/analysis/per_class_ap_coupling_3seed_cache'
OUTPUT_JSON = REPO / 'experiments/analysis/per_class_ap_coupling_3seed_results.json'

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


def _find_best_ckpt(strategy_dir: Path, seed: int) -> Path | None:
    """Locate best_coco_bbox_mAP_epoch_*.pth in a multi_seed run directory."""
    pattern = f'best_coco_bbox_mAP_epoch_*.pth'
    matches = sorted(strategy_dir.glob(pattern))
    return matches[0] if matches else None


def _load_config_safe(config_path: Path) -> Config:
    """Load config with custom_imports handling.

    All 3 multi_seed configs already use 'experiments.mmdet_bridge.*' imports
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


def _disable_ot_coupling(cfg: Config) -> Config:
    """Disable OT coupling at inference time.

    Coupling strategies only affect how training-time noise is paired with GT
    boxes; inference always uses random noise-to-box assignment. For a fair
    comparison across Hard OT / Random / Stoch Coupling models, we set
    ot_coupling=False and strip ot_* kwargs so the head __init__ doesn't reject
    unknown args.
    """
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'bbox_head'):
        bh = cfg.model.bbox_head
        if hasattr(bh, 'ot_coupling'):
            bh.ot_coupling = False
            print(f'  [FIX] Disabled ot_coupling (inference)')
        # Strip any ot_* kwargs from the coupling dict, keep coupling dict
        # intact so the head __init__ still receives the registered strategy
        # but it won't be invoked because ot_coupling=False.
        if hasattr(bh, 'coupling') and isinstance(bh.coupling, dict):
            # Keep coupling dict; behavior controlled by ot_coupling=False
            pass
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
    """Compute aggregate + per-class AP from COCO DT predictions."""
    if not predictions:
        # No detections: zero everywhere
        aggregate = {k: 0.0 for k in
                     ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l']}
        per_class_ap = {n: 0.0 for n in CLASS_NAMES}
        per_class_ap50 = {n: 0.0 for n in CLASS_NAMES}
        return {'aggregate': aggregate,
                'per_class_ap': per_class_ap,
                'per_class_ap50': per_class_ap50}

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

    # per-class AP @ IoU=0.5:0.95 and IoU=0.5
    # precision shape: [T, R, K, A, M]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategies', nargs='+',
        default=['Hard OT', 'Random (AdaLN)', 'Stochastic Coupling'],
        help='Subset of strategies to run (default: all 3)')
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[42, 123, 789],
        help='Seeds to run (default: 42 123 789)')
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip (strategy, seed) pairs whose JSON cache exists')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 100)
    print('Per-Class AP Analysis: Coupling Strategies × 3 Seeds (Dataset 1)')
    print('=' * 100)
    print(f'Device: {DEVICE}')
    print(f'Annotation: {ANN_FILE}')
    print(f'Output cache: {OUTPUT_DIR}')
    print(f'Final JSON:  {OUTPUT_JSON}')
    print(f'Strategies: {args.strategies}')
    print(f'Seeds:      {args.seeds}')
    print()

    # Load COCO GT once
    coco_gt = COCO(ANN_FILE)
    print(f'COCO GT: {len(coco_gt.imgs)} imgs, {len(coco_gt.cats)} cats')
    print()

    # Build run plan
    plan: list[tuple[str, int, Path, Path, Path]] = []
    for strat_name, strat_dir_rel, config_name in STRATEGIES:
        if strat_name not in args.strategies:
            continue
        strat_dir = _resolve(strat_dir_rel)
        for seed in args.seeds:
            if seed not in args.seeds:
                continue
            seed_dir = strat_dir / f'seed_{seed}'
            config_path = seed_dir / config_name
            ckpt_path = _find_best_ckpt(seed_dir, seed)
            cache_path = OUTPUT_DIR / f'{strat_name.replace(" ", "_")}_seed{seed}.json'
            if ckpt_path is None:
                print(f'[MISS] {strat_name} seed{seed}: no best ckpt in {seed_dir}')
                continue
            if args.skip_existing and cache_path.exists():
                print(f'[SKIP] {strat_name} seed{seed}: cache exists')
                continue
            plan.append((strat_name, seed, config_path, ckpt_path, cache_path))

    if not plan:
        print('Nothing to run.')
        return

    print(f'Will run {len(plan)} inference jobs:')
    for s, sd, cfg, ckpt, _ in plan:
        print(f'  - {s} seed{sd}')
        print(f'      cfg={cfg.name}')
        print(f'      ckpt={ckpt.name}')
    print()

    # ---- Run inference + compute metrics ----
    results: dict[tuple[str, int], dict] = {}

    for strat_name, seed, config_path, ckpt_path, cache_path in plan:
        print('-' * 100)
        print(f'>>> {strat_name}  seed={seed}')
        print(f'    cfg:  {config_path}')
        print(f'    ckpt: {ckpt_path}')
        print(f'    cache: {cache_path}')

        cfg = _load_config_safe(config_path)
        cfg = _disable_ot_coupling(cfg)

        predictions = _run_inference(cfg, ckpt_path, DEVICE)
        metrics = _compute_metrics(coco_gt, predictions)

        # Persist per-run cache (predictions + metrics)
        cache_data = {
            'strategy': strat_name,
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
              f'AP_S={metrics["aggregate"]["AP_s"]:.4f}  '
              f'AP_M={metrics["aggregate"]["AP_m"]:.4f}  '
              f'AP_L={metrics["aggregate"]["AP_l"]:.4f}')
        y_ap = metrics['per_class_ap'].get('Y', float('nan'))
        print(f'    Y AP={y_ap:.4f}')

        results[(strat_name, seed)] = metrics
        print()

    # ---- Also load any skipped caches so we have the full picture ----
    for strat_name, _, _ in STRATEGIES:
        if strat_name not in args.strategies:
            continue
        for seed in args.seeds:
            key = (strat_name, seed)
            if key in results:
                continue
            cache_path = OUTPUT_DIR / f'{strat_name.replace(" ", "_")}_seed{seed}.json'
            if cache_path.exists():
                with open(cache_path) as f:
                    cache_data = json.load(f)
                results[key] = cache_data['metrics']
                print(f'[LOAD] {strat_name} seed{seed} from cache')

    # ---- Summary table: per-class AP across (strategy × seed) ----
    strat_order = [s[0] for s in STRATEGIES if s[0] in args.strategies]
    print()
    print('=' * 130)
    print('Per-Class AP @ IoU=0.5:0.95 (mean over 3 seeds)')
    print('=' * 130)
    header = f'{"Class":>5} {"Group":<14}'
    for s in strat_order:
        header += f' {s:>30} mean±std'
    print(header)
    print('-' * 130)

    summary_per_class: dict[str, dict[str, dict]] = {}
    for cls_name in CLASS_NAMES:
        row = f'{cls_name:>5} {GROUP_LABELS[CLASS_NAMES.index(cls_name)]:<14}'
        summary_per_class[cls_name] = {}
        for s in strat_order:
            vals = []
            for seed in args.seeds:
                key = (s, seed)
                if key in results:
                    v = results[key]['per_class_ap'].get(cls_name, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
                summary_per_class[cls_name][s] = {
                    'mean': mean, 'std': std,
                    'values': vals,
                }
            else:
                row += f' {"N/A":>26}'
        print(row)

    # ---- Aggregate row ----
    print('-' * 130)
    agg_metrics = ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l']
    for agg in agg_metrics:
        row = f'{agg:>5} {"":<14}'
        for s in strat_order:
            vals = []
            for seed in args.seeds:
                key = (s, seed)
                if key in results:
                    v = results[key]['aggregate'].get(agg, float('nan'))
                    vals.append(v)
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=0))
                row += f' {mean:>7.4f}±{std:.4f}        '
            else:
                row += f' {"N/A":>26}'
        print(row)
    print()

    # ---- Y chromosome focus ----
    print('=' * 130)
    print('Y CHROMOSOME ANALYSIS (the original motivation)')
    print('=' * 130)
    y_per_seed = {}
    for s in strat_order:
        y_per_seed[s] = []
        for seed in args.seeds:
            key = (s, seed)
            if key in results:
                y_per_seed[s].append(results[key]['per_class_ap'].get('Y', float('nan')))

    print(f'{"Strategy":<24} {"seed42":>10} {"seed123":>10} {"seed789":>10} {"mean":>10} {"std":>10} {"cv(%)":>10}')
    print('-' * 90)
    for s in strat_order:
        vals = y_per_seed[s]
        if len(vals) == 3:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=0))
            cv = (std / mean * 100) if mean > 0 else 0
            print(f'{s:<24} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {mean:>10.4f} {std:>10.4f} {cv:>10.2f}')
    print()

    # ---- Paired Wilcoxon tests on per-class AP ----
    print('=' * 130)
    print('Paired Wilcoxon Signed-Rank Tests on Per-Class AP (3 seeds × 24 classes = 72 pairs)')
    print('=' * 130)
    if len(strat_order) == 3:
        comparisons = [
            ('Stochastic Coupling', 'Hard OT'),
            ('Stochastic Coupling', 'Random (AdaLN)'),
            ('Random (AdaLN)', 'Hard OT'),
        ]
        print(f'{"Comparison":<55} {"Δ mean":>10} {"Wilcoxon W":>12} {"p-value":>12} {"Significance":>15}')
        print('-' * 110)
        for a_strat, b_strat in comparisons:
            a_vals = []
            b_vals = []
            for seed in args.seeds:
                a_key = (a_strat, seed)
                b_key = (b_strat, seed)
                if a_key in results and b_key in results:
                    a_ap = results[a_key]['per_class_ap']
                    b_ap = results[b_key]['per_class_ap']
                    for cls_name in CLASS_NAMES:
                        a_vals.append(a_ap.get(cls_name, 0.0))
                        b_vals.append(b_ap.get(cls_name, 0.0))
            if len(a_vals) >= 5 and len(b_vals) >= 5:
                mean_delta = float(np.mean(np.array(a_vals) - np.array(b_vals)))
                stat, p = _wilcoxon_paired(a_vals, b_vals)
                sig = ('***' if p < 0.001 else
                       '**' if p < 0.01 else
                       '*' if p < 0.05 else
                       'n.s.')
                print(f'{a_strat} vs {b_strat:<30} {mean_delta:>+10.4f} {stat:>12.1f} {p:>12.4e} {sig:>15}')
        print()

    # ---- Save final JSON ----
    output_data = {
        'experiment': 'coupling_strategy_per_class_ap_3seed',
        'description': 'Per-class AP for Hard OT / Random / Stoch Coupling × 3 seeds on Dataset 1',
        'ann_file': ANN_FILE,
        'dataset': 'Chromosome20240904_NoAug_NoResize_coco (valid, 440 imgs; train=1540)',
        'class_names': CLASS_NAMES,
        'group_labels': GROUP_LABELS,
        'seeds': args.seeds,
        'strategies': strat_order,
        'note': (
            'OT coupling was disabled at inference (ot_coupling=False). '
            'Coupling is a training-time strategy; inference always uses random '
            'noise-to-box assignment. This mirrors the treatment in baseline_vs_sota.py.'
        ),
        'results': {
            f'{s}__seed{seed}': {
                'strategy': s,
                'seed': seed,
                'aggregate': results[(s, seed)]['aggregate'],
                'per_class_ap': results[(s, seed)]['per_class_ap'],
                'per_class_ap50': results[(s, seed)]['per_class_ap50'],
            }
            for s in strat_order
            for seed in args.seeds
            if (s, seed) in results
        },
        'summary_per_class': summary_per_class,
        'y_chromosome_per_seed': {
            s: y_per_seed[s] for s in strat_order
        },
        'cache_dir': str(OUTPUT_DIR),
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\nResults saved to: {OUTPUT_JSON}')
    print(f'Per-run caches in: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
