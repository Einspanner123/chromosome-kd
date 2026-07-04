#!/usr/bin/env python3
"""Baseline vs SOTA 综合对比分析 — 多数据集版

收集并对比以下指标 (步数对齐 4-step):
1. 每类 AP (COCO mAP per class, IoU=0.5:0.95)
2. 混淆矩阵 (24×24+1, IoU=0.5, score=0.3)
3. 聚合 P/R 指标 (mAP/AP50/AP75/AR + 混淆矩阵 P/R/F1)

支持数据集与模型:
- chromo (Chromosome20240904):
    - DiffusionDet (3-seed, baseline 0.729)
    - SOTA (1-seed, Sinkhorn Stochastic OT, 训练 best 0.753, 推理 mAP=0.748)
- 24obj (24_chromosomes_object):
    - DiffusionDet (1-seed, baseline)
    - SOTA (1-seed, Sinkhorn Stochastic OT eps=5, 训练 best 0.853)

输出:
    experiments/analysis/baseline_vs_sota_results.json
    experiments/analysis/baseline_vs_sota_cache/  (每模型×seed 的预测缓存)

Usage:
    python experiments/analysis/baseline_vs_sota.py
    python experiments/analysis/baseline_vs_sota.py --force-inference  # 强制重推理
    python experiments/analysis/baseline_vs_sota.py --device cuda:1
    python experiments/analysis/baseline_vs_sota.py --datasets chromo  # 仅 chromo
    python experiments/analysis/baseline_vs_sota.py --datasets chromo 24obj
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, _ANALYSIS_DIR)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 复用 bottleneck_diagnosis 的混淆矩阵分析
from bottleneck_diagnosis import analyze_confusion_matrix

# ──────────────────────────────────────────────
# 配置: 多数据集
# ──────────────────────────────────────────────

DATASETS = {
    'chromo': {
        'ann_file': 'data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json',
        'models': {
            # DiffusionDet baseline: 3-seed (步数对齐 DDIM 4-step = 1-step = 0.729)
            'DiffusionDet': {
                'config': 'experiments/configs/baselines/diffusiondet_ddpm.py',
                'checkpoints': {
                    42: 'work_dirs/multi_seed_aug/ddpm/seed_42/best_coco_bbox_mAP_epoch_79.pth',
                    123: 'work_dirs/multi_seed_aug/ddpm/seed_123/best_coco_bbox_mAP_epoch_66.pth',
                    789: 'work_dirs/multi_seed_aug/ddpm/seed_789/best_coco_bbox_mAP_epoch_87.pth',
                },
                'sampling_timesteps': 4,
            },
            # SOTA 0.753: 单 checkpoint (reproduce_0751_stochot_eps5_v2, best epoch 59)
            # ⚠ checkpoint 不在当前服务器, 仅使用从另一服务器带来的缓存预测
            'SOTA': {
                'config': 'experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py',
                'checkpoints': {
                    42: 'work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth',
                },
                'sampling_timesteps': 4,
            },
        },
    },
    '24obj': {
        'ann_file': 'data/24_chromosomes_object/coco/valid/_annotations.coco.json',
        'models': {
            # DiffusionDet baseline: 单 checkpoint (benchmark_diffusiondet_24obj, best epoch 26)
            'DiffusionDet': {
                'config': 'experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py',
                'checkpoints': {
                    42: 'work_dirs/benchmark_diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth',
                },
                'sampling_timesteps': 4,
            },
            # SOTA: Sinkhorn Stochastic OT (eps=5), 单 checkpoint (best epoch 89, 训练 mAP=0.853)
            # 使用训练时保存的完整 config (已解析, 无 _base_ 依赖)
            'SOTA': {
                'config': '/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/ldmdet_rf_heun_adaln_stochot_eps5/ldmdet_rf_heun_adaln_stochot_eps5.py',
                'checkpoints': {
                    42: '/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/ldmdet_rf_heun_adaln_stochot_eps5/best_coco_bbox_mAP_epoch_89.pth',
                },
                'sampling_timesteps': 4,
            },
        },
    },
}

SEEDS = [42, 123, 789]
CACHE_DIR = Path(_PROJECT_ROOT) / 'experiments/analysis/baseline_vs_sota_cache'
OUTPUT_PATH = Path(_PROJECT_ROOT) / 'experiments/analysis/baseline_vs_sota_results.json'


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def set_seed(seed: int):
    """固定随机种子, 确保推理可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_inference(config_path: str, checkpoint_path: str, sampling_timesteps: int,
                  device: str = 'cuda:0') -> list[dict]:
    """运行推理, 返回 COCO DT 格式的预测列表

    支持 sampling_timesteps 覆盖 (用于 DDPM 步数对齐)
    自动修正旧 config 的 custom_imports (projects.LDMDet → experiments.mmdet_bridge)
    """
    import tempfile
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    # 旧训练 config (如 24obj SOTA) 引用已删除的 projects.LDMDet, 需在加载前替换 custom_imports
    # Config.fromfile() 会在加载时自动 import custom_imports, 无法在加载后覆盖
    with open(config_path, 'r') as f:
        config_text = f.read()

    if 'projects.LDMDet' in config_text:
        config_text = config_text.replace(
            'projects.LDMDet.model',
            'experiments.mmdet_bridge.registry',
        ).replace(
            'projects.LDMDet.hooks',
            'experiments.mmdet_bridge.hooks',
        )
        # 添加缺失的 detector 和 transforms 导入
        config_text = config_text.replace(
            "'experiments.mmdet_bridge.registry',",
            "'experiments.mmdet_bridge.registry',\n"
            "        'experiments.mmdet_bridge.detector',\n"
            "        'experiments.mmdet_bridge.transforms',",
        )
        print(f'  [FIX] 已替换 custom_imports: projects.LDMDet → experiments.mmdet_bridge')

    # 写入临时文件并加载
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', dir=_PROJECT_ROOT, delete=False
    ) as tmp:
        tmp.write(config_text)
        tmp_path = tmp.name

    try:
        cfg = Config.fromfile(tmp_path)
    finally:
        os.unlink(tmp_path)

    # 覆盖采样步数 (步数对齐)
    cfg.model.bbox_head.sampling_timesteps = sampling_timesteps

    # 旧训练 config 使用 ot_coupling/ot_* 参数 (训练时 OT 配对), 推理时不需要
    # sinkhorn_stochastic 耦合策略未在当前代码注册, 设为 False 使用默认 random 耦合
    if hasattr(cfg.model.bbox_head, 'ot_coupling'):
        cfg.model.bbox_head.ot_coupling = False
        # 移除所有 ot_* 参数, 避免 DiffusionDetHead.__init__ 拒绝未知参数
        for key in list(cfg.model.bbox_head.keys()):
            if key.startswith('ot_'):
                cfg.model.bbox_head.pop(key)
        print(f'  [FIX] 已禁用 ot_coupling (推理不需要)')

    model = init_detector(cfg, checkpoint_path, device=device)

    val_dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(val_dataset_cfg)

    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            out = model.test_step(data)
            for r in (out if isinstance(out, list) else [out]):
                pi = r.pred_instances
                img_id = getattr(r, 'img_id', i)
                for j in range(len(pi.bboxes)):
                    bbox = pi.bboxes[j].cpu().tolist()
                    bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    predictions.append({
                        'image_id': img_id,
                        'category_id': int(pi.labels[j].cpu().item()) + 1,
                        'bbox': bbox_xywh,
                        'score': float(pi.scores[j].cpu().item()),
                    })

            if (i + 1) % 50 == 0:
                print(f'    推理进度: {i+1}/{len(dataset)}')

    # 释放显存
    del model
    torch.cuda.empty_cache()

    return predictions


def compute_coco_metrics(coco_gt: COCO, predictions: list[dict]) -> dict:
    """计算 COCO 标准指标: 聚合 AP/AR + 每类 AP

    COCOeval.stats 有 12 个元素:
        [0] AP all  [1] AP50  [2] AP75
        [3] AP_s  [4] AP_m  [5] AP_l
        [6] AR@1  [7] AR@10  [8] AR@100
        [9] AR_s  [10] AR_m  [11] AR_l

    返回:
        aggregate: {mAP, AP50, AP75, AP_s, AP_m, AP_l, AR@1, AR@10, AR@100, AR_s, AR_m, AR_l}
        per_class_ap: {class_name: ap}  (IoU=0.5:0.95)
        per_class_ap50: {class_name: ap50}  (IoU=0.5)
    """
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
        'AR@1': float(stats[6]),
        'AR@10': float(stats[7]),
        'AR@100': float(stats[8]),
        'AR_s': float(stats[9]),
        'AR_m': float(stats[10]),
        'AR_l': float(stats[11]),
    }

    # 每类 AP (IoU=0.5:0.95)
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    cat_ids = coco_eval.params.catIds
    per_class_ap = {}
    per_class_ap50 = {}

    for k, cat_id in enumerate(cat_ids):
        cat_name = coco_gt.cats[cat_id]['name']

        # AP @[IoU=0.5:0.95], area=all, maxDets=100
        ap_cat = precision[:, :, k, 0, -1]
        ap_cat = ap_cat[ap_cat > -1]
        per_class_ap[cat_name] = float(ap_cat.mean()) if len(ap_cat) > 0 else 0.0

        # AP @[IoU=0.5], area=all, maxDets=100
        ap50_cat = precision[0, :, k, 0, -1]
        ap50_cat = ap50_cat[ap50_cat > -1]
        per_class_ap50[cat_name] = float(ap50_cat.mean()) if len(ap50_cat) > 0 else 0.0

    return {
        'aggregate': aggregate,
        'per_class_ap': per_class_ap,
        'per_class_ap50': per_class_ap50,
    }


def compute_detection_pr(confusion_matrix: np.ndarray, num_classes: int) -> dict:
    """从混淆矩阵计算检测 P/R/F1 (IoU=0.5, score=0.3)

    confusion_matrix: [num_classes+1, num_classes+1]
        - [i, j] (i,j < num_classes): GT class i 匹配到 pred class j
        - [i, num_classes]: GT class i 被漏检 (FN)
        - [num_classes, j]: pred class j 为误检 (FP)
    """
    cm = np.array(confusion_matrix)

    # 聚合
    tp = np.trace(cm[:num_classes, :num_classes])
    fp = cm[num_classes, :num_classes].sum()
    fn = cm[:num_classes, num_classes].sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # 每类
    per_class = {}
    for i in range(num_classes):
        tp_i = cm[i, i]
        # 列和 = 所有预测为 i 的 (含 FP 和误分类)
        total_pred_i = cm[:, i].sum()
        # 行和 = 所有 GT 为 i 的 (含漏检和误分类)
        total_gt_i = cm[i, :].sum()
        p_i = tp_i / max(total_pred_i, 1)
        r_i = tp_i / max(total_gt_i, 1)
        per_class[i] = {
            'precision': float(p_i),
            'recall': float(r_i),
            'f1': float(2 * p_i * r_i / max(p_i + r_i, 1e-8)),
        }

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'per_class': per_class,
    }


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def evaluate_model(model_name: str, model_cfg: dict, coco_gt: COCO,
                   device: str, force_inference: bool, dataset_name: str = '') -> dict:
    """评估单个模型的所有 seeds, 返回聚合结果

    单 checkpoint 模型 (checkpoints 仅含 1 个 seed) 也能正确处理。
    缓存文件名包含 dataset_name 以区分不同数据集。

    缓存策略:
        - 缓存存在且未强制推理 → 加载缓存 (默认, 避免重复推理)
        - 缓存不存在或强制推理 → 运行推理 (checkpoint 不存在则 skip)
    """
    print(f'\n{"="*70}')
    print(f'评估模型: {model_name}' + (f'  [{dataset_name}]' if dataset_name else ''))
    print(f'  Config: {model_cfg["config"]}')
    print(f'  Sampling steps: {model_cfg["sampling_timesteps"]}')
    print(f'{"="*70}')

    # config 可能是绝对路径 (如 24obj SOTA 的训练 config) 或相对项目根的路径
    config_path = model_cfg['config']
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)

    seeds_result = {}
    # 使用 model 自身的 checkpoints keys 作为 seeds (支持单/multi-seed)
    model_seeds = list(model_cfg['checkpoints'].keys())
    ds_prefix = f'{dataset_name}_' if dataset_name else ''

    for seed in model_seeds:
        ckpt_path = model_cfg['checkpoints'][seed]
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(_PROJECT_ROOT, ckpt_path)
        cache_path = CACHE_DIR / f'{ds_prefix}{model_name.replace("+", "_")}_seed{seed}_preds.json'

        print(f'\n--- {model_name} seed={seed} [{dataset_name}] ---')
        print(f'  Checkpoint: {ckpt_path}')

        # 缓存优先 (除非强制推理)
        if cache_path.exists() and not force_inference:
            print(f'  加载缓存: {cache_path}')
            with open(cache_path, 'r') as f:
                predictions = json.load(f)
        else:
            if not os.path.exists(ckpt_path):
                print(f'  [SKIP] Checkpoint 不存在且无缓存')
                continue
            set_seed(seed)
            predictions = run_inference(
                config_path, ckpt_path, model_cfg['sampling_timesteps'], device
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(predictions, f)
            print(f'  缓存已保存: {cache_path}')

        print(f'  总预测数: {len(predictions)}')

        # COCO 指标
        print(f'  计算 COCO 指标...')
        coco_metrics = compute_coco_metrics(coco_gt, predictions)
        agg = coco_metrics['aggregate']
        print(f'  mAP={agg["mAP"]:.4f}  AP50={agg["AP50"]:.4f}  AP75={agg["AP75"]:.4f}')
        print(f'  AP_s={agg["AP_s"]:.4f}  AP_m={agg["AP_m"]:.4f}  AP_l={agg["AP_l"]:.4f}')
        print(f'  AR@100={agg["AR@100"]:.4f}  AR_s={agg["AR_s"]:.4f}  '
              f'AR_m={agg["AR_m"]:.4f}  AR_l={agg["AR_l"]:.4f}')

        # 混淆矩阵
        print(f'  计算混淆矩阵...')
        cm_result = analyze_confusion_matrix(
            coco_gt, predictions, iou_thr=0.5, score_thr=0.3
        )

        # 检测 P/R (从混淆矩阵)
        num_classes = len(cm_result['cat_names'])
        det_pr = compute_detection_pr(
            np.array(cm_result['confusion_matrix']), num_classes
        )
        print(f'  Det P={det_pr["precision"]:.4f}  R={det_pr["recall"]:.4f}  '
              f'F1={det_pr["f1"]:.4f}  (TP={det_pr["tp"]}, FP={det_pr["fp"]}, '
              f'FN={det_pr["fn"]})')

        seeds_result[str(seed)] = {
            'per_class_ap': coco_metrics['per_class_ap'],
            'per_class_ap50': coco_metrics['per_class_ap50'],
            'aggregate': agg,
            'confusion_matrix': cm_result['confusion_matrix'],
            'cat_names': cm_result['cat_names'],
            'per_class_accuracy': cm_result['per_class_accuracy'],
            'detection_pr': det_pr,
        }

    # 计算 seed 间的均值
    if not seeds_result:
        return {'seeds': {}, 'mean': {}}

    mean_result = compute_mean_across_seeds(seeds_result)
    return {'seeds': seeds_result, 'mean': mean_result}


def compute_mean_across_seeds(seeds_result: dict) -> dict:
    """计算各 seed 的均值"""
    seeds = list(seeds_result.keys())
    n = len(seeds)

    # 聚合指标均值
    aggregate_keys = seeds_result[seeds[0]]['aggregate'].keys()
    mean_aggregate = {k: np.mean([seeds_result[s]['aggregate'][k] for s in seeds]) for k in aggregate_keys}

    # 每类 AP 均值
    cat_names = seeds_result[seeds[0]]['cat_names']
    mean_per_class_ap = {}
    mean_per_class_ap50 = {}
    for cn in cat_names:
        mean_per_class_ap[cn] = float(np.mean([seeds_result[s]['per_class_ap'][cn] for s in seeds]))
        mean_per_class_ap50[cn] = float(np.mean([seeds_result[s]['per_class_ap50'][cn] for s in seeds]))

    # 混淆矩阵均值 (归一化后取均值, 行归一化 = 每类 recall)
    num_classes = len(cat_names)
    cm_sum = np.zeros((num_classes + 1, num_classes + 1))
    for s in seeds:
        cm_sum += np.array(seeds_result[s]['confusion_matrix'])
    cm_mean = cm_sum / n  # 平均计数
    # 行归一化 (不含背景行)
    cm_norm = cm_mean.copy()
    for i in range(num_classes):
        row_sum = cm_norm[i, :].sum()
        if row_sum > 0:
            cm_norm[i, :] = cm_norm[i, :] / row_sum

    # 检测 P/R 均值
    det_pr_keys = ['precision', 'recall', 'f1', 'tp', 'fp', 'fn']
    mean_det_pr = {k: float(np.mean([seeds_result[s]['detection_pr'][k] for s in seeds]))
                   for k in det_pr_keys}
    # 每类 P/R 均值
    mean_det_pr_per_class = {}
    for i in range(num_classes):
        mean_det_pr_per_class[i] = {
            mk: float(np.mean([seeds_result[s]['detection_pr']['per_class'][i][mk] for s in seeds]))
            for mk in ['precision', 'recall', 'f1']
        }

    return {
        'aggregate': mean_aggregate,
        'per_class_ap': mean_per_class_ap,
        'per_class_ap50': mean_per_class_ap50,
        'confusion_matrix_mean': cm_mean.tolist(),
        'confusion_matrix_norm': cm_norm.tolist(),
        'detection_pr': mean_det_pr,
        'detection_pr_per_class': mean_det_pr_per_class,
        'n_seeds': n,
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline vs SOTA 综合对比分析 (多数据集)')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--force-inference', action='store_true',
                        help='强制重新推理 (默认缓存优先)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='指定数据集 (默认全部), e.g. --datasets chromo 24obj')
    args = parser.parse_args()

    selected_datasets = args.datasets if args.datasets else list(DATASETS.keys())

    print('=' * 70)
    print('Baseline vs SOTA 综合对比分析 (多数据集)')
    print('=' * 70)
    print(f'Device: {args.device}')
    print(f'Datasets: {selected_datasets}')
    print(f'Cache policy: {"force-inference" if args.force_inference else "cache-first"}')

    # 评估所有数据集 × 所有模型
    all_results = {}
    for ds_name in selected_datasets:
        if ds_name not in DATASETS:
            print(f'[WARN] 未知数据集: {ds_name}, 跳过')
            continue

        ds_cfg = DATASETS[ds_name]
        ann_path = os.path.join(_PROJECT_ROOT, ds_cfg['ann_file'])
        print(f'\n{"#"*70}')
        print(f'# Dataset: {ds_name}')
        print(f'# Annotation: {ann_path}')
        print(f'# Models: {list(ds_cfg["models"].keys())}')
        print(f'{"#"*70}')

        if not os.path.exists(ann_path):
            print(f'[SKIP] 标注文件不存在: {ann_path}')
            continue

        coco_gt = COCO(ann_path)

        ds_results = {}
        for model_name, model_cfg in ds_cfg['models'].items():
            ds_results[model_name] = evaluate_model(
                model_name, model_cfg, coco_gt, args.device,
                args.force_inference, dataset_name=ds_name
            )

        all_results[ds_name] = ds_results

    # 保存结果
    output = {
        'description': 'Baseline vs SOTA 综合对比 (多数据集, 步数对齐 4-step)',
        'iou_thr': 0.5,
        'score_thr': 0.3,
        'datasets': {
            ds_name: {
                'ann_file': DATASETS[ds_name]['ann_file'],
                'models': ds_results,
            }
            for ds_name, ds_results in all_results.items()
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {OUTPUT_PATH}')

    # 打印汇总
    print('\n' + '=' * 70)
    print('汇总')
    print('=' * 70)
    for ds_name, ds_results in all_results.items():
        print(f'\n=== Dataset: {ds_name} ===')
        for model_name, result in ds_results.items():
            mean = result.get('mean', {})
            if not mean:
                print(f'  {model_name}: 无数据')
                continue
            n_seeds = mean.get('n_seeds', 0)
            agg = mean['aggregate']
            pr = mean['detection_pr']
            print(f'  {model_name} (n_seeds={n_seeds}):')
            print(f'    mAP={agg["mAP"]:.4f}  AP50={agg["AP50"]:.4f}  AP75={agg["AP75"]:.4f}')
            print(f'    AP_s={agg["AP_s"]:.4f}  AP_m={agg["AP_m"]:.4f}  AP_l={agg["AP_l"]:.4f}')
            print(f'    AR@100={agg["AR@100"]:.4f}  AR_s={agg["AR_s"]:.4f}  '
                  f'AR_m={agg["AR_m"]:.4f}  AR_l={agg["AR_l"]:.4f}')
            print(f'    Det P={pr["precision"]:.4f}  R={pr["recall"]:.4f}  F1={pr["f1"]:.4f}')

        # Delta: SOTA - DiffusionDet (展示完整提升: DiffusionDet → SOTA)
        sota_name = 'SOTA'
        baseline_name = 'DiffusionDet'
        if sota_name in ds_results and baseline_name in ds_results:
            b = ds_results[baseline_name]['mean']
            s = ds_results[sota_name]['mean']
            if b and s:
                print(f'\n  Delta ({sota_name} - {baseline_name}) on {ds_name}:')
                for k in ['mAP', 'AP50', 'AP75', 'AP_s', 'AP_m', 'AP_l',
                          'AR@100', 'AR_s', 'AR_m', 'AR_l']:
                    print(f'    Δ{k}: {s["aggregate"][k] - b["aggregate"][k]:+.4f}')
                for k in ['precision', 'recall', 'f1']:
                    print(f'    ΔDet_{k}: {s["detection_pr"][k] - b["detection_pr"][k]:+.4f}')


if __name__ == '__main__':
    main()
