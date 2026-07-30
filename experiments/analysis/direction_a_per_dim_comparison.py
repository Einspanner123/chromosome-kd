#!/usr/bin/env python3
"""方向 A Phase 2: per-dim solver 对比实验 (零成本推理)

在 +DPM-Solver++ checkpoint 上对比 2 个 solver 的 mAP 和延迟:
  1. dpm_solver_pp            (2 阶, +DPM-Solver++ baseline, 全维度 2 阶)
  2. dpm_solver_pp_per_dim    (per-dim: h=1阶 Euler, cx/cy/w=2阶 DPM-Solver++)

无需重训练, 仅推理时修改 solver_type.

诊断结论 (方向 A Phase 1):
  per-dim eta_str:
    h 维度 (index 3) eta_str = 4-11 (曲率最小)
    cx,cy 维度 (index 0,1) eta_str = 17-50 (曲率最大)
    w 维度 (index 2) eta_str 介于二者之间, 与 cx/cy 接近
  → h 维度轨迹最接近直线, 1 阶 (Euler) 应足够; cx/cy/w 需要 2 阶校正

预期:
  - per-dim mAP 与 baseline 持平或略降 (因 h 维度降为 1 阶, 增益可能有限)
  - per-dim 延迟略低 (1 阶 step 省去 D1 计算, 但幅度极小)

风险:
  - per-dim 阶数分配可能破坏 bbox 4 维度耦合性 (位置 cx,cy 与尺度 w,h 物理相关)

Usage:
  python experiments/analysis/direction_a_per_dim_comparison.py \
      --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
      --ann data/24_chromosomes_object/coco/valid/_annotations.coco.json \
      --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run_eval_with_solver(
    config_path, checkpoint, ann_file, solver_type, device='cuda:0',
):
    """在指定 solver_type 下评估模型 mAP + 延迟 + per-class AP"""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)

    # 覆盖 solver_type
    cfg.model.bbox_head.solver_type = solver_type

    # 禁用 SwanLab (推理评估不应污染训练项目)
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=cfg.vis_backends,
        name='visualizer',
    )

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    # 构建验证集
    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    # 构建 evaluator (classwise=True 输出 per-class AP)
    evaluator = METRICS.build(dict(
        type='CocoMetric',
        ann_file=ann_file,
        metric='bbox',
        classwise=True,
        format_only=False,
    ))
    evaluator.dataset_meta = dataset.metainfo

    # 推理 + 计时
    latencies = []
    num_samples = 0
    # 收集 per-dim eta_str 诊断 (仅第一个 batch, 用于统计)
    eta_str_per_dim_all = []

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            torch.cuda.synchronize() if 'cuda' in device else None
            t0 = time.time()
            out = model.test_step(data)
            if 'cuda' in device:
                torch.cuda.synchronize()
            latencies.append(time.time() - t0)

            # 收集 per-dim eta_str (仅第一个 batch)
            if solver_type == 'dpm_solver_pp_per_dim' and i == 0:
                bbox_head = model.bbox_head
                if hasattr(bbox_head, '_last_eta_str_per_dim_log'):
                    eta_str_per_dim_all = [
                        list(x) for x in bbox_head._last_eta_str_per_dim_log
                    ]

            # 将模型输出送入 evaluator
            out_list = out if isinstance(out, list) else [out]
            eval_samples = []
            for r in out_list:
                d = {}
                if hasattr(r, 'pred_instances') and r.pred_instances is not None:
                    pi = r.pred_instances
                    d['pred_instances'] = {
                        'bboxes': pi.bboxes.cpu(),
                        'scores': pi.scores.cpu(),
                        'labels': pi.labels.cpu(),
                    }
                d['img_id'] = getattr(r, 'img_id', i)
                d['ori_shape'] = getattr(r, 'ori_shape', (1, 1))
                eval_samples.append(d)
            evaluator.process({}, eval_samples)
            num_samples += len(eval_samples)

            if (i + 1) % 50 == 0:
                print(f'  推理进度: {i+1}/{len(dataset)}')

    metrics = evaluator.evaluate(num_samples)

    avg_latency = np.mean(latencies) * 1000  # ms
    p99_latency = np.percentile(latencies, 99) * 1000

    # 提取 per-class AP (classwise 输出)
    # CocoMetric classwise 输出 key 格式: coco/<classname>_<metric>
    # <metric> ∈ {precision, mAP_50, mAP_75, mAP_s, mAP_m, mAP_l}
    # <classname>_precision 即该类的 AP@[0.50:0.95] (mAP)
    KNOWN_METRICS = ['mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'precision']
    per_class_ap = {}
    per_class_detail = {}
    for key, val in metrics.items():
        if not key.startswith('coco/'):
            continue
        if key in (
            'coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75',
            'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l',
            'coco/bbox_mAP_copypaste',
        ):
            continue
        suffix = key.replace('coco/', '')
        for metric in KNOWN_METRICS:
            if suffix.endswith(metric):
                class_name = suffix[:-len(metric)].rstrip('_')
                if class_name:
                    val_f = float(val) if val == val else 0.0  # NaN → 0
                    if metric == 'precision':
                        per_class_ap[class_name] = val_f
                    if class_name not in per_class_detail:
                        per_class_detail[class_name] = {}
                    per_class_detail[class_name][metric] = val_f
                break

    return {
        'solver_type': solver_type,
        'mAP': float(metrics.get('coco/bbox_mAP', 0)),
        'AP50': float(metrics.get('coco/bbox_mAP_50', 0)),
        'AP75': float(metrics.get('coco/bbox_mAP_75', 0)),
        'APs': float(metrics.get('coco/bbox_mAP_s', 0)),
        'APm': float(metrics.get('coco/bbox_mAP_m', 0)),
        'APl': float(metrics.get('coco/bbox_mAP_l', 0)),
        'avg_latency_ms': float(avg_latency),
        'p99_latency_ms': float(p99_latency),
        'fps': float(1000.0 / avg_latency),
        'per_class_ap': per_class_ap,
        'per_class_detail': per_class_detail,
        'eta_str_per_dim': eta_str_per_dim_all,
    }


def main():
    parser = argparse.ArgumentParser(description='方向 A Phase 2: per-dim solver 对比')
    parser.add_argument(
        '--config', required=True,
        help='+DPM-Solver++ 配置文件',
    )
    parser.add_argument('--checkpoint', required=True, help='+DPM-Solver++ checkpoint')
    parser.add_argument('--ann', required=True, help='COCO 标注文件')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument(
        '--output', default=None,
        help='输出 JSON (默认 work_dirs/diagnosis/direction_a_per_dim_comparison.json)',
    )
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    output_path = args.output or os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        'direction_a_per_dim_comparison.json'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2 个 solver 配置
    solvers = [
        {'solver_type': 'dpm_solver_pp', 'label': 'DPM-Solver++ 2阶 (+DPM-Solver++ baseline, 全维度2阶)'},
        {'solver_type': 'dpm_solver_pp_per_dim', 'label': 'Per-dim (h=1阶, cxcy/w=2阶)'},
    ]

    print('=' * 80)
    print('方向 A Phase 2: per-dim solver 对比实验')
    print('=' * 80)
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Ann: {args.ann}')
    print(f'Device: {device}')
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print()

    all_results = []
    for cfg_solver in solvers:
        print(f'\n--- {cfg_solver["label"]} ---')
        result = run_eval_with_solver(
            args.config, args.checkpoint, args.ann,
            solver_type=cfg_solver['solver_type'],
            device=device,
        )
        result['label'] = cfg_solver['label']
        all_results.append(result)
        print(f'  mAP={result["mAP"]:.4f}  AP50={result["AP50"]:.4f}  '
              f'AP75={result["AP75"]:.4f}')
        print(f'  延迟: avg={result["avg_latency_ms"]:.1f}ms  '
              f'p99={result["p99_latency_ms"]:.1f}ms  '
              f'FPS={result["fps"]:.1f}')
        if result['eta_str_per_dim']:
            dim_names = ['cx', 'cy', 'w', 'h']
            print('  per-dim eta_str (per step):')
            for si, eta_dim in enumerate(result['eta_str_per_dim']):
                eta_str = ', '.join(
                    f'{dim_names[j]}={v:.3f}' for j, v in enumerate(eta_dim)
                )
                print(f'    step{si}: {eta_str}')

    # 汇总表
    print('\n' + '=' * 80)
    print('方向 A Phase 2 汇总')
    print('=' * 80)
    print(f'{"Solver":<45} {"mAP":>8} {"AP50":>8} {"AP75":>8} '
          f'{"Lat(ms)":>8} {"FPS":>8}')
    print('-' * 80)
    for r in all_results:
        print(f'{r["label"]:<45} {r["mAP"]:>8.4f} {r["AP50"]:>8.4f} '
              f'{r["AP75"]:>8.4f} {r["avg_latency_ms"]:>8.1f} '
              f'{r["fps"]:>8.1f}')

    # 分析
    baseline = all_results[0]  # dpm_solver_pp 2阶
    per_dim = all_results[1]   # per-dim

    print('\n分析:')
    print(f'  2阶 → per-dim: ΔmAP = {per_dim["mAP"] - baseline["mAP"]:+.4f}  '
          f'Δ延迟 = {per_dim["avg_latency_ms"] - baseline["avg_latency_ms"]:+.1f}ms')

    # per-class AP 对比 (小类别: Y, G22, F19, F20)
    print('\n小类别 per-class AP 对比 (Y, G22, F19, F20):')
    small_classes = ['Y', 'G22', 'F19', 'F20']
    print(f'  {"Class":<8} {"2阶":>10} {"per-dim":>10} {"Δ":>10}')
    for cls in small_classes:
        v0 = baseline['per_class_ap'].get(cls, 0.0)
        v1 = per_dim['per_class_ap'].get(cls, 0.0)
        print(f'  {cls:<8} {v0:>10.4f} {v1:>10.4f} {v1-v0:>+10.4f}')

    # 保存结果
    output = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'ann_file': args.ann,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results,
        'analysis': {
            'delta_map_baseline_to_per_dim': per_dim['mAP'] - baseline['mAP'],
            'delta_latency_baseline_to_per_dim': per_dim['avg_latency_ms'] - baseline['avg_latency_ms'],
        },
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
