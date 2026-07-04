#!/usr/bin/env python3
"""采样步数扫描实验 — 定位采样瓶颈

在已训练好的模型上, 测试不同采样步数 (1, 2, 4, 8, 16, 32) 下的 mAP 和延迟.
无需重新训练, 仅修改推理时的 sampling_timesteps.

Usage:
    python experiments/analysis/sampling_step_sweep.py \
        experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py \
        work_dirs/ldmdet_rf_heun_adaln/best_coco_bbox_mAP_epoch_102.pth \
        --ann data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run_eval_with_steps(config_path, checkpoint, ann_file, steps, device='cuda:0'):
    """在指定采样步数下评估模型"""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)

    # 修改采样步数
    cfg.model.bbox_head.sampling_timesteps = steps

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    # 构建验证集
    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    # 构建 evaluator
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

            # 将模型输出送入 evaluator (转换为 dict 格式, CocoMetric 期望可下标对象)
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

    return {
        'steps': steps,
        'mAP': float(metrics.get('coco/bbox_mAP', 0)),
        'AP50': float(metrics.get('coco/bbox_mAP_50', 0)),
        'AP75': float(metrics.get('coco/bbox_mAP_75', 0)),
        'APs': float(metrics.get('coco/bbox_mAP_s', 0)),
        'APm': float(metrics.get('coco/bbox_mAP_m', 0)),
        'APl': float(metrics.get('coco/bbox_mAP_l', 0)),
        'avg_latency_ms': float(avg_latency),
        'p99_latency_ms': float(p99_latency),
        'fps': float(1000.0 / avg_latency),
    }


def main():
    parser = argparse.ArgumentParser(description='采样步数扫描')
    parser.add_argument('config', help='配置文件')
    parser.add_argument('checkpoint', help='检查点')
    parser.add_argument('--ann', required=True, help='COCO 标注文件')
    parser.add_argument('--steps', default='1,2,4,8,16', help='采样步数列表')
    parser.add_argument('--output', default='sampling_sweep_results.json', help='输出文件')
    parser.add_argument('--device', default='cuda:0', help='设备')
    args = parser.parse_args()

    steps_list = [int(s) for s in args.steps.split(',')]

    print('=' * 80)
    print('采样步数扫描实验')
    print('=' * 80)
    print(f'Config: {args.config}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Steps: {steps_list}')
    print()

    all_results = []
    for steps in steps_list:
        print(f'\n--- 采样步数 = {steps} ---')
        result = run_eval_with_steps(args.config, args.checkpoint, args.ann, steps, args.device)
        all_results.append(result)
        print(f'  mAP={result["mAP"]:.4f}  AP50={result["AP50"]:.4f}  AP75={result["AP75"]:.4f}')
        print(f'  延迟: avg={result["avg_latency_ms"]:.1f}ms  p99={result["p99_latency_ms"]:.1f}ms  FPS={result["fps"]:.1f}')

    # 汇总表
    print('\n' + '=' * 80)
    print('采样步数扫描汇总')
    print('=' * 80)
    print(f'{"Steps":>6} {"mAP":>8} {"AP50":>8} {"AP75":>8} {"Latency(ms)":>12} {"FPS":>8}')
    print('-' * 60)
    for r in all_results:
        print(f'{r["steps"]:>6} {r["mAP"]:>8.4f} {r["AP50"]:>8.4f} {r["AP75"]:>8.4f} '
              f'{r["avg_latency_ms"]:>12.1f} {r["fps"]:>8.1f}')

    # 分析
    best_map = max(all_results, key=lambda x: x['mAP'])
    fastest = min(all_results, key=lambda x: x['avg_latency_ms'])
    map_range = best_map['mAP'] - min(r['mAP'] for r in all_results)

    print(f'\n分析:')
    print(f'  最佳 mAP: {best_map["mAP"]:.4f} (steps={best_map["steps"]})')
    print(f'  最快推理: {fastest["fps"]:.1f} FPS (steps={fastest["steps"]})')
    print(f'  mAP 变化范围: {map_range:.4f}')
    if map_range < 0.01:
        print(f'  -> 采样步数对性能影响很小, 采样不是瓶颈, 可用最少步数加速推理.')
    elif map_range < 0.03:
        print(f'  -> 采样步数有一定影响, 存在采样瓶颈但不大.')
    else:
        print(f'  -> 采样步数显著影响性能, 采样是重要瓶颈, 需增加步数或改进采样器.')

    # 保存结果
    output = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'results': all_results,
        'best_map': best_map,
        'fastest': fastest,
        'map_range': map_range,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n结果已保存到: {args.output}')


if __name__ == '__main__':
    main()
