#!/usr/bin/env python3
"""D1 消融实验: RoI 空间信息贡献度验证 (零成本推理)

在 +DPM-Solver++ checkpoint 上对比:
  1. Baseline: RoIAlign 7×7 原始空间特征 (有空间信息)
  2. Ablation: RoIAlign 7×7 → 空间平均池化 → 广播回 7×7 (抹平空间信息)

通过 forward hook 在 roi_extractor 输出后, 将 7×7 特征替换为全局平均 (所有 49 个
位置值相同), 完全抹平空间结构但保持维度不变, 确保 DynamicConv 预训练权重可完整加载。

目的: 验证 M1 (形态感知 RoI 编码器) 的必要性
  - 若 mAP 不降: 7×7 空间结构对当前架构无贡献, M1 有大改进空间
  - 若 mAP 显著下降: 空间信息有贡献, M1 应增强而非重建空间编码

Usage:
  python experiments/analysis/d1_roi_ablation.py \
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


def make_spatial_avg_hook():
    """创建抹平 RoI 空间信息的 forward hook.

    将 roi_extractor 输出 [N, C, H, W] 的空间维度做全局平均,
    然后广播回原始尺寸 — 所有 H×W 位置值相同, 空间信息被完全抹平,
    但张量维度不变, DynamicConv 预训练权重可完整加载。
    """
    def hook(module, input, output):
        # output: [N, C, H, W] (e.g., [N, 256, 7, 7])
        spatial_avg = output.mean(dim=[2, 3], keepdim=True)
        return spatial_avg.expand_as(output)
    return hook


def run_eval(config_path, checkpoint, ann_file, device='cuda:0',
             ablation=False):
    """评估模型 mAP + 延迟 + per-class AP.

    Args:
        ablation: True=抹平空间信息, False=baseline
    """
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)

    # 禁用 SwanLab
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=cfg.vis_backends,
        name='visualizer',
    )

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    # 注册 hook (消融模式)
    hook_handle = None
    if ablation:
        roi_extractor = model.bbox_head.roi_extractor
        hook_handle = roi_extractor.register_forward_hook(
            make_spatial_avg_hook()
        )
        print(f"  [消融] 已在 {type(roi_extractor).__name__} 上注册空间抹平 hook")
    else:
        print("  [Baseline] 原始 7×7 RoI 空间特征")

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

    with torch.inference_mode():
        for i in range(len(dataset)):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.test_step(data)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

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

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(dataset)} done")

    metrics = evaluator.evaluate(num_samples)

    # 移除 hook
    if hook_handle is not None:
        hook_handle.remove()

    # 提取结果
    mAP = metrics.get('coco/bbox_mAP', 0.0)
    mAP_50 = metrics.get('coco/bbox_mAP_50', 0.0)
    mAP_75 = metrics.get('coco/bbox_mAP_75', 0.0)
    mAP_s = metrics.get('coco/bbox_mAP_s', 0.0)
    mAP_m = metrics.get('coco/bbox_mAP_m', 0.0)
    mAP_l = metrics.get('coco/bbox_mAP_l', 0.0)

    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency

    # per-class AP (classwise: coco/<classname>_precision = AP@[0.50:0.95])
    KNOWN_METRICS = ['mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l', 'precision']
    per_class = {}
    for key, val in metrics.items():
        if not key.startswith('coco/'):
            continue
        if key in ('coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75',
                   'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l',
                   'coco/bbox_mAP_copypaste'):
            continue
        suffix = key.replace('coco/', '')
        for metric in KNOWN_METRICS:
            if suffix.endswith(metric):
                classname = suffix[:-len(metric)].rstrip('_')
                if metric == 'precision':
                    per_class[classname] = float(val)
                break

    return {
        'mAP': round(mAP, 4),
        'mAP_50': round(mAP_50, 4),
        'mAP_75': round(mAP_75, 4),
        'mAP_s': round(mAP_s, 4),
        'mAP_m': round(mAP_m, 4),
        'mAP_l': round(mAP_l, 4),
        'latency_ms': round(avg_latency, 1),
        'fps': round(fps, 1),
        'per_class_AP': per_class,
    }


def main():
    parser = argparse.ArgumentParser(description='D1 RoI 空间信息消融实验')
    parser.add_argument(
        '--config',
        default='experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
    )
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
    )
    parser.add_argument(
        '--ann',
        default='data/24_chromosomes_object/coco/valid/_annotations.coco.json',
    )
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'

    print("=" * 70)
    print("D1 消融实验: RoI 空间信息贡献度验证")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Annotation: {args.ann}")
    print()

    # Baseline
    print("[1/2] Baseline (7×7 RoI 原始空间特征)")
    baseline = run_eval(args.config, args.checkpoint, args.ann, device,
                        ablation=False)
    print(f"  mAP={baseline['mAP']}, AP50={baseline['mAP_50']}, "
          f"AP75={baseline['mAP_75']}, latency={baseline['latency_ms']}ms, "
          f"FPS={baseline['fps']}")
    print()

    # Ablation
    print("[2/2] Ablation (7×7 → 空间平均 → 广播回 7×7, 抹平空间信息)")
    ablation = run_eval(args.config, args.checkpoint, args.ann, device,
                        ablation=True)
    print(f"  mAP={ablation['mAP']}, AP50={ablation['mAP_50']}, "
          f"AP75={ablation['mAP_75']}, latency={ablation['latency_ms']}ms, "
          f"FPS={ablation['fps']}")
    print()

    # 对比
    print("=" * 70)
    print("对比结果")
    print("=" * 70)
    delta_map = ablation['mAP'] - baseline['mAP']
    delta_ap50 = ablation['mAP_50'] - baseline['mAP_50']
    delta_ap75 = ablation['mAP_75'] - baseline['mAP_75']
    delta_latency = ablation['latency_ms'] - baseline['latency_ms']

    print(f"{'指标':<15} {'Baseline':>12} {'Ablation':>12} {'Δ':>12}")
    print("-" * 55)
    print(f"{'mAP':<15} {baseline['mAP']:>12.4f} {ablation['mAP']:>12.4f} "
          f"{delta_map:>+12.4f}")
    print(f"{'AP50':<15} {baseline['mAP_50']:>12.4f} {ablation['mAP_50']:>12.4f} "
          f"{delta_ap50:>+12.4f}")
    print(f"{'AP75':<15} {baseline['mAP_75']:>12.4f} {ablation['mAP_75']:>12.4f} "
          f"{delta_ap75:>+12.4f}")
    print(f"{'APs':<15} {baseline['mAP_s']:>12.4f} {ablation['mAP_s']:>12.4f} "
          f"{ablation['mAP_s'] - baseline['mAP_s']:>+12.4f}")
    print(f"{'APm':<15} {baseline['mAP_m']:>12.4f} {ablation['mAP_m']:>12.4f} "
          f"{ablation['mAP_m'] - baseline['mAP_m']:>+12.4f}")
    print(f"{'APl':<15} {baseline['mAP_l']:>12.4f} {ablation['mAP_l']:>12.4f} "
          f"{ablation['mAP_l'] - baseline['mAP_l']:>+12.4f}")
    print(f"{'Latency(ms)':<15} {baseline['latency_ms']:>12.1f} "
          f"{ablation['latency_ms']:>12.1f} {delta_latency:>+12.1f}")
    print()

    # per-class AP 对比
    print("Per-class AP 对比:")
    print(f"{'Class':<10} {'Baseline':>10} {'Ablation':>10} {'Δ':>10}")
    print("-" * 42)
    all_classes = sorted(set(baseline['per_class_AP'].keys()) |
                         set(ablation['per_class_AP'].keys()))
    for cls in all_classes:
        b = baseline['per_class_AP'].get(cls, 0.0)
        a = ablation['per_class_AP'].get(cls, 0.0)
        print(f"{cls:<10} {b:>10.4f} {a:>10.4f} {a - b:>+10.4f}")
    print()

    # 结论
    print("=" * 70)
    print("结论")
    print("=" * 70)
    if abs(delta_map) < 0.002:
        print(f"ΔmAP = {delta_map:+.4f} (|Δ| < 0.002, 持平)")
        print("→ 7×7 空间结构对当前架构无显著贡献")
        print("→ M1 (形态感知 RoI 编码器) 有大改进空间: 当前空间信息未被有效利用")
    elif delta_map < -0.005:
        print(f"ΔmAP = {delta_map:+.4f} (显著下降)")
        print("→ 7×7 空间结构对当前架构有显著贡献")
        print("→ M1 应增强空间编码而非重建: 现有 DynamicConv 已部分利用空间信息")
    else:
        print(f"ΔmAP = {delta_map:+.4f} (边际变化)")
        print("→ 空间结构有一定贡献但不显著, M1 改进空间中等")

    # 保存结果
    output_dir = Path('work_dirs/diagnosis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'd1_roi_ablation.json'
    result = {
        'timestamp': datetime.now().isoformat(),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'baseline': baseline,
        'ablation': ablation,
        'delta_mAP': delta_map,
    }
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
