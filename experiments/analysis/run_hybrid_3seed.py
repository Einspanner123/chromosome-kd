#!/usr/bin/env python3
"""Hybrid 求解器 3-seed 评估 (与其他 4 配置对齐)

checkpoint:
  seed_42:  work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth (原始训练)
  seed_123: /media/ross/8TB/.../multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth
  seed_789: /media/ross/8TB/.../multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth

Usage:
    python experiments/analysis/run_hybrid_3seed.py --gpu 0 --no-box-renewal
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.analysis.per_dim_d1_clean_repro import run_config


SEEDS = {
    42: os.path.join(
        _PROJECT_ROOT,
        'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
    ),
    123: '/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth',
    789: '/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth',
}

CONFIG = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
)
ANN = os.path.join(
    _PROJECT_ROOT,
    'data/24_chromosomes_object/coco/valid/_annotations.coco.json',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-box-renewal', action='store_true')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    box_renewal = not args.no_box_renewal

    print('=' * 80)
    print('Hybrid 求解器 3-seed 评估 (cx/cy DPM++, w/h Heun)')
    print('=' * 80)
    print(f'box_renewal: {"ON" if box_renewal else "OFF"}')
    print()

    results = {}
    for seed, ckpt in SEEDS.items():
        if not os.path.exists(ckpt):
            print(f'  [跳过] seed={seed} checkpoint 不存在: {ckpt}')
            continue
        print(f'\n--- seed={seed} ---')
        print(f'  checkpoint: {ckpt}')
        r = run_config(
            CONFIG, ckpt, ANN, 'hybrid',
            device=device, max_imgs=None, box_renewal=box_renewal,
        )
        results[seed] = r
        print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  '
              f'AP75={r["AP75"]:.4f}  APs={r["APs"]:.4f}  '
              f'lat={r["avg_latency_ms"]:.1f}ms')
        pd = r['per_dim_l1']
        print(f'  per-dim L1: cx={pd["cx"]:.5f} cy={pd["cy"]:.5f} '
              f'w={pd["w"]:.5f} h={pd["h"]:.5f}')

    # ===================== 汇总 =====================
    if len(results) < 2:
        print(f'\n只完成 {len(results)} 个 seed, 无法计算 std')
        return

    print('\n' + '=' * 80)
    print('3-seed 汇总')
    print('=' * 80)

    metrics = ['mAP', 'AP50', 'AP75', 'APs']
    dims = ['cx', 'cy', 'w', 'h']

    print(f'\n{"指标":<12} {"seed42":>10} {"seed123":>10} {"seed789":>10} '
          f'{"mean":>10} {"std":>10}')
    print('-' * 65)
    for m in metrics:
        vals = [results[s][m] for s in sorted(results)]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0
        row = f'{m:<12}'
        for v in vals:
            row += f' {v:>10.4f}'
        row += f' {mean:>10.4f} {std:>10.4f}'
        print(row)

    print(f'\n{"per-dim L1":<12} {"seed42":>10} {"seed123":>10} {"seed789":>10} '
          f'{"mean":>10} {"std":>10}')
    print('-' * 65)
    for d in dims:
        vals = [results[s]['per_dim_l1'][d] for s in sorted(results)]
        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if len(vals) > 1 else 0
        row = f'{d+"_L1":<12}'
        for v in vals:
            row += f' {v:>10.5f}'
        row += f' {mean:>10.5f} {std:>10.5f}'
        print(row)

    # 与 baseline 3-seed 对比 (从 JSON 读取, 消除硬编码循环引用)
    print('\n--- 与 baseline [1,1,1,1] 3-seed 对比 ---')
    baseline_json_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        'baseline_heun_3seed_norenewal.json',
    )
    with open(baseline_json_path) as f:
        baseline_data = json.load(f)

    def _agg(config_key, metric):
        """从 baseline_heun_3seed_norenewal.json 聚合 3-seed 统计量."""
        vals = [baseline_data['results'][config_key]['seeds'][s][metric]
                for s in ('42', '123', '789')]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    baseline_mAP_mean, baseline_mAP_std = _agg('baseline', 'mAP')
    baseline_APs_mean, baseline_APs_std = _agg('baseline', 'APs')
    wh_euler_mAP_mean, wh_euler_mAP_std = _agg('wh_euler', 'mAP')
    wh_euler_APs_mean, wh_euler_APs_std = _agg('wh_euler', 'APs')

    hybrid_mAP = np.mean([results[s]['mAP'] for s in sorted(results)])
    hybrid_mAP_std = np.std([results[s]['mAP'] for s in sorted(results)], ddof=1)
    hybrid_APs = np.mean([results[s]['APs'] for s in sorted(results)])
    hybrid_APs_std = np.std([results[s]['APs'] for s in sorted(results)], ddof=1)

    print(f'  baseline mAP: {baseline_mAP_mean:.4f} ± {baseline_mAP_std:.4f}')
    print(f'  hybrid   mAP: {hybrid_mAP:.4f} ± {hybrid_mAP_std:.4f}  '
          f'Δ={hybrid_mAP - baseline_mAP_mean:+.4f}')
    print(f'  baseline APs: {baseline_APs_mean:.4f} ± {baseline_APs_std:.4f}')
    print(f'  hybrid   APs: {hybrid_APs:.4f} ± {hybrid_APs_std:.4f}  '
          f'Δ={hybrid_APs - baseline_APs_mean:+.4f}')

    # 与 [1,1,0,0] 3-seed 对比 (Q3)
    print('\n--- Q3: hybrid (w/h Heun 2阶) vs [1,1,0,0] (w/h Euler 1阶) 3-seed ---')
    print(f'  [1,1,0,0] mAP: {wh_euler_mAP_mean:.4f} ± {wh_euler_mAP_std:.4f}')
    print(f'  hybrid   mAP: {hybrid_mAP:.4f} ± {hybrid_mAP_std:.4f}  '
          f'Δ={hybrid_mAP - wh_euler_mAP_mean:+.4f}')
    print(f'  [1,1,0,0] APs: {wh_euler_APs_mean:.4f} ± {wh_euler_APs_std:.4f}')
    print(f'  hybrid   APs: {hybrid_APs:.4f} ± {hybrid_APs_std:.4f}  '
          f'Δ={hybrid_APs - wh_euler_APs_mean:+.4f}')

    d_map = hybrid_mAP - wh_euler_mAP_mean
    d_map_base = hybrid_mAP - baseline_mAP_mean
    noise_threshold = max(baseline_mAP_std, wh_euler_mAP_std, hybrid_mAP_std) * 2
    if abs(d_map_base) <= noise_threshold:
        print(f'  → vs baseline ΔmAP={d_map_base:+.4f} 在 noise 范围内 '
              f'(threshold={noise_threshold:.4f}), null result')
    else:
        print(f'  → vs baseline ΔmAP={d_map_base:+.4f} 超 noise 阈值, 需进一步分析')
    if abs(d_map) <= noise_threshold:
        print(f'  → vs [1,1,0,0] ΔmAP={d_map:+.4f} 在 noise 范围内, '
              f'Heun 2阶 vs Euler 1阶 对 mAP 无显著差异')
    else:
        print(f'  → vs [1,1,0,0] ΔmAP={d_map:+.4f} 超 noise 阈值, 需进一步分析')

    # 保存
    suffix = '_norenewal' if not box_renewal else ''
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'hybrid_3seed{suffix}.json',
    )
    output = {
        'config': CONFIG,
        'ann_file': ANN,
        'box_renewal': box_renewal,
        'timestamp': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seeds': {str(s): r for s, r in results.items()},
        'summary': {
            'mAP_mean': float(hybrid_mAP),
            'mAP_std': float(hybrid_mAP_std),
            'APs_mean': float(hybrid_APs),
            'APs_std': float(hybrid_APs_std),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
