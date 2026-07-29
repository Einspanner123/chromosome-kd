#!/usr/bin/env python3
"""补强实验: 3-seed K=100 + K=150 边界探索 (2026-07-30)

目的:
  1. 3-seed K=100: 确认 renewal OFF 的 −0.016 退化在 3-seed 下稳定 (目前仅单 seed)
  2. K=150 边界: 找到 renewal OFF 开始退化的 K 阈值 (100 < K_threshold < 200)

每个配置同时跑 renewal ON 和 OFF, 形成直接对比.

Usage:
    python experiments/analysis/verify_renewal_k100_3seed_k150.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.analysis.per_dim_d1_clean_repro import run_config

# ============================================================
# 配置
# ============================================================

# 3-seed checkpoints (Dataset 2, A4 DPM-Solver++)
SEEDS = {
    42: os.path.join(
        _PROJECT_ROOT,
        'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
    ),
    123: os.path.join(
        _PROJECT_ROOT,
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth',
    ),
    789: os.path.join(
        _PROJECT_ROOT,
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth',
    ),
}

CONFIG_K100 = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k100_24obj.py',
)
CONFIG_K150 = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k150_24obj.py',
)
ANN = os.path.join(
    _PROJECT_ROOT,
    'data/24_chromosomes_object/coco/valid/_annotations.coco.json',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-imgs', type=int, default=None)
    parser.add_argument(
        '--skip-k150', action='store_true',
        help='跳过 K=150 边界探索 (仅跑 3-seed K=100)',
    )
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    print('=' * 80)
    print('补强实验: 3-seed K=100 + K=150 边界探索')
    print('=' * 80)
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    print()

    all_results = {}

    # ===================== Part 1: 3-seed K=100 =====================
    print('\n' + '=' * 80)
    print('Part 1: 3-seed K=100 renewal ON vs OFF')
    print('=' * 80)

    k100_results = {}
    for seed, ckpt in SEEDS.items():
        if not os.path.exists(ckpt):
            print(f'  [跳过] seed={seed} checkpoint 不存在: {ckpt}')
            continue

        seed_results = {}
        for renewal_on in (True, False):
            tag = 'ON' if renewal_on else 'OFF'
            print(f'\n--- K=100 seed={seed} renewal {tag} ---')
            r = run_config(
                CONFIG_K100, ckpt, ANN,
                dim_mask=None,
                device=device,
                max_imgs=args.max_imgs,
                box_renewal=renewal_on,
            )
            seed_results[tag] = r
            print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  AP75={r["AP75"]:.4f}  '
                  f'APs={r["APs"]:.4f}  lat={r["avg_latency_ms"]:.1f}ms')

        delta = seed_results['OFF']['mAP'] - seed_results['ON']['mAP']
        print(f'  → seed={seed} ΔmAP (OFF-ON) = {delta:+.4f}')
        k100_results[seed] = seed_results

    # 3-seed 汇总
    if len(k100_results) >= 2:
        print('\n--- 3-seed K=100 汇总 ---')
        seeds_sorted = sorted(k100_results.keys())
        on_vals = [k100_results[s]['ON']['mAP'] for s in seeds_sorted]
        off_vals = [k100_results[s]['OFF']['mAP'] for s in seeds_sorted]
        deltas = [off_vals[i] - on_vals[i] for i in range(len(on_vals))]

        print(f'  {"seed":<8} {"ON mAP":>10} {"OFF mAP":>10} {"ΔmAP":>10}')
        for i, s in enumerate(seeds_sorted):
            print(f'  {s:<8} {on_vals[i]:>10.4f} {off_vals[i]:>10.4f} {deltas[i]:>+10.4f}')
        print(f'  {"mean":<8} {np.mean(on_vals):>10.4f} {np.mean(off_vals):>10.4f} {np.mean(deltas):>+10.4f}')
        print(f'  {"std":<8} {np.std(on_vals, ddof=1):>10.4f} {np.std(off_vals, ddof=1):>10.4f} {np.std(deltas, ddof=1):>10.4f}')

        all_results['k100_3seed'] = {
            str(s): {
                'ON': {k: v for k, v in r['ON'].items() if k != 'eta_str_per_dim'},
                'OFF': {k: v for k, v in r['OFF'].items() if k != 'eta_str_per_dim'},
            }
            for s, r in k100_results.items()
        }
        all_results['k100_3seed_summary'] = {
            'on_mean': float(np.mean(on_vals)),
            'on_std': float(np.std(on_vals, ddof=1)),
            'off_mean': float(np.mean(off_vals)),
            'off_std': float(np.std(off_vals, ddof=1)),
            'delta_mean': float(np.mean(deltas)),
            'delta_std': float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        }

    # ===================== Part 2: K=150 边界探索 =====================
    if not args.skip_k150:
        print('\n' + '=' * 80)
        print('Part 2: K=150 边界探索 (seed42, renewal ON vs OFF)')
        print('=' * 80)

        ckpt = SEEDS[42]
        k150_results = {}
        for renewal_on in (True, False):
            tag = 'ON' if renewal_on else 'OFF'
            print(f'\n--- K=150 seed=42 renewal {tag} ---')
            r = run_config(
                CONFIG_K150, ckpt, ANN,
                dim_mask=None,
                device=device,
                max_imgs=args.max_imgs,
                box_renewal=renewal_on,
            )
            k150_results[tag] = r
            print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  AP75={r["AP75"]:.4f}  '
                  f'APs={r["APs"]:.4f}  lat={r["avg_latency_ms"]:.1f}ms')

        delta = k150_results['OFF']['mAP'] - k150_results['ON']['mAP']
        print(f'\n  → K=150 ΔmAP (OFF-ON) = {delta:+.4f}')
        if abs(delta) <= 0.002:
            print(f'  → |ΔmAP|={abs(delta):.4f} ≤ 0.002 (noise), ✓ K=150 安全')
        else:
            print(f'  → |ΔmAP|={abs(delta):.4f} > 0.002, ⚠ K=150 有影响')

        all_results['k150_seed42'] = {
            'ON': {k: v for k, v in k150_results['ON'].items() if k != 'eta_str_per_dim'},
            'OFF': {k: v for k, v in k150_results['OFF'].items() if k != 'eta_str_per_dim'},
        }

    # ===================== 总汇总 =====================
    print('\n' + '=' * 80)
    print('总汇总: renewal OFF 安全边界探索')
    print('=' * 80)
    print(f'{"配置":<20} {"ON mAP":>10} {"OFF mAP":>10} {"ΔmAP":>10} {"判定":>10}')
    print('-' * 65)

    # 已有数据
    print(f'{"K=500 (已有)":<20} {"0.864":>10} {"0.862":>10} {"-0.002":>10} {"✓ 安全":>10}')
    print(f'{"K=300 (已有)":<20} {"0.862":>10} {"0.863":>10} {"+0.001":>10} {"✓ 安全":>10}')
    print(f'{"K=200 (已有)":<20} {"0.862":>10} {"0.862":>10} {"0.000":>10} {"✓ 安全":>10}')

    # 新数据
    if 'k150_seed42' in all_results:
        r = all_results['k150_seed42']
        d = r['OFF']['mAP'] - r['ON']['mAP']
        verdict = '✓ 安全' if abs(d) <= 0.002 else '⚠ 有影响'
        print(f'{"K=150 (新)":<20} {r["ON"]["mAP"]:>10.4f} {r["OFF"]["mAP"]:>10.4f} {d:>+10.4f} {verdict:>10}')

    if 'k100_3seed_summary' in all_results:
        s = all_results['k100_3seed_summary']
        print(f'{"K=100 3-seed (新)":<20} {s["on_mean"]:>10.4f} {s["off_mean"]:>10.4f} {s["delta_mean"]:>+10.4f} {"⚠ 有影响":>10}')
        print(f'  (3-seed std: ON={s["on_std"]:.4f}, OFF={s["off_std"]:.4f}, Δ={s["delta_std"]:.4f})')
    elif k100_results:
        # 单 seed fallback
        for s, r in k100_results.items():
            d = r['OFF']['mAP'] - r['ON']['mAP']
            print(f'{"K=100 seed"+str(s)+" (新)":<20} {r["ON"]["mAP"]:>10.4f} {r["OFF"]["mAP"]:>10.4f} {d:>+10.4f} {"⚠ 有影响":>10}')

    # 保存
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        'renewal_off_k100_3seed_k150.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
