#!/usr/bin/env python3
"""D1 Top-K 3-seed 结果聚合 (2026-07-30)

聚合 d1_topk_validation_seed{42,123,789}.json, 计算 mean ± std,
生成 3-seed 汇总表用于更新 LINEAGE §四/§六.

Usage:
    python experiments/analysis/d1_topk_3seed_aggregate.py
    python experiments/analysis/d1_topk_3seed_aggregate.py --seeds 42 123 789
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


def load_seed_result(seed):
    """加载单个 seed 的 Top-K 验证结果."""
    path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'd1_topk_validation_seed{seed}.json',
    )
    if not os.path.exists(path):
        print(f'[警告] seed{seed} 结果不存在: {path}')
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='D1 Top-K 3-seed 结果聚合',
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 789],
                        help='要聚合的 seed 列表')
    args = parser.parse_args()

    print('=' * 90)
    print('Dataset 1 Top-K Pruning 3-seed 聚合')
    print('=' * 90)

    # 加载所有 seed 结果
    all_seed_data = {}
    for seed in args.seeds:
        data = load_seed_result(seed)
        if data is not None:
            all_seed_data[seed] = data['results']
            print(f'  seed{seed}: checkpoint={os.path.basename(data["checkpoint"])}')

    if len(all_seed_data) < 2:
        print(f'\n[错误] 仅 {len(all_seed_data)} 个 seed 结果可用, 至少需要 2 个')
        sys.exit(1)

    seeds = sorted(all_seed_data.keys())
    n_seeds = len(seeds)
    print(f'\n可用 seeds: {seeds} ({n_seeds} seeds)')

    # 场景列表
    scenarios = [
        ('K=500 renewal ON',  500, True),
        ('K=500 renewal OFF', 500, False),
        ('K=300 renewal ON',  300, True),
        ('K=300 renewal OFF', 300, False),
        ('K=200 renewal ON',  200, True),
        ('K=200 renewal OFF', 200, False),
        ('K=100 renewal ON',  100, True),
        ('K=100 renewal OFF', 100, False),
    ]

    # 聚合
    print(f'\n{"=" * 90}')
    print(f'3-seed 聚合结果 (D1 DPM-Solver++ 4-step)')
    print(f'{"=" * 90}')
    print(f'{"场景":<28} {"mAP mean":>10} {"±std":>8} ', end='')
    for s in seeds:
        print(f'{"seed"+str(s):>8}', end='')
    print(f'  {"ΔvsK500ON":>10}')
    print('-' * 90)

    aggregated = {}
    k500_on_mean = None

    for label, topk_k, renewal_on in scenarios:
        mAPs = []
        for s in seeds:
            r = all_seed_data[s].get(label, {})
            if 'mAP' in r:
                mAPs.append(r['mAP'])

        if len(mAPs) == 0:
            print(f'{label:<28} {"N/A":>10}')
            continue

        mean_mAP = float(np.mean(mAPs))
        std_mAP = float(np.std(mAPs, ddof=1)) if len(mAPs) > 1 else 0.0

        if label == 'K=500 renewal ON':
            k500_on_mean = mean_mAP

        aggregated[label] = {
            'mAP_mean': mean_mAP,
            'mAP_std': std_mAP,
            'mAP_per_seed': dict(zip(seeds, mAPs)),
            'n_seeds': len(mAPs),
        }

    # 第二遍打印 (需要 k500_on_mean)
    for label, topk_k, renewal_on in scenarios:
        if label not in aggregated:
            continue
        agg = aggregated[label]
        mean_mAP = agg['mAP_mean']
        std_mAP = agg['mAP_std']
        delta = mean_mAP - k500_on_mean if k500_on_mean else 0.0

        print(f'{label:<28} {mean_mAP:>10.4f} ±{std_mAP:>6.4f} ', end='')
        for s in seeds:
            v = agg['mAP_per_seed'].get(s)
            print(f'{v:>8.4f}' if v is not None else f'{"N/A":>8}', end='')
        print(f'  {delta:>+10.4f}')

    # D1 vs D2 对照
    print(f'\n--- D1 vs D2 对照 (3-seed mean) ---')
    D2_REF = {
        'K=500': 0.863, 'K=300': 0.861, 'K=200': 0.860, 'K=100': 0.850,
    }
    for k_label, d2_mAP in D2_REF.items():
        k_val = int(k_label.split('=')[1])
        on_label = f'K={k_val} renewal ON'
        if on_label in aggregated:
            d1_mean = aggregated[on_label]['mAP_mean']
            d1_std = aggregated[on_label]['mAP_std']
            print(f'  {k_label:>8}: D1={d1_mean:.4f}±{d1_std:.4f}  '
                  f'D2={d2_mAP:.4f}  Δ(D1-D2)={d1_mean-d2_mAP:+.4f}')

    # K 值依赖性表 (§六)
    print(f'\n--- K 值依赖性 (renewal ON vs OFF, 3-seed) ---')
    print(f'{"K":<8} {"ON mean":>10} {"OFF mean":>10} {"Δ(ON-OFF)":>10} {"判定":>8}')
    for k_val in [500, 300, 200, 100]:
        on_label = f'K={k_val} renewal ON'
        off_label = f'K={k_val} renewal OFF'
        if on_label in aggregated and off_label in aggregated:
            on_mean = aggregated[on_label]['mAP_mean']
            off_mean = aggregated[off_label]['mAP_mean']
            delta = on_mean - off_mean
            judge = '✓ 不影响' if abs(delta) < 0.005 else '⚠ 有影响'
            print(f'K={k_val:<5} {on_mean:>10.4f} {off_mean:>10.4f} '
                  f'{delta:>+10.4f} {judge:>8}')

    # 保存
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'd1_topk_validation_3seed_agg.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        'seeds': seeds,
        'n_seeds': n_seeds,
        'aggregated': aggregated,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n聚合结果已保存: {out_path}')


if __name__ == '__main__':
    main()
