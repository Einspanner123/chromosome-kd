#!/usr/bin/env python3
"""验证 Top-K + renewal OFF 的 mAP

确认去掉 box_renewal 在 Top-K 场景也不影响精度。

测试矩阵 (seed42, +DPM-Solver++ checkpoint, box_renewal OFF):
  1. K=500 (no pruning)  — 对照 (已有 renewal ON=0.864, OFF=0.863)
  2. K=200              — 新数据 (renewal ON=0.862)
  3. K=300              — 新数据 (renewal ON=0.862)

Usage:
    /home/linkst/data/miniconda3/envs/chromo/bin/python \
        experiments/analysis/verify_renewal_off_topk.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.analysis.per_dim_d1_clean_repro import run_config

# 配置路径
CONFIG_DPMPP = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
)
CONFIG_K200 = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k200_24obj.py',
)
CONFIG_K300 = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k300_24obj.py',
)
CKPT = os.path.join(
    _PROJECT_ROOT,
    'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
)
ANN = os.path.join(
    _PROJECT_ROOT,
    'data/24_chromosomes_object/coco/valid/_annotations.coco.json',
)

# 已知 renewal ON 数据 (seed42, 来自 LINEAGE/CATALOG)
RENEWAL_ON_REF = {
    'K=500': {'mAP': 0.864, 'lat': 93.7},
    'K=200': {'mAP': 0.862, 'lat': None},
    'K=300': {'mAP': 0.862, 'lat': None},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-imgs', type=int, default=None)
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    print('=' * 80)
    print('Top-K + renewal OFF 验证 (确认去掉 box_renewal 在所有场景不影响)')
    print('=' * 80)
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    print(f'Checkpoint: {CKPT}')
    print()

    configs = [
        ('K=500', CONFIG_DPMPP),
        ('K=200', CONFIG_K200),
        ('K=300', CONFIG_K300),
    ]

    results = {}
    for label, config_path in configs:
        print(f'\n--- {label} + renewal OFF ---')
        print(f'  Config: {config_path}')
        r = run_config(
            config_path, CKPT, ANN,
            dim_mask=None,  # [1,1,1,1] baseline
            device=device,
            max_imgs=args.max_imgs,
            box_renewal=False,
        )
        results[label] = r
        ref = RENEWAL_ON_REF.get(label, {})
        ref_mAP = ref.get('mAP', '?')
        delta = r['mAP'] - ref_mAP if isinstance(ref_mAP, float) else '?'
        print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  AP75={r["AP75"]:.4f}  '
              f'APs={r["APs"]:.4f}  lat={r["avg_latency_ms"]:.1f}ms')
        print(f'  vs renewal ON ({ref_mAP}): ΔmAP={delta}')

    # 汇总
    print('\n' + '=' * 80)
    print('汇总: renewal OFF vs ON')
    print('=' * 80)
    print(f'{"配置":<12} {"renewal ON":>12} {"renewal OFF":>12} {"ΔmAP":>10} {"lat(ms)":>10}')
    print('-' * 60)
    for label, _ in configs:
        r = results[label]
        ref_mAP = RENEWAL_ON_REF.get(label, {}).get('mAP', 0)
        delta = r['mAP'] - ref_mAP
        print(f'{label:<12} {ref_mAP:>12.4f} {r["mAP"]:>12.4f} {delta:>+10.4f} '
              f'{r["avg_latency_ms"]:>10.1f}')

    # 判定
    print('\n--- 判定 ---')
    all_ok = True
    for label, _ in configs:
        r = results[label]
        ref_mAP = RENEWAL_ON_REF.get(label, {}).get('mAP', 0)
        delta = abs(r['mAP'] - ref_mAP)
        if delta <= 0.002:
            print(f'  {label}: |ΔmAP|={delta:.4f} ≤ 0.002 (noise), ✓ 不影响')
        else:
            print(f'  {label}: |ΔmAP|={delta:.4f} > 0.002, ⚠ 有影响')
            all_ok = False

    if all_ok:
        print('\n→ 结论: 所有 Top-K 配置下去掉 box_renewal 均不影响 mAP')
    else:
        print('\n→ 结论: 部分配置有影响, 需进一步分析')

    # 保存
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        'renewal_off_topk_verify.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': CKPT,
        'box_renewal': False,
        'renewal_on_ref': RENEWAL_ON_REF,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'eta_str_per_dim'}
                     for k, v in results.items()},
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
