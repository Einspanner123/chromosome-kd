#!/usr/bin/env python3
"""只跑 hybrid 配置 (cx/cy DPM++, w/h Heun), 与已有 4 配置结果合并.

Usage:
    python experiments/analysis/run_hybrid_only.py \
        --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \
        --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \
        --ann data/24_chromosomes_object/coco/valid/_annotations.coco.json \
        --gpu 0 --no-box-renewal
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.analysis.per_dim_d1_clean_repro import run_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ann', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--no-box-renewal', action='store_true')
    parser.add_argument(
        '--existing',
        default=os.path.join(
            _PROJECT_ROOT, 'work_dirs', 'diagnosis',
            'per_dim_d1_clean_repro_norenewal.json',
        ),
        help='已有的 4 配置 JSON (用于合并)',
    )
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    box_renewal = not args.no_box_renewal

    print('=' * 80)
    print('Hybrid 求解器评估 (cx/cy DPM++, w/h Heun)')
    print('=' * 80)
    print(f'box_renewal: {"ON" if box_renewal else "OFF"}')
    print()

    # 只跑 hybrid
    print('--- 配置: hybrid (cx/cy DPM++, w/h Heun) ---')
    hybrid = run_config(
        args.config, args.checkpoint, args.ann, 'hybrid',
        device=device, max_imgs=None, box_renewal=box_renewal,
    )
    print(f'  mAP={hybrid["mAP"]:.4f}  AP50={hybrid["AP50"]:.4f}  '
          f'AP75={hybrid["AP75"]:.4f}  lat={hybrid["avg_latency_ms"]:.1f}ms')
    pd = hybrid['per_dim_l1']
    print(f'  per-dim L1: cx={pd["cx"]:.5f} cy={pd["cy"]:.5f} '
          f'w={pd["w"]:.5f} h={pd["h"]:.5f}  (n={pd["n_matched"]})')
    if hybrid.get('eta_str_per_dim'):
        print('  eta_str_per_dim (cx,cy,w,h) 各步:')
        for si, row in enumerate(hybrid['eta_str_per_dim']):
            print(f'    step{si}: [{row[0]:.2f}, {row[1]:.2f}, '
                  f'{row[2]:.2f}, {row[3]:.2f}]')

    # 加载已有 4 配置
    existing = None
    if os.path.exists(args.existing):
        with open(args.existing) as f:
            existing = json.load(f)
        print(f'\n已加载已有结果: {args.existing}')
    else:
        print(f'\n未找到已有结果: {args.existing}')

    # 合并保存
    suffix = '_norenewal' if not box_renewal else ''
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'hybrid_only{suffix}.json',
    )
    output = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'ann_file': args.ann,
        'box_renewal': box_renewal,
        'hybrid_result': hybrid,
        'existing_results': existing['results'] if existing else None,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')

    # ===================== 对比分析 =====================
    if existing and len(existing['results']) >= 4:
        base = existing['results'][0]    # [1,1,1,1]
        center = existing['results'][1]  # [1,1,0,0]
        euler = existing['results'][3]   # [0,0,0,0]

        print('\n' + '=' * 80)
        print('汇总对比 (box_renewal OFF, 500 图)')
        print('=' * 80)
        print(f'{"配置":<32} {"mAP":>8} {"AP75":>8} {"cx_L1":>9} {"cy_L1":>9} '
              f'{"w_L1":>9} {"h_L1":>9} {"lat(ms)":>8}')
        print('-' * 100)
        for r in existing['results']:
            pd = r['per_dim_l1']
            print(f'{r["label"]:<32} {r["mAP"]:>8.4f} {r["AP75"]:>8.4f} '
                  f'{pd["cx"]:>9.5f} {pd["cy"]:>9.5f} {pd["w"]:>9.5f} {pd["h"]:>9.5f} '
                  f'{r["avg_latency_ms"]:>8.1f}')
        pd = hybrid['per_dim_l1']
        print(f'{"hybrid (cxcyDPM++/whHeun)":<32} {hybrid["mAP"]:>8.4f} {hybrid["AP75"]:>8.4f} '
              f'{pd["cx"]:>9.5f} {pd["cy"]:>9.5f} {pd["w"]:>9.5f} {pd["h"]:>9.5f} '
              f'{hybrid["avg_latency_ms"]:>8.1f}')

        print('\n--- Q3: w/h 用 Heun 2阶 vs Euler 1阶 (cx/cy 均用 DPM++)? ---')
        print(f'  [1,1,0,0] w/h Euler 1阶: mAP={center["mAP"]:.4f}  '
              f'w_L1={center["per_dim_l1"]["w"]:.5f} h_L1={center["per_dim_l1"]["h"]:.5f}  '
              f'lat={center["avg_latency_ms"]:.1f}ms')
        print(f'  hybrid    w/h Heun  2阶: mAP={hybrid["mAP"]:.4f}  '
              f'w_L1={hybrid["per_dim_l1"]["w"]:.5f} h_L1={hybrid["per_dim_l1"]["h"]:.5f}  '
              f'lat={hybrid["avg_latency_ms"]:.1f}ms')
        d_map = hybrid["mAP"] - center["mAP"]
        d_w = hybrid["per_dim_l1"]["w"] - center["per_dim_l1"]["w"]
        d_h = hybrid["per_dim_l1"]["h"] - center["per_dim_l1"]["h"]
        d_lat = hybrid["avg_latency_ms"] - center["avg_latency_ms"]
        print(f'  ΔmAP={d_map:+.4f}  Δw_L1={d_w:+.6f}  Δh_L1={d_h:+.6f}  '
              f'Δlat={d_lat:+.1f}ms')
        if abs(d_map) <= 0.001 and abs(d_w) < 1e-4 and abs(d_h) < 1e-4:
            print(f'  → Q3 结论: Heun 2阶 vs Euler 1阶 对 w/h 无显著差异 (低曲率维度, '
                  f'梯形≈前向; 网络重预测 x0 补偿求解器差异). 额外 NFE 无收益.')
        else:
            print(f'  → Q3 结论: Heun 对 w/h 有可测量差异, 见上方数据.')

        print('\n--- 全配置 mAP 对比 ---')
        print(f'  baseline [1,1,1,1] 全 DPM++:   mAP={base["mAP"]:.4f}  '
              f'lat={base["avg_latency_ms"]:.1f}ms')
        print(f'  all-Euler [0,0,0,0]:           mAP={euler["mAP"]:.4f}  '
              f'lat={euler["avg_latency_ms"]:.1f}ms')
        print(f'  hybrid (cx/cy DPM++, w/h Heun): mAP={hybrid["mAP"]:.4f}  '
              f'lat={hybrid["avg_latency_ms"]:.1f}ms')
        print(f'  Δ(hybrid vs baseline)={hybrid["mAP"]-base["mAP"]:+.4f}  '
              f'Δ(hybrid vs all-Euler)={hybrid["mAP"]-euler["mAP"]:+.4f}')


if __name__ == '__main__':
    main()
