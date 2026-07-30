#!/usr/bin/env python3
"""box_renewal 去除全场景验证 (2026-07-29)

填补两个数据缺口, 确认 "推理时关闭 box_renewal 不影响精度" 在所有场景成立:

  Gap 1: Dataset 1 (Chromosome20240904) + DPM-Solver++
    - bottleneck no_box_renewal (训练消融, Heun) 显示 -0.016 mAP
    - 需验证: 推理切换 (非训练消融) 在 DPM-Solver++ 上是否也不影响
    - renewal ON 基准: seed42 mAP=0.746 (训练日志 epoch 49)

  Gap 2: Dataset 2 + Top-K K=100 (最脆弱配置)
    - K=100 proposal 数量最少, box_renewal 的"低置信度 proposal 重置"理论上影响最大
    - renewal ON 基准: seed42 mAP=0.852 (3-seed mean 0.850±0.003)

每个场景同时跑 renewal ON 和 OFF, 形成直接对比 (不依赖历史数据).

Usage:
    python experiments/analysis/verify_renewal_off_all_scenarios.py --gpu 0
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
# 场景配置
# ============================================================

# Gap 1: Dataset 1 + DPM-Solver++ (seed42)
DATASET1_CONFIG = os.path.join(
    _PROJECT_ROOT, 'experiments/configs/ldmdet/a4_dpm_pp_chr2024.py',
)
DATASET1_CKPT = os.path.join(
    _PROJECT_ROOT,
    'work_dirs/a4_dpm_pp_chr2024_seed42/best_coco_bbox_mAP_epoch_49.pth',
)
DATASET1_ANN = os.path.join(
    _PROJECT_ROOT,
    'data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json',
)

# Gap 2: Dataset 2 + Top-K K=100 (seed42)
K100_CONFIG = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/inference_opt/a4_io3_eval_k100_24obj.py',
)
K100_CKPT = os.path.join(
    _PROJECT_ROOT,
    'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
)
K100_ANN = os.path.join(
    _PROJECT_ROOT,
    'data/24_chromosomes_object/coco/valid/_annotations.coco.json',
)

# 历史参考 (renewal ON)
REF = {
    'Dataset1 (+DPM-Solver++)': {'mAP': 0.746, 'source': '训练日志 epoch 49 seed42'},
    'Dataset2_K100': {'mAP': 0.852, 'source': 'LINEAGE seed42 (3-seed mean 0.850±0.003)'},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max-imgs', type=int, default=None)
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    print('=' * 80)
    print('box_renewal 去除全场景验证 (Dataset 1 + K=100 缺口填补)')
    print('=' * 80)
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    print()

    scenarios = [
        ('Dataset1 (+DPM-Solver++)', DATASET1_CONFIG, DATASET1_CKPT, DATASET1_ANN),
        ('Dataset2_K100', K100_CONFIG, K100_CKPT, K100_ANN),
    ]

    all_results = {}

    for label, config, ckpt, ann in scenarios:
        print(f'\n{"=" * 80}')
        print(f'场景: {label}')
        print(f'  Config: {config}')
        print(f'  Checkpoint: {ckpt}')
        print(f'  Ann: {ann}')
        print(f'  renewal ON 参考: mAP={REF[label]["mAP"]} ({REF[label]["source"]})')
        print(f'{"=" * 80}')

        if not os.path.exists(ckpt):
            print(f'  [跳过] checkpoint 不存在: {ckpt}')
            continue

        scene_results = {}
        for renewal_on in (True, False):
            tag = 'ON' if renewal_on else 'OFF'
            print(f'\n--- {label} renewal {tag} ---')
            r = run_config(
                config, ckpt, ann,
                dim_mask=None,  # [1,1,1,1] baseline
                device=device,
                max_imgs=args.max_imgs,
                box_renewal=renewal_on,
            )
            scene_results[tag] = r
            ref_mAP = REF[label]['mAP']
            print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  AP75={r["AP75"]:.4f}  '
                  f'APs={r["APs"]:.4f}  lat={r["avg_latency_ms"]:.1f}ms')
            print(f'  vs ref ({ref_mAP}): Δ={r["mAP"] - ref_mAP:+.4f}')

        # ON vs OFF 直接对比
        on_mAP = scene_results['ON']['mAP']
        off_mAP = scene_results['OFF']['mAP']
        delta = off_mAP - on_mAP
        print(f'\n--- {label} ON vs OFF 直接对比 ---')
        print(f'  renewal ON:  mAP={on_mAP:.4f}  lat={scene_results["ON"]["avg_latency_ms"]:.1f}ms')
        print(f'  renewal OFF: mAP={off_mAP:.4f}  lat={scene_results["OFF"]["avg_latency_ms"]:.1f}ms')
        print(f'  ΔmAP (OFF-ON): {delta:+.4f}')
        print(f'  Δlat (ON-OFF): {scene_results["ON"]["avg_latency_ms"] - scene_results["OFF"]["avg_latency_ms"]:+.1f}ms')

        if abs(delta) <= 0.002:
            print(f'  → |ΔmAP|={abs(delta):.4f} ≤ 0.002 (noise), ✓ 不影响')
        else:
            print(f'  → |ΔmAP|={abs(delta):.4f} > 0.002, ⚠ 有影响')

        all_results[label] = scene_results

    # ===================== 总汇总 =====================
    print('\n' + '=' * 80)
    print('总汇总: 全场景 renewal ON vs OFF')
    print('=' * 80)
    print(f'{"场景":<18} {"ON mAP":>10} {"OFF mAP":>10} {"ΔmAP":>10} '
          f'{"ON lat":>10} {"OFF lat":>10} {"判定":>10}')
    print('-' * 80)

    all_ok = True
    for label in all_results:
        on_r = all_results[label]['ON']
        off_r = all_results[label]['OFF']
        delta = off_r['mAP'] - on_r['mAP']
        ok = abs(delta) <= 0.002
        verdict = '✓ 不影响' if ok else '⚠ 有影响'
        if not ok:
            all_ok = False
        print(f'{label:<18} {on_r["mAP"]:>10.4f} {off_r["mAP"]:>10.4f} '
              f'{delta:>+10.4f} {on_r["avg_latency_ms"]:>9.1f}ms '
              f'{off_r["avg_latency_ms"]:>9.1f}ms {verdict:>10}')

    # 合并已有 Dataset 2 K=500/200/300 数据
    print(f'\n{"(已有) D2 K=500":<18} {"0.864":>10} {"0.862":>10} {"-0.002":>10} '
          f'{"93.7ms":>10} {"93.0ms":>10} {"✓ 不影响":>10}')
    print(f'{"(已有) D2 K=200":<18} {"0.862":>10} {"0.862":>10} {"0.000":>10} '
          f'{"-":>10} {"81.8ms":>10} {"✓ 不影响":>10}')
    print(f'{"(已有) D2 K=300":<18} {"0.862":>10} {"0.863":>10} {"+0.001":>10} '
          f'{"-":>10} {"83.0ms":>10} {"✓ 不影响":>10}')

    if all_ok:
        print('\n→ 结论: 所有场景 (Dataset 1/2, K=100/200/300/500) 推理时关闭 box_renewal 均不影响 mAP')
    else:
        print('\n→ 结论: 部分场景有影响, 需进一步分析')

    # 保存
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        'renewal_off_all_scenarios.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ref': REF,
        'results': {
            label: {
                tag: {k: v for k, v in r.items() if k != 'eta_str_per_dim'}
                for tag, r in scene.items()
            }
            for label, scene in all_results.items()
        },
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
