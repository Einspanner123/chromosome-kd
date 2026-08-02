#!/usr/bin/env python3
"""3-seed test mAP 补跑 (2026-08-03)

补全所有 3-seed 实验缺失的 test mAP 数据。
复用 test_eval_per_size.py 的 run_test_eval 函数 (已验证可用)。

覆盖范围:
  D1 (Dataset 1, chr2024, 220 test imgs):
    - KaryoFlow +DPM-Solver++ seed123/789 (seed42 已有 test=0.739)
    - KaryoFlow +Stoch. Coupling 3-seed (无aug, seed42 已有 test=0.740 来自 reproduce_0751)
    - KaryoFlow v-prediction 3-seed
    - Hard OT (无aug) 3-seed (OT collapse 论证)
    - Random (无aug, AdaLN) 3-seed (OT collapse baseline)

  D2 (Dataset 2, 24obj, 1000 test imgs):
    - Random Coupling 3-seed
    - GHSS Coupling 3-seed
    - v-prediction 3-seed

用法:
    python experiments/analysis/test_eval_3seed_gap.py --gpu 0
    python experiments/analysis/test_eval_3seed_gap.py --gpu 0 --group d1   # 仅 D1
    python experiments/analysis/test_eval_3seed_gap.py --gpu 0 --group d2   # 仅 D2
    python experiments/analysis/test_eval_3seed_gap.py --gpu 0 --only "D1 DPM++ seed123"  # 单个
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 复用 test_eval_per_size.py 的评估函数
from experiments.analysis.test_eval_per_size import run_test_eval


# ============================================================
# D1 补缺模型 (Dataset 1, chr2024 test split, 220 imgs)
# 格式: (label, config, checkpoint, dataset)
# ============================================================
D1_GAP_MODELS = [
    # --- KaryoFlow +DPM-Solver++ 3-seed (seed42 已有 test=0.739, 补 seed123/789) ---
    ('D1 KaryoFlow +DPM-Solver++ seed123',
     'experiments/configs/ldmdet/a4_dpm_pp_chr2024.py',
     'work_dirs/a4_dpm_pp_chr2024_seed123/best_coco_bbox_mAP_epoch_85.pth',
     'chr2024'),
    ('D1 KaryoFlow +DPM-Solver++ seed789',
     'experiments/configs/ldmdet/a4_dpm_pp_chr2024.py',
     'work_dirs/a4_dpm_pp_chr2024_seed789/best_coco_bbox_mAP_epoch_72.pth',
     'chr2024'),

    # --- KaryoFlow +Stoch. Coupling 3-seed (无aug, multi_seed/stochot_eps5_old) ---
    # 注: seed42 已有 test=0.740 来自 reproduce_0751_stochot_eps5_v2 (不同实验),
    #     此处补 multi_seed/stochot_eps5_old 3-seed 的 test (与 val 3-seed 同源)
    ('D1 KaryoFlow +StochOT eps5 (3seed) seed42',
     'work_dirs/multi_seed/stochot_eps5_old/seed_42/stochot_eps5_old_multiseed.py',
     'work_dirs/multi_seed/stochot_eps5_old/seed_42/best_coco_bbox_mAP_epoch_60.pth',
     'chr2024'),
    ('D1 KaryoFlow +StochOT eps5 (3seed) seed123',
     'work_dirs/multi_seed/stochot_eps5_old/seed_42/stochot_eps5_old_multiseed.py',
     'work_dirs/multi_seed/stochot_eps5_old/seed_123/best_coco_bbox_mAP_epoch_57.pth',
     'chr2024'),
    ('D1 KaryoFlow +StochOT eps5 (3seed) seed789',
     'work_dirs/multi_seed/stochot_eps5_old/seed_42/stochot_eps5_old_multiseed.py',
     'work_dirs/multi_seed/stochot_eps5_old/seed_789/best_coco_bbox_mAP_epoch_69.pth',
     'chr2024'),

    # --- KaryoFlow v-prediction 3-seed (r3_vpred_chr2024) ---
    ('D1 KaryoFlow v-prediction seed42',
     'work_dirs/r3_vpred_chr2024_seed42/r3_vpred_chr2024.py',
     'work_dirs/r3_vpred_chr2024_seed42/best_coco_bbox_mAP_epoch_72.pth',
     'chr2024'),
    ('D1 KaryoFlow v-prediction seed123',
     'work_dirs/r3_vpred_chr2024_seed42/r3_vpred_chr2024.py',  # 复用 seed42 config
     'work_dirs/r3_vpred_chr2024_seed123/best_coco_bbox_mAP_epoch_57.pth',
     'chr2024'),
    ('D1 KaryoFlow v-prediction seed789',
     'work_dirs/r3_vpred_chr2024_seed42/r3_vpred_chr2024.py',
     'work_dirs/r3_vpred_chr2024_seed789/best_coco_bbox_mAP_epoch_84.pth',
     'chr2024'),

    # --- Hard OT (无aug) 3-seed (OT collapse 论证, multi_seed/hard_ot) ---
    ('D1 Hard OT (NoAug) seed42',
     'work_dirs/multi_seed/hard_ot/seed_42/hard_ot.py',
     'work_dirs/multi_seed/hard_ot/seed_42/best_coco_bbox_mAP_epoch_30.pth',
     'chr2024'),
    ('D1 Hard OT (NoAug) seed123',
     'work_dirs/multi_seed/hard_ot/seed_42/hard_ot.py',
     'work_dirs/multi_seed/hard_ot/seed_123/best_coco_bbox_mAP_epoch_49.pth',
     'chr2024'),
    ('D1 Hard OT (NoAug) seed789',
     'work_dirs/multi_seed/hard_ot/seed_42/hard_ot.py',
     'work_dirs/multi_seed/hard_ot/seed_789/best_coco_bbox_mAP_epoch_43.pth',
     'chr2024'),

    # --- Random (无aug, AdaLN) 3-seed (OT collapse baseline, multi_seed/rf_heun_adaln) ---
    ('D1 Random (NoAug AdaLN) seed42',
     'work_dirs/multi_seed/rf_heun_adaln/seed_42/rf_heun_adaln.py',
     'work_dirs/multi_seed/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_47.pth',
     'chr2024'),
    ('D1 Random (NoAug AdaLN) seed123',
     'work_dirs/multi_seed/rf_heun_adaln/seed_42/rf_heun_adaln.py',
     'work_dirs/multi_seed/rf_heun_adaln/seed_123/best_coco_bbox_mAP_epoch_47.pth',
     'chr2024'),
    ('D1 Random (NoAug AdaLN) seed789',
     'work_dirs/multi_seed/rf_heun_adaln/seed_42/rf_heun_adaln.py',
     'work_dirs/multi_seed/rf_heun_adaln/seed_789/best_coco_bbox_mAP_epoch_25.pth',
     'chr2024'),
]


# ============================================================
# D2 补缺模型 (Dataset 2, 24obj test split, 1000 imgs)
# ============================================================
D2_GAP_MODELS = [
    # --- Random Coupling 3-seed (24obj_ablation/random) ---
    ('D2 Random Coupling seed42',
     'work_dirs/24obj_ablation/random/seed_42/chromo_24obj_random.py',
     'work_dirs/24obj_ablation/random/seed_42/best_coco_bbox_mAP_epoch_59.pth',
     '24obj'),
    ('D2 Random Coupling seed123',
     'work_dirs/24obj_ablation/random/seed_42/chromo_24obj_random.py',
     'work_dirs/24obj_ablation/random/seed_123/best_coco_bbox_mAP_epoch_115.pth',
     '24obj'),
    ('D2 Random Coupling seed789',
     'work_dirs/24obj_ablation/random/seed_42/chromo_24obj_random.py',
     'work_dirs/24obj_ablation/random/seed_789/best_coco_bbox_mAP_epoch_82.pth',
     '24obj'),

    # --- GHSS Coupling 3-seed (24obj_ablation/ghss) ---
    ('D2 GHSS Coupling seed42',
     'work_dirs/24obj_ablation/ghss/seed_42/chromo_24obj.py',
     'work_dirs/24obj_ablation/ghss/seed_42/best_coco_bbox_mAP_epoch_83.pth',
     '24obj'),
    ('D2 GHSS Coupling seed123',
     'work_dirs/24obj_ablation/ghss/seed_42/chromo_24obj.py',
     'work_dirs/24obj_ablation/ghss/seed_123/best_coco_bbox_mAP_epoch_102.pth',
     '24obj'),
    ('D2 GHSS Coupling seed789',
     'work_dirs/24obj_ablation/ghss/seed_42/chromo_24obj.py',
     'work_dirs/24obj_ablation/ghss/seed_789/best_coco_bbox_mAP_epoch_75.pth',
     '24obj'),

    # --- v-prediction 3-seed (r3_vpred_24obj) ---
    ('D2 v-prediction seed42',
     'work_dirs/r3_vpred_24obj_seed42/r3_vpred_24obj.py',
     'work_dirs/r3_vpred_24obj_seed42/best_coco_bbox_mAP_epoch_34.pth',
     '24obj'),
    ('D2 v-prediction seed123',
     'work_dirs/r3_vpred_24obj_seed42/r3_vpred_24obj.py',
     'work_dirs/r3_vpred_24obj_seed123/best_coco_bbox_mAP_epoch_48.pth',
     '24obj'),
    ('D2 v-prediction seed789',
     'work_dirs/r3_vpred_24obj_seed42/r3_vpred_24obj.py',
     'work_dirs/r3_vpred_24obj_seed789/best_coco_bbox_mAP_epoch_51.pth',
     '24obj'),
]


def main():
    parser = argparse.ArgumentParser(
        description='3-seed test mAP 补跑 (补全所有缺失 test 数据)',
    )
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (默认 0, A6000)')
    parser.add_argument('--group', type=str, default='all',
                        choices=['d1', 'd2', 'all'],
                        help='评估组 (d1/d2/all)')
    parser.add_argument('--only', type=str, default=None,
                        help='只评估指定 label (子串匹配)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 路径 (默认带时间戳)')
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    # 确定评估列表
    if args.group == 'd1':
        models = D1_GAP_MODELS
    elif args.group == 'd2':
        models = D2_GAP_MODELS
    else:
        models = D1_GAP_MODELS + D2_GAP_MODELS

    if args.only:
        models = [m for m in models if args.only in m[0]]

    print('=' * 80)
    print('3-seed test mAP 补跑')
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    print(f'待评估: {len(models)} 个模型')
    print('=' * 80)

    # 输出路径
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f'work_dirs/diagnosis/test_eval_3seed_gap_{ts}.json'
    else:
        output = args.output

    all_results = []
    total = len(models)
    success = 0
    skip = 0
    fail = 0

    for idx, (label, config, ckpt, dataset) in enumerate(models):
        config_path = os.path.join(_PROJECT_ROOT, config) if not os.path.isabs(config) else config
        ckpt_path = os.path.join(_PROJECT_ROOT, ckpt) if not os.path.isabs(ckpt) else ckpt

        print(f'\n{"=" * 80}')
        print(f'[{idx + 1}/{total}] {label}')
        print(f'{"=" * 80}')

        # 检查 config 和 ckpt 存在性
        if not os.path.exists(config_path):
            print(f'  [跳过] config 不存在: {config_path}')
            skip += 1
            continue
        if not os.path.exists(ckpt_path):
            print(f'  [跳过] checkpoint 不存在: {ckpt_path}')
            skip += 1
            continue
        # 文件完整性检查
        ckpt_size = os.path.getsize(ckpt_path)
        if ckpt_size < 10 * 1024 * 1024:
            print(f'  [跳过] checkpoint 不完整 ({ckpt_size} bytes): {ckpt_path}')
            skip += 1
            continue

        start = time.time()
        try:
            result = run_test_eval(config_path, ckpt_path, dataset, device, label)
            if result:
                result['eval_time_sec'] = round(time.time() - start, 1)
                all_results.append(result)
                success += 1
                print(f'\n  [OK] mAP={result["mAP"]:.4f} ({result["eval_time_sec"]}s)')
            else:
                fail += 1
                print(f'\n  [FAIL] 评估返回 None')
        except Exception as e:
            fail += 1
            print(f'\n  [错误] {label}: {e}')
            import traceback
            traceback.print_exc()
            import torch
            torch.cuda.empty_cache()

    # ===== 汇总 =====
    print('\n' + '=' * 80)
    print('汇总')
    print('=' * 80)
    print(f'{"Label":55s} {"mAP":>8s} {"AP50":>8s} {"AP75":>8s} {"AP_S":>8s} {"AP_M":>8s} {"AP_L":>8s}')
    print('-' * 103)
    for r in all_results:
        print(f'{r["label"]:55s} {r["mAP"]:8.4f} {r["AP50"]:8.4f} {r["AP75"]:8.4f} '
              f'{r["AP_S"]:8.4f} {r["AP_M"]:8.4f} {r["AP_L"]:8.4f}')

    print(f'\n总计: {total} | 成功: {success} | 跳过: {skip} | 失败: {fail}')

    # 保存 JSON
    os.makedirs(os.path.dirname(output), exist_ok=True)
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'total': total,
        'success': success,
        'skip': skip,
        'fail': fail,
        'results': all_results,
    }
    with open(output, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {output}')


if __name__ == '__main__':
    main()
