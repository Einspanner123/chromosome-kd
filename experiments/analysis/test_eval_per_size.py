#!/usr/bin/env python3
"""Test 集 per-size AP 评估 (2026-07-30)

在 test split 上评估所有模型的 per-size AP (mAP, AP_S, AP_M, AP_L, AP50, AP75)。
不统计 FPS 数据 (FPS 统一在 ross 上进行)。

支持:
  - Dataset 1 (Chromosome20240904) test split (220 张图)
  - Dataset 2 (24 Chromosomes Object) test split (1000 张图)

用法:
    # 评估单个模型
    python experiments/analysis/test_eval_per_size.py \
        --config experiments/configs/ldmdet/a4_dpm_pp_chr2024.py \
        --checkpoint work_dirs/a4_dpm_pp_chr2024_seed42/best_coco_bbox_mAP_epoch_49.pth \
        --dataset chr2024 --gpu 1 --label "D1 A4 DPM++ seed42"

    # 评估所有 D1 模型
    python experiments/analysis/test_eval_per_size.py --batch d1 --gpu 1

    # 评估 D2 补缺 (DINO R50 + RTMDet-L)
    python experiments/analysis/test_eval_per_size.py --batch d2_missing --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# 模型注册表: (label, config, checkpoint, dataset)
# dataset: 'chr2024' (D1) or '24obj' (D2)
# ============================================================

D1_DATA_ROOT = 'data/Chromosome20240904_NoAug_NoResize_coco/'
D2_DATA_ROOT = 'data/24_chromosomes_object/coco/'

D1_TEST_ANN = D1_DATA_ROOT + 'test/_annotations.coco.json'
D2_TEST_ANN = D2_DATA_ROOT + 'test/_annotations.coco.json'

# D1 模型清单
D1_MODELS = [
    # KaryoFlow RF+Heun 3 seeds
    ('D1 KaryoFlow RF+Heun seed42',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/rf_heun_adaln.py',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth',
     'chr2024'),
    ('D1 KaryoFlow RF+Heun seed123',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/rf_heun_adaln.py',  # 复用 seed42 config
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_123/best_coco_bbox_mAP_epoch_101.pth',
     'chr2024'),
    ('D1 KaryoFlow RF+Heun seed789',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/rf_heun_adaln.py',
     'work_dirs/multi_seed_aug/rf_heun_adaln/seed_789/best_coco_bbox_mAP_epoch_75.pth',
     'chr2024'),
    # DDPM 3 seeds
    ('D1 DiffusionDet DDPM seed42',
     'work_dirs/multi_seed_aug/ddpm/seed_42/diffusiondet_ddpm.py',
     'work_dirs/multi_seed_aug/ddpm/seed_42/best_coco_bbox_mAP_epoch_79.pth',
     'chr2024'),
    ('D1 DiffusionDet DDPM seed123',
     'work_dirs/multi_seed_aug/ddpm/seed_42/diffusiondet_ddpm.py',
     'work_dirs/multi_seed_aug/ddpm/seed_123/best_coco_bbox_mAP_epoch_66.pth',
     'chr2024'),
    ('D1 DiffusionDet DDPM seed789',
     'work_dirs/multi_seed_aug/ddpm/seed_42/diffusiondet_ddpm.py',
     'work_dirs/multi_seed_aug/ddpm/seed_789/best_coco_bbox_mAP_epoch_87.pth',
     'chr2024'),
    # A4 DPM++ D1 seed42
    ('D1 KaryoFlow A4 DPM++ seed42',
     'experiments/configs/ldmdet/a4_dpm_pp_chr2024.py',
     'work_dirs/a4_dpm_pp_chr2024_seed42/best_coco_bbox_mAP_epoch_49.pth',
     'chr2024'),
    # D1 SOTA baselines (支持 ross 和 workstation 两种路径)
    ('D1 Cascade R-CNN R50',
     'experiments/configs/baselines/benchmark/cascade_rcnn_r50.py',
     ['work_dirs/baselines/cascade_rcnn_r50_20240904/best_coco_bbox_mAP_epoch_86.pth',
      'work_dirs/benchmark/cascade_rcnn_r50/best_coco_bbox_mAP_epoch_86.pth'],
     'chr2024'),
    ('D1 RTMDet-L',
     'experiments/configs/baselines/benchmark/rtmdet_l.py',
     ['work_dirs/baselines/rtmdet_l_20240904/best_coco_bbox_mAP_epoch_52.pth',
      'work_dirs/benchmark/rtmdet_l/best_coco_bbox_mAP_epoch_52.pth'],
     'chr2024'),
    ('D1 YOLOX-S',
     'experiments/configs/baselines/benchmark/yolox_s.py',
     ['work_dirs/baselines/yolox_s_20240904/best_coco_bbox_mAP_epoch_150.pth',
      'work_dirs/benchmark/yolox_s/best_coco_bbox_mAP_epoch_150.pth'],
     'chr2024'),
    ('D1 DINO R50 (训练中 Ep101)',
     'experiments/configs/baselines/benchmark/dino_r50.py',
     'work_dirs/baselines/dino_r50_20240904/epoch_101.pth',
     'chr2024'),
]

# D2 补缺模型
D2_MISSING_MODELS = [
    ('D2 DINO R50',
     'experiments/configs/baselines/benchmark_24obj/dino_r50.py',
     'work_dirs/baselines/dino_r50_24obj/best_coco_bbox_mAP_epoch_102.pth',
     '24obj'),
    ('D2 RTMDet-L',
     'experiments/configs/baselines/benchmark_24obj/rtmdet_l.py',
     'work_dirs/baselines/rtmdet_l_24obj/epoch_85.pth',
     '24obj'),
]


def run_test_eval(config_path, checkpoint, dataset, device='cuda:1',
                  label='', max_retries=1):
    """在 test split 上评估单个模型, 返回 per-size AP dict。

    关键: 强制将 test_dataloader 和 test_evaluator 的 ann_file 修改为 test split,
    避免某些 config 中 test_evaluator = val_evaluator 导致 ann_file 指向 valid/。
    """
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)

    # 确定数据集路径
    if dataset == 'chr2024':
        data_root = D1_DATA_ROOT
        test_ann = D1_TEST_ANN
    else:
        data_root = D2_DATA_ROOT
        test_ann = D2_TEST_ANN

    # ===== 强制修改 test_dataloader 指向 test split =====
    # 某些 config 中 test_dataloader = val_dataloader, 需要覆盖
    test_dataset_cfg = dict(
        type=cfg.test_dataloader.dataset.type if hasattr(cfg, 'test_dataloader') else
        cfg.val_dataloader.dataset.type,
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=cfg.test_dataloader.dataset.pipeline if hasattr(cfg, 'test_dataloader') else
        cfg.val_dataloader.dataset.pipeline,
    )
    # 保留 metainfo
    if hasattr(cfg, 'test_dataloader') and 'metainfo' in cfg.test_dataloader.dataset:
        test_dataset_cfg['metainfo'] = cfg.test_dataloader.dataset.metainfo
    elif 'metainfo' in cfg.val_dataloader.dataset:
        test_dataset_cfg['metainfo'] = cfg.val_dataloader.dataset.metainfo

    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=test_dataset_cfg,
    )

    # ===== 强制修改 test_evaluator 指向 test split =====
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=test_ann,
        metric='bbox',
        classwise=False,
        format_only=False,
    )

    # 禁用 SwanLab
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends,
        name='visualizer',
    )

    # 禁用 EMA (test 模式下可能未初始化)
    custom_hooks = getattr(cfg, 'custom_hooks', [])
    cfg.custom_hooks = [
        h for h in custom_hooks if h.get('type') != 'EMAHook'
    ]

    print(f'\n{"=" * 80}')
    print(f'评估: {label}')
    print(f'{"=" * 80}')
    print(f'  Config: {config_path}')
    print(f'  Checkpoint: {checkpoint}')
    print(f'  Dataset: {dataset} (test split)')
    print(f'  Ann file: {test_ann}')
    print(f'  Device: {device}')

    if not os.path.exists(checkpoint):
        print(f'  [错误] checkpoint 不存在!')
        return None

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    eval_dataset = DATASETS.build(cfg.test_dataloader.dataset)
    evaluator = METRICS.build(cfg.test_evaluator)
    evaluator.dataset_meta = eval_dataset.metainfo

    num_samples = 0
    n_imgs = len(eval_dataset)

    with torch.no_grad():
        for i in range(n_imgs):
            data = eval_dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            out = model.test_step(data)

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
                print(f'    进度 {i+1}/{n_imgs}')

    metrics = evaluator.evaluate(num_samples)

    result = {
        'label': label,
        'config': config_path,
        'checkpoint': checkpoint,
        'dataset': dataset,
        'n_images': n_imgs,
        'mAP': metrics.get('coco/bbox_mAP', 0.0),
        'AP50': metrics.get('coco/bbox_mAP_50', 0.0),
        'AP75': metrics.get('coco/bbox_mAP_75', 0.0),
        'AP_S': metrics.get('coco/bbox_mAP_s', 0.0),
        'AP_M': metrics.get('coco/bbox_mAP_m', 0.0),
        'AP_L': metrics.get('coco/bbox_mAP_l', 0.0),
    }

    print(f'\n  结果: mAP={result["mAP"]:.4f} AP50={result["AP50"]:.4f} '
          f'AP75={result["AP75"]:.4f} AP_S={result["AP_S"]:.4f} '
          f'AP_M={result["AP_M"]:.4f} AP_L={result["AP_L"]:.4f}')

    # 清理 GPU 缓存
    del model
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Test 集 per-size AP 评估 (不统计 FPS)',
    )
    parser.add_argument('--config', type=str, default=None,
                        help='单个模型 config 路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='单个模型 checkpoint 路径')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['chr2024', '24obj'],
                        help='数据集 (chr2024=D1, 24obj=D2)')
    parser.add_argument('--label', type=str, default='',
                        help='模型标签 (用于结果记录)')
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU ID (默认 1, 即 A4000)')
    parser.add_argument('--batch', type=str, default=None,
                        choices=['d1', 'd2_missing', 'all'],
                        help='批量评估模式')
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    print('=' * 80)
    print('Test 集 per-size AP 评估')
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    print('=' * 80)

    # 确定评估列表
    if args.batch:
        if args.batch == 'd1':
            models = D1_MODELS
        elif args.batch == 'd2_missing':
            models = D2_MISSING_MODELS
        elif args.batch == 'all':
            models = D1_MODELS + D2_MISSING_MODELS
        else:
            print(f'未知 batch 模式: {args.batch}')
            sys.exit(1)
    elif args.config and args.checkpoint and args.dataset:
        models = [(args.label, args.config, args.checkpoint, args.dataset)]
    else:
        print('请指定 --batch 或 --config + --checkpoint + --dataset')
        sys.exit(1)

    all_results = []
    for label, config, ckpt, dataset in models:
        config_path = os.path.join(_PROJECT_ROOT, config) if not os.path.isabs(config) else config

        # 支持 checkpoint 路径为列表 (多候选路径, 自动选择第一个存在的)
        if isinstance(ckpt, list):
            ckpt_path = None
            for c in ckpt:
                c_full = os.path.join(_PROJECT_ROOT, c) if not os.path.isabs(c) else c
                if os.path.exists(c_full):
                    ckpt_path = c_full
                    break
        else:
            ckpt_path = os.path.join(_PROJECT_ROOT, ckpt) if not os.path.isabs(ckpt) else ckpt

        if not os.path.exists(config_path):
            print(f'\n[跳过] {label}: config 不存在 ({config_path})')
            continue
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f'\n[跳过] {label}: checkpoint 不存在')
            continue

        result = run_test_eval(config_path, ckpt_path, dataset, device, label)
        if result:
            all_results.append(result)

    # ===== 汇总表 =====
    print('\n' + '=' * 80)
    print('汇总: Test 集 per-size AP')
    print('=' * 80)
    print(f'{"模型":<35} {"mAP":>8} {"AP50":>8} {"AP75":>8} '
          f'{"AP_S":>8} {"AP_M":>8} {"AP_L":>8}')
    print('-' * 95)
    for r in all_results:
        print(f'{r["label"]:<35} {r["mAP"]:>8.4f} {r["AP50"]:>8.4f} '
              f'{r["AP75"]:>8.4f} {r["AP_S"]:>8.4f} {r["AP_M"]:>8.4f} '
              f'{r["AP_L"]:>8.4f}')

    # 保存结果
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'test_eval_per_size_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'results': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
