#!/usr/bin/env python3
"""跨数据集 per-class AP 分析 (2026-07-30)

用 D1 训练的模型在 D2 test 上评估 per-class AP,
判断跨域检测失效是类别顺序不一致占主导还是数据集跨域占主导。

D1 类别顺序: RF+Heun +AdaLN-Zero +Stoch. Coupling B4 B5 [C10 C11 C12 C6 C7 C8 C9] D13... (index 5-11 为 C 组, 与 D2 不一致)
D2 类别顺序: RF+Heun +AdaLN-Zero +Stoch. Coupling B4 B5 [C6  C7  C8  C9 C10 C11 C12] D13... (index 5-11 为 C 组, 数字序)

matched indices (0-4, 12-23): 两数据集类别相同
mismatched indices (5-11): C 组类别顺序不同

用法:
    python experiments/analysis/cross_dataset_per_class.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# D1 类别顺序 (模型训练时的顺序)
D1_CLASSES = [
    'A1', 'A2', 'A3', 'B4', 'B5',
    'C10', 'C11', 'C12', 'C6', 'C7', 'C8', 'C9',  # index 5-11: 与 D2 不一致
    'D13', 'D14', 'D15', 'E16', 'E17', 'E18',
    'F19', 'F20', 'G21', 'G22', 'X', 'Y',
]

# D2 类别顺序 (评估数据集的顺序)
D2_CLASSES = [
    'A1', 'A2', 'A3', 'B4', 'B5',
    'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',  # index 5-11: 数字序
    'D13', 'D14', 'D15', 'E16', 'E17', 'E18',
    'F19', 'F20', 'G21', 'G22', 'X', 'Y',
]

# matched: index 0-4, 12-23 (两数据集类别名相同)
# mismatched: index 5-11 (C 组顺序不同)
MATCHED_INDICES = list(range(5)) + list(range(12, 24))
MISMATCHED_INDICES = list(range(5, 12))


def run_per_class_eval(config_path, checkpoint, device='cuda:0'):
    """在 D2 test 上评估 per-class AP"""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)

    D2_DATA_ROOT = 'data/24_chromosomes_object/coco/'
    D2_TEST_ANN = D2_DATA_ROOT + 'test/_annotations.coco.json'

    # 强制 D2 test split
    test_dataset_cfg = dict(
        type=cfg.test_dataloader.dataset.type if hasattr(cfg, 'test_dataloader')
        else cfg.val_dataloader.dataset.type,
        data_root=D2_DATA_ROOT,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=cfg.test_dataloader.dataset.pipeline
        if hasattr(cfg, 'test_dataloader')
        else cfg.val_dataloader.dataset.pipeline,
    )
    # 使用 D1 的 metainfo (模型训练时的类别顺序)
    test_dataset_cfg['metainfo'] = dict(classes=tuple(D1_CLASSES))

    cfg.test_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=test_dataset_cfg,
    )

    # classwise=True 启用 per-class AP
    cfg.test_evaluator = dict(
        type='CocoMetric',
        ann_file=D2_TEST_ANN,
        metric='bbox',
        classwise=True,
        format_only=False,
    )

    # 禁用 SwanLab / EMA
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends,
        name='visualizer',
    )
    custom_hooks = getattr(cfg, 'custom_hooks', [])
    cfg.custom_hooks = [
        h for h in custom_hooks if h.get('type') != 'EMAHook'
    ]

    print(f'Config: {config_path}')
    print(f'Checkpoint: {checkpoint}')
    print(f'Device: {device}')
    print(f'D1 classes (model): {D1_CLASSES}')
    print(f'D2 classes (eval):   {D2_CLASSES}')
    print(f'Matched indices: {MATCHED_INDICES}')
    print(f'Mismatched indices: {MISMATCHED_INDICES}')

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

            if (i + 1) % 200 == 0:
                print(f'  进度 {i+1}/{n_imgs}')

    metrics = evaluator.evaluate(num_samples)

    # 提取 per-class AP
    per_class_ap = {}
    per_class_ap50 = {}
    for i, cls_name in enumerate(D1_CLASSES):
        # CocoMetric 输出格式: coco/CLASS_precision, coco/CLASS_mAP_50 等
        # 也可能用 class name 直接作为 key
        prefix = f'coco/{cls_name}'
        ap_key = f'{prefix}_mAP'
        ap50_key = f'{prefix}_mAP_50'

        # 尝试不同的 key 格式
        ap = None
        ap50 = None
        for k, v in metrics.items():
            if cls_name in k and 'mAP_50' in k:
                ap50 = v
            elif cls_name in k and k.endswith('_mAP') and '_50' not in k and '_75' not in k:
                ap = v
            elif cls_name in k and 'precision' in k and 'mAP' not in k:
                ap = v

        per_class_ap[cls_name] = ap if ap is not None else 0.0
        per_class_ap50[cls_name] = ap50 if ap50 is not None else 0.0

    return per_class_ap, per_class_ap50, metrics


def main():
    parser = argparse.ArgumentParser(
        description='跨数据集 per-class AP 分析'
    )
    parser.add_argument(
        '--config', type=str,
        default='experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py',
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='work_dirs/ablation_old/reproduce_0751_stochot_eps5_v2/'
                'best_coco_bbox_mAP_epoch_59.pth',
    )
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    config_path = os.path.join(_PROJECT_ROOT, args.config)
    checkpoint = os.path.join(_PROJECT_ROOT, args.checkpoint)

    per_class_ap, per_class_ap50, raw_metrics = run_per_class_eval(
        config_path, checkpoint, device
    )

    # 打印所有 metrics keys 用于调试
    print('\n=== All metric keys ===')
    for k in sorted(raw_metrics.keys()):
        if any(c in k for c in D1_CLASSES):
            print(f'  {k}: {raw_metrics[k]}')

    # 打印 per-class AP 表
    print('\n' + '=' * 90)
    print('Per-class AP (D1 model on D2 test)')
    print('=' * 90)
    print(f'{"Index":<7} {"D1类":<6} {"D2类":<6} {"匹配?":<8} {"AP":>8} {"AP50":>8}')
    print('-' * 50)

    matched_aps = []
    mismatched_aps = []
    matched_ap50s = []
    mismatched_ap50s = []

    for i, d1_cls in enumerate(D1_CLASSES):
        d2_cls = D2_CLASSES[i]
        matched = '✓' if i in MATCHED_INDICES else '✗'
        ap = per_class_ap[d1_cls]
        ap50 = per_class_ap50[d1_cls]

        print(f'{i:<7} {d1_cls:<6} {d2_cls:<6} {matched:<8} {ap:>8.4f} {ap50:>8.4f}')

        if i in MATCHED_INDICES:
            matched_aps.append(ap)
            matched_ap50s.append(ap50)
        else:
            mismatched_aps.append(ap)
            mismatched_ap50s.append(ap50)

    # 汇总统计
    import statistics
    print('\n' + '=' * 50)
    print('汇总统计')
    print('=' * 50)
    print(f'Matched classes (n={len(matched_aps)}):')
    print(f'  AP   mean={statistics.mean(matched_aps):.4f} '
          f'median={statistics.median(matched_aps):.4f} '
          f'min={min(matched_aps):.4f} max={max(matched_aps):.4f}')
    print(f'  AP50 mean={statistics.mean(matched_ap50s):.4f} '
          f'median={statistics.median(matched_ap50s):.4f}')

    print(f'Mismatched classes (n={len(mismatched_aps)}):')
    print(f'  AP   mean={statistics.mean(mismatched_aps):.4f} '
          f'median={statistics.median(mismatched_aps):.4f} '
          f'min={min(mismatched_aps):.4f} max={max(mismatched_aps):.4f}')
    print(f'  AP50 mean={statistics.mean(mismatched_ap50s):.4f} '
          f'median={statistics.median(mismatched_ap50s):.4f}')

    # 判断主导因素
    matched_mean = statistics.mean(matched_aps)
    mismatched_mean = statistics.mean(mismatched_aps)
    print(f'\nMatched mean AP:    {matched_mean:.4f}')
    print(f'Mismatched mean AP: {mismatched_mean:.4f}')
    print(f'Ratio (mismatched/matched): {mismatched_mean/max(matched_mean, 1e-6):.4f}')

    if matched_mean > 0.15 and mismatched_mean < matched_mean * 0.3:
        print('\n结论: 类别顺序不一致占主导 (matched AP 远高于 mismatched AP)')
    elif matched_mean < 0.10:
        print('\n结论: 数据集跨域占主导 (matched AP 也很低, 模型无法泛化)')
    else:
        print(f'\n结论: 混合因素 (matched AP={matched_mean:.4f}, '
              f'mismatched AP={mismatched_mean:.4f})')

    # 保存结果
    from datetime import datetime
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'cross_dataset_per_class_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'reproduce_0751_stochot_eps5_v2 (D1 trained)',
        'eval_dataset': 'D2 test (24 Chromosomes Object)',
        'per_class': [
            {
                'index': i,
                'd1_class': D1_CLASSES[i],
                'd2_class': D2_CLASSES[i],
                'matched': i in MATCHED_INDICES,
                'AP': per_class_ap[D1_CLASSES[i]],
                'AP50': per_class_ap50[D1_CLASSES[i]],
            }
            for i in range(24)
        ],
        'summary': {
            'matched_mean_AP': matched_mean,
            'mismatched_mean_AP': mismatched_mean,
            'matched_mean_AP50': statistics.mean(matched_ap50s),
            'mismatched_mean_AP50': statistics.mean(mismatched_ap50s),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
