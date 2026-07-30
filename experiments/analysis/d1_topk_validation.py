#!/usr/bin/env python3
"""Dataset 1 Top-K Proposal Pruning 验证 (2026-07-30)

填补 §四 Top-K 双数据集验证缺口: D2 K=100/200/300 已完成, D1 未跑。
推理零成本实验, 复用 D1 DPM-Solver++ checkpoint (a4_dpm_pp_chr2024_seed42).

实验目的:
  1. 验证 Top-K K=100/200/300 在 Dataset 1 (1540 张, ~46 GT/图) 上的 mAP
  2. 与 Dataset 2 结果对照: D2 K=200 最优 (0.860), K=100 掉点 (0.850 single-seed)
  3. 预测: D1 重叠冗余更少, K=100 可能已足够 (D1 K 最优解可能与 D2 不同)
  4. 同时跑 renewal on/off, 闭合 §六 box_renewal K 值依赖性在 D1 的验证

关联文档: EXPERIMENT_LINEAGE.md §四 (Top-K) + §六 (D3 Box Renewal)
关联 checkpoint: work_dirs/a4_dpm_pp_chr2024_seed42/best_coco_bbox_mAP_epoch_49.pth
关联 config: experiments/configs/ldmdet/a4_dpm_pp_chr2024.py

Usage:
    python experiments/analysis/d1_topk_validation.py --gpu 0 --seed 42
    python experiments/analysis/d1_topk_validation.py --gpu 0 --seed 123
    python experiments/analysis/d1_topk_validation.py --gpu 0 --seed 789
    python experiments/analysis/d1_topk_validation.py --gpu 0 --seed 42 --max-imgs 50  # 快速测试
    python experiments/analysis/d1_topk_validation.py --gpu 0 --checkpoint /path/to/ckpt.pth  # 自定义 checkpoint
"""
from __future__ import annotations

import argparse
import json
import os
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
# 配置
# ============================================================

D1_CONFIG = os.path.join(
    _PROJECT_ROOT, 'experiments/configs/ldmdet/a4_dpm_pp_chr2024.py',
)
D1_ANN = os.path.join(
    _PROJECT_ROOT,
    'data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json',
)


def find_d1_checkpoint(seed):
    """根据 seed 自动查找 D1 +DPM-Solver++ checkpoint.

    优先查找 best_coco_bbox_mAP_*.pth, 回退到 epoch_50.pth.
    """
    seed_dir = os.path.join(
        _PROJECT_ROOT, f'work_dirs/a4_dpm_pp_chr2024_seed{seed}',
    )
    if not os.path.isdir(seed_dir):
        return None
    # 优先 best checkpoint
    bests = sorted([f for f in os.listdir(seed_dir)
                    if f.startswith('best_coco_bbox_mAP_') and f.endswith('.pth')])
    if bests:
        return os.path.join(seed_dir, bests[0])
    # 回退到最后一个 epoch checkpoint
    epochs = sorted([f for f in os.listdir(seed_dir)
                     if f.startswith('epoch_') and f.endswith('.pth')])
    if epochs:
        return os.path.join(seed_dir, epochs[-1])
    return None

# D2 参考数据 (LINEAGE §四, seed42)
D2_REF = {
    'K=500': {'mAP': 0.863, 'APs': None, 'lat': None},
    'K=300': {'mAP': 0.861, 'APs': None, 'lat': 71.3},
    'K=200': {'mAP': 0.860, 'APs': None, 'lat': 70.5},
    'K=100': {'mAP': 0.850, 'APs': None, 'lat': 69.7},
}

# D1 参考数据 (LINEAGE §三, seed42, renewal ON)
D1_REF = {
    'K=500_ON': {'mAP': 0.744, 'source': 'renewal_off_all_scenarios.json Dataset1 (+DPM-Solver++) ON'},
    'K=500_OFF': {'mAP': 0.743, 'source': 'renewal_off_all_scenarios.json Dataset1 (+DPM-Solver++) OFF'},
}


def run_topk_eval(config_path, checkpoint, ann_file, topk_k, box_renewal,
                  device='cuda:0', max_imgs=None):
    """在给定 Top-K 和 box_renewal 配置下评估 D1 mAP + 延迟.

    与 verify_renewal_off_all_scenarios.py 的 run_config 不同, 本函数
    额外覆盖 topk_pruning_enabled / topk_k 参数.
    """
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)

    # DPM-Solver++ 4 步 (config 已设, 显式确认)
    cfg.model.bbox_head.solver_type = 'dpm_solver_pp'
    cfg.model.bbox_head.sampling_timesteps = 4

    # box_renewal 开关
    cfg.model.bbox_head.box_renewal = box_renewal

    # Top-K 剪枝开关
    if topk_k is not None and topk_k < 500:
        cfg.model.bbox_head.topk_pruning_enabled = True
        cfg.model.bbox_head.topk_k = topk_k
        cfg.model.bbox_head.topk_pruning_step = 0
    else:
        cfg.model.bbox_head.topk_pruning_enabled = False

    # 禁用 SwanLab
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer',
    )

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    # 验证 Top-K 配置传递
    head = model.bbox_head
    print(f'    [验证] topk_pruning_enabled={head.topk_pruning_enabled}, '
          f'topk_k={head.topk_k}, topk_pruning_step={head.topk_pruning_step}')
    # sampling_timesteps 传给 sampler (非 head 属性), 从 cfg 读取确认
    print(f'    [验证] solver_type={head.solver_type}, '
          f'sampling_timesteps={cfg.model.bbox_head.sampling_timesteps}, '
          f'box_renewal={head.box_renewal}')

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    evaluator = METRICS.build(dict(
        type='CocoMetric', ann_file=ann_file, metric='bbox',
        classwise=False, format_only=False,
    ))
    evaluator.dataset_meta = dataset.metainfo

    latencies = []
    num_samples = 0
    n_imgs = len(dataset) if max_imgs is None else min(max_imgs, len(dataset))

    with torch.no_grad():
        for i in range(n_imgs):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            if 'cuda' in device:
                torch.cuda.synchronize()
            t0 = time.time()
            out = model.test_step(data)
            if 'cuda' in device:
                torch.cuda.synchronize()
            latencies.append(time.time() - t0)

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

    metrics = evaluator.evaluate(num_samples)

    import numpy as np
    lat_arr = np.array(latencies[5:])  # 跳过前 5 张 warmup
    result = {
        'mAP': metrics.get('coco/bbox_mAP', 0.0),
        'AP50': metrics.get('coco/bbox_mAP_50', 0.0),
        'AP75': metrics.get('coco/bbox_mAP_75', 0.0),
        'APs': metrics.get('coco/bbox_mAP_s', 0.0),
        'APm': metrics.get('coco/bbox_mAP_m', 0.0),
        'APl': metrics.get('coco/bbox_mAP_l', 0.0),
        'avg_latency_ms': float(lat_arr.mean() * 1000) if len(lat_arr) > 0 else 0.0,
        'fps': float(1.0 / lat_arr.mean()) if len(lat_arr) > 0 else 0.0,
        'n_images': n_imgs,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Dataset 1 Top-K Pruning 验证 (闭合 §四 双数据集缺口)',
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (42/123/789), 用于自动查找 checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='手动指定 checkpoint 路径 (覆盖 --seed 自动查找)')
    parser.add_argument('--max-imgs', type=int, default=None,
                        help='限制评估图像数 (调试用, None=全量)')
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    # 确定 checkpoint
    if args.checkpoint is not None:
        d1_ckpt = args.checkpoint
    else:
        d1_ckpt = find_d1_checkpoint(args.seed)
    if d1_ckpt is None or not os.path.exists(d1_ckpt):
        print(f'[错误] checkpoint 不存在: seed={args.seed}, '
              f'查找路径=work_dirs/a4_dpm_pp_chr2024_seed{args.seed}/')
        sys.exit(1)
    if not os.path.exists(D1_ANN):
        print(f'[错误] annotation 不存在: {D1_ANN}')
        sys.exit(1)

    print('=' * 80)
    print(f'Dataset 1 Top-K Proposal Pruning 验证 (seed {args.seed})')
    print('=' * 80)
    print(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Device: {device}')
    print(f'Seed: {args.seed}')
    print(f'Config: {D1_CONFIG}')
    print(f'Checkpoint: {d1_ckpt}')
    print(f'Annotation: {D1_ANN}')
    print(f'Max imgs: {args.max_imgs or "全量"}')
    print()

    # 实验矩阵: K × renewal on/off (完整 8 场景)
    # 同时闭合 §四 (Top-K mAP) + §六 (box_renewal K 值依赖性) 双数据集缺口
    # D2 已有 K={500,300,200,100} × renewal {ON,OFF} 全矩阵, D1 此前仅 K=500 ON/OFF
    scenarios = [
        # (label, topk_k, renewal_on)
        ('D1 K=500 renewal ON',  500, True),
        ('D1 K=500 renewal OFF', 500, False),
        ('D1 K=300 renewal ON',  300, True),
        ('D1 K=300 renewal OFF', 300, False),
        ('D1 K=200 renewal ON',  200, True),
        ('D1 K=200 renewal OFF', 200, False),
        ('D1 K=100 renewal ON',  100, True),
        ('D1 K=100 renewal OFF', 100, False),
    ]

    all_results = {}

    for label, topk_k, renewal_on in scenarios:
        tag = 'ON' if renewal_on else 'OFF'
        k_label = f'K={topk_k}'
        print(f'\n{"=" * 80}')
        print(f'场景: {label}')
        print(f'{"=" * 80}')

        r = run_topk_eval(
            D1_CONFIG, d1_ckpt, D1_ANN,
            topk_k=topk_k, box_renewal=renewal_on,
            device=device, max_imgs=args.max_imgs,
        )
        all_results[label] = r

        # 对比参考
        if topk_k == 500:
            ref = D1_REF.get(f'K=500_{tag}', {})
            ref_mAP = ref.get('mAP')
            if ref_mAP:
                print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  '
                      f'AP75={r["AP75"]:.4f}  APs={r["APs"]:.4f}  '
                      f'lat={r["avg_latency_ms"]:.1f}ms')
                print(f'  vs ref ({ref_mAP}): Δ={r["mAP"] - ref_mAP:+.4f}')
        else:
            d2_ref = D2_REF.get(k_label, {})
            d2_mAP = d2_ref.get('mAP')
            d1_base = all_results.get('D1 K=500 renewal ON', {}).get('mAP', 0.744)
            print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  '
                  f'AP75={r["AP75"]:.4f}  APs={r["APs"]:.4f}  '
                  f'lat={r["avg_latency_ms"]:.1f}ms')
            print(f'  vs D1 K=500 baseline ({d1_base:.4f}): '
                  f'Δ={r["mAP"] - d1_base:+.4f}')
            if d2_mAP:
                print(f'  vs D2 {k_label} ({d2_mAP:.4f}): '
                      f'D1-D2 Δ={r["mAP"] - d2_mAP:+.4f}')

    # ===================== 汇总表 =====================
    print('\n' + '=' * 80)
    print(f'汇总: Dataset 1 Top-K Pruning (DPM-Solver++ 4-step, seed{args.seed})')
    print('=' * 80)
    print(f'{"场景":<28} {"mAP":>8} {"AP50":>8} {"AP75":>8} {"APs":>8} '
          f'{"lat(ms)":>8} {"ΔvsK500":>8}')
    print('-' * 80)

    d1_k500_on = all_results.get('D1 K=500 renewal ON', {}).get('mAP', 0.0)
    for label in [s[0] for s in scenarios]:
        r = all_results[label]
        delta = r['mAP'] - d1_k500_on
        print(f'{label:<28} {r["mAP"]:>8.4f} {r["AP50"]:>8.4f} {r["AP75"]:>8.4f} '
              f'{r["APs"]:>8.4f} {r["avg_latency_ms"]:>7.1f} {delta:>+8.4f}')

    # D2 对照
    print(f'\n--- Dataset 2 对照 (seed42, LINEAGE §四) ---')
    print(f'{"D2 K=500 (baseline)":<28} {D2_REF["K=500"]["mAP"]:>8.4f}')
    print(f'{"D2 K=300":<28} {D2_REF["K=300"]["mAP"]:>8.4f}  '
          f'Δ={D2_REF["K=300"]["mAP"]-D2_REF["K=500"]["mAP"]:>+8.4f}')
    print(f'{"D2 K=200 [最优]":<28} {D2_REF["K=200"]["mAP"]:>8.4f}  '
          f'Δ={D2_REF["K=200"]["mAP"]-D2_REF["K=500"]["mAP"]:>+8.4f}')
    print(f'{"D2 K=100 [掉点]":<28} {D2_REF["K=100"]["mAP"]:>8.4f}  '
          f'Δ={D2_REF["K=100"]["mAP"]-D2_REF["K=500"]["mAP"]:>+8.4f}')

    # 保存结果
    out_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        f'd1_topk_validation_seed{args.seed}.json',
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed': args.seed,
        'checkpoint': d1_ckpt,
        'config': D1_CONFIG,
        'annotation': D1_ANN,
        'max_imgs': args.max_imgs,
        'd2_ref': D2_REF,
        'd1_ref': D1_REF,
        'results': all_results,
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
