#!/usr/bin/env python3
"""Baseline + Full-Heun 3-seed 评估 (box_renewal OFF, 与 hybrid 3-seed 对齐)

目的:
  1. 生成可验证的 baseline [1,1,1,1] 3-seed JSON (替代硬编码循环引用)
  2. 补全缺失的 "全 Heun" 对照, 区分:
     - Heun 本身有害 (全 Heun 也掉点) vs
     - 混合策略有害 (仅 w/h Heun 掉点, 全 Heun 不掉)

条件: box_renewal OFF (与 hybrid_3seed_norenewal.json 一致, 干净隔离 solver 效果)

配置:
  - baseline: solver_type='dpm_solver_pp', dim_d1_mask=None (全 DPM++ 2阶)
  - full_heun: solver_type='heun' (全维度 Heun 2阶, 额外 NFE)
  - wh_euler: solver_type='dpm_solver_pp', dim_d1_mask=[1,1,0,0] (cx/cy DPM++, w/h Euler 1阶)

输出: work_dirs/diagnosis/baseline_heun_3seed_norenewal.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SEEDS = {
    42: os.path.join(
        _PROJECT_ROOT,
        'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'),
    123: os.path.join(
        _PROJECT_ROOT,
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth'),
    789: os.path.join(
        _PROJECT_ROOT,
        'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth'),
}

CONFIG = os.path.join(
    _PROJECT_ROOT,
    'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py')
ANN = os.path.join(
    _PROJECT_ROOT, 'data/24_chromosomes_object/coco/valid/_annotations.coco.json')


def run_one(checkpoint, device, solver_type, dim_d1_mask):
    """评估单配置, 返回 mAP + per-dim L1 + 延迟."""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS
    from experiments.analysis.per_dim_d1_clean_repro import (
        per_dim_error_for_image, _to_plain_tensor)

    cfg = Config.fromfile(CONFIG)
    cfg.model.bbox_head.solver_type = solver_type
    cfg.model.bbox_head.box_renewal = False
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='vis')

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    if dim_d1_mask is not None:
        model.bbox_head._sampler.dim_d1_mask = torch.tensor(
            dim_d1_mask, dtype=torch.float32)
    else:
        model.bbox_head._sampler.dim_d1_mask = None

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    evaluator = METRICS.build(dict(
        type='CocoMetric', ann_file=ANN, metric='bbox',
        classwise=False, format_only=False))
    evaluator.dataset_meta = dataset.metainfo

    per_dim_accum = {'cx': [], 'cy': [], 'w': [], 'h': []}
    latencies = []
    n_imgs = len(dataset)

    with torch.no_grad():
        for i in range(n_imgs):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            if 'cuda' in device:
                torch.cuda.synchronize()
            import time
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

                pred_xyxy = _to_plain_tensor(pi.bboxes) if hasattr(r, 'pred_instances') else torch.zeros(0, 4)
                gt_xyxy = torch.zeros(0, 4)
                img_shape = getattr(r, 'img_shape', None) or getattr(r, 'ori_shape', (1, 1))
                ds_in = data['data_samples'][0] if data['data_samples'] else None
                if ds_in is not None and hasattr(ds_in, 'gt_instances') and ds_in.gt_instances is not None:
                    gt_xyxy = _to_plain_tensor(ds_in.gt_instances.bboxes)
                    img_shape = getattr(ds_in, 'img_shape', img_shape)
                if pred_xyxy.shape[0] > 0 and gt_xyxy.shape[0] > 0:
                    errs = per_dim_error_for_image(pred_xyxy, gt_xyxy, img_shape)
                    if errs is not None:
                        for k in per_dim_accum:
                            per_dim_accum[k].extend(errs[k])

            evaluator.process({}, eval_samples)
            if (i + 1) % 200 == 0:
                print(f'    进度 {i+1}/{n_imgs}')

    metrics = evaluator.evaluate(n_imgs)
    avg_lat = float(np.mean(latencies) * 1000)
    per_dim_mean = {}
    for k in ('cx', 'cy', 'w', 'h'):
        vals = per_dim_accum[k]
        per_dim_mean[k] = float(np.mean(vals)) if vals else float('nan')
    per_dim_mean['n_matched'] = sum(len(v) for v in per_dim_accum.values()) // 4

    return {
        'mAP': float(metrics.get('coco/bbox_mAP', 0)),
        'AP50': float(metrics.get('coco/bbox_mAP_50', 0)),
        'AP75': float(metrics.get('coco/bbox_mAP_75', 0)),
        'APs': float(metrics.get('coco/bbox_mAP_s', 0)),
        'avg_latency_ms': avg_lat,
        'fps': float(1000.0 / avg_lat) if avg_lat > 0 else 0,
        'per_dim_l1': per_dim_mean,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--configs', nargs='+', default=['baseline', 'full_heun', 'wh_euler'],
                        choices=['baseline', 'full_heun', 'wh_euler'])
    args = parser.parse_args()
    device = f'cuda:{args.gpu}'

    CONFIGS = {
        'baseline': dict(solver_type='dpm_solver_pp', dim_d1_mask=None,
                         label='[1,1,1,1] baseline (全 DPM++ 2阶)'),
        'full_heun': dict(solver_type='heun', dim_d1_mask=None,
                          label='full Heun (全维度 Heun 2阶, 额外 NFE)'),
        'wh_euler': dict(solver_type='dpm_solver_pp', dim_d1_mask=[1.0, 1.0, 0.0, 0.0],
                         label='[1,1,0,0] (cx/cy DPM++, w/h Euler 1阶)'),
    }

    print('=' * 70)
    print('Baseline + Full-Heun 3-seed 评估 (box_renewal OFF)')
    print('=' * 70)
    print(f'Configs: {args.configs}')
    print(f'Seeds: {sorted(SEEDS.keys())}')

    all_results = {}
    for cfg_name in args.configs:
        cfg_info = CONFIGS[cfg_name]
        print(f'\n{"="*70}')
        print(f'配置: {cfg_info["label"]}')
        print(f'{"="*70}')
        all_results[cfg_name] = {'label': cfg_info['label'], 'seeds': {}}
        for seed, ckpt in sorted(SEEDS.items()):
            if not os.path.exists(ckpt):
                print(f'  [跳过] seed={seed} checkpoint 不存在: {ckpt}')
                continue
            print(f'\n--- seed={seed} ---')
            print(f'  checkpoint: {ckpt}')
            r = run_one(ckpt, device, cfg_info['solver_type'], cfg_info['dim_d1_mask'])
            all_results[cfg_name]['seeds'][seed] = r
            print(f'  mAP={r["mAP"]:.4f}  AP50={r["AP50"]:.4f}  AP75={r["AP75"]:.4f}  '
                  f'lat={r["avg_latency_ms"]:.1f}ms  n_matched={r["per_dim_l1"]["n_matched"]}')

    # 汇总
    print('\n' + '=' * 70)
    print('3-seed 汇总')
    print('=' * 70)
    print(f'{"配置":<35} {"mAP mean±std":>16} {"lat(ms)":>10}')
    print('-' * 65)
    summary = {}
    for cfg_name, cfg_data in all_results.items():
        maps = [s['mAP'] for s in cfg_data['seeds'].values()]
        lats = [s['avg_latency_ms'] for s in cfg_data['seeds'].values()]
        if len(maps) >= 2:
            mean = float(np.mean(maps))
            std = float(np.std(maps, ddof=1))
            mean_lat = float(np.mean(lats))
            summary[cfg_name] = {'mAP_mean': mean, 'mAP_std': std, 'lat_mean': mean_lat}
            print(f'{cfg_data["label"]:<35} {mean:.4f}±{std:.4f}   {mean_lat:>8.1f}')
        else:
            print(f'{cfg_data["label"]:<35} (insufficient seeds)')

    # 与 hybrid 对比
    print('\n--- 与 hybrid 3-seed 对比 (box_renewal OFF) ---')
    hybrid_summary = {'mAP_mean': 0.858, 'mAP_std': 0.0035, 'lat_mean': 145.3}
    print(f'{"hybrid (cx/cy DPM++, w/h Heun)":<35} {hybrid_summary["mAP_mean"]:.4f}±{hybrid_summary["mAP_std"]:.4f}   {hybrid_summary["lat_mean"]:>8.1f}')
    for cfg_name, s in summary.items():
        delta = s['mAP_mean'] - hybrid_summary['mAP_mean']
        print(f'{all_results[cfg_name]["label"]:<35} {s["mAP_mean"]:.4f}±{s["mAP_std"]:.4f}   '
              f'{s["lat_mean"]:>8.1f}  Δ_hybrid={delta:+.4f}')

    print('\n判定 (全 Heun 对照):')
    if 'full_heun' in summary and 'baseline' in summary:
        fh = summary['full_heun']['mAP_mean']
        bl = summary['baseline']['mAP_mean']
        hy = hybrid_summary['mAP_mean']
        if fh < bl - 0.003:
            print(f'  全 Heun 也掉点 ({fh:.4f} vs baseline {bl:.4f}) → Heun 本身有害, 非混合策略问题')
        else:
            print(f'  全 Heun 不掉点 ({fh:.4f} vs baseline {bl:.4f}) → Heun 本身无害, '
                  f'hybrid 掉点 ({hy:.4f}) 是混合策略问题')

    # 保存
    output_path = os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        'baseline_heun_3seed_norenewal.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        'config': CONFIG, 'ann_file': ANN, 'box_renewal': False,
        'timestamp': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'baseline + full Heun 3-seed, box_renewal OFF, 与 hybrid 3-seed 对齐',
        'seeds_available': {s: os.path.exists(p) for s, p in SEEDS.items()},
        'results': all_results,
        'summary': summary,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {output_path}')


if __name__ == '__main__':
    main()
