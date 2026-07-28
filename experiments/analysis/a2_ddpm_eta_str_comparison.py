#!/usr/bin/env python3
"""A.2 + DDPM η_str 对比实验 (零成本推理)

实验 1 (A.2): w,h 维度均用 1 阶, 仅 cx/cy 用 2 阶
  - 基于 Phase 2 per-dim η_str 诊断: w (0.5-1.0) 与 h (0.4-0.9) 接近
  - 预期: mAP 持平, 延迟略降

实验 2 (DDPM vs RF η_str): 在 DDPM checkpoint 上以 RF 路径运行 DPM-Solver++
  - 目的: 对比 DDPM-trained vs RF-trained 模型的速度场非恒常性
  - 方法: 覆盖 diffusion_type='rectified_flow', solver_type='dpm_solver_pp'
  - 注意: DDPM 模型在 RF 框架下评估, η_str 反映训练范式对速度场线性的影响

Usage:
  python experiments/analysis/a2_ddpm_eta_str_comparison.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run_eval(config_path, checkpoint, ann_file, solver_type, device='cuda:0',
             diffusion_type_override=None, sampling_steps=None,
             collect_eta_str=False, num_images=None):
    """在指定配置下评估模型 mAP + 延迟 + η_str"""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS, METRICS

    cfg = Config.fromfile(config_path)

    # 覆盖参数
    cfg.model.bbox_head.solver_type = solver_type
    if sampling_steps is not None:
        cfg.model.bbox_head.sampling_timesteps = sampling_steps
    if diffusion_type_override is not None:
        cfg.model.bbox_head.diffusion_type = diffusion_type_override

    # 禁用 SwanLab
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=cfg.vis_backends,
        name='visualizer',
    )

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    evaluator = METRICS.build(dict(
        type='CocoMetric',
        ann_file=ann_file,
        metric='bbox',
        classwise=True,
        format_only=False,
    ))
    evaluator.dataset_meta = dataset.metainfo

    latencies = []
    num_samples = 0
    eta_str_all = []
    eta_str_per_dim_all = []
    max_images = num_images or len(dataset)

    with torch.no_grad():
        for i in range(min(max_images, len(dataset))):
            data = dataset[i]
            data['inputs'] = data['inputs'].unsqueeze(0).to(device)
            if not isinstance(data['data_samples'], list):
                data['data_samples'] = [data['data_samples']]

            torch.cuda.synchronize()
            t0 = time.time()
            out = model.test_step(data)
            torch.cuda.synchronize()
            latencies.append(time.time() - t0)

            # 收集 η_str
            if collect_eta_str:
                bbox_head = model.bbox_head
                if hasattr(bbox_head, '_last_eta_str_log'):
                    eta_str_all.append(list(bbox_head._last_eta_str_log))
                if hasattr(bbox_head, '_last_eta_str_per_dim_log'):
                    eta_str_per_dim_all.append([
                        list(x) for x in bbox_head._last_eta_str_per_dim_log
                    ])

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
                print(f'  推理进度: {i+1}/{max_images}')

    metrics = evaluator.evaluate(num_samples)
    avg_latency = np.mean(latencies) * 1000

    # 聚合 η_str
    eta_str_stats = {}
    if eta_str_all:
        all_vals = [v for r in eta_str_all for v in r if v is not None]
        if all_vals:
            arr = np.array(all_vals)
            eta_str_stats['overall'] = {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'median': float(np.median(arr)),
                'p95': float(np.percentile(arr, 95)),
            }
    if eta_str_per_dim_all:
        dim_names = ['cx', 'cy', 'w', 'h']
        per_dim = {dim: [] for dim in dim_names}
        for record in eta_str_per_dim_all:
            for step_vals in record:
                for j, dim_name in enumerate(dim_names):
                    if j < len(step_vals) and step_vals[j] is not None:
                        per_dim[dim_name].append(step_vals[j])
        eta_str_stats['per_dim'] = {}
        for dim_name, vals in per_dim.items():
            if vals:
                arr = np.array(vals)
                eta_str_stats['per_dim'][dim_name] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'median': float(np.median(arr)),
                }

    return {
        'mAP': float(metrics.get('coco/bbox_mAP', 0)),
        'AP50': float(metrics.get('coco/bbox_mAP_50', 0)),
        'AP75': float(metrics.get('coco/bbox_mAP_75', 0)),
        'APs': float(metrics.get('coco/bbox_mAP_s', 0)),
        'APm': float(metrics.get('coco/bbox_mAP_m', 0)),
        'APl': float(metrics.get('coco/bbox_mAP_l', 0)),
        'avg_latency_ms': float(avg_latency),
        'fps': float(1000.0 / avg_latency),
        'eta_str_stats': eta_str_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='A.2 + DDPM η_str 对比实验')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-images', type=int, default=None,
                        help='限制评估图像数 (None=全部 500)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    num_images = args.num_images

    # 通用路径
    ann_file = os.path.join(
        _PROJECT_ROOT,
        'data/24_chromosomes_object/coco/valid/_annotations.coco.json'
    )
    output_dir = os.path.join(_PROJECT_ROOT, 'work_dirs', 'diagnosis')
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # ================================================================
    # 实验 1 (A.2): w,h=1阶, cx/cy=2阶
    # ================================================================
    print('\n' + '=' * 80)
    print('实验 1 (A.2): per-dim-w solver (w,h=1阶, cx/cy=2阶)')
    print('=' * 80)

    a4_config = os.path.join(
        _PROJECT_ROOT,
        'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py'
    )
    a4_ckpt = os.path.join(
        _PROJECT_ROOT,
        'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'
    )

    if os.path.exists(a4_ckpt):
        # Baseline: 全 2 阶
        print('\n--- A.2 Baseline: DPM-Solver++ 全 2 阶 ---')
        r_base = run_eval(a4_config, a4_ckpt, ann_file,
                          solver_type='dpm_solver_pp', device=device,
                          collect_eta_str=True, num_images=num_images)
        print(f'  mAP={r_base["mAP"]:.4f}  AP50={r_base["AP50"]:.4f}  '
              f'AP75={r_base["AP75"]:.4f}  延迟={r_base["avg_latency_ms"]:.1f}ms')
        if r_base['eta_str_stats'].get('per_dim'):
            print('  per-dim η_str:', {k: f'{v["mean"]:.3f}' for k, v in r_base['eta_str_stats']['per_dim'].items()})

        # A.2: w,h=1阶
        print('\n--- A.2: per-dim-w (w,h=1阶, cx/cy=2阶) ---')
        r_a2 = run_eval(a4_config, a4_ckpt, ann_file,
                        solver_type='dpm_solver_pp_per_dim_w', device=device,
                        collect_eta_str=True, num_images=num_images)
        print(f'  mAP={r_a2["mAP"]:.4f}  AP50={r_a2["AP50"]:.4f}  '
              f'AP75={r_a2["AP75"]:.4f}  延迟={r_a2["avg_latency_ms"]:.1f}ms')
        if r_a2['eta_str_stats'].get('per_dim'):
            print('  per-dim η_str:', {k: f'{v["mean"]:.3f}' for k, v in r_a2['eta_str_stats']['per_dim'].items()})

        print(f'\n  ΔmAP = {r_a2["mAP"] - r_base["mAP"]:+.4f}  '
              f'Δ延迟 = {r_a2["avg_latency_ms"] - r_base["avg_latency_ms"]:+.1f}ms')

        results['a2'] = {
            'baseline_2nd': r_base,
            'per_dim_w': r_a2,
            'delta_mAP': r_a2['mAP'] - r_base['mAP'],
            'delta_latency': r_a2['avg_latency_ms'] - r_base['avg_latency_ms'],
        }
    else:
        print(f'  ⚠ Checkpoint 不存在: {a4_ckpt}')
        results['a2'] = {'error': f'checkpoint not found: {a4_ckpt}'}

    # ================================================================
    # 实验 2 (DDPM vs RF η_str): DDPM checkpoint 以 RF 路径运行 DPM-Solver++
    # ================================================================
    print('\n' + '=' * 80)
    print('实验 2: DDPM vs RF η_str 对比')
    print('=' * 80)

    ddpm_config = os.path.join(
        _PROJECT_ROOT,
        'experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py'
    )
    ddpm_ckpt = os.path.join(
        _PROJECT_ROOT,
        'work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth'
    )

    if os.path.exists(ddpm_ckpt):
        # DDPM checkpoint 以 RF 路径运行 DPM-Solver++ 4 步
        print('\n--- DDPM checkpoint (RF 路径, DPM-Solver++ 4 步) ---')
        try:
            r_ddpm = run_eval(ddpm_config, ddpm_ckpt, ann_file,
                              solver_type='dpm_solver_pp', device=device,
                              diffusion_type_override='rectified_flow',
                              sampling_steps=4,
                              collect_eta_str=True, num_images=num_images)
            print(f'  mAP={r_ddpm["mAP"]:.4f}  AP50={r_ddpm["AP50"]:.4f}  '
                  f'AP75={r_ddpm["AP75"]:.4f}')
            if r_ddpm['eta_str_stats'].get('overall'):
                s = r_ddpm['eta_str_stats']['overall']
                print(f'  η_str overall: mean={s["mean"]:.4f}  '
                      f'median={s["median"]:.4f}  p95={s["p95"]:.4f}')
            if r_ddpm['eta_str_stats'].get('per_dim'):
                print('  per-dim η_str:', {k: f'{v["mean"]:.3f}' for k, v in r_ddpm['eta_str_stats']['per_dim'].items()})

            # 对比 RF checkpoint 的 η_str (从实验 1 baseline 获取)
            if 'a2' in results and isinstance(results['a2'].get('baseline_2nd'), dict):
                rf_eta = results['a2']['baseline_2nd']['eta_str_stats']
                print('\n--- RF vs DDPM η_str 对比 ---')
                if rf_eta.get('overall') and r_ddpm['eta_str_stats'].get('overall'):
                    print(f'  RF  η_str overall: mean={rf_eta["overall"]["mean"]:.4f}')
                    print(f'  DDPM η_str overall: mean={r_ddpm["eta_str_stats"]["overall"]["mean"]:.4f}')
                    ratio = r_ddpm['eta_str_stats']['overall']['mean'] / max(rf_eta['overall']['mean'], 1e-6)
                    print(f'  DDPM/RF 比值: {ratio:.2f}x')
                if rf_eta.get('per_dim') and r_ddpm['eta_str_stats'].get('per_dim'):
                    print(f'\n  {"Dim":<6} {"RF":>10} {"DDPM":>10} {"Ratio":>10}')
                    for dim in ['cx', 'cy', 'w', 'h']:
                        rf_v = rf_eta['per_dim'].get(dim, {}).get('mean', 0)
                        ddpm_v = r_ddpm['eta_str_stats']['per_dim'].get(dim, {}).get('mean', 0)
                        ratio = ddpm_v / max(rf_v, 1e-6)
                        print(f'  {dim:<6} {rf_v:>10.4f} {ddpm_v:>10.4f} {ratio:>10.2f}x')

            results['ddpm_eta_str'] = r_ddpm
        except Exception as e:
            print(f'  ⚠ DDPM η_str 测量失败: {e}')
            import traceback
            traceback.print_exc()
            results['ddpm_eta_str'] = {'error': str(e)}
    else:
        print(f'  ⚠ Checkpoint 不存在: {ddpm_ckpt}')
        results['ddpm_eta_str'] = {'error': f'checkpoint not found: {ddpm_ckpt}'}

    # ================================================================
    # 保存结果
    # ================================================================
    output_path = os.path.join(output_dir, 'a2_ddpm_eta_str_comparison.json')
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_images': num_images,
        'results': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
