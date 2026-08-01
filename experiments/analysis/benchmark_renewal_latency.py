#!/usr/bin/env python3
"""测量 box_renewal ON vs OFF 的推理延迟 (3-seed, CUDA timing)

实验设计:
  - 模型: a4_dpm_pp_24obj (+DPM-Solver++ 4-step)
  - 3 seeds: 42, 123, 789
  - 对比: box_renewal=True (默认) vs box_renewal=False
  - 测量: warmup=50, iters=200, 10 张验证集图片循环
  - 硬件: NVIDIA RTX A6000 (ross)

输出: work_dirs/diagnosis/renewal_latency_3seed.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

def run_latency_test(config_path, checkpoint, device='cuda:0',
                     box_renewal=True, warmup=50, iters=200, n_imgs=10):
    """加载模型并测量推理延迟"""
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS

    cfg = Config.fromfile(config_path)
    cfg.model.bbox_head.box_renewal = box_renewal
    # 禁用 SwanLab
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer', vis_backends=cfg.vis_backends, name='visualizer',
    )

    model = init_detector(cfg, checkpoint, device=device)
    model.eval()

    # 验证 box_renewal 设置
    actual_renewal = model.bbox_head.box_renewal
    assert actual_renewal == box_renewal, \
        f"box_renewal 设置失败: 期望 {box_renewal}, 实际 {actual_renewal}"

    # 加载数据集
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    n_imgs = min(n_imgs, len(dataset))

    test_data = []
    for i in range(n_imgs):
        data = dataset[i]
        data['inputs'] = data['inputs'].unsqueeze(0).to(device)
        if not isinstance(data['data_samples'], list):
            data['data_samples'] = [data['data_samples']]
        test_data.append(data)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            data = test_data[_ % n_imgs]
            model.test_step(data)
        torch.cuda.synchronize()

    # 测量
    latencies = []
    with torch.no_grad():
        for i in range(iters):
            data = test_data[i % n_imgs]
            torch.cuda.synchronize()
            t0 = time.time()
            model.test_step(data)
            torch.cuda.synchronize()
            latencies.append((time.time() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        'total_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'fps': float(1000.0 / np.mean(latencies)),
        'warmup': warmup,
        'iters': iters,
        'n_imgs': n_imgs,
        'box_renewal': box_renewal,
    }


def main():
    parser = argparse.ArgumentParser(description='Renewal ON vs OFF 延迟测量 (3-seed)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--output', type=str,
                        default='work_dirs/diagnosis/renewal_latency_3seed.json')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    config_path = 'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py'

    # 3-seed checkpoints
    seeds = {
        42: 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth',
        123: 'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth',
        789: 'work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth',
    }

    # 验证 checkpoint 存在
    for seed, ckpt in seeds.items():
        if not os.path.exists(ckpt):
            print(f"[ERROR] seed={seed} checkpoint 不存在: {ckpt}")
            sys.exit(1)

    gpu_name = torch.cuda.get_device_name(args.gpu)
    print(f"=== Renewal ON vs OFF 延迟测量 (3-seed) ===")
    print(f"GPU: {gpu_name}")
    print(f"Config: {config_path}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")
    print(f"Device: {device}")
    print()

    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': gpu_name,
        'config': config_path,
        'warmup': args.warmup,
        'iters': args.iters,
        'seeds': {},
    }

    for seed, ckpt in seeds.items():
        print(f"--- Seed {seed} (checkpoint: {os.path.basename(ckpt)}) ---")

        # Renewal ON
        print(f"  [1/2] box_renewal=True ...", end=' ', flush=True)
        res_on = run_latency_test(
            config_path, ckpt, device=device,
            box_renewal=True, warmup=args.warmup, iters=args.iters,
        )
        print(f"avg={res_on['total_ms']:.2f}ms  std={res_on['std_ms']:.2f}ms  "
              f"p99={res_on['p99_ms']:.2f}ms  FPS={res_on['fps']:.1f}")

        # Renewal OFF
        print(f"  [2/2] box_renewal=False ...", end=' ', flush=True)
        res_off = run_latency_test(
            config_path, ckpt, device=device,
            box_renewal=False, warmup=args.warmup, iters=args.iters,
        )
        print(f"avg={res_off['total_ms']:.2f}ms  std={res_off['std_ms']:.2f}ms  "
              f"p99={res_off['p99_ms']:.2f}ms  FPS={res_off['fps']:.1f}")

        delta_ms = res_on['total_ms'] - res_off['total_ms']
        delta_pct = delta_ms / res_on['total_ms'] * 100
        print(f"  Δ (ON-OFF) = {delta_ms:+.2f}ms ({delta_pct:+.1f}%)  "
              f"FPS提升: {res_off['fps'] - res_on['fps']:+.2f}")
        print()

        results['seeds'][str(seed)] = {
            'checkpoint': ckpt,
            'renewal_on': res_on,
            'renewal_off': res_off,
            'delta_ms': delta_ms,
            'delta_pct': delta_pct,
            'fps_gain': res_off['fps'] - res_on['fps'],
        }

    # 汇总
    deltas = [results['seeds'][s]['delta_ms'] for s in ['42', '123', '789']]
    delta_pcts = [results['seeds'][s]['delta_pct'] for s in ['42', '123', '789']]
    fps_gains = [results['seeds'][s]['fps_gain'] for s in ['42', '123', '789']]

    results['summary'] = {
        'mean_delta_ms': float(np.mean(deltas)),
        'std_delta_ms': float(np.std(deltas)),
        'mean_delta_pct': float(np.mean(delta_pcts)),
        'mean_fps_gain': float(np.mean(fps_gains)),
    }

    print("=== 汇总 (3-seed) ===")
    print(f"Δ延迟 (ON-OFF): {np.mean(deltas):+.2f} ± {np.std(deltas):.2f} ms "
          f"({np.mean(delta_pcts):+.1f}%)")
    print(f"FPS 提升: {np.mean(fps_gains):+.2f}")
    print(f"结论: {'关闭 renewal 有加速' if np.mean(deltas) > 0.5 else '加速不显著 (<0.5ms)'}")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
