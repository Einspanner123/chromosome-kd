"""Checkpoint 推理测试

Usage:
    python experiments/runners/test.py experiments/configs/ldmdet/sinkhorn_stochastic.py \\
        --checkpoint work_dirs/xxx/best_coco_bbox_mAP_epoch_59.pth \\
        --dataset val

采样器/步数覆盖:
    python experiments/runners/test.py experiments/configs/ldmdet/rf_heun_adaln.py \\
        --checkpoint work_dirs/xxx/best.pth --dataset test \\
        --sampling-steps 3 --solver-type euler --seed 42

DDPM baseline:
    python experiments/runners/test.py experiments/configs/baselines/diffusiondet_ddpm.py \\
        --checkpoint work_dirs/multi_seed_aug/ddpm/seed_42/best_coco_bbox_mAP_epoch_79.pth \\
        --dataset test --sampling-steps 4 --seed 42

SwanLab: 推理测试独立到 'ldmdet-inference' 项目, 实验名默认 '{solver}_{steps}step'
         (可用 --exp-name 覆盖)

输出: mAP, AP50, AP75, per-class AP
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner

# 推理测试独立 SwanLab 项目名 (与训练 'ldmdet-ablation' 分离)
_INFERENCE_SWANLAB_PROJECT = 'ldmdet-inference'


def set_seed(seed: int):
    """固定随机种子, 确保推理可复现 (初始噪声 + box_renewal 均受 seed 控制)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='LDMDet Test (Inference)')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--dataset', default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible inference')
    parser.add_argument('--sampling-steps', type=int, default=None, help='Override sampling steps')
    parser.add_argument('--solver-type', type=str, default=None,
                        choices=['euler', 'heun', 'ddim', 'dpm_solver_pp', 'dpm_solver_pp_3'],
                        help='Override solver type')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='SwanLab experiment name (default: auto {solver}_{steps}step)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # 固定随机种子 (推理随机性来源: 初始噪声 proposals + box_renewal)
    set_seed(args.seed)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.dirname(args.checkpoint) or 'work_dirs/test'

    if args.sampling_steps is not None:
        cfg.model.bbox_head.sampling_timesteps = args.sampling_steps

    if args.solver_type is not None:
        cfg.model.bbox_head.solver_type = args.solver_type

    # 生成 SwanLab 实验名 (推理测试独立项目)
    if args.exp_name is None:
        solver = args.solver_type or cfg.model.bbox_head.get('solver_type', 'unknown')
        steps = (args.sampling_steps if args.sampling_steps is not None
                 else cfg.model.bbox_head.get('sampling_timesteps', 0))
        args.exp_name = f'{solver}_{steps}step'

    # 覆盖 SwanLab 配置: 推理独立项目 + 明确实验名
    # 注意: cfg.vis_backends 与 cfg.visualizer['vis_backends'] 可能是不同对象, 需同时修改
    _vis_backends_list = cfg.get('vis_backends', [])
    _visualizer_backends = cfg.get('visualizer', {}).get('vis_backends', [])
    for _vis_backends in [_vis_backends_list, _visualizer_backends]:
        for backend in _vis_backends:
            if isinstance(backend, dict) and backend.get('type') == 'SwanlabVisBackend':
                backend.setdefault('init_kwargs', {})
                backend['init_kwargs']['project'] = _INFERENCE_SWANLAB_PROJECT
                backend['init_kwargs']['experiment_name'] = args.exp_name

    print(f'[SwanLab] project={_INFERENCE_SWANLAB_PROJECT}, exp_name={args.exp_name}, seed={args.seed}')

    # 切换为测试模式
    if args.dataset == 'test':
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)

    # 运行测试
    metrics = runner.test()
    print(f'\n{"="*60}')
    print(f'Test Results ({args.dataset}) [{args.exp_name}, seed={args.seed}]')
    print(f'{"="*60}')
    for k, v in sorted(metrics.items()):
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
