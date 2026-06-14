"""LDMDet 训练入口

Usage:
    python experiments/runners/train.py experiments/configs/ldmdet/rf_heun_adaln.py
    python experiments/runners/train.py experiments/configs/ldmdet/sinkhorn_stochastic.py --work-dir work_dirs/my_exp --seed 42
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='LDMDet Training')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--work-dir', default=None, help='Work directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    # 设置环境
    if args.seed is not None:
        os.environ['RANDOM_SEED'] = str(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # 使用 mmengine 的 runner
    from mmengine.runner import Runner
    from mmengine.config import Config

    cfg = Config.fromfile(args.config)

    # Override work_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif 'work_dir' not in cfg:
        # 自动生成 work_dir 名称
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        seed_suffix = f'_seed{args.seed}' if args.seed else ''
        cfg.work_dir = os.path.join('work_dirs', f'{config_name}{seed_suffix}')

    if args.resume:
        cfg.resume = True

    # 注入 async checkpoint hook
    try:
        from projects.LDMDet.async_checkpoint_hook import AsyncCheckpointHook
        from mmdet.registry import HOOKS
        HOOKS.register_module(module=AsyncCheckpointHook, name='AsyncCheckpointHook', force=True)
    except (ImportError, ModuleNotFoundError):
        pass

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
