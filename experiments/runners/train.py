"""LDMDet 训练入口

Usage:
    python experiments/runners/train.py experiments/configs/ldmdet/rf_heun_adaln.py
    python experiments/runners/train.py experiments/configs/ldmdet/sinkhorn_stochastic.py --work-dir work_dirs/my_exp --seed 42
"""

import argparse
import os
import sys

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _set_swanlab_name(cfg, exp_name: str):
    """向 SwanlabVisBackend 注入 experiment_name"""
    raw = cfg._cfg_dict
    for key in ('vis_backends', 'visualizer.vis_backends'):
        parts = key.split('.')
        d = raw
        for p in parts:
            d = d.get(p, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict) and 'value' in d:
            d = d['value']
        if not isinstance(d, list):
            continue
        for backend in d:
            if isinstance(backend, dict) and backend.get('type') == 'SwanlabVisBackend':
                backend.setdefault('init_kwargs', {})['experiment_name'] = exp_name


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

    # 设置 SwanLab 实验名 = config名 + seed
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    seed_suffix = f'_seed{args.seed}' if args.seed else ''
    exp_name = f'{config_name}{seed_suffix}'
    _set_swanlab_name(cfg, exp_name)

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
