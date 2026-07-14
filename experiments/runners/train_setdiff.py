"""SetDiff 训练入口 — work around CUDA Error 304.

train.py 在 Runner.from_cfg 的 collect_env 中触发 CUDA Error 304
(cudaGetDeviceCount 返回 OS call failed)。通过在导入 mmengine 之前
提前初始化 CUDA 上下文 (torch.zeros(1, device='cuda')) 来规避此问题。

Usage:
    python experiments/runners/train_setdiff.py experiments/configs/setdiff/setdiff_24obj.py --work-dir work_dirs/setdiff_24obj --seed 42
"""

import os
import sys

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    import argparse
    import glob
    import os.path as osp

    parser = argparse.ArgumentParser(description='SetDiff Training (CUDA init workaround)')
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

    # === CUDA Error 304 workaround ===
    # 在导入 mmengine/mmdet 之前, 提前初始化 CUDA 上下文.
    # train.py 中 collect_env 调用 cudaGetDeviceCount 时偶尔返回 Error 304
    # (OS call failed). 提前通过 torch.zeros(1, device='cuda') 建立CUDA
    # 上下文后, 后续的 cudaGetDeviceCount 将返回正确结果.
    import torch
    try:
        _cuda_init = torch.zeros(1, device='cuda')
        print(f'[CUDA workaround] CUDA context initialized: {_cuda_init.device}', flush=True)
    except Exception as e:
        print(f'[CUDA workaround] Failed to init CUDA: {e}', flush=True)
        print('Falling back to CPU (training will be very slow)...', flush=True)

    # 使用 mmengine 的 runner
    from mmengine.runner import Runner
    from mmengine.config import Config

    cfg = Config.fromfile(args.config)

    # Override work_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif 'work_dir' not in cfg:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        seed_suffix = f'_seed{args.seed}' if args.seed else ''
        cfg.work_dir = os.path.join('work_dirs', f'{config_name}{seed_suffix}')

    if args.resume:
        cfg.resume = True

    # 设置 SwanLab 实验名 = config名 + seed
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    seed_suffix = f'_seed{args.seed}' if args.seed else ''
    exp_name = f'{config_name}{seed_suffix}'

    # SwanLab name injection
    from experiments.runners.train import _set_swanlab_name, _patch_swanlab_save_id, _get_swanlab_run_id, _inject_swanlab_resume
    _set_swanlab_name(cfg, exp_name)

    # 续训时接续到已有 SwanLab 实验
    if args.resume:
        swanlab_id = _get_swanlab_run_id(cfg.work_dir)
        if swanlab_id:
            _inject_swanlab_resume(cfg, swanlab_id)
            print(f'[swanlab] Resuming SwanLab experiment {swanlab_id}')
        else:
            print(f'[swanlab] No SwanLab run ID found in {cfg.work_dir}; a new experiment will be created.')

    _patch_swanlab_save_id(cfg.work_dir)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
