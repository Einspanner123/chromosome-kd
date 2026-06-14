"""Checkpoint 推理测试

Usage:
    python experiments/runners/test.py experiments/configs/ldmdet/sinkhorn_stochastic.py \\
        --checkpoint work_dirs/xxx/best_coco_bbox_mAP_epoch_59.pth \\
        --dataset val

输出: mAP, AP50, AP75, per-class AP
"""

import argparse
import os
import sys

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='LDMDet Test (Inference)')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--dataset', default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--sampling-steps', type=int, default=None, help='Override sampling steps')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.dirname(args.checkpoint) or 'work_dirs/test'

    if args.sampling_steps is not None:
        cfg.model.bbox_head.sampling_timesteps = args.sampling_steps

    # 切换为测试模式
    if args.dataset == 'test':
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator

    # 注入 async checkpoint hook
    try:
        from projects.LDMDet.async_checkpoint_hook import AsyncCheckpointHook
        from mmdet.registry import HOOKS
        HOOKS.register_module(module=AsyncCheckpointHook, name='AsyncCheckpointHook', force=True)
    except (ImportError, ModuleNotFoundError):
        pass

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)

    # 运行测试
    metrics = runner.test()
    print(f'\n{"="*60}')
    print(f'Test Results ({args.dataset})')
    print(f'{"="*60}')
    for k, v in sorted(metrics.items()):
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')


if __name__ == '__main__':
    main()
