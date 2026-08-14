"""诊断 CUDA 初始化问题 — 模拟 train.py 的执行流程。

Usage:
    python experiments/runners/diag_cuda.py experiments/configs/setdiff/setdiff_24obj.py
"""

import argparse
import os
import sys

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def check_cuda(tag):
    import torch

    try:
        avail = torch.cuda.is_available()
        count = torch.cuda.device_count() if avail else 0
        print(
            f'[{tag}] cuda.is_available={avail}, device_count={count}',
            flush=True,
        )
        if avail:
            x = torch.zeros(1, device='cuda')
            print(f'[{tag}] cuda tensor ok: {x.device}', flush=True)
        return avail
    except Exception as e:
        print(f'[{tag}] CUDA check FAILED: {e}', flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    check_cuda('0. after basic imports')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    check_cuda('1. after CUDA_VISIBLE_DEVICES set')

    from mmengine.config import Config

    check_cuda('2. after mmengine Config import')

    cfg = Config.fromfile(args.config)
    check_cuda('3. after Config.fromfile (custom_imports executed)')

    from mmengine.runner import Runner

    check_cuda('4. after Runner import')

    # Try building the model only
    from mmdet.registry import MODELS

    model_cfg = cfg.model.copy()
    check_cuda('5. before MODELS.build')
    try:
        model = MODELS.build(model_cfg)
        check_cuda('6. after MODELS.build (model constructed)')
    except Exception as e:
        print(f'[6. MODELS.build FAILED] {e}', flush=True)

    # Try Runner.from_cfg
    check_cuda('7. before Runner.from_cfg')
    try:
        if 'work_dir' not in cfg:
            cfg.work_dir = '/tmp/diag_workdir'
        os.makedirs(cfg.work_dir, exist_ok=True)
        runner = Runner.from_cfg(cfg)
        check_cuda('8. after Runner.from_cfg')
    except Exception as e:
        print(f'[8. Runner.from_cfg FAILED] {e}', flush=True)
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
