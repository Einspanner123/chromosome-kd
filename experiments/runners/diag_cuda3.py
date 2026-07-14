"""诊断 CUDA 问题 v3 — 不提前调用任何 CUDA 函数，直接 Runner.from_cfg。

关键假设: 诊断脚本 diag_cuda2.py 中的 check_cuda() 提前初始化了 CUDA 上下文,
导致后续 collect_env 返回 True。如果不提前调用 CUDA, collect_env 可能返回 False。
"""

import argparse
import os
import os.path as osp
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import glob


def _iter_swanlab_backends(cfg):
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
                yield backend


def _set_swanlab_name(cfg, exp_name):
    for backend in _iter_swanlab_backends(cfg):
        init_kwargs = backend.setdefault('init_kwargs', {})
        if 'experiment_name' not in init_kwargs:
            init_kwargs['experiment_name'] = exp_name


def _patch_swanlab_save_id(work_dir):
    try:
        from swanlab.integration.mmengine import SwanlabVisBackend
    except ImportError:
        return
    _orig_init_env = SwanlabVisBackend._init_env

    def _patched_init_env(self):
        _orig_init_env(self)
        run_id = None
        try:
            import swanlab
            run = swanlab.get_run()
            if run is not None:
                run_id = run.id
        except Exception:
            pass
        if run_id is not None:
            id_file = osp.join(work_dir, '.swanlab_id')
            os.makedirs(work_dir, exist_ok=True)
            with open(id_file, 'w') as f:
                f.write(run_id)

    SwanlabVisBackend._init_env = _patched_init_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--work-dir', default='/tmp/diag3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    # === 完全复制 train.py, 但不调用任何 CUDA 函数 ===
    if args.seed is not None:
        os.environ['RANDOM_SEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # 不 import torch, 不调用任何 CUDA
    from mmengine.runner import Runner
    from mmengine.config import Config

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    seed_suffix = f'_seed{args.seed}' if args.seed else ''
    exp_name = f'{config_name}{seed_suffix}'
    _set_swanlab_name(cfg, exp_name)
    _patch_swanlab_save_id(cfg.work_dir)

    print('About to call Runner.from_cfg (NO prior CUDA calls)...', flush=True)
    # Runner.from_cfg -> Runner.__init__ -> _log_env -> collect_env
    # collect_env will call torch.cuda.is_available() for the FIRST time
    runner = Runner.from_cfg(cfg)

    # Now check CUDA after collect_env
    import torch
    print(f'After Runner.from_cfg: cuda.is_available={torch.cuda.is_available()}', flush=True)


if __name__ == '__main__':
    main()
