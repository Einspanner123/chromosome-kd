"""诊断 CUDA 问题 v2 — 完全复制 train.py 的执行流程。

在 runner.train() 之前和内部关键点检查 CUDA。
"""

import argparse
import os
import os.path as osp
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 复制 train.py 的所有函数
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


def check_cuda(tag):
    import torch
    try:
        avail = torch.cuda.is_available()
        count = torch.cuda.device_count() if avail else 0
        print(f'[{tag}] cuda.is_available={avail}, device_count={count}', flush=True)
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
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    # === 完全复制 train.py 的执行流程 ===
    if args.seed is not None:
        os.environ['RANDOM_SEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    check_cuda('A. after env setup')

    from mmengine.runner import Runner
    from mmengine.config import Config
    check_cuda('B. after mmengine imports')

    cfg = Config.fromfile(args.config)
    check_cuda('C. after Config.fromfile')

    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif 'work_dir' not in cfg:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        seed_suffix = f'_seed{args.seed}' if args.seed else ''
        cfg.work_dir = os.path.join('work_dirs', f'{config_name}{seed_suffix}')

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    seed_suffix = f'_seed{args.seed}' if args.seed else ''
    exp_name = f'{config_name}{seed_suffix}'
    _set_swanlab_name(cfg, exp_name)
    check_cuda('D. after _set_swanlab_name')

    _patch_swanlab_save_id(cfg.work_dir)
    check_cuda('E. after _patch_swanlab_save_id')

    check_cuda('F. before Runner.from_cfg')
    runner = Runner.from_cfg(cfg)
    check_cuda('G. after Runner.from_cfg')

    # 不调用 runner.train()，只检查到这里
    print('\n=== Diagnosis complete. NOT calling runner.train() ===', flush=True)


if __name__ == '__main__':
    main()
