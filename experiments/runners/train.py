"""LDMDet 训练入口

Usage:
    python experiments/runners/train.py experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py
    python experiments/runners/train.py experiments/configs/ldmdet/sinkhorn_stochastic.py --work-dir work_dirs/my_exp --seed 42
    python experiments/runners/train.py experiments/configs/ldmdet/directions/nonlinear_trajectory/nonlinear_trajectory_e43_eps3.py --work-dir work_dirs/nonlinear_trajectory_e43_eps3 --resume --gpu-id 1
"""

import argparse
import glob
import os
import os.path as osp
import sys

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _iter_swanlab_backends(cfg):
    """遍历配置中的 SwanlabVisBackend。

    兼容 mmengine Config 的 _cfg_dict 结构（含 'value' key）。
    """
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


def _set_swanlab_name(cfg, exp_name: str):
    """向 SwanlabVisBackend 注入 experiment_name（若配置未显式指定）。"""
    for backend in _iter_swanlab_backends(cfg):
        init_kwargs = backend.setdefault('init_kwargs', {})
        if 'experiment_name' not in init_kwargs:
            init_kwargs['experiment_name'] = exp_name


def _get_swanlab_run_id(work_dir):
    """从 work_dir 恢复 SwanLab run ID 用于续训。

    优先读取 .swanlab_id 文件；若不存在，则从 vis_data/run-* 目录名解析。
    支持 work_dir/vis_data/ 和 work_dir/<timestamp>/vis_data/ 两种结构。
    """
    # 1. 直接读取保存的 run ID 文件
    id_file = osp.join(work_dir, '.swanlab_id')
    if osp.exists(id_file):
        with open(id_file, 'r') as f:
            run_id = f.read().strip()
            if run_id:
                return run_id

    # 2. 从 vis_data/run-<timestamp>-<run_id> 目录名解析
    run_dirs = glob.glob(osp.join(work_dir, '**', 'vis_data', 'run-*'), recursive=True)
    run_dirs = [d for d in run_dirs if osp.isdir(d)]
    if not run_dirs:
        return None
    # 取最新的 run 目录（按修改时间）
    run_dirs.sort(key=lambda p: osp.getmtime(p), reverse=True)
    latest = osp.basename(run_dirs[0])
    # 格式: run-<timestamp>-<run_id>
    parts = latest.split('-')
    if len(parts) >= 3:
        return parts[-1]
    return None


def _inject_swanlab_resume(cfg, run_id):
    """注入 id 和 resume='allow' 到 SwanlabVisBackend init_kwargs,
    使 swanlab.init() 续接到已有实验。

    遍历所有 SwanlabVisBackend（vis_backends 和 visualizer.vis_backends
    可能是独立副本），确保全部注入续训参数。
    """
    injected = False
    for backend in _iter_swanlab_backends(cfg):
        init_kwargs = backend.setdefault('init_kwargs', {})
        init_kwargs['id'] = run_id
        init_kwargs['resume'] = 'allow'
        injected = True
    return injected


def _patch_swanlab_save_id(work_dir):
    """Patch SwanlabVisBackend._init_env 以在初始化后保存 run ID,
    便于未来 --resume 调用续接到同一实验。"""
    try:
        from swanlab.integration.mmengine import SwanlabVisBackend
    except ImportError:
        return

    _orig_init_env = SwanlabVisBackend._init_env

    def _patched_init_env(self):
        _orig_init_env(self)
        run_id = None
        try:
            # SwanLab 0.8.x: 通过 swanlab.get_run() 获取活跃 run
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
    parser = argparse.ArgumentParser(description='LDMDet Training')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--work-dir', default=None, help='Work directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint and continue SwanLab logging')
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

    # 设置 SwanLab 实验名 = config名 + seed（配置文件未显式指定时）
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    seed_suffix = f'_seed{args.seed}' if args.seed else ''
    exp_name = f'{config_name}{seed_suffix}'
    _set_swanlab_name(cfg, exp_name)

    # 续训时接续到已有 SwanLab 实验
    if args.resume:
        swanlab_id = _get_swanlab_run_id(cfg.work_dir)
        if swanlab_id:
            _inject_swanlab_resume(cfg, swanlab_id)
            print(f'[swanlab] Resuming SwanLab experiment {swanlab_id}')
        else:
            print(f'[swanlab] No SwanLab run ID found in {cfg.work_dir}; a new experiment will be created.')

    # Patch SwanLab 以便每次运行都保存 run ID 供未来 resume
    _patch_swanlab_save_id(cfg.work_dir)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
