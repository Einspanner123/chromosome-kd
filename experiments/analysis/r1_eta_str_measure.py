"""R1 实验: 测量 DPM-Solver++ 直线度指标 eta_str

理论 (docs/paper/theory_analysis_RF_DPM.md §1):
  eta_str = ||D1|| / ||x0||, 其中 D1 = (x0_n - x0_p) / (t_n - t_p)
  理想 RF (直线 ODE) 下 D1=0, eta_str=0
  eta_str 大表示学习轨迹非直线, DPM-Solver++ 二阶校正生效

实验目的:
  1. 验证 +Stoch. Coupling (DPM-Solver++) 在 4 步推理下的 eta_str 分布
  2. 验证 "2 步收敛" 的定量解释: 是否 eta_str^(2) 已接近 0
  3. 比较 RF+Heun (Heun) 和 +Stoch. Coupling (DPM-Solver++) 的 eta_str (前者不通过 DPM 框架)
  4. 比较 box_renewal on/off 下的 eta_str (验证 D3 矛盾: renewal 是否污染 eta_str)

Usage:
  python experiments/analysis/r1_eta_str_measure.py \\
      --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \\
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
      --seed 42 --gpu-id 0 \\
      --output experiments/analysis/r1_eta_str_a3_seed42.json

  多 seed 对比:
  for s in 42 123 789; do
    python experiments/analysis/r1_eta_str_measure.py ... --seed $s \\
      --output experiments/analysis/r1_eta_str_a3_seed${s}.json
  done
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.runner import Runner


class EtaStrCollectorHook(Hook):
    """收集每次 val_iter 后 model.bbox_head._last_eta_str_log

    eta_str_history 在 RFDPMSolverMultistep.step() 中记录:
      - 4 步推理下, step 0 是 linear (无 D1), step 1/2/3 记录
      - 故 eta_str_history 长度 = 3 (对应 step 1, 2, 3)
      - box_renewal 会重置 dpm_solver 时可能清空 eta_str_history
    """

    def __init__(self):
        super().__init__()
        self.eta_str_records = []  # list[list[float]], 每张图一条
        self.per_step_stats = {}  # 聚合后统计

    def _collect_once(self, runner, batch_idx):
        """单次 iter 收集逻辑 (after_val_iter / after_test_iter 共用)"""
        model = runner.model
        # 诊断: 首次 iter 打印 model 类型
        if batch_idx == 0:
            print(f'[EtaStrHook] model type: {type(model).__name__}', flush=True)
            head = self._get_head(model)
            if head is not None:
                print(f'[EtaStrHook] head type: {type(head).__name__}', flush=True)
                print(f'[EtaStrHook] head has _last_eta_str_log: '
                      f'{hasattr(head, "_last_eta_str_log")}', flush=True)
                if hasattr(head, '_last_eta_str_log'):
                    print(f'[EtaStrHook] first record: {head._last_eta_str_log}', flush=True)
                private_attrs = [a for a in dir(head) if a.startswith('_last')]
                print(f'[EtaStrHook] head _last* attrs: {private_attrs}', flush=True)
            else:
                print(f'[EtaStrHook] WARNING: cannot access bbox_head', flush=True)
                print(f'[EtaStrHook] model attrs: {[a for a in dir(model) if not a.startswith("_")][:20]}', flush=True)
                if hasattr(model, 'module'):
                    inner = model.module
                    print(f'[EtaStrHook] model.module type: {type(inner).__name__}', flush=True)
                    print(f'[EtaStrHook] model.module attrs: {[a for a in dir(inner) if not a.startswith("_")][:20]}', flush=True)

        head = self._get_head(model)
        if head is not None and hasattr(head, '_last_eta_str_log'):
            record = list(head._last_eta_str_log)
            self.eta_str_records.append(record)
        else:
            self.eta_str_records.append([])

    @staticmethod
    def _get_head(model):
        """从 (可能被 wrap 的) model 中提取 bbox_head"""
        if hasattr(model, 'bbox_head'):
            return model.bbox_head
        if hasattr(model, 'module'):
            inner = model.module
            if hasattr(inner, 'bbox_head'):
                return inner.bbox_head
            if hasattr(inner, 'model') and hasattr(inner.model, 'bbox_head'):
                return inner.model.bbox_head
        return None

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._collect_once(runner, batch_idx)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._collect_once(runner, batch_idx)

    def compute_stats(self):
        """聚合所有图的 eta_str, 按步索引计算 mean/std/median/max"""
        if not self.eta_str_records:
            return {}
        # 找最大长度 (理论上应一致, 但 box_renewal/reset 可能使部分图更短)
        max_len = max(len(r) for r in self.eta_str_records)
        stats = {
            'n_images': len(self.eta_str_records),
            'max_steps_recorded': max_len,
            'per_step': {},
            'aggregate': {},
        }
        # 按步索引聚合
        for step_idx in range(max_len):
            values = [
                r[step_idx] for r in self.eta_str_records
                if step_idx < len(r) and r[step_idx] is not None
            ]
            if not values:
                continue
            arr = np.array(values)
            stats['per_step'][f'step_{step_idx + 1}'] = {
                'n': len(arr),
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'median': float(np.median(arr)),
                'p25': float(np.percentile(arr, 25)),
                'p75': float(np.percentile(arr, 75)),
                'p95': float(np.percentile(arr, 95)),
                'max': float(arr.max()),
                'min': float(arr.min()),
            }
        # 聚合 (所有步所有图)
        all_values = [v for r in self.eta_str_records for v in r if v is not None]
        if all_values:
            arr = np.array(all_values)
            stats['aggregate'] = {
                'n': len(arr),
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'median': float(np.median(arr)),
                'p95': float(np.percentile(arr, 95)),
                'max': float(arr.max()),
            }
        return stats


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='R1: eta_str 直线度指标测量')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--dataset', default='val', choices=['val', 'test'])
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampling-steps', type=int, default=None)
    parser.add_argument('--solver-type', type=str, default=None,
                        choices=['euler', 'heun', 'ddim', 'dpm_solver_pp', 'dpm_solver_pp_3'])
    parser.add_argument('--box-renewal', type=str, default=None, choices=['on', 'off'],
                        help='推理时覆盖 box_renewal (D3 实验: 验证 renewal 对 eta_str 的污染)')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--exp-name', default='r1_eta_str_measure')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    set_seed(args.seed)

    cfg = Config.fromfile(args.config)
    cfg.work_dir = os.path.dirname(args.checkpoint) or 'work_dirs/r1_eta_str'

    if args.sampling_steps is not None:
        cfg.model.bbox_head.sampling_timesteps = args.sampling_steps
    if args.solver_type is not None:
        cfg.model.bbox_head.solver_type = args.solver_type
    if args.box_renewal is not None:
        cfg.model.bbox_head.box_renewal = (args.box_renewal == 'on')
        print(f'[D3] box_renewal overridden: {cfg.model.bbox_head.box_renewal}')

    # 切换为 val 模式 (与 test.py 一致)
    if args.dataset == 'val':
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator

    # 禁用 SwanLab (实验脚本, 不需要上传)
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    if 'visualizer' in cfg:
        cfg.visualizer['vis_backends'] = [dict(type='LocalVisBackend')]

    # 挂载 EtaStrCollectorHook
    eta_hook = EtaStrCollectorHook()
    if 'custom_hooks' in cfg:
        cfg.custom_hooks.append(dict(type='EtaStrCollectorHook'))
    else:
        cfg.custom_hooks = [dict(type='EtaStrCollectorHook')]
    # 直接传实例 (绕过 config 反序列化)
    # Runner 会通过 custom_hooks 配置构建, 我们改用 register + 实例注入
    # 更简单方式: Runner 构造后用 runner.register_hook
    cfg.custom_hooks = []  # 清空, 后面手动注册

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    runner.register_hook(eta_hook, priority='LOW')

    print(f'[R1] 开始推理: {args.checkpoint}')
    print(f'[R1] solver={cfg.model.bbox_head.get("solver_type", "unknown")}, '
          f'steps={cfg.model.bbox_head.get("sampling_timesteps", "unknown")}, '
          f'seed={args.seed}')

    metrics = runner.test()

    # 汇总
    stats = eta_hook.compute_stats()
    result = {
        'config': {
            'checkpoint': args.checkpoint,
            'solver_type': cfg.model.bbox_head.get('solver_type', 'unknown'),
            'sampling_timesteps': cfg.model.bbox_head.get('sampling_timesteps', 'unknown'),
            'dataset': args.dataset,
            'seed': args.seed,
            'box_renewal': cfg.model.bbox_head.get('box_renewal', True),
            'topk_pruning_enabled': cfg.model.bbox_head.get('topk_pruning_enabled', False),
            'topk_k': cfg.model.bbox_head.get('topk_k', None),
        },
        'mAP': metrics.get('coco/bbox_mAP', None) if metrics else None,
        'eta_str_stats': stats,
        # 保存前 50 张图的原始记录用于分布检查
        'sample_records': eta_hook.eta_str_records[:50],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n{"="*70}')
    print(f'[R1] eta_str 统计结果 ({args.checkpoint})')
    print(f'{"="*70}')
    print(f'图像数: {stats.get("n_images", 0)}')
    print(f'记录步数: {stats.get("max_steps_recorded", 0)}')
    for step_key, step_stat in stats.get('per_step', {}).items():
        print(f'  {step_key}: mean={step_stat["mean"]:.6f}, '
              f'std={step_stat["std"]:.6f}, '
              f'median={step_stat["median"]:.6f}, '
              f'p95={step_stat["p95"]:.6f}')
    agg = stats.get('aggregate', {})
    if agg:
        print(f'  聚合: mean={agg["mean"]:.6f}, std={agg["std"]:.6f}, '
              f'median={agg["median"]:.6f}, p95={agg["p95"]:.6f}')
    print(f'\n[R1] 结果已保存到: {args.output}')


if __name__ == '__main__':
    main()
