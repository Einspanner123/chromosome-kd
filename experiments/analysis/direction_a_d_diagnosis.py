"""方向 A + D 零成本前置诊断脚本

目的:
  方向 A: 验证 bbox 4 维度 (cx,cy,w,h) 的曲率是否一致
          假设: w,h 维度的 eta_str 显著小于 cx,cy → 可设计检测专用 solver
  方向 D: 验证自适应阶次假设 (重构后)
          假设: 早期 step (t大) eta_3rd 高 → 用 3 阶; 后期 step (t小) eta_3rd 低 → 用 2 阶
          对比: 固定 3 阶 vs 固定 2 阶的 mAP 差异

输入:
  - A4 checkpoint (work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth)
  - 24obj val set (data/24_chromosomes_object/coco/valid/)

输出:
  - work_dirs/diagnosis/direction_a_d_diagnosis.json
    {
      "per_dim_eta_str": {
        "step_1": {"cx": ..., "cy": ..., "w": ..., "h": ...},
        "step_2": {...},
        ...
      },
      "eta_str_scalar": [...],   # 原 R1 诊断量, 用于对比
      "eta_3rd": [...],          # ||D2||/||x0|| per step
      "n_images": 50,
      "checkpoint": "...",
      "config": "a4_dpm_pp_24obj.py",
    }

用法:
  python experiments/analysis/direction_a_d_diagnosis.py \\
      --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \\
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
      --num-images 50 --gpu-id 0

  # 用 3 阶 solver 跑同样的诊断 (对比 eta_3rd 是否被实际应用)
  python experiments/analysis/direction_a_d_diagnosis.py \\
      --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py \\
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
      --solver-type dpm_solver_pp_3 --num-images 50 --gpu-id 0

  # 测试自适应 solver (方向 D Phase 2)
  python experiments/analysis/direction_a_d_diagnosis.py \\
      --config experiments/configs/ldmdet/directions/mainline_ablation_24obj/a7_dpm_pp_adaptive_24obj.py \\
      --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
      --num-images 50 --gpu-id 0
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# 确保项目根目录在 Python path 中
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='方向 A+D 零成本诊断')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='checkpoint 路径')
    parser.add_argument(
        '--num-images', type=int, default=50,
        help='诊断使用的图像数 (默认 50, 足够稳定的 eta_str 估计)',
    )
    parser.add_argument(
        '--solver-type', default=None,
        help='覆盖 solver_type (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)',
    )
    parser.add_argument(
        '--adaptive-mode', default=None,
        help='自适应 solver 模式 (static / eta_threshold), 仅 dpm_solver_pp_adaptive 生效',
    )
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--output', default=None,
        help='输出 JSON 路径 (默认 work_dirs/diagnosis/direction_a_d_diagnosis.json)',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 固定随机种子, 确保诊断可复现
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 加载配置
    cfg = Config.fromfile(args.config)

    # 覆盖 solver_type (用于对比 2 阶 / 3 阶 / adaptive)
    if args.solver_type is not None:
        cfg.model.bbox_head.solver_type = args.solver_type
        print(f"[diagnosis] 覆盖 solver_type = {args.solver_type}")
    if args.adaptive_mode is not None:
        cfg.model.bbox_head.adaptive_solver_mode = args.adaptive_mode
        print(f"[diagnosis] 覆盖 adaptive_solver_mode = {args.adaptive_mode}")

    # 限制 num_images: 取 val_dataloader 的前 N 张
    cfg.val_dataloader = dict(cfg.val_dataloader)
    # 通过 num_workers=0 + batch_size=1 简化诊断
    cfg.val_dataloader['num_workers'] = 0
    cfg.val_dataloader['batch_size'] = 1

    # 构建 Runner (用于加载 checkpoint 和构建模型)
    cfg.load_from = args.checkpoint
    cfg.work_dir = '/tmp/diagnosis_workdir'  # 临时目录, 不污染 work_dirs
    cfg.gpu_id = args.gpu_id
    cfg.experiment_name = 'direction_a_d_diagnosis'

    # 禁用 SwanLab 上传 (诊断脚本不应污染训练项目)
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=cfg.vis_backends,
        name='visualizer',
    )

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    bbox_head = model.bbox_head
    bbox_head.eval()

    # 收集诊断量
    # eta_str_per_dim: list of [cx, cy, w, h] per step, 累加后取均值
    # eta_str_scalar: list of float per step
    # eta_3rd: list of float per step
    # applied_3rd: list of bool per step (仅 adaptive solver 有意义)
    per_dim_accum = defaultdict(list)  # step_idx -> list of [cx,cy,w,h]
    scalar_accum = defaultdict(list)   # step_idx -> list of float
    eta_3rd_accum = defaultdict(list)  # step_idx -> list of float
    applied_3rd_accum = defaultdict(list)  # step_idx -> list of bool

    val_dataloader = runner.val_dataloader
    n_processed = 0
    print(f"[diagnosis] 开始处理前 {args.num_images} 张验证图像...")

    with torch.no_grad():
        for data in val_dataloader:
            if n_processed >= args.num_images:
                break

            # 标准 predict 调用 (会自动收集 _last_eta_str_log 等)
            outputs = bbox_head.predict(
                data['inputs'] if 'inputs' in data else
                runner.model.data_processor(data, False)['inputs'],
                data['data_samples'] if 'data_samples' in data else
                runner.model.data_processor(data, False)['data_samples'],
            )

            # 收集诊断量 (head.py predict() 末尾保存到 self._last_*_log)
            # 直接访问属性: 若 head.py 未赋值 (代码改动或 solver 不支持),
            # 让 AttributeError 显式抛出, 而非静默返回空 list 掩盖问题.
            scalar_log = bbox_head._last_eta_str_log
            per_dim_log = bbox_head._last_eta_str_per_dim_log
            eta_3rd_log = bbox_head._last_eta_3rd_log
            applied_3rd_log = bbox_head._last_applied_3rd_log

            for step_idx, val in enumerate(scalar_log):
                scalar_accum[step_idx].append(float(val))
            for step_idx, vals in enumerate(per_dim_log):
                per_dim_accum[step_idx].append([float(v) for v in vals])
            for step_idx, val in enumerate(eta_3rd_log):
                eta_3rd_accum[step_idx].append(float(val))
            for step_idx, val in enumerate(applied_3rd_log):
                applied_3rd_accum[step_idx].append(bool(val))

            n_processed += 1
            if n_processed % 10 == 0:
                print(f"  [{n_processed}/{args.num_images}] 处理中...")

    # 聚合统计
    def mean_std(vals):
        if not vals:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        arr = np.array(vals)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n": len(arr),
        }

    results = {
        "config": os.path.relpath(args.config, _PROJECT_ROOT),
        "checkpoint": args.checkpoint,
        "solver_type": cfg.model.bbox_head.get('solver_type', 'unknown'),
        "n_images": n_processed,
        "eta_str_scalar_per_step": {
            str(k): mean_std(v) for k, v in sorted(scalar_accum.items())
        },
        "eta_str_per_dim_per_step": {
            str(k): {
                "cx": mean_std([row[0] for row in v]),
                "cy": mean_std([row[1] for row in v]),
                "w": mean_std([row[2] for row in v]),
                "h": mean_std([row[3] for row in v]),
            }
            for k, v in sorted(per_dim_accum.items())
        },
        "eta_3rd_per_step": {
            str(k): mean_std(v) for k, v in sorted(eta_3rd_accum.items())
        },
        "applied_3rd_ratio_per_step": {
            str(k): {
                "ratio_applied": float(np.mean(v)) if v else 0.0,
                "n": len(v),
            }
            for k, v in sorted(applied_3rd_accum.items())
        },
    }

    # 方向 A 关键诊断: per-dim eta_str 比值 (w,h vs cx,cy)
    # 如果 w,h 的 eta_str 显著小于 cx,cy, 则方向 A 假设成立
    per_dim_summary = {}
    for step_str, dim_stats in results["eta_str_per_dim_per_step"].items():
        cx_mean = dim_stats["cx"]["mean"]
        cy_mean = dim_stats["cy"]["mean"]
        w_mean = dim_stats["w"]["mean"]
        h_mean = dim_stats["h"]["mean"]
        cxcy_avg = (cx_mean + cy_mean) / 2 + 1e-8
        wh_avg = (w_mean + h_mean) / 2 + 1e-8
        per_dim_summary[step_str] = {
            "cx_mean": cx_mean,
            "cy_mean": cy_mean,
            "w_mean": w_mean,
            "h_mean": h_mean,
            "wh_to_cxcy_ratio": float(wh_avg / cxcy_avg),
        }
    results["direction_a_summary"] = per_dim_summary

    # 方向 D 关键诊断: eta_3rd 随 step 的变化趋势
    # 如果早期 step eta_3rd 高, 后期低, 则方向 D 重构假设成立
    eta_3rd_trend = []
    for step_str, stats in results["eta_3rd_per_step"].items():
        eta_3rd_trend.append(stats["mean"])
    results["direction_d_summary"] = {
        "eta_3rd_by_step": eta_3rd_trend,
        "trend": (
            "decreasing (假设成立: 早期高, 后期低)"
            if len(eta_3rd_trend) >= 2 and eta_3rd_trend[0] > eta_3rd_trend[-1]
            else "increasing (假设不成立: 早期低, 后期高)"
            if len(eta_3rd_trend) >= 2 and eta_3rd_trend[0] < eta_3rd_trend[-1]
            else "insufficient data"
        ),
    }

    # 输出
    output_path = args.output or os.path.join(
        _PROJECT_ROOT, 'work_dirs', 'diagnosis',
        'direction_a_d_diagnosis.json'
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[diagnosis] 完成! 处理 {n_processed} 张图")
    print(f"[diagnosis] 结果保存到: {output_path}")

    # 打印关键结论
    print("\n" + "=" * 60)
    print("方向 A 关键诊断: per-dim eta_str (||D1||/||x0||)")
    print("=" * 60)
    print(f"{'step':<6} {'cx':<10} {'cy':<10} {'w':<10} {'h':<10} {'wh/cxcy':<10}")
    for step_str, s in per_dim_summary.items():
        print(f"{step_str:<6} {s['cx_mean']:<10.4f} {s['cy_mean']:<10.4f} "
              f"{s['w_mean']:<10.4f} {s['h_mean']:<10.4f} "
              f"{s['wh_to_cxcy_ratio']:<10.4f}")
    print("\n→ 若 wh/cxcy 显著 < 1 (如 < 0.5), 则方向 A 假设成立 "
          "(w,h 维度曲率显著小于 cx,cy)")

    print("\n" + "=" * 60)
    print("方向 D 关键诊断: eta_3rd (||D2||/||x0||) 随 step 变化")
    print("=" * 60)
    for step_str, stats in results["eta_3rd_per_step"].items():
        print(f"  step {step_str}: mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"\n趋势: {results['direction_d_summary']['trend']}")
    print("→ 若 decreasing, 则方向 D 重构假设成立 "
          "(早期 step 用 3 阶, 后期用 2 阶)")


if __name__ == '__main__':
    main()
