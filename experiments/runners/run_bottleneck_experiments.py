#!/usr/bin/env python3
"""瓶颈实验编排脚本 — 一键运行所有瓶颈诊断实验

分三个阶段:
阶段 1 (无需训练): 后验瓶颈诊断 + 采样步数扫描 (使用已有 checkpoint)
阶段 2 (需训练): 组件消融实验 (每个实验需重新训练)
阶段 3 (需训练): 损失/调度消融实验

Usage:
    # 仅运行阶段 1 (快速, 无需训练)
    python experiments/runners/run_bottleneck_experiments.py \
        --config experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py \
        --checkpoint work_dirs/ldmdet_rf_heun_adaln/best_coco_bbox_mAP_epoch_102.pth \
        --phase 1

    # 运行全部三个阶段
    python experiments/runners/run_bottleneck_experiments.py \
        --config experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py \
        --checkpoint work_dirs/ldmdet_rf_heun_adaln/best_coco_bbox_mAP_epoch_102.pth \
        --phase all

    # 仅运行特定消融实验
    python experiments/runners/run_bottleneck_experiments.py \
        --config experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py \
        --checkpoint work_dirs/ldmdet_rf_heun_adaln/best_coco_bbox_mAP_epoch_102.pth \
        --phase 2 --experiments no_box_renewal,no_ensemble
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

REPO = Path(_PROJECT_ROOT)
ANN_FILE = 'data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json'

# 阶段 2: 组件消融实验 (需训练)
PHASE2_EXPERIMENTS = {
    'no_box_renewal': 'experiments/configs/bottleneck/no_box_renewal.py',
    'no_ensemble': 'experiments/configs/bottleneck/no_ensemble.py',
    'no_deep_supervision': 'experiments/configs/bottleneck/no_deep_supervision.py',
    'proposals_100': 'experiments/configs/bottleneck/proposals_100.py',
    'proposals_1000': 'experiments/configs/bottleneck/proposals_1000.py',
    'single_step_sampling': 'experiments/configs/bottleneck/single_step_sampling.py',
    'multi_step_sampling': 'experiments/configs/bottleneck/multi_step_sampling.py',
    'heads_2': 'experiments/configs/bottleneck/heads_2.py',
    'heads_12': 'experiments/configs/bottleneck/heads_12.py',
}

# 阶段 3: 损失/调度消融实验 (需训练)
# 分为 3a (定位瓶颈), 3b (分类瓶颈), 3c (扩散调度)
PHASE3_EXPERIMENTS = {
    # 3a: 定位瓶颈 (针对 AP-IoU 下降和高 IoU 下 Loc 误差)
    'scale_aware_loss': 'experiments/configs/bottleneck/scale_aware_loss.py',
    'relative_l1_loss': 'experiments/configs/bottleneck/relative_l1_loss.py',
    'high_giou_weight': 'experiments/configs/bottleneck/high_giou_weight.py',
    # 3b: 分类瓶颈 (针对 Cls 误差和类别不平衡)
    'focal_gamma_3': 'experiments/configs/bottleneck/focal_gamma_3.py',
    'focal_gamma_1_5': 'experiments/configs/bottleneck/focal_gamma_1_5.py',
    'high_cls_weight': 'experiments/configs/bottleneck/high_cls_weight.py',
    'class_balanced_sampling': 'experiments/configs/bottleneck/class_balanced_sampling.py',
    # 3c: 扩散调度
    'dpm_solver_pp': 'experiments/configs/bottleneck/dpm_solver_pp.py',
    'ddpm_baseline': 'experiments/configs/bottleneck/ddpm_baseline.py',
    'rf_shift_1': 'experiments/configs/bottleneck/rf_shift_1.py',
    'rf_shift_5': 'experiments/configs/bottleneck/rf_shift_5.py',
}


def run_command(cmd: list[str], cwd: str = None) -> int:
    """运行命令, 实时打印输出"""
    print(f'\n$ {" ".join(cmd)}\n')
    result = subprocess.run(cmd, cwd=cwd or str(REPO))
    return result.returncode


def phase1_diagnosis(config: str, checkpoint: str, output_dir: str):
    """阶段 1: 后验瓶颈诊断 (无需训练)"""
    print('\n' + '=' * 80)
    print('阶段 1: 后验瓶颈诊断 (无需训练)')
    print('=' * 80)

    # 1a. 综合瓶颈诊断
    print('\n--- 1a. 综合瓶颈诊断 ---')
    bottleneck_report = os.path.join(output_dir, 'bottleneck_report.json')
    cmd = [
        sys.executable, 'experiments/analysis/bottleneck_diagnosis.py',
        config, checkpoint,
        '--ann', ANN_FILE,
        '--output', bottleneck_report,
        '--cache', os.path.join(output_dir, 'bottleneck_preds.json'),
    ]
    run_command(cmd)

    # 1b. 采样步数扫描
    print('\n--- 1b. 采样步数扫描 ---')
    sampling_report = os.path.join(output_dir, 'sampling_sweep_results.json')
    cmd = [
        sys.executable, 'experiments/analysis/sampling_step_sweep.py',
        config, checkpoint,
        '--ann', ANN_FILE,
        '--steps', '1,2,4,8,16',
        '--output', sampling_report,
    ]
    run_command(cmd)

    print(f'\n阶段 1 完成. 报告:')
    print(f'  瓶颈诊断: {bottleneck_report}')
    print(f'  采样扫描: {sampling_report}')


def phase2_component_ablation(
    config: str,
    checkpoint: str,
    output_dir: str,
    experiments: list[str] = None,
    gpu_id: int = 0,
):
    """阶段 2: 组件消融实验 (需训练)"""
    print('\n' + '=' * 80)
    print('阶段 2: 组件消融实验 (需训练)')
    print('=' * 80)

    exps = experiments or list(PHASE2_EXPERIMENTS.keys())
    results = {}

    for exp_name in exps:
        if exp_name not in PHASE2_EXPERIMENTS:
            print(f'  [SKIP] 未知实验: {exp_name}')
            continue

        exp_config = PHASE2_EXPERIMENTS[exp_name]
        work_dir = os.path.join(output_dir, 'ablation', exp_name)

        print(f'\n--- 训练: {exp_name} ---')
        print(f'  Config: {exp_config}')
        print(f'  WorkDir: {work_dir}')

        cmd = [
            sys.executable, 'experiments/runners/train.py',
            exp_config,
            '--work-dir', work_dir,
            '--gpu-id', str(gpu_id),
        ]
        ret = run_command(cmd)
        if ret != 0:
            print(f'  [FAIL] {exp_name} 训练失败 (exit={ret})')
            results[exp_name] = {'status': 'failed', 'returncode': ret}
            continue

        # 找到 best checkpoint
        best_ckpts = list(Path(work_dir).glob('best_coco_bbox_mAP_epoch_*.pth'))
        if not best_ckpts:
            print(f'  [WARN] {exp_name} 未找到 best checkpoint')
            results[exp_name] = {'status': 'no_checkpoint'}
            continue

        best_ckpt = str(best_ckpts[0])
        print(f'  Best checkpoint: {best_ckpt}')

        # 评估 (使用 test.py runner)
        cmd = [
            sys.executable, 'experiments/runners/test.py',
            exp_config,
            '--checkpoint', best_ckpt,
        ]
        run_command(cmd)
        results[exp_name] = {
            'status': 'ok',
            'checkpoint': best_ckpt,
            'config': exp_config,
        }

    # 保存汇总
    summary_path = os.path.join(output_dir, 'phase2_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n阶段 2 汇总已保存到: {summary_path}')


def phase3_loss_ablation(
    config: str,
    checkpoint: str,
    output_dir: str,
    experiments: list[str] = None,
    gpu_id: int = 0,
):
    """阶段 3: 损失/调度消融实验 (需训练)"""
    print('\n' + '=' * 80)
    print('阶段 3: 损失/调度消融实验 (需训练)')
    print('=' * 80)

    exps = experiments or list(PHASE3_EXPERIMENTS.keys())
    results = {}

    for exp_name in exps:
        if exp_name not in PHASE3_EXPERIMENTS:
            print(f'  [SKIP] 未知实验: {exp_name}')
            continue

        exp_config = PHASE3_EXPERIMENTS[exp_name]
        work_dir = os.path.join(output_dir, 'ablation', exp_name)

        print(f'\n--- 训练: {exp_name} ---')
        print(f'  Config: {exp_config}')
        print(f'  WorkDir: {work_dir}')

        cmd = [
            sys.executable, 'experiments/runners/train.py',
            exp_config,
            '--work-dir', work_dir,
            '--gpu-id', str(gpu_id),
        ]
        ret = run_command(cmd)
        if ret != 0:
            print(f'  [FAIL] {exp_name} 训练失败 (exit={ret})')
            results[exp_name] = {'status': 'failed', 'returncode': ret}
            continue

        best_ckpts = list(Path(work_dir).glob('best_coco_bbox_mAP_epoch_*.pth'))
        if not best_ckpts:
            print(f'  [WARN] {exp_name} 未找到 best checkpoint')
            results[exp_name] = {'status': 'no_checkpoint'}
            continue

        best_ckpt = str(best_ckpts[0])
        results[exp_name] = {
            'status': 'ok',
            'checkpoint': best_ckpt,
            'config': exp_config,
        }

    summary_path = os.path.join(output_dir, 'phase3_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n阶段 3 汇总已保存到: {summary_path}')


def generate_final_report(output_dir: str):
    """生成最终汇总报告"""
    print('\n' + '=' * 80)
    print('最终汇总报告')
    print('=' * 80)

    report = {
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
    }

    # 加载阶段 1 报告
    bottleneck_path = os.path.join(output_dir, 'bottleneck_report.json')
    if os.path.exists(bottleneck_path):
        with open(bottleneck_path) as f:
            report['bottleneck_diagnosis'] = json.load(f)

    sampling_path = os.path.join(output_dir, 'sampling_sweep_results.json')
    if os.path.exists(sampling_path):
        with open(sampling_path) as f:
            report['sampling_sweep'] = json.load(f)

    # 加载阶段 2/3 汇总
    for phase_file in ['phase2_summary.json', 'phase3_summary.json']:
        path = os.path.join(output_dir, phase_file)
        if os.path.exists(path):
            with open(path) as f:
                report[phase_file.replace('.json', '')] = json.load(f)

    # 打印关键发现
    print('\n关键发现:')
    if 'bottleneck_diagnosis' in report:
        bd = report['bottleneck_diagnosis']
        for key, val in bd.items():
            if isinstance(val, dict) and 'analysis' in val:
                print(f'  [{key}] {val["analysis"]}')

    if 'sampling_sweep' in report:
        ss = report['sampling_sweep']
        if 'map_range' in ss:
            print(f'  [sampling] mAP 变化范围: {ss["map_range"]:.4f}')

    final_path = os.path.join(output_dir, 'final_report.json')
    with open(final_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n最终报告已保存到: {final_path}')


def main():
    parser = argparse.ArgumentParser(description='瓶颈实验编排')
    parser.add_argument('--config', required=True, help='Baseline 配置文件')
    parser.add_argument('--checkpoint', required=True, help='Baseline checkpoint')
    parser.add_argument('--phase', default='1', help='运行阶段: 1, 2, 3, all')
    parser.add_argument('--experiments', default=None, help='指定实验 (逗号分隔, 仅 phase 2/3)')
    parser.add_argument('--output-dir', default='work_dirs/bottleneck',
                        help='输出目录')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    phase = args.phase.lower()
    experiments = args.experiments.split(',') if args.experiments else None

    if phase in ('1', 'all'):
        phase1_diagnosis(args.config, args.checkpoint, args.output_dir)

    if phase in ('2', 'all'):
        phase2_component_ablation(
            args.config, args.checkpoint, args.output_dir,
            experiments, args.gpu_id,
        )

    if phase in ('3', 'all'):
        phase3_loss_ablation(
            args.config, args.checkpoint, args.output_dir,
            experiments, args.gpu_id,
        )

    generate_final_report(args.output_dir)


if __name__ == '__main__':
    main()
