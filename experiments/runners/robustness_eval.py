"""Robustness 实验 §4.8 — 批量推理 + 结果聚合

对 work_dirs/robustness_noise/perturbed/ 下的所有扰动 JSON (clean + 9 个噪声级别)
逐一运行 test.py 推理, 解析 mAP, 并聚合写入 results JSON。

输出:
  work_dirs/robustness_noise/results.json  — 每个噪声级别的 mAP/AP50/AP75/per-class AP

Usage:
    python experiments/runners/robustness_eval.py \\
        --perturbed-dir work_dirs/robustness_noise/perturbed \\
        --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
        --config experiments/configs/robustness/noise_test_a4.py \\
        --out work_dirs/robustness_noise/results.json \\
        --seed 42 --gpu-id 0

可选:
    --skip-existing   跳过 results.json 中已存在的噪声级别 (增量重跑)
    --only TAG        仅运行指定的噪声 tag (例如 clean 或 noise_both_s10_f020)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import timedelta


def parse_test_stdout(stdout: str):
    """从 test.py stdout 中解析 mAP / AP50 / AP75 / per-class AP。

    test.py 末尾会 print:
        Test Results (test) [...]
          ...
          coco/bbox_mAP: 0.8630
          coco/bbox_mAP_eas[50-95]_cls_A1: 0.9210
          ...
    """
    metrics = {}
    # 先定位 "Test Results" 区段 (test.py 末尾的 sorted metrics print)
    marker = 'Test Results'
    idx = stdout.rfind(marker)
    if idx < 0:
        return metrics
    section = stdout[idx:]
    # 只匹配 coco/ 前缀的 metric (避免误匹配 "torch: 2.1.0" 之类环境信息)
    pattern = re.compile(r'^\s+(coco/\S+):\s+([0-9.]+)\s*$', re.MULTILINE)
    for m in pattern.finditer(section):
        key = m.group(1)
        val = float(m.group(2))
        metrics[key] = val
    return metrics


def run_one(config, checkpoint, noise_ann, seed, gpu_id, exp_name=None):
    """运行一次 test.py 推理, 返回 (metrics_dict, stdout_tail)。"""
    env = dict(os.environ)
    env['NOISE_ANN_FILE'] = noise_ann
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable, 'experiments/runners/test.py',
        config,
        '--checkpoint', checkpoint,
        '--dataset', 'test',
        '--seed', str(seed),
        '--gpu-id', '0',  # CUDA_VISIBLE_DEVICES 已经把物理 GPU 映射为 0
    ]
    if exp_name:
        cmd += ['--exp-name', exp_name]

    print(f'\n[robustness_eval] RUN: {" ".join(cmd)}')
    print(f'[robustness_eval]   NOISE_ANN_FILE={noise_ann}')
    t0 = time.time()
    proc = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        cwd='/home/linkst/workspace/projects/chromosome-kd',
        timeout=3600,  # 单次最多 1 小时
    )
    dt = time.time() - t0
    print(f'[robustness_eval]   elapsed={timedelta(seconds=int(dt))}  exit={proc.returncode}')
    if proc.returncode != 0:
        # 失败: 打印尾部 stderr 帮助调试
        tail = '\n'.join(proc.stderr.splitlines()[-40:])
        print(f'[robustness_eval] FAILED. stderr tail:\n{tail}')
        return None, proc.stdout + '\n--- STDERR ---\n' + proc.stderr
    metrics = parse_test_stdout(proc.stdout)
    return metrics, proc.stdout


def main():
    parser = argparse.ArgumentParser(description='Robustness §4.8 批量推理 + 聚合')
    parser.add_argument('--perturbed-dir', default='work_dirs/robustness_noise/perturbed',
                        help='扰动 JSON 目录 (由 robustness_noise.py --grid 生成)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='experiments/configs/robustness/noise_test_a4.py')
    parser.add_argument('--out', default='work_dirs/robustness_noise/results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过 results.json 中已存在的 tag (增量重跑)')
    parser.add_argument('--only', default=None,
                        help='仅运行指定 tag (例如 clean 或 noise_both_s10_f020)')
    args = parser.parse_args()

    # 读 manifest (含所有 tag 的列表)
    manifest_path = os.path.join(args.perturbed_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    runs = manifest['runs']  # [{file, sigma, flip_rate, ...}, ...]

    # 已有结果 (增量模式)
    existing = {}
    if args.skip_existing and os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f).get('runs', {})
        existing = {r['file']: r for r in existing} if isinstance(existing, list) else existing

    all_results = dict(existing) if args.skip_existing else {}

    for run in runs:
        tag = run['file']
        if args.only and args.only not in tag:
            continue
        if tag in all_results and args.skip_existing:
            print(f'[robustness_eval] SKIP (existing): {tag}')
            continue

        noise_ann = os.path.join(args.perturbed_dir, tag)
        if not os.path.exists(noise_ann):
            print(f'[robustness_eval] SKIP (missing file): {noise_ann}')
            continue

        # exp_name = a4_noise_<tag_without_ext>
        exp_tag = os.path.splitext(tag)[0]
        metrics, stdout = run_one(
            args.config, args.checkpoint, noise_ann,
            args.seed, args.gpu_id, exp_name=f'a4_noise_{exp_tag}',
        )

        record = {
            'file': tag,
            'sigma': run['sigma'],
            'flip_rate': run['flip_rate'],
            'seed': run['seed'],
            'metrics': metrics,
            'stdout_tail': '\n'.join(stdout.splitlines()[-20:]),
        }
        if metrics is None:
            record['error'] = 'inference_failed'
        all_results[tag] = record

        # 增量保存 (每次跑完就写, 防止中途挂掉丢结果)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({
                'config': args.config,
                'checkpoint': args.checkpoint,
                'seed': args.seed,
                'runs': list(all_results.values()),
            }, f, indent=2)
        print(f'[robustness_eval] saved -> {args.out}')

        # 单条结果即时打印
        if metrics:
            mAP = metrics.get('coco/bbox_mAP', float('nan'))
            AP50 = metrics.get('coco/bbox_mAP_50', float('nan'))
            AP75 = metrics.get('coco/bbox_mAP_75', float('nan'))
            print(f'[robustness_eval] {tag}: mAP={mAP:.4f}  AP50={AP50:.4f}  AP75={AP75:.4f}')

    print(f'\n[robustness_eval] DONE. {len(all_results)} runs in {args.out}')


if __name__ == '__main__':
    main()
