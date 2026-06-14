"""多 seed 批量训练

Usage:
    python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/sinkhorn_stochastic.py --seeds 42,123,456 --gpus 0,0,1

输出: work_dirs/sinkhorn_stochastic/multi_seed/report.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_single(config: str, seed: int, gpu: int, work_dir: str) -> dict:
    """运行单次训练，返回 best mAP"""
    cmd = [
        sys.executable, '-m', 'experiments.runners.train',
        config,
        '--work-dir', work_dir,
        '--seed', str(seed),
        '--gpu-id', str(gpu),
    ]

    print(f'\n{"="*60}')
    print(f'Seed {seed} | GPU {gpu} | {work_dir}')
    print(f'{"="*60}')

    result = subprocess.run(cmd, capture_output=False)
    ok = result.returncode == 0

    # 读取 best mAP
    best_map = None
    try:
        from mmengine.logging import HistoryBuffer
        from mmengine.runner import load_checkpoint
        # 从 work_dir 的日志中提取 best mAP
        log_path = Path(work_dir) / 'vis_data' / 'scalars.json'
        if log_path.exists():
            with open(log_path) as f:
                lines = f.readlines()
                maps = []
                for line in lines:
                    try:
                        data = json.loads(line)
                        if 'coco/bbox_mAP' in data:
                            maps.append(data['coco/bbox_mAP'])
                    except json.JSONDecodeError:
                        continue
                if maps:
                    best_map = max(maps)
    except Exception:
        pass

    return {
        'seed': seed,
        'gpu': gpu,
        'work_dir': work_dir,
        'ok': ok,
        'best_mAP': best_map,
    }


def aggregate(results: list[dict]) -> dict:
    """聚合多 seed 结果"""
    maps = [r['best_mAP'] for r in results if r['best_mAP'] is not None]
    if not maps:
        return {'error': 'no valid mAP results', 'results': results}

    import statistics
    return {
        'n_seeds': len(results),
        'n_valid': len(maps),
        'mAP_mean': statistics.mean(maps),
        'mAP_std': statistics.stdev(maps) if len(maps) > 1 else 0.0,
        'mAP_max': max(maps),
        'mAP_min': min(maps),
        'per_seed': results,
    }


def main():
    parser = argparse.ArgumentParser(description='LDMDet Multi-Seed Training')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--seeds', default='42,123,456', help='Comma-separated seeds')
    parser.add_argument('--gpus', default='0', help='Comma-separated GPU IDs (repeated if fewer than seeds)')
    parser.add_argument('--base-dir', default=None, help='Base work directory')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    gpus = [int(g.strip()) for g in args.gpus.split(',')]

    # 循环 GPU 如果数量不足
    while len(gpus) < len(seeds):
        gpus.extend(gpus)
    gpus = gpus[:len(seeds)]

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = args.base_dir or os.path.join('work_dirs', 'multi_seed', f'{config_name}_{ts}')
    os.makedirs(base_dir, exist_ok=True)

    results = []
    for seed, gpu in zip(seeds, gpus):
        work_dir = os.path.join(base_dir, f'seed_{seed}')
        r = run_single(args.config, seed, gpu, work_dir)
        results.append(r)

    report = aggregate(results)

    # 保存报告
    report_path = os.path.join(base_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # 打印摘要
    print(f'\n{"="*60}')
    print(f'Multi-Seed Report: {config_name}')
    print(f'{"="*60}')
    if 'mAP_mean' in report:
        print(f'mAP: {report["mAP_mean"]:.4f} ± {report["mAP_std"]:.4f} (max={report["mAP_max"]:.4f}, min={report["mAP_min"]:.4f})')
    print(f'Per seed:')
    for r in results:
        status = '✓' if r['ok'] else '✗'
        mAP_str = f'{r["best_mAP"]:.4f}' if r['best_mAP'] else 'N/A'
        print(f'  seed={r["seed"]:4d}  {status}  mAP={mAP_str}  {r["work_dir"]}')
    print(f'\nReport saved to {report_path}')


if __name__ == '__main__':
    main()
