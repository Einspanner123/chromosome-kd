"""逐项排查推理优化的性能影响 (简化版)

在 ross 上运行, 隔离 dit_head.py 和 rectified_flow.py 的影响。

Usage (在 ross 上):
    cd /media/ross/8TB/linkst/chromo/chromosome-kd
    /home/linkst/data/miniconda3/envs/chromo/bin/python benchmark_ablation.py
"""

import os
import shutil
import subprocess

PROJECT_ROOT = '/media/ross/8TB/linkst/chromo/chromosome-kd'
PYTHON = '/home/linkst/data/miniconda3/envs/chromo/bin/python'
DIT_HEAD = 'projects/LDMDetDiT/mods/dit_head.py'
RF_FILE = 'projects/LDMDetDiT/mods/rectified_flow.py'

BENCH_CMD = [
    PYTHON, 'ldmdet/tools/benchmark_inference.py',
    '--solvers', 'heun', 'dpm_solver_pp',
    '--steps', '4', '--bs', '1', '--num-proposals', '500',
    '--warmup', '20', '--iters', '100', '--no-profile', '--gpu', '0',
]


def run_bench(label):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    result = subprocess.run(
        BENCH_CMD, capture_output=True, text=True,
        cwd=PROJECT_ROOT, timeout=300,
    )
    output = result.stdout + result.stderr
    heun_ms = dpm_ms = None
    for line in output.split('\n'):
        parts = line.split()
        # 只匹配 "heun 4 ..." 或 "dpm_solver_pp 4 ..." 格式 (Steps=4)
        if len(parts) >= 5 and parts[0] in ('heun', 'dpm_solver_pp') and parts[1] == '4':
            try:
                heun_ms = float(parts[-2]) if parts[0] == 'heun' else heun_ms
                dpm_ms = float(parts[-2]) if parts[0] == 'dpm_solver_pp' else dpm_ms
            except ValueError:
                pass
    print(f'  Heun: {heun_ms} ms | DPM++: {dpm_ms} ms')
    return {'label': label, 'heun_ms': heun_ms, 'dpm_ms': dpm_ms}


def checkout(commit, files):
    """git checkout 指定 commit 的文件版本"""
    subprocess.run(
        ['git', 'checkout', commit, '--'] + files,
        cwd=PROJECT_ROOT, check=True,
    )


def main():
    os.chdir(PROJECT_ROOT)
    results = []

    # 0. 基线: 全部回退 (ccbccdd6)
    checkout('ccbccdd6', [DIT_HEAD, RF_FILE])
    results.append(run_bench('0. 基线 (ccbccdd6, 全部回退)'))

    # 1. 优化后全部 (356935c9)
    checkout('HEAD', [DIT_HEAD, RF_FILE])
    results.append(run_bench('1. 优化后全部 (356935c9)'))

    # 2. 仅 dit_head 优化 (rectified_flow 回退)
    checkout('ccbccdd6', [RF_FILE])
    results.append(run_bench('2. 仅 dit_head 优化 (RF 回退)'))

    # 3. 仅 rectified_flow 优化 (dit_head 回退)
    checkout('HEAD', [RF_FILE])
    checkout('ccbccdd6', [DIT_HEAD])
    results.append(run_bench('3. 仅 RF 优化 (dit_head 回退)'))

    # 恢复优化后
    checkout('HEAD', [DIT_HEAD, RF_FILE])

    # 汇总
    print(f'\n{"="*70}')
    print('汇总结果')
    print(f'{"="*70}')
    print(f'{"配置":<50} {"Heun (ms)":>12} {"DPM++ (ms)":>12}')
    print('-' * 74)
    for r in results:
        h = f'{r["heun_ms"]:.2f}' if r['heun_ms'] else 'N/A'
        d = f'{r["dpm_ms"]:.2f}' if r['dpm_ms'] else 'N/A'
        print(f'{r["label"]:<50} {h:>12} {d:>12}')

    # 分析
    base = results[0]
    print(f'\n分析 (相对基线):')
    for r in results:
        if r['heun_ms'] and base['heun_ms']:
            d = r['heun_ms'] - base['heun_ms']
            p = d / base['heun_ms'] * 100
            tag = '✓ 加速' if d < -0.5 else ('✗ 变慢' if d > 0.5 else '~ 持平')
            print(f'  {r["label"]}: Heun {d:+.2f} ms ({p:+.1f}%) {tag}')


if __name__ == '__main__':
    main()
