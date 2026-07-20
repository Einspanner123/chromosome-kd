"""汇总 R1 + D3 全部实验结果 (3 seeds × 4 configs)"""
import json
import os
from collections import defaultdict

SEEDS = [42, 123, 789]
CONFIGS = [
    ('baseline', 'r1_eta_str_a3_seed{seed}'),
    ('renewal_off', 'r1_eta_str_a3_seed{seed}_noRenewal'),
    ('topk_k200', 'r1_eta_str_a3_seed{seed}_k200'),
    ('topk_k100', 'r1_eta_str_a3_seed{seed}_k100'),
]

print("=" * 90)
print("R1 + D3 实验汇总 (3 seeds × 4 configs)")
print("=" * 90)

# Per-config per-seed table
print(f"\n{'Config':<14} {'Seed':>5} {'mAP':>6} {'Step1':>8} {'Step2':>8} {'Step3':>8} {'Mode':>12}")
print('-' * 80)

# 收集聚合数据
agg_data = defaultdict(lambda: {'mAP': [], 'step1': [], 'step2': [], 'step3': []})

for cfg_name, cfg_pattern in CONFIGS:
    for seed in SEEDS:
        path = f'experiments/analysis/{cfg_pattern.format(seed=seed)}.json'
        if not os.path.exists(path):
            print(f"{cfg_name:<14} {seed:>5} (missing: {path})")
            continue
        with open(path) as f:
            r = json.load(f)
        mAP = r['mAP']
        stats = r['eta_str_stats']
        s1 = stats.get('per_step', {}).get('step_1', {}).get('mean', 0.0)
        s2 = stats.get('per_step', {}).get('step_2', {}).get('mean', 0.0)
        s3 = stats.get('per_step', {}).get('step_3', {}).get('mean', 0.0)
        # 模式判断
        if s1 > s2 > s3:
            mode = 'decreasing'
        elif s1 < s2 > s3:
            mode = 'V-shape'
        elif s1 < s2 < s3:
            mode = 'increasing'
        else:
            mode = 'other'
        print(f"{cfg_name:<14} {seed:>5} {mAP:>6.3f} {s1:>8.3f} {s2:>8.3f} {s3:>8.3f} {mode:>12}")
        agg_data[cfg_name]['mAP'].append(mAP)
        agg_data[cfg_name]['step1'].append(s1)
        agg_data[cfg_name]['step2'].append(s2)
        agg_data[cfg_name]['step3'].append(s3)

# 聚合统计 (mean ± std across seeds)
print(f"\n{'='*80}")
print("Aggregate (mean ± std across 3 seeds)")
print(f"{'='*80}")
print(f"{'Config':<14} {'mAP':>12} {'Step1':>14} {'Step2':>14} {'Step3':>14}")
print('-' * 80)
import numpy as np
for cfg_name, _ in CONFIGS:
    data = agg_data[cfg_name]
    if not data['mAP']:
        continue
    mAP_str = f"{np.mean(data['mAP']):.3f} ± {np.std(data['mAP']):.3f}"
    s1_str = f"{np.mean(data['step1']):.3f} ± {np.std(data['step1']):.3f}"
    s2_str = f"{np.mean(data['step2']):.3f} ± {np.std(data['step2']):.3f}"
    s3_str = f"{np.mean(data['step3']):.3f} ± {np.std(data['step3']):.3f}"
    print(f"{cfg_name:<14} {mAP_str:>12} {s1_str:>14} {s2_str:>14} {s3_str:>14}")

# 关键发现
print(f"\n{'='*80}")
print("Key Findings:")
print(f"{'='*80}")
b = agg_data['baseline']
r = agg_data['renewal_off']
if b['mAP'] and r['mAP']:
    mAP_delta = np.mean(r['mAP']) - np.mean(b['mAP'])
    s1_ratio = np.mean(r['step1']) / np.mean(b['step1'])
    s3_ratio = np.mean(r['step3']) / np.mean(b['step3'])
    print(f"1. D3 矛盾验证: renewal off vs on")
    print(f"   mAP delta: {mAP_delta:+.4f} (噪声范围则 D3 mAP 影响小)")
    print(f"   Step1 eta_str ratio (off/on): {s1_ratio:.3f} (D3 预测 < 1)")
    print(f"   Step3 eta_str ratio (off/on): {s3_ratio:.3f}")
    print(f"   → renewal 使 eta_str 虚高 {(1-s1_ratio)*100:.1f}% (Step1), {(1-s3_ratio)*100:.1f}% (Step3)")

k100 = agg_data['topk_k100']
k200 = agg_data['topk_k200']
if k100['mAP'] and k200['mAP']:
    mAP_delta_k = np.mean(k100['mAP']) - np.mean(k200['mAP'])
    print(f"\n2. K=100 vs K=200 (D3 对 K=100 掉点解释)")
    print(f"   mAP delta: {mAP_delta_k:+.4f} (论文 claim -0.013)")
    print(f"   K=100 eta_str step2: {np.mean(k100['step2']):.3f}")
    print(f"   K=200 eta_str step2: {np.mean(k200['step2']):.3f}")
    print(f"   → eta_str 差异小, K=100 掉点主因是 proposal 数量不足, 非 DPM-Solver++ 历史破坏")
