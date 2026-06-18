#!/usr/bin/env python3
"""消融实验预警检查 — 每轮结束后自动评估"""
import json, sys, os

# 预警阈值
THRESHOLDS = {
    'rf_heun_adaln': {'min': 0.73, 'max': 0.77, 'label': 'Random baseline'},
    'ddpm':           {'min': 0.70, 'max': 0.75, 'label': 'DDPM baseline'},
    'hard_ot':        {'max': None,  'label': 'Hard OT'},
    'sinkhorn_stochastic': {'min': None, 'label': 'Sinkhorn ε=5'},
    'ghss':           {'min': None, 'label': 'GHSS ε=5'},
}

# 顶会阈值 (CVPR/ICCV detection)
VENUE_THRESHOLD = {
    'min_improvement': 0.005,   # 至少 +0.005 才算正面
    'std_max': 0.01,            # seed 波动不能太大
}

def check_experiment(exp_name, results):
    t = THRESHOLDS.get(exp_name, {})
    label = t.get('label', exp_name)
    maps = [r['best_mAP'] for r in results if r.get('best_mAP')]
    
    if len(maps) < 2:
        return f"⚠️  {label}: only {len(maps)} seed(s), need ≥2"
    
    import statistics
    mean = statistics.mean(maps)
    std = statistics.stdev(maps) if len(maps) > 1 else 0
    
    issues = []
    
    # 检查 min threshold
    if t.get('min') and mean < t['min']:
        issues.append(f"🔴 BELOW MIN: mean={mean:.4f} < {t['min']}")
    if t.get('max') and mean > t['max']:
        issues.append(f"🟡 ABOVE MAX: mean={mean:.4f} > {t['max']} (unexpected)")
    
    if std > VENUE_THRESHOLD['std_max']:
        issues.append(f"🟡 HIGH VARIANCE: std={std:.4f} > {VENUE_THRESHOLD['std_max']}")
    
    if not issues:
        return f"✅ {label}: mean={mean:.4f}±{std:.4f}  [OK]"
    return f"{chr(10)}".join(issues)


def check_relative(prev, curr, prev_label, curr_label):
    """检查相对顺序是否符合预期"""
    if prev is None or curr is None:
        return ""
    prev_m = sum(r['best_mAP'] for r in prev) / len(prev)
    curr_m = sum(r['best_mAP'] for r in curr) / len(curr)
    diff = curr_m - prev_m
    
    lines = [f"{curr_label} vs {prev_label}: Δ={diff:+.4f}"]
    
    if 'Random' in prev_label and 'Hard OT' in curr_label:
        # OT should be LOWER than Random
        if diff > 0:
            lines.append("🔴 REVERSED! OT > Random — theory contradicted")
        elif abs(diff) < VENUE_THRESHOLD['min_improvement']:
            lines.append("⚠️  OT ≈ Random — no degradation, coupling irrelevant?")
    
    if 'DDPM' in prev_label and 'Random' in curr_label:
        # RF should be HIGHER than DDPM
        if diff < 0:
            lines.append("🔴 DDPM > RF — RF not improving, check implementation")
    
    if 'GHSS' in curr_label:
        if diff <= 0:
            lines.append("🔴 GHSS ≤ Random — method fails")
    
    return chr(10).join(lines)


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else 'work_dirs/multi_seed_aug'
    
    exps = {}
    for d in sorted(os.listdir(base)):
        rpt = os.path.join(base, d, 'report.json')
        if os.path.exists(rpt):
            with open(rpt) as f:
                data = json.load(f)
            if 'error' not in data:
                exps[d] = data
    
    print("=== 实验预警检查 ===")
    prev_data = None
    prev_name = None
    
    for name, data in exps.items():
        exp_key = name.replace('rf_heun_adaln', 'rf_heun_adaln')  # normalize
        for key in THRESHOLDS:
            if key in name:
                results = data if isinstance(data, list) else data.get('results', [data])
                if isinstance(data, dict) and 'results' in data:
                    results = data['results']
                print(check_experiment(key, results))
                if prev_data:
                    print(check_relative(prev_data, results, prev_name, key))
                prev_data = results
                prev_name = key
                break
    
    if not exps:
        print("No completed experiments yet.")
