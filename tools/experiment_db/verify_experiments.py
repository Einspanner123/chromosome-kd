"""文档核对脚本: 对比 SQLite 数据库与文档中的实验数据, 标记不一致.

用法:
  python tools/experiment_db/verify_experiments.py
  python tools/experiment_db/verify_experiments.py --doc docs/EXPERIMENT_LINEAGE.md
  python tools/experiment_db/verify_experiments.py --check-augmentation
  python tools/experiment_db/verify_experiments.py --check-mAP

核对内容:
  1. mAP 核对: 文档记录的 mAP vs 数据库实际值
  2. 增强策略核对: 文档标注 vs 数据库实际
  3. 数据集核对: 文档标注 vs 数据库实际
  4. 缺失/幽灵实验检查
"""

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')


# ─── 文档解析 ───────────────────────────────────────────────

def extract_workdir_refs(doc_path):
    """从文档中提取所有 work_dirs/... 路径引用.

    Returns:
        list[dict]: [{'path': str, 'line': int, 'context': str}]
    """
    refs = []
    with open(doc_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 匹配 work_dirs/... 路径
            matches = re.finditer(r'work_dirs/[^\s,)\]]+', line)
            for m in matches:
                path = m.group()
                # 清理路径 (去掉末尾的标点)
                path = re.sub(r'[/]+$', '', path)
                refs.append({
                    'path': path,
                    'line': line_num,
                    'context': line.strip()[:200],
                })
    return refs


def extract_documented_mAPs(doc_path):
    """从文档中提取所有 mAP=0.XXX 记录及其上下文.

    Returns:
        list[dict]: [{'mAP': float, 'line': int, 'context': str, 'work_dir': str|None}]
    """
    records = []
    with open(doc_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 匹配 mAP=0.XXX 或 mAP: 0.XXX
            matches = re.finditer(r'mAP[=:]\s*([\d.]+)', line)
            for m in matches:
                mAP_str = m.group(1)
                try:
                    mAP = float(mAP_str)
                except ValueError:
                    continue
                # 尝试提取同行的 work_dir 引用
                wd_match = re.search(r'work_dirs/([^\s,)\]]+)', line)
                work_dir = f'work_dirs/{wd_match.group(1)}' if wd_match else None

                records.append({
                    'mAP': mAP,
                    'line': line_num,
                    'context': line.strip()[:200],
                    'work_dir': work_dir,
                })
    return records


def extract_augmentation_labels(doc_path):
    """从文档中提取增强策略标注及其上下文.

    查找 "标准增强", "简单增强", "NoAug", "无aug", "简化设置" 等标签.

    Returns:
        list[dict]: [{'label': str, 'line': int, 'context': str}]
    """
    labels = []
    patterns = [
        (r'标准增强', 'standard'),
        (r'简单增强', 'simple'),
        (r'NoAug', 'simple'),
        (r'无aug', 'simple'),
        (r'简化设置', 'simple'),
        (r'无增强', 'simple'),
    ]

    with open(doc_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            for pattern, label in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    labels.append({
                        'label': label,
                        'pattern': pattern,
                        'line': line_num,
                        'context': line.strip()[:200],
                    })
    return labels


# ─── 数据库查询 ─────────────────────────────────────────────

def get_experiment_by_workdir(conn, work_dir):
    """通过 work_dir 查询实验."""
    cursor = conn.cursor()
    # 尝试精确匹配
    cursor.execute('''
        SELECT e.experiment_id, e.best_val_mAP, e.best_val_epoch, e.seed, e.status,
               c.dataset, c.coupling_type, c.ot_epsilon,
               a.pipeline_name
        FROM experiment e LEFT JOIN config c ON e.config_path = c.config_path
        LEFT JOIN aug_pipeline a ON c.aug_pipeline_hash = a.pipeline_hash
        WHERE e.experiment_id = ?
    ''', (work_dir,))
    row = cursor.fetchone()
    if row:
        return row
    # 尝试模糊匹配 (work_dir 可能是目录前缀)
    cursor.execute('''
        SELECT e.experiment_id, e.best_val_mAP, e.best_val_epoch, e.seed, e.status,
               c.dataset, c.coupling_type, c.ot_epsilon,
               a.pipeline_name
        FROM experiment e LEFT JOIN config c ON e.config_path = c.config_path
        LEFT JOIN aug_pipeline a ON c.aug_pipeline_hash = a.pipeline_hash
        WHERE e.experiment_id LIKE ?
        LIMIT 1
    ''', (f'{work_dir}%',))
    return cursor.fetchone()


def get_all_experiments(conn):
    """获取所有实验的 work_dir."""
    cursor = conn.cursor()
    cursor.execute('SELECT experiment_id FROM experiment WHERE server != "swanlab"')
    return [row[0] for row in cursor.fetchall()]


# ─── 核对逻辑 ───────────────────────────────────────────────

def verify_mAP(conn, doc_path, tolerance=0.002):
    """核对文档中的 mAP 值与数据库.

    Args:
        tolerance: 允许的误差 (0.002 = 0.2%)
    """
    records = extract_documented_mAPs(doc_path)
    verified = 0
    mismatched = 0
    no_db = 0

    print('\n=== mAP 核对 ===')
    print(f'文档中找到 {len(records)} 条 mAP 记录')

    for rec in records:
        if not rec['work_dir']:
            continue  # 没有 work_dir 引用, 无法核对

        row = get_experiment_by_workdir(conn, rec['work_dir'])
        if not row:
            no_db += 1
            continue

        db_mAP = row[1]  # best_val_mAP
        if db_mAP is None:
            no_db += 1
            continue

        diff = abs(db_mAP - rec['mAP'])
        if diff <= tolerance:
            verified += 1
        else:
            mismatched += 1
            print(f'  ❌ L{rec["line"]}: 文档={rec["mAP"]:.4f} vs DB={db_mAP:.4f} (Δ={diff:.4f})')
            print(f'     {rec["work_dir"]}')
            print(f'     {rec["context"][:150]}')

    print(f'\n核对结果: ✅ {verified} 一致 | ❌ {mismatched} 不一致 | ⚠️ {no_db} 无数据库记录')
    return mismatched


def verify_augmentation(conn, doc_path):
    """核对文档中的增强策略标注."""
    labels = extract_augmentation_labels(doc_path)
    simple_labels = [l for l in labels if l['label'] == 'simple']
    standard_labels = [l for l in labels if l['label'] == 'standard']

    print('\n=== 增强策略标注核对 ===')
    print(f'文档中标注: 标准增强 {len(standard_labels)} 处, 简单增强/NoAug {len(simple_labels)} 处')

    if simple_labels:
        print(f'\n⚠️ 仍有 {len(simple_labels)} 处简单增强/NoAug 标注 (需清除):')
        for l in simple_labels:
            print(f'  L{l["line"]}: [{l["pattern"]}] {l["context"][:120]}')

    return len(simple_labels)


def verify_experiment_coverage(conn, doc_path):
    """核对数据库实验是否都在文档中被引用, 反之亦然."""
    doc_refs = extract_workdir_refs(doc_path)
    doc_paths = set(r['path'] for r in doc_refs)

    db_paths = set(get_all_experiments(conn))

    # 文档引用但数据库没有的实验
    doc_only = doc_paths - db_paths
    # 数据库有但文档没有引用的实验
    db_only = db_paths - doc_paths

    print('\n=== 实验覆盖核对 ===')
    print(f'文档引用的 work_dir: {len(doc_paths)} 个')
    print(f'数据库实验: {len(db_paths)} 个')

    if doc_only:
        print(f'\n⚠️ 文档引用但数据库无记录 ({len(doc_only)} 个):')
        for p in sorted(doc_only)[:10]:
            print(f'  {p}')

    if db_only:
        # 过滤掉非主路线实验
        main_route_patterns = ['multi_seed', '24obj_ablation', 'stochot', 'hard_ot', 'rf_heun']
        main_route_db_only = [p for p in db_only if any(pat in p for pat in main_route_patterns)]
        if main_route_db_only:
            print(f'\nℹ️ 数据库有但文档未引用的主路线实验 ({len(main_route_db_only)} 个):')
            for p in sorted(main_route_db_only)[:10]:
                print(f'  {p}')

    return len(doc_only)


def verify_key_experiments(conn):
    """核对关键实验的增强策略一致性."""
    print('\n=== 关键实验增强策略核对 ===')

    key_exps = [
        ('work_dirs/multi_seed_aug/rf_heun_adaln/seed_42', 'D1 Random', 'standard'),
        ('work_dirs/multi_seed_aug/rf_heun_adaln/seed_123', 'D1 Random', 'standard'),
        ('work_dirs/multi_seed_aug/rf_heun_adaln/seed_789', 'D1 Random', 'standard'),
        ('work_dirs/multi_seed_aug/hard_ot/seed_42', 'D1 Hard OT', 'standard'),
        ('work_dirs/multi_seed_aug/hard_ot/seed_123', 'D1 Hard OT', 'standard'),
        ('work_dirs/multi_seed/stochot_eps5_old/seed_42', 'D1 StochOT', 'standard'),
        ('work_dirs/multi_seed/stochot_eps5_old/seed_123', 'D1 StochOT', 'standard'),
        ('work_dirs/multi_seed/stochot_eps5_old/seed_789', 'D1 StochOT', 'standard'),
        ('work_dirs/24obj_ablation/random/seed_42', 'D2 Random', 'standard'),
        ('work_dirs/24obj_ablation/sinkhorn/seed_42', 'D2 StochOT', 'standard'),
        # 旧实验 (应该使用简单增强)
        ('work_dirs/multi_seed/rf_heun_adaln/seed_42', 'D1 Random (旧)', 'simple'),
        ('work_dirs/multi_seed/hard_ot/seed_42', 'D1 Hard OT (旧)', 'simple'),
    ]

    all_ok = True
    for work_dir, name, expected_aug in key_exps:
        row = get_experiment_by_workdir(conn, work_dir)
        if not row:
            print(f'  ❌ {name}: {work_dir} 不在数据库中')
            all_ok = False
            continue

        actual_aug = row[8] if row[8] else 'unknown'
        db_mAP = row[1]
        dataset = row[5]
        coupling = row[6]

        if actual_aug == expected_aug:
            print(f'  ✅ {name}: aug={actual_aug} mAP={db_mAP} dataset={dataset} coupling={coupling}')
        else:
            print(f'  ❌ {name}: 期望 aug={expected_aug}, 实际 aug={actual_aug} | mAP={db_mAP}')
            all_ok = False

    return all_ok


# ─── 主流程 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='文档核对: 对比 SQLite 数据库与文档')
    parser.add_argument('--db', default=DB_PATH, help='SQLite 数据库路径')
    parser.add_argument('--doc', default=os.path.join(_PROJECT_ROOT, 'docs/EXPERIMENT_LINEAGE.md'),
                        help='文档路径')
    parser.add_argument('--check-mAP', action='store_true', help='核对 mAP 值')
    parser.add_argument('--check-augmentation', action='store_true', help='核对增强策略标注')
    parser.add_argument('--check-coverage', action='store_true', help='核对实验覆盖')
    parser.add_argument('--check-key', action='store_true', help='核对关键实验')
    parser.add_argument('--all', action='store_true', help='执行所有核对')
    args = parser.parse_args()

    if args.all or not any([args.check_mAP, args.check_augmentation, args.check_coverage, args.check_key]):
        args.check_mAP = args.check_augmentation = args.check_coverage = args.check_key = True

    conn = sqlite3.connect(args.db)

    print(f'数据库: {args.db}')
    print(f'文档: {args.doc}')

    issues = 0

    if args.check_key:
        if not verify_key_experiments(conn):
            issues += 1

    if args.check_augmentation:
        issues += verify_augmentation(conn, args.doc)

    if args.check_mAP:
        issues += verify_mAP(conn, args.doc)

    if args.check_coverage:
        issues += verify_experiment_coverage(conn, args.doc)

    print(f'\n=== 总结 ===')
    if issues == 0:
        print('✅ 所有核对通过, 无不一致')
    else:
        print(f'❌ 发现 {issues} 类问题, 请查看上方详情')

    conn.close()


if __name__ == '__main__':
    main()
