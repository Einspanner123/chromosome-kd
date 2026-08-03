"""文档数据校准: 对比 docs/ 下所有文档与 SQLite 数据库, 逐条核对实验数据.

用法:
  python tools/experiment_db/calibrate_docs.py                    # 校准所有文档
  python tools/experiment_db/calibrate_docs.py --doc EXPERIMENT_LINEAGE.md
  python tools/experiment_db/calibrate_docs.py --report-only       # 只输出报告不修复
  python tools/experiment_db/calibrate_docs.py --verbose

校准内容:
  1. work_dir 引用 → 数据库匹配 (存在/缺失)
  2. mAP 值核对 (文档 vs 数据库, 容差 0.002)
  3. 增强策略核对 (文档标注 vs 数据库实际)
  4. 数据集核对 (文档标注 vs 数据库实际)
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')
DOCS_DIR = os.path.join(_PROJECT_ROOT, 'docs')

# 校准容差
MAP_TOLERANCE = 0.002  # 0.2%


# ─── 文档解析 ───────────────────────────────────────────────

def expand_template_path(path):
    """展开模板路径 work_dirs/.../seed_{42,123,789} 为具体路径列表."""
    m = re.search(r'\{([^}]+)\}', path)
    if not m:
        return [path]
    options = m.group(1).split(',')
    results = []
    for opt in options:
        opt = opt.strip()
        expanded = path[:m.start()] + opt + path[m.end():]
        results.append(expanded)
    return results


def extract_workdir_refs(doc_path):
    """从文档中提取所有 work_dirs/... 路径引用.

    支持模板路径: work_dirs/.../seed_{42,123,789}/

    Returns:
        list[dict]: [{'path': str, 'expanded': list[str], 'line': int, 'context': str}]
    """
    refs = []
    seen = set()

    with open(doc_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 匹配 work_dirs/... 路径
            # 允许 {42,123,789} 模板语法 (不在排除字符中排除 {} 和 ,)
            matches = re.finditer(r'work_dirs/[^\s`)\]}>]+', line)
            for m in matches:
                raw_path = m.group()
                # 清理末尾标点
                raw_path = re.sub(r'[/\.]+$', '', raw_path)
                # 去掉末尾的单个逗号 (模板路径结尾可能带逗号)
                raw_path = re.sub(r',$', '', raw_path)

                # 跳过非实验路径
                skip_patterns = ['.json', '.png', '.log', '/vis_data/',
                                '/LDMDet_backup/', '/experiments_backup/',
                                '/diagnosis/', '<exp_dir', '<timestamp']
                if any(sp in raw_path for sp in skip_patterns):
                    continue
                # 跳过单独的 .pth 文件引用
                if raw_path.endswith('.pth'):
                    continue

                # 展开模板路径
                expanded = expand_template_path(raw_path)

                key = (raw_path, line_num)
                if key in seen:
                    continue
                seen.add(key)

                refs.append({
                    'path': raw_path,
                    'expanded': expanded,
                    'line': line_num,
                    'context': line.strip()[:400],
                })
    return refs


def extract_mAP_near(work_dir_ref):
    """从 work_dir 引用的上下文中提取 mAP 值.

    提取多种格式的 mAP:
      - mAP=0.XXX / mAP: 0.XXX / mAP 0.XXX
      - 0.XXX@epNN  (无空格)
      - 0.XXX @ epNN (有空格)
      - 0.XXX (best...) / 0.XXX (val...
      - **0.XXX**  (加粗)
      - | 0.XXX |  (表格单元格, mAP 列)
    """
    context = work_dir_ref['context']
    mAPs = []
    seen_vals = set()

    def _add(val_str):
        try:
            v = float(val_str)
            if 0.3 < v < 1.0 and v not in seen_vals:  # 合理 mAP 范围 + 去重
                seen_vals.add(v)
                mAPs.append(v)
        except ValueError:
            pass

    # 1. mAP=0.XXX, mAP: 0.XXX, mAP 0.XXX
    for m in re.finditer(r'mAP[=: ]+([\d.]+)', context):
        _add(m.group(1))

    # 2. 0.XXX@epNN  (无空格)
    for m in re.finditer(r'(?<!\d)(0\.\d{3,4})@ep\d+', context):
        _add(m.group(1))

    # 3. 0.XXX @ epNN (有空格)
    for m in re.finditer(r'(?<!\d)(0\.\d{3,4})\s*@\s*ep\d+', context):
        _add(m.group(1))

    # 4. 0.XXX (best... / 0.XXX (val... / 0.XXX (3-seed...
    for m in re.finditer(r'(?<!\d)(0\.\d{3,4})\s*\((best|val|3-seed|max)', context, re.IGNORECASE):
        _add(m.group(1))

    # 5. **0.XXX** (加粗, 紧跟 **)
    for m in re.finditer(r'\*\*(0\.\d{3,4})\*\*', context):
        _add(m.group(1))
    # 5b. **0.XXX ...  (加粗开头, 数字紧跟 ** 后, 如 **0.863 val**)
    for m in re.finditer(r'\*\*(0\.\d{3,4})\s', context):
        _add(m.group(1))

    # 6. | 0.XXX |  (表格单元格 — mAP 列, 紧跟 | 分隔符)
    for m in re.finditer(r'\|\s*\*{0,2}(0\.\d{3,4})\*{0,2}\s*[|（(]', context):
        _add(m.group(1))

    # 7. 0.XXX±0.XXX  (mean±std 格式, 提取 mean)
    for m in re.finditer(r'(?<!\d)(0\.\d{3,4})±', context):
        _add(m.group(1))

    # 8. 0.XXX val / 0.XXX test  (带 val/test 后缀)
    for m in re.finditer(r'(?<!\d)(0\.\d{3,4})\s+(val|test)\b', context, re.IGNORECASE):
        _add(m.group(1))

    return mAPs


def extract_dataset_label(context, work_dir_path=None):
    """从上下文或路径中提取数据集标注.

    优先从 work_dir 路径判断 (更可靠), 其次从上下文判断.
    忽略 "D1 config" / "D2 config" 等指源配置的模式.
    """
    # 1. 路径优先 (最可靠)
    if work_dir_path:
        path_lower = work_dir_path.lower()
        if '24obj' in path_lower or '/d2_' in path_lower or '24_chromosomes' in path_lower:
            return 'D2'
        if 'chr2024' in path_lower or '/d1_' in path_lower or 'chromosome20240904' in path_lower:
            return 'D1'

    # 2. 上下文判断 — 移除 "D1/D2 config" 等源配置引用后再匹配
    cleaned = re.sub(r'[Dd][12]\s*(config|configuration|0\.\d{3})', '', context)
    cleaned = re.sub(r'Dataset\s*[12]\s*(config|configuration)', '', cleaned)

    if re.search(r'Dataset\s*1|chr2024|Chromosome20240904', cleaned):
        return 'D1'
    if re.search(r'Dataset\s*2|24obj|24_chromosomes', cleaned):
        return 'D2'
    # 仅当上下文中独立出现 D1/D2 (非 config 引用) 时才匹配
    if re.search(r'(?<!config )(?<!configuration )\bD1\b', cleaned):
        return 'D1'
    if re.search(r'(?<!config )(?<!configuration )\bD2\b', cleaned):
        return 'D2'
    return None


def extract_aug_label(context):
    """从上下文中提取增强策略标注."""
    if re.search(r'标准增强|standard', context, re.IGNORECASE):
        return 'standard'
    if re.search(r'简单增强|NoAug|无aug|简化设置|simple', context, re.IGNORECASE):
        return 'simple'
    return None


# ─── 数据库查询 ─────────────────────────────────────────────

def query_experiment(conn, work_dir):
    """查询单个实验, 返回完整信息."""
    cursor = conn.cursor()
    # 精确匹配
    cursor.execute('''
        SELECT e.experiment_id, e.best_val_mAP, e.best_val_epoch, e.seed, e.status,
               c.dataset, c.coupling_type, c.ot_epsilon, c.solver_type,
               a.pipeline_name, a.description
        FROM experiment e LEFT JOIN config c ON e.config_path = c.config_path
        LEFT JOIN aug_pipeline a ON c.aug_pipeline_hash = a.pipeline_hash
        WHERE e.experiment_id = ?
    ''', (work_dir,))
    row = cursor.fetchone()
    if row:
        return row
    # 模糊匹配 (路径前缀)
    cursor.execute('''
        SELECT e.experiment_id, e.best_val_mAP, e.best_val_epoch, e.seed, e.status,
               c.dataset, c.coupling_type, c.ot_epsilon, c.solver_type,
               a.pipeline_name, a.description
        FROM experiment e LEFT JOIN config c ON e.config_path = c.config_path
        LEFT JOIN aug_pipeline a ON c.aug_pipeline_hash = a.pipeline_hash
        WHERE e.experiment_id LIKE ?
        LIMIT 3
    ''', (f'{work_dir}%',))
    rows = cursor.fetchall()
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        # 返回最匹配的
        for r in rows:
            if r[0] == work_dir:
                return r
        return rows[0]
    return None


def query_test_mAP(conn, experiment_id):
    """查询实验的 test mAP."""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT mAP, AP50, AP75, epoch, source
        FROM evaluation
        WHERE experiment_id = ? AND split = 'test'
        ORDER BY mAP DESC
        LIMIT 1
    ''', (experiment_id,))
    return cursor.fetchone()


# ─── 校准逻辑 ───────────────────────────────────────────────

def calibrate_document(conn, doc_path, verbose=False):
    """校准单个文档."""
    doc_name = os.path.basename(doc_path)
    refs = extract_workdir_refs(doc_path)

    results = {
        'doc': doc_name,
        'total_refs': len(refs),
        'matched': 0,
        'not_in_db': 0,
        'mAP_verified': 0,
        'mAP_mismatch': [],
        'aug_verified': 0,
        'aug_mismatch': [],
        'dataset_verified': 0,
        'dataset_mismatch': [],
        'missing_details': [],
    }

    seen_experiments = set()

    for ref in refs:
        # 展开模板路径, 逐个匹配
        for exp_path in ref['expanded']:
            if exp_path in seen_experiments:
                continue
            seen_experiments.add(exp_path)

            row = query_experiment(conn, exp_path)

            if not row:
                results['not_in_db'] += 1
                if verbose:
                    results['missing_details'].append({
                        'path': exp_path,
                        'line': ref['line'],
                        'context': ref['context'][:150],
                    })
                continue

            results['matched'] += 1
            exp_id, db_mAP, db_epoch, db_seed, db_status, db_dataset, db_coupling, db_eps, db_solver, db_aug, aug_desc = row

            # mAP 核对
            # 策略: 如果行内任一 mAP 与数据库匹配, 则视为一致;
            #       仅当所有 mAP 都不匹配时才标记不一致
            doc_mAPs = extract_mAP_near(ref)
            if doc_mAPs and db_mAP is not None:
                any_match = False
                worst_mismatch = None
                for doc_mAP in doc_mAPs:
                    diff = abs(doc_mAP - db_mAP)
                    if diff <= MAP_TOLERANCE:
                        any_match = True
                        break
                    elif worst_mismatch is None or diff > worst_mismatch['diff']:
                        worst_mismatch = {
                            'path': exp_id,
                            'line': ref['line'],
                            'doc_mAP': doc_mAP,
                            'db_mAP': db_mAP,
                            'diff': diff,
                            'context': ref['context'][:150],
                        }
                if any_match:
                    results['mAP_verified'] += 1
                elif worst_mismatch:
                    results['mAP_mismatch'].append(worst_mismatch)

            # 增强策略核对
            doc_aug = extract_aug_label(ref['context'])
            if doc_aug and db_aug:
                if doc_aug == db_aug:
                    results['aug_verified'] += 1
                else:
                    results['aug_mismatch'].append({
                        'path': exp_id,
                        'line': ref['line'],
                        'doc_aug': doc_aug,
                        'db_aug': db_aug,
                        'context': ref['context'][:150],
                    })

            # 数据集核对
            doc_dataset = extract_dataset_label(ref['context'], work_dir_path=exp_path)
            if doc_dataset and db_dataset and db_dataset != 'unknown':
                if doc_dataset == db_dataset:
                    results['dataset_verified'] += 1
                else:
                    results['dataset_mismatch'].append({
                        'path': exp_id,
                        'line': ref['line'],
                        'doc_dataset': doc_dataset,
                        'db_dataset': db_dataset,
                        'context': ref['context'][:150],
                    })

    return results


def print_report(results):
    """打印校准报告."""
    doc = results['doc']
    print(f'\n{"=" * 80}')
    print(f'  {doc}')
    print(f'{"=" * 80}')
    print(f'  引用总数: {results["total_refs"]}')
    print(f'  数据库匹配: {results["matched"]}')
    print(f'  数据库无记录: {results["not_in_db"]}')
    print()

    # mAP 核对
    print(f'  mAP 核对: ✅ {results["mAP_verified"]} 一致, ❌ {len(results["mAP_mismatch"])} 不一致')
    if results['mAP_mismatch']:
        for m in results['mAP_mismatch'][:20]:
            print(f'    ❌ L{m["line"]}: {m["path"]}')
            print(f'       文档={m["doc_mAP"]:.4f} vs DB={m["db_mAP"]:.4f} (Δ={m["diff"]:.4f})')
            print(f'       {m["context"][:120]}')

    # 增强策略核对
    print(f'\n  增强策略核对: ✅ {results["aug_verified"]} 一致, ❌ {len(results["aug_mismatch"])} 不一致')
    if results['aug_mismatch']:
        for m in results['aug_mismatch']:
            print(f'    ❌ L{m["line"]}: {m["path"]}')
            print(f'       文档={m["doc_aug"]} vs DB={m["db_aug"]}')
            print(f'       {m["context"][:120]}')

    # 数据集核对
    print(f'\n  数据集核对: ✅ {results["dataset_verified"]} 一致, ❌ {len(results["dataset_mismatch"])} 不一致')
    if results['dataset_mismatch']:
        for m in results['dataset_mismatch']:
            print(f'    ❌ L{m["line"]}: {m["path"]}')
            print(f'       文档={m["doc_dataset"]} vs DB={m["db_dataset"]}')
            print(f'       {m["context"][:120]}')

    # 缺失实验
    if results['missing_details']:
        print(f'\n  数据库无记录的引用 ({len(results["missing_details"])} 个):')
        for m in results['missing_details'][:15]:
            print(f'    ⚠️ L{m["line"]}: {m["path"]}')
            print(f'       {m["context"][:120]}')


# ─── 主流程 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='文档数据校准')
    parser.add_argument('--db', default=DB_PATH, help='SQLite 数据库路径')
    parser.add_argument('--doc', default=None, help='只校准指定文档 (文件名)')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示缺失实验详情')
    parser.add_argument('--json', action='store_true', help='输出 JSON 格式报告')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # 收集文档
    if args.doc:
        doc_paths = [os.path.join(DOCS_DIR, args.doc)]
    else:
        doc_paths = sorted(glob_for_docs())

    print(f'数据库: {args.db}')
    print(f'待校准文档: {len(doc_paths)} 个')

    all_results = []
    for doc_path in doc_paths:
        if not os.path.isfile(doc_path):
            print(f'  [SKIP] {doc_path} 不存在')
            continue
        results = calibrate_document(conn, doc_path, verbose=args.verbose)
        all_results.append(results)
        if not args.json:
            print_report(results)

    # 总结
    if not args.json:
        print(f'\n{"=" * 80}')
        print('  总结')
        print(f'{"=" * 80}')
        total_matched = sum(r['matched'] for r in all_results)
        total_not_in_db = sum(r['not_in_db'] for r in all_results)
        total_mAP_mismatch = sum(len(r['mAP_mismatch']) for r in all_results)
        total_aug_mismatch = sum(len(r['aug_mismatch']) for r in all_results)
        total_ds_mismatch = sum(len(r['dataset_mismatch']) for r in all_results)

        print(f'  总引用: {sum(r["total_refs"] for r in all_results)}')
        print(f'  数据库匹配: {total_matched}')
        print(f'  数据库无记录: {total_not_in_db}')
        print(f'  mAP 不一致: {total_mAP_mismatch}')
        print(f'  增强策略不一致: {total_aug_mismatch}')
        print(f'  数据集不一致: {total_ds_mismatch}')

        if total_mAP_mismatch == 0 and total_aug_mismatch == 0 and total_ds_mismatch == 0:
            print('\n  ✅ 所有可核对数据一致')
        else:
            print(f'\n  ❌ 发现 {total_mAP_mismatch + total_aug_mismatch + total_ds_mismatch} 处不一致')
    else:
        print(json.dumps(all_results, indent=2, ensure_ascii=False))

    conn.close()


def glob_for_docs():
    """获取 docs/ 下所有 .md 文件."""
    import glob
    return glob.glob(os.path.join(DOCS_DIR, '*.md'))


if __name__ == '__main__':
    main()
