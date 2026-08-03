"""SwanLab 同步: 拉取云端实验元数据, 匹配本地实验, 补充缺失数据.

用法:
  python tools/experiment_db/swanlab_sync.py                    # 同步所有项目
  python tools/experiment_db/swanlab_sync.py --project ldmdet-ablation  -- 只同步指定项目
  python tools/experiment_db/swanlab_sync.py --verbose           -- 详细输出

依赖: swanlab (conda activate chromo)
"""

import argparse
import glob
import os
import re
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.db')


# ─── 本地 SwanLab ID 提取 ──────────────────────────────────

def extract_swanlab_run_ids(work_dir):
    """从 work_dir 中提取 SwanLab run_id.

    来源:
      1. .swanlab_id 文件
      2. vis_data/run-<timestamp>-<run_id> 目录名
    """
    run_ids = set()
    abs_dir = os.path.join(_PROJECT_ROOT, work_dir)

    # 1. .swanlab_id 文件
    id_file = os.path.join(abs_dir, '.swanlab_id')
    if os.path.isfile(id_file):
        with open(id_file, 'r') as f:
            rid = f.read().strip()
            if rid:
                run_ids.add(rid)

    # 2. vis_data/run-* 目录
    run_dirs = glob.glob(os.path.join(abs_dir, '**', 'vis_data', 'run-*'), recursive=True)
    for rd in run_dirs:
        basename = os.path.basename(rd)
        # 格式: run-<timestamp>-<run_id>
        parts = basename.split('-')
        if len(parts) >= 3:
            run_id = parts[-1]
            run_ids.add(run_id)

    return run_ids


def build_local_run_id_map():
    """构建 work_dir → run_id 映射."""
    mapping = {}
    work_dirs_root = os.path.join(_PROJECT_ROOT, 'work_dirs')
    for root, dirs, _ in os.walk(work_dirs_root):
        dirs[:] = [d for d in dirs if d not in ('vis_data', '__pycache__')]
        # 检查是否有 .swanlab_id 或 vis_data/run-*
        rel_path = os.path.relpath(root, _PROJECT_ROOT)
        run_ids = extract_swanlab_run_ids(rel_path)
        if run_ids:
            mapping[rel_path] = run_ids
    return mapping


# ─── SwanLab API 同步 ──────────────────────────────────────

def sync_swanlab(conn, api, project_filter=None, verbose=False):
    """同步 SwanLab 实验到数据库.

    策略:
      1. 构建本地 work_dir → run_id 映射
      2. 遍历所有 SwanLab 项目和实验
      3. 匹配本地实验, 更新 swanlab_project 和 swanlab_run_id
      4. 云端 only 实验也记录到数据库
    """
    # 1. 构建本地映射
    local_map = build_local_run_id_map()
    # 反转: run_id → work_dir
    run_id_to_workdir = {}
    for work_dir, run_ids in local_map.items():
        for rid in run_ids:
            run_id_to_workdir[rid] = work_dir

    if verbose:
        print(f'本地 run_id 映射: {len(run_id_to_workdir)} 个')

    # 2. 列出所有项目
    projects_resp = api.list_projects()
    projects = projects_resp.data

    if project_filter:
        projects = [p for p in projects if project_filter in p.name]

    matched = 0
    cloud_only = 0

    for project in projects:
        project_name = project.name
        if verbose:
            print(f'\n项目: {project_name}')

        # 3. 列出项目中的实验
        try:
            exps_resp = api.list_project_exps(project_name)
            experiments = exps_resp.data
        except Exception as e:
            print(f'  [WARN] 无法列出 {project_name} 的实验: {e}')
            continue

        for exp in experiments:
            run_id = exp.cuid
            exp_name = exp.name
            state = getattr(exp, 'state', 'UNKNOWN')
            description = getattr(exp, 'description', '') or ''
            created_at = getattr(exp, 'createdAt', '') or ''
            finished_at = getattr(exp, 'finishedAt', '') or ''

            # 4. 匹配本地实验
            work_dir = run_id_to_workdir.get(run_id)

            if work_dir:
                # 更新本地实验的 SwanLab 信息
                conn.execute('''
                    UPDATE experiment SET
                        swanlab_project = ?,
                        swanlab_run_id = ?,
                        notes = CASE WHEN notes IS NULL OR notes = '' THEN ? ELSE notes END
                    WHERE experiment_id = ?
                ''', (project_name, run_id, description, work_dir))
                matched += 1
                if verbose:
                    print(f'  ✅ {exp_name} → {work_dir} ({state})')
            else:
                # 云端 only 实验, 记录到数据库
                cloud_exp_id = f'swanlab/{project_name}/{exp_name}'
                # 检查是否已存在
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM experiment WHERE experiment_id = ?', (cloud_exp_id,))
                if not cursor.fetchone():
                    conn.execute('''
                        INSERT OR IGNORE INTO experiment
                            (experiment_id, name, work_dir, server, config_path,
                             swanlab_project, swanlab_run_id, seed, status, notes)
                        VALUES (?, ?, ?, ?, NULL, ?, ?, NULL, ?, ?)
                    ''', (
                        cloud_exp_id,
                        exp_name,
                        f'swanlab://{project_name}/{exp_name}',
                        'swanlab',
                        project_name,
                        run_id,
                        state.lower(),
                        f'云端实验: {description}',
                    ))
                    cloud_only += 1
                    if verbose:
                        print(f'  ☁️  {exp_name} (cloud-only, {state})')

    conn.commit()

    print(f'\n=== SwanLab 同步完成 ===')
    print(f'匹配本地实验: {matched}')
    print(f'云端 only 实验: {cloud_only}')

    return matched, cloud_only


def update_local_swanlab_ids(conn):
    """扫描本地 work_dirs, 提取 SwanLab run_id, 更新数据库."""
    work_dirs_root = os.path.join(_PROJECT_ROOT, 'work_dirs')
    updated = 0

    for root, dirs, _ in os.walk(work_dirs_root):
        dirs[:] = [d for d in dirs if d not in ('vis_data', '__pycache__')]
        rel_path = os.path.relpath(root, _PROJECT_ROOT)
        run_ids = extract_swanlab_run_ids(rel_path)

        if run_ids:
            # 取第一个 run_id (通常只有一个)
            run_id = list(run_ids)[0]
            cursor = conn.cursor()
            cursor.execute('UPDATE experiment SET swanlab_run_id = ? WHERE experiment_id = ? AND swanlab_run_id IS NULL',
                          (run_id, rel_path))
            if cursor.rowcount > 0:
                updated += 1

    conn.commit()
    print(f'本地 SwanLab run_id 更新: {updated} 个实验')
    return updated


def main():
    parser = argparse.ArgumentParser(description='SwanLab 实验同步')
    parser.add_argument('--db', default=DB_PATH, help='SQLite 数据库路径')
    parser.add_argument('--project', default=None, help='只同步指定项目 (子串匹配)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--skip-sync', action='store_true', help='只更新本地 run_id, 不同步云端')
    args = parser.parse_args()

    import swanlab
    api = swanlab.OpenApi()

    conn = sqlite3.connect(args.db)

    # 1. 更新本地 SwanLab run_id
    print('=== 步骤 1: 更新本地 SwanLab run_id ===')
    update_local_swanlab_ids(conn)

    if not args.skip_sync:
        # 2. 同步云端实验
        print('\n=== 步骤 2: 同步 SwanLab 云端实验 ===')
        sync_swanlab(conn, api, project_filter=args.project, verbose=args.verbose)

    # 3. 统计
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM experiment WHERE swanlab_run_id IS NOT NULL')
    total_with_swanlab = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM experiment WHERE server = "swanlab"')
    total_cloud_only = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM experiment')
    total = cursor.fetchone()[0]

    print(f'\n=== 数据库统计 ===')
    print(f'总实验: {total}')
    print(f'有 SwanLab run_id: {total_with_swanlab}')
    print(f'云端 only: {total_cloud_only}')

    conn.close()


if __name__ == '__main__':
    main()
