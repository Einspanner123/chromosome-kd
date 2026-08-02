"""搜寻实验目录的 best mAP（val/test）并核对数据真实性。

用法:
  python tools/find_best_map.py [work_dir1] [work_dir2] ...
  python tools/find_best_map.py --all              # 递归扫描 work_dirs/ 下所有含 ckpt 的目录
  python tools/find_best_map.py --pattern a4_dpm   # 只扫描名称匹配的目录
  python tools/find_best_map.py --pattern random --recursive  # 递归扫描名称匹配的子目录

输出:
  work_dir | best_epoch | best_val_mAP | best_score_line | ckpt_file | log_source

数据来源优先级:
  1. best_coco_bbox_mAP_epoch_*.pth 文件名 → best epoch
  2. train.log / {timestamp}/{timestamp}.log 中 "best score: X.XXX" 行 → best mAP
  3. log 中 "Epoch(val) [{epoch}]" 行 → 该 epoch 的 val mAP（交叉验证）
  4. vis_data/scalars.json 中 coco/bbox_mAP 字段 → 所有 val mAP（备份）

注: --all 现已递归扫描 (发现 24obj_ablation/random/seed_42 等嵌套目录)。
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path


def find_best_ckpt(work_dir: str) -> tuple[int | None, str | None]:
    """从 best_coco_bbox_mAP_epoch_*.pth 文件名提取 best epoch。"""
    pattern = os.path.join(work_dir, 'best_coco_bbox_mAP_epoch_*.pth')
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None, None
    ckpt = ckpts[0]  # 只取第一个（通常只有一个）
    m = re.search(r'best_coco_bbox_mAP_epoch_(\d+)\.pth', os.path.basename(ckpt))
    if m:
        return int(m.group(1)), ckpt
    return None, ckpt


def find_log_files(work_dir: str) -> list[str]:
    """查找所有可能的 log 文件（train.log + timestamp 子目录下的 .log）。"""
    logs = []
    # 根目录 train.log
    train_log = os.path.join(work_dir, 'train.log')
    if os.path.isfile(train_log):
        logs.append(train_log)
    # timestamp 子目录下的 .log (如 20260728_205221/20260728_205221.log)
    for item in os.listdir(work_dir):
        subdir = os.path.join(work_dir, item)
        if os.path.isdir(subdir) and re.match(r'\d{8}_\d{6}', item):
            for f in os.listdir(subdir):
                if f.endswith('.log') and f.startswith(item):
                    logs.append(os.path.join(subdir, f))
    return logs


def extract_best_score_from_log(log_path: str) -> tuple[float | None, str | None]:
    """从 log 中提取 'best score: X.XXX' 行。"""
    try:
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            best_score = None
            best_line = None
            for line in f:
                if 'best score:' in line:
                    m = re.search(r'best score:\s*([\d.]+)', line)
                    if m:
                        best_score = float(m.group(1))
                        best_line = line.strip()
            return best_score, best_line
    except Exception:
        return None, None


def extract_val_mAP_for_epoch(log_path: str, epoch: int) -> tuple[float | None, str | None]:
    """从 log 中提取指定 epoch 的 val mAP。"""
    try:
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 匹配 "Epoch(val) [49][440/440] ... coco/bbox_mAP: 0.7460"
                if f'Epoch(val) [{epoch}][' in line and 'coco/bbox_mAP:' in line:
                    m = re.search(r'coco/bbox_mAP:\s*([\d.]+)', line)
                    if m:
                        return float(m.group(1)), line.strip()[:200]
            return None, None
    except Exception:
        return None, None


def extract_all_val_mAPs_from_log(log_path: str) -> list[tuple[int, float]]:
    """从 log 中提取所有 epoch 的 val mAP，返回 [(epoch, mAP), ...]。"""
    results = []
    try:
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'Epoch(val)' in line and 'coco/bbox_mAP:' in line:
                    ep_m = re.search(r'Epoch\(val\)\s*\[(\d+)\]', line)
                    map_m = re.search(r'coco/bbox_mAP:\s*([\d.]+)', line)
                    if ep_m and map_m:
                        results.append((int(ep_m.group(1)), float(map_m.group(1))))
    except Exception:
        pass
    return results


def extract_from_scalars_json(work_dir: str) -> list[tuple[int, float]]:
    """从 vis_data/scalars.json 提取所有 val mAP（备份来源）。"""
    results = []
    for item in os.listdir(work_dir):
        subdir = os.path.join(work_dir, item)
        if not os.path.isdir(subdir):
            continue
        scalars_path = os.path.join(subdir, 'vis_data', 'scalars.json')
        if not os.path.isfile(scalars_path):
            continue
        try:
            with open(scalars_path, encoding='utf-8') as f:
                step_to_epoch = {}
                # 先收集 train 行的 step→epoch 映射
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get('epoch') is not None:
                            step_to_epoch[d['step']] = d['epoch']
                    except json.JSONDecodeError:
                        continue
                # 再收集 val 行
                f.seek(0)
                current_epoch = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if 'coco/bbox_mAP' in d and d.get('coco/bbox_mAP') is not None:
                            mAP = d['coco/bbox_mAP']
                            if mAP > 0:  # 跳过初始 0 值
                                step = d.get('step', 0)
                                # val 行的 epoch 通常为 None，用最近的 train epoch
                                ep = d.get('epoch') or step_to_epoch.get(step, current_epoch)
                                if d.get('epoch') is not None:
                                    current_epoch = d['epoch']
                                results.append((ep, mAP))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
    return results


def analyze_work_dir(work_dir: str) -> dict:
    """分析单个实验目录，返回 best mAP 信息。"""
    result = {
        'work_dir': work_dir,
        'dir_name': os.path.relpath(work_dir, 'work_dirs') if work_dir.startswith('work_dirs') else os.path.basename(work_dir),
        'best_epoch': None,
        'best_val_mAP': None,
        'best_score_line': None,
        'val_mAP_at_best_epoch': None,
        'ckpt_file': None,
        'log_source': None,
        'all_val_mAPs': [],
        'max_val_mAP': None,
        'max_val_epoch': None,
        'test_mAP': None,
    }

    # 1. best checkpoint → epoch
    best_epoch, ckpt = find_best_ckpt(work_dir)
    result['best_epoch'] = best_epoch
    result['ckpt_file'] = os.path.basename(ckpt) if ckpt else None

    # 2. 查找 log 文件
    log_files = find_log_files(work_dir)

    # 3. 从 log 提取 best score
    for log_path in log_files:
        best_score, best_line = extract_best_score_from_log(log_path)
        if best_score is not None:
            result['best_val_mAP'] = best_score
            result['best_score_line'] = best_line[:150] if best_line else None
            result['log_source'] = os.path.relpath(log_path, work_dir)
            break

    # 4. 从 log 提取 best epoch 的 val mAP（交叉验证）
    if best_epoch is not None:
        for log_path in log_files:
            val_mAP, _ = extract_val_mAP_for_epoch(log_path, best_epoch)
            if val_mAP is not None:
                result['val_mAP_at_best_epoch'] = val_mAP
                break

    # 5. 提取所有 val mAP，找最大值
    all_vals = []
    for log_path in log_files:
        vals = extract_all_val_mAPs_from_log(log_path)
        if vals:
            all_vals.extend(vals)
            if not result['log_source']:
                result['log_source'] = os.path.relpath(log_path, work_dir)

    # 如果 log 中没找到，从 scalars.json 提取
    if not all_vals:
        all_vals = extract_from_scalars_json(work_dir)

    result['all_val_mAPs'] = all_vals
    if all_vals:
        max_ep, max_map = max(all_vals, key=lambda x: x[1])
        result['max_val_mAP'] = max_map
        result['max_val_epoch'] = max_ep

    # 6. 检查 test mAP（在 log 中搜索 test 评估行）
    for log_path in log_files:
        try:
            with open(log_path, encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'test' in line.lower() and 'coco/bbox_mAP:' in line and 'Epoch(val)' not in line:
                        m = re.search(r'coco/bbox_mAP:\s*([\d.]+)', line)
                        if m:
                            result['test_mAP'] = float(m.group(1))
                            break
        except Exception:
            pass

    return result


def print_results(results: list[dict], verbose: bool = False):
    """打印结果表格。"""
    print(f'{"="*130}')
    print(f'{"work_dir":<52} {"best_ep":>7} {"val_mAP":>8} {"best_score":>10} {"val@best_ep":>12} {"max_val":>8} {"max_ep":>7} {"ckpt":<30}')
    print(f'{"="*130}')
    for r in results:
        print(
            f'{r["dir_name"]:<52} '
            f'{str(r["best_epoch"] or "-"):>7} '
            f'{str(r["best_val_mAP"] or "-"):>8} '
            f'{str(r["best_val_mAP"] or "-"):>10} '
            f'{str(r["val_mAP_at_best_epoch"] or "-"):>12} '
            f'{str(r["max_val_mAP"] or "-"):>8} '
            f'{str(r["max_val_epoch"] or "-"):>7} '
            f'{str(r["ckpt_file"] or "-"):<30}'
        )
    print(f'{"="*130}')

    if verbose:
        print()
        for r in results:
            print(f'\n--- {r["dir_name"]} ---')
            print(f'  work_dir:       {r["work_dir"]}')
            print(f'  best_epoch:     {r["best_epoch"]}')
            print(f'  best_val_mAP:   {r["best_val_mAP"]}')
            print(f'  val@best_epoch: {r["val_mAP_at_best_epoch"]}')
            print(f'  max_val_mAP:    {r["max_val_mAP"]} @ ep{r["max_val_epoch"]}')
            print(f'  ckpt:           {r["ckpt_file"]}')
            print(f'  log_source:     {r["log_source"]}')
            print(f'  best_score_line: {r["best_score_line"]}')
            if r['all_val_mAPs']:
                print(f'  all_val_mAPs:   {len(r["all_val_mAPs"])} entries')
                # 显示 top 3
                top3 = sorted(r['all_val_mAPs'], key=lambda x: x[1], reverse=True)[:3]
                for ep, mAP in top3:
                    print(f'    ep{ep}: {mAP}')
            if r['test_mAP']:
                print(f'  test_mAP:       {r["test_mAP"]}')


def find_nested_experiment_dirs(base: str, pattern: str | None = None) -> list[str]:
    """递归查找所有含 best checkpoint 或 log 的实验目录。

    支持嵌套结构如 24obj_ablation/random/seed_42/。
    """
    result = []
    for root, dirs, files in os.walk(base):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        has_ckpt = any(f.startswith('best_coco_bbox_mAP_epoch_') and f.endswith('.pth') for f in files)
        has_log = any(f.endswith('.log') for f in files)
        if has_ckpt or has_log:
            if pattern is None or pattern in root:
                result.append(root)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description='搜寻实验目录的 best mAP')
    parser.add_argument('work_dirs', nargs='*', help='实验目录列表')
    parser.add_argument('--all', action='store_true', help='递归扫描 work_dirs/ 下所有含 ckpt/log 的目录')
    parser.add_argument('--pattern', type=str, default=None, help='只扫描路径中含此字符串的目录')
    parser.add_argument('--recursive', '-r', action='store_true', help='配合 --pattern 递归扫描子目录')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    args = parser.parse_args()

    # 确定要扫描的目录
    if args.all:
        work_dirs = find_nested_experiment_dirs('work_dirs')
    elif args.pattern:
        if args.recursive:
            work_dirs = find_nested_experiment_dirs('work_dirs', args.pattern)
        else:
            base = 'work_dirs'
            work_dirs = sorted([
                os.path.join(base, d) for d in os.listdir(base)
                if os.path.isdir(os.path.join(base, d))
                and args.pattern in d
            ])
    elif args.work_dirs:
        work_dirs = args.work_dirs
    else:
        parser.print_help()
        sys.exit(1)

    # 过滤出有 best checkpoint 或 log 的目录
    valid_dirs = []
    for wd in work_dirs:
        if not os.path.isdir(wd):
            continue
        has_ckpt = glob.glob(os.path.join(wd, 'best_coco_bbox_mAP_epoch_*.pth'))
        has_log = find_log_files(wd)
        if has_ckpt or has_log:
            valid_dirs.append(wd)

    if not valid_dirs:
        print('No valid experiment directories found.')
        sys.exit(1)

    # 分析每个目录
    results = []
    for wd in valid_dirs:
        r = analyze_work_dir(wd)
        results.append(r)

    # 打印结果
    print_results(results, verbose=args.verbose)

    # 检查不一致
    print('\n=== 不一致检查 ===')
    for r in results:
        issues = []
        if r['best_val_mAP'] and r['val_mAP_at_best_epoch']:
            if abs(r['best_val_mAP'] - r['val_mAP_at_best_epoch']) > 0.0005:
                issues.append(
                    f'best_score({r["best_val_mAP"]}) != val@best_ep({r["val_mAP_at_best_epoch"]})'
                )
        if r['best_val_mAP'] and r['max_val_mAP']:
            if abs(r['best_val_mAP'] - r['max_val_mAP']) > 0.0005:
                issues.append(
                    f'best_score({r["best_val_mAP"]}) != max_val({r["max_val_mAP"]}@ep{r["max_val_epoch"]})'
                )
        if r['best_epoch'] and r['max_val_epoch']:
            if r['best_epoch'] != r['max_val_epoch'] and r['best_val_mAP'] and r['max_val_mAP']:
                if r['max_val_mAP'] > r['best_val_mAP'] + 0.0005:
                    issues.append(
                        f'best_ep({r["best_epoch"]}) != max_val_ep({r["max_val_epoch"]}), '
                        f'max_val({r["max_val_mAP"]}) > best_score({r["best_val_mAP"]})'
                    )
        if issues:
            print(f'  ⚠ {r["dir_name"]}: {"; ".join(issues)}')

    if not any(
        r['best_val_mAP'] and r['val_mAP_at_best_epoch']
        and abs(r['best_val_mAP'] - r['val_mAP_at_best_epoch']) > 0.0005
        for r in results
    ):
        print('  (无不一致)')


if __name__ == '__main__':
    main()
