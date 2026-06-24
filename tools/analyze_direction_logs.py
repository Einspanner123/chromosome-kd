"""分析 5 个突破方向的实验日志.

用法:
    python tools/analyze_direction_logs.py
"""
import json
import os
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent / 'work_dirs'

DIRECTIONS = [
    ('方向一 非平衡OT', 'unbalanced_ghss'),
    ('方向二 计数先验', 'count_prior'),
    ('方向三 SNR匹配', 'snr_matching'),
    ('方向四 非线性轨迹', 'nonlinear_trajectory'),
    ('方向五 分层分类', 'hierarchical_classification'),
]


def load_scalars(run_dir: Path):
    """加载一个 run 目录下的 scalars.json, 返回所有条目列表."""
    scalars_path = run_dir / 'vis_data' / 'scalars.json'
    if not scalars_path.exists():
        return None
    entries = []
    with open(scalars_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def find_latest_run_with_scalars(direction_dir: Path):
    """找到最新的包含 scalars.json 的 run 目录."""
    runs = sorted([d for d in direction_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    for run in reversed(runs):
        scalars = load_scalars(run)
        if scalars:
            return run, scalars
    return None, None


def analyze_direction(name: str, dirname: str):
    """分析单个方向."""
    print(f'\n{"=" * 70}')
    print(f'  {name}  ({dirname})')
    print(f'{"=" * 70}')

    direction_dir = WORK_DIR / dirname
    if not direction_dir.exists():
        print('  [!] 目录不存在')
        return

    # 列出所有 run
    runs = sorted([d for d in direction_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    print(f'  Run 数量: {len(runs)}')
    for run in runs:
        scalars_path = run / 'vis_data' / 'scalars.json'
        has_scalars = scalars_path.exists()
        log_path = run / f'{run.name}.log'
        log_size = log_path.stat().st_size if log_path.exists() else 0
        print(f'    - {run.name}: scalars={"Y" if has_scalars else "N"}, log={log_size}B')

    # 检查 checkpoint
    ckpts = sorted(direction_dir.glob('*.pth'))
    best_ckpt = [c for c in ckpts if 'best' in c.name]
    last_ckpt_file = direction_dir / 'last_checkpoint'
    print(f'  Checkpoint: {len(ckpts)} 个')
    if best_ckpt:
        print(f'    Best: {best_ckpt[0].name}')
    if last_ckpt_file.exists():
        print(f'    Last: {last_ckpt_file.read_text().strip()}')

    # 找最新有 scalars 的 run
    run, scalars = find_latest_run_with_scalars(direction_dir)
    if not scalars:
        print('  [!] 无 scalars.json — 训练可能未完成或崩溃')
        # 检查 log 末尾
        if runs:
            log_path = runs[-1] / f'{runs[-1].name}.log'
            if log_path.exists():
                print(f'  --- 日志末尾 ({log_path.name}) ---')
                with open(log_path) as f:
                    lines = f.readlines()
                for line in lines[-15:]:
                    print(f'    {line.rstrip()}')
        return

    print(f'\n  分析 Run: {run.name}')
    print(f'  日志条目总数: {len(scalars)}')

    # 分离 train 和 val
    val_entries = [e for e in scalars if 'coco/bbox_mAP' in e]
    train_entries = [e for e in scalars if 'coco/bbox_mAP' not in e and 'loss' in e]
    print(f'  Train 条目: {len(train_entries)}, Val 条目: {len(val_entries)}')

    if not val_entries:
        print('  [!] 无验证条目')
        return

    # mAP 分析
    maps = [(e.get('step', i), e['coco/bbox_mAP'], e) for i, e in enumerate(val_entries)]
    best_step, best_map, best_entry = max(maps, key=lambda x: x[1])
    last_step, last_map, last_entry = maps[-1]
    first_step, first_map, _ = maps[0]

    print(f'\n  --- mAP 概览 ---')
    print(f'  首次验证: step={first_step}, mAP={first_map:.4f}')
    print(f'  最佳:     step={best_step}, mAP={best_map:.4f}')
    print(f'  最后:     step={last_step}, mAP={last_map:.4f}')

    # 最佳点的详细指标
    print(f'\n  --- 最佳 step={best_step} 详细指标 ---')
    for k, v in sorted(best_entry.items()):
        if k.startswith('coco/'):
            print(f'    {k}: {v:.4f}')

    # mAP 轨迹 (每 10 个 val 取一个)
    print(f'\n  --- mAP 轨迹 (采样) ---')
    n = len(maps)
    sample_indices = list(range(0, n, max(1, n // 10)))
    if sample_indices[-1] != n - 1:
        sample_indices.append(n - 1)
    for idx in sample_indices:
        step, m, e = maps[idx]
        ap50 = e.get('coco/bbox_mAP_50', 0)
        ap75 = e.get('coco/bbox_mAP_75', 0)
        print(f'    val#{idx:3d} step={step:4d}: mAP={m:.4f} AP50={ap50:.4f} AP75={ap75:.4f}')

    # 训练损失分析
    if train_entries:
        print(f'\n  --- 训练损失 ---')
        first_loss = train_entries[0].get('loss', None)
        last_loss = train_entries[-1].get('loss', None)
        min_loss = min(e.get('loss', 1e9) for e in train_entries)
        print(f'    初始 loss: {first_loss:.4f}')
        print(f'    最小 loss: {min_loss:.4f}')
        print(f'    最后 loss: {last_loss:.4f}')

        # 方向特定损失
        special_losses = set()
        for e in train_entries:
            for k in e.keys():
                if k.startswith('loss_') and k not in ('loss', 'loss_cls', 'loss_bbox', 'loss_giou'):
                    special_losses.add(k)
        if special_losses:
            print(f'    方向特定损失项: {sorted(special_losses)}')
            for sl in sorted(special_losses):
                vals = [e[sl] for e in train_entries if sl in e]
                if vals:
                    print(f'      {sl}: 初始={vals[0]:.4f}, 最小={min(vals):.4f}, 最后={vals[-1]:.4f}')


def main():
    print('=' * 70)
    print('  LDMDet 5 个突破方向实验日志分析')
    print(f'  Work dir: {WORK_DIR}')
    print('=' * 70)
    for name, dirname in DIRECTIONS:
        analyze_direction(name, dirname)
    print(f'\n{"=" * 70}')
    print('  分析完成')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
