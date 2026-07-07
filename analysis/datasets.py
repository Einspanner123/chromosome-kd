"""数据集配置 — 统一管理数据集别名与路径的对应关系。

所有分析脚本通过 --dataset 别名从此模块获取数据集路径。
数据集必须为 COCO 格式 (train/valid/test 三个 split, 每个 split 下有
_annotations.coco.json), 否则报错。

Usage:
    from datasets import get_dataset_path, get_output_dir

    root = get_dataset_path('24obj')        # Path('data/24_chromosomes_object/coco/')
    out  = get_output_dir('24obj', 'overlap')  # Path('analysis/24obj/overlap/')
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ──────────────────────────────────────────────
# 数据集字典: 别名 → 路径
# ──────────────────────────────────────────────

DATASETS: dict[str, dict] = {
    '24obj': {
        'root': 'data/24_chromosomes_object/coco/',
        'splits': ['train', 'valid', 'test'],
    },
    'autokary': {
        'root': 'data/AutoKary2022_v1_coco/',
        'splits': ['train', 'valid', 'test'],
    },
}

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────


def get_dataset_path(alias: str) -> Path:
    """根据别名获取数据集根路径, 并验证 COCO 格式。

    Args:
        alias: 数据集别名 (如 '24obj', 'autokary')

    Returns:
        数据集根目录 Path

    Raises:
        ValueError: 别名不存在
        FileNotFoundError: COCO 标注文件缺失 (非 COCO 格式)
    """
    if alias not in DATASETS:
        available = ', '.join(DATASETS.keys())
        raise ValueError(
            f"Unknown dataset alias '{alias}'. Available: {available}"
        )

    info = DATASETS[alias]
    root = Path(info['root'])

    # 验证 COCO 格式: 每个 split 必须有 _annotations.coco.json
    for split in info['splits']:
        ann_file = root / split / '_annotations.coco.json'
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Dataset '{alias}' is not valid COCO format: "
                f"annotation file not found: {ann_file}"
            )

    return root


def get_splits(alias: str) -> list[str]:
    """获取数据集的 split 列表。"""
    if alias not in DATASETS:
        raise ValueError(f"Unknown dataset alias '{alias}'")
    return DATASETS[alias]['splits']


def get_ann_file(alias: str, split: str) -> Path:
    """获取指定 split 的 COCO 标注文件路径。"""
    root = get_dataset_path(alias)
    return root / split / '_annotations.coco.json'


def get_img_dir(alias: str, split: str) -> Path:
    """获取指定 split 的图片目录路径。"""
    root = get_dataset_path(alias)
    return root / split


def get_output_dir(alias: str, feature: str) -> Path:
    """获取输出目录: analysis/{alias}/{feature}/

    自动创建目录 (含父目录)。

    Args:
        alias: 数据集别名
        feature: 功能名 (如 'overlap', 'problematic', 'split')

    Returns:
        输出目录 Path
    """
    out_dir = Path(__file__).parent / alias / feature
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def add_dataset_arg(parser: argparse.ArgumentParser) -> None:
    """为 argparse 添加 --dataset 参数。"""
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help='数据集别名',
    )


# ──────────────────────────────────────────────
# CLI 入口: 列出所有数据集
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print('Available datasets:')
    for alias, info in DATASETS.items():
        root = Path(info['root'])
        print(f'  {alias:12s} → {root}')
        for split in info['splits']:
            ann = root / split / '_annotations.coco.json'
            status = '✓' if ann.exists() else '✗ (missing)'
            print(f'    {split:8s} {status}  {ann}')
