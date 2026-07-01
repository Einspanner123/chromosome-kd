"""Few-shot 数据集工具

跨数据集迁移少样本 benchmark 的核心工具:
1. generate_few_shot_coco: 从 COCO 标注生成 k-shot 子集
2. remap_categories: 24obj ↔ chromo 类别 ID 映射
3. get_class_mapping: 获取两个数据集的类别对齐映射
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def get_class_mapping() -> Dict[int, int]:
    """获取 24obj → chromo 的类别 ID 映射

    24obj C 组顺序: C6(6), C7(7), C8(8), C9(9), C10(10), C11(11), C12(12)
    chromo C 组顺序: C10(6), C11(7), C12(8), C6(9), C7(10), C8(11), C9(12)

    Returns:
        Dict[24obj_cat_id, chromo_cat_id]
    """
    # 24obj name → id
    obj_name_to_id = {
        'A1': 1, 'A2': 2, 'A3': 3, 'B4': 4, 'B5': 5,
        'C6': 6, 'C7': 7, 'C8': 8, 'C9': 9,
        'C10': 10, 'C11': 11, 'C12': 12,
        'D13': 13, 'D14': 14, 'D15': 15,
        'E16': 16, 'E17': 17, 'E18': 18,
        'F19': 19, 'F20': 20, 'G21': 21, 'G22': 22, 'X': 23, 'Y': 24,
    }
    # chromo name → id
    chromo_name_to_id = {
        'A1': 1, 'A2': 2, 'A3': 3, 'B4': 4, 'B5': 5,
        'C10': 6, 'C11': 7, 'C12': 8, 'C6': 9, 'C7': 10, 'C8': 11, 'C9': 12,
        'D13': 13, 'D14': 14, 'D15': 15,
        'E16': 16, 'E17': 17, 'E18': 18,
        'F19': 19, 'F20': 20, 'G21': 21, 'G22': 22, 'X': 23, 'Y': 24,
    }

    # 24obj cat_id → chromo cat_id (by name)
    mapping = {}
    for name, obj_id in obj_name_to_id.items():
        chromo_id = chromo_name_to_id[name]
        mapping[obj_id] = chromo_id
    return mapping


def remap_annotations(
    annotations: List[dict],
    source_to_target: Dict[int, int],
) -> List[dict]:
    """将标注的 category_id 从 source 映射到 target

    Args:
        annotations: COCO 标注列表
        source_to_target: source cat_id → target cat_id 映射

    Returns:
        映射后的标注列表
    """
    remapped = []
    for ann in annotations:
        ann = dict(ann)  # 浅拷贝
        old_cat = ann['category_id']
        if old_cat in source_to_target:
            ann['category_id'] = source_to_target[old_cat]
        remapped.append(ann)
    return remapped


def generate_few_shot_coco(
    source_json: str,
    output_json: str,
    k: int,
    seed: int = 42,
) -> dict:
    """从 COCO 标注文件生成 k-shot 子集

    策略: 每类随机选 k 个标注, 包含这些标注的所有图片和标注
    (一张图可能包含多个类别的标注, 所以图片数 ≤ 24*k)

    Args:
        source_json: 源 COCO 标注文件路径
        output_json: 输出 COCO 标注文件路径
        k: 每类采样的标注数
        seed: 随机种子

    Returns:
        生成的 COCO 数据字典
    """
    random.seed(seed)

    with open(source_json) as f:
        data = json.load(f)

    # 按 category_id 分组标注
    cat_to_anns = defaultdict(list)
    for ann in data['annotations']:
        cat_to_anns[ann['category_id']].append(ann)

    # 每类采样 k 个标注
    selected_ann_ids = set()
    for cat_id, anns in cat_to_anns.items():
        if len(anns) <= k:
            selected_ann_ids.update(a['id'] for a in anns)
        else:
            sampled = random.sample(anns, k)
            selected_ann_ids.update(a['id'] for a in sampled)

    # 找到包含这些标注的图片
    selected_img_ids = set()
    selected_anns = []
    for ann in data['annotations']:
        if ann['id'] in selected_ann_ids:
            selected_img_ids.add(ann['image_id'])
            selected_anns.append(ann)

    # 过滤图片
    selected_imgs = [img for img in data['images'] if img['id'] in selected_img_ids]

    # 构建输出
    few_shot_data = {
        'images': selected_imgs,
        'annotations': selected_anns,
        'categories': data['categories'],
    }

    # 写入文件
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(few_shot_data, f)

    print(
        f'Few-shot k={k}: {len(selected_imgs)} imgs, '
        f'{len(selected_anns)} anns, '
        f'{len(few_shot_data["categories"])} cats'
    )

    return few_shot_data


def prepare_few_shot_data(
    data_root: str,
    k_values: Tuple[int, ...] = (5, 10),
    seed: int = 42,
):
    """预生成所有 k-shot COCO 标注文件

    Args:
        data_root: chromo 数据集根目录
        k_values: k-shot 值列表
        seed: 随机种子
    """
    source_json = os.path.join(data_root, 'train', '_annotations.coco.json')
    for k in k_values:
        output_json = os.path.join(data_root, 'train', f'few_shot_k{k}.json')
        if not os.path.exists(output_json):
            generate_few_shot_coco(source_json, output_json, k, seed)
        else:
            print(f'k={k} few-shot data already exists: {output_json}')
