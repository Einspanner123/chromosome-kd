"""BBox质量评估指标

包含:
  - 类别分布匹配: 生成vs真实的类别频率分布 (KL散度, JS散度)
  - 尺寸分布匹配: 生成vs真实的bbox尺寸分布 (Wasserstein距离)
  - 数量分布匹配: 生成vs真实的每图目标数分布
  - IoU质量: 生成bbox与LDMDet伪标注的IoU分布
  - 空间分布匹配: 生成vs真实的bbox中心点分布
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

# 添加项目根目录以导入共享常量
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))
from projects.ChromoGen.constants import NUM_CLASSES


def load_coco_annotations(ann_path: str) -> Dict:
    """加载COCO标注"""
    with open(ann_path) as f:
        return json.load(f)


def load_generation_info(gen_path: str) -> List[Dict]:
    """加载生成结果信息"""
    with open(gen_path) as f:
        return json.load(f)


# ============================================================
# 类别分布匹配
# ============================================================


def compute_class_distribution(
    real_ann_path: str,
    gen_info_path: str,
    score_threshold: float = 0.3,
) -> Dict[str, float]:
    """计算类别分布匹配度

    Args:
        real_ann_path: 真实COCO标注路径
        gen_info_path: 生成结果信息路径
        score_threshold: bbox置信度阈值
    Returns:
        dict with 'kl_divergence', 'js_divergence', 'class_counts_real', 'class_counts_gen'
    """
    # 真实分布
    real_coco = load_coco_annotations(real_ann_path)
    real_counts = np.zeros(NUM_CLASSES)
    for ann in real_coco['annotations']:
        cat_idx = ann['category_id'] - 1  # COCO从1开始
        if 0 <= cat_idx < NUM_CLASSES:
            real_counts[cat_idx] += 1

    # 生成分布
    gen_data = load_generation_info(gen_info_path)
    gen_counts = np.zeros(NUM_CLASSES)
    for item in gen_data:
        if 'labels' in item and 'scores' in item:
            for label, score in zip(item['labels'], item['scores']):
                if score >= score_threshold and 0 <= label < NUM_CLASSES:
                    gen_counts[label] += 1

    # 归一化为概率分布
    real_prob = real_counts / (real_counts.sum() + 1e-8)
    gen_prob = gen_counts / (gen_counts.sum() + 1e-8)

    # KL散度
    kl_div = _kl_divergence(real_prob, gen_prob)

    # JS散度
    js_div = _js_divergence(real_prob, gen_prob)

    return {
        'kl_divergence': float(kl_div),
        'js_divergence': float(js_div),
        'class_counts_real': real_counts.astype(int).tolist(),
        'class_counts_gen': gen_counts.astype(int).tolist(),
        'class_prob_real': real_prob.tolist(),
        'class_prob_gen': gen_prob.tolist(),
    }


# ============================================================
# 尺寸分布匹配
# ============================================================


def compute_size_distribution(
    real_ann_path: str,
    gen_info_path: str,
    score_threshold: float = 0.3,
) -> Dict[str, float]:
    """计算bbox尺寸分布匹配度

    将bbox按面积分为小/中/大三档，比较分布差异。

    Args:
        real_ann_path: 真实COCO标注路径
        gen_info_path: 生成结果信息路径
        score_threshold: bbox置信度阈值
    Returns:
        dict with 'wasserstein_width', 'wasserstein_height', 'size_category_match'
    """
    from scipy.stats import wasserstein_distance

    # 真实尺寸
    real_coco = load_coco_annotations(real_ann_path)
    real_widths = []
    real_heights = []
    for ann in real_coco['annotations']:
        _, _, w, h = ann['bbox']
        # 归一化到图像尺寸
        img = next(
            img for img in real_coco['images'] if img['id'] == ann['image_id']
        )
        real_widths.append(w / img['width'])
        real_heights.append(h / img['height'])

    # 生成尺寸
    gen_data = load_generation_info(gen_info_path)
    gen_widths = []
    gen_heights = []
    for item in gen_data:
        if 'bboxes' in item and 'scores' in item:
            for bbox, score in zip(item['bboxes'], item['scores']):
                if score >= score_threshold:
                    _, _, w, h = bbox  # cxcywh归一化
                    gen_widths.append(w)
                    gen_heights.append(h)

    if not gen_widths or not real_widths:
        return {
            'wasserstein_width': -1.0,
            'wasserstein_height': -1.0,
            'size_category_match': -1.0,
        }

    # Wasserstein距离
    w_dist_width = wasserstein_distance(real_widths, gen_widths)
    w_dist_height = wasserstein_distance(real_heights, gen_heights)

    # 按COCO标准分档: small<32², medium<96², large>=96² (归一化后)
    # 假设768×768, 阈值: 32/768=0.042, 96/768=0.125
    small_thresh = 32 / 768
    medium_thresh = 96 / 768

    def size_category(widths, heights):
        areas = np.array(widths) * np.array(heights)
        cats = np.zeros(3)
        cats[0] = (areas < small_thresh**2).sum()
        cats[1] = (
            (areas >= small_thresh**2) & (areas < medium_thresh**2)
        ).sum()
        cats[2] = (areas >= medium_thresh**2).sum()
        return cats / (cats.sum() + 1e-8)

    real_cats = size_category(real_widths, real_heights)
    gen_cats = size_category(gen_widths, gen_heights)

    # 分类匹配度 (1 - JS散度)
    size_match = 1.0 - _js_divergence(real_cats, gen_cats)

    return {
        'wasserstein_width': float(w_dist_width),
        'wasserstein_height': float(w_dist_height),
        'size_category_match': float(size_match),
        'size_dist_real': {
            'small': float(real_cats[0]),
            'medium': float(real_cats[1]),
            'large': float(real_cats[2]),
        },
        'size_dist_gen': {
            'small': float(gen_cats[0]),
            'medium': float(gen_cats[1]),
            'large': float(gen_cats[2]),
        },
    }


# ============================================================
# 数量分布匹配
# ============================================================


def compute_count_distribution(
    real_ann_path: str,
    gen_info_path: str,
    score_threshold: float = 0.3,
) -> Dict[str, float]:
    """计算每图目标数量分布匹配度

    Args:
        real_ann_path: 真实COCO标注路径
        gen_info_path: 生成结果信息路径
        score_threshold: bbox置信度阈值
    Returns:
        dict with 'mean_count_real', 'mean_count_gen', 'wasserstein_count', 'count_histogram'
    """
    from scipy.stats import wasserstein_distance

    # 真实数量
    real_coco = load_coco_annotations(real_ann_path)
    real_counts_per_img = Counter()
    for ann in real_coco['annotations']:
        real_counts_per_img[ann['image_id']] += 1
    real_num_objs = list(real_counts_per_img.values())

    # 生成数量
    gen_data = load_generation_info(gen_info_path)
    gen_num_objs = []
    for item in gen_data:
        if 'labels' in item and 'scores' in item:
            n = sum(1 for s in item['scores'] if s >= score_threshold)
            gen_num_objs.append(n)
        else:
            gen_num_objs.append(0)

    if not gen_num_objs:
        return {
            'mean_count_real': -1,
            'mean_count_gen': -1,
            'wasserstein_count': -1,
        }

    w_dist = wasserstein_distance(real_num_objs, gen_num_objs)

    return {
        'mean_count_real': float(np.mean(real_num_objs)),
        'mean_count_gen': float(np.mean(gen_num_objs)),
        'std_count_real': float(np.std(real_num_objs)),
        'std_count_gen': float(np.std(gen_num_objs)),
        'wasserstein_count': float(w_dist),
    }


# ============================================================
# IoU质量评估
# ============================================================


def compute_iou_quality(
    gen_info_path: str,
    pseudo_ann_path: Optional[str] = None,
    score_threshold: float = 0.3,
) -> Dict[str, float]:
    """计算生成bbox的质量

    如果有LDMDet伪标注，计算生成bbox与伪标注的IoU。
    否则计算生成bbox的置信度分布。

    Args:
        gen_info_path: 生成结果信息路径
        pseudo_ann_path: LDMDet伪标注路径 (可选)
        score_threshold: bbox置信度阈值
    Returns:
        dict with 'mean_score', 'score_std', 'high_conf_ratio'
    """
    gen_data = load_generation_info(gen_info_path)

    all_scores = []
    for item in gen_data:
        if 'scores' in item:
            all_scores.extend(item['scores'])

    if not all_scores:
        return {'mean_score': -1, 'score_std': -1, 'high_conf_ratio': -1}

    scores = np.array(all_scores)

    result = {
        'mean_score': float(scores.mean()),
        'score_std': float(scores.std()),
        'median_score': float(np.median(scores)),
        'high_conf_ratio': float((scores >= 0.7).mean()),
        'low_conf_ratio': float((scores < score_threshold).mean()),
    }

    # 如果有伪标注，计算IoU
    if pseudo_ann_path is not None:
        pseudo_coco = load_coco_annotations(pseudo_ann_path)
        # TODO: 实现生成bbox与伪标注的匹配和IoU计算
        # 这需要更复杂的匹配逻辑，暂时跳过
        result['pseudo_ann_available'] = True

    return result


# ============================================================
# 空间分布匹配
# ============================================================


def compute_spatial_distribution(
    real_ann_path: str,
    gen_info_path: str,
    score_threshold: float = 0.3,
    grid_size: int = 8,
) -> Dict[str, float]:
    """计算bbox中心点空间分布匹配度

    将图像划分为grid_size×grid_size网格，统计各网格中bbox中心点的频率，
    比较生成vs真实的分布差异。

    Args:
        real_ann_path: 真实COCO标注路径
        gen_info_path: 生成结果信息路径
        score_threshold: bbox置信度阈值
        grid_size: 网格大小
    Returns:
        dict with 'spatial_js_divergence', 'spatial_kl_divergence'
    """
    # 真实空间分布
    real_coco = load_coco_annotations(real_ann_path)
    real_grid = np.zeros(grid_size * grid_size)
    for ann in real_coco['annotations']:
        img = next(
            img for img in real_coco['images'] if img['id'] == ann['image_id']
        )
        x, y, w, h = ann['bbox']
        cx = (x + w / 2) / img['width']
        cy = (y + h / 2) / img['height']
        gx = min(int(cx * grid_size), grid_size - 1)
        gy = min(int(cy * grid_size), grid_size - 1)
        real_grid[gy * grid_size + gx] += 1

    # 生成空间分布
    gen_data = load_generation_info(gen_info_path)
    gen_grid = np.zeros(grid_size * grid_size)
    for item in gen_data:
        if 'bboxes' in item and 'scores' in item:
            for bbox, score in zip(item['bboxes'], item['scores']):
                if score >= score_threshold:
                    cx, cy, _, _ = bbox  # cxcywh归一化
                    gx = min(int(cx * grid_size), grid_size - 1)
                    gy = min(int(cy * grid_size), grid_size - 1)
                    gen_grid[gy * grid_size + gx] += 1

    # 归一化
    real_prob = real_grid / (real_grid.sum() + 1e-8)
    gen_prob = gen_grid / (gen_grid.sum() + 1e-8)

    return {
        'spatial_kl_divergence': float(_kl_divergence(real_prob, gen_prob)),
        'spatial_js_divergence': float(_js_divergence(real_prob, gen_prob)),
    }


# ============================================================
# 工具函数
# ============================================================


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL散度 KL(p||q)"""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS散度 (KL散度的对称版本)"""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
