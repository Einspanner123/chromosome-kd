"""离线估计类条件先验 (mu_p^c, sigma_bar_sq^c) 供 TRIP 使用

TRIP (Tikhonov-Regularized Inverse-problem Prediction) 需要每类的
先验均值 mu_p^c 和各向同性平均方差 sigma_bar_sq^c = tr(Sigma_p^c) / d.

本工具从 COCO 格式标注文件中统计每类 GT bbox (cxcywh, 归一化 [0,1]),
计算类条件先验并保存为 pickle.

用法:
    python ldmdet/tools/estimate_class_priors.py \\
        --ann-file data/24_chromosomes_object/coco/train/_annotations.coco.json \\
        --output data/class_priors_24obj.pkl

输出格式 (pickle):
    {
        'mu': Tensor[num_classes, 4],          # per-class 均值 (cxcywh, [0,1])
        'sigma_bar_sq': Tensor[num_classes],   # per-class 各向同性平均方差
        'count': Tensor[num_classes],          # per-class 样本数 (诊断用)
    }
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import torch

# 确保项目根目录在 path 中
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='估计类条件先验 (mu_p^c, sigma_bar_sq^c) 供 TRIP 使用'
    )
    parser.add_argument(
        '--ann-file', required=True,
        help='COCO 格式标注文件路径 (如 train/_annotations.coco.json)'
    )
    parser.add_argument(
        '--output', required=True,
        help='输出 pickle 文件路径 (如 data/class_priors_24obj.pkl)'
    )
    parser.add_argument(
        '--num-classes', type=int, default=24,
        help='类别数 (默认 24, 染色体数据集)'
    )
    return parser.parse_args()


def load_coco_annotations(ann_file):
    """加载 COCO 格式标注, 返回 (images, annotations)"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data['images'], data['annotations'], data.get('categories', [])


def collect_class_bboxes(images, annotations, num_classes):
    """收集每类 GT bbox (cxcywh, 归一化 [0,1])

    COCO bbox 格式: [x, y, w, h] (像素, 左上角+尺寸)
    转换: cx = x + w/2, cy = y + h/2, 归一化 by image size
    """
    # 建立 image_id → (w, h) 映射
    img_size = {img['id']: (img['width'], img['height']) for img in images}

    # 收集每类 bbox
    class_bboxes = defaultdict(list)
    for ann in annotations:
        # COCO category_id 通常 1-indexed; 减 1 转为 0-indexed
        cat_id = ann['category_id'] - 1
        if cat_id < 0 or cat_id >= num_classes:
            continue
        img_w, img_h = img_size.get(ann['image_id'], (None, None))
        if img_w is None:
            continue
        x, y, w, h = ann['bbox']
        # 转换 cxcywh 并归一化
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        class_bboxes[cat_id].append([cx, cy, w_norm, h_norm])

    return class_bboxes


def compute_priors(class_bboxes, num_classes):
    """计算每类先验 (mu_p^c, sigma_bar_sq^c)

    sigma_bar_sq^c = tr(Sigma_p^c) / d  (d=4, 各向同性平均方差)
    """
    mu = torch.zeros(num_classes, 4)
    sigma_bar_sq = torch.zeros(num_classes)
    count = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        bboxes = class_bboxes.get(c, [])
        if len(bboxes) == 0:
            print(f"  [警告] 类 {c} 无样本, mu_p=0, sigma_p²=0 (TRIP 将退化为 GT)")
            continue
        bboxes_t = torch.tensor(bboxes, dtype=torch.float32)  # [N, 4]
        mu[c] = bboxes_t.mean(dim=0)
        if len(bboxes) > 1:
            # 样本协方差 (无偏估计, ddof=1)
            cov = torch_cov(bboxes_t, rowvar=False)  # [4, 4]
            sigma_bar_sq[c] = cov.trace() / 4.0  # tr(Sigma)/d
        else:
            # 单样本: 方差设为极小值 (避免除零)
            sigma_bar_sq[c] = 1e-6
        count[c] = len(bboxes)

    return mu, sigma_bar_sq, count


def torch_cov(x, rowvar=False):
    """计算协方差矩阵 (类似 numpy.cov, ddof=1)"""
    if x.shape[0] < 2:
        return torch.zeros(x.shape[1], x.shape[1])
    mean = x.mean(dim=0, keepdim=True)
    xm = x - mean
    cov = xm.T @ xm / (x.shape[0] - 1)
    return cov


def main():
    args = parse_args()

    print(f"加载标注: {args.ann_file}")
    images, annotations, categories = load_coco_annotations(args.ann_file)
    print(f"  图像数: {len(images)}")
    print(f"  标注数: {len(annotations)}")
    if categories:
        print(f"  类别数 (文件): {len(categories)}")

    print(f"\n收集每类 GT bbox (cxcywh, 归一化)...")
    class_bboxes = collect_class_bboxes(images, annotations, args.num_classes)

    print(f"\n计算类条件先验 (mu_p^c, sigma_bar_sq^c)...")
    mu, sigma_bar_sq, count = compute_priors(class_bboxes, args.num_classes)

    # 打印统计
    print(f"\n=== 类条件先验统计 ===")
    print(f"{'类':>4} {'类名':>6} {'样本数':>8} {'mu_cx':>8} {'mu_cy':>8} "
          f"{'mu_w':>8} {'mu_h':>8} {'sigma²':>10}")
    for c in range(args.num_classes):
        name = categories[c]['name'] if c < len(categories) else f'cls{c}'
        print(f"{c:>4} {name:>6} {count[c].item():>8} "
              f"{mu[c, 0].item():>8.4f} {mu[c, 1].item():>8.4f} "
              f"{mu[c, 2].item():>8.4f} {mu[c, 3].item():>8.4f} "
              f"{sigma_bar_sq[c].item():>10.6f}")

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    priors = {
        'mu': mu,
        'sigma_bar_sq': sigma_bar_sq,
        'count': count,
    }
    torch.save(priors, args.output)
    print(f"\n已保存到: {args.output}")
    print(f"  mu: shape={mu.shape}, dtype={mu.dtype}")
    print(f"  sigma_bar_sq: shape={sigma_bar_sq.shape}, dtype={sigma_bar_sq.dtype}")

    # 验证: s(t) 曲线 (MAP, λ=t²)
    print(f"\n=== 收缩因子 s(t) 验证 (MAP, λ=t²) ===")
    print(f"{'t':>6} {'s (min σ²)':>12} {'s (max σ²)':>12} {'s (mean σ²)':>12}")
    sigma_min = sigma_bar_sq.min().item()
    sigma_max = sigma_bar_sq.max().item()
    sigma_mean = sigma_bar_sq.mean().item()
    for t_val in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        # s(t) = (t²/σ²) / ((1-t)² + t²/σ²)
        def s(t, sig):
            if sig < 1e-8:
                return 0.0
            gamma = t ** 2 / sig
            return gamma / ((1 - t) ** 2 + gamma)
        print(f"{t_val:>6.2f} {s(t_val, sigma_min):>12.4f} "
              f"{s(t_val, sigma_max):>12.4f} {s(t_val, sigma_mean):>12.4f}")


if __name__ == '__main__':
    main()
