"""COCO 标注噪声注入 (用于 robustness 实验 §4.8)

生成扰动版本的 COCO 标注 JSON, 用于测试推理阶段对标注噪声的鲁棒性。
所有扰动仅修改 annotations, images/categories 保持不变 (图片本身不动)。

支持的噪声类型 (可叠加):
  - bbox jitter : 对每个 GT bbox 中心点加高斯噪声 (sigma 像素)
                  bbox 仍保持原宽高, 仅中心偏移; 若偏移后越界会自动 clip
  - class flip  : 以概率 p 随机翻转类别标签 (在 24 类内均匀重采样)
                  这模拟标注员的细粒度类别混淆 (例如 C6↔C7, G21↔G22)

输出: 单个扰动 JSON 文件, 与原 COCO 格式完全兼容, 可直接喂给 mmdet test_dataloader

Usage:
    # 单个噪声级别
    python experiments/runners/robustness_noise.py \
        --src data/24_chromosomes_object/coco/test/_annotations.coco.json \
        --out-dir work_dirs/robustness_noise/perturbed \
        --noise-type both --sigma 5 --flip-rate 0.10 --seed 42

    # 批量生成 3x3 = 9 个扰动 + 1 个 baseline (clean copy)
    python experiments/runners/robustness_noise.py \
        --src data/24_chromosomes_object/coco/test/_annotations.coco.json \
        --out-dir work_dirs/robustness_noise/perturbed \
        --grid --seed 42

命名约定:
    clean.json                                  (sigma=0, flip=0)
    noise_both_s02_f005.json                    (sigma=2,  flip=0.05)
    noise_both_s05_f010.json                    (sigma=5,  flip=0.10)
    noise_both_s10_f020.json                    (sigma=10, flip=0.20)
    ... (3x3 = 9 个组合)
"""

import argparse
import json
import os
import random
import shutil

import numpy as np


def perturb_bbox(bbox, sigma, img_w, img_h, rng):
    """对单个 bbox [x, y, w, h] 中心点加高斯噪声, 保持 w/h 不变。

    若 sigma=0, 原样返回 (不做任何改动, 与 clean 一致)。
    偏移后中心点会被 clip 到图像边界内。
    """
    x, y, w, h = bbox
    if sigma <= 0:
        return [float(x), float(y), float(w), float(h)]
    dx = float(rng.normal(0.0, sigma))
    dy = float(rng.normal(0.0, sigma))
    # 中心点
    cx = x + w / 2.0 + dx
    cy = y + h / 2.0 + dy
    # clip 中心点到 [w/2, img_w - w/2] 之内 (保证 bbox 完全在图像内)
    cx = max(w / 2.0, min(img_w - w / 2.0, cx))
    cy = max(h / 2.0, min(img_h - h / 2.0, cy))
    new_x = cx - w / 2.0
    new_y = cy - h / 2.0
    return [new_x, new_y, float(w), float(h)]


def perturb_class(category_id, num_classes, flip_rate, rng):
    """以概率 flip_rate 把 category_id 替换为另一个均匀随机类别。

    若 flip_rate <= 0, 原样返回。否则保留原类别 (1-p 概率), 或从其余
    num_classes-1 个类别中均匀采样一个 (p 概率)。
    """
    if flip_rate <= 0:
        return int(category_id)
    if rng.random() < flip_rate:
        # 从 [1, num_classes] 中随机选一个, 排除原始 id
        candidates = [c for c in range(1, num_classes + 1) if c != category_id]
        return int(rng.choice(candidates))
    return int(category_id)


def perturb_coco(src_data, sigma, flip_rate, seed):
    """对 COCO dict 应用 bbox jitter + class flip 扰动。

    返回新的 dict (深拷贝, 不修改原对象)。
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(
        seed + 1
    )  # bbox 用单独的 numpy RNG, 与 class flip 解耦

    images = src_data.get('images', [])
    categories = src_data.get('categories', [])
    num_classes = len(categories)
    # category_id 集合 (用于验证扰动后仍合法)
    valid_cat_ids = {c['id'] for c in categories}

    # image_id -> (width, height) 查找表 (bbox clip 需要)
    img_size = {
        img['id']: (img.get('width', 1), img.get('height', 1))
        for img in images
    }

    new_annotations = []
    bbox_perturbed_count = 0
    class_flipped_count = 0
    for ann in src_data.get('annotations', []):
        new_ann = dict(ann)  # shallow copy (够用, 我们只改 bbox/category_id)

        # bbox jitter
        if sigma > 0 and 'bbox' in new_ann and 'image_id' in new_ann:
            img_w, img_h = img_size.get(new_ann['image_id'], (1, 1))
            old_bbox = list(new_ann['bbox'])
            new_bbox = perturb_bbox(old_bbox, sigma, img_w, img_h, np_rng)
            if new_bbox != old_bbox:
                bbox_perturbed_count += 1
            new_ann['bbox'] = new_bbox
            # area 同步重算 (COCO eval 用 area 选 small/medium/large)
            if len(new_bbox) == 4:
                new_ann['area'] = float(new_bbox[2] * new_bbox[3])

        # class flip
        if flip_rate > 0 and 'category_id' in new_ann:
            old_cat = new_ann['category_id']
            new_cat = perturb_class(old_cat, num_classes, flip_rate, rng)
            if new_cat != old_cat:
                class_flipped_count += 1
            new_ann['category_id'] = new_cat

        # 防御: 确保 category_id 合法
        if new_ann.get('category_id') not in valid_cat_ids:
            # 极端情况下兜底: 回退到原始 id
            new_ann['category_id'] = ann.get('category_id')

        new_annotations.append(new_ann)

    new_data = {
        'images': images,
        'annotations': new_annotations,
        'categories': categories,
    }
    # 保留原 JSON 中的其它字段 (info, licenses 等, 若存在)
    for k, v in src_data.items():
        if k not in new_data:
            new_data[k] = v

    stats = {
        'n_annotations': len(new_annotations),
        'n_bbox_perturbed': bbox_perturbed_count,
        'n_class_flipped': class_flipped_count,
        'sigma': sigma,
        'flip_rate': flip_rate,
        'seed': seed,
    }
    return new_data, stats


def fname_for(noise_type, sigma, flip_rate):
    """生成扰动 JSON 文件名 (与 Usage 文档一致)。"""
    if noise_type == 'bbox' and flip_rate <= 0:
        return f'noise_bbox_s{int(sigma):02d}.json'
    if noise_type == 'class' and sigma <= 0:
        return f'noise_class_f{int(flip_rate * 100):03d}.json'
    # 默认 both
    return f'noise_both_s{int(sigma):02d}_f{int(flip_rate * 100):03d}.json'


def main():
    parser = argparse.ArgumentParser(
        description='COCO 标注噪声注入 (robustness 实验)'
    )
    parser.add_argument(
        '--src', required=True, help='原始 COCO 标注 JSON 路径'
    )
    parser.add_argument('--out-dir', required=True, help='输出目录')
    parser.add_argument(
        '--noise-type',
        default='both',
        choices=['bbox', 'class', 'both'],
        help='噪声类型 (both = bbox + class 同时扰动)',
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.0,
        help='bbox jitter sigma (像素); 0 = 不扰动',
    )
    parser.add_argument(
        '--flip-rate',
        type=float,
        default=0.0,
        help='类别翻转概率 (0~1); 0 = 不扰动',
    )
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument(
        '--grid',
        action='store_true',
        help='批量生成 3x3 = 9 个扰动 + 1 个 clean baseline',
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[robustness_noise] loading src: {args.src}')
    with open(args.src) as f:
        src_data = json.load(f)
    print(
        f'[robustness_noise] src: {len(src_data["images"])} images, '
        f'{len(src_data["annotations"])} annotations, '
        f'{len(src_data["categories"])} categories'
    )

    if args.grid:
        # 3x3 grid: sigma ∈ {2, 5, 10} px × flip ∈ {5%, 10%, 20%}
        # 同时复制一份 clean.json 作为 baseline
        sigmas = [2, 5, 10]
        flip_rates = [0.05, 0.10, 0.20]
        all_runs = [(0.0, 0.0)]  # clean baseline
        for s in sigmas:
            for f in flip_rates:
                all_runs.append((float(s), float(f)))
    else:
        # 单次运行: 根据 noise-type 决定哪些维度扰动
        sigma = args.sigma if args.noise_type in ('bbox', 'both') else 0.0
        flip_rate = (
            args.flip_rate if args.noise_type in ('class', 'both') else 0.0
        )
        all_runs = [(sigma, flip_rate)]

    summary = []
    for sigma, flip_rate in all_runs:
        # 命名
        if sigma == 0.0 and flip_rate == 0.0:
            out_name = 'clean.json'
            out_path = os.path.join(args.out_dir, out_name)
            # clean = 直接复制原文件 (保证格式完全一致)
            if os.path.abspath(args.src) != os.path.abspath(out_path):
                shutil.copyfile(args.src, out_path)
            print(f'[robustness_noise] clean copy -> {out_path}')
            summary.append(
                {
                    'file': out_name,
                    'sigma': 0.0,
                    'flip_rate': 0.0,
                    'n_annotations': len(src_data['annotations']),
                    'n_bbox_perturbed': 0,
                    'n_class_flipped': 0,
                    'seed': args.seed,
                }
            )
            continue

        new_data, stats = perturb_coco(src_data, sigma, flip_rate, args.seed)
        out_name = fname_for(args.noise_type, sigma, flip_rate)
        out_path = os.path.join(args.out_dir, out_name)
        with open(out_path, 'w') as f:
            json.dump(new_data, f)
        print(
            f'[robustness_noise] {out_name}: '
            f'bbox_perturbed={stats["n_bbox_perturbed"]}/{stats["n_annotations"]} '
            f'({100 * stats["n_bbox_perturbed"] / max(1, stats["n_annotations"]):.1f}%), '
            f'class_flipped={stats["n_class_flipped"]}/{stats["n_annotations"]} '
            f'({100 * stats["n_class_flipped"] / max(1, stats["n_annotations"]):.1f}%) '
            f'-> {out_path}'
        )
        summary.append({'file': out_name, **stats})

    # 写一份 manifest 供下游脚本读取
    manifest_path = os.path.join(args.out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(
            {'src': args.src, 'seed': args.seed, 'runs': summary}, f, indent=2
        )
    print(f'[robustness_noise] manifest -> {manifest_path}')
    print(f'[robustness_noise] done. {len(summary)} files written.')


if __name__ == '__main__':
    main()
