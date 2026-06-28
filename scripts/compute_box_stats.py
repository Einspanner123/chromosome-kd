"""预计算训练集框统计, 生成 StructuredPrior 所需的 GMM 参数.

用法:
    python scripts/compute_box_stats.py \
        --ann-file data/Chromosome20240904_NoAug_NoResize_coco/train/_annotations.coco.json \
        --out work_dirs/box_stats.pt --num-components 4

输出: work_dirs/box_stats.pt (含 means, stds, weights 三个 tensor)
"""

import argparse
import json
import os
import os.path as osp

import torch
from torchvision.ops import box_convert  # xyxy -> cxcywh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann-file',
        default='data/Chromosome20240904_NoAug_NoResize_coco/train/_annotations.coco.json',
    )
    parser.add_argument('--out', default='work_dirs/box_stats.pt')
    parser.add_argument('--num-components', type=int, default=4)
    args = parser.parse_args()

    # 读取 COCO 标注
    with open(args.ann_file) as f:
        coco = json.load(f)

    # 收集所有框 (xywh -> xyxy -> cxcywh 归一化)
    boxes = []
    img_wh = {img['id']: (img['width'], img['height']) for img in coco['images']}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        w, h = img_wh[img_id]
        x, y, bw, bh = ann['bbox']  # xywh (COCO 格式)
        # 归一化 cxcywh
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        boxes.append([cx, cy, nw, nh])

    boxes = torch.tensor(boxes, dtype=torch.float32)
    print(f'共 {boxes.shape[0]} 个框')

    # K-means 聚类 + 各簇统计 (简化 GMM)
    N = boxes.shape[0]
    K = min(args.num_components, N)
    idx = torch.randperm(N)[:K]
    centers = boxes[idx].clone()
    for _ in range(20):
        dists = torch.cdist(boxes, centers)
        assign = dists.argmin(dim=-1)
        for k in range(K):
            mask = assign == k
            if mask.any():
                centers[k] = boxes[mask].mean(dim=0)

    means, stds, weights = [], [], []
    for k in range(K):
        mask = assign == k
        if mask.any():
            cluster = boxes[mask]
            means.append(cluster.mean(dim=0))
            stds.append(cluster.std(dim=0) + 1e-4)
            weights.append(mask.sum().float() / N)
        else:
            means.append(centers[k])
            stds.append(torch.full((4,), 0.01))
            weights.append(torch.tensor(1.0 / K))

    stats = {
        'means': torch.stack(means),
        'stds': torch.stack(stds),
        'weights': torch.stack(weights),
    }
    os.makedirs(osp.dirname(args.out) or '.', exist_ok=True)
    torch.save(stats, args.out)
    print(f'保存到 {args.out}')
    print(f'means: {stats["means"]}')
    print(f'stds: {stats["stds"]}')
    print(f'weights: {stats["weights"]}')


if __name__ == '__main__':
    main()
