"""Phase 0 探针实验: Linear Probe 运行脚本

在 ChromoGen 冻结特征上训练线性分类器, 评估特征的检测信息量。

实验设计:
  1. 对训练集和验证集图像提取多尺度特征
  2. 对每个 bbox, 用 ROI Align 提取对应位置的特征
  3. 训练线性分类器预测类别
  4. 在验证集上评估准确率

对比层:
  - vae_latent (4ch): VAE 潜变量
  - down1 (320ch): UNet 浅层特征
  - down2 (640ch): UNet 中层特征
  - down3 (1280ch): UNet 深层特征
  - mid (1280ch): UNet bottleneck 特征

用法:
  python projects/ChromoGen/tools/run_linear_probe.py --gpu 0
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import roi_align

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from pycocotools.coco import COCO

from projects.ChromoGen.evaluation.linear_probe import (
    FeatureExtractor,
    LinearProbe,
    RoIFeatureCollector,
)


class ChromoProbeDataset(Dataset):
    """染色体探针数据集"""

    def __init__(self, data_root, ann_file, img_dir, image_size=768):
        self.coco = COCO(os.path.join(data_root, ann_file))
        self.img_dir = os.path.join(data_root, img_dir)
        self.image_size = image_size
        self.image_ids = list(self.coco.imgs.keys())

        # 图像预处理 (与 ChromoGen 训练一致)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -1 到 1
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 加载图像
        from PIL import Image

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # 预处理
        img_tensor = self.transform(img)

        # bbox 归一化 [x1, y1, x2, y2] (0-1)
        bboxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            # 归一化到 0-1
            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h
            bboxes.append([x1, y1, x2, y2])
            # category_id 1-24 → 0-23
            labels.append(ann['category_id'] - 1)

        return {
            'image': img_tensor,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'orig_size': (orig_h, orig_w),
        }


def collate_fn(batch):
    """自定义 collate: 图像 stack, bbox/labels 保留为 list"""
    images = torch.stack([b['image'] for b in batch], dim=0)
    bboxes = [b['bboxes'] for b in batch]
    labels = [b['labels'] for b in batch]
    orig_sizes = [b['orig_size'] for b in batch]
    return {
        'image': images,
        'bboxes': bboxes,
        'labels': labels,
        'orig_size': orig_sizes,
    }


def extract_features_for_all_images(
    extractor, dataset, device, batch_size=1
):
    """提取所有图像的特征和 bbox 标签

    Returns:
        features_by_layer: dict {layer_name: list of (roi_feats, labels)}
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        collate_fn=collate_fn,
    )

    # 收集所有层的 ROI 特征和标签
    features_by_layer = {
        'vae_latent': [],
        'down1': [],
        'down2': [],
        'down3': [],
        'mid': [],
    }
    all_labels_by_layer = {
        'vae_latent': [],
        'down1': [],
        'down2': [],
        'down3': [],
        'mid': [],
    }

    total = len(dataset)
    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device)
        # batch_size=1: 取第一个
        bboxes = batch['bboxes'][0]
        labels = batch['labels'][0]
        orig_size = batch['orig_size'][0]  # (H, W)

        if bboxes.shape[0] == 0:
            continue

        # 提取特征
        feats = extractor(images)

        # 对每层提取 ROI 特征
        for layer_name, feat in feats.items():
            # feat: (1, C, H, W)
            # bboxes: (N, 4) 归一化
            # 需要根据特征图尺寸调整 spatial_scale
            feat_h, feat_w = feat.shape[2:]
            # 图像被 resize 到 768x768, bbox 已归一化到 0-1
            # 使用 768 作为参考尺寸
            img_h, img_w = 768.0, 768.0
            spatial_scale = feat_h / img_h  # 特征图相对于 resize 后图像的比例

            # 转换 bbox 到像素坐标
            bboxes_pixel = bboxes.clone()
            bboxes_pixel[:, 0] *= img_w
            bboxes_pixel[:, 1] *= img_h
            bboxes_pixel[:, 2] *= img_w
            bboxes_pixel[:, 3] *= img_h

            # roi_align 需要 (N, 5) [batch_idx, x1, y1, x2, y2]
            n = bboxes.shape[0]
            rois = torch.zeros(n, 5, device=feat.device)
            rois[:, 1:] = bboxes_pixel

            roi_feats = roi_align(
                input=feat,
                boxes=rois,
                output_size=(7, 7),
                spatial_scale=spatial_scale,
                aligned=True,
            )

            features_by_layer[layer_name].append(roi_feats.cpu())
            all_labels_by_layer[layer_name].append(labels)

        if (batch_idx + 1) % 10 == 0:
            print(f'  Extracted {batch_idx + 1}/{total} images')

    # 合并所有 ROI 特征
    merged_features = {}
    merged_labels = {}
    for layer in features_by_layer:
        if features_by_layer[layer]:
            merged_features[layer] = torch.cat(features_by_layer[layer], dim=0)
            merged_labels[layer] = torch.cat(all_labels_by_layer[layer], dim=0)
        else:
            merged_features[layer] = torch.empty(0)
            merged_labels[layer] = torch.empty(0, dtype=torch.long)

    return merged_features, merged_labels


def train_linear_probe(
    train_feats, train_labels, val_feats, val_labels, num_classes=24,
    epochs=50, lr=1e-3, device='cpu'
):
    """训练线性探针分类器"""
    in_channels = train_feats.shape[1]
    probe = LinearProbe(
        in_channels=in_channels, num_classes=num_classes, roi_size=7
    ).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    best_val_acc = 0
    batch_size = 256
    n_samples = train_feats.shape[0]

    for epoch in range(epochs):
        probe.train()
        # 随机打乱
        perm = torch.randperm(n_samples)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            batch_feats = train_feats[idx]
            batch_labels = train_labels[idx]

            logits = probe(batch_feats)
            loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # 验证
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_feats)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(
                f'  Epoch {epoch + 1}/{epochs}: '
                f'loss={epoch_loss / n_batches:.4f}, val_acc={val_acc:.4f}'
            )

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Linear Probe Experiment')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=str,
                        default='work_dirs/chromogen_phase1/final_model.pt')
    parser.add_argument('--vae_path', type=str,
                        default='work_dirs/chromogen_phase1/vae')
    parser.add_argument('--data_root', type=str,
                        default='data/Chromosome20240904_NoAug_NoResize_coco/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_images', type=int, default=200,
                        help='最大提取图像数 (节省时间)')
    parser.add_argument('--output', type=str,
                        default='work_dirs/chromogen_phase1/probe_results.json')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    torch.manual_seed(42)

    # 1. 加载特征提取器
    print('Loading ChromoGen model...')
    extractor = FeatureExtractor(
        checkpoint=args.checkpoint,
        vae_path=args.vae_path,
        device=device,
    )
    print('Model loaded.')

    # 2. 加载数据集
    train_dataset = ChromoProbeDataset(
        data_root=args.data_root,
        ann_file='train/_annotations.coco.json',
        img_dir='train',
    )
    val_dataset = ChromoProbeDataset(
        data_root=args.data_root,
        ann_file='valid/_annotations.coco.json',
        img_dir='valid',
    )

    # 限制图像数量
    if args.max_images > 0:
        train_dataset.image_ids = train_dataset.image_ids[:args.max_images]
        val_dataset.image_ids = val_dataset.image_ids[:args.max_images // 4]

    print(f'Train images: {len(train_dataset)}, Val images: {len(val_dataset)}')

    # 3. 提取特征
    print('\nExtracting train features...')
    t0 = time.time()
    train_feats, train_labels = extract_features_for_all_images(
        extractor, train_dataset, device, batch_size=args.batch_size
    )
    print(f'Train extraction done in {time.time() - t0:.1f}s')

    print('\nExtracting val features...')
    t0 = time.time()
    val_feats, val_labels = extract_features_for_all_images(
        extractor, val_dataset, device, batch_size=args.batch_size
    )
    print(f'Val extraction done in {time.time() - t0:.1f}s')

    # 4. 对每层特征训练 Linear Probe
    results = {}
    for layer in ['vae_latent', 'down1', 'down2', 'down3', 'mid']:
        n_train = train_feats[layer].shape[0]
        n_val = val_feats[layer].shape[0]
        print(f'\n{"=" * 60}')
        print(f'Layer: {layer} (train={n_train}, val={n_val})')
        print(f'{"=" * 60}')

        if n_train == 0 or n_val == 0:
            print('  Skipping (no data)')
            results[layer] = {'val_acc': 0, 'n_train': n_train, 'n_val': n_val}
            continue

        # 统计类别分布
        class_counts = torch.bincount(train_labels[layer], minlength=24)
        print(f'  Class distribution: min={class_counts.min().item()}, '
              f'max={class_counts.max().item()}')

        # 训练
        best_acc = train_linear_probe(
            train_feats[layer], train_labels[layer],
            val_feats[layer], val_labels[layer],
            num_classes=24, epochs=args.epochs, lr=1e-3,
            device=device,
        )

        results[layer] = {
            'val_acc': best_acc,
            'n_train': n_train,
            'n_val': n_val,
            'in_channels': train_feats[layer].shape[1],
        }
        print(f'  Best Val Acc: {best_acc:.4f} ({best_acc * 100:.1f}%)')

    # 5. 打印汇总
    print(f'\n{"=" * 60}')
    print('Linear Probe Results Summary')
    print(f'{"=" * 60}')
    print(f'{"Layer":<15} {"Channels":<10} {"Val Acc":<10} {"Random":<10}')
    print(f'{"-" * 45}')
    for layer, res in results.items():
        print(
            f'{layer:<15} {res["in_channels"]:<10} '
            f'{res["val_acc"] * 100:>6.1f}%    {100 / 24:>6.1f}%'
        )

    # 保存结果
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
