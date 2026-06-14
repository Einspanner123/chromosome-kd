"""图像质量评估指标

包含:
  - FID (Fréchet Inception Distance): 生成图像与真实图像的分布距离
  - IS (Inception Score): 生成图像的清晰度和多样性
  - LPIPS (Learned Perceptual Image Patch Similarity): 感知相似度
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ============================================================
# InceptionV3 特征提取器 (FID & IS 共用)
# ============================================================


class InceptionFeatureExtractor(nn.Module):
    """InceptionV3特征提取器

    提取2048维pool3特征用于FID，以及1000维logits用于IS。
    """

    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # 加载预训练InceptionV3
        inception = models.inception_v3(pretrained=True)
        inception.eval()

        # 截取到pool3层 (FID用)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头 (IS用)
        self.fc = inception.fc

        self.to(self.device)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取特征

        Args:
            x: [B, 3, 299, 299] 归一化图像
        Returns:
            features: [B, 2048] pool3特征 (FID用)
            logits: [B, 1000] 分类logits (IS用)
        """
        x = self.blocks(x)
        x = self.avgpool(x)
        features = x.flatten(1)  # [B, 2048]
        logits = self.fc(features)  # [B, 1000]
        return features, logits


# ============================================================
# 预处理
# ============================================================

INCEPTION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


class ImageFolderDataset(Dataset):
    """简单图像文件夹数据集"""

    def __init__(self, image_dir: str, transform=None, max_images: int = None):
        self.image_dir = image_dir
        self.transform = transform or INCEPTION_TRANSFORM

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_paths = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith(exts)
            ]
        )

        if max_images and len(self.image_paths) > max_images:
            self.image_paths = self.image_paths[:max_images]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


# ============================================================
# FID 计算
# ============================================================


def compute_fid(
    real_dir: str,
    gen_dir: str,
    batch_size: int = 32,
    max_images: int = 5000,
    device: torch.device = None,
) -> float:
    """计算FID分数

    Args:
        real_dir: 真实图像目录
        gen_dir: 生成图像目录
        batch_size: 批大小
        max_images: 最大评估图像数
        device: 计算设备
    Returns:
        fid: FID分数 (越低越好)
    """
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    extractor = InceptionFeatureExtractor(device)

    # 提取真实图像特征
    real_dataset = ImageFolderDataset(real_dir, max_images=max_images)
    real_loader = DataLoader(
        real_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    real_features = []
    for batch in real_loader:
        batch = batch.to(device)
        feat, _ = extractor(batch)
        real_features.append(feat.cpu())
    real_features = torch.cat(real_features, dim=0)

    # 提取生成图像特征
    gen_dataset = ImageFolderDataset(gen_dir, max_images=max_images)
    gen_loader = DataLoader(
        gen_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    gen_features = []
    for batch in gen_loader:
        batch = batch.to(device)
        feat, _ = extractor(batch)
        gen_features.append(feat.cpu())
    gen_features = torch.cat(gen_features, dim=0)

    # 计算FID
    fid = _calculate_fid_from_features(real_features, gen_features)
    return fid


def _calculate_fid_from_features(
    real_features: torch.Tensor, gen_features: torch.Tensor
) -> float:
    """从特征计算FID

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2 * (Sigma_r * Sigma_g)^(1/2))
    """
    import numpy as np

    real_np = real_features.numpy()
    gen_np = gen_features.numpy()

    mu_r = np.mean(real_np, axis=0)
    mu_g = np.mean(gen_np, axis=0)

    sigma_r = np.cov(real_np, rowvar=False)
    sigma_g = np.cov(gen_np, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = _matrix_sqrt(sigma_r @ sigma_g)

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def _matrix_sqrt(mat):
    """矩阵平方根 (使用SVD)"""
    import numpy as np

    U, S, Vh = np.linalg.svd(mat)
    s_sqrt = np.sqrt(np.maximum(S, 0))
    return U @ np.diag(s_sqrt) @ Vh, None


# ============================================================
# IS 计算
# ============================================================


def compute_is(
    gen_dir: str,
    batch_size: int = 32,
    max_images: int = 5000,
    num_splits: int = 10,
    device: torch.device = None,
) -> Tuple[float, float]:
    """计算Inception Score

    Args:
        gen_dir: 生成图像目录
        batch_size: 批大小
        max_images: 最大评估图像数
        num_splits: 分割数
        device: 计算设备
    Returns:
        is_mean: IS均值 (越高越好)
        is_std: IS标准差
    """
    import numpy as np

    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    extractor = InceptionFeatureExtractor(device)

    gen_dataset = ImageFolderDataset(gen_dir, max_images=max_images)
    gen_loader = DataLoader(
        gen_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    all_logits = []
    for batch in gen_loader:
        batch = batch.to(device)
        _, logits = extractor(batch)
        all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)

    # 计算IS
    probs = F.softmax(all_logits, dim=1).numpy()

    scores = []
    split_size = len(probs) // num_splits
    for i in range(num_splits):
        part = probs[i * split_size : (i + 1) * split_size]
        kl = part * (
            np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0))
        )
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    is_mean = float(np.mean(scores))
    is_std = float(np.std(scores))
    return is_mean, is_std


# ============================================================
# LPIPS 计算
# ============================================================


def compute_lpips(
    real_dir: str,
    gen_dir: str,
    batch_size: int = 16,
    max_pairs: int = 1000,
    device: torch.device = None,
) -> float:
    """计算LPIPS感知相似度

    随机配对真实图像和生成图像，计算感知距离。

    Args:
        real_dir: 真实图像目录
        gen_dir: 生成图像目录
        batch_size: 批大小
        max_pairs: 最大配对数
        device: 计算设备
    Returns:
        lpips_mean: 平均LPIPS距离 (越低越相似)
    """
    try:
        import lpips
    except ImportError:
        print('Warning: lpips not installed. Install with: pip install lpips')
        return -1.0

    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    real_dataset = ImageFolderDataset(
        real_dir, transform=transform, max_images=max_pairs
    )
    gen_dataset = ImageFolderDataset(
        gen_dir, transform=transform, max_images=max_pairs
    )

    n = min(len(real_dataset), len(gen_dataset), max_pairs)

    distances = []
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        real_batch = torch.stack([real_dataset[j] for j in range(i, end)]).to(
            device
        )
        gen_batch = torch.stack([gen_dataset[j] for j in range(i, end)]).to(
            device
        )

        with torch.no_grad():
            dist = lpips_fn(real_batch, gen_batch)
        distances.extend(dist.squeeze().cpu().tolist())

    return float(torch.tensor(distances).mean())


# ============================================================
# LPIPS 多样性 (生成图像之间的LPIPS距离)
# ============================================================


def compute_diversity_lpips(
    gen_dir: str,
    batch_size: int = 16,
    max_pairs: int = 500,
    device: torch.device = None,
) -> float:
    """计算生成图像之间的LPIPS距离作为多样性指标

    随机配对生成图像，距离越大表示多样性越高。

    Args:
        gen_dir: 生成图像目录
        batch_size: 批大小
        max_pairs: 最大配对数
        device: 计算设备
    Returns:
        diversity: 平均LPIPS距离 (越高越多样)
    """
    try:
        import lpips
    except ImportError:
        print('Warning: lpips not installed. Install with: pip install lpips')
        return -1.0

    import random

    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = ImageFolderDataset(gen_dir, transform=transform)
    n = len(dataset)

    # 随机配对
    pairs = []
    for _ in range(max_pairs):
        i, j = random.sample(range(n), 2)
        pairs.append((i, j))

    distances = []
    for k in range(0, len(pairs), batch_size):
        batch_pairs = pairs[k : k + batch_size]
        imgs_a = torch.stack([dataset[i] for i, _ in batch_pairs]).to(device)
        imgs_b = torch.stack([dataset[j] for _, j in batch_pairs]).to(device)

        with torch.no_grad():
            dist = lpips_fn(imgs_a, imgs_b)
        distances.extend(dist.squeeze().cpu().tolist())

    return float(torch.tensor(distances).mean())
