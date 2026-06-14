"""ChromoGen 评估指标模块

支持 FID (Fréchet Inception Distance) 和 IS (Inception Score) 计算。
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.models import inception_v3


def _get_inception_model(device: torch.device) -> torch.nn.Module:
    """加载预训练 InceptionV3 模型"""
    try:
        from torchvision.models import Inception_V3_Weights

        model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=True
        )
    except (ImportError, AttributeError):
        model = inception_v3(pretrained=True, transform_input=True)
    model.fc = torch.nn.Identity()  # 移除最后一层全连接，输出 2048-dim 特征
    model.eval()
    model.to(device)
    return model


def _extract_inception_features(
    images: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int = 32,
) -> np.ndarray:
    """提取 InceptionV3 特征 (2048-dim pool3)

    Args:
        images: [N, 3, H, W] 像素值范围 [0, 1]
        model: InceptionV3 特征提取器
        batch_size: 批处理大小
    Returns:
        features: [N, 2048] numpy array
    """
    device = next(model.parameters()).device
    features_list = []

    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i : i + batch_size].to(device)
            feats = model(batch)
            features_list.append(feats.cpu().numpy())

    return np.concatenate(features_list, axis=0)


def compute_fid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
) -> float:
    """计算 Fréchet Inception Distance

    FID = |mu_r - mu_f|^2 + Tr(Σ_r + Σ_f - 2(Σ_r Σ_f)^{1/2})

    Args:
        real_features: [N_real, 2048]
        fake_features: [N_fake, 2048]
    Returns:
        FID 值 (越低越好)
    """
    mu_r = np.mean(real_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)

    mu_f = np.mean(fake_features, axis=0)
    sigma_f = np.cov(fake_features, rowvar=False)

    # 数值稳定性: 给协方差矩阵加一个小偏移
    eps = 1e-6
    sigma_r = sigma_r + np.eye(sigma_r.shape[0]) * eps
    sigma_f = sigma_f + np.eye(sigma_f.shape[0]) * eps

    # Fréchet 距离
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

    # 处理 sqrtm 可能产生的复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
    return fid


def compute_inception_score(
    features: np.ndarray,
    splits: int = 10,
) -> Tuple[float, float]:
    """计算 Inception Score

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Args:
        features: [N, 2048] InceptionV3 特征 (需要包含分类信息)
        splits: 分片数 (用于计算标准差)
    Returns:
        (mean_is, std_is)
    """
    # IS 需要 softmax 输出，用线性层近似 (InceptionV3 的 fc 层输出 1000-way logits)
    # 因为我们用了 Identity fc，这里从 2048-dim 特征无法直接得到类别分布。
    # 需要重新加载带 fc 层的模型。
    return 0.0, 0.0


def compute_inception_score_full(
    images: torch.Tensor,
    device: torch.device,
    splits: int = 10,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """完整 IS 计算 - 使用 InceptionV3 的 softmax 输出

    Args:
        images: [N, 3, H, W] 像素值范围 [0, 1]
        device: 计算设备
        splits: 分片数
        batch_size: 批处理大小
    Returns:
        (mean_is, std_is)
    """
    try:
        from torchvision.models import Inception_V3_Weights

        model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=True
        )
    except (ImportError, AttributeError):
        model = inception_v3(pretrained=True, transform_input=True)
    model.eval()
    model.to(device)

    preds_list = []
    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch = images[i : i + batch_size].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            preds_list.append(probs.cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)  # [N, 1000]

    # 分片计算
    N = preds.shape[0]
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits)]
        py = np.mean(part, axis=0)  # p(y) 边际分布
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i]
            scores.append(np.sum(pyx * (np.log(pyx) - np.log(py))))
        split_scores.append(np.exp(np.mean(scores)))

    return float(np.mean(split_scores)), float(np.std(split_scores))


class ChromoGenEvaluator:
    """ChromoGen 评估器

    管理真实图像特征缓存，计算 FID/IS 等指标。
    """

    def __init__(
        self,
        real_images: torch.Tensor,
        device: torch.device,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            real_images: 真实图像张量 [N, 3, H, W]，范围 [0, 1]
            device: 计算设备
            cache_dir: 特征缓存目录 (可选)
        """
        self.device = device
        self.cache_dir = cache_dir

        # 加载 Inception 模型
        self.inception = _get_inception_model(device)

        # 预计算真实图像特征
        print(
            f'Extracting inception features from {real_images.shape[0]} real images...'
        )
        self.real_features = _extract_inception_features(
            real_images, self.inception
        )
        print(f'Real features shape: {self.real_features.shape}')

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        device: torch.device,
        max_images: int = 256,
        cache_dir: Optional[str] = None,
    ) -> 'ChromoGenEvaluator':
        """从 DataLoader 构建评估器 (预加载真实图像)

        Args:
            dataloader: 验证集 DataLoader
            device: 计算设备
            max_images: 最多加载的图像数
            cache_dir: 特征缓存目录
        """
        images_list = []
        for batch in dataloader:
            # 反归一化: [-1, 1] → [0, 1]
            pixel_values = batch['pixel_values']
            pixel_values = (pixel_values * 0.5 + 0.5).clamp(0, 1)
            images_list.append(pixel_values)
            if sum(img.shape[0] for img in images_list) >= max_images:
                break
        real_images = torch.cat(images_list, dim=0)[:max_images]
        return cls(real_images, device, cache_dir)

    def evaluate(
        self,
        fake_images: torch.Tensor,
    ) -> dict:
        """评估生成图像质量

        Args:
            fake_images: 生成图像 [N, 3, H, W]，范围 [0, 1]
        Returns:
            metrics: {'fid': float, 'is_mean': float, 'is_std': float}
        """
        # FID
        fake_features = _extract_inception_features(
            fake_images, self.inception
        )
        fid = compute_fid(self.real_features, fake_features)

        # IS
        N = fake_images.shape[0]
        if N >= 10:
            is_mean, is_std = compute_inception_score_full(
                fake_images, self.device
            )
        else:
            is_mean, is_std = 0.0, 0.0

        return {
            'fid': fid,
            'is_mean': is_mean,
            'is_std': is_std,
        }
