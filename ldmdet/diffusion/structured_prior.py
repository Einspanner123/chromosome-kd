"""StructuredPrior — 方向 F1: 结构化噪声先验

用训练集框的统计分布 (高斯混合 GMM) 代替标准高斯噪声 N(0,I) 作为扩散起点.
降低"从纯噪声找 GT"的学习难度.

设计:
- 预计算训练集框 (cxcywh 归一化) 的 GMM 统计
- 训练时 x_1 ~ GMM 采样, 而非 randn
- 采样时从 GMM 采样初始框
- 支持 clamp 到 [0,1] 保证有效坐标
"""

import torch
import torch.nn as nn


class StructuredPrior(nn.Module):
    """结构化噪声先验 (高斯混合).

    Args:
        means: (K, 4) 各高斯分量的均值
        stds: (K, 4) 各高斯分量的标准差
        weights: (K,) 各分量的混合权重
    """

    def __init__(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
        weights: torch.Tensor,
    ):
        super().__init__()
        # 注册为 buffer (不参与训练, 但随 model.to() 移动)
        self.register_buffer('means', means.float())
        self.register_buffer('stds', stds.float())
        self.register_buffer('weights', weights.float())

    @classmethod
    def from_statistics(
        cls,
        boxes: torch.Tensor,
        num_components: int = 4,
    ) -> 'StructuredPrior':
        """从训练集框统计构建 GMM (简化版: K-means 聚类 + 各簇统计).

        Args:
            boxes: (N, 4) 归一化 cxcywh 框
            num_components: 高斯分量数

        Returns:
            StructuredPrior 实例
        """
        N = boxes.shape[0]
        K = min(num_components, N)
        # 简化: 随机选 K 个中心, 分配到最近簇
        idx = torch.randperm(N)[:K]
        centers = boxes[idx].clone()
        for _ in range(10):  # K-means 迭代
            dists = torch.cdist(boxes, centers)  # (N, K)
            assign = dists.argmin(dim=-1)  # (N,)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    centers[k] = boxes[mask].mean(dim=0)
        # 统计各簇
        means = []
        stds = []
        weights = []
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
        return cls(
            means=torch.stack(means),
            stds=torch.stack(stds),
            weights=torch.stack(weights),
        )

    def sample(
        self,
        n: int,
        clamp_to_unit: bool = False,
    ) -> torch.Tensor:
        """从 GMM 采样.

        Args:
            n: 采样数
            clamp_to_unit: 是否 clamp 到 [0, 1]

        Returns:
            (n, 4) 采样框 (cxcywh 归一化)
        """
        # 按权重选分量
        idx = torch.multinomial(self.weights, n, replacement=True)  # (n,)
        # 从对应分量采样
        means = self.means[idx]  # (n, 4)
        stds = self.stds[idx]  # (n, 4)
        samples = means + stds * torch.randn_like(means)
        if clamp_to_unit:
            samples = samples.clamp(0.0, 1.0)
        return samples
