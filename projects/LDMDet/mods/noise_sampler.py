"""
结构化噪声采样器 — FlowDet

用网格初始化+扰动替代纯高斯噪声，减小 OT 传输距离，
使单步检测质量大幅提升。

策略:
- 'pure': 纯高斯噪声（消融用，与 DiffusionDet 一致）
- 'grid': 网格中心 + 默认尺寸 + 高斯扰动
- 'grid_multiscale': 多尺度网格 + 扰动
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class StructuredNoiseSampler(nn.Module):
    """结构化噪声采样器

    在扩散空间 (cxcywh, [-snr, snr]) 中生成带有空间结构的初始噪声框，
    而非完全随机的高斯噪声。

    Args:
        num_proposals: 提议框数量
        noise_scale: 高斯扰动的标准差
        strategy: 'pure', 'grid', 'grid_multiscale'
        snr_scale: 信噪比缩放 (需要与 DiffusionDetHead 一致)
    """

    def __init__(
        self,
        num_proposals: int = 500,
        noise_scale: float = 1.0,
        strategy: str = "grid",
        snr_scale: float = 2.0,
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.noise_scale = noise_scale
        self.strategy = strategy
        self.snr_scale = snr_scale

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        """采样噪声框

        Returns:
            (bs, num_proposals, 4) 扩散空间的噪声框
        """
        if self.strategy == "pure":
            return self._sample_pure(batch_size, device)
        elif self.strategy == "grid":
            return self._sample_grid(batch_size, device)
        elif self.strategy == "grid_multiscale":
            return self._sample_grid_multiscale(batch_size, device)
        else:
            raise ValueError(f"Unknown noise strategy: {self.strategy}")

    def _sample_pure(self, batch_size: int, device: torch.device) -> Tensor:
        """纯高斯噪声（与 DiffusionDet 一致）"""
        return torch.randn(batch_size, self.num_proposals, 4, device=device)

    def _sample_grid(self, batch_size: int, device: torch.device) -> Tensor:
        """网格初始化 + 高斯扰动

        在归一化空间 [0, 1] 中均匀撒点，转换到扩散空间 [-snr, snr]，
        再加高斯扰动。
        """
        N = self.num_proposals
        # 计算网格大小 (近似正方形)
        grid_h = int(math.sqrt(N))
        grid_w = N // grid_h
        actual_n = grid_h * grid_w

        # 网格中心 (归一化坐标 [0, 1])
        cx = torch.linspace(0, 1, grid_w + 2, device=device)[1:-1]
        cy = torch.linspace(0, 1, grid_h + 2, device=device)[1:-1]
        cx, cy = torch.meshgrid(cx, cy, indexing="xy")
        centers = torch.stack([cx.flatten(), cy.flatten()], dim=-1)[:actual_n]
        # (actual_n, 2) — cx, cy

        # 默认宽高 (适中的默认尺寸)
        default_wh = torch.full((actual_n, 2), 0.1, device=device)

        # 组合为 cxcywh
        boxes = torch.cat([centers, default_wh], dim=-1)  # (actual_n, 4)

        # 补足到 num_proposals
        if actual_n < N:
            extra = torch.rand(N - actual_n, 4, device=device)
            extra[:, 2:] = 0.1  # 默认宽高
            boxes = torch.cat([boxes, extra], dim=0)

        # [0, 1] 空间 → 扩散空间 [-snr, snr]
        boxes = (boxes * 2 - 1) * self.snr_scale

        # 加高斯扰动 (仅对中心坐标和宽高分别控制)
        noise = self.noise_scale * torch.randn(batch_size, N, 4, device=device)
        boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1).clone() + noise

        return boxes

    def _sample_grid_multiscale(
        self, batch_size: int, device: torch.device
    ) -> Tensor:
        """多尺度网格: 不同尺度的框交错排列"""
        N = self.num_proposals
        scales = [0.05, 0.1, 0.2, 0.4]
        per_scale = N // len(scales)

        all_boxes = []
        for s_idx, scale in enumerate(scales):
            n = per_scale if s_idx < len(scales) - 1 else N - per_scale * (len(scales) - 1)
            grid_h = int(math.sqrt(n))
            grid_w = n // grid_h
            actual_n = grid_h * grid_w

            cx = torch.linspace(0, 1, grid_w + 2, device=device)[1:-1]
            cy = torch.linspace(0, 1, grid_h + 2, device=device)[1:-1]
            cx, cy = torch.meshgrid(cx, cy, indexing="xy")
            centers = torch.stack([cx.flatten(), cy.flatten()], dim=-1)[:actual_n]

            wh = torch.full((actual_n, 2), scale, device=device)
            boxes = torch.cat([centers, wh], dim=-1)

            # 补足
            if actual_n < n:
                extra = torch.rand(n - actual_n, 4, device=device)
                extra[:, 2:] = scale
                boxes = torch.cat([boxes, extra], dim=0)

            all_boxes.append(boxes)

        boxes = torch.cat(all_boxes, dim=0)[:N]

        # → 扩散空间
        boxes = (boxes * 2 - 1) * self.snr_scale

        noise = self.noise_scale * torch.randn(batch_size, N, 4, device=device)
        boxes = boxes.unsqueeze(0).expand(batch_size, -1, -1) + noise

        return boxes
