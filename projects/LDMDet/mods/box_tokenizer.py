from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class BoxTokenizer(nn.Module):
    """将 bbox 坐标转换为 Box Tokens

    策略: 坐标位置编码 + 层级编码
    备选: 可学习 Query + 坐标编码 (通过 init_mode 参数切换)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        num_fpn_levels: int = 4,
        init_mode: str = 'zero',
    ):
        """
        Args:
            feat_channels: 特征通道数
            num_fpn_levels: FPN 层级数
            init_mode: "zero" (零初始化) 或 "learnable" (可学习 Query)
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.num_fpn_levels = num_fpn_levels
        self.init_mode = init_mode

        self.bbox_pos_embed = nn.Sequential(
            nn.Linear(4, feat_channels),
            nn.ReLU(),
            nn.Linear(feat_channels, feat_channels),
        )
        self.level_embed = nn.Embedding(num_fpn_levels, feat_channels)

        if init_mode == 'learnable':
            self.query_embed = nn.Parameter(torch.randn(1, 100, feat_channels))

    def _assign_fpn_level(
        self, bboxes: Tensor
    ) -> Tensor:
        """根据 bbox 面积分配 FPN 层级

        遵循 FPN 的 k = floor(k0 + log2(sqrt(wh) / 224)) 规则

        Args:
            bboxes: (bs, N, 4) xyxy 归一化坐标 [0,1]

        Returns:
            level_indices: (bs, N) 层级索引
        """
        bs, num_boxes, _ = bboxes.shape
        device = bboxes.device

        widths = (bboxes[..., 2] - bboxes[..., 0]).clamp(min=1e-8)
        heights = (bboxes[..., 3] - bboxes[..., 1]).clamp(min=1e-8)
        areas = widths * heights

        k0 = 4
        k = k0 + torch.log2(areas.sqrt() + 1e-8)
        k = k.nan_to_num(0).clamp(0, self.num_fpn_levels - 1).long()

        return k

    def forward(
        self,
        bboxes: Tensor,
        fpn_features: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """将 bbox 坐标转换为 Box Tokens

        Args:
            bboxes: (bs, N, 4) xyxy 归一化坐标 [0,1]
            fpn_features: List[(bs, C, H_l, W_l)] P2-P5 特征

        Returns:
            box_tokens: (bs, N, C)
            level_indices: (bs, N) FPN 层级索引
        """
        level_indices = self._assign_fpn_level(bboxes)

        if self.init_mode == 'learnable':
            bs = bboxes.shape[0]
            sampled_feat = self.query_embed.expand(bs, -1, -1)
            num_proposals = bboxes.shape[1]
            if sampled_feat.shape[1] != num_proposals:
                if sampled_feat.shape[1] > num_proposals:
                    sampled_feat = sampled_feat[:, :num_proposals]
                else:
                    repeat = (num_proposals // sampled_feat.shape[1]) + 1
                    sampled_feat = sampled_feat.repeat(1, repeat, 1)[
                        :, :num_proposals
                    ]
        else:
            sampled_feat = torch.zeros(
                bboxes.shape[0],
                bboxes.shape[1],
                self.feat_channels,
                device=bboxes.device,
            )

        pos_embed = self.bbox_pos_embed(bboxes)
        lvl_embed = self.level_embed(level_indices)

        box_tokens = sampled_feat + pos_embed + lvl_embed

        # 出口防护: 确保 token 不含 NaN/Inf，防止传播到 DiTBlock
        box_tokens = torch.nan_to_num(box_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        return box_tokens, level_indices


def bbox_to_reference_points(
    bboxes: Tensor,
    img_metas: Optional[list] = None,
) -> Tensor:
    """将 xyxy bbox 转换为归一化参考点 (中心坐标)

    Args:
        bboxes: (bs, N, 4) xyxy 格式，归一化到 [0,1]
        img_metas: 未使用，保留接口兼容

    Returns:
        reference_points: (bs, N, 2) 归一化中心坐标
    """
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2
    ref = torch.stack([cx, cy], dim=-1)
    return ref


def reference_points_with_levels(
    reference_points: Tensor,
    num_levels: int,
) -> Tensor:
    """扩展参考点到所有 FPN 层级

    Args:
        reference_points: (bs, N, 2)
        num_levels: FPN 层级数

    Returns:
        (bs, N, num_levels, 2)
    """
    return reference_points.unsqueeze(2).expand(-1, -1, num_levels, -1)
