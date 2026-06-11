import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .utils import sanitize_features


class BoxTokenizer(nn.Module):
    """将 bbox 坐标转换为 Box Tokens

    策略: 坐标位置编码 + 层级编码
    备选: 可学习 Query + 坐标编码 (通过 init_mode 参数切换)

    init_mode:
        - "zero": 零初始化 token embedding
        - "learnable" / "query": 可学习 Query embedding
        - "spatial_prior": 均匀分布锚点位置编码 (DAB-DETR 思路)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        num_fpn_levels: int = 4,
        init_mode: str = 'zero',
        num_proposals: int = 100,
    ):
        """
        Args:
            feat_channels: 特征通道数
            num_fpn_levels: FPN 层级数
            init_mode: "zero" / "learnable" / "query" / "spatial_prior"
            num_proposals: 提案数量 (spatial_prior 模式需要用此参数生成锚点)
        """
        super().__init__()
        self.feat_channels = feat_channels
        self.num_fpn_levels = num_fpn_levels
        self.init_mode = init_mode
        self.num_proposals = num_proposals

        self.bbox_pos_embed = nn.Sequential(
            nn.Linear(4, feat_channels),
            nn.ReLU(),
            nn.Linear(feat_channels, feat_channels),
        )
        # LayerNorm 确保 pos_embed 与其他组件 magnitude 匹配
        # 诊断发现: 不加时 pos_embed magnitude~1.5, 其他组件~20
        # 导致所有 token 相似 (cos_sim=0.99), Self-Attention 前就已无法区分不同框
        self.pos_norm = nn.LayerNorm(feat_channels)
        self.level_embed = nn.Embedding(num_fpn_levels, feat_channels)

        if init_mode in ('learnable', 'query'):
            self.query_embed = nn.Parameter(
                torch.randn(1, num_proposals, feat_channels)
            )

        if init_mode == 'spatial_prior':
            # DAB-DETR 思路: 生成均匀分布在全图的锚点框 (cx, cy, w, h)
            grid_size = int(math.ceil(math.sqrt(num_proposals)))
            cx = torch.linspace(0.05, 0.95, grid_size)
            cy = torch.linspace(0.05, 0.95, grid_size)
            cx_grid, cy_grid = torch.meshgrid(cx, cy, indexing='ij')
            anchors = torch.stack(
                [
                    cx_grid.flatten()[:num_proposals],
                    cy_grid.flatten()[:num_proposals],
                    torch.full((num_proposals,), 0.1),
                    torch.full((num_proposals,), 0.1),
                ],
                dim=-1,
            )  # (num_proposals, 4) 归一化坐标
            self.register_buffer('anchor_boxes', anchors)
            # 独立的 anchor 位置编码，避免与 bbox_pos_embed 重复叠加
            self.anchor_pos_embed = nn.Sequential(
                nn.Linear(4, feat_channels),
                nn.ReLU(),
                nn.Linear(feat_channels, feat_channels),
            )

        # ---- 初始化平衡 ----
        # level_embed (nn.Embedding) 默认 N(0,1) init，std≈1.0
        # bbox_pos_embed (2-layer MLP, Xavier init) 输出 std≈0.02-0.05
        # anchor_pos_embed 同理 std≈0.02-0.05
        # 不加处理时 level_embed 贡献 ~50x 于其他分量，导致 box_tokens ≈ level_embed
        # 所有同 FPN level 的 proposal 得到相同 token → 100 个框全同 → NMS 塌缩为 1 框
        # → GIoU 梯度同质化 → displacement loss 被淹没 → mAP=0
        nn.init.normal_(self.level_embed.weight, std=0.01)
        # 同时把 anchor_pos_embed 最后一层 bias 初始化为 0，确保
        # spatial prior 不会在大特征空间中压制 bbox 变化信号
        nn.init.zeros_(self.anchor_pos_embed[-1].bias)
        nn.init.zeros_(self.bbox_pos_embed[-1].bias)


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

        bs = bboxes.shape[0]
        num_proposals = bboxes.shape[1]

        if self.init_mode in ('learnable', 'query'):
            sampled_feat = self.query_embed.expand(bs, -1, -1)
            if sampled_feat.shape[1] != num_proposals:
                if sampled_feat.shape[1] > num_proposals:
                    sampled_feat = sampled_feat[:, :num_proposals]
                else:
                    repeat = (num_proposals // sampled_feat.shape[1]) + 1
                    sampled_feat = sampled_feat.repeat(1, repeat, 1)[
                        :, :num_proposals
                    ]
        elif self.init_mode == 'spatial_prior':
            # DAB-DETR: 用锚点框的位置编码作为 token 先验
            anchors = self.anchor_boxes[:num_proposals].unsqueeze(0).expand(
                bs, -1, -1
            )
            sampled_feat = self.anchor_pos_embed(anchors)
        else:
            sampled_feat = torch.zeros(
                bs,
                num_proposals,
                self.feat_channels,
                device=bboxes.device,
            )

        pos_embed = self.bbox_pos_embed(bboxes)
        pos_embed = self.pos_norm(pos_embed)  # 归一化使 magnitude ~sqrt(C)，与其他组件匹配
        lvl_embed = self.level_embed(level_indices)

        box_tokens = sampled_feat + pos_embed + lvl_embed

        # 出口防护: 确保 token 不含 NaN/Inf，防止传播到 DiTBlock
        box_tokens = sanitize_features(box_tokens)

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
