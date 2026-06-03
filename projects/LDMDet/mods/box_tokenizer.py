import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BoxTokenizer(nn.Module):
    """将 bbox 坐标转换为 Box Tokens

    策略: 双线性插值采样 + 位置编码 + 层级编码
    备选: 可学习 Query + 坐标编码 (通过 init_mode 参数切换)
    """

    def __init__(
        self,
        feat_channels: int = 256,
        num_fpn_levels: int = 4,
        init_mode: str = 'bilinear',
    ):
        """
        Args:
            feat_channels: 特征通道数
            num_fpn_levels: FPN 层级数
            init_mode: "bilinear" (双线性插值) 或 "learnable" (可学习 Query)
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
            self.query_embed = None

    def _assign_fpn_level(
        self, bboxes: Tensor, spatial_shapes: List[Tuple[int, int]]
    ) -> Tensor:
        """根据 bbox 面积分配 FPN 层级

        遵循 FPN 的 k = floor(k0 + log2(sqrt(wh) / 224)) 规则

        Args:
            bboxes: (bs, N, 4) xyxy 归一化坐标 [0,1]
            spatial_shapes: [(H_0, W_0), ...] 各层分辨率

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

    def _bilinear_sample(
        self,
        fpn_features: List[Tensor],
        bboxes: Tensor,
        level_indices: Tensor,
    ) -> Tensor:
        """在 FPN 对应层级上双线性插值采样 bbox 中心点特征

        Args:
            fpn_features: List[(bs, C, H_l, W_l)] P2-P5 特征
            bboxes: (bs, N, 4) xyxy 归一化坐标 [0,1]
            level_indices: (bs, N) 层级索引

        Returns:
            sampled: (bs, N, C)
        """
        bs, num_boxes, _ = bboxes.shape
        device = bboxes.device
        C = fpn_features[0].shape[1]

        cx = (bboxes[..., 0] + bboxes[..., 2]) / 2
        cy = (bboxes[..., 1] + bboxes[..., 3]) / 2

        sampled = torch.zeros(bs, num_boxes, C, device=device)

        for b in range(bs):
            for level_idx, feat in enumerate(fpn_features):
                mask = level_indices[b] == level_idx
                if not mask.any():
                    continue

                grid_x = cx[b, mask] * 2 - 1
                grid_y = cy[b, mask] * 2 - 1

                n_pts = mask.sum().item()
                grid = torch.stack([grid_x, grid_y], dim=-1)
                grid = grid.reshape(1, 1, n_pts, 2)

                feat_b = feat[b:b+1]
                out = F.grid_sample(
                    feat_b,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )
                out = out.squeeze(2).squeeze(0).T

                sampled[b, mask] = out

        return sampled

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
        level_indices = self._assign_fpn_level(bboxes, None)

        if self.init_mode == 'bilinear':
            sampled_feat = self._bilinear_sample(
                fpn_features, bboxes, level_indices
            )
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

        # #region debug-point B:box-tokens
        try:
            from .dit_head import _dbg_post  # lazy: avoid circular import
            if not hasattr(self, '_dit_dbg_step_b'):
                self._dit_dbg_step_b = 0
            self._dit_dbg_step_b += 1
            if self._dit_dbg_step_b % 50 == 1:
                _bt = box_tokens.detach()
                # diversity: std across instance dimension (high = distinguishable)
                _bt_std_inst = float(_bt[0].std(dim=0).mean())
                _bt_std_feat = float(_bt[0].std(dim=1).mean())
                # pairwise L2 distance between consecutive instances (mean)
                if _bt.shape[1] > 1:
                    _a = _bt[0, :-1]
                    _b = _bt[0, 1:]
                    _pd = float((_a - _b).norm(dim=-1).mean())
                else:
                    _pd = 0.0
                # level-index distribution
                _lvl_hist = [int((level_indices[0] == k).sum().item())
                             for k in range(self.num_fpn_levels)]
                _dbg_post(
                    'B',
                    'box_tokenizer.py:BoxTokenizer.forward',
                    f'[DEBUG] box_tokens diversity iter={self._dit_dbg_step_b}',
                    {
                        'bt_std_across_instances': _bt_std_inst,
                        'bt_std_across_features': _bt_std_feat,
                        'bt_pairwise_l2_consec': _pd,
                        'level_hist': _lvl_hist,
                        'init_mode': self.init_mode,
                    },
                )
        except Exception:
            pass
        # #endregion

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
        reference_points: (bs, N, num_levels, 2) 归一化中心坐标
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
    return reference_points.unsqueeze(2).expand(
        -1, -1, num_levels, -1
    )
