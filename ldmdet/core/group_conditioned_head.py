"""组条件化 Single Head — 方向五: 组条件化分层分类

扩展 SingleDiffusionDetHead 的 AdaLN-Zero, 将组嵌入注入扩散过程,
使同组框的生成更聚焦 (同组框形态相似).

数学: adaln_params = MLP(time_emb + group_emb)
组嵌入: g_emb = Embed(g), g ∈ {0, ..., 7}

若方向五废弃, 删除本文件 + tests/unit/test_hierarchical_classification.py 即可回滚.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldmdet.core.single_head import SingleDiffusionDetHead
from ldmdet.utils.constants import NUM_CHROMO_GROUPS


class GroupConditionedSingleHead(SingleDiffusionDetHead):
    """组条件化 Single Head.

    在 AdaLN-Zero 中注入组嵌入, 使扩散过程组感知.
    仅在 time_conditioning='adaln_zero' 时生效.

    Args:
        *args, **kwargs: 传给父类 SingleDiffusionDetHead
        num_groups: 组数 (默认 8)
        group_emb_dim: 组嵌入维度 (默认 feat_channels * 4, 与 time_emb 一致)
    """

    def __init__(
        self,
        *args,
        num_groups: int = NUM_CHROMO_GROUPS,
        group_emb_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # 仅在 adaln_zero 模式下启用组条件化
        if self.time_conditioning != 'adaln_zero':
            raise ValueError(
                f"GroupConditionedSingleHead 仅支持 time_conditioning='adaln_zero', "
                f"got '{self.time_conditioning}'"
            )

        feat_channels = self.feat_channels
        if group_emb_dim is None:
            group_emb_dim = feat_channels * 4

        self.num_groups = num_groups
        self.group_emb_dim = group_emb_dim

        # 组嵌入表
        self.group_embedding = nn.Embedding(num_groups, group_emb_dim)

        # 替换 adaln_mlp: 输入维度 = time_emb_dim + group_emb_dim
        # 原: adaln_mlp = Sequential(SiLU, Linear(feat*4, feat*4*6))
        # 新: adaln_mlp = Sequential(SiLU, Linear(feat*4 + group_emb_dim, feat*4*6))
        self.adaln_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_channels * 4 + group_emb_dim, feat_channels * 6),
        )
        # AdaLN-Zero: 最后一层零初始化
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)

    def _forward_adaln_zero(
        self,
        proposals: torch.Tensor,
        roi_features: torch.Tensor,
        time_emb: torch.Tensor,
        bs: int,
        num_boxes: int,
        group_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """组条件化 AdaLN-Zero.

        Args:
            proposals: [bs, num_boxes, feat_channels] proposal 特征
            roi_features: ROI 特征
            time_emb: [bs, feat_channels*4] 时间嵌入
            bs: batch size
            num_boxes: 每张图的框数
            group_labels: [bs, num_boxes] 每个框的组标签, 若 None 则用 0 (无条件)

        Returns:
            obj_flat: [bs*num_boxes, feat_channels] 输出特征
        """
        device = time_emb.device

        if group_labels is None:
            group_labels = torch.zeros(
                bs, num_boxes, dtype=torch.long, device=device
            )

        # 组嵌入: [bs, num_boxes, group_emb_dim]
        group_emb = self.group_embedding(group_labels)

        # 广播 time_emb 到每个框: [bs, 1, feat*4] → [bs, num_boxes, feat*4]
        time_emb_expanded = time_emb.unsqueeze(1).expand(-1, num_boxes, -1)

        # 融合: [bs, num_boxes, feat*4 + group_emb_dim]
        cond_emb = torch.cat([time_emb_expanded, group_emb], dim=-1)

        # reshape for adaln_mlp
        proposals = proposals.view(bs, num_boxes, self.feat_channels).permute(1, 0, 2)
        cond_flat = cond_emb.reshape(bs * num_boxes, -1)
        adaln_params = self.adaln_mlp(cond_flat)
        adaln_params = adaln_params.view(num_boxes, bs, -1)

        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = adaln_params.chunk(6, dim=-1)

        # 后续与父类相同, 但用 reshape 替代 view 以确保连续性
        proposals_flat = proposals.reshape(num_boxes * bs, self.feat_channels)
        q_modulated = F.layer_norm(proposals_flat, [self.feat_channels]) * (
            1 + gamma1.reshape(-1, self.feat_channels)
        ) + beta1.reshape(-1, self.feat_channels)
        q_modulated = q_modulated.view(num_boxes, bs, self.feat_channels)
        attn_out, _ = self._self_attn(q_modulated)
        attn_out_flat = attn_out.reshape(num_boxes * bs, self.feat_channels)
        proposals_flat = proposals_flat + alpha1.reshape(-1, self.feat_channels) * attn_out_flat

        proposals = proposals_flat.view(num_boxes, bs, self.feat_channels)
        proposals = proposals.permute(1, 0, 2).reshape(1, bs * num_boxes, self.feat_channels)
        inst_out = self.inst_interact(proposals, roi_features)
        proposals = proposals + self.dropout2(inst_out)
        obj_features = self.norm2(proposals)

        obj_flat = obj_features.squeeze(0)
        ffn_input = F.layer_norm(obj_flat, [self.feat_channels]) * (
            1 + gamma2.reshape(-1, self.feat_channels)
        ) + beta2.reshape(-1, self.feat_channels)
        ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
        obj_flat = obj_flat + alpha2.reshape(-1, self.feat_channels) * ffn_out
        return obj_flat

    def forward(
        self,
        features: torch.Tensor,
        bboxes: torch.Tensor,
        proposals: Optional[torch.Tensor],
        pooler: nn.Module,
        time_emb: torch.Tensor,
        group_labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        """前向传播 (支持 group_labels 注入).

        Args:
            features: FPN 特征
            bboxes: [bs, num_boxes, 4] 框
            proposals: proposal 特征
            pooler: ROI extractor
            time_emb: [bs, feat*4] 时间嵌入
            group_labels: [bs, num_boxes] 组标签, 若 None 则用 0

        Returns:
            (class_logits, pred_bboxes, curr_proposals)
        """
        from ldmdet.utils.box_ops import bbox2roi

        bs, num_boxes = bboxes.shape[:2]
        rois = bbox2roi([bboxes[i] for i in range(bs)])
        roi_features = pooler(features, rois)

        if proposals is None:
            proposals = roi_features.flatten(2).mean(-1).view(
                bs, num_boxes, self.feat_channels
            )

        roi_features = roi_features.view(
            bs * num_boxes, self.feat_channels, -1
        ).permute(2, 0, 1)

        # 调用组条件化的 _forward_adaln_zero
        fc_feature = self._forward_adaln_zero(
            proposals, roi_features, time_emb, bs, num_boxes, group_labels
        )
        return self._predict(fc_feature, bboxes, bs, num_boxes)
