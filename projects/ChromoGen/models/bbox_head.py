"""BBox扩散生成头

并行的bbox扩散分支，与UNet共享特征，使用Rectified Flow生成bbox坐标。
通过配置 enable_bbox_head 控制是否启用此分支参与训练。

启用时：联合训练图像生成 + bbox生成
禁用时：仅训练图像生成分支（Phase 1）
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class BBoxDiffusionHead(nn.Module):
    """BBox扩散生成头

    从UNet bottleneck特征中提取信息，通过Rectified Flow生成bbox坐标。

    流程：
      1. UNet bottleneck特征 → 全局池化 → 特征向量
      2. 特征向量 + 时间步 → MLP → 预测x0 (bbox坐标)
      3. Rectified Flow: x_t = (1-t)*x_0 + t*x_1, 预测x_0
    """

    def __init__(
        self,
        in_channels: int = 1280,  # UNet bottleneck通道数
        feat_channels: int = 512,
        num_classes: int = 24,
        num_proposals: int = 100,  # 最大生成bbox数
        num_heads: int = 8,
        num_layers: int = 3,
        snr_scale: float = 2.0,
        num_cls_convs: int = 1,
        num_reg_convs: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.snr_scale = snr_scale

        # 从UNet特征中提取全局特征
        self.feat_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, feat_channels),
            nn.GELU(),
            nn.Linear(feat_channels, feat_channels),
        )

        # 时间步embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(feat_channels),
            nn.Linear(feat_channels, feat_channels),
            nn.GELU(),
        )

        # 条件embedding (从condition encoder获取)
        self.cond_proj = nn.Sequential(
            nn.Linear(feat_channels, feat_channels),
            nn.GELU(),
        )

        # Proposal queries: 可学习的查询向量
        self.proposal_queries = nn.Parameter(
            torch.randn(1, num_proposals, feat_channels) * 0.02
        )

        # Transformer decoder layers: queries attend to features
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_channels,
            nhead=num_heads,
            dim_feedforward=feat_channels * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 分类头
        cls_layers = []
        for _ in range(num_cls_convs):
            cls_layers.extend(
                [nn.Linear(feat_channels, feat_channels), nn.GELU()]
            )
        cls_layers.append(nn.Linear(feat_channels, num_classes))
        self.cls_head = nn.Sequential(*cls_layers)

        # 回归头 (预测cxcywh格式的x0)
        reg_layers = []
        for _ in range(num_reg_convs):
            reg_layers.extend(
                [nn.Linear(feat_channels, feat_channels), nn.GELU()]
            )
        reg_layers.append(nn.Linear(feat_channels, 4))
        self.reg_head = nn.Sequential(*reg_layers)

        # LayerNorm
        self.norm = nn.LayerNorm(feat_channels)

    def forward(
        self,
        bottleneck_feat: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        gt_bboxes: Optional[torch.Tensor] = None,
        gt_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            bottleneck_feat: UNet bottleneck特征 [B, C, H, W]
            condition: 条件embedding [B, seq_len, feat_channels]
            t: 时间步 [B] 范围[0,1]
            gt_bboxes: GT bbox列表 (训练时), 每个元素 [N_i, 4] (cxcywh归一化)
            gt_labels: GT标签列表 (训练时), 每个元素 [N_i]
        Returns:
            训练时: dict with 'loss_bbox', 'loss_cls', 'loss_total'
            推理时: dict with 'pred_bboxes', 'pred_labels', 'pred_scores'
        """
        B = bottleneck_feat.shape[0]

        # 1. 提取全局特征
        global_feat = self.feat_proj(bottleneck_feat)  # [B, feat_channels]

        # 2. 时间步embedding
        t_emb = self.time_embed(t)  # [B, feat_channels]

        # 3. 条件特征 (取全局token)
        cond_feat = self.cond_proj(condition[:, 0, :])  # [B, feat_channels]

        # 4. 融合: global + time + condition → memory
        memory_feat = (global_feat + t_emb + cond_feat).unsqueeze(
            1
        )  # [B, 1, feat_channels]

        # 5. Proposal queries + time conditioning
        queries = self.proposal_queries.expand(
            B, -1, -1
        )  # [B, num_proposals, feat_channels]
        # 将时间信息加到queries上
        queries = queries + t_emb.unsqueeze(1)

        # 6. Transformer decoder
        decoded = self.decoder(
            queries, memory_feat
        )  # [B, num_proposals, feat_channels]
        decoded = self.norm(decoded)

        # 7. 预测
        cls_logits = self.cls_head(decoded)  # [B, num_proposals, num_classes]
        pred_x0 = self.reg_head(
            decoded
        )  # [B, num_proposals, 4] (cxcywh归一化)

        if self.training and gt_bboxes is not None and gt_labels is not None:
            return self._compute_loss(
                pred_x0, cls_logits, t, gt_bboxes, gt_labels
            )
        else:
            return self._inference_output(pred_x0, cls_logits)

    def _compute_loss(
        self,
        pred_x0: torch.Tensor,
        cls_logits: torch.Tensor,
        t: torch.Tensor,
        gt_bboxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """计算训练损失

        使用Rectified Flow: 给定GT x_0和噪声x_1，计算x_t，然后让模型从x_t预测x_0。
        使用匈牙利匹配将GT与proposals最优配对。
        """
        all_loss_bbox = []
        all_loss_cls = []

        for i in range(len(gt_bboxes)):
            n_gt = gt_bboxes[i].shape[0]
            if n_gt == 0:
                continue

            # GT bbox: [N_gt, 4] cxcywh归一化 → 扩散空间
            x0_gt = (gt_bboxes[i] * 2 - 1) * self.snr_scale  # [N_gt, 4]

            # 噪声
            x1_noise = torch.randn_like(x0_gt)

            # 当前时间步
            t_i = t[i]

            # Rectified Flow: x_t = (1-t)*x_0 + t*x_1
            t_view = t_i.view(-1, *([1] * (x0_gt.dim() - 1)))
            x_t = (1.0 - t_view) * x0_gt + t_view * x1_noise

            # 匈牙利匹配: 基于L1代价矩阵将GT与proposals最优配对
            n_proposals = min(self.num_proposals, pred_x0.shape[1])
            cost_matrix = (
                torch.cdist(
                    pred_x0[i, :n_proposals].detach(),
                    x0_gt.detach(),
                    p=1,
                )
                .cpu()
                .numpy()
            )  # [n_proposals, N_gt]

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            # row_indices: 匹配的proposal索引, col_indices: 匹配的GT索引

            matched_pred = pred_x0[i, row_indices]  # [N_matched, 4]
            matched_cls = cls_logits[
                i, row_indices
            ]  # [N_matched, num_classes]
            matched_gt = x0_gt[col_indices]  # [N_matched, 4]
            matched_labels = gt_labels[i][col_indices]  # [N_matched]

            # Bbox L1 loss
            loss_bbox = F.l1_loss(matched_pred, matched_gt)

            # 分类loss
            loss_cls = F.cross_entropy(matched_cls, matched_labels)

            all_loss_bbox.append(loss_bbox)
            all_loss_cls.append(loss_cls)

        if len(all_loss_bbox) == 0:
            return {
                'loss_bbox': torch.tensor(
                    0.0, device=pred_x0.device, requires_grad=True
                ),
                'loss_cls': torch.tensor(
                    0.0, device=pred_x0.device, requires_grad=True
                ),
                'loss_total': torch.tensor(
                    0.0, device=pred_x0.device, requires_grad=True
                ),
            }

        loss_bbox = torch.stack(all_loss_bbox).mean()
        loss_cls = torch.stack(all_loss_cls).mean()
        loss_total = loss_bbox + 0.5 * loss_cls

        return {
            'loss_bbox': loss_bbox,
            'loss_cls': loss_cls,
            'loss_total': loss_total,
        }

    def _inference_output(
        self,
        pred_x0: torch.Tensor,
        cls_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """推理输出

        将预测的x0从扩散空间转回归一化cxcywh
        """
        # 扩散空间 → 归一化cxcywh
        bboxes = ((pred_x0 / self.snr_scale) + 1) / 2  # [B, num_proposals, 4]
        bboxes = bboxes.clamp(0, 1)

        scores = torch.sigmoid(cls_logits)  # [B, num_proposals, num_classes]
        labels = scores.argmax(dim=-1)  # [B, num_proposals]
        confidences = scores.max(dim=-1)[0]  # [B, num_proposals]

        return {
            'pred_bboxes': bboxes,
            'pred_labels': labels,
            'pred_scores': confidences,
        }


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步embedding"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] 时间步
        Returns:
            emb: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        if emb.shape[-1] < self.dim:
            emb = torch.cat(
                [
                    emb,
                    torch.zeros(
                        *emb.shape[:-1],
                        self.dim - emb.shape[-1],
                        device=device,
                    ),
                ],
                dim=-1,
            )

        return emb
