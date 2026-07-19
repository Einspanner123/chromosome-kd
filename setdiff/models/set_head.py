"""JointDiffusionHead — diffusion over N boxes as a 4N joint state.

Key difference from DiffusionDet/DiffuDETR:
- DiffusionDet: per-proposal diffusion, v_i depends only on x_i.
- SetDiff: joint diffusion, v_i can depend on x_j (via Self-Attention).

Training flow:
1. Sample noise z ~ N(0, I) [N, 4].
2. Global coupled matching: match z to GT boxes (one-to-one).
3. Forward diffusion: x_t = (1-t)*x_0_matched + t*z.
4. Predict x_0: x_0_pred = SetEncoder(x_t, t, image_features).
5. Loss: cls (focal) + bbox (L1) + giou (对齐 LDMDet/DiffusionDet 2:5:2).

Inference flow:
1. Sample noise z ~ N(0, I) [N, 4].
2. Iterative denoising: x_t → SetEncoder → x_0_pred → step → x_{t-dt}.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from setdiff.core.set_encoder import SetEncoder
from setdiff.criterion.set_loss import SetCriterion
from setdiff.diffusion.set_rf import SetRectifiedFlow
from setdiff.matching.hungarian import HungarianMatcher
from setdiff.matching.random_matcher import RandomMatcher


class JointDiffusionHead(nn.Module):
    """Joint Diffusion Head: diffusion over N boxes as a 4N joint state."""

    def __init__(
        self,
        num_queries: int,
        feat_channels: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        snr_scale: float = 2.0,
        num_sample_steps: int = 4,
        sampler: str = 'euler',
        box_renewal: bool = True,
        score_thr: float = 0.3,
        min_keep: int = 75,
        matcher_type: str = 'hungarian',
        unmatched_strategy: str = 'noise',
    ):
        super().__init__()
        self.num_queries = num_queries
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.num_sample_steps = num_sample_steps
        self.sampler = sampler
        # snr_scale: GT 从 [0,1] 缩放到 [-snr_scale, +snr_scale] 匹配 N(0,1) 噪声
        # 由 SetDiffDetector 在 loss()/predict() 中应用 (数据预处理, 不参与 RF 公式)
        self.snr_scale = snr_scale

        # Box renewal (推理时): 低置信 slot 重置为 randn.
        # 核心修复: 验证 unmatched slot 发散假设 —
        # 训练时 unmatched slot 的 x_t 恒为 noise 且 pred_boxes 无监督,
        # 推理时这些 slot 输出垃圾 pred_boxes, Euler step 把 x_t 推到 OOD,
        # 通过 self-attention 污染 matched slot → mAP=0.
        # 修复: 每步 Euler 后, 低置信 slot 重置为 randn (回到训练分布),
        # 保留高置信 slot 继续迭代. 对齐 LDMDet apply_box_renewal.
        self.box_renewal = box_renewal
        self.score_thr = score_thr
        # min_keep: 至少保留的 slot 数 (防止全被重置).
        # 默认 num_queries//4=75 (300 queries 时).
        self.min_keep = min_keep

        # Set encoder (joint function approximator)
        self.encoder = SetEncoder(
            num_queries=num_queries,
            feat_channels=feat_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
        )

        # Set-level Rectified Flow
        self.rf = SetRectifiedFlow(snr_scale=snr_scale)

        # Matcher 选择 (方案 A/B):
        #   - 'hungarian' (默认): global coupled one-to-one matching.
        #     unmatched_strategy='noise' (原行为, mAP=0 根因) |
        #     'random_gt' (方案 B, unmatched slot 分配随机 GT)
        #   - 'random' (方案 A): 所有 slot 随机分配 GT, 对齐 LDMDet
        #     (unmatched_strategy 对 random matcher 无意义, 静默忽略)
        if matcher_type == 'hungarian':
            self.matcher = HungarianMatcher(
                unmatched_strategy=unmatched_strategy
            )
        elif matcher_type == 'random':
            if unmatched_strategy != 'noise':
                import warnings
                warnings.warn(
                    f"matcher_type='random' 忽略 unmatched_strategy="
                    f"'{unmatched_strategy}' (random matcher 无 unmatched "
                    f"slot 概念). 建议移除该参数避免误配.",
                    stacklevel=2,
                )
            self.matcher = RandomMatcher()
        else:
            raise ValueError(
                f"matcher_type 必须是 'hungarian' 或 'random', "
                f"实际: {matcher_type}"
            )

        # Loss
        self.criterion = SetCriterion(
            num_classes=num_classes, snr_scale=snr_scale
        )

        # Time embedding: Sinusoidal → small MLP for learnable projection
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, feat_channels),
            nn.SiLU(),
            nn.Linear(feat_channels, feat_channels),
        )

    def forward(
        self,
        image_features: Tensor,
        gt_boxes: Optional[List[Tensor]] = None,
        gt_labels: Optional[List[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Training forward pass.

        Args:
            image_features: [B, HW, C] flattened multi-scale features.
            gt_boxes: list of [M_i, 4] GT boxes (cxcywh, diffusion space),
                None for inference.
            gt_labels: list of [M_i] GT labels, None for inference.

        Returns:
            Training: dict with loss terms (loss_cls, loss_bbox, loss_giou,
                loss).
            Inference: dict with 'pred_logits' and 'pred_boxes'.
        """
        if gt_boxes is not None and gt_labels is not None:
            return self._forward_train(image_features, gt_boxes, gt_labels)
        return self.predict(image_features)

    def _forward_train(
        self,
        image_features: Tensor,
        gt_boxes: List[Tensor],
        gt_labels: List[Tensor],
    ) -> Dict[str, Tensor]:
        B = image_features.shape[0]
        device = image_features.device

        # 1. Sample noise z ~ N(0, I) [B, N, 4]
        noise = torch.randn(B, self.num_queries, 4, device=device)

        # 2. Global coupled matching: match z to GT boxes (one-to-one)
        # coupling 阶段: 构造训练轨迹 x_t = (1-t)*x_0_matched + t*noise.
        # 方案 A (random): 所有 slot 随机分配 GT (对齐 LDMDet _couple_single_image).
        # 方案 B (hungarian+random_gt): matched slot 最优一对一, unmatched 随机 GT.
        # 关键: coupling 阶段保证所有 slot 在 [GT, noise] 插值轨迹上,
        # 避免训练-推理分布不匹配 (mAP=0 根因).
        matched_boxes, matched_labels = self.matcher.match_batch(
            noise, gt_boxes, gt_labels
        )

        # 3. Sample time t ~ U(0, 1) and forward diffusion
        t = torch.rand(B, device=device)
        x_t, _velocity = self.rf.q_sample(matched_boxes, noise, t)

        # 4. Time embedding (对齐 LDMDet: t * 1000 提高正弦嵌入分辨率)
        t_scaled = t * 1000.0
        t_emb = self.time_embed(t_scaled)  # [B, feat_channels]

        # 5. Predict x_0
        cls_logits, pred_boxes = self.encoder(
            x_t,
            t_emb,
            image_features,
            matched_mask=(matched_labels >= 0),
        )

        # 6. Compute loss (返回加权单项, 对齐 LDMDet)
        # 关键修复 1: 不添加 'loss' key (weighted total), 否则 mmengine parse_losses
        #   会 sum 所有含 'loss' 的 key, 导致 double-counting:
        #   实际 total = cls + bbox + giou + (2*cls + 5*bbox + 2*giou) = 3:6:3
        #   修复后: 每项预乘权重, parse_losses 直接 sum 得到 2*cls + 5*bbox + 2*giou.
        #
        # 关键修复 2 (2026-07-19): 传入原始 GT 给 criterion, 不传 coupling 阶段的
        #   matched_boxes/matched_labels. criterion 内部用 Hungarian 重新匹配
        #   pred_boxes 和 GT, num_pos=M (不是 N=300), 避免梯度稀释 37.5 倍.
        #   对齐 LDMDet criterion.py: coupling 阶段 (随机分配 GT) 与 loss 阶段
        #   (Hungarian 重匹配) 分离, 两阶段独立决策.
        outputs = {
            'pred_logits': cls_logits,
            'pred_boxes': pred_boxes,
        }
        loss_dict, _ = self.criterion(outputs, gt_boxes, gt_labels)
        # 权重预乘到单项 (mmengine parse_losses 直接 sum 各项)
        return {
            k: v * self.criterion.weight_dict.get(k, 1.0)
            for k, v in loss_dict.items()
        }

    @torch.no_grad()
    def predict(self, image_features: Tensor) -> Dict[str, Tensor]:
        """Inference: iterative denoising.

        Args:
            image_features: [B, HW, C] flattened multi-scale features.

        Returns:
            dict with 'pred_logits' [B, N, C] and 'pred_boxes' [B, N, 4].

        Box renewal (核心修复):
            每步 Euler 后 (非最后一步), 低置信 slot 的 x_t 重置为 randn.
            根因: 训练时 unmatched slot 的 x_t 恒为 noise, pred_boxes 无监督;
            推理时这些 slot 输出垃圾, Euler 把 x_t 推到 OOD, 污染整个 state.
            修复: 重置低置信 slot 为 randn, 让它们回到训练分布 (noise),
            保留高置信 slot 继续迭代. 对齐 LDMDet apply_box_renewal.
        """
        B = image_features.shape[0]
        device = image_features.device

        # 1. Sample noise z ~ N(0, I) [B, N, 4]
        x_t = torch.randn(B, self.num_queries, 4, device=device)

        # 2. Build time schedule: 1.0 → 0.0
        timesteps = torch.linspace(
            1.0, 0.0, self.num_sample_steps + 1, device=device
        )

        cls_logits = None
        pred_boxes = None
        for i in range(self.num_sample_steps):
            t_curr = float(timesteps[i].item())
            t_next = float(timesteps[i + 1].item())

            t = torch.full((B,), t_curr, device=device)
            # 对齐 LDMDet: t * 1000 提高正弦嵌入分辨率
            t_scaled = t * 1000.0
            t_emb = self.time_embed(t_scaled)

            cls_logits, pred_boxes = self.encoder(x_t, t_emb, image_features)

            # Euler step on joint state
            x_t = self.rf.step(x_t, pred_boxes, t_curr, t_next)

            # Box renewal: 非最后一步时, 低置信 slot 重置为 randn
            if self.box_renewal and i < self.num_sample_steps - 1:
                x_t = self._apply_box_renewal(x_t, cls_logits)

        return {
            'pred_logits': cls_logits,
            'pred_boxes': pred_boxes,
        }

    @torch.no_grad()
    def _apply_box_renewal(
        self,
        x_t: Tensor,
        cls_logits: Tensor,
    ) -> Tensor:
        """低置信 slot 重置为 randn, 保留高置信 slot.

        对齐 LDMDet apply_box_renewal (sampling.py:143-204):
            1. 计算 slot 置信度 = sigmoid(cls_logits).max(-1)
            2. 置信度 < score_thr 的 slot 重置为 randn
            3. 保证至少保留 min_keep 个 slot (topk 补足)

        核心目的: 让 unmatched slot 的 x_t 保持在训练分布 (noise) 内,
        避免发散到 OOD 后通过 self-attention 污染 matched slot.

        Args:
            x_t: [B, N, 4] 当前扩散状态 (Euler step 后).
            cls_logits: [B, N, C] 分类 logits (encoder 输出).

        Returns:
            x_t_new: [B, N, 4] renewal 后的状态.
        """
        B, N, _ = x_t.shape
        device = x_t.device

        # slot 置信度: [B, N]
        scores = torch.sigmoid(cls_logits).max(dim=-1)[0]
        x_t_new = x_t.clone()

        for b in range(B):
            keep = scores[b] > self.score_thr
            # 保证至少 min_keep 个 slot 保留 (topk 补足)
            if keep.sum() < self.min_keep:
                _, topk_idx = scores[b].topk(
                    min(self.min_keep, N)
                )
                keep[topk_idx] = True

            num_renew = (~keep).sum().item()
            if num_renew > 0:
                # 重置为 randn (回到训练 unmatched 分布)
                x_t_new[b, ~keep] = torch.randn(
                    num_renew, 4, device=device
                )

        return x_t_new
