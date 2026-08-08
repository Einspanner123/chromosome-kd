"""检测损失计算核心类"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import ops

from ldmdet.data.structures import InstanceData, ModelOutput
from ldmdet.diagnostics.instrumentation import probe
from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy

logger = logging.getLogger(__name__)


class DiffusionDetCriterion(nn.Module):
    """DiffusionDet 损失计算核心"""

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        loss_giou: nn.Module,
        deep_supervision: bool = True,
        quality_loss_weight: float = 0.25,
        quality_focal_alpha: float = 0.75,
        quality_focal_gamma: float = 2.0,
        # R3: v-prediction 等价的损失重加权
        # 理论 (theory_analysis_RF_DPM.md §4.3): L_v = (1/t²) L_x0 (L2 squared 下)
        # L1 loss 下严格等价应为 1/t, 这里用 1/t² 匹配理论 doc 的梯度放大 claim
        # 并对权重做 batch normalization (均值=1), 避免训练崩溃
        v_prediction: bool = False,
        v_prediction_t_eps: float = 1e-2,
        # ReFlow (Standard MSE): box target 模式
        # 'gt' (默认): box target = GT bboxes (现有行为)
        # 'x0_pred': box target = x_0^pred (A4 推理预测, RF 拉直目标)
        # 'trip': box target = Tikhonov/MAP 收缩估计 (TRIP, SNR 退化正则化)
        # 详见 REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1.2 / FEASIBLE_TRIP.md §7
        box_target_mode: str = 'gt',
        # TRIP: 类条件先验 (Tikhonov/MAP 收缩目标)
        # 可为 dict (已加载) 或 str (pickle 文件路径, 懒加载)
        # 格式: {'mu': Tensor[num_classes, 4] (cxcywh, [0,1]),
        #        'sigma_bar_sq': Tensor[num_classes] (各向同性平均方差, 归一化空间)}
        # 由 tools/estimate_class_priors.py 离线估计
        class_priors: Optional[Union[str, Dict]] = None,
        trip_lambda_mode: str = 'map',  # 'map' (λ=t², 主) / 'morozov' (备选)
        trip_tau: float = 1.0,           # Morozov 偏差原理的 τ 参数
        # TRIP 空间一致性: snr_scale 用于将 σ_p² 从归一化 [0,1] 空间转换到
        # raw 扩散空间 [-snr_scale, snr_scale] (与噪声 ε~N(0,I) 同空间).
        # 转换公式: σ_p²_raw = 4·snr_scale²·σ_p²_norm
        # (因 raw = (norm*2-1)*snr_scale → Var(raw) = 4·snr_scale²·Var(norm))
        # 若不设置 (默认 1.0), σ_p² 不做转换 — 仅当 priors 已在 raw 空间时正确.
        snr_scale: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.quality_loss_weight = quality_loss_weight
        self.quality_focal_alpha = quality_focal_alpha
        self.quality_focal_gamma = quality_focal_gamma
        self.deep_supervision = deep_supervision
        self.v_prediction = v_prediction
        self.v_prediction_t_eps = v_prediction_t_eps
        assert box_target_mode in ('gt', 'x0_pred', 'trip'), (
            f"box_target_mode 必须是 'gt' / 'x0_pred' / 'trip', got {box_target_mode}"
        )
        self.box_target_mode = box_target_mode

        # TRIP: 类条件先验 (懒加载, 首次 _compute_trip_target 时载入)
        # 详见 FEASIBLE_TRIP.md §7.2
        self._class_priors_arg = class_priors  # 原始参数 (str 路径或 dict)
        self._class_priors: Optional[Dict] = None  # 加载后的先验 (tensor 格式)
        self.trip_lambda_mode = trip_lambda_mode
        self.trip_tau = trip_tau
        self.snr_scale = snr_scale
        assert trip_lambda_mode in ('map', 'morozov'), (
            f"trip_lambda_mode 必须是 'map' 或 'morozov', got {trip_lambda_mode}"
        )
        if box_target_mode == 'trip':
            assert class_priors is not None, (
                "box_target_mode='trip' 时必须提供 class_priors "
                "(dict 或 pickle 文件路径)"
            )

    # ================================================================
    # TRIP: Tikhonov/MAP 正则化回归目标
    # ================================================================

    def _load_class_priors(self) -> Dict:
        """懒加载类条件先验 (首次 _compute_trip_target 调用时载入).

        支持两种格式:
          - dict: {'mu': Tensor[num_classes, 4], 'sigma_bar_sq': Tensor[num_classes]}
          - str: pickle 文件路径, 加载后期望为上述 dict 格式

        Returns:
            priors: dict with 'mu' [num_classes, 4] 和 'sigma_bar_sq' [num_classes]
        """
        if self._class_priors is not None:
            return self._class_priors

        arg = self._class_priors_arg
        if isinstance(arg, dict):
            priors = arg
        elif isinstance(arg, str):
            if not os.path.isfile(arg):
                raise FileNotFoundError(
                    f"TRIP class_priors 文件不存在: {arg}. "
                    f"请先运行 tools/estimate_class_priors.py 生成."
                )
            priors = torch.load(arg, map_location='cpu')
            logger.info(f"TRIP: 类条件先验已从 {arg} 加载")
        else:
            raise TypeError(
                f"class_priors 必须为 dict 或 str (文件路径), got {type(arg)}"
            )

        # 校验格式
        assert 'mu' in priors, "class_priors 缺少 'mu' 键"
        assert 'sigma_bar_sq' in priors, "class_priors 缺少 'sigma_bar_sq' 键"
        mu = priors['mu']
        sigma_bar_sq = priors['sigma_bar_sq']
        assert mu.shape[0] == self.num_classes, (
            f"mu 的 num_classes={mu.shape[0]} 与 criterion "
            f"num_classes={self.num_classes} 不匹配"
        )
        assert sigma_bar_sq.shape[0] == self.num_classes, (
            f"sigma_bar_sq 的 num_classes={sigma_bar_sq.shape[0]} 与 criterion "
            f"num_classes={self.num_classes} 不匹配"
        )
        # 缓存 (转为 tensor, 后续 .to(device) 按需)
        self._class_priors = priors
        return priors

    def _compute_trip_target(
        self,
        gt_xyxy: Tensor,
        matched_gt_inds: Tensor,
        fg_masks: Tensor,
        gt_labels_padded: Tensor,
        t: Tensor,
    ) -> Tensor:
        r"""TRIP: Tikhonov/MAP 正则化目标值.

        将 GT 回归目标 x_0 替换为贝叶斯 MAP 收缩估计:
          x_tilde^c(t) = (1-s(t)) * x_0 + s(t) * mu_p^c
        其中 s(t) = (t²/σ_p²) / ((1-t)² + t²/σ_p²) ∈ [0,1] (MAP, λ=t²)

        边界行为: s(0)≈0 (目标=x_0=GT, 零正则); s(1)≈1 (目标=mu_p, 完全收缩)
        详见 FEASIBLE_TRIP.md §2.5 定理 2.8, §7.2 代码方案

        Args:
            gt_xyxy: GT bboxes padded [bs, max_gt, 4] (归一化 xyxy [0,1])
            matched_gt_inds: 每个 proposal 匹配的 GT 索引 [bs, N]
            fg_masks: 正样本掩码 [bs, N] (False=背景, 不参与 TRIP)
            gt_labels_padded: GT 类标签 padded [bs, max_gt] (含背景填充)
            t: 扩散时间 [bs] (RF 路径下范围 [0,1])
        Returns:
            trip_tgt_xyxy: TRIP 目标 [bs, N, 4] (归一化 xyxy [0,1], 与 src_boxes 同空间)
        """
        bs, N = matched_gt_inds.shape
        device = gt_xyxy.device

        # 1. Gather 每个 proposal 匹配的 GT (xyxy [0,1])
        matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)
        tgt_gt_xyxy = torch.gather(
            gt_xyxy, 1,
            matched_gt_inds_clamped.unsqueeze(-1).expand(-1, -1, 4),
        )  # [bs, N, 4]

        # 2. 转换到 cxcywh [0,1] (TRIP 收缩在 cxcywh 空间, 与 L1 loss 一致)
        tgt_gt_cxcywh = bbox_xyxy_to_cxcywh(tgt_gt_xyxy)  # [bs, N, 4]

        # 3. Gather 每个 proposal 匹配的 GT 类标签
        matched_gt_labels = torch.gather(
            gt_labels_padded, 1, matched_gt_inds_clamped
        )  # [bs, N]
        # 背景位置 (fg=False) 的 label 可能是 num_classes (填充值),
        # clamp 到有效类范围以避免索引越界 (后续用 fg_masks 屏蔽)
        matched_gt_labels = matched_gt_labels.clamp(max=self.num_classes - 1)

        # 4. 加载类条件先验
        # priors 中 mu 在归一化 [0,1] cxcywh 空间 (用于下方凸组合, 与 tgt_gt_cxcywh 同空间);
        # sigma_bar_sq 也在归一化空间, 但 MAP 公式 s(t) = (t²/σ_p²)/((1-t)²+t²/σ_p²) 要求
        # σ_p² 与噪声 ε~N(0,I) 同空间 (raw 扩散空间 [-snr_scale, snr_scale]).
        # 转换: σ_p²_raw = 4·snr_scale²·σ_p²_norm
        # (因 raw = (norm*2-1)*snr_scale → Var(raw) = 4·snr_scale²·Var(norm))
        # 不转换会使 s(t) 激进 4·snr_scale² 倍 (snr_scale=2 时 16×), 导致训练崩溃.
        priors = self._load_class_priors()
        mu_p = priors['mu'].to(device)             # [num_classes, 4] (归一化空间, 用于凸组合)
        sigma_bar_sq_norm = priors['sigma_bar_sq'].to(device)  # [num_classes] (归一化空间)
        # 空间转换: 归一化 → raw (与噪声同空间, 供 s(t) 使用)
        sigma_bar_sq = sigma_bar_sq_norm * (4.0 * self.snr_scale ** 2)  # [num_classes]

        # 5. 向量化: 收集每个 proposal 对应类的先验 (R2 修正: 避免 for b,c 循环)
        mu_p_per_prop = mu_p[matched_gt_labels]             # [bs, N, 4]
        sigma_p_sq_per_prop = sigma_bar_sq[matched_gt_labels]  # [bs, N]

        # 6. 计算收缩因子 s(t)
        tb = t.view(bs, 1)  # [bs, 1] → broadcast to [bs, N]
        if self.trip_lambda_mode == 'map':
            # 贝叶斯 MAP: λ = t², γ = t² / σ_p²
            gamma = (tb ** 2) / sigma_p_sq_per_prop.clamp(min=1e-8)  # [bs, N]
        elif self.trip_lambda_mode == 'morozov':
            # Morozov 自适应 (备选, 类比应用, 详见 FEASIBLE_TRIP.md §2.4)
            gamma = (
                tb * (tb + (tb ** 2 + sigma_p_sq_per_prop * (1 - tb) ** 2).sqrt())
                / sigma_p_sq_per_prop.clamp(min=1e-8)
            ) / self.trip_tau  # [bs, N]
        else:
            raise ValueError(f"Unknown trip_lambda_mode: {self.trip_lambda_mode}")

        # s(t) = γ / ((1-t)² + γ) ∈ [0, 1]
        s = gamma / ((1 - tb) ** 2 + gamma)  # [bs, N]
        s = s.unsqueeze(-1)  # [bs, N, 1]

        # 7. TRIP 目标: (1-s) * x_0 + s * μ_p^c (cxcywh 空间)
        trip_tgt_cxcywh = (1 - s) * tgt_gt_cxcywh + s * mu_p_per_prop  # [bs, N, 4]

        # 8. 转回 xyxy [0,1] (与 src_boxes 同空间, 供 L1 + GIoU loss 使用)
        trip_tgt_xyxy = bbox_cxcywh_to_xyxy(trip_tgt_cxcywh)  # [bs, N, 4]

        # 9. 背景位置: 用原 GT (不参与 TRIP, 后续 fg_masks 会屏蔽)
        # (避免背景位置 TRIP 目标被 GIoU loss 误用; 实际 GIoU/L1 均用 fg_masks 屏蔽)
        trip_tgt_xyxy = torch.where(
            fg_masks.unsqueeze(-1), trip_tgt_xyxy, tgt_gt_xyxy
        )

        # 诊断: 收缩因子分布
        with torch.no_grad():
            probe.record_scalar('train/trip_s_mean', s.mean().item())
            probe.record_scalar('train/trip_s_max', s.max().item())
            probe.record_scalar('train/trip_s_min', s.min().item())
            probe.record_scalar(
                'train/trip_s_fg_mean',
                s.squeeze(-1)[fg_masks].mean().item()
                if fg_masks.any() else 0.0,
            )
        return trip_tgt_xyxy

    def forward(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        t: Optional[Tensor] = None,
        # ReFlow (Standard MSE): box target = A4 预测 (x_0^pred), 替代 GT
        # 仅当 box_target_mode='x0_pred' 时生效; 'gt' 模式忽略此参数
        # 形状: [bs, max_gt, 4] (xyxy, 已 padded) 或 list[[n_i, 4]]
        box_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # 主输出: 使用 matcher.forward 并建立 GT 缓存
        indices, gt_cache = self.matcher.forward_with_gt_cache(
            outputs, targets
        )
        losses = self._get_loss(
            outputs, targets, indices, t, box_targets=box_targets
        )

        if self.deep_supervision and outputs.aux_outputs is not None:
            for i, aux_out in enumerate(outputs.aux_outputs):
                # aux_outputs 复用 GT 缓存，避免重复计算 gt_ctrs/gt_wh/center 区域
                aux_indices, gt_cache = self.matcher.forward_with_gt_cache(
                    aux_out, targets, gt_cache
                )
                aux_losses = self._get_loss(
                    aux_out, targets, aux_indices, t, box_targets=box_targets
                )
                for name, val in aux_losses.items():
                    losses[f'aux_{i}_{name}'] = val
            # 探针: deep_supervision aux loss 数量
            probe.record_scalar('criterion/n_aux_outputs', len(outputs.aux_outputs))
        self._last_indices = indices
        return losses

    def _get_loss(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        indices: List[Tuple[Tensor, Tensor]] = None,
        t: Optional[Tensor] = None,
        box_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if indices is None:
            indices = self.matcher(outputs, targets)
        loss_cls = self._loss_classification(outputs, targets, indices)
        loss_bbox, loss_giou = self._loss_boxes(
            outputs, targets, indices, t, box_targets=box_targets
        )

        # 探针: 损失分量标量 (训练时每 100 步)
        probe.record_scalar('criterion/loss_cls', loss_cls.item())
        probe.record_scalar('criterion/loss_bbox', loss_bbox.item())
        probe.record_scalar('criterion/loss_giou', loss_giou.item())

        losses = {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
        }
        if outputs.pred_quality is not None:
            losses['loss_quality'] = self._loss_quality(
                outputs, targets, indices)
            probe.record_scalar(
                'criterion/loss_quality', losses['loss_quality'].item())
        return losses

    def _loss_quality(self, outputs, targets, indices) -> Tensor:
        """Varifocal-style continuous IoU quality supervision.

        Hungarian positives receive their detached aligned IoU as target;
        unmatched proposals receive zero. Geometry gradients therefore remain
        exclusively in the existing box loss, while the new branch learns the
        ranking statistic required by COCO AP at strict IoU thresholds.
        """
        logits = outputs.pred_quality.squeeze(-1)
        boxes = outputs.pred_boxes.detach()
        bs, num_queries = logits.shape
        targets_iou = logits.new_zeros(bs, num_queries)
        fg_masks = torch.stack([idx[0] for idx in indices])
        matched_gt_inds = torch.stack([idx[1] for idx in indices]).clamp(min=0)

        max_gt = max(target.bboxes.shape[0] for target in targets)
        if max_gt > 0:
            gt_padded = boxes.new_zeros(bs, max_gt, 4)
            for batch_index, target in enumerate(targets):
                count = target.bboxes.shape[0]
                if count:
                    gt_padded[batch_index, :count] = target.bboxes
            matched = torch.gather(
                gt_padded, 1,
                matched_gt_inds.unsqueeze(-1).expand(-1, -1, 4))
            lt = torch.maximum(boxes[..., :2], matched[..., :2])
            rb = torch.minimum(boxes[..., 2:], matched[..., 2:])
            wh = (rb - lt).clamp(min=0)
            intersection = wh[..., 0] * wh[..., 1]
            box_wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
            gt_wh = (matched[..., 2:] - matched[..., :2]).clamp(min=0)
            union = (box_wh[..., 0] * box_wh[..., 1]
                     + gt_wh[..., 0] * gt_wh[..., 1] - intersection)
            aligned_iou = intersection / union.clamp(min=1e-7)
            targets_iou[fg_masks] = aligned_iou[fg_masks].clamp(0, 1)

        probability = logits.sigmoid()
        positive_weight = targets_iou
        negative_weight = (
            self.quality_focal_alpha
            * probability.pow(self.quality_focal_gamma))
        focal_weight = torch.where(
            fg_masks, positive_weight, negative_weight).detach()
        loss = F.binary_cross_entropy_with_logits(
            logits, targets_iou, reduction='none') * focal_weight
        num_pos = fg_masks.sum().clamp(min=1)
        return self.quality_loss_weight * loss.sum() / num_pos

    def _loss_classification(self, outputs, targets, indices) -> Tensor:
        src_logits = outputs.pred_logits  # [bs, num_queries, num_classes+1]
        bs, num_queries = src_logits.shape[:2]

        # 构建 padded GT labels: [bs, max_gt]
        max_gt = max(t.labels.shape[0] for t in targets)
        if max_gt == 0:
            # 全部为背景
            target_classes = src_logits.new_full(
                (bs, num_queries), self.num_classes, dtype=torch.long
            )
            num_pos = src_logits.new_tensor(1, dtype=torch.long)
            loss_cls = self.loss_cls(
                src_logits.flatten(0, 1), target_classes.flatten(0, 1)
            )
            return loss_cls / num_pos

        gt_labels_padded = src_logits.new_full(
            (bs, max_gt), self.num_classes, dtype=torch.long
        )
        for i, t in enumerate(targets):
            n = t.labels.shape[0]
            if n > 0:
                gt_labels_padded[i, :n] = t.labels

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack(
            [idx[1] for idx in indices]
        )  # [bs, N] long
        matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)

        # Gather GT labels: [bs, N]
        matched_gt_labels = torch.gather(
            gt_labels_padded, 1, matched_gt_inds_clamped
        )
        # 背景位置设为 num_classes
        matched_gt_labels[~fg_masks] = self.num_classes

        target_classes = matched_gt_labels
        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        loss_cls = self.loss_cls(
            src_logits.flatten(0, 1), target_classes.flatten(0, 1)
        )
        return loss_cls / num_pos

    def _loss_boxes(
        self,
        outputs,
        targets,
        indices,
        t: Optional[Tensor] = None,
        box_targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """计算 box L1 + GIoU 损失.

        空间一致性约定 (ReFlow 修复 BUG #1):
          - src_boxes (outputs.pred_boxes): 归一化 xyxy [0,1]
          - targets[*].bboxes (GT): 归一化 xyxy [0,1] (由 head._normalize_targets 转换)
          - box_targets (ReFlow x0_pred): **必须** 归一化 xyxy [0,1]
            (head.loss() 中已将 raw cxcywh scaled 转换为归一化 xyxy,
             切勿直接传 raw 扩散空间的 coupling x0_pred)

        三种 box_target_mode:
          - 'gt' (默认): box target = GT bboxes (现有行为)
          - 'x0_pred': box target = A4 预测 (x_0^pred), 仅当传入 box_targets 时生效;
                       未传则安全降级到 GT (避免训练崩溃)
          - 'trip': box target = Tikhonov/MAP 收缩估计 (TRIP, SNR 退化正则化)
                    x_tilde(t) = (1-s(t)) * x_0 + s(t) * mu_p^c
                    s(t) 由贝叶斯 MAP (λ=t²) 导出, 详见 FEASIBLE_TRIP.md §7
        cls target 始终用 GT (matcher 基于 GT 分配正负样本, 不受 box_target_mode 影响)
        """
        src_boxes = outputs.pred_boxes  # [bs, num_queries, 4] 归一化 xyxy [0,1]
        bs = src_boxes.shape[0]

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        max_gt = max(t_data.bboxes.shape[0] for t_data in targets)
        if max_gt == 0:
            # 无正样本，返回零损失
            return src_boxes.sum() * 0, src_boxes.sum() * 0

        # 批量化: 将 indices 堆叠为 [bs, N] 张量
        fg_masks = torch.stack([idx[0] for idx in indices])  # [bs, N] bool
        matched_gt_inds = torch.stack(
            [idx[1] for idx in indices]
        )  # [bs, N] long

        # 构建 padded GT bboxes: [bs, max_gt, 4]
        gt_bboxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
        for i, t_data in enumerate(targets):
            n = t_data.bboxes.shape[0]
            if n > 0:
                gt_bboxes_padded[i, :n] = t_data.bboxes

        # box target 来源选择 (三种模式)
        # 'gt' (默认): box target = GT bboxes (现有行为)
        # 'x0_pred': box target = A4 预测 (x_0^pred), 仅当传入 box_targets 时生效;
        #            未传则安全降级到 GT (避免训练崩溃)
        # 'trip': box target = Tikhonov/MAP 收缩估计 (SNR 退化正则化)
        # cls target 始终用 GT (matcher 基于 GT 分配正负样本, 不受此处影响)
        use_x0_pred = (
            self.box_target_mode == 'x0_pred' and box_targets is not None
        )
        use_trip = (
            self.box_target_mode == 'trip'
            and t is not None
        )
        if use_x0_pred:
            # REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1.2: box target = x_0^pred (RF 拉直目标)
            # 两种布局 (按形状自动分流):
            #   - per-proposal [bs, num_proposals, 4]: 每个 proposal 的 A4 预测 (轨迹端点),
            #     与 pred_boxes 直接对齐, 不经 matcher gather (canonical ReFlow)
            #   - per-GT [bs, max_gt, 4]: 每个 GT 的 A4 预测, 按 matched_gt_inds gather
            is_per_proposal = (
                not isinstance(box_targets, (list, tuple))
                and box_targets.shape[1] == src_boxes.shape[1]
            )
            if is_per_proposal:
                # per-proposal: 直接对齐 (RF 每个提案有独立轨迹端点 x_0^pred)
                tgt_boxes = box_targets.to(src_boxes)  # [bs, num_proposals, 4]
            else:
                # per-GT: 对齐到 [bs, max_gt, 4] 并 gather (与 GT 布局一致)
                if isinstance(box_targets, (list, tuple)):
                    target_boxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
                    for i, bt in enumerate(box_targets):
                        n = bt.shape[0]
                        if n > 0:
                            target_boxes_padded[i, :n] = bt
                else:
                    n_box = box_targets.shape[1]
                    if n_box >= max_gt:
                        target_boxes_padded = box_targets[:, :max_gt].to(src_boxes)
                    else:
                        target_boxes_padded = src_boxes.new_zeros(bs, max_gt, 4)
                        target_boxes_padded[:, :n_box] = box_targets.to(src_boxes)
                matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)
                tgt_boxes = torch.gather(
                    target_boxes_padded,
                    1,
                    matched_gt_inds_clamped.unsqueeze(-1).expand(-1, -1, 4),
                )
        elif use_trip:
            # TRIP: box target = Tikhonov/MAP 收缩估计 (FEASIBLE_TRIP.md §7)
            # 构建 padded GT labels (与 _loss_classification 一致)
            gt_labels_padded = src_boxes.new_full(
                (bs, max_gt), self.num_classes, dtype=torch.long
            )
            for i, t_data in enumerate(targets):
                n = t_data.labels.shape[0]
                if n > 0:
                    gt_labels_padded[i, :n] = t_data.labels
            # 计算 TRIP 收缩目标: (1-s(t))*x_0 + s(t)*mu_p^c
            tgt_boxes = self._compute_trip_target(
                gt_xyxy=gt_bboxes_padded,
                matched_gt_inds=matched_gt_inds,
                fg_masks=fg_masks,
                gt_labels_padded=gt_labels_padded,
                t=t,
            )  # [bs, N, 4] 归一化 xyxy [0,1]
        else:
            # 'gt' 模式 或 'x0_pred' 未传 box_targets → 用 GT
            matched_gt_inds_clamped = matched_gt_inds.clamp(min=0)
            tgt_boxes = torch.gather(
                gt_bboxes_padded,
                1,
                matched_gt_inds_clamped.unsqueeze(-1).expand(-1, -1, 4),
            )

        # num_pos 保留为张量，避免 .item() 同步
        num_pos = fg_masks.sum().clamp(min=1)

        tgt_cxcywh = bbox_xyxy_to_cxcywh(tgt_boxes)  # [bs, N, 4]
        src_cxcywh = bbox_xyxy_to_cxcywh(src_boxes)  # [bs, N, 4]

        # L1 loss: 逐元素计算后 mask
        per_elem_l1 = F.l1_loss(
            src_cxcywh, tgt_cxcywh, reduction='none'
        )  # [bs, N, 4]
        if self.v_prediction and t is not None:
            # R3: v-prediction 等价的 1/t² 损失加权
            # t 是 [bs,] 张量 (shifted schedule 后, 范围 [0,1])
            # 理论 (§4.3): v-prediction 在 t→0 时梯度放大 1/t²
            # batch normalization (均值=1) 控制绝对幅度, 避免训练崩溃
            t_clamped = t.clamp(min=self.v_prediction_t_eps)
            v_weight = 1.0 / (t_clamped ** 2)  # [bs,]
            v_weight = v_weight / v_weight.mean().detach()
            v_weight = v_weight.view(-1, 1, 1)  # [bs, 1, 1]
            masked_l1 = (
                per_elem_l1
                * fg_masks.unsqueeze(-1).float()
                * v_weight
            )
        else:
            masked_l1 = per_elem_l1 * fg_masks.unsqueeze(-1).float()

        loss_bbox = self.loss_bbox.loss_weight * masked_l1.sum() / num_pos
        # GIoU loss: 逐元素计算后 mask
        # GIoU 不是 t 的简单函数, v_prediction 下保持不加权
        per_giou = ops.generalized_box_iou_loss(
            src_boxes.reshape(-1, 4),
            tgt_boxes.reshape(-1, 4),
            reduction='none',
        ).reshape(bs, -1)
        per_giou = torch.nan_to_num(per_giou, nan=0.0)

        loss_giou = (
            self.loss_giou.loss_weight
            * (per_giou * fg_masks.float()).sum()
            / num_pos
        )

        # 探针: box 损失详细统计 (正样本数, L1/GIoU per-elem 分布)
        probe.record_scalar('criterion/num_pos', num_pos.item())
        probe.record_scalar('criterion/fg_ratio', fg_masks.float().mean().item())
        # per-elem L1 分布 (仅正样本)
        if per_elem_l1 is not None:
            pos_l1 = per_elem_l1[fg_masks]
            if pos_l1.numel() > 0:
                probe.record_tensor_stats('criterion/l1_per_elem', pos_l1)
        # per-giou 分布 (仅正样本)
        if per_giou is not None:
            pos_giou = per_giou[fg_masks]
            if pos_giou.numel() > 0:
                probe.record_tensor_stats('criterion/giou_per_elem', pos_giou)
        # 预测 box vs GT box 的 cxcywh 差异 (仅正样本, per-dim)
        if src_cxcywh is not None and tgt_cxcywh is not None:
            box_diff = (src_cxcywh - tgt_cxcywh).abs()[fg_masks]
            if box_diff.numel() > 0:
                probe.record_tensor_stats('criterion/box_diff', box_diff)

        return loss_bbox, loss_giou
