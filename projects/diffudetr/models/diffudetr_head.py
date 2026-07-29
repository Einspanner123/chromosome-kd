"""DiffuDETR 扩散头 — 训练 (q_sample + 预测 x0) + 推理 (DDIM + box_renewal)

从原仓库 (MBadran2000/DiffuDETR) 的 dino_diffu_det_noise.py 移植核心扩散逻辑,
适配 mmdet 框架 (纯 PyTorch head, 无 detrex/detectron2 依赖).

训练流程:
  1. GT box (扩散空间 [-scale,scale]) → pad 到 num_queries (随机框 + 背景标签)
  2. 固定随机 shuffle: GT 放到随机 query 位置
  3. 采样时间步 t → q_sample(x_start, t) → 噪声框 x_t
  4. transformer(x_t, t) → 中间层特征 → class_embed/bbox_embed → 预测 x0 + logits
  5. 预测框/目标框转换到归一化 cxcywh [0,1] → SNR 加权损失

推理流程:
  1. 纯噪声 randn [B, N, 4] (扩散空间, 匹配 N(0,1) 先验)
  2. DDIM 25 步 (eta=0, 确定性):
     a. transformer(noise, t) → pred_x0 + pred_logits
     b. box_renewal: threshold = time_next/100, 低分框替换为 randn
     c. DDIM step → 下一轮噪声
     d. ensemble: 累积 (scores, labels, boxes)
  3. NMS 去重

关键设计决策:
  - num_classes=24, num_queries=300, 背景标签 = num_classes (24)
  - scale=2: 扩散空间 [-2,2], box [0,1] ↔ [-2,2]
  - parameterization="x0": 直接预测干净框 x0
  - box_renewal 自适应阈值: threshold = time_next/100
    (高 timestep 阈值大 → 几乎全部 renew; 低 timestep 阈值小 → 保留可靠预测)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.ops import batched_nms

from mmdet.registry import MODELS
from .criterion import (
    HungarianMatcher,
    SetCriterion,
    box_cxcywh_to_xyxy,
)
from .diffusion_scheduler import DiffusionScheduler
from .timestep_block import TimeStepBlock
from .transformer import MLP, DiffuDETRTransformer


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """inverse_sigmoid: x → log(x / (1-x)).

    对齐 mmdet/detectron2 inverse_sigmoid. 用于 DINO iterative refinement:
        pred_boxes = (bbox_embed(feat) + inverse_sigmoid(reference)).sigmoid()

    Args:
        x: [B, N, 4] 归一化 [0,1] 参考点.
        eps: 数值稳定 clamp, 避免 log(0).

    Returns:
        [B, N, 4] inverse sigmoid 空间值.
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


@MODELS.register_module(name='DiffuDETRHead', force=True)
class DiffuDETRHead(nn.Module):
    """DiffuDETR 扩散检测头.

    Args:
        num_classes: 类别数 (24, Dataset2 染色体).
        num_queries: 查询数 (300).
        embed_dim: 嵌入维度 (256).
        num_heads: 注意力头数 (8).
        dim_feedforward: FFN 中间维度 (2048).
        num_layers: decoder 层数 (6).
        num_feature_levels: 特征层数 (4).
        timesteps: DDPM 总步数 (1000).
        sampling_timesteps: DDIM 采样步数 (25).
        scale: 扩散空间缩放 (2.0).
        box_renewal: 是否启用 box renewal (True).
        use_ensemble: 是否启用集成 (True).
        use_nms: 是否启用 NMS (True).
        nms_thr: NMS IoU 阈值 (0.7).
        aux_loss: 是否计算辅助损失 (True).
    """

    def __init__(
        self,
        num_classes: int = 24,
        num_queries: int = 300,
        embed_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 6,
        num_feature_levels: int = 4,
        timesteps: int = 1000,
        sampling_timesteps: int = 25,
        scale: float = 2.0,
        box_renewal: bool = True,
        use_ensemble: bool = True,
        use_nms: bool = True,
        nms_thr: float = 0.7,
        aux_loss: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.scale = scale
        self.aux_loss = aux_loss
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self.use_nms = use_nms
        self.nms_thr = nms_thr

        # 扩散调度器
        self.scheduler = DiffusionScheduler(
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            ddim_eta=0.0,
            scale=scale,
        )

        # Transformer (decoder)
        self.transformer = DiffuDETRTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            dropout=dropout,
        )

        # 每层预测头 (class_embed + bbox_embed), 对齐原仓库 ModuleList
        num_pred = num_layers
        self.class_embed = nn.ModuleList(
            [nn.Linear(embed_dim, num_classes) for _ in range(num_pred)]
        )
        self.bbox_embed = nn.ModuleList(
            [MLP(embed_dim, embed_dim, 4, 3) for _ in range(num_pred)]
        )

        # 分类头: 分类预测前注入时间步 (对齐原仓库 ClassEmbed)
        # 用 TimeStepBlock 在 class_embed 前注入时间步
        time_embed_dim = embed_dim * 4
        self.class_time_embed = nn.ModuleList(
            [
                TimeStepBlock(
                    channels=embed_dim,
                    emb_channels=time_embed_dim,
                    out_channels=embed_dim,
                )
                for _ in range(num_pred)
            ]
        )

        # 损失: Hungarian 匹配 + SNR 加权
        matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            num_classes=num_classes,
        )
        weight_dict = {
            'loss_ce': 2.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
        }
        self.criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=['labels', 'boxes'],
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化分类/回归头 (对齐原仓库 prior_prob 初始化)."""
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls in self.class_embed:
            cls.bias.data = torch.ones(self.num_classes) * bias_value
        for bbox in self.bbox_embed:
            nn.init.constant_(bbox.layers[-1].weight.data, 0)
            nn.init.constant_(bbox.layers[-1].bias.data, 0)

    # ========== 空间转换 ==========

    def diffusion_to_norm(self, x: torch.Tensor) -> torch.Tensor:
        """扩散空间 [-scale, scale] → 归一化 cxcywh [0,1].

        对齐 setdiff_detector.diffusion_to_norm_space:
            (clamp(x, -s, s) / s + 1) / 2
        """
        s = self.scale
        return (x.clamp(-s, s) / s + 1.0) / 2.0

    def norm_to_diffusion(self, x: torch.Tensor) -> torch.Tensor:
        """归一化 cxcywh [0,1] → 扩散空间 [-scale, scale].

        对齐 setdiff_detector.gt_to_diffusion_space:
            (x * 2 - 1) * scale
        """
        return (x * 2.0 - 1.0) * self.scale

    # ========== 训练 ==========

    def _prepare_x_start(
        self,
        gt_boxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备 x_start: GT 框 + 随机填充 → 固定随机 shuffle.

        对齐 DiffusionDet prepare_targets + DiffuDETR 的 shuffle 逻辑:
          1. GT 框 (扩散空间) 放到 query 位置
          2. 不足 num_queries 的位置用随机框填充 (N(0,1), 扩散空间)
          3. 随机 shuffle GT 位置 (固定随机匹配)
          4. 填充框标签 = num_classes (背景)

        Args:
            gt_boxes_list: 每张图 GT 框 [M, 4] (扩散空间 [-scale,scale]).
            gt_labels_list: 每张图 GT 标签 [M].
            device: 设备.

        Returns:
            x_start: [B, N, 4] 干净框 (扩散空间).
            labels: [B, N] 标签 (背景 = num_classes).
        """
        B = len(gt_boxes_list)
        N = self.num_queries
        x_start = torch.randn(B, N, 4, device=device)  # 填充: N(0,1)
        labels = torch.full(
            (B, N), self.num_classes, dtype=torch.long, device=device
        )  # 填充: 背景类

        for i in range(B):
            gt_boxes = gt_boxes_list[i]  # [M, 4]
            gt_labels = gt_labels_list[i]  # [M]
            M = gt_boxes.shape[0]
            if M == 0:
                continue
            if M > N:
                # GT 多于 query: 随机选 N 个
                perm = torch.randperm(M, device=device)[:N]
                gt_boxes = gt_boxes[perm]
                gt_labels = gt_labels[perm]
                M = N
            # 固定随机 shuffle: GT 放到随机位置
            positions = torch.randperm(N, device=device)[:M]
            x_start[i, positions] = gt_boxes
            labels[i, positions] = gt_labels

        return x_start, labels

    def _run_heads(
        self,
        inter_states: torch.Tensor,
        time_emb: torch.Tensor,
        init_reference: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """对每层 decoder 输出应用 class_embed + bbox_embed (DINO iterative refinement).

        对齐原仓库 dino_diffu_det_noise.py:541-549:
            reference = inverse_sigmoid(reference)
            tmp = bbox_embed(feat) + reference
            outputs_coord = tmp.sigmoid()

        sigmoid 保证预测在 (0,1), 梯度有界 (最大 0.25), 避免无界输出导致的
        clamp 死梯度问题 (原实现 pred_boxes = bbox_embed(feat) 无约束).

        Args:
            inter_states: [num_layers, B, N, C] 中间层特征.
            time_emb: [B, 4*embed_dim] 时间步嵌入.
            init_reference: [B, N, 4] 初始参考点 (归一化 [0,1] cxcywh).

        Returns:
            pred_logits_list: [num_layers] 每层 [B, N, num_classes].
            pred_boxes_list: [num_layers] 每层 [B, N, 4] (归一化 [0,1] cxcywh,
                             sigmoid 保证有界).
        """
        pred_logits_list = []
        pred_boxes_list = []
        reference = init_reference  # [B, N, 4] in [0,1]
        for i in range(self.num_layers):
            feat = inter_states[i]  # [B, N, C]
            # 分类: 先注入时间步, 再 class_embed
            feat_cls = self.class_time_embed[i](feat, time_emb)
            pred_logits = self.class_embed[i](feat_cls)
            # 回归: DINO iterative refinement
            # sigmoid(offset + inverse_sigmoid(ref)) → 保证 [0,1], 梯度有界
            pred_offset = self.bbox_embed[i](feat)
            pred_boxes = (pred_offset + inverse_sigmoid(reference)).sigmoid()
            pred_logits_list.append(pred_logits)
            pred_boxes_list.append(pred_boxes)
            reference = pred_boxes  # 下一层以当前预测为参考
        return pred_logits_list, pred_boxes_list

    def forward_train(
        self,
        multi_level_feats: List[torch.Tensor],
        gt_boxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """训练前向: q_sample + 预测 x0 + SNR 加权损失.

        Args:
            multi_level_feats: List[[B,C,H,W]] 多尺度图像特征.
            gt_boxes_list: 每张图 GT 框 [M, 4] (扩散空间 [-scale,scale]).
            gt_labels_list: 每张图 GT 标签 [M].

        Returns:
            loss_dict: 各损失项.
        """
        device = multi_level_feats[0].device
        B = multi_level_feats[0].shape[0]

        # 1. 准备 x_start (GT + 填充 + shuffle)
        x_start, all_labels = self._prepare_x_start(
            gt_boxes_list, gt_labels_list, device
        )

        # 2. 采样时间步 t (每张图独立采样)
        t = torch.randint(0, self.scheduler.num_timesteps, (B,), device=device)

        # 3. 前向扩散: q_sample(x_start, t) → 噪声框 x_t
        x_t = self.scheduler.q_sample(x_start, t)

        # 4. 时间步嵌入
        from .diffusion_scheduler import timestep_embedding

        time_emb = timestep_embedding(t.float(), self.embed_dim)
        time_emb_full = self.transformer.time_embed(
            time_emb
        )  # [B, 4*embed_dim]

        # 5. Transformer → 中间层特征
        inter_states = self.transformer(
            multi_level_feats, x_t, t, scale=self.scale
        )

        # 6. init_reference: x_t (扩散空间 [-scale,scale]) → 归一化 [0,1]
        #    用于 DINO iterative refinement 的初始参考点
        init_reference = self.diffusion_to_norm(x_t)  # [B, N, 4] in [0,1]

        # 7. 每层预测头 (DINO iterative refinement, sigmoid 保证输出 [0,1])
        pred_logits_list, pred_boxes_list = self._run_heads(
            inter_states, time_emb_full, init_reference
        )
        # pred_boxes 已在 [0,1] (sigmoid 输出), 无需 diffusion_to_norm 转换
        pred_boxes_norm_list = pred_boxes_list

        # 8. 构造 targets — 仅包含真实 GT (不含 padding)
        # 对齐 DETR SetCriterion: N 个预测匹配 M 个 GT, 未匹配的 query 为背景
        # GT 框从扩散空间转换到归一化 cxcywh [0,1]
        targets = []
        for i in range(B):
            gt_boxes_diff = gt_boxes_list[i]  # [M, 4] 扩散空间
            gt_labels = gt_labels_list[i]  # [M]
            # 扩散空间 [-scale, scale] → 归一化 cxcywh [0,1]
            gt_boxes_norm = self.diffusion_to_norm(gt_boxes_diff)
            targets.append(
                {
                    'labels': gt_labels,
                    'boxes': gt_boxes_norm,
                }
            )

        # 9. SNR 损失权重
        loss_weight = self.scheduler.loss_weight[t]  # [B]

        # 10. 计算损失
        loss_dict = self.criterion(
            pred_logits_list,
            pred_boxes_norm_list,
            targets,
            time_steps=t,
            loss_weight=loss_weight,
        )

        # nan 防御: 防止个别 inf/nan 污染权重导致永久崩溃 (治标, 根因已通过归一化+clip 修复)
        for k, v in loss_dict.items():
            if not torch.isfinite(v).all():
                loss_dict[k] = torch.nan_to_num(
                    v, nan=0.0, posinf=1e4, neginf=-1e4
                )
        return loss_dict

    # ========== 推理 ==========

    def _single_step_predict(
        self,
        multi_level_feats: List[torch.Tensor],
        noise: torch.Tensor,
        t: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步推理: transformer + heads → pred_x0_norm + pred_logits.

        对齐训练路径: DINO iterative refinement, sigmoid 保证输出 [0,1].

        Args:
            multi_level_feats: 多尺度特征.
            noise: [B, N, 4] 当前噪声框 (扩散空间 [-scale,scale]).
            t: 当前时间步 (标量).
            device: 设备.

        Returns:
            pred_x0_norm: [B, N, 4] 预测干净框 (归一化 [0,1] cxcywh, sigmoid 输出).
            pred_logits: [B, N, num_classes] 分类 logits.
        """
        B = noise.shape[0]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)

        # 时间步嵌入
        from .diffusion_scheduler import timestep_embedding

        time_emb = timestep_embedding(t_batch.float(), self.embed_dim)
        time_emb_full = self.transformer.time_embed(time_emb)

        # Transformer
        inter_states = self.transformer(
            multi_level_feats, noise, t_batch, scale=self.scale
        )

        # init_reference: noise (扩散空间) → 归一化 [0,1]
        init_reference = self.diffusion_to_norm(noise)  # [B, N, 4] in [0,1]

        # 逐层 iterative refinement, 取最后一层 (对齐训练路径)
        pred_logits_list, pred_boxes_list = self._run_heads(
            inter_states, time_emb_full, init_reference
        )
        pred_x0_norm = pred_boxes_list[-1]  # [B, N, 4] in [0,1]
        pred_logits = pred_logits_list[-1]

        return pred_x0_norm, pred_logits

    def forward_inference(
        self,
        multi_level_feats: List[torch.Tensor],
        img_shapes: List[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        """推理前向: DDIM 25 步 + box_renewal + ensemble + NMS.

        Args:
            multi_level_feats: List[[B,C,H,W]] 多尺度特征.
            img_shapes: 每张图 (H, W).

        Returns:
            results: 每张图 dict('boxes' [K,4] xyxy像素, 'scores' [K], 'labels' [K]).
        """
        device = multi_level_feats[0].device
        B = multi_level_feats[0].shape[0]
        N = self.num_queries

        # 1. 初始噪声: 纯 randn (扩散空间, 匹配 N(0,1) 先验)
        noise = torch.randn(B, N, 4, device=device)

        # 2. DDIM 时间对
        time_pairs = self.scheduler.get_time_pairs()

        # 3. ensemble 累积
        ensemble_scores = []
        ensemble_labels = []
        ensemble_boxes = []  # xyxy 像素坐标

        # 4. DDIM 循环
        for time, time_next in time_pairs:
            # a. 预测 x0_norm [0,1] (sigmoid 输出) + logits
            pred_x0_norm, pred_logits = self._single_step_predict(
                multi_level_feats, noise, time, device
            )

            # b. 计算分数 (用于 box_renewal 和 ensemble)
            scores = torch.sigmoid(pred_logits)  # [B, N, num_classes]
            conf, labels = scores.max(dim=-1)  # [B, N], [B, N]

            # c. box_renewal: 自适应阈值, 低分框替换为 randn
            if self.box_renewal and time_next >= 0:
                threshold = time_next / 100.0  # 自适应阈值
                # 低分框 → 替换为 randn (扩散空间)
                keep = conf > threshold  # [B, N]
                for i in range(B):
                    renew_mask = ~keep[i]
                    num_renew = renew_mask.sum().item()
                    if num_renew > 0:
                        noise[i, renew_mask] = torch.randn(
                            num_renew, 4, device=device
                        )

            # d. ensemble: 累积预测 (pred_x0_norm 已在 [0,1], 直接转 xyxy 像素)
            if self.use_ensemble:
                for i in range(B):
                    img_h, img_w = img_shapes[i]
                    scale = pred_x0_norm.new_tensor(
                        [img_w, img_h, img_w, img_h]
                    )
                    boxes_xyxy = box_cxcywh_to_xyxy(pred_x0_norm[i]) * scale
                    ensemble_scores.append(conf[i])
                    ensemble_labels.append(labels[i])
                    ensemble_boxes.append(boxes_xyxy)

            # e. DDIM step → 下一轮噪声 (需要扩散空间 x0)
            if time_next < 0:
                break
            # pred_x0_norm [0,1] → 扩散空间 [-scale, scale] 用于 DDIM
            pred_x0_diffusion = self.norm_to_diffusion(pred_x0_norm)
            noise = self.scheduler.ddim_step(
                noise, time, time_next, pred_x0_diffusion
            )

        # 5. ensemble + NMS
        results = []
        for i in range(B):
            if self.use_ensemble and len(ensemble_scores) > 0:
                # 收集该图的所有步预测
                step_scores = torch.stack(
                    ensemble_scores[i::B]
                )  # [num_steps, N]
                step_labels = torch.stack(ensemble_labels[i::B])
                step_boxes = torch.stack(ensemble_boxes[i::B])

                all_scores = step_scores.flatten()
                all_labels = step_labels.flatten()
                all_boxes = step_boxes.reshape(-1, 4)
            else:
                # 仅用最后一步 (pred_x0_norm 已在 [0,1], 来自最后循环)
                all_scores = conf[i]
                all_labels = labels[i]
                img_h, img_w = img_shapes[i]
                scale = pred_x0_norm.new_tensor([img_w, img_h, img_w, img_h])
                all_boxes = box_cxcywh_to_xyxy(pred_x0_norm[i]) * scale

            # NMS
            if self.use_nms:
                nms_cfg = dict(iou_threshold=self.nms_thr)
                # batched_nms 返回 (dets[K,5], keep_indices[K])
                _, keep = batched_nms(
                    all_boxes, all_scores, all_labels, nms_cfg
                )
                # 限制数量
                max_det = self.num_queries
                keep = keep[:max_det]
                all_scores = all_scores[keep]
                all_labels = all_labels[keep]
                all_boxes = all_boxes[keep]

            results.append(
                {
                    'boxes': all_boxes,
                    'scores': all_scores,
                    'labels': all_labels,
                }
            )

        return results

    def forward(
        self,
        multi_level_feats: List[torch.Tensor],
        gt_boxes_list: Optional[List[torch.Tensor]] = None,
        gt_labels_list: Optional[List[torch.Tensor]] = None,
        img_shapes: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """统一前向入口.

        训练模式: forward_train (gt_boxes_list 必须提供).
        推理模式: forward_inference (img_shapes 必须提供).
        """
        if self.training:
            assert gt_boxes_list is not None and gt_labels_list is not None
            return self.forward_train(
                multi_level_feats, gt_boxes_list, gt_labels_list
            )
        else:
            assert img_shapes is not None
            return self.forward_inference(multi_level_feats, img_shapes)
