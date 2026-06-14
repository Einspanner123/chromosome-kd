"""DiffusionDetHead — 核心迭代去噪检测头

协调 SingleHead 序列、ROI 提取、损失计算和扩散采样。
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ldmdet.coupling import build_coupling
from ldmdet.coupling.base import CouplingStrategy
from ldmdet.data.structures import DetectionResult, ImageMeta, InstanceData, ModelOutput
from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from ldmdet.diffusion.noise_schedule import cosine_noise_schedule, load_buffer
from ldmdet.diffusion.rectified_flow import RectifiedFlow
from ldmdet.diffusion.sampling import DiffusionSampler, _get_img_shape
from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh


class DiffusionDetHead(nn.Module):
    """扩散检测头。

    支持 DDPM 和 Rectified Flow 两种范式，
    Euler / Heun / DPM-Solver++ 等多种采样策略。
    """

    def __init__(
        self,
        num_classes: int = 80,
        feat_channels: int = 256,
        num_proposals: int = 500,
        num_heads: int = 6,
        prior_prob: float = 0.01,
        snr_scale: float = 2.0,
        timesteps: int = 1000,
        sampling_timesteps: int = 1,
        solver_type: str = 'euler',
        box_renewal: bool = True,
        use_ensemble: bool = True,
        deep_supervision: bool = True,
        ddim_sampling_eta: float = 1.0,
        diffusion_type: str = 'ddpm',
        rf_schedule: str = 'linear',
        rf_power: float = 1.0,
        rf_shift: float = 1.0,
        single_head: nn.Module = None,
        roi_extractor: nn.Module = None,
        criterion: nn.Module = None,
        use_nms: bool = True,
        nms_thr: float = 0.5,
        score_thr: float = 0.05,
        min_keep: int = 10,
        filter_unknown: bool = True,
        gt_reweight: bool = True,
        use_checkpoint: bool = False,
        counting_branch: Optional[nn.Module] = None,
        consistency_loss: Optional[nn.Module] = None,
        coupling: Optional[CouplingStrategy] = None,
        pre_noise_layer: int = 2,
        loss_aux: Optional[Dict] = None,
        torch_compile: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.diffusion_type = diffusion_type
        self.deep_supervision = deep_supervision
        self.filter_unknown = filter_unknown
        self.gt_reweight = gt_reweight
        self.timesteps = timesteps
        self.use_checkpoint = use_checkpoint
        self.rf_schedule = rf_schedule
        self.rf_shift = rf_shift
        self.rf_power = rf_power

        self.loss_aux = loss_aux
        self.counting_branch = counting_branch
        self.consistency_loss = consistency_loss

        # 扩散组件
        self.rf = RectifiedFlow(snr_scale=snr_scale)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, feat_channels * 4),
            nn.SiLU(),
            nn.Linear(feat_channels * 4, feat_channels * 4),
        )

        # 级联 Head
        self.head_series = nn.ModuleList([
            copy.deepcopy(single_head) for _ in range(num_heads)
        ])
        self.roi_extractor = roi_extractor
        self.criterion = criterion
        self.pre_noise_layer = pre_noise_layer

        # 耦合策略
        self.coupling = coupling if coupling is not None else build_coupling('random')

        # 采样器
        self._sampler = DiffusionSampler(
            diffusion_type=diffusion_type,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            solver_type=solver_type,
            ddim_sampling_eta=ddim_sampling_eta,
            rf_schedule=rf_schedule,
            rf_power=rf_power,
            rf_shift=rf_shift,
            snr_scale=snr_scale,
            box_renewal=box_renewal,
            use_ensemble=use_ensemble,
            use_nms=use_nms,
            nms_thr=nms_thr,
            score_thr=score_thr,
            min_keep=min_keep,
        )

        # 初始化
        self._init_weights(prior_prob)

        if torch_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward, dynamic=True)

    def _init_weights(self, prior_prob):
        for head in self.head_series:
            if hasattr(head, 'cls_head'):
                bias_value = -(math.log((1 - prior_prob) / prior_prob))
                nn.init.constant_(head.cls_head[-1].bias, bias_value)

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(self, features, bboxes, t):
        bs = features[0].shape[0]
        time_emb = self.time_mlp(t)

        all_cls_logits = []
        all_pred_bboxes = []
        proposals = None

        for head_idx, single_head in enumerate(self.head_series):
            if self.use_checkpoint and self.training:
                cls_logits, pred_bboxes, proposals = torch.utils.checkpoint.checkpoint(
                    single_head, features, bboxes, proposals, self.roi_extractor, time_emb,
                    use_reentrant=False,
                )
            else:
                cls_logits, pred_bboxes, proposals = single_head(
                    features, bboxes, proposals, self.roi_extractor, time_emb
                )

            all_cls_logits.append(cls_logits)
            all_pred_bboxes.append(pred_bboxes)

        return torch.stack(all_cls_logits), torch.stack(all_pred_bboxes), proposals

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def loss(
        self,
        features: Tuple[Tensor],
        img_metas: List[ImageMeta],
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
    ) -> Dict[str, Tensor]:
        """训练 loss

        Args:
            features: FPN 特征图
            img_metas: 图像元信息列表
            gt_bboxes: GT 框列表 (xyxy 格式)
            gt_labels: GT 标签列表

        Returns:
            loss dict
        """
        bs = len(img_metas)

        # 生成噪声提案
        noise = torch.randn(bs * self.num_proposals, 4, device=features[0].device)
        t = torch.rand((bs,), device=features[0].device)

        # 对每张图运行耦合 → 构建 x_t
        loss_dict = {}
        all_cls_logits = []
        all_pred_bboxes = []

        for i in range(bs):
            start = i * self.num_proposals
            end = start + self.num_proposals
            noise_i = noise[start:end]

            if len(gt_bboxes[i]) == 0:
                x_start = noise_i
            else:
                gt_xyxy = gt_bboxes[i]
                gt_norm = self._gt_to_diffusion(gt_xyxy, img_metas[i])
                x_start, _ = self.coupling.couple(
                    noise_i, gt_norm, gt_labels[i], noise.device
                )

            # Rectified Flow: x_t = (1-t)x_0 + t*x_1
            t_view = t[i].view(-1, 1)
            x_t = (1.0 - t_view) * x_start + t_view * noise_i

            # 将扩散空间转到图像空间供 head 使用
            bboxes_i = self._raw_to_xyxy(x_t.unsqueeze(0), [img_metas[i]])[0]
            noise[start:end] = x_t

        noise = noise.view(bs, self.num_proposals, 4)
        bboxes = self._raw_to_xyxy(noise, img_metas)

        # Head 前向
        time_emb = self.time_mlp(t)
        proposals = None
        for head_idx, single_head in enumerate(self.head_series):
            cls_logits, pred_bboxes, proposals = single_head(
                features, bboxes, proposals, self.roi_extractor, time_emb
            )
            all_cls_logits.append(cls_logits)
            all_pred_bboxes.append(pred_bboxes)

        all_cls_logits = torch.stack(all_cls_logits)
        all_pred_bboxes = torch.stack(all_pred_bboxes)

        if self.criterion is not None:
            # 包装为 ModelOutput + List[InstanceData]
            outputs = ModelOutput(
                pred_logits=all_cls_logits[-1],
                pred_boxes=all_pred_bboxes[-1],
                aux_outputs=[
                    ModelOutput(pred_logits=all_cls_logits[i], pred_boxes=all_pred_bboxes[i])
                    for i in range(len(all_cls_logits) - 1)
                ] if self.deep_supervision and len(all_cls_logits) > 1 else None,
            )
            targets = [
                InstanceData(bboxes=gt_bboxes[i], labels=gt_labels[i], img_shape=img_metas[i].img_shape)
                for i in range(bs)
            ]
            losses = self.criterion(outputs, targets)
            loss_dict.update(losses)

        return loss_dict

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def predict(
        self,
        features: Tuple[Tensor],
        img_metas: List[ImageMeta],
        rescale: bool = True,
        return_trajectory: bool = False,
    ) -> List[DetectionResult]:
        """推理: 迭代去噪 → 预测框"""
        bs = len(img_metas)
        device = features[0].device

        noise_raw = torch.randn(bs, self.num_proposals, 4, device=device)
        curr_bboxes = self._raw_to_xyxy(noise_raw, img_metas)
        curr_raw = noise_raw

        time_pairs = self._sampler.build_time_pairs(device)
        dpm_solver = self._sampler.create_dpm_solver()
        ensemble_results = []

        if dpm_solver is not None:
            dpm_solver.reset()

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            t_batch = torch.full((bs,), t_curr, device=device, dtype=torch.float32)
            cls_logits, pred_bboxes, _ = self(features, curr_bboxes, t_batch)

            cls_logits_last = cls_logits[-1]
            pred_bboxes_last = pred_bboxes[-1]

            if dpm_solver is not None and t_curr > 0.0:
                x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)
                curr_raw = dpm_solver.step(curr_raw, x0, t_curr, step_idx)
                curr_bboxes = self._sampler.raw_to_xyxy(curr_raw, img_metas)
            elif self.diffusion_type == 'rectified_flow':
                if self._sampler.solver_type == 'heun':
                    def model_fn(x, t_val):
                        t_b = torch.full((bs,), t_val, device=device)
                        return self(features, x, t_b)[-2][-1], None
                    curr_bboxes = self.rf.heun_step(
                        curr_bboxes, pred_bboxes_last, t_curr, t_next, model_fn
                    )
                else:
                    curr_bboxes = self.rf.step(
                        curr_bboxes, pred_bboxes_last, t_curr, t_next
                    )
                curr_raw = self._sampler.xyxy_to_raw(curr_bboxes, img_metas)
            else:
                # DDPM (deprecated)
                raise NotImplementedError("DDPM sampling is deprecated. Use RF.")

            if self._sampler.box_renewal:
                curr_raw = self._sampler.apply_box_renewal(curr_raw, cls_logits_last)

        # 最后一步预测
        t_batch = torch.zeros(bs, device=device)
        cls_logits, pred_bboxes, _ = self(features, curr_bboxes, t_batch)
        ensemble_results.append((cls_logits[-1], pred_bboxes[-1]))

        return self._sampler.post_process(ensemble_results, img_metas, rescale)

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

    def _gt_to_diffusion(self, gt_xyxy: Tensor, img_meta: ImageMeta) -> Tensor:
        """GT xyxy → 扩散空间"""
        h, w = _get_img_shape(img_meta)[:2]
        scale = gt_xyxy.new_tensor([w, h, w, h])
        gt_norm = gt_xyxy / scale
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_norm)
        return (gt_cxcywh * 2 - 1) * self.snr_scale

    def _raw_to_xyxy(self, raw, img_metas):
        return self._sampler.raw_to_xyxy(raw, img_metas)

    # ------------------------------------------------------------------
    # DDPM (deprecated)
    # ------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        alphas_cumprod = cosine_noise_schedule(self.timesteps).to(x_start.device)
        sqrt_alpha = load_buffer(alphas_cumprod.sqrt(), t, x_start.shape)
        sqrt_one_minus = load_buffer((1 - alphas_cumprod).sqrt(), t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise
