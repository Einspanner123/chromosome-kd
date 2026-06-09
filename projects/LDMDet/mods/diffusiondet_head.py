"""DiffusionDet 检测头

核心迭代去噪模块，协调 SingleHead 序列、ROI 提取、损失计算和采样推理。

实验证实的有效特性:
- Rectified Flow (RF) 替代 DDPM: mAP 0.733 vs 0.725
- Shifted Schedule (s=3): mAP 0.747
- AdaLN-Zero: mAP 0.751
- Heun Solver: mAP 0.751
- Stochastic OT (eps=5): mAP 0.751
- Group-Hierarchical Stochastic OT: mAP 0.752

已移除的无效特性 (详见 MASTER_TIMELINE.md):
- KCEC/DAEC: 信息论不可行 / 无增益
- 硬 OT (argmax): 消除训练多样性
- TRD: 增益极小且 velocity target 符号有误
- CAT: 增益极小且实现非纯曲率
- LSAS: 增益极小
- Velocity 预测模式: Reflow 退化，梯度冲突
- Objectness 分支: 低于基线
- Stratified 时间采样: 无实验支持
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .embeddings import SinusoidalPositionEmbeddings
from .noise_schedule import cosine_noise_schedule, load_buffer
from .ot_coupling import OTCoupling
from .rectified_flow import RectifiedFlow
from .sampling import DiffusionSampler, _get_img_shape
from .structures import DetectionResult, ImageMeta, InstanceData, ModelOutput
from .utils import bbox_xyxy_to_cxcywh


class DiffusionDetHead(nn.Module):
    """DiffusionDet 检测头

    通过迭代去噪实现目标检测。支持 DDPM 和 Rectified Flow 两种扩散范式，
    以及 Euler、Heun、DPM-Solver++ 等多种采样策略。

    关键子模块:
    - head_series: N 个 SingleHead 串联，每步细化预测
    - time_mlp: 时间步嵌入，注入条件信息
    - roi_extractor: 从 FPN 特征中提取 ROI 特征
    - criterion: 检测损失 (分类 + 回归 + GIoU)
    - ot_module: OT 耦合模块 (可选，用于训练配对)
    - _sampler: 扩散采样器 (非 nn.Module)
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
        min_keep: int = 60,
        # OT 耦合参数
        ot_coupling: bool = False,
        ot_epsilon: float = 1.0,
        ot_num_iters: int = 20,
        ot_sample_seed: Optional[int] = None,
        ot_group_hierarchical: bool = False,
        # 训练稳定化参数
        use_flash_attn: bool = False,
    ):
        super().__init__()

        # ---- 基础配置 ----
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self.deep_supervision = deep_supervision
        self.ddim_sampling_eta = ddim_sampling_eta
        self.diffusion_type = diffusion_type
        self.rf_schedule = rf_schedule
        self.rf_power = rf_power
        self.rf_shift = rf_shift
        self.solver_type = solver_type
        self.use_flash_attn = use_flash_attn

        # 测试配置
        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

        # OT 参数
        self.ot_coupling = ot_coupling

        # ---- 子模块 ----
        self.roi_extractor = roi_extractor
        self.criterion = criterion

        self.ot_module = OTCoupling(
            ot_coupling=ot_coupling,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
            ot_sample_seed=ot_sample_seed,
            ot_group_hierarchical=ot_group_hierarchical,
        )

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

        # ---- 扩散过程参数 ----
        if self.diffusion_type == 'ddpm':
            # DEPRECATED: DDPM — RF 已完全替代 DDPM，保留仅用于对比实验
            self._build_diffusion_buffers()
        elif self.diffusion_type == 'rectified_flow':
            self.rf = RectifiedFlow(snr_scale=snr_scale)

        # ---- 检测头序列 (迭代去噪) ----
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head) for _ in range(num_heads)]
        )
        for head in self.head_series:
            head.use_flash_attn = use_flash_attn

        # ---- 时间步嵌入 MLP ----
        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.prior_prob = prior_prob
        self._init_weights()

    # ================================================================
    # 前向传播
    # ================================================================

    def forward(
        self,
        features: Tuple[Tensor],
        bboxes: Tensor,
        t: Tensor,
        proposals: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List, List]:
        """前向传播，迭代去噪

        Args:
            features: FPN 特征元组
            bboxes: 当前边界框 [bs, num_proposals, 4] (xyxy)
            t: 当前时间步 [bs]
            proposals: 可选的提案特征

        Returns:
            all_cls_logits: [num_heads, bs, num_proposals, num_classes]
            all_pred_bboxes: [num_heads, bs, num_proposals, 4]
            all_curr_proposals: list of [bs, num_proposals, feat_dim]
        """
        time_emb = self.time_mlp(t)

        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_curr_proposals = []

        curr_bboxes = bboxes
        curr_proposals = proposals

        for head in self.head_series:
            result = head(
                features,
                curr_bboxes,
                curr_proposals,
                self.roi_extractor,
                time_emb,
            )

            if len(result) == 4:
                cls_logits, pred_bboxes, curr_proposals, _ = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_curr_proposals.append(curr_proposals)

            curr_bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return (
                torch.stack(inter_cls_logits),
                torch.stack(inter_pred_bboxes),
                inter_curr_proposals,
            )
        else:
            return (
                torch.stack(inter_cls_logits[-1:]),
                torch.stack(inter_pred_bboxes[-1:]),
                inter_curr_proposals[-1:],
            )

    # ================================================================
    # 训练损失
    # ================================================================

    def loss(
        self,
        features: Tuple[Tensor],
        img_metas: List[ImageMeta],
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
    ) -> Dict[str, Tensor]:
        """训练损失计算接口"""
        device = features[0].device
        bs = len(img_metas)

        # 1. 准备真值数据并归一化
        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)

        # 2. 时间步采样
        t = self._sample_t(bs, device)

        # 3. 构建训练配对
        x_boxes, x_starts, x_noises, matched_gt_indices = (
            self._build_training_targets(
                bs,
                device,
                t,
                targets,
                gt_bboxes,
                img_metas,
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

        # 4. 前向传播
        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps
        all_cls_logits, all_pred_bboxes, all_curr_proposals = self(
            features, curr_bboxes, t_input
        )

        # 5. 归一化并计算检测损失
        norm_pred_bboxes = self._normalize_pred_bboxes(
            all_pred_bboxes, img_metas
        )
        outputs = self._build_outputs(all_cls_logits, norm_pred_bboxes)
        losses = self.criterion(outputs, targets)

        return losses

    # ================================================================
    # 推理
    # ================================================================

    @torch.no_grad()
    def predict(
        self,
        features: Tuple[Tensor],
        img_metas: List[ImageMeta],
        rescale: bool = True,
        return_trajectory: bool = False,
    ) -> List[DetectionResult]:
        """推理模式的前向传播"""
        device = features[0].device
        bs = len(img_metas)

        # 1. 准备采样时间序列
        time_pairs = self._sampler.build_time_pairs(device)

        # 初始噪声框
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        ensemble_results = []
        trajectory = []

        dpm_solver = self._sampler.create_dpm_solver()

        # 2. 迭代采样
        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))

            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            if self.diffusion_type == 'ddpm':
                # DEPRECATED: DDPM — RF 已完全替代 DDPM
                curr_bboxes_xyxy, x_raw = self._sampler.ddim_step(
                    t_curr,
                    t_next,
                    x_raw,
                    cls_logits,
                    pred_bboxes,
                    img_metas,
                    self.alphas_cumprod,
                )
                if t_next < 0:
                    break
            else:
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx)
                elif self.solver_type == 'heun' and t_next > 0:

                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = self._forward_at_t(
                            features, x_tmp, t_tmp, img_metas
                        )
                        return x0_tmp, None

                    x_raw = self.rf.heun_step(
                        x_raw,
                        x0_raw,
                        t_curr,
                        t_next,
                        model_fn,
                    )
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                if self.box_renewal:
                    x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)

                if t_next <= 0:
                    break

        # 3. 后处理
        results = self._sampler.post_process(
            ensemble_results, img_metas, rescale
        )

        if return_trajectory:
            return results, trajectory

        return results

    # ================================================================
    # 训练辅助方法
    # ================================================================

    def _normalize_targets(
        self,
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
        img_metas: List[ImageMeta],
        bs: int,
    ) -> List[InstanceData]:
        targets = []
        for i in range(bs):
            h, w = _get_img_shape(img_metas[i])[:2]
            scale = gt_bboxes[i].new_tensor([w, h, w, h])
            targets.append(
                InstanceData(
                    labels=gt_labels[i],
                    bboxes=gt_bboxes[i] / scale,
                    img_shape=(h, w),
                )
            )
        return targets

    def _sample_t(self, bs: int, device: torch.device) -> Tensor:
        """采样训练时间步"""
        if self.diffusion_type == 'ddpm':
            # DEPRECATED: DDPM
            return torch.randint(
                0, self.timesteps, (bs,), device=device
            ).long()
        t = torch.rand((bs,), device=device)
        if self.rf_schedule == 'shifted':
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        return t

    def _build_training_targets(
        self,
        bs: int,
        device: torch.device,
        t: Tensor,
        targets: List[InstanceData],
        gt_bboxes: List[Tensor],
        img_metas: List[ImageMeta],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """构建训练配对: 为每张图生成噪声框和对应的 GT 起点"""
        x_boxes = []
        x_starts = []
        x_noises = []
        matched_gt_indices = []

        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt == 0:
                noise = torch.randn(self.num_proposals, 4, device=device)
                x_boxes.append(noise)
                x_starts.append(torch.zeros_like(noise))
                x_noises.append(noise)
                matched_gt_indices.append(
                    torch.zeros(
                        self.num_proposals, dtype=torch.long, device=device
                    )
                )
                continue

            # GT in diffusion space
            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale
            noise = torch.randn(self.num_proposals, 4, device=device)

            # Coupling
            x_start, matched_idx = self._couple_single_image(
                i, noise, gt_diffusion, targets[i].labels, device
            )
            matched_gt_indices.append(matched_idx)

            # Forward diffusion
            x_noisy, x_noise = self._forward_diffusion(
                x_start, noise, t[i : i + 1]
            )
            x_starts.append(x_start)
            x_noises.append(x_noise)
            x_boxes.append(x_noisy)

        return x_boxes, x_starts, x_noises, matched_gt_indices

    def _couple_single_image(
        self,
        img_idx: int,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """单张图像的 OT 耦合或随机配对"""
        if self.ot_coupling and self.diffusion_type == 'rectified_flow':
            x_start, matched_idx = self.ot_module.couple(
                noise, gt_diffusion, gt_labels, device
            )
            return x_start, matched_idx

        # 随机配对
        num_gt = gt_diffusion.shape[0]
        idx = torch.randint(0, num_gt, (self.num_proposals,), device=device)
        x_start = gt_diffusion[idx]
        return x_start, idx

    def _forward_diffusion(
        self,
        x_start: Tensor,
        noise: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """前向扩散: 返回 (x_noisy, x_noise_used)"""
        if self.diffusion_type == 'ddpm':
            # DEPRECATED: DDPM
            x_noisy = self.q_sample(x_start, t)
            return x_noisy, torch.zeros_like(x_start)
        else:
            x_noisy, _ = self.rf.q_sample(x_start, x_noise=noise, t=t)
            return x_noisy, noise

    def _forward_at_t(
        self,
        features: Tuple[Tensor],
        x_raw: Tensor,
        t: float,
        img_metas: List[ImageMeta],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """在指定时间步 t 进行前向预测"""
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)

        t_input = torch.full((bs,), t * self.timesteps, device=device)

        cls_logits_seq, pred_bboxes_seq, _ = self(
            features, curr_bboxes, t_input
        )

        last_cls_logits = cls_logits_seq[-1]
        last_pred_bboxes = pred_bboxes_seq[-1]

        x0_raw = self._sampler.xyxy_to_raw(last_pred_bboxes, img_metas)

        return last_cls_logits, last_pred_bboxes, x0_raw

    # ================================================================
    # 损失辅助方法
    # ================================================================

    def _normalize_pred_bboxes(
        self, all_pred_bboxes: Tensor, img_metas
    ) -> Tensor:
        normed = all_pred_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            normed[:, i] /= normed.new_tensor([w, h, w, h])
        return normed

    def _build_outputs(self, all_cls_logits, norm_pred_bboxes) -> ModelOutput:
        outputs = ModelOutput(
            pred_logits=all_cls_logits[-1],
            pred_boxes=norm_pred_bboxes[-1],
        )
        if self.deep_supervision and self.num_heads > 1:
            outputs.aux_outputs = [
                ModelOutput(
                    pred_logits=all_cls_logits[i],
                    pred_boxes=norm_pred_bboxes[i],
                )
                for i in range(self.num_heads - 1)
            ]
        return outputs

    # ================================================================
    # DDPM 扩散过程 (DEPRECATED)
    # ================================================================

    # DEPRECATED: DDPM — RF 已完全替代 DDPM，保留仅用于对比实验

    def _build_diffusion_buffers(self):
        """构建并注册扩散过程所需的常量 buffer"""
        betas = cosine_noise_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        alphas_cumprod = alphas_cumprod.float()
        alphas_cumprod_prev = alphas_cumprod_prev.float()

        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas.float()
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """前向扩散采样: x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * noise"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = load_buffer(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = load_buffer(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    # ================================================================
    # 初始化
    # ================================================================

    def _init_weights(self):
        """初始化权重"""
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m.out_features in [
                        self.num_classes,
                        self.num_classes + 1,
                    ]:
                        nn.init.constant_(m.bias, bias_value)
                    else:
                        nn.init.constant_(m.bias, 0)

        # 重新零初始化 AdaLN-Zero 的输出层
        for head in self.head_series:
            if hasattr(head, 'adaln_mlp'):
                nn.init.zeros_(head.adaln_mlp[-1].weight)
                nn.init.zeros_(head.adaln_mlp[-1].bias)
