"""DiffusionDet 检测头

核心迭代去噪模块，协调 SingleHead 序列、ROI 提取、损失计算和采样推理。

实验证实的有效特性:
- Rectified Flow (RF) 替代 DDPM: mAP 0.733 vs 0.725
- Shifted Schedule (s=3): mAP 0.747
- AdaLN-Zero: mAP 0.751
- Heun Solver: mAP 0.751
- Stochastic OT (eps=5): mAP 0.751
- Group-Hierarchical Stochastic OT: mAP 0.752

实验性特性 (需消融验证):
- TRD (Transport-Refinement Decomposition): 速度分解为传输+残差
- CAT (Curvature-Aware Training): 曲率正则化 (x0_consistency / velocity_curvature)
- Velocity Loss: 辅助速度场拟合监督

已移除的无效特性 (详见 MASTER_TIMELINE.md):
- KCEC/DAEC: 信息论不可行 / 无增益
- 硬 OT (argmax): 消除训练多样性
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
        # TRD 参数
        use_trd: bool = False,
        trd_delta_t: float = 0.05,
        trd_use_analytic_v: bool = True,
        trd_weight: float = 1.0,
        # CAT 参数
        use_cat: bool = False,
        cat_delta_t: float = 0.02,
        cat_loss_type: str = 'x0_consistency',
        cat_weight: float = 1.0,
        # Velocity loss 参数
        use_velocity_loss: bool = False,
        velocity_loss_weight: float = 1.0,
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

        # TRD/CAT/Velocity loss 参数
        self.use_trd = use_trd
        self.trd_delta_t = trd_delta_t
        self.trd_use_analytic_v = trd_use_analytic_v
        self.trd_weight = trd_weight
        self.use_cat = use_cat
        self.cat_delta_t = cat_delta_t
        self.cat_loss_type = cat_loss_type
        self.cat_weight = cat_weight
        self.use_velocity_loss = use_velocity_loss
        self.velocity_loss_weight = velocity_loss_weight

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

        # 6. 辅助 loss (TRD / CAT / Velocity)
        x_starts_batch = torch.stack(x_starts)
        x_noises_batch = torch.stack(x_noises)

        if self.use_velocity_loss and self.diffusion_type == 'rectified_flow':
            losses.update(
                self._add_velocity_loss(
                    all_pred_bboxes, x_starts_batch, x_noises_batch,
                    t, img_metas,
                )
            )

        if self.use_cat:
            losses.update(
                self._add_cat_loss(
                    features, all_pred_bboxes, x_starts_batch, x_noises_batch,
                    t, img_metas,
                )
            )

        if self.use_trd:
            losses.update(
                self._add_trd_loss(
                    features, all_pred_bboxes, x_starts_batch, x_noises_batch,
                    t, img_metas,
                )
            )

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

    # ================================================================
    # TRD / CAT / Velocity Loss
    # ================================================================

    def _add_velocity_loss(
        self,
        all_pred_bboxes: Tensor,
        x_starts: Tensor,
        x_noises: Tensor,
        t: Tensor,
        img_metas: List[ImageMeta],
    ) -> Dict[str, Tensor]:
        """Velocity loss: 匹配 RF 目标速度 v* = x_noise - x_start

        仅在 prediction_mode='x0' 时作为辅助 loss 使用。
        当模型预测 x0 时，velocity loss 提供额外的速度场拟合监督。
        """
        # 从预测 bboxes 推导 x0_pred (raw diffusion space)
        last_pred_bboxes = all_pred_bboxes[-1]  # [bs, num_proposals, 4]
        x0_pred = self._sampler.xyxy_to_raw(last_pred_bboxes, img_metas)

        # 目标速度: v* = x_1 - x_0
        v_target = x_noises - x_starts

        # 从 x0_pred 推导预测速度: v_pred = (x_t - x0_pred) / t
        t_view = t.view(-1, 1, 1)
        x_noisy = (1.0 - t_view) * x_starts + t_view * x_noises
        v_pred = (x_noisy - x0_pred) / torch.clamp(t_view, min=1e-5)

        loss = F.mse_loss(v_pred, v_target) * self.velocity_loss_weight
        return {'loss_velocity': loss}

    def _add_cat_loss(
        self,
        features: Tuple[Tensor],
        all_pred_bboxes: Tensor,
        x_starts: Tensor,
        x_noises: Tensor,
        t: Tensor,
        img_metas: List[ImageMeta],
    ) -> Dict[str, Tensor]:
        """Curvature-Aware Training (CAT): 惩罚速度场的时间变化率

        支持两种模式:
        - 'x0_consistency': 惩罚 |x0_pred(t) - x0_pred(t+dt)|^2
        - 'velocity_curvature': 惩罚 |v(t+dt) - v(t)|^2 (纯曲率正则)
        """
        t_view = t.view(-1, 1, 1)
        t2 = (t + self.cat_delta_t).clamp(max=1.0)
        t2_view = t2.view(-1, 1, 1)

        # 构造 x_{t+dt} = (1 - t2) * x_start + t2 * x_noise
        x_t2 = (1.0 - t2_view) * x_starts + t2_view * x_noises

        # 在 t+dt 处前向传播
        curr_bboxes_t2 = self._sampler.raw_to_xyxy(x_t2, img_metas)
        t2_input = t2 * self.timesteps
        _, all_pred_bboxes_t2, _ = self(features, curr_bboxes_t2, t2_input)

        # 取最后一层 head 的预测
        x0_pred_t1 = self._sampler.xyxy_to_raw(all_pred_bboxes[-1], img_metas)
        x0_pred_t2 = self._sampler.xyxy_to_raw(all_pred_bboxes_t2[-1], img_metas)

        if self.cat_loss_type == 'x0_consistency':
            # 模式 1: x0 一致性
            loss = F.mse_loss(x0_pred_t1, x0_pred_t2.detach()) * self.cat_weight
        elif self.cat_loss_type == 'velocity_curvature':
            # 模式 2: 纯曲率正则化 |v(t+dt) - v(t)|^2
            x_noisy_t1 = (1.0 - t_view) * x_starts + t_view * x_noises
            t_safe = torch.clamp(t_view, min=0.01)
            t2_safe = torch.clamp(t2_view, min=0.01)
            v_t1 = (x_noisy_t1 - x0_pred_t1) / t_safe
            v_t2 = (x_t2 - x0_pred_t2.detach()) / t2_safe
            loss = F.mse_loss(v_t1, v_t2) * self.cat_weight
        else:
            raise ValueError(
                f"Unknown cat_loss_type: {self.cat_loss_type}. "
                f"Expected 'x0_consistency' or 'velocity_curvature'."
            )

        return {'loss_curvature': loss}

    def _add_trd_loss(
        self,
        features: Tuple[Tensor],
        all_pred_bboxes: Tensor,
        x_starts: Tensor,
        x_noises: Tensor,
        t: Tensor,
        img_metas: List[ImageMeta],
    ) -> Dict[str, Tensor]:
        """Transport-Refinement Decomposition (TRD) loss

        将速度分解为传输分量 v_pi 和特征修正分量 delta_v:
        v_theta = v_pi + delta_v

        训练时:
        1. 从 x_t 前进到 x_{t+dt} (自条件化)
        2. 在 x_{t+dt} 处预测 x0
        3. 计算残差速度 delta_v = v_pred - v_pi
        4. 惩罚 delta_v 与 (v* - v_pi) 的差异

        推理时使用自条件化: 将 v_pi 注入模型输入
        """
        t_view = t.view(-1, 1, 1)
        t2 = (t + self.trd_delta_t).clamp(max=1.0)
        t2_view = t2.view(-1, 1, 1)

        # 1. 计算传输速度 v_pi
        if self.trd_use_analytic_v:
            # 使用解析传输速度: v* = x_1 - x_0 (当前 OT 配对)
            v_pi = x_noises - x_starts
        else:
            # 使用模型预测估计 v_pi
            x0_pred = self._sampler.xyxy_to_raw(all_pred_bboxes[-1], img_metas)
            x_noisy = (1.0 - t_view) * x_starts + t_view * x_noises
            v_pi = (x_noisy - x0_pred) / torch.clamp(t_view, min=1e-5)

        # 2. 自条件化: 从 x_t 前进到 x_{t+dt}
        x_noisy = (1.0 - t_view) * x_starts + t_view * x_noises
        x_t2_sc = x_noisy + self.trd_delta_t * v_pi  # Euler step

        # 3. 在 x_{t+dt} 处前向传播
        curr_bboxes_t2 = self._sampler.raw_to_xyxy(x_t2_sc, img_metas)
        t2_input = t2 * self.timesteps
        _, all_pred_bboxes_t2, _ = self(features, curr_bboxes_t2, t2_input)

        # 4. 计算残差速度
        x0_pred_t2 = self._sampler.xyxy_to_raw(all_pred_bboxes_t2[-1], img_metas)
        v_pred_t2 = (x_t2_sc - x0_pred_t2) / torch.clamp(t2_view, min=1e-5)

        # 目标速度 v* = x_1 - x_0
        v_target = x_noises - x_starts

        # 残差: delta_v = v_pred - v_pi, 目标: v* - v_pi
        delta_v_pred = v_pred_t2 - v_pi
        delta_v_target = v_target - v_pi

        loss = F.mse_loss(delta_v_pred, delta_v_target.detach()) * self.trd_weight
        return {'loss_trd': loss}

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
