"""DiffusionDet 检测头

核心迭代去噪模块，协调 SingleHead 序列、ROI 提取、损失计算和采样推理。
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
        prediction_mode: str = 'x0',
        velocity_loss_weight: float = 1.0,
        # OT 耦合参数
        ot_coupling: bool = False,
        ot_matcher: str = 'nearest',
        ot_epsilon: float = 1.0,
        ot_num_iters: int = 20,
        ot_sample: bool = False,
        ot_sample_seed: Optional[int] = None,
        ot_group_hierarchical: bool = False,
        ot_kcec: bool = False,
        kcec_morph_weight: float = 0.25,
        kcec_group_weight: float = 0.1,
        kcec_cls_weight: float = 0.0,
        kcec_quota_strength: float = 1.0,
        kcec_slack: float = 0.05,
        kcec_conf_threshold: float = 0.5,
        kcec_scale_anneal: bool = False,
        kcec_log_interval: int = 100,
        # DAEC 参数
        daec_contrastive_weight: float = 0.1,
        daec_temperature: float = 0.05,
        # TRD 参数
        use_trd: bool = False,
        trd_self_cond_prob: float = 0.5,
        trd_delta_t: Optional[float] = None,
        # CAT 参数
        use_cat: bool = False,
        cat_weight: float = 0.1,
        cat_delta_t: float = 0.01,
        cat_loss_type: str = 'x0_consistency',
        # LSAS 参数
        use_lsas: bool = False,
        lsas_num_bins: int = 100,
        lsas_temp: float = 1.0,
        # 训练稳定化参数
        t_sampling: str = 'uniform',
        t_sampling_bins: int = 8,
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
        self.prediction_mode = prediction_mode
        self.velocity_loss_weight = velocity_loss_weight
        self.use_trd = use_trd
        self.trd_self_cond_prob = trd_self_cond_prob
        self.trd_delta_t = cat_delta_t if trd_delta_t is None else trd_delta_t
        self.use_cat = use_cat
        self.cat_weight = cat_weight
        self.cat_delta_t = cat_delta_t
        self.cat_loss_type = cat_loss_type
        self.use_lsas = use_lsas
        self.lsas_num_bins = lsas_num_bins
        self.lsas_temp = lsas_temp
        self.t_sampling = t_sampling
        self.t_sampling_bins = t_sampling_bins
        self.use_flash_attn = use_flash_attn

        # 测试配置
        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

        # OT / DAEC 参数 (代理到 ot_module，同时保留顶层属性以便外部访问)
        self.ot_coupling = ot_coupling
        self.ot_kcec = ot_kcec
        self.kcec_cls_weight = kcec_cls_weight
        self.daec_contrastive_weight = daec_contrastive_weight
        self.daec_temperature = daec_temperature

        # ---- 子模块 ----
        self.roi_extractor = roi_extractor
        self.criterion = criterion

        self.ot_module = OTCoupling(
            ot_coupling=ot_coupling,
            ot_matcher=ot_matcher,
            ot_epsilon=ot_epsilon,
            ot_num_iters=ot_num_iters,
            ot_sample=ot_sample,
            ot_sample_seed=ot_sample_seed,
            ot_group_hierarchical=ot_group_hierarchical,
            ot_kcec=ot_kcec,
            kcec_morph_weight=kcec_morph_weight,
            kcec_group_weight=kcec_group_weight,
            kcec_cls_weight=kcec_cls_weight,
            kcec_quota_strength=kcec_quota_strength,
            kcec_slack=kcec_slack,
            kcec_conf_threshold=kcec_conf_threshold,
            kcec_scale_anneal=kcec_scale_anneal,
            kcec_log_interval=kcec_log_interval,
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
            use_trd=use_trd,
            trd_delta_t=self.trd_delta_t,
        )

        # ---- 扩散过程参数 ----
        if self.diffusion_type == 'ddpm':
            self._build_diffusion_buffers()
        elif self.diffusion_type == 'rectified_flow':
            self.rf = RectifiedFlow(snr_scale=snr_scale)

        # ---- 检测头序列 (迭代去噪) ----
        if (
            hasattr(single_head, 'prediction_mode')
            and single_head.prediction_mode != prediction_mode
        ):
            single_head.prediction_mode = prediction_mode
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head) for _ in range(num_heads)]
        )

        if prediction_mode == 'velocity':
            for head in self.head_series:
                if (
                    not hasattr(head, 'velocity_head')
                    or head.velocity_head is None
                ):
                    head.velocity_head = nn.Sequential(
                        nn.Linear(feat_channels, feat_channels, bias=False),
                        nn.LayerNorm(feat_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(feat_channels, feat_channels, bias=False),
                        nn.LayerNorm(feat_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(feat_channels, 4),
                    )
                    head.prediction_mode = 'velocity'
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

        # LSAS: 可学习时间分布
        if self.use_lsas and self.diffusion_type == 'rectified_flow':
            self.lsas_logits = nn.Parameter(torch.zeros(lsas_num_bins))
            self.lsas_bin_edges = torch.linspace(0, 1, lsas_num_bins + 1)

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
            all_objectness: list of [bs, num_proposals, 1] or [None, ...]
            all_velocity: list of [bs, num_proposals, 4] or [None, ...]
        """
        time_emb = self.time_mlp(t)

        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_objectness = []
        inter_velocity = []
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

            if len(result) == 5:
                (
                    cls_logits,
                    pred_bboxes,
                    curr_proposals,
                    objectness,
                    velocity,
                ) = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
                objectness = None
                velocity = None

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_objectness.append(objectness)
            inter_velocity.append(velocity)
            inter_curr_proposals.append(curr_proposals)

            curr_bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return (
                torch.stack(inter_cls_logits),
                torch.stack(inter_pred_bboxes),
                inter_objectness,
                inter_velocity,
                inter_curr_proposals,
            )
        else:
            return (
                torch.stack(inter_cls_logits[-1:]),
                torch.stack(inter_pred_bboxes[-1:]),
                inter_objectness[-1:],
                inter_velocity[-1:],
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
        t, lsas_log_probs = self._sample_t(bs, device)

        # DAEC Phase 2: 获取语义信息用于耦合
        cls_logits_for_coupling = self._get_cls_logits_for_coupling(
            features, t, bs, device
        )

        # 3. 构建训练配对
        x_boxes, x_starts, x_noises, kcec_log, matched_gt_indices = (
            self._build_training_targets(
                bs,
                device,
                t,
                targets,
                gt_bboxes,
                img_metas,
                cls_logits_for_coupling=cls_logits_for_coupling,
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

        # 4. 前向传播 (含 TRD 自条件化)
        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps
        (
            all_cls_logits,
            all_pred_bboxes,
            all_objectness,
            all_velocity,
            all_curr_proposals,
        ) = self._forward_trd(
            features, curr_bboxes, t, t_input, x_noisy_batch, img_metas, bs
        )

        # 5. 归一化并计算检测损失
        norm_pred_bboxes = self._normalize_pred_bboxes(
            all_pred_bboxes, img_metas
        )
        outputs = self._build_outputs(
            all_cls_logits, norm_pred_bboxes, all_objectness
        )
        losses = self.criterion(outputs, targets)

        # DAEC: 监督对比损失
        self._add_contrastive_loss(
            losses, all_curr_proposals, matched_gt_indices, targets, bs, device
        )

        # 6. 辅助损失
        self._add_velocity_loss(
            losses, all_velocity, x_starts, x_noises, device
        )
        self._add_cat_loss(
            losses,
            features,
            t,
            x_starts,
            x_noises,
            all_pred_bboxes,
            img_metas,
            bs,
            device,
        )
        self._add_lsas_loss(losses, lsas_log_probs)

        if kcec_log:
            for k, v in kcec_log.items():
                losses[k] = v.detach()

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
        x0_prev = None

        ensemble_results = []
        trajectory = []

        dpm_solver = self._sampler.create_dpm_solver()

        # 2. 迭代采样
        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            if (
                self.use_trd
                and x0_prev is not None
                and self.diffusion_type == 'rectified_flow'
            ):
                t_curr_t = torch.full((bs,), t_curr, device=device)
                t_view = t_curr_t.view(-1, 1, 1)
                v_ot_est = (x_raw - x0_prev) / torch.clamp(t_view, min=1e-5)
                x_raw_refined = x_raw + self.trd_delta_t * v_ot_est
                cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                    features, x_raw_refined, t_curr, img_metas
                )
            else:
                cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                    features, x_raw, t_curr, img_metas
                )

            x0_prev = x0_raw.detach()

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))

            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            if self.diffusion_type == 'ddpm':
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

    def _sample_t(
        self, bs: int, device: torch.device
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """采样训练时间步"""
        if self.diffusion_type == 'ddpm':
            return torch.randint(
                0, self.timesteps, (bs,), device=device
            ).long(), None
        if self.use_lsas:
            return self._sample_time_lsas(bs, device)
        if self.t_sampling == 'stratified':
            n_bins = self.t_sampling_bins
            bin_size = bs // n_bins
            remainder = bs % n_bins
            parts = []
            for i in range(n_bins):
                n = bin_size + (1 if i < remainder else 0)
                t_bin = (i + torch.rand(n, device=device)) / n_bins
                parts.append(t_bin)
            t = torch.cat(parts).clamp(1e-5, 1.0 - 1e-5)
            t = t[torch.randperm(bs, device=device)]
        else:
            t = torch.rand((bs,), device=device)
        if self.rf_schedule == 'shifted':
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        return t, None

    def _sample_time_lsas(
        self, bs: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """LSAS 时间步采样"""
        probs = F.softmax(self.lsas_logits / self.lsas_temp, dim=0)
        bin_indices = torch.multinomial(probs, bs, replacement=True)
        bin_width = 1.0 / self.lsas_num_bins
        t = (
            bin_indices.float() * bin_width
            + torch.rand(bs, device=device) * bin_width
        )
        t = t.clamp(1e-5, 1.0 - 1e-5)
        log_probs = torch.log(probs[bin_indices] + 1e-10)
        return t, log_probs

    def _get_cls_logits_for_coupling(
        self, features, t, bs, device
    ) -> Optional[Tensor]:
        """DAEC Phase 2: 获取语义信息用于耦合"""
        if not (
            self.training
            and self.ot_coupling
            and self.ot_kcec
            and self.kcec_cls_weight > 0
        ):
            return None

        with torch.no_grad():
            noise_probe = torch.randn(bs, self.num_proposals, 4, device=device)
            curr_bboxes_probe = self._sampler.raw_to_xyxy(noise_probe, [])
            t_input_probe = (
                t if self.diffusion_type == 'ddpm' else t * self.timesteps
            )
            all_cls_logits_probe, _, _, _, _ = self(
                features, curr_bboxes_probe, t_input_probe
            )
            return all_cls_logits_probe[-1]

    def _build_training_targets(
        self,
        bs: int,
        device: torch.device,
        t: Tensor,
        targets: List[InstanceData],
        gt_bboxes: List[Tensor],
        img_metas: List[ImageMeta],
        cls_logits_for_coupling: Optional[Tensor] = None,
    ) -> Tuple[
        List[Tensor],
        List[Tensor],
        List[Tensor],
        Dict[str, Tensor],
        List[Tensor],
    ]:
        """构建训练配对: 为每张图生成噪声框和对应的 GT 起点"""
        x_boxes = []
        x_starts = []
        x_noises = []
        matched_gt_indices = []
        kcec_log_stats: Dict[str, Tensor] = {}

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
                i,
                noise,
                gt_diffusion,
                targets[i].labels,
                device,
                cls_logits_for_coupling,
            )
            matched_gt_indices.append(matched_idx)

            # Forward diffusion
            x_noisy, x_noise = self._forward_diffusion(
                x_start, noise, t[i : i + 1]
            )
            x_starts.append(x_start)
            x_noises.append(x_noise)
            x_boxes.append(x_noisy)

        # 合并 KCEC 日志
        kcec_log = self._merge_kcec_logs(kcec_log_stats)

        return x_boxes, x_starts, x_noises, kcec_log, matched_gt_indices

    def _couple_single_image(
        self,
        img_idx: int,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
        cls_logits_for_coupling: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """单张图像的 OT 耦合或随机配对"""
        if self.ot_coupling and self.diffusion_type == 'rectified_flow':
            cls_logits_i = (
                cls_logits_for_coupling[img_idx]
                if cls_logits_for_coupling is not None
                else None
            )
            x_start, _, matched_idx = self.ot_module.couple(
                noise,
                gt_diffusion,
                gt_labels,
                device,
                cls_logits=cls_logits_i,
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
            x_noisy = self.q_sample(x_start, t)
            return x_noisy, torch.zeros_like(x_start)
        else:
            x_noisy, _ = self.rf.q_sample(x_start, x_noise=noise, t=t)
            return x_noisy, noise

    @staticmethod
    def _merge_kcec_logs(log_stats: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """合并 KCEC 日志 (当前为空，OTCoupling 内部处理)"""
        return log_stats

    def _forward_trd(
        self,
        features,
        curr_bboxes,
        t,
        t_input,
        x_noisy_batch,
        img_metas,
        bs,
    ) -> Tuple:
        """前向传播, 可选 TRD 自条件化"""
        if not (
            self.use_trd
            and self.diffusion_type == 'rectified_flow'
            and self.training
        ):
            return self(features, curr_bboxes, t_input)

        sc_mask = torch.rand(bs, device=t.device) < self.trd_self_cond_prob
        if not sc_mask.any():
            return self(features, curr_bboxes, t_input)

        with torch.no_grad():
            _, all_pred_sc, _, _, _ = self(features, curr_bboxes, t_input)
            x0_sc = self._sampler.xyxy_to_raw(all_pred_sc[-1], img_metas)
        x_noisy_sc = x_noisy_batch.clone()
        t_view = t.view(-1, 1, 1)
        v_ot_est = (x_noisy_sc - x0_sc) / torch.clamp(t_view, min=1e-5)
        x_noisy_sc[sc_mask] = (x_noisy_sc + self.trd_delta_t * v_ot_est)[
            sc_mask
        ]
        t_sc = t.clone()
        t_sc[sc_mask] = (t[sc_mask] + self.trd_delta_t).clamp(0, 1)
        curr_bboxes_sc = self._sampler.raw_to_xyxy(x_noisy_sc, img_metas)
        return self(features, curr_bboxes_sc, t_sc * self.timesteps)

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

        cls_logits_seq, pred_bboxes_seq, _, _, _ = self(
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

    def _build_outputs(
        self, all_cls_logits, norm_pred_bboxes, all_objectness
    ) -> ModelOutput:
        outputs = ModelOutput(
            pred_logits=all_cls_logits[-1],
            pred_boxes=norm_pred_bboxes[-1],
            pred_objectness=all_objectness[-1]
            if all_objectness[0] is not None
            else None,
        )
        if self.deep_supervision and self.num_heads > 1:
            outputs.aux_outputs = [
                ModelOutput(
                    pred_logits=all_cls_logits[i],
                    pred_boxes=norm_pred_bboxes[i],
                    pred_objectness=(
                        all_objectness[i]
                        if all_objectness[0] is not None
                        else None
                    ),
                )
                for i in range(self.num_heads - 1)
            ]
        return outputs

    def _add_contrastive_loss(
        self,
        losses,
        all_curr_proposals,
        matched_gt_indices,
        targets,
        bs,
        device,
    ):
        """DAEC: 监督对比损失"""
        last_proposals = all_curr_proposals[-1]
        if last_proposals is None:
            return

        last_proposals = last_proposals.squeeze(0).view(
            bs, self.num_proposals, -1
        )
        loss_contrastive = torch.tensor(0.0, device=device)
        valid_bs = 0
        for i in range(bs):
            gt_idx = matched_gt_indices[i]
            if len(targets[i].labels) == 0:
                continue
            matched_labels = targets[i].labels[gt_idx]

            features_i = F.normalize(last_proposals[i], dim=1)

            if features_i.shape[0] != matched_labels.shape[0]:
                continue

            temperature = self.daec_temperature
            sim_matrix = torch.matmul(features_i, features_i.T) / temperature

            mask = torch.eq(
                matched_labels.unsqueeze(1), matched_labels.unsqueeze(0)
            ).float()
            mask.fill_diagonal_(0)

            if mask.sum() == 0:
                continue

            exp_sim = torch.exp(sim_matrix)
            log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

            mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (
                mask.sum(dim=1) + 1e-8
            )
            loss_contrastive = loss_contrastive - mean_log_prob_pos.mean()
            valid_bs += 1

        if valid_bs > 0:
            weight = getattr(self, 'daec_contrastive_weight', 0.1)
            losses['loss_contrastive'] = (loss_contrastive / valid_bs) * weight

    def _add_velocity_loss(
        self,
        losses,
        all_velocity,
        x_starts,
        x_noises,
        device,
    ):
        if not (
            self.prediction_mode == 'velocity'
            and all_velocity[0] is not None
            and self.diffusion_type == 'rectified_flow'
        ):
            return
        v_target = torch.stack(x_noises) - torch.stack(x_starts)
        vel_weight = self.velocity_loss_weight

        last_v = all_velocity[-1]
        if last_v is not None:
            losses['loss_velocity'] = F.mse_loss(last_v, v_target) * vel_weight
        if self.deep_supervision and self.num_heads > 1:
            aux = torch.tensor(0.0, device=device)
            n = 0
            for hi in range(self.num_heads - 1):
                if all_velocity[hi] is not None:
                    aux = aux + F.mse_loss(all_velocity[hi], v_target)
                    n += 1
            if n > 0:
                losses['loss_velocity_aux'] = aux / n * vel_weight * 0.5

    def _add_cat_loss(
        self,
        losses,
        features,
        t,
        x_starts,
        x_noises,
        all_pred_bboxes,
        img_metas,
        bs,
        device,
    ):
        if not (
            self.use_cat
            and self.diffusion_type == 'rectified_flow'
            and self.training
        ):
            return
        x_start_batch = torch.stack(x_starts)
        x_noise_batch = torch.stack(x_noises)
        dt = self.cat_delta_t
        with torch.no_grad():
            t2 = (t + dt).clamp(0, 1)
            t2_view = t2.view(-1, 1, 1)
            x_t2 = (1.0 - t2_view) * x_start_batch + t2_view * x_noise_batch
            curr_bboxes_t2 = self._sampler.raw_to_xyxy(x_t2, img_metas)
            _, all_pred_t2, _, _, _ = self(
                features, curr_bboxes_t2, t2 * self.timesteps
            )
            x0_t2 = self._sampler.xyxy_to_raw(all_pred_t2[-1], img_metas)
        x0_t1 = self._sampler.xyxy_to_raw(all_pred_bboxes[-1], img_metas)
        if self.cat_loss_type == 'velocity_curvature':
            t1_safe = t.view(-1, 1, 1).clamp_min(1e-3)
            t2_safe = t2_view.clamp_min(1e-3)
            v_t1 = (x_noise_batch - x0_t1) / t1_safe
            v_t2 = (x_noise_batch - x0_t2.detach()) / t2_safe
            losses['loss_curvature'] = F.mse_loss(v_t1, v_t2) * self.cat_weight
        else:
            losses['loss_curvature'] = (
                F.mse_loss(x0_t1, x0_t2.detach()) * self.cat_weight
            )

    @staticmethod
    def _add_lsas_loss(losses, lsas_log_probs):
        if lsas_log_probs is None:
            return
        with torch.no_grad():
            total = sum(losses.values())
        losses['loss_lsas'] = total.detach() * (-lsas_log_probs).mean() * 0.01

    # ================================================================
    # DDPM 扩散过程
    # ================================================================

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
