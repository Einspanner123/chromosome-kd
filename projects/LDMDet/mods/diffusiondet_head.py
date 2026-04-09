import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms

from .modules import (
    SinusoidalPositionEmbeddings,
    cosine_noise_schedule,
    load_buffer,
)
from .rectified_flow import RectifiedFlow
from .structures import DetectionResult, ImageMeta, InstanceData, ModelOutput
from .utils import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh


class DiffusionDetHead(nn.Module):
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
        solver_type: str = "euler",  # "euler" or "heun"
        self_condition: bool = False,
        box_renewal: bool = True,
        use_ensemble: bool = True,
        deep_supervision: bool = True,
        ddim_sampling_eta: float = 1.0,
        diffusion_type: str = "ddpm",  # "ddpm" or "rectified_flow"
        rf_schedule: str = "linear",  # "linear" or "power" or "shifted"
        rf_power: float = 1.0,  # 用于 power schedule
        rf_shift: float = 1.0,  # 用于 shifted schedule (如 SD3 中的 3.0)
        single_head: nn.Module = None,
        roi_extractor: nn.Module = None,
        criterion: nn.Module = None,
        use_nms: bool = True,
        nms_thr: float = 0.5,
        score_thr: float = 0.05,
        min_keep: int = 60,
        # === FlowDet 新增参数 ===
        noise_sampler: nn.Module = None,  # 结构化噪声采样器
        prediction_mode: str = "x0",  # "x0" 或 "velocity"
        velocity_loss_weight: float = 1.0,  # velocity loss 权重
        ot_coupling: bool = False,  # 训练时用 OT 最优耦合 (noise, GT) 配对
        ot_matcher: str = "nearest",  # "nearest" (原始 argmin) 或 "sinkhorn" (均衡配对)
        ot_epsilon: float = 1.0,  # Sinkhorn OT 的正则化参数
        ot_num_iters: int = 20,  # Sinkhorn OT 的迭代次数
        ot_init_mode: str = "replace",  # "replace" (替换 x_start) 或 "guided" (引导噪声初始化)
        ot_init_scale: float = 0.5,  # guided 模式下噪声到 GT 的缩放因子
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.self_condition = self_condition
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
        self.ot_coupling = ot_coupling
        self.ot_matcher = ot_matcher
        self.ot_epsilon = ot_epsilon
        self.ot_num_iters = ot_num_iters
        self.ot_init_mode = ot_init_mode
        self.ot_init_scale = ot_init_scale

        # 测试配置
        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

        # ROI 特征提取器和损失函数
        self.roi_extractor = roi_extractor
        self.criterion = criterion

        # 结构化噪声采样器 (Phase 3A)
        self.noise_sampler = noise_sampler

        # 构建扩散过程参数
        if self.diffusion_type == "ddpm":
            self._build_diffusion_buffers()
        elif self.diffusion_type == "rectified_flow":
            self.rf = RectifiedFlow(snr_scale=snr_scale)

        # 构建检测头序列 (迭代去噪)
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head) for _ in range(num_heads)]
        )

        # 时间步嵌入 MLP
        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.prior_prob = prior_prob
        self._init_weights()

    def _sample_noise(self, bs: int, device: torch.device) -> Tensor:
        """采样噪声框，支持结构化噪声"""
        if self.noise_sampler is not None:
            return self.noise_sampler.sample(bs, device)
        else:
            return torch.randn(bs, self.num_proposals, 4, device=device)

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
        targets = []
        for i in range(bs):
            h, w = self._get_img_shape(img_metas[i])[:2]
            scale = gt_bboxes[i].new_tensor([w, h, w, h])
            norm_bboxes = gt_bboxes[i] / scale
            targets.append(
                InstanceData(
                    labels=gt_labels[i],
                    bboxes=norm_bboxes,
                    img_shape=(h, w),
                )
            )

        # 2. 扩散过程：生成训练所需的噪声框
        if self.diffusion_type == "ddpm":
            t = torch.randint(0, self.timesteps, (bs,), device=device).long()
        else:
            t = torch.rand((bs,), device=device)
            if self.rf_schedule == "shifted":
                s = self.rf_shift
                t = s * t / (1 + (s - 1) * t)

        x_boxes = []
        x_starts = []  # 记录每个 proposal 对应的 x_start (扩散空间 GT)
        x_noises = []  # 记录每个 proposal 的噪声源
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt > 0:
                # --- 先准备 GT 在扩散空间的表示 ---
                norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)  # (K, 4)
                gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale  # (K, 4)

                # --- 采样噪声 ---
                noise = torch.randn(self.num_proposals, 4, device=device)  # (N, 4)

                if self.ot_coupling and self.diffusion_type == "rectified_flow":
                    if self.ot_matcher == "sinkhorn":
                        cost = torch.cdist(noise, gt_diffusion, p=2)
                        N, K = cost.shape
                        a = torch.ones(N, device=device) / N
                        proposals_per_gt = max(N // max(K, 1), 1)
                        gt_mass = torch.full((K,), proposals_per_gt / N, device=device)
                        b = gt_mass / gt_mass.sum()
                        log_K_mat = -cost / self.ot_epsilon
                        log_u = torch.zeros(N, device=device)
                        log_v = torch.zeros(K, device=device)
                        for _ in range(self.ot_num_iters):
                            log_u = torch.log(a + 1e-10) - torch.logsumexp(log_K_mat + log_v.unsqueeze(0), dim=1)
                            log_v = torch.log(b + 1e-10) - torch.logsumexp(log_K_mat + log_u.unsqueeze(1), dim=0)
                        transport = torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))
                        matched_gt_idx = transport.argmax(dim=1)
                        x_start = gt_diffusion[matched_gt_idx]
                    else:
                        cost = torch.cdist(noise, gt_diffusion, p=2)
                        matched_gt_idx = cost.argmin(dim=1)
                        x_start = gt_diffusion[matched_gt_idx]

                    if self.ot_init_mode == "guided":
                        x_start = noise + self.ot_init_scale * (x_start - noise)
                else:
                    # === 原始随机耦合 ===
                    idx = torch.randint(0, num_gt, (self.num_proposals,), device=device)
                    sample_bboxes = targets[i].bboxes[idx]
                    sample_bboxes = bbox_xyxy_to_cxcywh(sample_bboxes)
                    x_start = (sample_bboxes * 2 - 1) * self.snr_scale

                if self.diffusion_type == "ddpm":
                    x_noisy = self.q_sample(x_start, t[i : i + 1])
                    x_starts.append(x_start)
                    x_noises.append(torch.zeros_like(x_start))  # placeholder
                else:
                    x_noisy, _ = self.rf.q_sample(x_start, x_noise=noise, t=t[i : i + 1])
                    x_starts.append(x_start)
                    x_noises.append(noise)
                x_boxes.append(x_noisy)
            else:
                noise = torch.randn(self.num_proposals, 4, device=device)
                x_boxes.append(noise)
                x_starts.append(torch.zeros_like(noise))
                x_noises.append(noise)

        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._raw_to_xyxy(x_noisy_batch, img_metas)

        # 3. 前向传播获取预测结果
        t_input = t if self.diffusion_type == "ddpm" else t * self.timesteps
        all_cls_logits, all_pred_bboxes, all_objectness, all_velocity = self(
            features, curr_bboxes, t_input
        )

        # 4. 归一化预测框
        norm_pred_bboxes = all_pred_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = norm_pred_bboxes.new_tensor([w, h, w, h])
            norm_pred_bboxes[:, i] /= scale

        outputs = ModelOutput(
            pred_logits=all_cls_logits[-1],
            pred_boxes=norm_pred_bboxes[-1],
            pred_objectness=all_objectness[-1] if all_objectness[0] is not None else None,
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

        if self.criterion is None:
            raise ValueError("Criterion is not initialized in DiffusionDetHead")

        losses = self.criterion(outputs, targets)

        # 5. Velocity loss (Phase 4)
        # velocity_head 预测扩散空间的速度 v = x_start - x_noise
        # 使用训练时记录的 x_starts 和 x_noises (一一对应)
        if (
            self.prediction_mode == "velocity"
            and all_velocity[0] is not None
            and self.diffusion_type == "rectified_flow"
        ):
            x_start_batch = torch.stack(x_starts)  # (bs, N, 4)
            x_noise_batch = torch.stack(x_noises)  # (bs, N, 4)
            # target velocity: 从噪声到数据的方向 (与 RF 约定一致)
            v_target = x_start_batch - x_noise_batch  # (bs, N, 4)

            # 仅对最后一个 head 计算 velocity loss (避免过度正则化)
            last_v = all_velocity[-1]
            if last_v is not None:
                losses["loss_velocity"] = F.mse_loss(last_v, v_target) * self.velocity_loss_weight

        return losses

    def _init_weights(self):
        """初始化权重"""
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m.out_features in [self.num_classes, self.num_classes + 1]:
                        nn.init.constant_(m.bias, bias_value)
                    else:
                        nn.init.constant_(m.bias, 0)

        # 重新零初始化 AdaLN-Zero 的输出层（_init_weights 会覆盖它）
        for head in self.head_series:
            if hasattr(head, "adaln_mlp"):
                nn.init.zeros_(head.adaln_mlp[-1].weight)
                nn.init.zeros_(head.adaln_mlp[-1].bias)

    def _build_diffusion_buffers(self):
        """构建并注册扩散过程所需的常量 buffer"""
        betas = cosine_noise_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        alphas_cumprod = alphas_cumprod.float()
        alphas_cumprod_prev = alphas_cumprod_prev.float()

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas.float() * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """前向扩散采样: x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * noise"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = load_buffer(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = load_buffer(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        """从预测的 x0 还原噪声"""
        sqrt_recip_alphas_cumprod_t = load_buffer(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = load_buffer(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return (sqrt_recip_alphas_cumprod_t * x_t - x0) / sqrt_recipm1_alphas_cumprod_t

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

        curr_bboxes = bboxes
        curr_proposals = proposals

        for head in self.head_series:
            result = head(
                features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb
            )

            # 兼容旧版 (3 返回值) 和新版 (5 返回值) 的 single_head
            if len(result) == 5:
                cls_logits, pred_bboxes, curr_proposals, objectness, velocity = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
                objectness = None
                velocity = None

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_objectness.append(objectness)
            inter_velocity.append(velocity)

            curr_bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return (
                torch.stack(inter_cls_logits),
                torch.stack(inter_pred_bboxes),
                inter_objectness,
                inter_velocity,
            )
        else:
            return (
                torch.stack(inter_cls_logits[-1:]),
                torch.stack(inter_pred_bboxes[-1:]),
                inter_objectness[-1:],
                inter_velocity[-1:],
            )

    def _forward_at_t(
        self,
        features: Tuple[Tensor],
        x_raw: Tensor,
        t: float,
        img_metas: List[ImageMeta],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """在指定时间步 t 进行前向预测"""
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._raw_to_xyxy(x_raw, img_metas)

        t_input = torch.full((bs,), t * self.timesteps, device=device)

        cls_logits_seq, pred_bboxes_seq, _, _ = self(features, curr_bboxes, t_input)

        last_cls_logits = cls_logits_seq[-1]
        last_pred_bboxes = pred_bboxes_seq[-1]

        x0_raw = self._xyxy_to_raw(last_pred_bboxes, img_metas)
        logits_0_raw = last_cls_logits

        return last_cls_logits, last_pred_bboxes, x0_raw, logits_0_raw

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
        if self.diffusion_type == "ddpm":
            times = torch.linspace(
                -1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=device
            )
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
        else:
            times = torch.linspace(
                1.0, 0.0, steps=self.sampling_timesteps + 1, device=device
            )
            if self.rf_schedule == "power":
                times = times.pow(self.rf_power)
            elif self.rf_schedule == "shifted":
                s = self.rf_shift
                times = s * times / (1 + (s - 1) * times)

            time_pairs = []
            for i in range(len(times) - 1):
                time_pairs.append((times[i].item(), times[i + 1].item()))

        # 初始噪声框 (支持结构化噪声)
        x_raw = self._sample_noise(bs, device)

        ensemble_results = []
        trajectory = []

        # 2. 迭代采样
        for t_curr, t_next in time_pairs:
            cls_logits, pred_bboxes, x0_raw, logits_0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))

            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            if self.diffusion_type == "ddpm":
                curr_bboxes_xyxy, x_raw = self._ddim_step(
                    t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas
                )
                if t_next < 0:
                    break
            else:
                if self.solver_type == "heun" and t_next > 0:
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp, _ = self._forward_at_t(
                            features, x_tmp, t_tmp, img_metas
                        )
                        return x0_tmp, None

                    x_raw = self.rf.heun_step(
                        x_raw, x0_raw, t_curr, t_next, model_fn,
                    )
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                if self.box_renewal:
                    x_raw = self._apply_box_renewal(x_raw, cls_logits)

                if t_next <= 0:
                    break

        # 3. 后处理
        results = self._post_process(ensemble_results, img_metas, rescale)

        if return_trajectory:
            return results, trajectory

        return results

    def _apply_box_renewal(self, x_raw: Tensor, cls_logits: Tensor) -> Tensor:
        """通用的框更新策略"""
        bs, device = x_raw.shape[0], x_raw.device
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        x_raw_new = x_raw.clone()

        for i in range(bs):
            keep = scores[i] > self.score_thr
            if keep.sum() < self.min_keep:
                _, topk_idx = scores[i].topk(min(self.min_keep, scores.shape[1]))
                keep[topk_idx] = True

            num_renew = (~keep).sum()
            if num_renew > 0:
                x_raw_new[i, ~keep] = torch.randn(num_renew, 4, device=device)
        return x_raw_new

    @staticmethod
    def _get_img_shape(meta):
        """兼容 dict 和 ImageMeta 的 img_shape 获取"""
        if isinstance(meta, dict):
            return meta["img_shape"]
        return meta.img_shape

    @staticmethod
    def _get_scale_factor(meta):
        """兼容 dict 和 ImageMeta 的 scale_factor 获取"""
        if isinstance(meta, dict):
            return meta.get("scale_factor")
        return meta.scale_factor

    def _xyxy_to_raw(self, bboxes: Tensor, img_metas) -> Tensor:
        """将图像空间的 xyxy 框转回扩散空间的 raw 框"""
        x0 = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale
        return x0

    def _raw_to_xyxy(self, raw_bboxes: Tensor, img_metas) -> Tensor:
        """将扩散空间的 raw 框转为图像空间的 xyxy 框"""
        bboxes = (
            (raw_bboxes.clamp(-self.snr_scale, self.snr_scale) / self.snr_scale) + 1
        ) / 2
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        for i, meta in enumerate(img_metas):
            h, w = self._get_img_shape(meta)[:2]
            scale = bboxes.new_tensor([w, h, w, h])
            bboxes[i] *= scale
        return bboxes

    def _ddim_step(
        self, t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas: List[ImageMeta]
    ):
        """执行一步 DDIM 采样"""
        bs, device = x_raw.shape[0], x_raw.device

        x0 = self._xyxy_to_raw(pred_bboxes, img_metas)

        t_batch = torch.full((bs,), t_curr, device=device, dtype=torch.long)
        pred_noise = self.predict_noise_from_start(x_raw, t_batch, x0)

        alpha = self.alphas_cumprod[t_curr]
        alpha_next = self.alphas_cumprod[t_next]
        sigma = (
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        noise = torch.randn_like(x_raw)
        x_raw_next = x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise

        if self.box_renewal:
            x_raw_next = self._apply_box_renewal(x_raw_next, cls_logits)

        return self._raw_to_xyxy(x_raw_next, img_metas), x_raw_next

    def _post_process(
        self, ensemble_results, img_metas: List[ImageMeta], rescale: bool
    ) -> List[DetectionResult]:
        """后处理：集成、NMS、缩放"""
        results_list = []
        bs = len(img_metas)

        for i in range(bs):
            all_scores = []
            all_bboxes = []
            all_labels = []

            for cls_logits, pred_bboxes in ensemble_results:
                scores = torch.sigmoid(cls_logits[i])
                conf, labels = scores.max(-1)
                all_scores.append(conf)
                all_bboxes.append(pred_bboxes[i])
                all_labels.append(labels)

            final_scores = torch.cat(all_scores)
            final_bboxes = torch.cat(all_bboxes)
            final_labels = torch.cat(all_labels)

            if self.use_nms:
                keep = batched_nms(
                    final_bboxes, final_scores, final_labels, self.nms_thr,
                )
                final_scores = final_scores[keep]
                final_bboxes = final_bboxes[keep]
                final_labels = final_labels[keep]

            if rescale:
                scale_factor = self._get_scale_factor(img_metas[i])
                if scale_factor is None:
                    scale_factor = [1.0, 1.0, 1.0, 1.0]

                if isinstance(scale_factor, (list, tuple, Tensor)):
                    if len(scale_factor) == 2:
                        if isinstance(scale_factor, Tensor):
                            scale_factor = scale_factor.repeat(2)
                        else:
                            scale_factor = [
                                scale_factor[0], scale_factor[1],
                                scale_factor[0], scale_factor[1],
                            ]

                if not isinstance(scale_factor, Tensor):
                    scale_factor = final_bboxes.new_tensor(scale_factor)

                if scale_factor.dim() == 1:
                    scale_factor = scale_factor.unsqueeze(0)

                final_bboxes /= scale_factor

            results_list.append(
                DetectionResult(
                    bboxes=final_bboxes, scores=final_scores, labels=final_labels
                )
            )

        return results_list
