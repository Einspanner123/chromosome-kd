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
        self_condition: bool = False,
        box_renewal: bool = True,
        use_ensemble: bool = True,
        deep_supervision: bool = True,
        ddim_sampling_eta: float = 1.0,
        diffusion_type: str = "ddpm",  # "ddpm" or "rectified_flow"
        single_head: nn.Module = None,
        roi_extractor: nn.Module = None,
        criterion: nn.Module = None,
        use_nms: bool = True,
        nms_thr: float = 0.5,
        score_thr: float = 0.05,
        min_keep: int = 60,
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

        # 测试配置
        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

        # ROI 特征提取器和损失函数
        self.roi_extractor = roi_extractor
        self.criterion = criterion

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
            h, w = img_metas[i].img_shape[:2]
            scale = gt_bboxes[i].new_tensor([w, h, w, h])
            # 归一化 xyxy
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
            # Rectified Flow 使用 [0, 1] 之间的连续时间步
            t = torch.rand((bs,), device=device)

        x_boxes = []
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt > 0:
                # 随机重复采样 GT 框到 num_proposals 个
                idx = torch.randint(0, num_gt, (self.num_proposals,), device=device)
                sample_bboxes = targets[i].bboxes[idx]
                # xyxy -> cxcywh
                sample_bboxes = bbox_xyxy_to_cxcywh(sample_bboxes)
                # [0, 1] -> [-snr, snr]
                x_start = (sample_bboxes * 2 - 1) * self.snr_scale

                if self.diffusion_type == "ddpm":
                    # DDPM 加噪
                    x_noisy = self.q_sample(x_start, t[i : i + 1])
                else:
                    # Rectified Flow 加噪: x_t = (1-t)x_0 + t*x_1
                    x_noisy, _ = self.rf.q_sample(x_start, t=t[i : i + 1])
                x_boxes.append(x_noisy)
            else:
                # 全纯噪声
                x_boxes.append(torch.randn(self.num_proposals, 4, device=device))

        x_noisy_batch = torch.stack(x_boxes)  # [bs, num_proposals, 4]
        # 转换为 xyxy 格式用于 RoIAlign
        curr_bboxes = self._raw_to_xyxy(x_noisy_batch, img_metas)

        # 3. 前向传播获取预测结果
        # 对于 RF，我们可能需要将 t 缩放到 [0, timesteps] 以适配预训练的时间嵌入
        t_input = t if self.diffusion_type == "ddpm" else t * self.timesteps
        all_cls_logits, all_pred_bboxes = self(features, curr_bboxes, t_input)

        # 4. 整理输出格式并计算损失
        # all_cls_logits: [num_heads, bs, num_proposals, num_classes]
        # all_pred_bboxes: [num_heads, bs, num_proposals, 4] (图像空间的 xyxy)

        # 归一化预测框以适配 Matcher
        norm_pred_bboxes = all_pred_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = meta.img_shape[:2]
            scale = norm_pred_bboxes.new_tensor([w, h, w, h])
            norm_pred_bboxes[:, i] /= scale

        outputs = ModelOutput(
            pred_logits=all_cls_logits[-1],
            pred_boxes=norm_pred_bboxes[-1],
        )
        if self.deep_supervision and self.num_heads > 1:
            outputs.aux_outputs = [
                ModelOutput(
                    pred_logits=all_cls_logits[i], pred_boxes=norm_pred_bboxes[i]
                )
                for i in range(self.num_heads - 1)
            ]

        if self.criterion is None:
            raise ValueError("Criterion is not initialized in DiffusionDetHead")

        return self.criterion(outputs, targets)

    def _init_weights(self):
        """初始化权重"""
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 如果是分类层（最后一层），使用先验概率初始化偏置
                    if m.out_features in [self.num_classes, self.num_classes + 1]:
                        nn.init.constant_(m.bias, bias_value)
                    else:
                        nn.init.constant_(m.bias, 0)

    def _build_diffusion_buffers(self):
        """构建并注册扩散过程所需的常量 buffer"""
        betas = cosine_noise_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 转换为 float32 减少计算开销并保持类型一致性
        alphas_cumprod = alphas_cumprod.float()
        alphas_cumprod_prev = alphas_cumprod_prev.float()

        # 注册 buffers
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

        # 后验分布 q(x_{t-1} | x_t, x_0) 参数
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
    ) -> Tuple[Tensor, Tensor]:
        """前向传播，迭代去噪

        Args:
            features: FPN 特征元组
            bboxes: 当前边界框 [bs, num_proposals, 4] (xyxy)
            t: 当前时间步 [bs]
            proposals: 可选的提案特征 [bs, num_proposals, feat_channels]

        Returns:
            all_cls_logits: [num_heads, bs, num_proposals, num_classes]
            all_pred_bboxes: [num_heads, bs, num_proposals, 4]
        """
        time_emb = self.time_mlp(t)

        inter_cls_logits = []
        inter_pred_bboxes = []

        curr_bboxes = bboxes
        curr_proposals = proposals

        for head in self.head_series:
            # single_head 的 forward 签名: (features, bboxes, proposals, pooler, time_emb)
            cls_logits, pred_bboxes, curr_proposals = head(
                features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb
            )

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)

            # 迭代更新边界框 (detach 以防梯度在迭代间传导)
            curr_bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return torch.stack(inter_cls_logits), torch.stack(inter_pred_bboxes)
        else:
            return inter_cls_logits[-1:], inter_pred_bboxes[-1:]

    @torch.no_grad()
    def predict(
        self, features: Tuple[Tensor], img_metas: List[ImageMeta], rescale: bool = True
    ) -> List[DetectionResult]:
        """推理接口"""
        device = features[0].device
        bs = len(img_metas)

        # 1. 准备初始噪声框
        # 生成时间步序列
        if self.diffusion_type == "ddpm":
            times = torch.linspace(
                -1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=device
            )
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))
        else:
            # Rectified Flow: t 从 1.0 (noise) 到 0.0 (data)
            times = torch.linspace(
                1.0, 0.0, steps=self.sampling_timesteps + 1, device=device
            )
            time_pairs = []
            for i in range(len(times) - 1):
                time_pairs.append((times[i].item(), times[i + 1].item()))

        # 初始随机噪声框 [-snr, snr] -> [0, 1] -> xyxy
        noise_bboxes_raw = torch.randn(bs, self.num_proposals, 4, device=device)
        curr_bboxes = self._raw_to_xyxy(noise_bboxes_raw, img_metas)

        # 存储集成结果
        ensemble_results = []

        # 2. 迭代采样
        for t_curr, t_next in time_pairs:
            # 前向预测所需的时间步
            if self.diffusion_type == "ddpm":
                t_batch = torch.full((bs,), t_curr, device=device, dtype=torch.long)
                t_input = t_batch
            else:
                # RF: 输入给网络的时间步需要缩放
                t_batch = torch.full((bs,), t_curr, device=device)
                t_input = t_batch * self.timesteps

            # 前向预测
            cls_logits_seq, pred_bboxes_seq = self(features, curr_bboxes, t_input)

            # 取序列最后一个输出
            last_cls_logits = cls_logits_seq[-1]
            last_pred_bboxes = pred_bboxes_seq[-1]

            if self.diffusion_type == "ddpm":
                if t_next < 0:
                    if self.use_ensemble:
                        ensemble_results.append((last_cls_logits, last_pred_bboxes))
                    break
                # DDIM 采样步骤
                curr_bboxes, noise_bboxes_raw = self._ddim_step(
                    t_curr,
                    t_next,
                    noise_bboxes_raw,
                    last_cls_logits,
                    last_pred_bboxes,
                    img_metas,
                )
            else:
                if t_next <= 0:
                    if self.use_ensemble:
                        ensemble_results.append((last_cls_logits, last_pred_bboxes))
                    break
                # Rectified Flow Euler 采样步骤
                curr_bboxes, noise_bboxes_raw = self._rf_step(
                    t_curr,
                    t_next,
                    noise_bboxes_raw,
                    last_cls_logits,
                    last_pred_bboxes,
                    img_metas,
                )

            if self.use_ensemble:
                ensemble_results.append((last_cls_logits, last_pred_bboxes))

        # 3. 后处理结果
        return self._post_process(ensemble_results, img_metas, rescale)

    def _raw_to_xyxy(self, raw_bboxes: Tensor, img_metas: List[ImageMeta]) -> Tensor:
        """将扩散空间的 raw 框转为图像空间的 xyxy 框"""
        # [-snr, snr] -> [0, 1]
        bboxes = (
            (raw_bboxes.clamp(-self.snr_scale, self.snr_scale) / self.snr_scale) + 1
        ) / 2
        # cxcywh -> xyxy
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        # 映射到图像尺寸
        for i, meta in enumerate(img_metas):
            h, w = meta.img_shape[:2]
            scale = bboxes.new_tensor([w, h, w, h])
            bboxes[i] *= scale
        return bboxes

    def _ddim_step(
        self, t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas: List[ImageMeta]
    ):
        """执行一步 DDIM 采样"""
        bs, device = x_raw.shape[0], x_raw.device

        # 将预测的 xyxy 转回扩散空间的 cxcywh (x0)
        x0 = pred_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = meta.img_shape[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale

        # 预测噪声
        t_batch = torch.full((bs,), t_curr, device=device, dtype=torch.long)
        pred_noise = self.predict_noise_from_start(x_raw, t_batch, x0)

        # DDIM 参数
        alpha = self.alphas_cumprod[t_curr]
        alpha_next = self.alphas_cumprod[t_next]
        sigma = (
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        # 更新 x_raw
        noise = torch.randn_like(x_raw)
        x_raw_next = x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # 框更新策略 (Box Renewal)
        if self.box_renewal:
            scores = torch.sigmoid(cls_logits).max(-1)[0]
            for i in range(bs):
                keep = scores[i] > self.score_thr
                # 保证至少保留一些框（针对染色体任务的优化）
                if keep.sum() < self.min_keep:
                    _, topk_idx = scores[i].topk(min(self.min_keep, scores.shape[1]))
                    keep[topk_idx] = True

                # 替换低置信度框为新的随机噪声
                num_renew = (~keep).sum()
                if num_renew > 0:
                    x_raw_next[i, ~keep] = torch.randn(num_renew, 4, device=device)

        return self._raw_to_xyxy(x_raw_next, img_metas), x_raw_next

    def _rf_step(
        self, t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas: List[ImageMeta]
    ):
        """执行一步 Rectified Flow ODE 采样"""
        bs, device = x_raw.shape[0], x_raw.device

        # 将预测的 xyxy 转回扩散空间的 cxcywh (x0)
        x0 = pred_bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = meta.img_shape[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale

        # RF 采样一步
        x_raw_next = self.rf.step(x_raw, x0, t_curr, t_next)

        # 框更新策略 (Box Renewal)
        if self.box_renewal:
            scores = torch.sigmoid(cls_logits).max(-1)[0]
            for i in range(bs):
                keep = scores[i] > self.score_thr
                if keep.sum() < self.min_keep:
                    _, topk_idx = scores[i].topk(min(self.min_keep, scores.shape[1]))
                    keep[topk_idx] = True

                num_renew = (~keep).sum()
                if num_renew > 0:
                    # 重新从噪声 (t=1.0) 采样
                    x_raw_next[i, ~keep] = torch.randn(num_renew, 4, device=device)

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
                # 获取每个框的最大得分和类别
                conf, labels = scores.max(-1)
                all_scores.append(conf)
                all_bboxes.append(pred_bboxes[i])
                all_labels.append(labels)

            # 合并集成结果
            final_scores = torch.cat(all_scores)
            final_bboxes = torch.cat(all_bboxes)
            final_labels = torch.cat(all_labels)

            # NMS
            if self.use_nms:
                keep = batched_nms(
                    final_bboxes,
                    final_scores,
                    final_labels,
                    self.nms_thr,
                )
                final_scores = final_scores[keep]
                final_bboxes = final_bboxes[keep]
                final_labels = final_labels[keep]

            # 缩放回原始尺寸
            if rescale:
                scale_factor = img_metas[i].scale_factor
                if scale_factor is None:
                    scale_factor = [1.0, 1.0, 1.0, 1.0]

                if isinstance(scale_factor, (list, tuple, Tensor)):
                    if len(scale_factor) == 2:
                        if isinstance(scale_factor, Tensor):
                            scale_factor = scale_factor.repeat(2)
                        else:
                            scale_factor = [
                                scale_factor[0],
                                scale_factor[1],
                                scale_factor[0],
                                scale_factor[1],
                            ]

                if not isinstance(scale_factor, Tensor):
                    scale_factor = final_bboxes.new_tensor(scale_factor)

                # Ensure scale_factor is [1, 4] for broadcasting if final_bboxes is [N, 4]
                if scale_factor.dim() == 1:
                    scale_factor = scale_factor.unsqueeze(0)

                final_bboxes /= scale_factor

            results_list.append(
                DetectionResult(
                    bboxes=final_bboxes, scores=final_scores, labels=final_labels
                )
            )

        return results_list
