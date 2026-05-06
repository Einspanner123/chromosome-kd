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
        solver_type: str = "euler",
        self_condition: bool = False,
        box_renewal: bool = True,
        use_ensemble: bool = True,
        deep_supervision: bool = True,
        ddim_sampling_eta: float = 1.0,
        diffusion_type: str = "ddpm",
        rf_schedule: str = "linear",
        rf_power: float = 1.0,
        rf_shift: float = 1.0,
        single_head: nn.Module = None,
        roi_extractor: nn.Module = None,
        criterion: nn.Module = None,
        use_nms: bool = True,
        nms_thr: float = 0.5,
        score_thr: float = 0.05,
        min_keep: int = 60,
        ot_coupling: bool = False,
        ot_matcher: str = "nearest",
        ot_epsilon: float = 1.0,
        ot_num_iters: int = 20,
        ot_sample: bool = False,
        roi_share: bool = False,
        roi_share_iou_thr: float = 0.95,
        proposal_prune_ratios: Optional[List[float]] = None,
        domain_adaptive_prior: bool = False,
        go_lsd: bool = False,
        go_lsd_weight: float = 1.0,
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
        self.roi_share = roi_share
        self.roi_share_iou_thr = roi_share_iou_thr
        self.proposal_prune_ratios = proposal_prune_ratios
        self.domain_adaptive_prior = domain_adaptive_prior
        self.go_lsd = go_lsd
        self.go_lsd_weight = go_lsd_weight

        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep
        self.ot_coupling = ot_coupling
        self.ot_matcher = ot_matcher
        self.ot_epsilon = ot_epsilon
        self.ot_num_iters = ot_num_iters
        self.ot_sample = ot_sample

        self.roi_extractor = roi_extractor
        self.criterion = criterion

        if self.diffusion_type == "ddpm":
            self._build_diffusion_buffers()
        elif self.diffusion_type == "rectified_flow":
            self.rf = RectifiedFlow(snr_scale=snr_scale)

        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head) for _ in range(num_heads)]
        )

        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.prior_prob = prior_prob
        self._init_weights()

        if self.domain_adaptive_prior:
            self.domain_encoder = nn.Sequential(
                nn.Linear(feat_channels, feat_channels),
                nn.LayerNorm(feat_channels),
                nn.ReLU(inplace=True),
                nn.Linear(feat_channels, 8),
            )
            nn.init.zeros_(self.domain_encoder[-1].weight)
            nn.init.zeros_(self.domain_encoder[-1].bias)
        else:
            self.domain_encoder = None

    def loss(
        self,
        features: Tuple[Tensor],
        img_metas: List[ImageMeta],
        gt_bboxes: List[Tensor],
        gt_labels: List[Tensor],
    ) -> Dict[str, Tensor]:
        device = features[0].device
        bs = len(img_metas)

        targets = []
        for i in range(bs):
            h, w = img_metas[i].img_shape[:2]
            scale = gt_bboxes[i].new_tensor([w, h, w, h])
            norm_bboxes = gt_bboxes[i] / scale
            targets.append(
                InstanceData(
                    labels=gt_labels[i],
                    bboxes=norm_bboxes,
                    img_shape=(h, w),
                )
            )

        if self.diffusion_type == "ddpm":
            t = torch.randint(0, self.timesteps, (bs,), device=device).long()
        else:
            t = torch.rand((bs,), device=device)
            if self.rf_schedule == "shifted":
                s = self.rf_shift
                t = s * t / (1 + (s - 1) * t)

        if self.domain_adaptive_prior:
            domain_mu, domain_sigma = self._get_domain_prior(
                features, bs, self.num_proposals, device
            )

        x_boxes = []
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt > 0:
                noise = None
                if self.ot_coupling and self.diffusion_type == "rectified_flow":
                    norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
                    gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale
                    noise = torch.randn(self.num_proposals, 4, device=device)

                    if self.ot_matcher == "sinkhorn":
                        cost = torch.cdist(noise, gt_diffusion, p=2)
                        n_prop, n_gt = cost.shape
                        a = torch.ones(n_prop, device=device) / n_prop
                        proposals_per_gt = max(n_prop // max(n_gt, 1), 1)
                        gt_mass = torch.full(
                            (n_gt,), proposals_per_gt / n_prop, device=device
                        )
                        b = gt_mass / gt_mass.sum()
                        log_k = -cost / self.ot_epsilon
                        log_u = torch.zeros(n_prop, device=device)
                        log_v = torch.zeros(n_gt, device=device)
                        for _ in range(self.ot_num_iters):
                            log_u = torch.log(a + 1e-10) - torch.logsumexp(
                                log_k + log_v.unsqueeze(0), dim=1
                            )
                            log_v = torch.log(b + 1e-10) - torch.logsumexp(
                                log_k + log_u.unsqueeze(1), dim=0
                            )
                        transport = torch.exp(
                            log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
                        )
                        if self.ot_sample:
                            row_probs = transport / transport.sum(
                                dim=1, keepdim=True
                            ).clamp_min(1e-10)
                            matched_gt_idx = torch.multinomial(row_probs, 1).squeeze(-1)
                        else:
                            matched_gt_idx = transport.argmax(dim=1)
                    else:
                        cost = torch.cdist(noise, gt_diffusion, p=2)
                        matched_gt_idx = cost.argmin(dim=1)

                    x_start = gt_diffusion[matched_gt_idx]
                else:
                    idx = torch.randint(0, num_gt, (self.num_proposals,), device=device)
                    sample_bboxes = targets[i].bboxes[idx]
                    sample_bboxes = bbox_xyxy_to_cxcywh(sample_bboxes)
                    x_start = (sample_bboxes * 2 - 1) * self.snr_scale

                if self.diffusion_type == "ddpm":
                    x_noisy = self.q_sample(x_start, t[i : i + 1])
                else:
                    if self.domain_adaptive_prior:
                        noise = self._sample_adaptive_noise(
                            domain_mu[i : i + 1], domain_sigma[i : i + 1], x_start.shape
                        )
                        x_noisy, _ = self.rf.q_sample(
                            x_start, x_noise=noise, t=t[i : i + 1]
                        )
                    else:
                        if noise is not None:
                            x_noisy, _ = self.rf.q_sample(
                                x_start, x_noise=noise, t=t[i : i + 1]
                            )
                        else:
                            x_noisy, _ = self.rf.q_sample(x_start, t=t[i : i + 1])
                x_boxes.append(x_noisy)
            else:
                if self.domain_adaptive_prior:
                    noise = self._sample_adaptive_noise(
                        domain_mu[i : i + 1],
                        domain_sigma[i : i + 1],
                        (self.num_proposals, 4),
                    )
                    x_boxes.append(noise)
                else:
                    x_boxes.append(torch.randn(self.num_proposals, 4, device=device))

        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._raw_to_xyxy(x_noisy_batch, img_metas)

        t_input = t if self.diffusion_type == "ddpm" else t * self.timesteps
        all_cls_logits, all_pred_bboxes = self(features, curr_bboxes, t_input)

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

        losses = self.criterion(outputs, targets)

        if self.go_lsd and self.training and self.num_heads > 1:
            lsd_losses = self._compute_go_lsd_loss(
                all_cls_logits, all_pred_bboxes, norm_pred_bboxes
            )
            for k, v in lsd_losses.items():
                losses[k] = v * self.go_lsd_weight

        return losses

    def _get_domain_prior(
        self, features: Tuple[Tensor], bs: int, num_boxes: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coarse_feat = features[0].mean(dim=[2, 3])
        params = self.domain_encoder(coarse_feat)
        mu = params[:, :4]
        log_sigma = params[:, 4:]
        sigma = torch.exp(log_sigma)
        mu = mu.view(bs, 1, 4)
        sigma = sigma.view(bs, 1, 4)
        return mu, sigma

    def _compute_go_lsd_loss(
        self, all_cls_logits: Tensor, all_pred_bboxes: Tensor, norm_pred_bboxes: Tensor
    ) -> Dict[str, Tensor]:
        num_heads = all_cls_logits.shape[0]
        teacher_cls = all_cls_logits[-1].detach()
        teacher_boxes = norm_pred_bboxes[-1].detach()
        lsd_losses = {}

        for h in range(num_heads - 1):
            student_cls = all_cls_logits[h]
            student_boxes = norm_pred_bboxes[h]
            lsd_losses[f"lsd_head{h}_cls"] = F.kl_div(
                F.log_softmax(student_cls, dim=-1),
                F.softmax(teacher_cls, dim=-1),
                reduction="batchmean",
            )
            lsd_losses[f"lsd_head{h}_bbox"] = F.l1_loss(student_boxes, teacher_boxes)

        return lsd_losses

    def _sample_adaptive_noise(self, mu, sigma, shape):
        raw = torch.randn(shape, device=mu.device)
        while mu.dim() < raw.dim():
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        result = mu + sigma * raw
        while result.dim() > len(shape):
            result = result.squeeze(0)
        return result

    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m.out_features in [self.num_classes, self.num_classes + 1]:
                        nn.init.constant_(m.bias, bias_value)
                    else:
                        nn.init.constant_(m.bias, 0)

    def _build_diffusion_buffers(self):
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
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = load_buffer(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = load_buffer(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t: Tensor, t: Tensor, x0: Tensor) -> Tensor:
        sqrt_recip_alphas_cumprod_t = load_buffer(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = load_buffer(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return (sqrt_recip_alphas_cumprod_t * x_t - x0) / sqrt_recipm1_alphas_cumprod_t

    @staticmethod
    def _bbox_mean_iou(bboxes_a: Tensor, bboxes_b: Tensor) -> float:
        lt = torch.max(bboxes_a[..., :2], bboxes_b[..., :2])
        rb = torch.min(bboxes_a[..., 2:], bboxes_b[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = (bboxes_a[..., 2] - bboxes_a[..., 0]) * (
            bboxes_a[..., 3] - bboxes_a[..., 1]
        )
        area_b = (bboxes_b[..., 2] - bboxes_b[..., 0]) * (
            bboxes_b[..., 3] - bboxes_b[..., 1]
        )
        iou = inter / (area_a + area_b - inter + 1e-6)
        return iou.mean().item()

    def _prune_proposals(
        self,
        cls_logits: Tensor,
        pred_bboxes: Tensor,
        curr_proposals: Tensor,
        keep_ratio: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if keep_ratio >= 1.0:
            return cls_logits, pred_bboxes, curr_proposals, None

        bs, num_boxes = cls_logits.shape[:2]
        keep_k = max(int(num_boxes * keep_ratio), self.min_keep)

        scores = torch.sigmoid(cls_logits).max(-1)[0]
        _, topk_idx = scores.topk(keep_k, dim=1)

        pruned_cls = torch.gather(
            cls_logits, 1, topk_idx.unsqueeze(-1).expand(-1, -1, cls_logits.shape[-1])
        )
        pruned_bboxes = torch.gather(
            pred_bboxes, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4)
        )

        if curr_proposals is not None:
            p = curr_proposals.view(bs, num_boxes, -1)
            pruned_p = torch.gather(
                p, 1, topk_idx.unsqueeze(-1).expand(-1, -1, p.shape[-1])
            ).view(1, bs * keep_k, -1)
        else:
            pruned_p = None

        return pruned_cls, pruned_bboxes, pruned_p, topk_idx

    def forward(
        self,
        features: Tuple[Tensor],
        bboxes: Tensor,
        t: Tensor,
        proposals: Optional[Tensor] = None,
        skip_prune: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        time_emb = self.time_mlp(t)

        inter_cls_logits = []
        inter_pred_bboxes = []

        curr_bboxes = bboxes
        curr_proposals = proposals
        cached_roi_features = None
        cached_roi_bboxes = None

        for head_idx, head in enumerate(self.head_series):
            if (
                self.roi_share
                and head_idx > 0
                and cached_roi_features is not None
                and cached_roi_bboxes is not None
            ):
                iou = self._bbox_mean_iou(curr_bboxes, cached_roi_bboxes)
                use_cached = iou >= self.roi_share_iou_thr
            else:
                use_cached = False

            if use_cached:
                cls_logits, pred_bboxes, curr_proposals = head.forward_with_cached_roi(
                    features, curr_bboxes, curr_proposals, cached_roi_features, time_emb
                )
            else:
                cls_logits, pred_bboxes, curr_proposals, roi_features = head(
                    features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb
                )
                if self.roi_share:
                    cached_roi_features = roi_features
                    cached_roi_bboxes = curr_bboxes.detach().clone()

            if (
                not skip_prune
                and self.proposal_prune_ratios is not None
                and not self.training
            ):
                ratio_idx = min(head_idx, len(self.proposal_prune_ratios) - 1)
                ratio = self.proposal_prune_ratios[ratio_idx]
                if ratio < 1.0:
                    cls_logits, pred_bboxes, curr_proposals, _ = self._prune_proposals(
                        cls_logits, pred_bboxes, curr_proposals, ratio
                    )
                    cached_roi_features = None
                    cached_roi_bboxes = None

            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)

            curr_bboxes = pred_bboxes.detach()

        if self.deep_supervision and self.training:
            return torch.stack(inter_cls_logits), torch.stack(inter_pred_bboxes)
        else:
            return inter_cls_logits[-1:], inter_pred_bboxes[-1:]

    def _forward_at_t(
        self,
        features: Tuple[Tensor],
        x_raw: Tensor,
        t: float,
        img_metas: List[ImageMeta],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._raw_to_xyxy(x_raw, img_metas)

        t_input = torch.full((bs,), t * self.timesteps, device=device)

        cls_logits_seq, pred_bboxes_seq = self(
            features, curr_bboxes, t_input, skip_prune=True
        )

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
        device = features[0].device
        bs = len(img_metas)

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

        if self.domain_adaptive_prior:
            domain_mu, domain_sigma = self._get_domain_prior(
                features, bs, self.num_proposals, device
            )
            x_raw = self._sample_adaptive_noise(
                domain_mu, domain_sigma, (bs, self.num_proposals, 4)
            )
        else:
            x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        ensemble_results = []
        trajectory = []

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
                        x_raw,
                        x0_raw,
                        t_curr,
                        t_next,
                        model_fn,
                    )
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                should_renew = self.box_renewal and self.sampling_timesteps > 1
                if should_renew:
                    x_raw = self._apply_box_renewal(x_raw, cls_logits)

                if t_next <= 0:
                    break

        results = self._post_process(ensemble_results, img_metas, rescale)

        if return_trajectory:
            return results, trajectory

        return results

    def _apply_box_renewal(self, x_raw: Tensor, cls_logits: Tensor) -> Tensor:
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

    def _xyxy_to_raw(self, bboxes: Tensor, img_metas: List[ImageMeta]) -> Tensor:
        x0 = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = meta.img_shape[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale
        return x0

    def _raw_to_xyxy(self, raw_bboxes: Tensor, img_metas: List[ImageMeta]) -> Tensor:
        bboxes = (
            (raw_bboxes.clamp(-self.snr_scale, self.snr_scale) / self.snr_scale) + 1
        ) / 2
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        for i, meta in enumerate(img_metas):
            h, w = meta.img_shape[:2]
            scale = bboxes.new_tensor([w, h, w, h])
            bboxes[i] *= scale
        return bboxes

    def _ddim_step(
        self, t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas: List[ImageMeta]
    ):
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

        should_renew = self.box_renewal and self.sampling_timesteps > 1
        if should_renew:
            x_raw_next = self._apply_box_renewal(x_raw_next, cls_logits)

        return self._raw_to_xyxy(x_raw_next, img_metas), x_raw_next

    def _post_process(
        self, ensemble_results, img_metas: List[ImageMeta], rescale: bool
    ) -> List[DetectionResult]:
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
                    final_bboxes, final_scores, final_labels, iou_threshold=self.nms_thr
                )
                final_scores = final_scores[keep]
                final_bboxes = final_bboxes[keep]
                final_labels = final_labels[keep]

            keep = final_scores > self.score_thr
            final_scores = final_scores[keep]
            final_bboxes = final_bboxes[keep]
            final_labels = final_labels[keep]

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

                if scale_factor.dim() == 1:
                    scale_factor = scale_factor.unsqueeze(0)

                final_bboxes /= scale_factor

            results_list.append(
                DetectionResult(
                    bboxes=final_bboxes, scores=final_scores, labels=final_labels
                )
            )

        return results_list
