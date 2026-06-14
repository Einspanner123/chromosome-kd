"""扩散采样模块

包含 DDIM、Euler、Heun、DPM-Solver++ 等采样策略，
以及框更新 (box renewal) 和后处理逻辑。
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.ops import batched_nms

from ldmdet.data.structures import DetectionResult, ImageMeta
from ldmdet.diffusion.noise_schedule import load_buffer
from ldmdet.diffusion.rectified_flow import RFDPMSolverMultistep
from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh


class DiffusionSampler:
    """扩散采样器，封装迭代去噪和后处理逻辑。

    注意：此类不持有 nn.Parameter，仅作为逻辑组织单元。
    """

    def __init__(
        self,
        diffusion_type: str,
        timesteps: int,
        sampling_timesteps: int,
        solver_type: str,
        ddim_sampling_eta: float,
        rf_schedule: str,
        rf_power: float,
        rf_shift: float,
        snr_scale: float,
        box_renewal: bool,
        use_ensemble: bool,
        use_nms: bool,
        nms_thr: float,
        score_thr: float,
        min_keep: int,
    ):
        self.diffusion_type = diffusion_type
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.solver_type = solver_type
        self.ddim_sampling_eta = ddim_sampling_eta
        self.rf_schedule = rf_schedule
        self.rf_power = rf_power
        self.rf_shift = rf_shift
        self.snr_scale = snr_scale
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self.use_nms = use_nms
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.min_keep = min_keep

    def build_time_pairs(
        self, device: torch.device
    ) -> List[Tuple[float, float]]:
        """构建采样时间序列"""
        if self.diffusion_type == 'ddpm':
            times = torch.linspace(
                -1,
                self.timesteps - 1,
                steps=self.sampling_timesteps + 1,
                device=device,
            )
            times = list(reversed(times.int().tolist()))
            return list(zip(times[:-1], times[1:]))

        times = torch.linspace(
            1.0, 0.0, steps=self.sampling_timesteps + 1, device=device
        )
        if self.rf_schedule == 'power':
            times = times.pow(self.rf_power)
        elif self.rf_schedule == 'shifted':
            s = self.rf_shift
            times = s * times / (1 + (s - 1) * times)

        time_pairs = []
        for i in range(len(times) - 1):
            time_pairs.append((times[i].item(), times[i + 1].item()))
        return time_pairs

    def create_dpm_solver(self) -> Optional[RFDPMSolverMultistep]:
        """创建 DPM-Solver++ 实例"""
        if self.solver_type in ('dpm_solver_pp', 'dpm_solver_pp_3'):
            solver_order = 3 if self.solver_type == 'dpm_solver_pp_3' else 2
            return RFDPMSolverMultistep(
                num_steps=self.sampling_timesteps, solver_order=solver_order
            )
        return None

    def apply_box_renewal(
        self, x_raw: Tensor, cls_logits: Tensor
    ) -> Tensor:
        """框更新：低置信度框替换为随机噪声"""
        bs, device = x_raw.shape[0], x_raw.device
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        x_raw_new = x_raw.clone()

        for i in range(bs):
            keep = scores[i] > self.score_thr
            if keep.sum() < self.min_keep:
                _, topk_idx = scores[i].topk(
                    min(self.min_keep, scores.shape[1])
                )
                keep[topk_idx] = True

            num_renew = (~keep).sum()
            if num_renew > 0:
                x_raw_new[i, ~keep] = torch.randn(
                    num_renew, 4, device=device
                )
        return x_raw_new

    def xyxy_to_raw(
        self, bboxes: Tensor, img_metas: List[ImageMeta]
    ) -> Tensor:
        """图像空间 xyxy → 扩散空间 raw"""
        x0 = bboxes.clone()
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            scale = x0.new_tensor([w, h, w, h])
            x0[i] /= scale
        x0 = bbox_xyxy_to_cxcywh(x0)
        x0 = (x0 * 2 - 1) * self.snr_scale
        return x0

    def raw_to_xyxy(
        self, raw_bboxes: Tensor, img_metas: List[ImageMeta]
    ) -> Tensor:
        """扩散空间 raw → 图像空间 xyxy"""
        bboxes = (
            (
                raw_bboxes.clamp(-self.snr_scale, self.snr_scale)
                / self.snr_scale
            )
            + 1
        ) / 2
        bboxes = bbox_cxcywh_to_xyxy(bboxes)
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            scale = bboxes.new_tensor([w, h, w, h])
            bboxes[i] *= scale
        return bboxes

    def ddim_step(
        self,
        t_curr: int,
        t_next: int,
        x_raw: Tensor,
        cls_logits: Tensor,
        pred_bboxes: Tensor,
        img_metas: List[ImageMeta],
        alphas_cumprod: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """一步 DDIM 采样 (DEPRECATED: DDPM)"""
        bs, device = x_raw.shape[0], x_raw.device

        x0 = self.xyxy_to_raw(pred_bboxes, img_metas)
        if t_next < 0:
            return self.raw_to_xyxy(x0, img_metas), x0

        t_batch = torch.full(
            (bs,), t_curr, device=device, dtype=torch.long
        )
        pred_noise = predict_noise_from_start(
            x_raw, t_batch, x0, alphas_cumprod
        )

        alpha = alphas_cumprod[t_curr]
        alpha_next = alphas_cumprod[t_next]
        sigma = (
            self.ddim_sampling_eta
            * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        noise = torch.randn_like(x_raw)
        x_raw_next = (
            x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise
        )

        if self.box_renewal:
            x_raw_next = self.apply_box_renewal(x_raw_next, cls_logits)

        return self.raw_to_xyxy(x_raw_next, img_metas), x_raw_next

    def post_process(
        self,
        ensemble_results: List[Tuple[Tensor, Tensor]],
        img_metas: List[ImageMeta],
        rescale: bool,
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
                    final_bboxes,
                    final_scores,
                    final_labels,
                    self.nms_thr,
                )
                final_scores = final_scores[keep]
                final_bboxes = final_bboxes[keep]
                final_labels = final_labels[keep]

            if rescale:
                scale_factor = _get_scale_factor(img_metas[i])
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
                    bboxes=final_bboxes,
                    scores=final_scores,
                    labels=final_labels,
                )
            )

        return results_list


def predict_noise_from_start(
    x_t: Tensor, t: Tensor, x0: Tensor, alphas_cumprod: Tensor
) -> Tensor:
    """从预测的 x0 还原噪声 (DDPM)"""
    sqrt_recip_alphas_cumprod_t = load_buffer(
        1.0 / alphas_cumprod.sqrt(), t, x_t.shape
    )
    sqrt_recipm1_alphas_cumprod_t = load_buffer(
        (1.0 / alphas_cumprod - 1).sqrt(), t, x_t.shape
    )
    return (
        sqrt_recip_alphas_cumprod_t * x_t - x0
    ) / sqrt_recipm1_alphas_cumprod_t


def _get_img_shape(meta):
    """兼容 dict 和 ImageMeta 的 img_shape 获取"""
    if isinstance(meta, dict):
        return meta['img_shape']
    return meta.img_shape


def _get_scale_factor(meta):
    """兼容 dict 和 ImageMeta 的 scale_factor 获取"""
    if isinstance(meta, dict):
        return meta.get('scale_factor')
    return meta.scale_factor
