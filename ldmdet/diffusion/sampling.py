"""扩散采样模块

包含 DDIM、Euler、Heun、DPM-Solver++ 等采样策略，
以及框更新 (box renewal)、PCSE 核型评分、CCBR 级联间更新和后处理逻辑。
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import batched_nms

from ldmdet.data.structures import DetectionResult, ImageMeta
from ldmdet.diffusion.noise_schedule import load_buffer
from ldmdet.diffusion.rectified_flow import (
    RFDPMSolverAdaptive,
    RFDPMSolverHybrid,
    RFDPMSolverMultistep,
    RFDPMSolverPerDim,
)
from ldmdet.diffusion.shts import build_shts_grid, build_shts_shifted_grid
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
        # SHTS params
        shts_alpha: float = 1.0,
        shts_sigma: float = 0.15,
        shts_shifted: bool = False,
        # 方向4: VGAR (Velocity-Guided Adaptive Renewal)
        velocity_guided_renewal: bool = False,
        # 方向 D: 自适应阶次 DPM-Solver++ (推理时改动, 无需重训练)
        # 仅当 solver_type='dpm_solver_pp_adaptive' 时生效
        adaptive_solver_mode: str = 'static',
        adaptive_num_3rd_steps: int = 2,
        adaptive_eta_3rd_threshold: float = 0.5,
        # P0: 自适应阈值 box_renewal (移植自 DiffuDETR)
        # 启用后, renewal 阈值随时间步递减: 早期高阈值(积极淘汰), 后期低阈值(保守保留)
        adaptive_renewal_threshold: bool = False,
        adaptive_renewal_scale: float = 0.9,
        # 分维度 D1 调制: cxcywh 空间中 [cx, cy, w, h] 的 D1 保留掩码
        # True=保留 DPM++ D1 校正, False=置零退化为 Euler
        # 典型值: [True, True, False, False] — cx/cy 保留二阶校正, w/h 退为一阶
        dim_d1_mask: Optional[list[bool]] = None,
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
        # SHTS params
        self.shts_alpha = shts_alpha
        self.shts_sigma = shts_sigma
        self.shts_shifted = shts_shifted
        # 方向4: VGAR — box_renewal × RF 速度场耦合
        # 启用后, renewal 不再纯随机, 而是保留部分 v_θ 预测的 x0 方向
        self.velocity_guided_renewal = velocity_guided_renewal
        # 方向 D: 自适应阶次 DPM-Solver++ 参数
        self.adaptive_solver_mode = adaptive_solver_mode
        self.adaptive_num_3rd_steps = adaptive_num_3rd_steps
        self.adaptive_eta_3rd_threshold = adaptive_eta_3rd_threshold
        # P0: 自适应阈值 box_renewal
        # threshold(t) = max(t_curr * scale, score_thr)
        # t_curr=1.0(早期) → threshold≈0.9, t_curr=0.0(后期) → threshold=score_thr
        self.adaptive_renewal_threshold = adaptive_renewal_threshold
        self.adaptive_renewal_scale = adaptive_renewal_scale
        # 分维度 D1 调制: 转为 tensor
        if dim_d1_mask is not None:
            self.dim_d1_mask = torch.tensor(dim_d1_mask, dtype=torch.float32)
        else:
            self.dim_d1_mask = None

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

        # SHTS schedule: 基于 SNR 导数的自适应网格
        if self.rf_schedule == 'shts':
            t_grid = self._build_shts_time_grid()
            times = torch.tensor(t_grid, device=device)
        else:
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

    def _build_shts_time_grid(self) -> List[float]:
        """构建 SHTS 时间步网格"""
        # 方向 D: dpm_solver_pp_adaptive 也按 3 阶准备网格 (允许最大阶次)
        # 方向 A: dpm_solver_pp_per_dim 按 2 阶准备网格
        if self.solver_type in ('dpm_solver_pp', 'heun', 'dpm_solver_pp_per_dim', 'dpm_solver_pp_per_dim_w', 'dpm_pp_heun_hybrid'):
            solver_order = 2
        elif self.solver_type in ('dpm_solver_pp_3', 'dpm_solver_pp_adaptive'):
            solver_order = 3
        else:
            solver_order = 1
        if self.shts_shifted:
            return build_shts_shifted_grid(
                self.sampling_timesteps, self.snr_scale,
                rf_shift=self.rf_shift,
                task_weight_alpha=self.shts_alpha,
                task_weight_sigma=self.shts_sigma,
                solver_order=solver_order,
            )
        return build_shts_grid(
            self.sampling_timesteps, self.snr_scale,
            task_weight_alpha=self.shts_alpha,
            task_weight_sigma=self.shts_sigma,
            solver_order=solver_order,
        )

    def create_dpm_solver(self) -> Optional[RFDPMSolverMultistep]:
        """创建 DPM-Solver++ 实例，支持 SHTS 网格传递"""
        if self.solver_type in ('dpm_solver_pp', 'dpm_solver_pp_3'):
            solver_order = 3 if self.solver_type == 'dpm_solver_pp_3' else 2
            # SHTS: 生成非均匀网格传递给 DPM-Solver++
            # 注意: dim_d1_mask 在 SHTS/非-SHTS 两个分支都要传递,
            # 否则 SHTS 路径下分维度 D1 调制会静默失效 (2026-07-29 修复)
            if self.rf_schedule == 'shts':
                t_grid = self._build_shts_time_grid()
                return RFDPMSolverMultistep(
                    num_steps=self.sampling_timesteps,
                    solver_order=solver_order,
                    timesteps=t_grid,
                    dim_d1_mask=self.dim_d1_mask,
                )
            return RFDPMSolverMultistep(
                num_steps=self.sampling_timesteps,
                solver_order=solver_order,
                dim_d1_mask=self.dim_d1_mask,
            )
        # 方向 D: 自适应阶次 solver (推理时改动, 无需重训练)
        if self.solver_type == 'dpm_solver_pp_adaptive':
            if self.rf_schedule == 'shts':
                t_grid = self._build_shts_time_grid()
                return RFDPMSolverAdaptive(
                    num_steps=self.sampling_timesteps,
                    timesteps=t_grid,
                    adaptive_mode=self.adaptive_solver_mode,
                    num_3rd_steps=self.adaptive_num_3rd_steps,
                    eta_3rd_threshold=self.adaptive_eta_3rd_threshold,
                )
            return RFDPMSolverAdaptive(
                num_steps=self.sampling_timesteps,
                adaptive_mode=self.adaptive_solver_mode,
                num_3rd_steps=self.adaptive_num_3rd_steps,
                eta_3rd_threshold=self.adaptive_eta_3rd_threshold,
            )
        # 方向 A Phase 2: per-dim 阶数分配 solver (推理时改动, 无需重训练)
        # h 维度 (index 3) 用 1 阶 Euler, cx/cy/w 维度 (index 0/1/2) 用 2 阶 DPM-Solver++
        if self.solver_type == 'dpm_solver_pp_per_dim':
            if self.rf_schedule == 'shts':
                t_grid = self._build_shts_time_grid()
                return RFDPMSolverPerDim(
                    num_steps=self.sampling_timesteps,
                    timesteps=t_grid,
                )
            return RFDPMSolverPerDim(
                num_steps=self.sampling_timesteps,
            )
        # 方向 A Phase 2 扩展 (A.2): w,h 维度均用 1 阶, 仅 cx/cy 用 2 阶
        # 基于 Phase 2 per-dim eta_str 诊断: w 维度 eta_str (0.5-1.0) 与 h (0.4-0.9) 接近
        if self.solver_type == 'dpm_solver_pp_per_dim_w':
            if self.rf_schedule == 'shts':
                t_grid = self._build_shts_time_grid()
                return RFDPMSolverPerDim(
                    num_steps=self.sampling_timesteps,
                    timesteps=t_grid,
                    euler_dims=(2, 3),    # w, h 维度用 1 阶
                    dpm_dims=(0, 1),      # cx, cy 维度用 2 阶
                )
            return RFDPMSolverPerDim(
                num_steps=self.sampling_timesteps,
                euler_dims=(2, 3),    # w, h 维度用 1 阶
                dpm_dims=(0, 1),      # cx, cy 维度用 2 阶
            )
        # 混合求解器: cx/cy 用 DPM-Solver++ 2阶 (x0 插值), w/h 用 Heun 2阶 (速度梯形)
        # Heun 校正项需额外 NFE; model_fn 由 predict() 在 step 循环前注入
        if self.solver_type == 'dpm_pp_heun_hybrid':
            if self.rf_schedule == 'shts':
                t_grid = self._build_shts_time_grid()
                return RFDPMSolverHybrid(
                    num_steps=self.sampling_timesteps,
                    timesteps=t_grid,
                )
            return RFDPMSolverHybrid(
                num_steps=self.sampling_timesteps,
            )
        return None

    def _compute_renewal_threshold(self, t_curr: Optional[float]) -> float:
        """计算 box_renewal 的置信度阈值。

        P0 自适应阈值 (移植自 DiffuDETR):
        - 启用 adaptive_renewal_threshold 且 t_curr 可用时, 阈值随时间步递减:
          threshold = max(t_curr * adaptive_renewal_scale, score_thr)
        - t_curr=1.0 (早期, 纯噪声) → 高阈值 (积极淘汰低分框)
        - t_curr=0.0 (后期, 接近真实) → score_thr (保守保留)
        - 禁用或 t_curr 不可用时, 回退到固定 score_thr (向后兼容)

        Args:
            t_curr: 当前归一化时间步 [0, 1], None 表示不可用

        Returns:
            threshold: 置信度阈值
        """
        if self.adaptive_renewal_threshold and t_curr is not None:
            return max(t_curr * self.adaptive_renewal_scale, self.score_thr)
        return self.score_thr

    def apply_box_renewal(
        self,
        x_raw: Tensor,
        cls_logits: Tensor,
        x0_pred: Tensor = None,
        t_curr: float = None,
    ) -> Tensor:
        """框更新：低置信度框替换为随机噪声或速度场引导的噪声

        方向4 VGAR: 当 x0_pred 和 t_curr 提供且 velocity_guided_renewal=True 时,
        renewal 不完全重置为纯随机噪声, 而是保留部分 v_θ 预测的 x0 方向:
            x_renewed = alpha(t) * x0_pred + (1 - alpha(t)) * randn
        其中 alpha(t) 随时间步自适应 (早期更随机, 后期更确定)。

        P0 自适应阈值: 当 adaptive_renewal_threshold=True 且 t_curr 可用时,
        阈值随时间步递减 (早期高阈值积极淘汰, 后期低阈值保守保留)。

        Args:
            x_raw: [bs, N, 4] 扩散空间框
            cls_logits: [bs, N, num_classes] 分类 logits
            x0_pred: [bs, N, 4] v_θ 预测的 x0 (可选, 不传则纯随机)
            t_curr: 当前归一化时间步 [0, 1] (可选, 不传则纯随机)
        """
        bs, device = x_raw.shape[0], x_raw.device
        scores = torch.sigmoid(cls_logits).max(-1)[0]
        x_raw_new = x_raw.clone()

        # P0: 计算自适应阈值
        threshold = self._compute_renewal_threshold(t_curr)

        # 方向4: 计算时间自适应 alpha
        use_vgar = (
            self.velocity_guided_renewal
            and x0_pred is not None
            and t_curr is not None
        )
        if use_vgar:
            # alpha(t) = 0.2 + 0.6 * sigmoid(5 * (0.5 - t))
            # t 大 (早期) → alpha 小 → 更随机 (探索)
            # t 小 (后期) → alpha 大 → 更确定 (利用 x0_pred)
            alpha = 0.2 + 0.6 * (
                1.0 / (1.0 + math.exp(5.0 * (t_curr - 0.5)))
            )
        else:
            alpha = 0.0  # 纯随机 renewal (向后兼容)

        for i in range(bs):
            keep = scores[i] > threshold
            if keep.sum() < self.min_keep:
                _, topk_idx = scores[i].topk(
                    min(self.min_keep, scores.shape[1])
                )
                keep[topk_idx] = True

            num_renew = (~keep).sum()
            if num_renew > 0:
                noise = torch.randn(num_renew, 4, device=device)
                if use_vgar:
                    # VGAR: alpha * x0_pred + (1 - alpha) * noise
                    # 利用 v_θ 预测的 x0 方向, 同时保持随机扰动 (探索)
                    x0_renew = x0_pred[i, ~keep]
                    x_raw_new[i, ~keep] = (
                        alpha * x0_renew + (1.0 - alpha) * noise
                    )
                else:
                    # 原始: 纯随机 renewal
                    x_raw_new[i, ~keep] = noise
        return x_raw_new

    # ================================================================
    # CCBR: 级联间软 renewal (跨级联框更新)
    # ================================================================

    def apply_inter_head_renewal(
        self,
        pred_bboxes: Tensor,      # [bs, N, 4] xyxy, Head k 的预测
        fused_conf: Tensor,       # [bs, N] 融合置信度
        threshold: float,         # renewal 阈值
        alpha: float = 0.7,       # 软 renewal 保留率
        sigma_scale: float = 0.1, # 扰动幅度 (框对角线比例)
    ) -> Tensor:
        """级联间软 box_renewal

        与 apply_box_renewal 的区别:
        1. 作用在 xyxy 空间 (级联间), 非 raw 空间 (时间步间)
        2. 软 renewal (alpha * pred + (1-alpha) * perturbed), 非纯噪声
        3. 用融合置信度 (本级 + 后向传播), 非仅最后一级
        """
        bs, N, _ = pred_bboxes.shape
        device = pred_bboxes.device

        # 框对角线长度作为扰动尺度
        diag = torch.sqrt(
            (pred_bboxes[..., 2] - pred_bboxes[..., 0]).clamp(min=1e-6) ** 2
            + (pred_bboxes[..., 3] - pred_bboxes[..., 1]).clamp(min=1e-6) ** 2
        )  # [bs, N]

        # 决策: 保留 or 软 renewal
        keep = fused_conf > threshold  # [bs, N]
        for i in range(bs):
            if keep[i].sum() < self.min_keep:
                _, topk_idx = fused_conf[i].topk(
                    min(self.min_keep, N)
                )
                keep[i, topk_idx] = True

        # 软 renewal: alpha * pred + (1-alpha) * (pred + sigma * noise)
        noise = torch.randn(bs, N, 4, device=device)
        sigma = sigma_scale * diag.unsqueeze(-1)  # [bs, N, 1]
        perturbed = pred_bboxes + sigma * noise
        renewed = alpha * pred_bboxes + (1 - alpha) * perturbed

        result = torch.where(keep.unsqueeze(-1), pred_bboxes, renewed)
        return result

    def apply_topk_pruning(
        self,
        x_raw: Tensor,
        cls_logits: Tensor,
        pred_bboxes: Tensor,
        x0_raw: Tensor,
        k: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Top-K 框剪枝：保留每张图置信度最高的 K 个框。

        数学理论:
            在扩散采样第 1 步后，分类置信度已具判别力。
            保留 Top-K 高置信框，丢弃其余，使后续步的
            RoIAlign (O(N)) 和 DynamicConv (O(N)) 计算量线性降低。

        Args:
            x_raw: [bs, N, 4] 扩散空间框
            cls_logits: [bs, N, num_classes] 分类 logits
            pred_bboxes: [bs, N, 4] 图像空间框
            x0_raw: [bs, N, 4] 扩散空间 x0 预测
            k: 保留的框数

        Returns:
            (x_raw_pruned, cls_logits_pruned, pred_bboxes_pruned,
             x0_raw_pruned, topk_indices)
             所有张量 N 维 → K 维，topk_indices: [bs, K]
        """
        bs, n = x_raw.shape[:2]
        if k >= n:
            idx = torch.arange(n, device=x_raw.device).unsqueeze(0).expand(bs, -1)
            return x_raw, cls_logits, pred_bboxes, x0_raw, idx

        k = min(k, n)
        scores = torch.sigmoid(cls_logits).max(dim=-1)[0]  # [bs, N]
        topk_idx = scores.topk(k, dim=1).indices  # [bs, K]

        idx_4d = topk_idx.unsqueeze(-1).expand(-1, -1, 4)  # [bs, K, 4]
        idx_cls = topk_idx.unsqueeze(-1).expand(
            -1, -1, cls_logits.shape[-1]
        )  # [bs, K, C]

        x_raw_pruned = x_raw.gather(1, idx_4d)
        cls_logits_pruned = cls_logits.gather(1, idx_cls)
        pred_bboxes_pruned = pred_bboxes.gather(1, idx_4d)
        x0_raw_pruned = x0_raw.gather(1, idx_4d)

        return (
            x_raw_pruned,
            cls_logits_pruned,
            pred_bboxes_pruned,
            x0_raw_pruned,
            topk_idx,
        )

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
        """一步 DDIM 采样 (DDPM 基线)"""
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
            # P0: DDIM 路径传递归一化 t_curr 用于自适应阈值
            # t_curr 是整数索引 [0, timesteps-1], 归一化到 [0, 1]
            t_curr_norm = t_curr / max(self.timesteps, 1)
            x_raw_next = self.apply_box_renewal(
                x_raw_next, cls_logits, t_curr=t_curr_norm
            )

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


# ================================================================
# PCSE: 配对一致性随机集成评分 (KaryotypeScorer)
# ================================================================


class KaryotypeScorer:
    """核型配对一致性评分器 (PCSE 核心)

    利用核型先验 (数量、配对、形态、性染色体) 对检测假设评分，
    从 K 个候选假设中选择最合法的输出。
    """

    def __init__(
        self,
        num_autosome: int = 22,
        x_class_idx: int = 22,
        y_class_idx: int = 23,
        target_count: int = 46,
        count_sigma: float = 2.0,
        alphas: Tuple[float, float, float, float] = (1.0, 2.0, 1.5, 1.5),
        lam: float = 0.5,
    ):
        self.num_autosome = num_autosome
        self.x_class_idx = x_class_idx
        self.y_class_idx = y_class_idx
        self.target_count = target_count
        self.count_sigma = count_sigma
        self.alphas = alphas  # (alpha_count, beta_pair, gamma_morph, delta_sex)
        self.lam = lam

    def score(self, bboxes: Tensor, scores: Tensor, labels: Tensor) -> float:
        """计算单个假设的综合评分

        Args:
            bboxes: [N, 4] xyxy 框坐标
            scores: [N] 检测置信度
            labels: [N] 类别标签

        Returns:
            float: 综合评分 (0~1), 越高越好
        """
        s_count = self._score_count(labels)
        s_pair = self._score_pair(labels)
        s_morph = self._score_morph(bboxes, labels)
        s_sex = self._score_sex(labels)
        s_karyo = (
            self.alphas[0] * s_count
            + self.alphas[1] * s_pair
            + self.alphas[2] * s_morph
            + self.alphas[3] * s_sex
        )
        s_karyo /= sum(self.alphas)  # 归一化到 [0, 1]
        s_conf = scores.mean().item() if len(scores) > 0 else 0.0
        return self.lam * s_conf + (1.0 - self.lam) * s_karyo

    def _score_count(self, labels: Tensor) -> float:
        """数量一致性评分: 染色体总数应接近 target_count"""
        n = len(labels)
        return math.exp(-((n - self.target_count) ** 2) / (2 * self.count_sigma ** 2))

    def _score_pair(self, labels: Tensor) -> float:
        """类别配对一致性评分: 每个常染色体类应恰好出现 2 次"""
        counts = torch.bincount(labels, minlength=self.num_autosome + 2)
        total = 0.0
        for k in range(self.num_autosome):
            n = counts[k].item()
            if n == 2:
                total += 1.0
            elif n in (1, 3):
                total += 0.5
        return total / self.num_autosome

    def _score_morph(self, bboxes: Tensor, labels: Tensor) -> float:
        """形态配对一致性评分: 同源染色体对大小、长宽比相似"""
        total_sim = 0.0
        num_pairs = 0
        for k in range(self.num_autosome):
            idx = (labels == k).nonzero(as_tuple=True)[0]
            if len(idx) != 2:
                continue
            b1, b2 = bboxes[idx[0]], bboxes[idx[1]]
            w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
            w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
            A1, A2 = w1 * h1, w2 * h2
            r1 = max(w1, h1) / max(min(w1, h1), 1e-6)
            r2 = max(w2, h2) / max(min(w2, h2), 1e-6)
            sim_size = 1.0 - abs(A1 - A2) / max(A1, A2, 1e-6)
            sim_aspect = 1.0 - abs(r1 - r2) / max(r1, r2, 1e-6)
            total_sim += 0.5 * (sim_size + sim_aspect)
            num_pairs += 1
        return total_sim / max(num_pairs, 1)

    def _score_sex(self, labels: Tensor) -> float:
        """性染色体一致性评分: 应为 XX (女) 或 XY (男)"""
        n_x = (labels == self.x_class_idx).sum().item()
        n_y = (labels == self.y_class_idx).sum().item()
        if (n_x, n_y) in [(2, 0), (1, 1)]:
            return 1.0
        elif (n_x, n_y) in [(1, 0), (2, 1), (3, 0)]:
            return 0.5
        return 0.0


def pcse_select(
    hypotheses: List[DetectionResult],
    scorer: KaryotypeScorer,
) -> DetectionResult:
    """PCSE 选择: 从 K 个假设中选择核型一致性最优的

    Args:
        hypotheses: K 个检测假设
        scorer: 核型评分器

    Returns:
        评分最高的假设
    """
    best_score = -float('inf')
    best_hypothesis = hypotheses[0]
    for hyp in hypotheses:
        s_total = scorer.score(hyp.bboxes, hyp.scores, hyp.labels)
        if s_total > best_score:
            best_score = s_total
            best_hypothesis = hyp
    return best_hypothesis


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
