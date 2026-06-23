"""DiffusionDetHead — 核心迭代去噪检测头

直接移植自 LDMDet/mods/diffusiondet_head.py (已验证工作),
仅更新 import 路径到 ldmdet 纯 PyTorch 库。
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ldmdet.coupling import build_coupling
from ldmdet.data.structures import DetectionResult, ImageMeta, InstanceData, ModelOutput
from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from ldmdet.diffusion.noise_schedule import cosine_noise_schedule
from ldmdet.diffusion.rectified_flow import RectifiedFlow
from ldmdet.diffusion.sampling import DiffusionSampler, _get_img_shape
from ldmdet.utils.box_ops import bbox_xyxy_to_cxcywh


class DiffusionDetHead(nn.Module):
    """扩散检测头。支持 DDPM 和 Rectified Flow，Euler/Heun/DPM-Solver++。"""

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
        coupling: Optional[nn.Module] = None,
        pre_noise_layer: int = 2,
        loss_aux: Optional[Dict] = None,
        torch_compile: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        # 方向二: 计数先验约束 (默认全关, 不影响 baseline)
        use_count_constraint: bool = False,
        default_target_count: int = 46,
        count_constraint_iou_threshold: float = 0.5,
        count_constraint_min_keep: int = 10,
        count_loss_weight: float = 1.0,
        # 方向四: 非线性轨迹 (默认 None, 不影响 baseline)
        # scale_conditioned_rf: ScaleConditionedRF 实例, 若提供则替换标准 RF
        scale_conditioned_rf: Optional[object] = None,
        # 方向五: 分层分类 (默认 None, 不影响 baseline)
        # hierarchical_head: HierarchicalClsHead 实例, 若提供则替换 cls_head
        hierarchical_head: Optional[nn.Module] = None,
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
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self.solver_type = solver_type

        self.loss_aux = loss_aux
        self.counting_branch = counting_branch
        self.consistency_loss = consistency_loss

        # 方向二: 计数先验约束参数
        self.use_count_constraint = use_count_constraint
        self.default_target_count = default_target_count
        self.count_constraint_iou_threshold = count_constraint_iou_threshold
        self.count_constraint_min_keep = count_constraint_min_keep
        self.count_loss_weight = count_loss_weight

        # 方向一诊断: 耦合诊断回调 (可选, 默认 None, 不影响 baseline)
        # 由 TrainingDiagnosticsHook 通过 diagnostics_callback 注入, 或手动设置
        self.coupling_diag_callback = None

        # 方向二诊断: 计数诊断回调 (可选, 默认 None, 不影响 baseline)
        self.count_diag_callback = None

        # 方向四诊断: 轨迹诊断回调 (可选, 默认 None, 不影响 baseline)
        self.trajectory_diag_callback = None

        # 方向五诊断: 分层分类诊断回调 (可选, 默认 None, 不影响 baseline)
        self.hierarchical_diag_callback = None

        # 方向四: 尺度条件化 RF (可选, 默认 None, 不影响 baseline)
        self.scale_conditioned_rf = scale_conditioned_rf

        # 方向五: 分层分类头 (可选, 默认 None, 不影响 baseline)
        self.hierarchical_head = hierarchical_head

        # 扩散组件
        self.rf = RectifiedFlow(snr_scale=snr_scale)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, feat_channels * 4),
            nn.SiLU(),
            nn.Linear(feat_channels * 4, feat_channels * 4),
        )

        # DDPM 缓存
        if diffusion_type == 'ddpm':
            betas = cosine_noise_schedule(timesteps).float()
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer('alphas_cumprod', alphas_cumprod)
        else:
            self.alphas_cumprod = None

        # 级联 Head
        self.head_series = nn.ModuleList([copy.deepcopy(single_head) for _ in range(num_heads)])
        self.roi_extractor = roi_extractor
        self.criterion = criterion
        self.pre_noise_layer = pre_noise_layer

        # 耦合策略
        self.ot_coupling = coupling is not None and not isinstance(coupling, bool)
        self.ot_module = coupling if coupling is not None else build_coupling('random')

        # 采样器
        self._sampler = DiffusionSampler(
            diffusion_type=diffusion_type, timesteps=timesteps,
            sampling_timesteps=sampling_timesteps, solver_type=solver_type,
            ddim_sampling_eta=ddim_sampling_eta, rf_schedule=rf_schedule,
            rf_power=rf_power, rf_shift=rf_shift, snr_scale=snr_scale,
            box_renewal=box_renewal, use_ensemble=use_ensemble,
            use_nms=use_nms, nms_thr=nms_thr, score_thr=score_thr, min_keep=min_keep,
        )

        self._init_weights(prior_prob)

        # AMP: 仅模型前向使用半精度，criterion 始终 FP32
        # 推荐值: torch.bfloat16 (同动态范围，无需 GradScaler)
        self.amp_dtype = amp_dtype

        if torch_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward, dynamic=True)

    def _init_weights(self, prior_prob):
        for head in self.head_series:
            if hasattr(head, 'cls_head'):
                bias_value = -(math.log((1 - prior_prob) / prior_prob))
                nn.init.constant_(head.cls_head[-1].bias, bias_value)

    # ================================================================
    # 前向传播
    # ================================================================

    def forward(self, features, bboxes, t):
        time_emb = self.time_mlp(t)
        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_curr_proposals = []
        curr_bboxes = bboxes
        curr_proposals = None

        for head in self.head_series:
            result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
            if len(result) == 4:
                cls_logits, pred_bboxes, curr_proposals, _ = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_curr_proposals.append(curr_proposals)
            curr_bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            return torch.stack(inter_cls_logits), torch.stack(inter_pred_bboxes), inter_curr_proposals
        return torch.stack(inter_cls_logits[-1:]), torch.stack(inter_pred_bboxes[-1:]), inter_curr_proposals[-1:]

    # ================================================================
    # 训练损失
    # ================================================================

    def loss(self, features, img_metas, gt_bboxes, gt_labels):
        device = features[0].device
        bs = len(img_metas)

        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
        t = self._sample_t(bs, device)
        x_boxes, x_starts, x_noises, matched_gt_indices = self._build_training_targets(
            bs, device, t, targets, gt_bboxes
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps
        # 模型前向：若启用 AMP，在 autocast 下执行（线性层/attention 用半精度加速）
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                all_cls_logits, all_pred_bboxes, all_curr_proposals = self(features, curr_bboxes, t_input)
            # autocast 输出可能为半精度，criterion 需 FP32（如 cdist 不支持 BF16）
            all_cls_logits = all_cls_logits.float()
            all_pred_bboxes = all_pred_bboxes.float()
        else:
            all_cls_logits, all_pred_bboxes, all_curr_proposals = self(features, curr_bboxes, t_input)

        norm_pred_bboxes = self._normalize_pred_bboxes(all_pred_bboxes, img_metas)
        outputs = self._build_outputs(all_cls_logits, norm_pred_bboxes)
        # 方向三: 传 t 给 criterion (若 criterion 不支持 t 则被忽略, 向后兼容)
        # t 是 [bs] 的扩散时间, 用于 SNR 感知匹配和损失加权
        losses = self.criterion(outputs, targets, t=t)

        # 方向二 路径 C: 计数分支训练 (开关控制, 默认不启用)
        if self.counting_branch is not None:
            gt_count = torch.tensor(
                [gt_bboxes[i].shape[0] for i in range(bs)],
                device=device, dtype=torch.long,
            )
            count_logits, _ = self.counting_branch(features)
            count_loss = self.counting_branch.compute_loss(count_logits, gt_count)
            losses['loss_count'] = count_loss * self.count_loss_weight

            # 方向二诊断: 更新计数预测统计 (若回调已注入)
            if self.count_diag_callback is not None:
                pred_count = count_logits.argmax(dim=-1)
                self.count_diag_callback.update(pred_count, gt_count)

        # 方向一诊断: 更新耦合统计 (若回调已注入)
        if self.coupling_diag_callback is not None:
            all_matched = torch.cat(matched_gt_indices)  # [bs * num_proposals]
            total_gt = sum(t.labels.shape[0] for t in targets)
            self.coupling_diag_callback.update(all_matched, total_gt)

        # 方向四诊断: 更新轨迹统计 (若回调已注入)
        if self.trajectory_diag_callback is not None:
            # 尺度条件化统计: 从 GT 框计算 scales, 从 t 计算 t_eff
            if self.scale_conditioned_rf is not None:
                # 收集所有 GT 框的尺度
                all_gt_boxes = []
                for tgt in targets:
                    if tgt.bboxes.shape[0] > 0:
                        # bboxes 是 cxcywh 归一化坐标
                        wh = tgt.bboxes[:, 2:]
                        scales = (wh.prod(dim=-1) ** 0.5).clamp_min(1e-6)
                        all_gt_boxes.append(scales)
                if all_gt_boxes:
                    all_scales = torch.cat(all_gt_boxes)
                    # 取 t 的均值作为标量 (RF 采样同 batch 同 t)
                    t_scalar = t.mean().unsqueeze(0).expand_as(all_scales)
                    t_eff = self.scale_conditioned_rf.compute_t_eff(
                        t_scalar, all_scales
                    )
                    kappa = self.scale_conditioned_rf.compute_kappa(all_scales)
                    self.trajectory_diag_callback.update_scale(
                        all_scales, t_scalar, t_eff, kappa,
                    )

            # OT 耦合统计: 若 ot_module 是 OTFlowCoupling, 重新计算 transport/cost
            ot_mod = getattr(self.ot_module, 'ot_module', None)
            if ot_mod is not None and hasattr(ot_mod, 'compute_coupling_cost'):
                # 从 x_starts 和 x_noises 重新计算 (取第一个样本)
                if x_starts[0].shape[0] > 1:
                    cost = self.ot_module.compute_coupling_cost(
                        x_starts[0], x_noises[0]
                    )
                    from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport
                    transport = sinkhorn_transport(
                        cost,
                        epsilon=self.ot_module.epsilon,
                        num_iters=self.ot_module.num_iters,
                    )
                    self.trajectory_diag_callback.update_ot(transport, cost)

        # 方向五: 分层分类辅助损失 + 诊断 (开关控制, 默认不启用)
        if self.hierarchical_head is not None:
            # 用最后一个 head 的 fc_feature 做分层分类
            # all_curr_proposals: list of [1, bs*num_boxes, feat_channels]
            fc_feature = all_curr_proposals[-1]  # [1, bs*num_boxes, C]
            fc_feature = fc_feature.squeeze(0)  # [bs*num_boxes, C]

            # 构建 targets: 每个 proposal 对应的 GT label
            # matched_gt_indices: list of [num_proposals], 指向 GT 索引
            # 背景proposal (matched_idx 指向不存在的 GT) 设为 -1
            hier_targets = []
            for i, matched_idx in enumerate(matched_gt_indices):
                gt_labels_i = targets[i].labels  # [num_gt_i]
                # 每个 proposal 的 label = gt_labels[matched_idx], 越界则 -1
                valid = matched_idx < gt_labels_i.shape[0]
                labels = torch.full_like(matched_idx, -1, dtype=torch.long)
                labels[valid] = gt_labels_i[matched_idx[valid]]
                hier_targets.append(labels)
            hier_targets = torch.cat(hier_targets)  # [bs*num_proposals]

            # 分层分类前向
            hier_out = self.hierarchical_head(fc_feature)
            # 计算辅助损失 (仅对有效 proposal, label >= 0)
            valid_mask = hier_targets >= 0
            if valid_mask.any():
                group_targets = self.hierarchical_head.group_of_class.to(device)[
                    hier_targets.clamp(min=0)
                ]
                hier_loss = self.hierarchical_head.compute_loss(
                    hier_out['group_logits'][valid_mask].unsqueeze(0),
                    [cl[valid_mask].unsqueeze(0) for cl in hier_out['class_logits_per_group']],
                    hier_out['flat_logits'][valid_mask].unsqueeze(0),
                    hier_targets[valid_mask].unsqueeze(0),
                    group_targets[valid_mask].unsqueeze(0),
                    valid_mask=valid_mask[valid_mask].unsqueeze(0),
                )
                losses['loss_hier'] = hier_loss['loss_total']

            # 方向五诊断: 更新分层分类统计 (若回调已注入)
            if self.hierarchical_diag_callback is not None and valid_mask.any():
                group_targets_all = self.hierarchical_head.group_of_class.to(device)[
                    hier_targets.clamp(min=0)
                ]
                self.hierarchical_diag_callback.update(
                    hier_out['group_logits'],
                    hier_out['class_logits_per_group'],
                    hier_out['flat_logits'],
                    hier_targets,
                    group_targets_all,
                    valid_mask=valid_mask,
                )

        return losses

    # ================================================================
    # 推理
    # ================================================================

    @torch.no_grad()
    def predict(self, features, img_metas, rescale=True, return_trajectory=False):
        device = features[0].device
        bs = len(img_metas)
        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        ensemble_results = []
        trajectory = []
        dpm_solver = self._sampler.create_dpm_solver()
        if dpm_solver is not None:
            dpm_solver.reset()

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(features, x_raw, t_curr, img_metas)
            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))
            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            # Always keep last result for non-ensemble (DDPM single-step)
            if not ensemble_results:
                ensemble_results.append((cls_logits, pred_bboxes))

            if self.diffusion_type == 'ddpm':
                curr_bboxes_xyxy, x_raw = self._sampler.ddim_step(
                    t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas, self.alphas_cumprod
                )
                if t_next < 0:
                    break
            else:
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx)
                elif self.solver_type == 'heun' and t_next > 0:
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = self._forward_at_t(features, x_tmp, t_tmp, img_metas)
                        return x0_tmp, None
                    x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                if self.box_renewal:
                    x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
                if t_next <= 0:
                    break

        results = self._sampler.post_process(ensemble_results, img_metas, rescale)

        # 方向二: 计数先验约束后处理 (开关控制, 默认不启用)
        # 路径 B (拉格朗日约束) + 路径 C (计数分支预测 count)
        if self.use_count_constraint:
            results = self._apply_count_constraint(features, results)

        if return_trajectory:
            return results, trajectory
        return results

    def _apply_count_constraint(self, features, results):
        """方向二: 对 post_process 结果应用计数约束 NMS.

        路径 C: 若 counting_branch 存在, 用其预测 count; 否则用 default_target_count.
        路径 B: 对每张图的结果重新做 count_constrained_nms.

        Args:
            features: FPN 特征 (用于 counting_branch 预测)
            results: List[DetectionResult] 原始 post_process 结果

        Returns:
            List[DetectionResult] 计数约束后的结果
        """
        from ldmdet.inference.count_constrained_nms import count_constrained_nms

        # 路径 C: 预测每张图的目标计数
        if self.counting_branch is not None:
            _, pred_count = self.counting_branch(features)
            target_counts = pred_count.tolist()
        else:
            target_counts = [self.default_target_count] * len(results)

        # 路径 B: 计数约束 NMS
        new_results = []
        for i, res in enumerate(results):
            if res.bboxes.numel() == 0:
                new_results.append(res)
                continue

            keep = count_constrained_nms(
                boxes=res.bboxes,
                scores=res.scores,
                labels=res.labels,
                target_count=target_counts[i],
                iou_threshold=self.count_constraint_iou_threshold,
                min_keep=self.count_constraint_min_keep,
            )
            new_results.append(DetectionResult(
                bboxes=res.bboxes[keep],
                scores=res.scores[keep],
                labels=res.labels[keep],
            ))
        return new_results

    # ================================================================
    # 训练辅助
    # ================================================================

    def _normalize_targets(self, gt_bboxes, gt_labels, img_metas, bs):
        targets = []
        for i in range(bs):
            h, w = _get_img_shape(img_metas[i])[:2]
            scale = gt_bboxes[i].new_tensor([w, h, w, h])
            targets.append(InstanceData(
                labels=gt_labels[i], bboxes=gt_bboxes[i] / scale, img_shape=(h, w)
            ))
        return targets

    def _sample_t(self, bs, device):
        if self.diffusion_type == 'ddpm':
            return torch.randint(0, self.timesteps, (bs,), device=device).long()
        t = torch.rand((bs,), device=device)
        if self.rf_schedule == 'shifted':
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        return t

    def _build_training_targets(self, bs, device, t, targets, gt_bboxes):
        x_boxes, x_starts, x_noises, matched_gt_indices = [], [], [], []
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt == 0:
                noise = torch.randn(self.num_proposals, 4, device=device)
                x_boxes.append(noise)
                x_starts.append(torch.zeros_like(noise))
                x_noises.append(noise)
                matched_gt_indices.append(torch.zeros(self.num_proposals, dtype=torch.long, device=device))
                continue
            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale
            noise = torch.randn(self.num_proposals, 4, device=device)
            x_start, matched_idx = self._couple_single_image(noise, gt_diffusion, targets[i].labels, device)
            matched_gt_indices.append(matched_idx)
            x_noisy, x_noise = self._forward_diffusion(x_start, noise, t[i:i+1])
            x_starts.append(x_start)
            x_noises.append(x_noise)
            x_boxes.append(x_noisy)
        return x_boxes, x_starts, x_noises, matched_gt_indices

    def _couple_single_image(self, noise, gt_diffusion, gt_labels, device):
        if self.ot_coupling and self.diffusion_type == 'rectified_flow':
            return self.ot_module.couple(noise, gt_diffusion, gt_labels, device)
        num_gt = gt_diffusion.shape[0]
        idx = torch.randint(0, num_gt, (self.num_proposals,), device=device)
        return gt_diffusion[idx], idx

    def _forward_diffusion(self, x_start, noise, t):
        if self.diffusion_type == 'ddpm':
            return self.q_sample(x_start, t), torch.zeros_like(x_start)
        x_noisy, _ = self.rf.q_sample(x_start, x_noise=noise, t=t)
        return x_noisy, noise

    def _forward_at_t(self, features, x_raw, t, img_metas):
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        cls_logits_seq, pred_bboxes_seq, _ = self(features, curr_bboxes, t_input)
        cls_logits_last = cls_logits_seq[-1]
        pred_bboxes_last = pred_bboxes_seq[-1]
        x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)
        return cls_logits_last, pred_bboxes_last, x0

    def _normalize_pred_bboxes(self, all_pred_bboxes, img_metas):
        # 构建 scale 张量并广播除法，消除逐 head 逐 image 的双重循环
        # all_pred_bboxes: [num_heads, bs, num_proposals, 4]
        num_heads = all_pred_bboxes.shape[0]
        bs = len(img_metas)
        # 构建 [bs, 4] 的 scale 张量
        scales = all_pred_bboxes.new_zeros(bs, 4)
        for i in range(bs):
            h, w = _get_img_shape(img_metas[i])[:2]
            scales[i] = all_pred_bboxes.new_tensor([w, h, w, h])
        # [num_heads, bs, 1, 4] / [1, bs, 1, 4] → [num_heads, bs, num_proposals, 4]
        return all_pred_bboxes / scales.unsqueeze(0).unsqueeze(2)

    def _build_outputs(self, all_cls_logits, norm_pred_bboxes):
        main_logits = all_cls_logits[-1]
        main_bboxes = norm_pred_bboxes[-1]
        aux_outputs = None
        if self.deep_supervision and all_cls_logits.shape[0] > 1:
            aux_outputs = [
                ModelOutput(pred_logits=all_cls_logits[i], pred_boxes=norm_pred_bboxes[i])
                for i in range(all_cls_logits.shape[0] - 1)
            ]
        return ModelOutput(pred_logits=main_logits, pred_boxes=main_bboxes, aux_outputs=aux_outputs)

    # ================================================================
    # DDPM 基线 (用于与 RF 对比实验)
    # ================================================================

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # 先按 t 索引取 bs 个值, 再 sqrt (bs << timesteps, 避免对全长向量开方)
        alpha_t = self.alphas_cumprod.gather(-1, t.long())
        reshape = (-1,) + (1,) * (x_start.dim() - 1)
        sqrt_alpha = alpha_t.sqrt().reshape(reshape)
        sqrt_one_minus = (1.0 - alpha_t).sqrt().reshape(reshape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise
