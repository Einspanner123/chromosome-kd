"""DiffusionDetHead — 核心迭代去噪检测头

直接移植自 LDMDet/mods/diffusiondet_head.py (已验证工作),
仅更新 import 路径到 ldmdet 纯 PyTorch 库。
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        cascade_detach: bool = True,
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
        coupling: Optional[nn.Module] = None,
        pre_noise_layer: int = 2,
        loss_aux: Optional[Dict] = None,
        torch_compile: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
        use_self_conditioning: bool = False,
        self_conditioning_prob: float = 0.5,
        use_distillation: bool = False,
        distill_lambda: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.diffusion_type = diffusion_type
        self.deep_supervision = deep_supervision
        self.cascade_detach = cascade_detach
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
        # SC-RF: 自条件化参数
        # use_self_conditioning: 是否启用自条件化 (模型条件化于自身上一步 x0 预测)
        # self_conditioning_prob: 训练时启用自条件化的概率 (0.5 = 50% 使用, 50% 零输入)
        self.use_self_conditioning = use_self_conditioning
        self.self_conditioning_prob = self_conditioning_prob

        # PD-RF: 直接蒸馏参数 (理论: PD-RF_Progressive_Distillation.md)
        # use_distillation: 是否启用知识蒸馏 (教师多步 → 学生1步)
        # distill_lambda: 蒸馏损失权重 λ, 控制蒸馏强度
        self.use_distillation = use_distillation
        self.distill_lambda = distill_lambda
        # teacher_model 通过 property 管理, 不注册为 nn.Module 子模块
        # (避免出现在 state_dict 中, 使 load_state_dict 不受教师权重干扰)
        object.__setattr__(self, '_teacher_model', None)

        self.loss_aux = loss_aux

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

    # ================================================================
    # PD-RF: 教师模型管理 (property 自动冻结, 不注册为子模块)
    # ================================================================

    @property
    def teacher_model(self):
        """教师模型 (外部注入). 不注册为 nn.Module 子模块, 避免污染 state_dict."""
        return getattr(self, '_teacher_model', None)

    def __setattr__(self, name, value):
        """重写以拦截 teacher_model 赋值, 自动冻结教师参数.

        nn.Module.__setattr__ 会将 nn.Module 值自动注册为子模块,
        绕过 property setter. 此处显式拦截 teacher_model 赋值,
        执行自动冻结并存储为普通属性 (非 _modules 注册).
        """
        if name == 'teacher_model':
            if value is not None:
                # 自动冻结教师参数 (教师作为固定监督源, 理论 2.1)
                for param in value.parameters():
                    param.requires_grad = False
                value.eval()
            # 使用 object.__setattr__ 绕过 nn.Module 的自动注册
            object.__setattr__(self, '_teacher_model', value)
        else:
            super().__setattr__(name, value)

    def _apply(self, fn):
        """重写以同步教师模型到当前设备 (to/cuda/cpu 等均经 _apply)."""
        super()._apply(fn)
        teacher = getattr(self, '_teacher_model', None)
        if teacher is not None:
            teacher._apply(fn)
        return self

    def _init_weights(self, prior_prob):
        for head in self.head_series:
            if hasattr(head, 'cls_head'):
                last_layer = head.cls_head[-1]
                if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                    bias_value = -(math.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(last_layer.bias, bias_value)

    # ================================================================
    # 前向传播
    # ================================================================

    def forward(self, features, bboxes, t, x0_prev=None):
        time_emb = self.time_mlp(t)
        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_curr_proposals = []
        curr_bboxes = bboxes
        curr_proposals = None

        for head in self.head_series:
            result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb, x0_prev)
            if len(result) == 4:
                cls_logits, pred_bboxes, curr_proposals, _ = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_curr_proposals.append(curr_proposals)
            curr_bboxes = pred_bboxes.detach() if self.cascade_detach else pred_bboxes

        if self.deep_supervision:
            return torch.stack(inter_cls_logits), torch.stack(inter_pred_bboxes), inter_curr_proposals
        return torch.stack(inter_cls_logits[-1:]), torch.stack(inter_pred_bboxes[-1:]), inter_curr_proposals[-1:]

    # ================================================================
    # 训练损失
    # ================================================================

    def loss(self, features, img_metas, gt_bboxes, gt_labels, x_raw_shared=None):
        device = features[0].device
        bs = len(img_metas)

        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
        t = self._sample_t(bs, device)
        x_boxes, x_starts, x_noises, matched_gt_indices = self._build_training_targets(
            bs, device, t, targets, gt_bboxes, external_noise=x_raw_shared
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps

        # SC-RF: 自条件化训练 (理论 2.4 残差学习 + 2.5 训练-推理失配缓解)
        # 训练时以 self_conditioning_prob 概率启用自条件化:
        #   - 启用: 无梯度前向获取 x0_pred_prev, 模型学习残差校正 Δ = x0 - x0_pred_prev
        #   - 不启用: 使用零输入, 保证模型有 fallback 路径 (无 x0_prev 时也能工作)
        x0_pred_prev = None
        sc_active = False
        if self.use_self_conditioning and self.training:
            if torch.rand(1, device=device).item() < self.self_conditioning_prob:
                # 无梯度前向获取 x0 预测 (残差学习的粗略估计)
                sc_active = True
                with torch.no_grad():
                    if self.amp_dtype is not None:
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                            _, sc_pred_bboxes, _ = self(features, curr_bboxes, t_input)
                        sc_pred_bboxes = sc_pred_bboxes.float()
                    else:
                        _, sc_pred_bboxes, _ = self(features, curr_bboxes, t_input)
                    # 使用最后一个 head 的预测 (最终预测), 转换到 raw 空间
                    x0_pred_prev = self._sampler.xyxy_to_raw(sc_pred_bboxes[-1], img_metas)
            else:
                # 零输入: fallback 路径, 模型在无 x0_prev 时也能工作
                x0_pred_prev = torch.zeros_like(x_noisy_batch)

        # 模型前向：若启用 AMP，在 autocast 下执行（线性层/attention 用半精度加速）
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                all_cls_logits, all_pred_bboxes, all_curr_proposals = self(features, curr_bboxes, t_input, x0_pred_prev)
            # autocast 输出可能为半精度，criterion 需 FP32（如 cdist 不支持 BF16）
            all_cls_logits = all_cls_logits.float()
            all_pred_bboxes = all_pred_bboxes.float()
        else:
            all_cls_logits, all_pred_bboxes, all_curr_proposals = self(features, curr_bboxes, t_input, x0_pred_prev)

        norm_pred_bboxes = self._normalize_pred_bboxes(all_pred_bboxes, img_metas)
        outputs = self._build_outputs(all_cls_logits, norm_pred_bboxes)
        # 方向三: 传 t 给 criterion (若 criterion 不支持 t 则被忽略, 向后兼容)
        # t 是 [bs] 的扩散时间, 用于 SNR 感知匹配和损失加权
        losses = self.criterion(outputs, targets, t=t)

        # SC-RF: 插桩监控关键数值 (上传 SwanLab, 非 loss_ 前缀不参与反传)
        if self.use_self_conditioning and self.training:
            with torch.no_grad():
                # sc_active: 本步是否启用了自条件化 (1.0=启用, 0.0=零输入)
                losses['sc_active'] = torch.tensor(float(sc_active), device=device)
                # sc_x0_prev_norm: x0_prev 的 L2 范数 (启用时非零, 零输入时为 0)
                losses['sc_x0_prev_norm'] = x0_pred_prev.norm(dim=-1).mean().detach()
                # sc_correction_magnitude: x0_prev_proj 输出的 L2 范数 (校正强度)
                # 初始零初始化时应为 ~0, 训练后逐渐增长
                correction = sum(
                    h.x0_prev_proj(x0_pred_prev).norm(dim=-1).mean()
                    for h in self.head_series
                ) / len(self.head_series)
                losses['sc_correction_magnitude'] = correction.detach()

        return losses

    # ================================================================
    # PD-RF: 直接知识蒸馏 (理论: PD-RF_Progressive_Distillation.md)
    # ================================================================

    def loss_with_distillation(self, features, img_metas, gt_bboxes, gt_labels):
        """含知识蒸馏的训练损失 (理论 3.1).

        核心流程:
        1. 生成共享噪声 x_raw_shared (教师和学生共用, 保证 proposal 对应)
        2. 教师多步推理获取 raw x0 + cls_logits (无梯度, 关闭 box_renewal, 不经 NMS)
        3. 学生前向计算检测损失 (有梯度, 使用共享噪声)
        4. 学生 1步 x0_pred + cls_logits (有梯度, 不在 no_grad 下!)
        5. 蒸馏损失: MSE(student_x0, teacher_x0) + KL(student_cls || teacher_cls)
        6. SwanLab 插桩监控关键数值
        """
        # 无教师或 lambda=0 时退化为标准训练
        if self.teacher_model is None or self.distill_lambda == 0:
            return self.loss(features, img_metas, gt_bboxes, gt_labels)

        device = features[0].device
        bs = len(img_metas)

        # === 1. 共享噪声 (理论 3.1: 教师和学生必须使用同一 x_raw) ===
        x_raw_shared = torch.randn(bs, self.num_proposals, 4, device=device)

        # === 2. 教师多步推理 (无梯度, 关闭 box_renewal, 不经 NMS) ===
        teacher_cls, teacher_x0 = self._teacher_multistep_x0(
            features, img_metas, x_raw_shared, return_cls=True
        )

        # === 3. 学生检测损失 (有梯度, 使用共享噪声) ===
        losses = self.loss(features, img_metas, gt_bboxes, gt_labels, x_raw_shared=x_raw_shared)

        # === 4. 学生 1步 x0_pred + cls_logits (有梯度! 蒸馏损失需反传到学生参数) ===
        student_cls, student_x0 = self._student_single_step_x0(
            features, img_metas, x_raw_shared, return_cls=True
        )

        # === 5. 蒸馏损失 (理论 2.5) ===
        # 5a. Box 蒸馏: raw 空间 MSE
        distill_loss_box = F.mse_loss(student_x0, teacher_x0.detach())
        # 5b. 分类蒸馏: KL 散度 (补充分类头在 t=1.0 的训练)
        student_cls_log = F.log_softmax(student_cls, dim=-1)
        teacher_cls_soft = F.softmax(teacher_cls.detach(), dim=-1)
        distill_loss_cls = F.kl_div(
            student_cls_log, teacher_cls_soft, reduction='batchmean'
        )
        losses['loss_distill'] = (distill_loss_box + distill_loss_cls) * self.distill_lambda

        # === 6. SwanLab 插桩 (非 loss_ 前缀, 不参与反传, 仅记录) ===
        with torch.no_grad():
            losses['pd_distill_loss_box'] = distill_loss_box.detach()
            losses['pd_distill_loss_cls'] = distill_loss_cls.detach()
            losses['pd_student_teacher_gap'] = (
                (student_x0.detach() - teacher_x0.detach()).norm(dim=-1).mean()
            )
            det_loss_sum = sum(
                v for k, v in losses.items()
                if k.startswith('loss_') and k != 'loss_distill'
            )
            if isinstance(det_loss_sum, torch.Tensor):
                det_loss_sum = det_loss_sum.detach()
                total = det_loss_sum + losses['loss_distill'].detach()
                losses['pd_det_loss_ratio'] = det_loss_sum / (total + 1e-8)

        return losses

    def _teacher_multistep_x0(self, features, img_metas, x_raw, return_cls=False):
        """教师多步推理获取 raw x0 (理论 3.1).

        关键实现点:
        - 使用 _forward_at_t 多步循环 (非 predict, 避免 post-NMS)
        - 关闭 box_renewal (保持 proposal 对应)
        - 无梯度 (教师冻结, torch.no_grad)
        - 返回 [bs, P, 4] raw 张量
        """
        teacher = self.teacher_model
        device = features[0].device
        time_pairs = teacher._sampler.build_time_pairs(device)
        x = x_raw.clone()

        # 创建 DPM-Solver++ (若教师配置为 dpm_solver_pp)
        dpm_solver = teacher._sampler.create_dpm_solver()
        if dpm_solver is not None:
            dpm_solver.reset()

        x0 = None
        cls_logits = None
        with torch.no_grad():
            for step_idx, (t_curr, t_next) in enumerate(time_pairs):
                cls_logits, _, x0 = teacher._forward_at_t(
                    features, x, t_curr, img_metas
                )
                if dpm_solver is not None:
                    x = dpm_solver.step(x, x0, t_curr, step_idx)
                elif teacher.solver_type == 'heun' and t_next > 0:
                    # Heun 二阶: 需要中点评估
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = teacher._forward_at_t(
                            features, x_tmp, t_tmp, img_metas
                        )
                        return x0_tmp, None
                    x = teacher.rf.heun_step(x, x0, t_curr, t_next, model_fn)
                else:
                    x = teacher.rf.step(x, x0, t_curr, t_next)
                # 不调用 apply_box_renewal! (保持 proposal 对应)
        if return_cls:
            return cls_logits, x0
        return x0

    def _student_single_step_x0(self, features, img_metas, x_raw, return_cls=False):
        """学生 1步 x0_pred (理论 3.1: t=1.0 单步前向).

        关键: 不在 no_grad 下! 蒸馏损失需对学生参数求梯度.
        """
        # t=1.0: 纯噪声输入, 模型预测 x0
        # SC-RF 兼容: 若启用自条件化, x0_prev 初始化为零 (第一步无先验)
        x0_prev = torch.zeros_like(x_raw) if self.use_self_conditioning else None
        cls_logits, _, x0 = self._forward_at_t(
            features, x_raw, 1.0, img_metas, x0_prev
        )
        if return_cls:
            return cls_logits, x0
        return x0

    # ================================================================
    # 推理
    # ================================================================

    @torch.no_grad()
    def predict(self, features, img_metas, rescale=True, return_trajectory=False):
        device = features[0].device
        bs = len(img_metas)
        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        # SC-RF: 初始化 x0_pred_prev 为零 (第一步无先验预测, 理论 2.2 零初始化保证)
        # 推理时每步将上一步的 x0 预测作为条件输入, 实现迭代精炼 (理论 2.3)
        x0_pred_prev = torch.zeros_like(x_raw) if self.use_self_conditioning else None

        ensemble_results = []
        trajectory = []
        dpm_solver = self._sampler.create_dpm_solver()
        if dpm_solver is not None:
            dpm_solver.reset()

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas, x0_pred_prev
            )
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
                    # SC-RF: Heun 中点评估使用当前步的 x0_pred_prev (跨时间步信息一致)
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = self._forward_at_t(
                            features, x_tmp, t_tmp, img_metas, x0_pred_prev
                        )
                        return x0_tmp, None
                    x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                # SC-RF: 更新 x0_pred_prev 为当前步的 x0 预测 (供下一步使用, 理论 2.3)
                if self.use_self_conditioning:
                    x0_pred_prev = x0_raw

                if self.box_renewal:
                    x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
                if t_next <= 0:
                    break

        results = self._sampler.post_process(ensemble_results, img_metas, rescale)

        if return_trajectory:
            return results, trajectory
        return results

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

    def _build_training_targets(self, bs, device, t, targets, gt_bboxes, external_noise=None):
        x_boxes, x_starts, x_noises, matched_gt_indices = [], [], [], []
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            if num_gt == 0:
                if external_noise is not None:
                    noise = external_noise[i]
                else:
                    noise = torch.randn(self.num_proposals, 4, device=device)
                x_boxes.append(noise)
                x_starts.append(torch.zeros_like(noise))
                x_noises.append(noise)
                matched_gt_indices.append(torch.zeros(self.num_proposals, dtype=torch.long, device=device))
                continue
            norm_gt_cxcywh = bbox_xyxy_to_cxcywh(targets[i].bboxes)
            gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale
            if external_noise is not None:
                noise = external_noise[i]
            else:
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

    def _forward_at_t(self, features, x_raw, t, img_metas, x0_prev=None):
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        # AMP: 推理前向用半精度加速 GEMM (FFN 占 42.4% 瓶颈),
        # 输出转回 fp32 以保证后续 box_renewal / NMS / solver 的数值精度
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                cls_logits_seq, pred_bboxes_seq, _ = self(features, curr_bboxes, t_input, x0_prev)
            cls_logits_seq = cls_logits_seq.float()
            pred_bboxes_seq = pred_bboxes_seq.float()
        else:
            cls_logits_seq, pred_bboxes_seq, _ = self(features, curr_bboxes, t_input, x0_prev)
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
