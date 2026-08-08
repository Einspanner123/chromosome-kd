"""DiffusionDetHead — 核心迭代去噪检测头

直接移植自 LDMDet/mods/diffusiondet_head.py (已验证工作),
仅更新 import 路径到 ldmdet 纯 PyTorch 库。
"""

import copy
import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ldmdet.coupling import build_coupling
from ldmdet.data.structures import (
    InstanceData,
    ModelOutput,
)
from ldmdet.diagnostics.instrumentation import probe
from ldmdet.diffusion.box_chart import ValidBoxChart
from ldmdet.diffusion.embeddings import SinusoidalPositionEmbeddings
from ldmdet.diffusion.noise_schedule import cosine_noise_schedule
from ldmdet.diffusion.rectified_flow import RectifiedFlow
from ldmdet.diffusion.sampling import DiffusionSampler, _get_img_shape, KaryotypeScorer, pcse_select
from ldmdet.utils.box_ops import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

logger = logging.getLogger(__name__)


def calibrate_class_logits(cls_logits, quality_logits, beta=2.0):
    """Fuse class probability and IoU quality, returning stable logits."""
    probability = cls_logits.sigmoid()
    quality = quality_logits.sigmoid()
    calibrated = probability * quality.pow(beta)
    calibrated = calibrated.clamp(min=1e-6, max=1.0 - 1e-6)
    return torch.logit(calibrated)


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
        box_parameterization: str = 'linear_cxcywh',
        box_chart_eps: float = 1e-6,
        box_chart_mean: Optional[list] = None,
        box_chart_covariance: Optional[list] = None,
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
        topk_pruning_enabled: bool = False,
        topk_k: int = 100,
        topk_pruning_step: int = 0,
        # PCSE: 配对一致性随机集成评分 (推理优化)
        use_pcse: bool = False,
        pcse_k: int = 5,
        pcse_lambda: float = 0.5,
        pcse_alphas: Tuple[float, float, float, float] = (1.0, 2.0, 1.5, 1.5),
        pcse_diversity: str = 'seed',
        # CCBR: 跨级联框更新 (推理优化)
        use_ccbr: bool = False,
        ccbr_alpha: float = 0.7,
        ccbr_sigma: float = 0.1,
        ccbr_lambda: float = 0.5,
        ccbr_beta: float = 0.5,
        # SHTS: SNR 层级时间步采样 (推理优化)
        shts_alpha: float = 1.0,
        shts_sigma: float = 0.15,
        shts_shifted: bool = False,
        # 方向2: Per-Head Time Reparameterization (PHTR)
        # 给每个级联头可学习的 time_scale/time_shift, 打破 time_emb 共享
        use_time_reparam: bool = False,
        # 方向4: VGAR (Velocity-Guided Adaptive Renewal)
        velocity_guided_renewal: bool = False,
        # P0: 自适应阈值 box_renewal (移植自 DiffuDETR)
        # 启用后, renewal 阈值随时间步递减: 早期高阈值, 后期低阈值
        adaptive_renewal_threshold: bool = False,
        adaptive_renewal_scale: float = 0.9,
        # 方向 D: 自适应阶次 DPM-Solver++ (推理时改动, 无需重训练)
        # 仅当 solver_type='dpm_solver_pp_adaptive' 时生效
        adaptive_solver_mode: str = 'static',
        adaptive_num_3rd_steps: int = 2,
        adaptive_eta_3rd_threshold: float = 0.5,
        # GACS: geometry-aware adaptive stopping (1/2-step inference).
        # The last two cascade stages provide a zero-extra-NFE posterior
        # consistency residual after the first solver evaluation.
        adaptive_stopping: bool = False,
        adaptive_stop_min_steps: int = 1,
        adaptive_stop_geo_threshold: float = 0.05,
        adaptive_stop_cls_threshold: float = 0.01,
        adaptive_stop_score_threshold: float = 0.5,
        adaptive_stop_min_box_scale: float = 0.0,
        adaptive_stop_topk: int = 100,
        # IQC: IoU Quality Calibration. The single head predicts q(IoU), and
        # inference ranks detections by p(class) * q ** beta.
        quality_score_beta: float = 2.0,
        quality_calibration_mode: str = 'solver_coupled',
        quality_only_training: bool = False,
        # Head Distillation v2: 少 Head (H=3) 蒸馏多 Head (H=6)
        # 详见 docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md §2
        use_distillation: bool = False,
        distill_lambda: float = 0.05,
        distill_head_map: Optional[Dict[int, int]] = None,
        deep_supervision_aux_weight: float = 1.0,
        freeze_backbone: bool = False,
        # ReFlow (Standard MSE): 预存 coupling 替代在线 OT
        # 详见 docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md §1
        use_reflow_coupling: bool = False,
        reflow_coupling_path: Optional[str] = None,
        # ReFlow per-dim 拉直维度选择 (标记用, 当前 criterion 仅实现 'all';
        # 'cxcy' per-dim 消融为可选未来扩展, 见方案 §1.4)
        reflow_dims: str = 'all',
        # LVD-RF: Lyapunov Velocity Direction Regularization (保守方向)
        # 详见 docs/research/proposals/FEASIBLE_LVD_RF.md
        # 在训练损失中增加 Lyapunov 方向余弦正则项 (默认 sin² 形式),
        # 利用 d=4 低维优势以零额外前向传播计算方向余弦,
        # 通过 Lyapunov 稳定性条件约束速度场方向, 间接降低 η_str 并减少
        # DPM-Solver++ 截断误差. 仅改训练, 推理零开销, NFE 保持 24 不变.
        use_lvd: bool = False,
        lvd_lambda: float = 0.1,
        lvd_eps: float = 1e-6,                 # 数值稳定常数
        lvd_t_threshold: float = 0.05,        # t 过小时跳过 (||x_t-x_0||→0 余弦不稳定)
        lvd_space: str = 'raw_cxcywh',        # 计算空间 (方案 §3.2: 唯一支持 raw_cxcywh)
        lvd_form: str = 'sin2',                # R1 K3: 默认 sin² (梯度比 1-cos 强 2×)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.snr_scale = snr_scale
        self.box_parameterization = box_parameterization
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
        self.adaptive_stopping = adaptive_stopping
        self.adaptive_stop_min_steps = int(adaptive_stop_min_steps)
        self.adaptive_stop_geo_threshold = float(adaptive_stop_geo_threshold)
        self.adaptive_stop_cls_threshold = float(adaptive_stop_cls_threshold)
        self.adaptive_stop_score_threshold = float(adaptive_stop_score_threshold)
        self.adaptive_stop_min_box_scale = float(adaptive_stop_min_box_scale)
        self.adaptive_stop_topk = int(adaptive_stop_topk)
        self._adaptive_stop_stats = None
        self._last_cascade_consistency = None
        self.quality_score_beta = float(quality_score_beta)
        self.quality_calibration_mode = quality_calibration_mode
        self.quality_only_training = bool(quality_only_training)
        self._last_quality_logits = None
        if self.quality_score_beta < 0:
            raise ValueError('quality_score_beta must be non-negative')
        if self.quality_calibration_mode not in ('solver_coupled', 'final_only'):
            raise ValueError(
                'quality_calibration_mode must be solver_coupled or final_only')

        if adaptive_stopping:
            if sampling_timesteps != 2:
                raise ValueError(
                    'adaptive_stopping currently requires sampling_timesteps=2 '
                    'so exits are exactly the validated 1/2-step schedules')
            if num_heads < 2:
                raise ValueError('adaptive_stopping requires at least two cascade heads')
            if not (1 <= adaptive_stop_min_steps <= sampling_timesteps):
                raise ValueError('adaptive_stop_min_steps must be in [1, 2]')
            if adaptive_stop_topk <= 0:
                raise ValueError('adaptive_stop_topk must be positive')

        if box_parameterization not in ('linear_cxcywh', 'gap_ilr'):
            raise ValueError(
                'box_parameterization must be linear_cxcywh or gap_ilr, '
                f'got {box_parameterization!r}')
        if box_parameterization == 'gap_ilr':
            if diffusion_type != 'rectified_flow':
                raise ValueError('gap_ilr is currently implemented for RF only')
            if box_chart_mean is None or box_chart_covariance is None:
                raise ValueError(
                    'gap_ilr requires box_chart_mean and box_chart_covariance')
            if use_reflow_coupling:
                raise ValueError('existing ReFlow couplings use linear_cxcywh coordinates')
            if use_lvd:
                raise ValueError('LVD raw_cxcywh loss is incompatible with gap_ilr')
            if solver_type in ('dpm_solver_pp_per_dim', 'dpm_solver_pp_per_dim_w'):
                raise ValueError('per-dimension cxcywh solvers are incompatible with gap_ilr')
            self.box_chart = ValidBoxChart(
                eps=box_chart_eps,
                mean=torch.as_tensor(box_chart_mean, dtype=torch.float64),
                covariance=torch.as_tensor(
                    box_chart_covariance, dtype=torch.float64),
            )
        else:
            self.box_chart = None

        self.loss_aux = loss_aux

        # 扩散组件
        # RectifiedFlow 的 use_reflow_coupling / reflow_dims 为标记参数,
        # q_sample 行为不变 (reflow 逻辑在 _build_training_targets 中处理);
        # reflow_dims 留作未来 per-dim 拉直扩展 (当前 criterion 仅实现 'all')
        self.rf = RectifiedFlow(
            snr_scale=snr_scale,
            use_reflow_coupling=use_reflow_coupling,
            reflow_dims=reflow_dims,
        )
        self.reflow_dims = reflow_dims
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
        self.head_series = nn.ModuleList(
            [copy.deepcopy(single_head) for _ in range(num_heads)]
        )
        # C2 is deliberately a last-stage-only intervention. Removing the
        # cloned quality modules from earlier cascade heads also avoids unused
        # parameters under DistributedDataParallel.
        if getattr(single_head, 'quality_head', None) is not None:
            for head in self.head_series[:-1]:
                head.quality_head = None
                head.predict_iou_quality = False
        self.roi_extractor = roi_extractor
        self.criterion = criterion
        self.pre_noise_layer = pre_noise_layer

        # 耦合策略
        self.ot_coupling = coupling is not None and not isinstance(
            coupling, bool
        )
        self.ot_module = (
            coupling if coupling is not None else build_coupling('random')
        )

        # SHTS: SNR 层级时间步采样 (推理优化, 需在_sampler前赋值)
        self.shts_alpha = shts_alpha
        self.shts_sigma = shts_sigma
        self.shts_shifted = shts_shifted

        # 方向4: VGAR — 透传给 DiffusionSampler
        self.velocity_guided_renewal = velocity_guided_renewal

        # 采样器
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
            # SHTS params
            shts_alpha=self.shts_alpha,
            shts_sigma=self.shts_sigma,
            shts_shifted=self.shts_shifted,
            # 方向4: VGAR
            velocity_guided_renewal=velocity_guided_renewal,
            # 方向 D: 自适应阶次 solver 参数
            adaptive_solver_mode=adaptive_solver_mode,
            adaptive_num_3rd_steps=adaptive_num_3rd_steps,
            adaptive_eta_3rd_threshold=adaptive_eta_3rd_threshold,
            # P0: 自适应阈值 box_renewal
            adaptive_renewal_threshold=adaptive_renewal_threshold,
            adaptive_renewal_scale=adaptive_renewal_scale,
            box_chart=self.box_chart,
        )

        self._init_weights(prior_prob)

        if self.quality_only_training:
            quality_head = getattr(self.head_series[-1], 'quality_head', None)
            if quality_head is None:
                raise ValueError(
                    'quality_only_training requires predict_iou_quality=True')
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in quality_head.parameters():
                parameter.requires_grad_(True)

        # AMP: 仅模型前向使用半精度，criterion 始终 FP32
        # 推荐值: torch.bfloat16 (同动态范围，无需 GradScaler)
        # 支持字符串 (配置文件无需 import torch, 避免 mmengine lazy_import 冲突)
        if isinstance(amp_dtype, str):
            amp_dtype = getattr(torch, amp_dtype)
        self.amp_dtype = amp_dtype

        if torch_compile and hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward, dynamic=True)

        # IO3: Top-K 框剪枝 (推理优化, 不影响训练)
        self.topk_pruning_enabled = topk_pruning_enabled
        self.topk_k = topk_k
        self.topk_pruning_step = topk_pruning_step
        self._pruning_stats = {}

        # ============================================================
        # PCSE: 配对一致性随机集成评分 (推理优化, 无需重训练)
        # ============================================================
        self.use_pcse = use_pcse
        self.pcse_k = pcse_k
        self.pcse_lambda = pcse_lambda
        self.pcse_alphas = pcse_alphas
        self.pcse_diversity = pcse_diversity
        self._pcse_scorer = None  # 延迟初始化

        # ============================================================
        # CCBR: 跨级联框更新 (推理优化, 无需重训练)
        # ============================================================
        self.use_ccbr = use_ccbr
        self.ccbr_alpha = ccbr_alpha
        self.ccbr_sigma = ccbr_sigma
        self.ccbr_lambda = ccbr_lambda
        self.ccbr_beta = ccbr_beta

        # ============================================================
        # 方向2: Per-Head Time Reparameterization (PHTR)
        # ============================================================
        # 给每个级联头可学习的 time_scale 和 time_shift, 对共享的 time_emb
        # 做 head 特有的仿射变换: time_emb_i = time_emb * scale_i + shift_i
        # 初始化为 identity (scale=1, shift=0), 确保不破坏预训练兼容性
        self.use_time_reparam = use_time_reparam
        if self.use_time_reparam:
            self.head_time_scale = nn.Parameter(
                torch.ones(num_heads, feat_channels * 4)
            )
            self.head_time_shift = nn.Parameter(
                torch.zeros(num_heads, feat_channels * 4)
            )

        # ============================================================
        # Head Distillation v2: 少 Head (H=3) 蒸馏多 Head (H=6)
        # ============================================================
        # Teacher (H=6, A4 冻结) 监督 Student (H=3) 的 fc_feature
        # L_distill = (1/K) Σ_k MSE(student_fc_k, teacher_fc_{map(k)}.detach())
        # Student head 0/1/2 ← Teacher head 0/2/5 (输入对齐 + 中间 + main 对齐)
        self.use_distillation = use_distillation
        self.distill_lambda = distill_lambda
        # 默认映射: Student H=3 → Teacher H=6
        #   head 0 ↔ head 0 (输入对齐, 都是第一个 head)
        #   head 1 ↔ head 2 (中间, 进度对齐 ~1/2 处)
        #   head 2 ↔ head 5 (main 对齐, 都是最后一个 head = 主输出)
        self.distill_head_map = distill_head_map or {0: 0, 1: 2, 2: 5}
        self.deep_supervision_aux_weight = deep_supervision_aux_weight
        self.freeze_backbone = freeze_backbone
        self._teacher: Optional[nn.Module] = None

        # ============================================================
        # ReFlow (Standard MSE): 预存 coupling (x_0^pred, x_1^noise)
        # ============================================================
        # 用 A4 推理生成的 (x_0^pred, x_1^noise) 替代在线 (GT, randn),
        # 训练 2-RF 拉直轨迹 (cls=GT, box=x_0^pred 混合 target, 详见方案 §1.2)
        # coupling 懒加载: __init__ 仅存路径, 首次 _build_training_targets 调用时加载
        # (避免 __init__ 阶段文件不存在导致测试/构建失败; 也便于测试 mock)
        self.use_reflow_coupling = use_reflow_coupling
        self.reflow_coupling_path = reflow_coupling_path
        self._reflow_couplings: Optional[dict] = None
        # ReFlow coupling 生成支持: predict() 结束时设置, __init__ 预初始化为 None
        # 避免 predict() 调用前访问导致 AttributeError (测试/诊断场景)
        self._last_x_raw_initial: Optional[torch.Tensor] = None
        self._last_x0_final: Optional[torch.Tensor] = None
        if self.use_reflow_coupling:
            assert self.reflow_coupling_path is not None, (
                "use_reflow_coupling=True 时必须指定 reflow_coupling_path "
                "(指向 generate_reflow_couplings.py 生成的 coupling 文件)"
            )

        # ============================================================
        # LVD-RF: Lyapunov Velocity Direction Regularization 状态
        # ============================================================
        # 仅训练时生效. 零额外前向传播: 复用训练步已有的 (x_t, x_hat_0, x_0)
        # 计算 v_θ 与 (x_t - x_0) 的方向余弦, 约束方向对齐 (Lyapunov 稳定性)
        # 详见 docs/research/proposals/FEASIBLE_LVD_RF.md §7
        self.use_lvd = use_lvd
        self.lvd_lambda = lvd_lambda
        self.lvd_eps = lvd_eps
        self.lvd_t_threshold = lvd_t_threshold
        # lvd_space: LVD-RF 方向余弦计算空间 (方案 §3.2).
        # 当前唯一支持 'raw_cxcywh' (与 x_t, x_0 同空间, 需将 xyxy 像素预测
        # 转换至此空间). 非该值直接报错退出 (方案约束: 不做自动降级).
        self.lvd_space = lvd_space
        assert lvd_space == 'raw_cxcywh', (
            f"lvd_space 当前仅支持 'raw_cxcywh' (方案 §3.2 唯一计算空间), "
            f"got {lvd_space!r}"
        )
        # lvd_form 运行时可被自适应切换 (sin2 → sqrt), 故存为可变属性
        self.lvd_form = lvd_form
        assert lvd_form in ('sin2', 'cos', 'sqrt'), (
            f"lvd_form 必须是 'sin2' / 'cos' / 'sqrt', got {lvd_form}"
        )
        # 自适应切换计数器: cos_sim_mean > 0.99 持续 1000 iter 时切换 sin2 → sqrt
        # (sin² 仍线性消失, sqrt 形式有非零梯度 0.5/sqrt(ε), 详见方案 §7.2 注 2.3)
        self._cos_sim_high_count = 0

    def _init_weights(self, prior_prob):
        for head in self.head_series:
            if hasattr(head, 'cls_head'):
                last_layer = head.cls_head[-1]
                if hasattr(last_layer, 'bias') and last_layer.bias is not None:
                    bias_value = -(math.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(last_layer.bias, bias_value)

    # ================================================================
    # Head Distillation v2: Teacher 注入
    # ================================================================

    def set_teacher(self, teacher: nn.Module):
        """注入 Teacher 模型并冻结其参数。

        Teacher 通过 _teacher 引用持有 (非 nn.Module 子模块),
        不出现在 self.parameters() 中, 避免参数进入 optimizer。
        使用 object.__setattr__ 绕过 nn.Module 的自动子模块注册。

        Args:
            teacher: Teacher DiffusionDetHead 实例 (H=6, A4 权重)
        """
        # object.__setattr__ 绕过 nn.Module.__setattr__ 的自动注册,
        # 使 Teacher 参数不进入 self.parameters() / optimizer
        object.__setattr__(self, '_teacher', teacher)
        # 冻结 Teacher 所有参数
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

    def _apply(self, fn):
        """覆写 _apply, 将 device/dtype 变更传播到 Teacher。

        Teacher 通过 object.__setattr__ 持有 (非 nn.Module 子模块),
        标准 nn.Module.to()/cuda()/float() 不会自动传播到 Teacher,
        导致 GPU 训练时 Teacher 留在 CPU 触发 device 不匹配。
        覆写 _apply (to/cuda 等的内部机制) 确保 Teacher 与 Student
        始终在同一 device 上。
        """
        super()._apply(fn)
        if self._teacher is not None:
            self._teacher._apply(fn)
        return self

    def init_student_from_teacher(self):
        """从 Teacher 初始化 Student 权重 (Head Distillation v2)

        Student head 0/1/2 ← Teacher head 0/2/5 (与 distill_head_map 一致),
        共享模块 (time_mlp, roi_extractor 等) 直接从 Teacher 复制。

        S1 已证随机初始化 H=3 训练失败 (N_cascade e2e 教训),
        因此 Student 必须从 Teacher 初始化而非随机。
        """
        if self._teacher is None:
            logger.warning(
                'init_student_from_teacher: Teacher 未注入, 跳过初始化'
            )
            return

        teacher_sd = self._teacher.state_dict()
        student_sd = self.state_dict()
        new_sd = {}

        # 1. 复制非 head_series 的共享模块 (time_mlp, roi_extractor, rf 等)
        for k, v in teacher_sd.items():
            if not k.startswith('head_series.'):
                if k in student_sd and student_sd[k].shape == v.shape:
                    new_sd[k] = v

        # 2. 按 distill_head_map 复制 head_series
        # Student head 0 ← Teacher head 0 (输入对齐)
        # Student head 1 ← Teacher head 2 (中间进度对齐)
        # Student head 2 ← Teacher head 5 (main 对齐)
        for s_idx, t_idx in self.distill_head_map.items():
            prefix_s = f'head_series.{s_idx}.'
            prefix_t = f'head_series.{t_idx}.'
            for k, v in teacher_sd.items():
                if k.startswith(prefix_t):
                    new_k = prefix_s + k[len(prefix_t):]
                    if (
                        new_k in student_sd
                        and student_sd[new_k].shape == v.shape
                    ):
                        new_sd[new_k] = v

        missing, unexpected = self.load_state_dict(new_sd, strict=False)
        if missing:
            logger.warning(
                f'init_student_from_teacher: missing keys ({len(missing)}): '
                f'{missing[:5]}'
            )
        if unexpected:
            logger.warning(
                f'init_student_from_teacher: unexpected keys ({len(unexpected)}): '
                f'{unexpected[:5]}'
            )
        logger.info(
            f'init_student_from_teacher: 已从 Teacher 初始化 Student '
            f'({len(new_sd)}/{len(student_sd)} keys)'
        )

    # ================================================================
    # 前向传播
    # ================================================================

    def forward(self, features, bboxes, t, img_metas=None):
        time_emb = self.time_mlp(t)
        # 探针: time_emb 激活统计 (训练时每 100 步, 推理时每次)
        if self.training:
            probe.record_tensor_stats('cascade/time_emb', time_emb)
        else:
            probe.record_inference_tensor_stats('cascade/time_emb', time_emb)
        inter_cls_logits = []
        inter_pred_bboxes = []
        inter_curr_proposals = []
        last_quality_logits = None
        curr_bboxes = bboxes
        curr_proposals = None
        prev_bboxes = None
        prev_logits = None

        for i, head in enumerate(self.head_series):
            # 方向2: PHTR — 每个头的 time_emb 做独立仿射变换
            # time_emb_i = time_emb * scale_i + shift_i
            if self.use_time_reparam:
                time_emb_i = (
                    time_emb * self.head_time_scale[i]
                    + self.head_time_shift[i]
                )
            else:
                time_emb_i = time_emb
            result = head(
                features, curr_bboxes, curr_proposals,
                self.roi_extractor, time_emb_i,
            )
            if len(result) == 4:
                cls_logits, pred_bboxes, curr_proposals, quality_logits = result
                last_quality_logits = quality_logits
            else:
                cls_logits, pred_bboxes, curr_proposals = result
            inter_cls_logits.append(cls_logits)
            inter_pred_bboxes.append(pred_bboxes)
            inter_curr_proposals.append(curr_proposals)

            # 探针: per-head 激活统计 (cls_logits + pred_bboxes)
            if self.training:
                probe.record_tensor_stats(f'cascade/head{i}/cls_logits', cls_logits)
                probe.record_tensor_stats(f'cascade/head{i}/pred_bboxes', pred_bboxes)
            else:
                probe.record_inference_tensor_stats(f'cascade/head{i}/cls_logits', cls_logits)
                probe.record_inference_tensor_stats(f'cascade/head{i}/pred_bboxes', pred_bboxes)

            # 级联: 将当前 head 输出作为下一 head 的输入
            curr_bboxes = (
                pred_bboxes.detach()
                if self.cascade_detach
                else pred_bboxes
            )
            prev_bboxes = pred_bboxes
            prev_logits = cls_logits

        self._last_quality_logits = last_quality_logits

        if self.deep_supervision:
            return (
                torch.stack(inter_cls_logits),
                torch.stack(inter_pred_bboxes),
                inter_curr_proposals,
            )
        return (
            torch.stack(inter_cls_logits[-1:]),
            torch.stack(inter_pred_bboxes[-1:]),
            inter_curr_proposals[-1:],
        )

    # ================================================================
    # 训练损失
    # ================================================================

    def loss(self, features, img_metas, gt_bboxes, gt_labels, x_raw_shared=None):
        # Head Distillation v2: 蒸馏模式分支
        if self.use_distillation and self._teacher is not None:
            return self._loss_with_distillation(
                features, img_metas, gt_bboxes, gt_labels
            )
        if self.use_distillation and self._teacher is None:
            logger.warning(
                'use_distillation=True 但 Teacher 未注入, 回退到普通 loss(). '
                '请检查 detector 是否正确构建了 Teacher.'
            )
        if self.quality_only_training:
            # The detector may still build backbone/neck features normally,
            # but detaching here makes C2 a strict ranking-only intervention:
            # no existing representation, classifier, or regressor can drift.
            features = tuple(feature.detach() for feature in features)
        device = features[0].device
        bs = len(img_metas)
        self._adaptive_stop_stats = None
        self._last_cascade_consistency = None

        targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
        t = self._sample_t(bs, device)
        # ReFlow: 用 img_id 索引预存 coupling (非 reflow 模式传 None, 回退 batch 索引)
        # ImageMeta 是 dataclass, 用属性访问 (img_id=None 时回退 batch 索引, 便于测试)
        img_ids = (
            [meta.img_id if meta.img_id is not None else i
             for i, meta in enumerate(img_metas)]
            if self.use_reflow_coupling else None
        )
        x_boxes, x_starts, x_noises, matched_gt_indices = (
            self._build_training_targets(
                bs, device, t, targets, gt_bboxes,
                external_noise=x_raw_shared, img_ids=img_ids,
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._sampler.raw_to_xyxy(x_noisy_batch, img_metas)

        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps

        # 模型前向：若启用 AMP，在 autocast 下执行（线性层/attention 用半精度加速）
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                all_cls_logits, all_pred_bboxes, all_curr_proposals = self(
                    features, curr_bboxes, t_input, img_metas
                )
            # autocast 输出可能为半精度，criterion 需 FP32（如 cdist 不支持 BF16）
            all_cls_logits = all_cls_logits.float()
            all_pred_bboxes = all_pred_bboxes.float()
        else:
            all_cls_logits, all_pred_bboxes, all_curr_proposals = self(
                features, curr_bboxes, t_input, img_metas
            )

        norm_pred_bboxes = self._normalize_pred_bboxes(
            all_pred_bboxes, img_metas
        )
        outputs = self._build_outputs(all_cls_logits, norm_pred_bboxes)
        if self._last_quality_logits is not None:
            # C2 isolates the final cascade stage. Auxiliary heads retain the
            # original losses so the experiment has a single causal change.
            outputs.pred_quality = self._last_quality_logits.float()
        # 方向三: 传 t 给 criterion (若 criterion 不支持 t 则被忽略, 向后兼容)
        # t 是 [bs] 的扩散时间, 用于 SNR 感知匹配和损失加权
        # ReFlow: 传 per-proposal x_0^pred 作为 box target
        # (criterion.box_target_mode='x0_pred' 时生效; 'gt' 模式忽略, 向后兼容)
        # x_starts 在 reflow 模式 = x_0^pred [num_proposals, 4], stack → [bs, num_proposals, 4]
        #
        # 空间一致性修复 (BUG #1): x_starts 来自 coupling 的 x0_pred 字段,
        # 处于 raw 扩散空间 (cxcywh, scaled [-snr_scale, snr_scale]); 而
        # outputs.pred_boxes 处于归一化 xyxy 空间 [0,1] (经 _normalize_pred_bboxes).
        # criterion._loss_boxes 假设 box_targets 与 src_boxes 同空间, 必须先转换:
        #   raw cxcywh [-snr, snr] → normalized cxcywh [0,1] → normalized xyxy [0,1]
        # (与 _sampler.raw_to_xyxy 的前两步一致, 但不乘 img_scale 以保持归一化)
        if self.use_reflow_coupling:
            raw_x_starts = torch.stack(x_starts)  # [bs, num_proposals, 4] raw cxcywh
            box_targets = self._sampler.raw_to_normalized_xyxy(raw_x_starts)
        else:
            box_targets = None
        losses = self.criterion(outputs, targets, t=t, box_targets=box_targets)

        # LVD-RF 正则化 (零额外前向, 复用 x_boxes/x_starts/all_pred_bboxes)
        # 详见 docs/research/proposals/FEASIBLE_LVD_RF.md §7.2
        # 仅训练时生效; 推理路径 (predict) 不触及此分支
        if self.use_lvd and self.training:
            lvd_loss = self._compute_lvd_loss(
                x_boxes=x_boxes,                    # x_t, list[bs] of [N, 4] raw cxcywh
                x_starts=x_starts,                  # x_0 (GT), list[bs] of [N, 4] raw cxcywh
                all_pred_bboxes=all_pred_bboxes,     # \hat{x}_0, [H, bs, N, 4] xyxy 像素
                t=t,                                  # [bs] 扩散时间
                img_metas=img_metas,
            )
            losses['loss_lvd'] = self.lvd_lambda * lvd_loss

        # 探针: 训练时 t 分布 + 损失分解
        probe.record_tensor_stats('train/t', t)
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                probe.record_scalar(f'train/loss/{name}', val.item())

        return losses

    # ================================================================
    # LVD-RF: Lyapunov Velocity Direction Regularization
    # ================================================================

    def _compute_lvd_loss(
        self,
        x_boxes: list,
        x_starts: list,
        all_pred_bboxes: torch.Tensor,
        t: torch.Tensor,
        img_metas,
    ) -> torch.Tensor:
        r"""LVD-RF: Lyapunov Velocity Direction Regularization.

        L_LVD = E[sin²(α)] = E[1 - cos²(α)]   (默认 sin² 形式, R1 K3 修正)

        由定理 2.4, 方向余弦尺度不变, 1/t 在分子分母抵消, 可直接用
        (x_t - x_hat_0) 和 (x_t - x_0) 计算余弦, 无需显式除以 t (数值稳定).

        空间一致性 (FEASIBLE_LVD_RF.md §3.2):
          - x_boxes (x_t), x_starts (x_0): raw cxcywh [-snr_scale, snr_scale]
          - all_pred_bboxes (x_hat_0): xyxy 像素 → 需转换到 raw cxcywh
          转换链: xyxy 像素 → 归一化 xyxy [0,1] → 归一化 cxcywh [0,1]
                  → raw cxcywh [-s, s]  (与 _sampler.raw_to_xyxy 互逆)

        Args:
            x_boxes: x_t, list[bs] of [N, 4] raw cxcywh
            x_starts: x_0 (GT), list[bs] of [N, 4] raw cxcywh
            all_pred_bboxes: x_hat_0, [num_heads, bs, N, 4] xyxy 像素
            t: [bs] 扩散时间
            img_metas: 图像元数据 (list[bs], 兼容 dict / ImageMeta)
        Returns:
            lvd_loss: 标量 (已对有效样本归一化)
        """
        bs = len(img_metas)
        device = x_boxes[0].device

        # 堆叠 list → tensor: [bs, N, 4]
        x_t = torch.stack(x_boxes)             # [bs, N, 4] raw cxcywh
        x_0 = torch.stack(x_starts)            # [bs, N, 4] raw cxcywh

        # 取末级 cascade head 的预测 (与 TFR/VCR 一致, 仅正则末级)
        # all_pred_bboxes: [num_heads, bs, N, 4] xyxy 像素
        pred_xyxy_pixel = all_pred_bboxes[-1]  # [bs, N, 4] xyxy 像素

        # 转换 \hat{x}_0 到 raw cxcywh (与 x_t, x_0 同空间)
        # xyxy 像素 → 归一化 xyxy [0,1] → 归一化 cxcywh [0,1] → raw cxcywh [-s, s]
        # 注: 使用 _get_img_shape 兼容 dict / ImageMeta 两种 img_metas 格式
        scales = x_t.new_zeros(bs, 4)
        for i in range(bs):
            h, w = _get_img_shape(img_metas[i])[:2]
            scales[i] = x_t.new_tensor([w, h, w, h])
        # [bs, 1, 4] 广播除法 (clamp 防 0 尺寸图像)
        norm_xyxy = pred_xyxy_pixel / scales.unsqueeze(1).clamp(min=1.0)
        norm_cxcywh = bbox_xyxy_to_cxcywh(norm_xyxy)        # [bs, N, 4] in [0,1]
        raw_cxcywh = (norm_cxcywh * 2 - 1) * self.snr_scale   # [bs, N, 4] in [-s, s]
        x_hat_0 = raw_cxcywh

        # 方向 1: 预测残差方向 (x_t - x_hat_0) ∝ v_θ (1/t 在余弦中抵消)
        d_pred = x_t - x_hat_0  # [bs, N, 4]

        # 方向 2: GT 锚定方向 (x_t - x_0) = t * (x_1 - x_0)
        d_gt = x_t - x_0  # [bs, N, 4]

        # 跳过 t 过小的样本 (||x_t - x_0|| = t * ||x_1 - x_0|| → 0, 余弦不稳定)
        # t_mask: [bs] → [bs, 1] (广播到 [bs, N])
        t_mask = (t >= self.lvd_t_threshold).float()  # [bs]
        t_mask = t_mask.view(bs, 1)  # [bs, 1] → broadcast to [bs, N]

        # 余弦相似度: cos(d_pred, d_gt) = (d_pred · d_gt) / (||d_pred|| ||d_gt||)
        dot = (d_pred * d_gt).sum(dim=-1)                # [bs, N]
        norm_pred = d_pred.norm(dim=-1).clamp(min=self.lvd_eps)  # [bs, N]
        norm_gt = d_gt.norm(dim=-1).clamp(min=self.lvd_eps)      # [bs, N]
        cos_sim = dot / (norm_pred * norm_gt)            # [bs, N]
        # R1 R2: clamp 到 [-1, 1] 防止浮点误差导致 sin² = 1 - cos² 略为负
        cos_sim = cos_sim.clamp(-1.0, 1.0)

        # R1 S4: xyxy 有效性检查 (x2 > x1, y2 > y1), 对无效框跳过 LVD 计算
        # 无效框 cos_sim 设为 1 (loss = 1 - 1² = 0, 不贡献)
        valid_mask = (
            (pred_xyxy_pixel[..., 2] > pred_xyxy_pixel[..., 0])
            & (pred_xyxy_pixel[..., 3] > pred_xyxy_pixel[..., 1])
        )  # [bs, N]
        cos_sim = torch.where(valid_mask, cos_sim, torch.ones_like(cos_sim))

        # R1 K3: 默认 sin² 形式 (梯度比 1-cos 强 2×, 缓解 cos_sim→1 梯度消失)
        # t_mask 广播: [bs, 1] * [bs, N] → [bs, N]
        if self.lvd_form == 'sin2':
            lvd_per_elem = (1.0 - cos_sim ** 2) * t_mask  # sin²(α)
        elif self.lvd_form == 'cos':
            lvd_per_elem = (1.0 - cos_sim) * t_mask       # 1 - cos(α)
        elif self.lvd_form == 'sqrt':
            # sqrt(1 - cos + ε): cos_sim→1 时梯度 → 0.5/sqrt(ε) (有界非零)
            lvd_per_elem = torch.sqrt(
                (1.0 - cos_sim).clamp(min=self.lvd_eps)
            ) * t_mask
        else:
            raise ValueError(f"Unknown lvd_form: {self.lvd_form}")

        # 有效样本数: 同时考虑 t_mask (大 t) 和 valid_mask (有效 xyxy)
        # R2 修正: 无效框 lvd=0 但原 n_valid 计入分母, 导致 loss 被低估
        # t_mask [bs,1] 广播到 [bs,N], 与 valid_mask [bs,N] 相乘得有效 proposal 掩码
        effective_mask = t_mask * valid_mask.float()  # [bs, N]
        n_valid = effective_mask.sum().clamp(min=1.0)
        lvd_loss = lvd_per_elem.sum() / n_valid

        # 诊断: 训练时 cos_sim 分布 + 自适应切换逻辑 (R1 K3 修正)
        # 诊断/切换仅在有效样本 (effective_mask = t_mask * valid_mask) 上统计,
        # 与 loss 归一化保持一致:
        #   - 无效框 cos_sim 被置 1.0 (loss 占位符, 非真实对齐度量)
        #   - 小 t 样本 ||x_t-x_0||=t·||x_1-x_0||→0, cos_sim 为数值噪声
        #   二者混入全量均值会虚高 cos_sim_mean, 训练早期无效框占比高时可
        #   误触发 sin2→sqrt 切换 (审计问题 2+4).
        with torch.no_grad():
            n_eff = effective_mask.sum().clamp(min=1.0)
            cos_sim_mean = (cos_sim * effective_mask).sum() / n_eff
            # cos_sim_min: 非有效样本填 1.0 (cos 上界), 不影响有效样本最小值
            cos_sim_min = torch.where(
                effective_mask.bool(), cos_sim, torch.ones_like(cos_sim)
            ).min()
            probe.record_scalar('train/lvd_loss', lvd_loss.item())
            probe.record_scalar('train/cos_sim_mean', cos_sim_mean.item())
            probe.record_scalar('train/cos_sim_min', cos_sim_min.item())
            probe.record_scalar(
                'train/lvd_valid_ratio', t_mask.mean().item()
            )
            # 梯度健康度: 若 cos_sim_mean > 0.99 持续 1000 iter, 触发切换
            if cos_sim_mean.item() > 0.99:
                self._cos_sim_high_count += 1
                if (
                    self._cos_sim_high_count > 1000
                    and self.lvd_form == 'sin2'
                ):
                    self.lvd_form = 'sqrt'  # 自动切换
                    logger.info(
                        'LVD-RF: cos_sim_mean > 0.99 持续 1000 iter, '
                        '切换至 sqrt 形式 (避免梯度消失)'
                    )
            else:
                self._cos_sim_high_count = 0
            # per-dim 方向偏差 (各维度对余弦的贡献, 仅有效样本)
            d_pred_normed = d_pred / norm_pred.unsqueeze(-1)
            d_gt_normed = d_gt / norm_gt.unsqueeze(-1)
            per_dim_contrib = d_pred_normed * d_gt_normed  # [bs, N, 4]
            per_dim_mean = (
                (per_dim_contrib * effective_mask.unsqueeze(-1))
                .sum(dim=(0, 1)) / n_eff
            )
            for i, name in enumerate(['cx', 'cy', 'w', 'h']):
                probe.record_scalar(
                    f'train/lvd_dir_{name}', per_dim_mean[i].item()
                )
            # LVD 损失对 cos_sim 的解析梯度量级 (监测 sqrt 切换后梯度放大,
            # 审计问题 3: cos→1 时 sqrt 梯度 → 0.5/√ε = 500, ×λ → 50)
            #   sin2:  d(1-cos²)/d(cos) = -2·cos        → |grad| = 2|cos|
            #   cos:   d(1-cos)/d(cos)   = -1            → |grad| = 1
            #   sqrt:  d(√((1-cos)∧ε))/d(cos):
            #          1-cos > ε 时 = -0.5/√(1-cos); 1-cos ≤ ε 时 = 0 (clamp 死区)
            uncos = 1.0 - cos_sim
            if self.lvd_form == 'sin2':
                grad_abs = 2.0 * cos_sim.abs()
            elif self.lvd_form == 'cos':
                grad_abs = torch.ones_like(cos_sim)
            else:  # sqrt
                in_bounds = (uncos > self.lvd_eps).float()
                grad_abs = (
                    0.5 / torch.sqrt(uncos.clamp(min=self.lvd_eps)) * in_bounds
                )
            grad_eff = grad_abs * effective_mask  # 非有效样本置 0
            probe.record_scalar(
                'train/lvd_grad_mean',
                (grad_eff.sum() / n_eff * self.lvd_lambda).item(),
            )
            probe.record_scalar(
                'train/lvd_grad_max',
                (grad_eff.max() * self.lvd_lambda).item(),
            )
            # sqrt 死区占比: 有效样本中 1-cos ≤ ε (梯度归 0) 的比例
            if self.lvd_form == 'sqrt':
                deadzone_ratio = (
                    ((uncos <= self.lvd_eps).float() * effective_mask)
                    .sum() / n_eff
                ).item()
                probe.record_scalar(
                    'train/lvd_grad_deadzone_ratio', deadzone_ratio
                )
        return lvd_loss

    # ================================================================
    # Head Distillation v2: 蒸馏训练损失
    # ================================================================

    def _loss_with_distillation(
        self, features, img_metas, gt_bboxes, gt_labels
    ):
        """Head Distillation v2 蒸馏损失。

        Student (H=3) 和 Teacher (H=6) 共享噪声/时间步, forward 后对
        fc_feature 做 headwise MSE 蒸馏:
          L_total = L_det(student) + λ · L_distill
          L_distill = (1/K) Σ_k MSE(student_fc_k, teacher_fc_{map(k)})

        v2 修正 (见 REFLOW_HEAD_DISTILL_IMPL_PLAN.md §2.2):
          - 蒸馏 fc_feature (非 pred_bboxes), 梯度更稳定
          - head 映射 {0→0, 1→2, 2→5} (输入/中间/main 对齐)
          - λ=0.05 保守起步
          - aux loss 权重降至 deep_supervision_aux_weight
        """
        device = features[0].device
        bs = len(img_metas)

        targets = self._normalize_targets(
            gt_bboxes, gt_labels, img_metas, bs
        )
        t = self._sample_t(bs, device)

        # 共享噪声: Student/Teacher 使用相同 x_raw (v2 修正 #5: 确定性 coupling)
        x_raw_shared = torch.randn(
            bs, self.num_proposals, 4, device=device
        )
        x_boxes, x_starts, x_noises, matched_gt_indices = (
            self._build_training_targets(
                bs, device, t, targets, gt_bboxes,
                external_noise=x_raw_shared,
            )
        )
        x_noisy_batch = torch.stack(x_boxes)
        curr_bboxes = self._sampler.raw_to_xyxy(x_noisy_batch, img_metas)
        t_input = t if self.diffusion_type == 'ddpm' else t * self.timesteps

        # Student forward (收集 fc_features)
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                s_cls, s_bbox, s_fc_feats = self(
                    features, curr_bboxes, t_input, img_metas
                )
            s_cls = s_cls.float()
            s_bbox = s_bbox.float()
            s_fc_feats = [f.float() for f in s_fc_feats]
        else:
            s_cls, s_bbox, s_fc_feats = self(
                features, curr_bboxes, t_input, img_metas
            )

        # Teacher forward (no_grad, 收集 fc_features)
        # AMP: Teacher forward 也用半精度, 与 Student 保持一致
        with torch.no_grad():
            if self.amp_dtype is not None:
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    t_cls, t_bbox, t_fc_feats = teacher(
                        features, curr_bboxes, t_input, img_metas
                    )
                t_fc_feats = [f.float() for f in t_fc_feats]
            else:
                t_cls, t_bbox, t_fc_feats = teacher(
                    features, curr_bboxes, t_input, img_metas
                )

        # 检测损失 L_det
        norm_pred_bboxes = self._normalize_pred_bboxes(s_bbox, img_metas)
        outputs = self._build_outputs(s_cls, norm_pred_bboxes)
        losses = self.criterion(outputs, targets, t=t)

        # v2 修正 #6: aux loss 权重降低
        if (
            self.deep_supervision
            and self.deep_supervision_aux_weight != 1.0
        ):
            for key in list(losses.keys()):
                if key.startswith('aux_'):
                    losses[key] = (
                        losses[key] * self.deep_supervision_aux_weight
                    )

        # 蒸馏损失 L_distill (headwise fc_feature MSE)
        # v2 修正 #6: 蒸馏所有 mapped head (§2.1 公式 (1/K)Σ_k)
        # §2.6 "仅 main head" 描述最小可用配置, 当前蒸馏全部 mapped head
        distill_loss = torch.tensor(
            0.0, device=device, dtype=s_bbox.dtype
        )
        n_distill = 0
        for s_idx, t_idx in self.distill_head_map.items():
            if s_idx < len(s_fc_feats) and t_idx < len(t_fc_feats):
                # 缓存 MSE 结果, 避免 probe 重复计算
                head_gap = F.mse_loss(
                    s_fc_feats[s_idx].float(),
                    t_fc_feats[t_idx].float().detach(),
                )
                distill_loss = distill_loss + head_gap
                n_distill += 1
                # 探针: 逐 head feature gap
                probe.record_scalar(
                    f'distill/head{s_idx}_feat_gap', head_gap.item()
                )
        if n_distill > 0:
            distill_loss = distill_loss / n_distill
        else:
            logger.warning(
                f'蒸馏 loss 计算了 0 个 head (n_distill=0). '
                f'distill_head_map={self.distill_head_map}, '
                f'student_heads={len(s_fc_feats)}, '
                f'teacher_heads={len(t_fc_feats)}. '
                f'请检查 head 映射配置.'
            )

        losses['loss_distill'] = self.distill_lambda * distill_loss

        # 探针: 蒸馏 loss 统计
        probe.record_scalar('distill/loss_distill', distill_loss.item())
        probe.record_scalar(
            'distill/loss_ratio',
            distill_loss.item() / max(
                sum(
                    v.item()
                    for k, v in losses.items()
                    if isinstance(v, torch.Tensor) and 'distill' not in k
                ),
                1e-8,
            ),
        )
        probe.record_tensor_stats('train/t', t)
        for name, val in losses.items():
            if isinstance(val, torch.Tensor):
                probe.record_scalar(f'train/loss/{name}', val.item())

        return losses

    # ================================================================
    # 推理
    # ================================================================

    @torch.no_grad()
    def predict(
        self, features, img_metas, rescale=True, return_trajectory=False
    ):
        device = features[0].device
        bs = len(img_metas)

        # 探针: 推理开始标记 (清空推理缓冲区)
        probe.on_inference_begin()

        # ================================================================
        # PCSE 分支: 多假设采样 + 核型评分选择
        # ================================================================
        if self.use_pcse:
            return self._predict_pcse(features, img_metas, rescale)

        # ================================================================
        # CCBR 分支: 跨级联框更新
        # ================================================================
        if self.use_ccbr:
            return self._predict_ccbr(features, img_metas, rescale, return_trajectory)

        # ================================================================
        # 原始 predict (基线)
        # ================================================================
        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
        # ReFlow coupling 生成支持: 记录初始噪声 x_1 (raw 空间), 供
        # generate_reflow_couplings.py 读取 (仅 box_renewal/ensemble/pruning 关闭时为干净轨迹)
        x_raw_initial = x_raw.detach().clone()
        x0_final = None  # 跟踪最后一步的 x0 预测

        ensemble_results = []
        trajectory = []
        dpm_solver = self._sampler.create_dpm_solver()
        # 显式校验: 防止 solver_type 拼写错误或未知值时静默降级到 Euler.
        # create_dpm_solver() 仅对 euler/heun 返回 None (合法), 其他 solver_type
        # 必须返回有效实例, 否则下方 if-elif-else 会静默走 Euler 分支.
        if dpm_solver is None:
            assert self.solver_type in ('euler', 'heun', 'ddim'), (
                f"solver_type='{self.solver_type}' 不被 create_dpm_solver() 支持, "
                f"且不属于 euler/heun/ddim. 请检查配置或扩展 create_dpm_solver()."
            )
        if dpm_solver is not None:
            dpm_solver.reset()
            # 混合求解器 (dpm_pp_heun_hybrid): 注入 model_fn 供 Heun 校正项额外前向.
            # 闭包捕获 features/img_metas, 签名 model_fn(x, t) -> (x0_pred, None)
            if self.solver_type == 'dpm_pp_heun_hybrid':
                def _hybrid_model_fn(x_tmp, t_tmp):
                    _, _, x0_tmp = self._forward_at_t(
                        features, x_tmp, t_tmp, img_metas
                    )
                    return x0_tmp, None
                dpm_solver.model_fn = _hybrid_model_fn

        # D3 化解路径 A: 记录上一步被 box_renewal 重置的 proposal mask
        # 在下一步 DPM-Solver++ step() 中传入, 对被 renewal 的 proposal
        # 置零 D1 校正项, 避免 renewal 噪声污染 x0_history 导致 D1 失效
        _renewal_mask: Optional[torch.Tensor] = None  # [bs, N] bool
        adaptive_gate_metrics = None
        adaptive_stopped = False
        actual_steps = 0

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            actual_steps = step_idx + 1
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )
            output_logits = (
                self._quality_ranking_logits(cls_logits)
                if self.quality_calibration_mode == 'final_only'
                else cls_logits
            )
            # ReFlow coupling 生成: 跟踪每步 x0, 循环结束后保留最后一步
            x0_final = x0_raw

            # 探针: 推理时 per-step 激活统计 (x0_pred + cls_logits + pred_bboxes)
            probe.record_inference_tensor_stats(
                f'inference/step{step_idx}/x0_pred', x0_raw
            )
            probe.record_inference_tensor_stats(
                f'inference/step{step_idx}/cls_logits', cls_logits
            )
            probe.record_inference_tensor_stats(
                f'inference/step{step_idx}/pred_bboxes', pred_bboxes
            )
            probe.record_inference_scalar(
                f'inference/step{step_idx}/t_curr', float(t_curr)
            )

            # IO3: Top-K 框剪枝 — 在指定步后保留 Top-K 高置信框
            if (
                self.topk_pruning_enabled
                and step_idx == self.topk_pruning_step
                and x_raw.shape[1] > self.topk_k
            ):
                x_raw, cls_logits, pred_bboxes, x0_raw, topk_indices = (
                    self._sampler.apply_topk_pruning(
                        x_raw, cls_logits, pred_bboxes, x0_raw,
                        k=self.topk_k,
                    )
                )
                output_logits = output_logits.gather(
                    1, topk_indices.unsqueeze(-1).expand(
                        -1, -1, output_logits.shape[-1]))
                # 剪枝后 DPM-Solver history 维度不匹配, 必须重置
                if dpm_solver is not None:
                    dpm_solver.reset()
                # 记录剪枝统计 (供 SwanLab 插桩)
                scores = torch.sigmoid(cls_logits).max(-1)[0]
                self._pruning_stats = {
                    'pruning_step': step_idx,
                    'n_before': self.num_proposals,
                    'n_after': self.topk_k,
                    'kept_mean_score': scores.mean().item(),
                }
            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))
            if self.use_ensemble:
                ensemble_results.append((output_logits, pred_bboxes))
            else:
                # Non-ensemble inference must return the latest solver result.
                # The previous ``if not ensemble_results`` implementation kept
                # step 0 forever, silently making later NFEs ineffective.
                ensemble_results[:] = [(output_logits, pred_bboxes)]

            # GACS early exit.  With sampling_timesteps=2 this check after the
            # first evaluation chooses exactly between the established 1-step
            # and 2-step schedules.  Batch inference exits only when every
            # image passes; per-image asynchronous stepping is intentionally
            # avoided to preserve tensor shapes and solver history.
            if (
                self.adaptive_stopping
                and actual_steps >= self.adaptive_stop_min_steps
                and actual_steps < len(time_pairs)
            ):
                adaptive_gate_metrics = self._last_cascade_consistency
                stop_mask = self._adaptive_stop_mask(adaptive_gate_metrics)
                if bool(stop_mask.all()):
                    adaptive_stopped = True
                    break

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
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx,
                                             renewal_mask=_renewal_mask)
                elif self.solver_type == 'heun' and t_next > 0:
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = self._forward_at_t(
                            features, x_tmp, t_tmp, img_metas
                        )
                        return x0_tmp, None
                    x_raw = self.rf.heun_step(
                        x_raw, x0_raw, t_curr, t_next, model_fn
                    )
                else:
                    x_raw = self.rf.step(
                        x_raw, x0_raw, t_curr, t_next
                    )

                if self.box_renewal:
                    # 方向4 VGAR: 传入 v_θ 预测的 x0 和当前时间步, 启用速度场引导
                    # 探针: 记录 box_renewal 前的 x_raw (用于计算重置率)
                    x_raw_before = x_raw.clone()
                    n_before = x_raw.shape[1]
                    x_raw = self._sampler.apply_box_renewal(
                        x_raw, cls_logits,
                        x0_pred=x0_raw,
                        t_curr=t_curr,
                    )
                    # D3 化解路径 A: 记录被 renewal 的 proposal mask
                    # 比较新旧 x_raw, 不一致的 proposal 即被 renewal
                    _renewal_mask = ~torch.isclose(
                        x_raw, x_raw_before, atol=1e-6
                    ).all(dim=-1)  # [bs, N]
                    # 探针: box_renewal 统计 (重置率 + 置信度分布)
                    n_after = x_raw.shape[1]
                    if n_before > 0:
                        renewal_rate = 1.0 - min(n_after, n_before) / n_before
                        probe.record_inference_scalar(
                            f'inference/step{step_idx}/box_renewal_rate', renewal_rate
                        )
                    # proposal 置信度分布 (max sigmoid score)
                    scores = torch.sigmoid(cls_logits).max(-1)[0]
                    probe.record_inference_tensor_stats(
                        f'inference/step{step_idx}/proposal_scores', scores
                    )
                # 探针: solver 推进后的 x_raw 统计
                probe.record_inference_tensor_stats(
                    f'inference/step{step_idx}/x_raw_after', x_raw
                )
                if t_next <= 0:
                    break

        if self.adaptive_stopping:
            metrics = adaptive_gate_metrics or self._last_cascade_consistency
            self._adaptive_stop_stats = {
                'actual_steps': actual_steps,
                'stopped_early': adaptive_stopped,
                'geo_residual': metrics['geo_residual'].detach().cpu().tolist(),
                'cls_residual': metrics['cls_residual'].detach().cpu().tolist(),
                'mean_topk_score': metrics['mean_topk_score'].detach().cpu().tolist(),
                'lower_box_scale': metrics['lower_box_scale'].detach().cpu().tolist(),
            }

        results = self._sampler.post_process(
            ensemble_results, img_metas, rescale
        )

        # ReFlow coupling 生成支持: 暴露初始噪声与最终 x0 (raw 空间, [bs, num_proposals, 4])
        # generate_reflow_couplings.py 读取这两个属性构造 (x_0^pred, x_1^noise) coupling.
        # 注意: 仅当 box_renewal/use_ensemble/topk_pruning 关闭时为干净的单轨迹;
        #       正常推理 (box_renewal 开) 下 x_raw_initial 仍为初始噪声, x0_final 为最后一步 x0.
        # 防御性断言: time_pairs 为空 (sampling_timesteps=0) 时 x0_final 仍为 None,
        # generate_reflow_couplings.py 读取会 TypeError, 此处显式报错便于定位
        assert x0_final is not None, (
            "predict() 结束时 x0_final 为 None — time_pairs 为空 "
            "(sampling_timesteps=0?). ReFlow coupling 生成需要至少一步采样."
        )
        self._last_x_raw_initial = x_raw_initial
        self._last_x0_final = x0_final

        # IO3: SwanLab 插桩 — 上传剪枝统计
        if self.topk_pruning_enabled and self._pruning_stats:
            try:
                import swanlab
                swanlab.log({
                    'inference/pruning_n_before': self._pruning_stats['n_before'],
                    'inference/pruning_n_after': self._pruning_stats['n_after'],
                    'inference/pruning_kept_mean_score': self._pruning_stats['kept_mean_score'],
                    'inference/pruning_step': self._pruning_stats['pruning_step'],
                })
            except Exception as e:
                logger.warning(f'[SwanLab] inference metrics log failed: {e}')

        # R1 诊断: 收集本次推理的 eta_str 历史 (DPM-Solver++ 直线度指标)
        # dpm_solver.eta_str_history 在每个 step 后 append 一个值;
        # 4 步推理下通常 length=3 (step 0 是 linear, 无 D1; step 1/2/3 记录)
        # 保存到 self._last_eta_str_log 供实验脚本读取 (不写入 SwanLab, 避免污染训练指标)
        # 注意: 不用 try-except 吞错 — 如果 dpm_solver 缺少诊断字段, 应显式报错而非静默回退
        if dpm_solver is not None:
            self._last_eta_str_log = list(dpm_solver.eta_str_history)
            # 方向 A/D 诊断: 同时收集 per-dim eta_str 和 eta_3rd
            self._last_eta_str_per_dim_log = [
                list(x) for x in dpm_solver.eta_str_per_dim_history
            ]
            self._last_eta_3rd_log = list(dpm_solver.eta_3rd_history)
            # 方向 D: 自适应 solver 才有 applied_3rd_history; 普通 solver 设为空
            self._last_applied_3rd_log = list(
                getattr(dpm_solver, 'applied_3rd_history', [])
            )
        else:
            self._last_eta_str_log = []
            self._last_eta_str_per_dim_log = []
            self._last_eta_3rd_log = []
            self._last_applied_3rd_log = []

        # 探针: 上传 solver 诊断量到推理缓冲区
        for i, eta in enumerate(self._last_eta_str_log):
            probe.record_inference_scalar(f'inference/eta_str/step{i}', eta)
        for i, eta in enumerate(self._last_eta_3rd_log):
            probe.record_inference_scalar(f'inference/eta_3rd/step{i}', eta)
        # per-dim eta_str (cx, cy, w, h)
        for i, eta_dim in enumerate(self._last_eta_str_per_dim_log):
            for j, dim_name in enumerate(['cx', 'cy', 'w', 'h']):
                if j < len(eta_dim):
                    probe.record_inference_scalar(
                        f'inference/eta_str_per_dim/step{i}/{dim_name}', eta_dim[j]
                    )
        # 方向 D: applied_3rd history
        for i, applied in enumerate(self._last_applied_3rd_log):
            probe.record_inference_scalar(
                f'inference/applied_3rd/step{i}', float(applied)
            )

        # 探针: 推理结束, flush 推理缓冲区到 SwanLab
        probe.on_inference_end()

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
            targets.append(
                InstanceData(
                    labels=gt_labels[i],
                    bboxes=gt_bboxes[i] / scale,
                    img_shape=(h, w),
                )
            )
        return targets

    def _sample_t(self, bs, device):
        if self.diffusion_type == 'ddpm':
            return torch.randint(
                0, self.timesteps, (bs,), device=device
            ).long()
        t = torch.rand((bs,), device=device)
        if self.rf_schedule == 'shifted':
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        elif self.rf_schedule == 'shts':
            # SHTS CDF 反演采样 (Phase 2 训练匹配, Phase 1 推理仅用网格)
            # 暂退化为 shifted 以保证训练稳定
            t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
        return t

    def _build_training_targets(
        self, bs, device, t, targets, gt_bboxes,
        external_noise=None, img_ids=None,
    ):
        """构建训练 targets (x_noisy, x_start, x_noise, matched_gt_idx).

        ReFlow 模式 (use_reflow_coupling=True): 从预存 coupling 加载
        x_start=x_0^pred 与 noise=x_1^noise (A4 推理生成), 替代在线 (GT, randn);
        matched_idx 仍在线重算 (基于 GT, 供 cls 正样本分配). 详见方案 §1.2.

        Args:
            img_ids: 可选, 每图的 coupling 键 (真实训练用 img_id; 测试/未传时回退到 batch 索引 i).
        """
        # ReFlow: 懒加载 coupling (首次调用时从磁盘载入, 后续复用)
        if self.use_reflow_coupling and self._reflow_couplings is None:
            self._reflow_couplings = self._load_reflow_coupling(
                self.reflow_coupling_path
            )

        x_boxes, x_starts, x_noises, matched_gt_indices = [], [], [], []
        for i in range(bs):
            num_gt = gt_bboxes[i].shape[0]
            # ReFlow coupling 键: 优先 img_id, 回退 batch 索引 (测试路径)
            img_key = img_ids[i] if img_ids is not None else i

            if self.use_reflow_coupling:
                coupling = self._reflow_couplings[img_key]
                # 预存 per-proposal noise (x_1) 与 x_0^pred (轨迹端点)
                noise = coupling['noise'].to(device)
                x_start = coupling['x0_pred'].to(device)

                if num_gt == 0:
                    # 无 GT: 无正样本, x_start 置零 (box target 不参与)
                    x_start = torch.zeros_like(noise)
                    matched_idx = torch.zeros(
                        self.num_proposals, dtype=torch.long, device=device
                    )
                else:
                    # matched_idx 在线重算: OT couple (noise, GT) 仅供 cls 分配
                    gt_diffusion = self._sampler.normalized_xyxy_to_raw(
                        targets[i].bboxes)
                    _, matched_idx = self._couple_single_image(
                        noise, gt_diffusion, targets[i].labels, device
                    )
                x_noisy, x_noise = self._forward_diffusion(
                    x_start, noise, t[i : i + 1]
                )
                matched_gt_indices.append(matched_idx)
                x_starts.append(x_start)
                x_noises.append(x_noise)
                x_boxes.append(x_noisy)
                continue

            # 标准 / OT 模式 (现有行为)
            if num_gt == 0:
                if external_noise is not None:
                    noise = external_noise[i]
                else:
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
            gt_diffusion = self._sampler.normalized_xyxy_to_raw(
                targets[i].bboxes)
            if external_noise is not None:
                noise = external_noise[i]
            else:
                noise = torch.randn(self.num_proposals, 4, device=device)
            x_start, matched_idx = self._couple_single_image(
                noise, gt_diffusion, targets[i].labels, device
            )
            matched_gt_indices.append(matched_idx)
            x_noisy, x_noise = self._forward_diffusion(
                x_start, noise, t[i : i + 1]
            )
            x_starts.append(x_start)
            x_noises.append(x_noise)
            x_boxes.append(x_noisy)
        return x_boxes, x_starts, x_noises, matched_gt_indices

    def _load_reflow_coupling(self, path: str):
        """加载预存的 reflow coupling 文件 (懒加载).

        格式 (generate_reflow_couplings.py 生成):
            {img_id: {'noise': tensor[num_proposals, 4],   # x_1^noise (A4 推理时的噪声)
                      'x0_pred': tensor[num_proposals, 4]}} # x_0^pred  (A4 推理预测, 轨迹端点)

        Returns:
            dict: img_id → {'noise', 'x0_pred'}
        """
        logger.info(f'ReFlow: 加载 coupling 文件 {path}')
        return torch.load(path, map_location='cpu')

    def _couple_single_image(self, noise, gt_diffusion, gt_labels, device):
        if self.ot_coupling and self.diffusion_type == 'rectified_flow':
            return self.ot_module.couple(
                noise, gt_diffusion, gt_labels, device
            )
        num_gt = gt_diffusion.shape[0]
        idx = torch.randint(0, num_gt, (self.num_proposals,), device=device)
        return gt_diffusion[idx], idx

    def _forward_diffusion(self, x_start, noise, t):
        if self.diffusion_type == 'ddpm':
            return self.q_sample(x_start, t), torch.zeros_like(x_start)
        x_noisy, _ = self.rf.q_sample(x_start, x_noise=noise, t=t)
        return x_noisy, noise

    def _cascade_consistency_metrics(
        self, cls_logits_seq, pred_bboxes_seq, img_metas
    ):
        """Return per-image posterior residuals of the final cascade update.

        Proposal indices are preserved across cascade stages.  Localization
        consistency is measured in the detector's RF coordinates (BoxChart
        coordinates for gap_ilr), while classification consistency is the
        change in the final winning-class sigmoid score.  Both are evaluated
        only on the final stage's top-scoring proposals.
        """
        cls_prev, cls_last = cls_logits_seq[-2], cls_logits_seq[-1]
        boxes_prev, boxes_last = pred_bboxes_seq[-2], pred_bboxes_seq[-1]
        z_prev = self._sampler.xyxy_to_raw(boxes_prev, img_metas)
        z_last = self._sampler.xyxy_to_raw(boxes_last, img_metas)

        probs_prev = cls_prev.sigmoid()
        probs_last = cls_last.sigmoid()
        scores, labels = probs_last.max(dim=-1)
        k = min(self.adaptive_stop_topk, scores.shape[1])
        top_scores, top_indices = scores.topk(k, dim=1)

        gather4 = top_indices.unsqueeze(-1).expand(-1, -1, 4)
        z_delta = (z_last.gather(1, gather4) - z_prev.gather(1, gather4))
        geo = z_delta.square().mean(dim=-1).sqrt().mean(dim=-1)

        winning_labels = labels.gather(1, top_indices)
        last_winning = probs_last.gather(1, top_indices.unsqueeze(-1).expand(
            -1, -1, probs_last.shape[-1]
        )).gather(2, winning_labels.unsqueeze(-1)).squeeze(-1)
        prev_winning = probs_prev.gather(1, top_indices.unsqueeze(-1).expand(
            -1, -1, probs_prev.shape[-1]
        )).gather(2, winning_labels.unsqueeze(-1)).squeeze(-1)
        cls = (last_winning - prev_winning).abs().mean(dim=-1)

        scales = boxes_last.new_empty(boxes_last.shape[0], 4)
        for i, meta in enumerate(img_metas):
            h, w = _get_img_shape(meta)[:2]
            scales[i] = boxes_last.new_tensor([w, h, w, h])
        normalized = boxes_last / scales.unsqueeze(1)
        top_boxes = normalized.gather(1, gather4)
        wh = (top_boxes[..., 2:] - top_boxes[..., :2]).clamp_min(0)
        # Lower-decile scale is robust to a few degenerate false proposals,
        # while retaining images whose confident proposal set contains small
        # targets for the second solver evaluation.
        lower_box_scale = torch.quantile(
            (wh[..., 0] * wh[..., 1]).sqrt(), 0.1, dim=1)

        return {
            'geo_residual': geo,
            'cls_residual': cls,
            'mean_topk_score': top_scores.mean(dim=-1),
            'lower_box_scale': lower_box_scale,
        }

    def _adaptive_stop_mask(self, metrics):
        return (
            (metrics['geo_residual'] <= self.adaptive_stop_geo_threshold)
            & (metrics['cls_residual'] <= self.adaptive_stop_cls_threshold)
            & (metrics['mean_topk_score'] >= self.adaptive_stop_score_threshold)
            & (metrics['lower_box_scale'] >= self.adaptive_stop_min_box_scale)
        )

    def _forward_at_t(self, features, x_raw, t, img_metas):
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        # AMP: 推理前向用半精度加速 GEMM (FFN 占 42.4% 瓶颈),
        # 输出转回 fp32 以保证后续 box_renewal / NMS / solver 的数值精度
        if self.amp_dtype is not None:
            with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                cls_logits_seq, pred_bboxes_seq, _ = self(
                    features, curr_bboxes, t_input, img_metas
                )
            cls_logits_seq = cls_logits_seq.float()
            pred_bboxes_seq = pred_bboxes_seq.float()
        else:
            cls_logits_seq, pred_bboxes_seq, _ = self(
                features, curr_bboxes, t_input, img_metas
            )
        if self.adaptive_stopping:
            self._last_cascade_consistency = self._cascade_consistency_metrics(
                cls_logits_seq, pred_bboxes_seq, img_metas)
        cls_logits_last = cls_logits_seq[-1]
        pred_bboxes_last = pred_bboxes_seq[-1]
        if (
            self._last_quality_logits is not None
            and self.quality_calibration_mode == 'solver_coupled'
        ):
            cls_logits_last = calibrate_class_logits(
                cls_logits_last,
                self._last_quality_logits.float(),
                self.quality_score_beta,
            )
        x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)

        return cls_logits_last, pred_bboxes_last, x0

    def _quality_ranking_logits(self, cls_logits):
        """Apply quality only to emitted detection scores.

        In ``final_only`` mode the caller keeps ``cls_logits`` for Top-K,
        renewal, DDIM and GACS, and uses this result only in post-processing.
        """
        if self._last_quality_logits is None:
            return cls_logits
        return calibrate_class_logits(
            cls_logits,
            self._last_quality_logits.float(),
            self.quality_score_beta,
        )

    # ================================================================
    # CCBR: 跨级联框更新 — 前向方法
    # ================================================================

    @torch.no_grad()
    def _forward_at_t_ccbr(
        self, features, x_raw, t, img_metas, prev_head6_scores=None
    ):
        """CCBR 前向: 6 级级联 + 级联间软 renewal

        Args:
            features: 多尺度特征
            x_raw: [bs, N, 4] 扩散空间框
            t: 当前时间步 (float, 0~1)
            img_metas: 图像元数据
            prev_head6_scores: [bs, N] 上一时间步 Head 6 置信度, 或 None

        Returns:
            (cls_logits_last, pred_bboxes_last, x0, inter_cls_logits)
        """
        bs = x_raw.shape[0]
        device = x_raw.device
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        time_emb = self.time_mlp(t_input)

        inter_cls_logits = []
        curr_proposals = None
        final_pred_bboxes = None

        for k, head in enumerate(self.head_series):
            result = head(
                features, curr_bboxes, curr_proposals,
                self.roi_extractor, time_emb,
            )
            if len(result) == 4:
                cls_logits, pred_bboxes, curr_proposals, _ = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
            inter_cls_logits.append(cls_logits)
            final_pred_bboxes = pred_bboxes

            # CCBR: 级联间 renewal (除最后一级外)
            if k < len(self.head_series) - 1 and self.use_ccbr:
                curr_scores = torch.sigmoid(cls_logits).max(-1)[0]  # [bs, N]

                # 融合置信度: 当前级 + 上一时间步 Head 6
                if prev_head6_scores is not None:
                    fused_conf = (
                        self.ccbr_lambda * curr_scores
                        + (1.0 - self.ccbr_lambda) * prev_head6_scores
                    )
                    # 自适应阈值
                    mean_diff = (
                        prev_head6_scores.mean() - curr_scores.mean()
                    ).item()
                    threshold = self._sampler.score_thr * (
                        1.0 - self.ccbr_beta * mean_diff
                    )
                    threshold = max(min(threshold, 0.5), 0.01)
                else:
                    fused_conf = curr_scores
                    threshold = self._sampler.score_thr

                # 软 renewal
                curr_bboxes = self._sampler.apply_inter_head_renewal(
                    pred_bboxes, fused_conf, threshold,
                    alpha=self.ccbr_alpha,
                    sigma_scale=self.ccbr_sigma,
                )
            else:
                curr_bboxes = pred_bboxes

            # 保持 cascade_detach (CCBR 核心: 不改训练动态)
            if self.cascade_detach:
                curr_bboxes = curr_bboxes.detach()

        cls_logits_last = inter_cls_logits[-1]
        pred_bboxes_last = final_pred_bboxes
        x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)

        return cls_logits_last, pred_bboxes_last, x0, inter_cls_logits

    # ================================================================
    # PCSE: 单次假设采样 (用于多种子集成)
    # ================================================================

    @torch.no_grad()
    def predict_single_hypothesis(
        self, features, img_metas, seed=None, use_ensemble=False, rescale=True
    ):
        """单次独立假设采样 (PCSE 使用)

        Args:
            features: 多尺度特征
            img_metas: 图像元数据
            seed: 随机种子 (None = 不固定)
            use_ensemble: 是否启用时间维 ensemble
            rescale: 是否缩放回原始尺寸

        Returns:
            DetectionResult: 单次采样的检测结果
        """
        device = features[0].device
        bs = len(img_metas)

        if seed is not None:
            torch.manual_seed(seed)

        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        ensemble_results = []
        dpm_solver = self._sampler.create_dpm_solver()
        # 显式校验: 防止 solver_type 拼写错误或未知值时静默降级到 Euler.
        # create_dpm_solver() 仅对 euler/heun 返回 None (合法), 其他 solver_type
        # 必须返回有效实例, 否则下方 if-elif-else 会静默走 Euler 分支.
        if dpm_solver is None:
            assert self.solver_type in ('euler', 'heun', 'ddim'), (
                f"solver_type='{self.solver_type}' 不被 create_dpm_solver() 支持, "
                f"且不属于 euler/heun/ddim. 请检查配置或扩展 create_dpm_solver()."
            )
        if dpm_solver is not None:
            dpm_solver.reset()

        # D3 化解路径 A: 记录上一步被 box_renewal 重置的 proposal mask
        # 在下一步 DPM-Solver++ step() 中传入, 对被 renewal 的 proposal
        # 置零 D1 校正项, 避免 renewal 噪声污染 x0_history 导致 D1 失效
        _renewal_mask: Optional[torch.Tensor] = None  # [bs, N] bool

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
                features, x_raw, t_curr, img_metas
            )
            output_logits = (
                self._quality_ranking_logits(cls_logits)
                if self.quality_calibration_mode == 'final_only'
                else cls_logits
            )

            if use_ensemble:
                ensemble_results.append((output_logits, pred_bboxes))

            if not ensemble_results:
                ensemble_results.append((output_logits, pred_bboxes))

            if self.diffusion_type == 'ddpm':
                curr_bboxes_xyxy, x_raw = self._sampler.ddim_step(
                    t_curr, t_next, x_raw, cls_logits, pred_bboxes,
                    img_metas, self.alphas_cumprod,
                )
                if t_next < 0:
                    break
            else:
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx,
                                             renewal_mask=_renewal_mask)
                elif self.solver_type == 'heun' and t_next > 0:
                    def model_fn(x_tmp, t_tmp):
                        _, _, x0_tmp = self._forward_at_t(
                            features, x_tmp, t_tmp, img_metas
                        )
                        return x0_tmp, None
                    x_raw = self.rf.heun_step(
                        x_raw, x0_raw, t_curr, t_next, model_fn
                    )
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                if self.box_renewal:
                    # 方向4 VGAR: 传入 v_θ 预测的 x0 和当前时间步, 启用速度场引导
                    x_raw = self._sampler.apply_box_renewal(
                        x_raw, cls_logits,
                        x0_pred=x0_raw,
                        t_curr=t_curr,
                    )
                if t_next <= 0:
                    break

        results = self._sampler.post_process(
            ensemble_results, img_metas, rescale
        )

        return results[0]

    # ================================================================
    # PCSE: 多假设采样 + 核型评分选择
    # ================================================================

    @torch.no_grad()
    def _predict_pcse(self, features, img_metas, rescale=True):
        """PCSE 推理: K 次独立假设采样 + 核型评分选择

        利用扩散采样的随机性生成 K 组候选检测假设,
        用核型先验 (配对一致性) 对假设集合评分, 选择最优假设输出。
        """
        device = features[0].device

        # 延迟初始化 KaryotypeScorer
        if self._pcse_scorer is None:
            # 对于 num_classes=24 (24obj), 0-21常染色体, 22=X, 23=Y
            x_class_idx = self.num_classes - 2
            y_class_idx = self.num_classes - 1
            self._pcse_scorer = KaryotypeScorer(
                num_autosome=22,
                x_class_idx=x_class_idx,
                y_class_idx=y_class_idx,
                target_count=46,
                count_sigma=2.0,
                alphas=self.pcse_alphas,
                lam=self.pcse_lambda,
            )

        # 生成 K 个独立假设
        base_seed = 2024
        hypotheses = []
        for k in range(self.pcse_k):
            hyp = self.predict_single_hypothesis(
                features, img_metas,
                seed=base_seed + k,
                use_ensemble=self.use_ensemble,
                rescale=rescale,
            )
            hypotheses.append(hyp)

        # 评分选择
        best_hyp = pcse_select(hypotheses, self._pcse_scorer)

        # PCSE 插桩: 记录评分分布
        try:
            scores = [self._pcse_scorer.score(h.bboxes, h.scores, h.labels)
                      for h in hypotheses]
            import swanlab
            swanlab.log({
                'inference/pcse_k': len(hypotheses),
                'inference/pcse_best_score': max(scores),
                'inference/pcse_mean_score': sum(scores) / len(scores),
                'inference/pcse_score_std': float(
                    torch.tensor(scores).std()
                ),
            })
        except Exception as e:
            logger.warning(f'[SwanLab] PCSE metrics log failed: {e}')

        return [best_hyp]

    # ================================================================
    # CCBR: 跨级联框更新推理
    # ================================================================

    @torch.no_grad()
    def _predict_ccbr(self, features, img_metas, rescale=True, return_trajectory=False):
        """CCBR 推理: 级联间 renewal + 后向置信度传播

        增强现有 box_renewal 机制, 在 6 级级联 Head 之间建立跨级信息共享。
        保持 cascade_detach=True, 仅在推理时共享信息。
        """
        device = features[0].device
        bs = len(img_metas)
        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        # CCBR: 缓存上一时间步 Head 6 的置信度
        prev_head6_scores = None  # [bs, N] or None (第一步)

        ensemble_results = []
        trajectory = []
        dpm_solver = self._sampler.create_dpm_solver()
        # 显式校验: 防止 solver_type 拼写错误或未知值时静默降级到 Euler.
        # create_dpm_solver() 仅对 euler/heun 返回 None (合法), 其他 solver_type
        # 必须返回有效实例, 否则下方 if-elif-else 会静默走 Euler 分支.
        if dpm_solver is None:
            assert self.solver_type in ('euler', 'heun', 'ddim'), (
                f"solver_type='{self.solver_type}' 不被 create_dpm_solver() 支持, "
                f"且不属于 euler/heun/ddim. 请检查配置或扩展 create_dpm_solver()."
            )
        if dpm_solver is not None:
            dpm_solver.reset()

        # D3 化解路径 A: 记录上一步被 box_renewal 重置的 proposal mask
        # 在下一步 DPM-Solver++ step() 中传入, 对被 renewal 的 proposal
        # 置零 D1 校正项, 避免 renewal 噪声污染 x0_history 导致 D1 失效
        _renewal_mask: Optional[torch.Tensor] = None  # [bs, N] bool

        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            # CCBR 前向 (带级联间 renewal)
            cls_logits, pred_bboxes, x0_raw, _ = self._forward_at_t_ccbr(
                features, x_raw, t_curr, img_metas, prev_head6_scores
            )

            # 缓存当前时间步 Head 6 的置信度, 供下一时间步使用
            prev_head6_scores = torch.sigmoid(cls_logits).max(-1)[0]  # [bs, N]

            if return_trajectory:
                trajectory.append((cls_logits.detach(), pred_bboxes.detach()))
            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            if not ensemble_results:
                ensemble_results.append((cls_logits, pred_bboxes))

            # 求解器步进 (不变)
            if self.diffusion_type == 'ddpm':
                curr_bboxes_xyxy, x_raw = self._sampler.ddim_step(
                    t_curr, t_next, x_raw, cls_logits, pred_bboxes,
                    img_metas, self.alphas_cumprod,
                )
                if t_next < 0:
                    break
            else:
                if dpm_solver is not None:
                    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx,
                                             renewal_mask=_renewal_mask)
                elif self.solver_type == 'heun' and t_next > 0:
                    def model_fn(x_tmp, t_tmp):
                        # Heun 第二阶也用 CCBR 前向
                        _, _, x0_tmp, _ = self._forward_at_t_ccbr(
                            features, x_tmp, t_tmp, img_metas, prev_head6_scores
                        )
                        return x0_tmp, None
                    x_raw = self.rf.heun_step(
                        x_raw, x0_raw, t_curr, t_next, model_fn
                    )
                else:
                    x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

                # 时间步间 box_renewal (现有, 保留)
                if self.box_renewal:
                    # 方向4 VGAR: 传入 v_θ 预测的 x0 和当前时间步, 启用速度场引导
                    x_raw = self._sampler.apply_box_renewal(
                        x_raw, cls_logits,
                        x0_pred=x0_raw,
                        t_curr=t_curr,
                    )
                if t_next <= 0:
                    break

        results = self._sampler.post_process(
            ensemble_results, img_metas, rescale
        )

        if return_trajectory:
            return results, trajectory
        return results

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
                ModelOutput(
                    pred_logits=all_cls_logits[i],
                    pred_boxes=norm_pred_bboxes[i],
                )
                for i in range(all_cls_logits.shape[0] - 1)
            ]
        return ModelOutput(
            pred_logits=main_logits,
            pred_boxes=main_bboxes,
            aux_outputs=aux_outputs,
        )

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
