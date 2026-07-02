"""LDMDet + Nonlinear Trajectory E6.1-EMA: EMA + weight_decay 调优

方向六优化器/训练策略改进实验 1: 在方向四最佳配置 (E4.3, mAP=0.752) 基础上,
添加 EMA 权重平滑 + 增大 weight_decay, 缓解训练后期波动 (±1.0%) 与过拟合.

改进点:
- EMA Hook: ExpMomentumEMA, momentum=0.0001, update_buffers=True
  平滑权重, 减少测试时权重抖动, 提升泛化
- weight_decay: 1e-4 → 5e-4 (5x)
  增强正则化, 抑制过拟合 (训练后期 train loss 下降但 val mAP 停滞)

对比:
- E4.3 baseline (nonlinear_trajectory.py): mAP=0.752, 后期波动 ±1.0%
- E6.1-EMA (本配置): 预期 mAP↑, 波动↓
"""

_base_ = ['nonlinear_trajectory.py']

# ── 训练策略改进 ──────────────────────────────────
# 增大 weight_decay: 1e-4 → 5e-4 (抑制过拟合)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=5e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# ── EMA Hook ─────────────────────────────────────
# 覆盖 custom_hooks, 在 baseline + 诊断 hooks 基础上追加 EMA
# EMA 优先级 49 (在 EarlyStopping 50 之前, 确保 EMA 权重用于评估)
custom_hooks = [
    # EMA 权重平滑: 减少训练后期权重抖动
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49,
    ),
    # baseline hooks
    dict(type='EarlyStoppingHook', priority=50, patience=30, min_delta=0.001, monitor='coco/bbox_mAP', rule='greater'),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
    # 诊断 hooks (保持与 baseline 一致)
    dict(
        type='TrainingDiagnosticsHook',
        priority='LOW',
        weight_grad_interval=100,
        activation_interval=500,
        log_weights=True,
        log_grads=True,
        log_activations=True,
        log_numerical_health=True,
        log_loss_breakdown=True,
        module_prefixes=dict(
            backbone='backbone',
            neck='neck',
            bbox_head_head_series='head',
            bbox_head_time_mlp='time_mlp',
            bbox_head_criterion='criterion',
        ),
        activation_layers=[],
        diagnostics_callback=None,
    ),
]
