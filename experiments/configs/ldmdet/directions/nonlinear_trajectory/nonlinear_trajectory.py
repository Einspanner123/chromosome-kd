"""LDMDet + Nonlinear Trajectory (方向四: 非线性轨迹)

继承自 rf_heun_adaln, 启用 OT Flow 耦合.
与 rf_heun_adaln.py (baseline) 差异: coupling=ot_flow.

实验目标: 验证 OT Flow 耦合对训练收敛的改善
- 假设: OT Flow 耦合减少路径交叉, 加速收敛

注意: ScaleConditionedRF 已证伪 (0.741 < 0.746 baseline), 已移除.
      之前的 0.752 mAP 全部来自 OTFlowCoupling, 与 SCRF 无关.
      详见 docs/EXPERIMENT_LINEAGE.md 第十一节.
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # 方向四: OT Flow 耦合 (mini-batch OT 计算最优配对)
        coupling=dict(
            type='ot_flow',
            epsilon=1.0,         # Sinkhorn 熵正则化
            num_iters=10,        # Sinkhorn 迭代次数
            coupling_mode='argmax',  # 'argmax' (确定性) 或 'multinomial' (随机)
        ),
    ),
)

# 方向四诊断: 覆盖 custom_hooks, 追加诊断 Hook
custom_hooks = [
    # baseline hooks
    dict(type='EarlyStoppingHook', priority=50, patience=30, min_delta=0.001, monitor='coco/bbox_mAP', rule='greater'),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
    # 诊断 hooks
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
