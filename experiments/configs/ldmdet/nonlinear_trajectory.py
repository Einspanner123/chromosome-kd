"""LDMDet + Nonlinear Trajectory (方向四: 非线性轨迹)

继承自 rf_heun_adaln, 启用尺度条件化 RF + OT Flow 耦合.
与 rf_heun_adaln.py (baseline) 差异: scale_conditioned_rf + coupling=ot_flow.

实验目标: 验证非线性轨迹对多尺度染色体生成的改善
- 假设: 尺度条件化调度让小目标 (A组) 在更早 t 去噪, 提升小目标 recall
- 假设: OT Flow 耦合减少路径交叉, 加速收敛
- 对比: rf_heun_adaln.py (baseline, 线性轨迹 + 随机耦合)

开关控制:
- scale_conditioned_rf: None=标准 RF, dict=尺度条件化 RF
- coupling: 'random'=随机耦合 (baseline), 'ot_flow'=OT Flow 耦合

消融实验:
- 仅尺度条件化: coupling 注释掉 (用 random), 保留 scale_conditioned_rf
- 仅 OT Flow: scale_conditioned_rf 注释掉, coupling=ot_flow
- 两者联合: 都启用 (推荐, 本配置默认)
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        # 方向四: 尺度条件化 RF (小目标用更陡的噪声调度)
        scale_conditioned_rf=dict(
            type='ScaleConditionedRF',
            lambda_mod=0.5,      # 调制强度 (0=标准 RF, 越大尺度差异越显著)
            s_max=0.15,          # 参考最大尺度 (归一化面积平方根)
            snr_scale=2.0,       # 与 RectifiedFlow 一致
        ),
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
    dict(type='TrajectoryDiagInjector', priority='NORMAL', interval=100),
]
