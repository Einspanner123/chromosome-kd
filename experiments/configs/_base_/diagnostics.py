"""训练诊断基础配置 — experiments/configs/_base_/diagnostics.py

提供权重/梯度/激活监控、数值健康度检查、训练动态记录.
本文件不作为 _base_ 继承 (避免 custom_hooks 列表覆盖冲突),
而是作为参考, 各方向配置直接在 custom_hooks 中追加诊断 Hook.

监控指标 (上传到 SwanLab):
- weights/{group}/{mean,std,norm,dead_ratio,has_nan,has_inf}: 权重统计
- grads/{group}/{mean,std,norm,zero_ratio,has_nan,has_inf}: 梯度统计
- acts/{group}/{mean,std,dead_ratio,saturation}: 激活统计
- dynamics/{loss_cls,loss_bbox,loss_giou,loss_total}: 损失分解
- coupling/{entropy,std_per_gt,mean_per_gt}: 方向一耦合诊断 (若启用)
- count/{mae,mse,bias,accuracy_2}: 方向二计数诊断 (若启用)
- snr/{weight_mean,weight_std,weight_t_correlation}: 方向三 SNR 诊断 (若启用)

采样频率:
- 权重/梯度: 每 100 iter
- 激活: 每 500 iter
- 损失分解: 每 iter

使用方式 (各方向配置):
    custom_hooks = [
        # baseline hooks (从 rf_heun_adaln 继承, 需手动写出)
        dict(type='EarlyStoppingHook', ...),
        dict(type='CopyProjectHook', ...),
        # 诊断 hooks
        dict(type='TrainingDiagnosticsHook', ...),
        dict(type='CouplingDiagInjector', ...),  # 方向一
    ]
"""

# 诊断 Hook 模板 (各方向配置复制到 custom_hooks)
DIAGNOSTIC_HOOKS = [
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
