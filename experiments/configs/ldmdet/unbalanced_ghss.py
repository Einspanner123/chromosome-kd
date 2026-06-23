"""LDMDet + Unbalanced GHSS (方向一: 非平衡最优传输耦合)

继承自 rf_heun_adaln, 切换 coupling 为 unbalanced_ghss.
与 ghss.py (baseline) 唯一差异: coupling type + lambda 参数.

实验目标: 验证非平衡 OT 在 GT 分布不均匀场景下的收益
- 假设: 边缘松弛提升稀疏图 recall, 改善组间均衡
- 对比: ghss.py (mAP=0.753 baseline)
- 消融: lambda_row, lambda_col 网格搜索
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        coupling=dict(
            type='unbalanced_ghss',
            epsilon=5.0,          # 与 baseline 一致
            num_iters=20,         # 与 baseline 一致
            lambda_row=1.0,       # 行边缘松弛 (推荐起点)
            lambda_col=1.0,       # 列边缘松弛 (推荐起点)
        ),
    ),
)

# 方向一诊断: 覆盖 custom_hooks, 追加诊断 Hook
# (mmengine 配置列表是覆盖合并, 需写完整列表)
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
    dict(type='CouplingDiagInjector', priority='NORMAL', interval=100),
]
