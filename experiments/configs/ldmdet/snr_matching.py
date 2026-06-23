"""LDMDet + SNR-Aware Dynamic Matching (方向三: SNR 感知的动态匹配)

继承自 rf_heun_adaln, 启用 SNR 感知匹配器 + SNR 加权损失.
与 rf_heun_adaln.py (baseline) 差异: matcher 类型 + snr_weighted_loss.

实验目标: 验证 SNR 感知匹配对训练稳定性和最终 mAP 的提升
- 假设: 高噪声时匹配更保守, 减少梯度污染, mAP 提升 0.5-1%
- 对比: rf_heun_adaln.py (baseline, 固定代价匹配)
- 机制: w(t) = (1-t)^2 / ((1-t)^2 + t^2), 高噪声时代价权重低

开关控制:
- matcher.type: 'SNRAwareMatcher' 启用, 'DiffusionDetMatcher' (默认) 关闭
- snr_weighted_loss: True 启用损失加权, False (默认) 关闭
- snr_mode: 'logistic' | 'exponential' | 'none'
- snr_w_min: 权重下界 (避免完全抑制, 推荐 0.1)
- k_max_scale: 是否用 w(t) 缩放 dynamic_k 上界
- cost_threshold: 高噪声保护阈值

消融实验:
- 仅匹配器: snr_weighted_loss=False (验证匹配改善的独立价值)
- 仅损失加权: matcher.type='DiffusionDetMatcher' + snr_weighted_loss=True
- 联合 (推荐, 本配置默认): 两者都启用
"""

_base_ = ['rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            # 方向三: SNR 感知匹配器
            assigner=dict(
                type='SNRAwareMatcher',
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                ],
                center_radius=2.5,
                candidate_topk=5,
                # SNR 参数
                snr_mode='logistic',
                snr_beta=3.0,
                snr_w_min=0.1,
                k_max_scale=True,
                cost_threshold=50.0,
            ),
            # 方向三: SNR 加权损失 (与匹配代价加权保持一致)
            snr_weighted_loss=True,
            snr_mode='logistic',
            snr_beta=3.0,
            snr_w_min=0.1,
        ),
    ),
)

# 方向三诊断: 覆盖 custom_hooks, 追加诊断 Hook
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
    dict(type='SNRDiagInjector', priority='NORMAL', interval=100),
]
