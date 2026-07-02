"""SOTA 叠加实验: focal_gamma=3 (Stochastic OT ε=5 + Focal Loss γ=3)

目标: 验证 focal_gamma=3 能否突破 SOTA 0.753 (单次试探, 通过后再补 3-seed)

基于 SOTA 原始训练配方 (bs2, lr=5e-5, no flash_attn):
  - coupling: ot_flow (sinkhorn, ε=5.0, multinomial)  ← 等价旧 sinkhorn_stochastic
  - focal_gamma: 2.0 → 3.0  ← 本实验唯一变量
  - 其余: RF + Heun + Shifted + AdaLN-Zero, 与 SOTA 完全一致

SwanLab: 新项目 'ldmdet-sota-stack' (与主线 ablation 分离)
对照: reproduce_0751_stochot_eps5_v2 (SOTA 0.753, bs2, gamma=2)
"""
_base_ = ['../ldmdet_rf_heun_shifted_bs2.py']

model = dict(
    bbox_head=dict(
        single_head=dict(time_conditioning='adaln_zero'),
        coupling=dict(
            type='ot_flow',
            epsilon=5.0,
            num_iters=20,
            coupling_mode='multinomial',
        ),
        criterion=dict(
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0, gamma=3.0),
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0, gamma=3.0),
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                ],
            ),
        ),
    ),
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=30,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]

# SwanLab 新项目: SOTA 叠加实验专用
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-sota-stack',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
