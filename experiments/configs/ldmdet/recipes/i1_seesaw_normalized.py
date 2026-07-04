"""方向I-1: Seesaw Loss + Normalized Classifier (长尾类别平衡)

目标: 验证 Seesaw Loss + Normalized Classifier 能否突破 SOTA 0.746 mAP (chromo)
预期收益: +0.005~0.010 mAP (保守估计)

基于 SOTA 训练配方 (bs2, lr=5e-5, RF+Heun+Shifted+AdaLN-Zero+StochasticOT ε=5):
  - loss_cls: FocalLoss → SeesawLoss (p=0.8, 误分类自校准)  ← 攻击分类主瓶颈 43.4%
  - cls_head: Linear → NormalizedLinear (L2归一化, τ=20)     ← 长尾特征对齐
  - 其余: 与 SOTA 完全一致

设计依据: docs/research/breakthrough_directions/方向I_长尾少样本类别平衡.md

SwanLab: 项目 'chromosome-kd-ablation', 实验 'i1_seesaw_normalized'
对照: SOTA 0.746 mAP (rf_heun_adaln_stochot_eps5)
"""
_base_ = ['../ldmdet_rf_heun_shifted_bs2.py']

num_classes = 24

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning='adaln_zero',
            # Normalized Classifier: L2归一化权重和特征, 温度缩放 τ=20
            use_normalized_classifier=True,
            classifier_temperature=20.0,
        ),
        coupling=dict(
            type='ot_flow',
            epsilon=5.0,
            num_iters=20,
            coupling_mode='multinomial',
        ),
        criterion=dict(
            # SeesawLoss: 头类样本→尾类负类时衰减, 保护尾类 (mmdet 方向)
            # p=0.8 衰减指数, gamma=2.0 Focal聚焦, alpha=0.25 平衡
            loss_cls=dict(
                type='PurePyTorchSeesawLoss',
                num_classes=num_classes,
                p=0.8,
                alpha=0.25,
                gamma=2.0,
                loss_weight=2.0,
            ),
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
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

# SwanLab: 方向I-1 消融实验
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='chromosome-kd-ablation',
            experiment_name='i1_seesaw_normalized',
            description='方向I-1: Seesaw Loss + Normalized Classifier | bs2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
