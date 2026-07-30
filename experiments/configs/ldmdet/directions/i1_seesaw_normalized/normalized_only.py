"""方向I-1 变体: 仅 Normalized Classifier (剥离 SeesawLoss)

目的: 隔离 NormalizedClassifier 的单独效果
背景: i1_seesaw_normalized (SeesawLoss+NormalizedClassifier) 未达预期 (+0.005~0.010),
      最佳 mAP=0.7440 vs SOTA 3seed 平均 0.7450 (-0.0010)
诊断: 数据集不平衡比仅 4.69:1 (RF+Heun=3253 vs Y=693), SeesawLoss 过度保护,
      22 个均衡类间互相衰减梯度, 损害类间区分 (如 RF+Heun/+AdaLN-Zero/+Stoch. Coupling 同字母组细分)
方案: 移除 SeesawLoss, 改回 FocalLoss, 只保留 NormalizedClassifier
预期: 验证 NormalizedClassifier 单独效果 (+0.003~0.008 mAP)

对照:
  - SOTA 3seed 平均: 0.7450 (sota_seed42/456/1000)
  - i1_seesaw_normalized: 0.7440 (SeesawLoss+NormalizedClassifier, 已证伪)

SwanLab: 项目 'ldmdet-breakthrough', 实验 'normalized_only'
"""
_base_ = ['../../ldmdet_rf_heun_shifted_bs2.py']

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
        # loss_cls 不覆盖, 继承 base 的 PurePyTorchFocalLoss (loss_weight=2.0)
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

# Val 评估器: 启用 classwise 输出 24 个 per-class AP (项目硬约束)
val_evaluator = dict(classwise=True)

# SwanLab: 方向I-1 变体 (仅 NormalizedClassifier)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-breakthrough',
            experiment_name='normalized_only',
            description='方向I-1 变体: 仅 Normalized Classifier (剥离 SeesawLoss) | bs2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
