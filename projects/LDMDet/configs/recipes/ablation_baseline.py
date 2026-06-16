"""Baseline: RF + Heun + AdaLN-Zero + Stochastic OT (eps=5)

当前 Chromosome20240904 数据集 SOTA 配置。
4-seed mean mAP ≈ 0.746 ± 0.008

此配置作为 TRD/CAT/Velocity 消融实验的对照组。
"""
_base_ = ['../ldmdet_rf_heun_shifted_bs8.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning='adaln_zero',
        ),
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
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
