"""24obj RF+Heun unshifted schedule (DeepSeek Major Concern 1 补充实验)

目的: 消融 shifted schedule 的贡献, 将 RF straight-line ODE 与 shifted schedule 解耦
基础: a1_rf_heun_24obj.py, 仅将 rf_schedule='shifted' → 'linear'
对照: RF+Heun (shifted, mAP=0.856) vs RF+Heun-unshifted (linear, 待训练)
       差值 = shifted schedule 的贡献

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a1_rf_heun_unshifted'
"""
_base_ = ['./a1_rf_heun_24obj.py']

# === 覆盖: 使用线性 schedule (无 shift) ===
model = dict(
    bbox_head=dict(
        rf_schedule='linear',  # shifted → linear
        rf_shift=1.0,          # 线性等价
    )
)

# === EarlyStopping patience=30 (reverted from 15: late burst risk) ===
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

# === 覆盖 SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a1_rf_heun_unshifted',
            description='24obj RF+Heun unshifted schedule (DeepSeek Concern 1) | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
