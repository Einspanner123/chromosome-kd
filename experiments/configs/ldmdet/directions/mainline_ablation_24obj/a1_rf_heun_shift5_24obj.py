"""24obj A1 RF+Heun shifted schedule 参数扫描: shift=5.0

目的: 验证 shift=3.0 是否为最优值, 更大 shift 是否对小目标有帮助
对照: shift=1 (unshifted) / shift=2 / shift=3 (A1, 0.856) / shift=5 (本实验)
公式: times = s * times / (1 + (s - 1) * times), s=rf_shift

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a1_rf_heun_shift5'
"""
_base_ = ['./a1_rf_heun_24obj.py']

# === 覆盖: shift=5.0 ===
model = dict(
    bbox_head=dict(
        rf_shift=5.0,
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
            experiment_name='a1_rf_heun_shift5',
            description='24obj A1 RF+Heun shift=5.0 | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
