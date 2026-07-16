"""24obj A1 RF+Heun shifted schedule 参数扫描: shift=2.0

目的: 验证 shift=3.0 是否为最优值
对照: shift=1 (unshifted) / shift=2 (本实验) / shift=3 (A1, 0.856) / shift=5
公式: times = s * times / (1 + (s - 1) * times), s=rf_shift

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a1_rf_heun_shift2'
"""
_base_ = ['./a1_rf_heun_24obj.py']

# === 覆盖: shift=2.0 ===
model = dict(
    bbox_head=dict(
        rf_shift=2.0,
    )
)

# === 覆盖 SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a1_rf_heun_shift2',
            description='24obj A1 RF+Heun shift=2.0 | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
