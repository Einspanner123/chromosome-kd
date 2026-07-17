"""24obj A0 baseline 多种子训练配置 (DeepSeek Major Concern 2 补充实验)

用途: A0 baseline 多种子实验 (seed 123, 789), 用于统计显著性分析
基础: a0_baseline_24obj.py, 移除 experiment_name 以便 train.py 自动添加 _seed{N} 后缀
注意: seed 42 已有结果 (a0_baseline, mAP=0.774), 此配置用于补充 seed 123/789

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验名由 train.py 自动生成
"""
_base_ = ['./a0_baseline_24obj.py']

# === 覆盖 EarlyStopping patience (30→15, 加速消融实验) ===
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=15,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]

# 覆盖 SwanLab: 移除 experiment_name, train.py 会自动添加 _seed{N} 后缀
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            description='24obj A0 baseline 多种子 | bs=4, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
