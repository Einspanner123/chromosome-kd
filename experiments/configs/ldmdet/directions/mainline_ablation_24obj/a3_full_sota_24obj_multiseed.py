"""24obj +Stoch. Coupling (LINEAGE) / +AdaLN-Zero (paper) RF+Heun+AdaLN+StochOT 多种子训练配置

用途: 论文 Table 5 主消融 +AdaLN-Zero (RF+Heun+StochOT) 多种子实验 (seed 123, 789)
       用于 DDPM baseline→RF+Heun→+AdaLN-Zero→+Stoch. Coupling 主消融的统计显著性分析
基础: a3_full_sota_24obj.py, 移除 experiment_name 以便 train.py 自动添加 _seed{N} 后缀
注意: seed 42 已有结果 (a3_full_sota, mAP=0.858), 此配置用于补充 seed 123/789

论文命名对应 (见 EXPERIMENT_CATALOG.md §C4):
  LINEAGE +Stoch. Coupling (a3_full_sota) = paper Table 5 +AdaLN-Zero (+ Stochastic Coupling)
  LINEAGE +DPM-Solver++ (a4_dpm_pp)    = paper Table 5 +Stoch. Coupling (DPM-Solver++)

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验名由 train.py 自动生成
"""
_base_ = ['./a3_full_sota_24obj.py']

# === EarlyStopping patience=30 (与 RF+Heun/+DPM-Solver++ multiseed 一致) ===
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

# 覆盖 SwanLab: 移除 experiment_name, train.py 会自动添加 _seed{N} 后缀
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            description='24obj +Stoch. Coupling (paper +AdaLN-Zero) RF+Heun+AdaLN+StochOT 多种子 | bs=4, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
