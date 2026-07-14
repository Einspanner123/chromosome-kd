"""旧数据集 StochOT ε=5 多种子训练配置

用途: 旧数据集 StochOT 多种子实验 (seed 42, 123, 789), 用于统计显著性分析
基础: stochot_eps5_old.py, 移除 experiment_name 以便 train.py 自动添加 _seed{N} 后缀
注意: 旧 seed 42 使用 sinkhorn_stochastic (旧 API), 新运行使用 ot_flow (新 API)
      算法等价但实现不同, 建议重新运行 seed 42 以保证一致性

SwanLab: 项目 'ldmdet-mainline-ablation-old', 实验名由 train.py 自动生成
"""
_base_ = ['./stochot_eps5_old.py']

# 覆盖 SwanLab: 移除 experiment_name, train.py 会自动添加 _seed{N} 后缀
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-old',
            description='旧数据集 StochOT ε=5 多种子 | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
