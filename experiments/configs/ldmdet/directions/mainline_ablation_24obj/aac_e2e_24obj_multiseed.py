"""24obj AAC Phase 2 端到端多种子训练配置 (统计显著性补充)

用途: AAC 端到端多种子实验 (seed 123, 789), 用于 AAC vs baseline (a1_rf_heun)
      端到端对比的统计显著性分析 (3-seed: 42/123/789)
基础: aac_e2e_24obj.py, 移除 experiment_name 以便 train.py 自动添加 _seed{N} 后缀
注意: seed 42 已有结果 (aac_e2e_24obj), 此配置用于补充 seed 123/789

机制 (详见 docs/research/proposals/AAC_DESIGN.md §4.3 Phase 2):
  - 3-seed 验证 (42, 123, 789), 与 baseline 多种子直接对比
  - AAC(m=2, β=1.0) 贯穿 150 epoch

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验名由 train.py 自动生成
"""

_base_ = ['./aac_e2e_24obj.py']

# 覆盖 SwanLab: 移除 experiment_name, train.py 会自动添加 _seed{N} 后缀
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            description='24obj AAC Phase 2 端到端多种子 | Anderson(m=2, β=1.0) | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
