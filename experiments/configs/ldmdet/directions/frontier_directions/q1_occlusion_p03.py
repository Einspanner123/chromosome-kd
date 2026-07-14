"""方向 Q1: 训练时 RoI 特征遮挡模拟 (Amodal Trajectory Diffusion)

目的: 在 a3_full_sota 基础上, 对 RoI 特征施加随机矩形零填充遮挡,
强制模型学习从部分观测恢复完整框 (amodal completion), 攻克染色体重叠漏检。

组件:
  + single_head.occlusion_prob=0.3 (30% RoI 被遮挡)
  + single_head.occlusion_ratio_range=(0.2, 0.6) (遮挡面积 20%-60%)

对照:
  - a3_full_sota (无遮挡) → 验证遮挡模拟的增益

理论: docs/research/frontier_directions/方向Q_ATD_Amodal轨迹扩散.md §3.1

SwanLab: 项目 'ldmdet-frontier-directions-24obj', 实验 'q1_occlusion_p03'
"""
_base_ = ['../mainline_ablation_24obj/a3_full_sota_24obj.py']

# Q1: RoI 特征遮挡模拟
model = dict(
    bbox_head=dict(
        single_head=dict(
            occlusion_prob=0.3,
            occlusion_ratio_range=(0.2, 0.6),
        ),
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-frontier-directions-24obj',
            experiment_name='q1_occlusion_p03',
            description='方向Q1: 训练时RoI特征遮挡模拟 (prob=0.3, ratio=0.2-0.6) | 基于 a3_full_sota | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
