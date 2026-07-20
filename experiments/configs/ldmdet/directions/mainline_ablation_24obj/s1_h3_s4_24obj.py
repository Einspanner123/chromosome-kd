"""S1 cascade head × solver step 解耦消融: H=3 S=4 (12 NFE)

理论依据: theory_analysis_RF_DPM.md §2 (S1)
目的: 验证 H × S 的可交换性边界 (命题 S1.3)
设置: H=3, S=4 → 12 NFE (DPM-Solver++ 1 NFE/step)
对照: A4 baseline (H=6, S=4 → 24 NFE) + s1_h6_s2 (H=6, S=2 → 12 NFE)
预期: 若 H×S 不可交换 (命题 S1.3 成立), H=3 S=4 mAP 应不同于 H=6 S=2

基线: a4_dpm_pp_24obj.py (DPM-Solver++ + RF + AdaLN + StochOT eps5)
变化: 仅 num_heads 6 → 3
"""
_base_ = ['./a4_dpm_pp_24obj.py']

model = dict(
    bbox_head=dict(
        num_heads=3,  # 6 → 3 (减半)
        sampling_timesteps=4,  # 保持 S=4
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-s1-cascade-decouple',
            experiment_name='s1_h3_s4',
            description='S1: H=3 S=4 (12 NFE) | 验证 H×S 可交换性边界 | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
