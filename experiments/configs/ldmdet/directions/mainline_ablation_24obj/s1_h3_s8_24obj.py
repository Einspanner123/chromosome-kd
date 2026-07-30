"""S1 cascade head × solver step 解耦消融: H=3 S=8 (24 NFE)

理论依据: theory_analysis_RF_DPM.md §2 (S1)
目的: 验证 H × S 的可交换性边界 (命题 S1.3)
设置: H=3, S=8 → 24 NFE (DPM-Solver++ 1 NFE/step), 与 +DPM-Solver++ baseline 同 NFE
对照: +DPM-Solver++ baseline (H=6, S=4 → 24 NFE) + s1_h3_s4 (H=3, S=4 → 12 NFE)
预期: 若 H×S 不可交换, 即使 NFE 相同, H=3 S=8 mAP 应低于 H=6 S=4
      (因 A_t 是二阶 solver, B_{t,k} 是一阶精化, H 减半需 S 增加多于两倍)

基线: a4_dpm_pp_24obj.py (DPM-Solver++ + RF + AdaLN + StochOT eps5)
变化: num_heads 6 → 3, sampling_timesteps 4 → 8
"""
_base_ = ['./a4_dpm_pp_24obj.py']

model = dict(
    bbox_head=dict(
        num_heads=3,  # 6 → 3 (减半)
        sampling_timesteps=8,  # 4 → 8 (加倍)
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-s1-cascade-decouple',
            experiment_name='s1_h3_s8',
            description='S1: H=3 S=8 (24 NFE, matched +DPM-Solver++) | 验证 H×S 可交换性 | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
