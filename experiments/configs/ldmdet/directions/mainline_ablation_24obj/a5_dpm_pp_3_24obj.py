"""24obj 方向 D 对照: DPM-Solver++ 3 阶 (全程 3 阶)

目的: 与 +DPM-Solver++ (2 阶) / 自适应阶次 DPM-Solver++ 对比, 验证 3 阶是否带来精度增益
组件:
  + solver_type='dpm_solver_pp_3' (RFDPMSolverMultistep, solver_order=3)
  + sampling_timesteps=4 (与 +DPM-Solver++ 一致)

无需重训练: 基于 +DPM-Solver++ checkpoint 直接推理

对照:
  - +DPM-Solver++ (DPM-Solver++ 2阶 4步)
  - 自适应阶次 DPM-Solver++ (前2步3阶 + 后2步2阶)

SwanLab: 项目 'ldmdet-inference', 实验 'a5_dpm_pp_3'
"""
_base_ = ['./a4_dpm_pp_24obj.py']

# === DPM-Solver++ 3 阶 ===
model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp_3',
        sampling_timesteps=4,
    ),
)

# === SwanLab (推理独立项目) ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference',
            experiment_name='a5_dpm_pp_3',
            description='24obj 方向 D 对照: DPM-Solver++ 3阶 4步 (全程3阶) | 推理时改动',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
