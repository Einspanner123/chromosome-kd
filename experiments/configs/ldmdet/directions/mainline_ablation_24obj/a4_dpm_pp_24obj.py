"""24obj 主路线消融实验 A4: DPM-Solver++ (验证采样器效果)

目的: 在 A3 完整 SOTA 基础上, 将 Heun 替换为 DPM-Solver++
组件:
  + solver_type='dpm_solver_pp' (DPM-Solver++ 高阶 ODE 求解器)
  + sampling_timesteps=4 (与 A3 Heun 4步保持一致, 公平对比采样器)
  - RF + AdaLN + StochOT eps5 保持不变

对照: A3 (Heun 4步) → 验证 DPM-Solver++ vs Heun 的采样器效果
理论: DPM-Solver++ 是高阶 ODE 求解器, 在相同步数下应有更小的截断误差

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a4_dpm_pp'
"""
_base_ = ['./a3_full_sota_24obj.py']

# === 替换采样器为 DPM-Solver++ ===
model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp',
        sampling_timesteps=4,
    ),
)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a4_dpm_pp',
            description='24obj 主路线消融 A4: DPM-Solver++ 4步 (vs A3 Heun 4步) | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
