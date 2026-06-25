"""瓶颈消融实验 — DPM-Solver++ 采样器

实验目标: 验证更高阶采样器是否提升性能
假设: DPM-Solver++ (2阶) 比 Heun 更精确, 可能在相同步数下提升 AP.
      若提升明显, 说明采样器精度是瓶颈.

对比: rf_heun_adaln.py (baseline, solver_type='heun', sampling_timesteps=4)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp',  # DPM-Solver++ 2阶
        sampling_timesteps=4,
    ),
)
