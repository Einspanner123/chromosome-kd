"""瓶颈消融实验 — DDPM 基线 (对比扩散框架)

实验目标: 对比 RF (Rectified Flow) 和 DDPM 的性能差异
假设: 若 DDPM 性能显著低于 RF, 说明 RF 是正确选择;
      若 DDPM 接近 RF, 说明扩散框架选择不是瓶颈.

对比: rf_heun_adaln.py (baseline, RF + Heun)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        diffusion_type='ddpm',
        solver_type='ddim',
        sampling_timesteps=4,
        rf_schedule='linear',
    ),
)
