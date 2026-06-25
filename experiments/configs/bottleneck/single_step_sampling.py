"""瓶颈消融实验 — 单步采样 (sampling_timesteps=1)

实验目标: 验证单步采样是否足够
假设: RF (Rectified Flow) 理论上支持单步采样.
      若单步性能接近 4 步, 说明采样步数不是瓶颈, 可大幅加速;
      若大幅下降, 说明多步采样是必要的.

对比: rf_heun_adaln.py (baseline, sampling_timesteps=4)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        sampling_timesteps=1,  # 单步采样
        solver_type='euler',   # 单步用 Euler 即可
    ),
)
