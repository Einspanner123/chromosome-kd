"""瓶颈消融实验 — 多步采样 (sampling_timesteps=8)

实验目标: 验证增加采样步数是否能提升性能
假设: 若 8 步比 4 步有提升, 说明采样精度是瓶颈;
      若无提升, 说明 4 步已收敛, 采样不是瓶颈.

对比: rf_heun_adaln.py (baseline, sampling_timesteps=4)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        sampling_timesteps=8,  # 增加采样步数
    ),
)
