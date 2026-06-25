"""瓶颈消融实验 — 关闭 ensemble

实验目标: 验证多步采样集成 (use_ensemble) 对检测性能的贡献
假设: 若关闭 ensemble 后 mAP 下降, 说明多步集成是关键;
      若不变, 说明单步预测已足够, ensemble 不是瓶颈.

对比: rf_heun_adaln.py (baseline, use_ensemble=True, sampling_timesteps=4)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        use_ensemble=False,  # 关闭多步集成
    ),
)
