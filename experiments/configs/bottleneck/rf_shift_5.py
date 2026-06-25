"""瓶颈消融实验 — rf_shift=5

实验目标: 验证更大的 rf_shift 是否改善小目标检测
假设: 更大的 shift 给高噪声区域更多时间步, 有助于困难样本.

对比: rf_heun_adaln.py (baseline, rf_shift=3.0)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        rf_shift=5.0,
    ),
)
