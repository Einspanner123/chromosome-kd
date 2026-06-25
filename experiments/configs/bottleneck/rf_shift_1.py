"""瓶颈消融实验 — 不同 rf_shift 值

实验目标: 验证 RF 噪声调度的 shift 参数对性能的影响
假设: shift 控制高噪声区域的时间分配, 影响小/大目标的去噪质量.
      扫描 shift=[1, 2, 3, 5, 8] 找到最优值.
      若性能随 shift 变化显著, 说明噪声调度是瓶颈.

对比: rf_heun_adaln.py (baseline, rf_shift=3.0)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        rf_shift=1.0,  # 无 shift (标准线性调度)
    ),
)
