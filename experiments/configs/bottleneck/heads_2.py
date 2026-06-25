"""瓶颈消融实验 — 减少级联 head 数 (6 -> 2)

实验目标: 验证 head 级联深度对性能的影响
假设: 若 2 级 head 性能接近 6 级, 说明级联深度不是瓶颈, 可减少计算量;
      若大幅下降, 说明深度级联是关键能力.

对比: rf_heun_adaln.py (baseline, num_heads=6)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        num_heads=2,  # 减少级联深度
    ),
)
