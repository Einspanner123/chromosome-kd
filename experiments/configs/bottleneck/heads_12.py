"""瓶颈消融实验 — 增加级联 head 数 (6 -> 12)

实验目标: 验证增加 head 级联深度是否能提升性能
假设: 若 12 级比 6 级有提升, 说明级联深度是瓶颈;
      若无提升, 说明 6 级已足够.

对比: rf_heun_adaln.py (baseline, num_heads=6)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        num_heads=12,  # 增加级联深度
    ),
)
