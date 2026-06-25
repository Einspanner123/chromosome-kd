"""瓶颈消融实验 — 增加 proposals (500 -> 1000)

实验目标: 验证增加 proposal 数量是否能提升性能
假设: 若 1000 proposals 比 500 有提升, 说明 proposal 容量是瓶颈;
      若无提升, 说明 500 已足够, 瓶颈在别处.

对比: rf_heun_adaln.py (baseline, num_proposals=500)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        num_proposals=1000,  # 增加 proposal 数量
    ),
)
