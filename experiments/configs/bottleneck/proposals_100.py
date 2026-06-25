"""瓶颈消融实验 — 减少 proposals (500 -> 100)

实验目标: 验证 proposal 数量对检测性能的影响
假设: 染色体图像平均 ~46 个目标, 500 proposals 可能远超需求.
      若 100 proposals 性能不降, 说明 500 是过度配置, 可加速;
      若性能下降, 说明高密度场景需要大量 proposals.

对比: rf_heun_adaln.py (baseline, num_proposals=500)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        num_proposals=100,  # 减少 proposal 数量
    ),
)
