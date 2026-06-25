"""瓶颈消融实验 — 关闭 deep_supervision

实验目标: 验证深度监督 (6 级 head 级联的中间监督) 对检测性能的贡献
假设: 若关闭 deep_supervision 后 mAP 大幅下降, 说明级联监督是关键;
      若变化不大, 说明单级 head 已足够.

对比: rf_heun_adaln.py (baseline, deep_supervision=True, num_heads=6)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        deep_supervision=False,  # 关闭深度监督
    ),
)
