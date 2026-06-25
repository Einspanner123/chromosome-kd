"""瓶颈消融实验 — 关闭 box_renewal

实验目标: 验证 box_renewal (低置信度框替换为随机噪声) 对检测性能的贡献
假设: 若关闭 box_renewal 后 mAP 大幅下降, 说明框更新机制是关键能力;
      若变化不大, 说明 box_renewal 不是瓶颈.

对比: rf_heun_adaln.py (baseline, box_renewal=True)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        box_renewal=False,  # 关闭框更新
    ),
)
