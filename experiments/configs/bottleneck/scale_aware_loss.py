"""瓶颈消融实验 — 尺度感知 L1 损失

实验目标: 验证框回归损失是否是定位瓶颈
假设: 染色体框尺度差异大, 标准 L1 损失对大框偏向严重.
      若尺度感知 L1 (按面积倒数加权) 提升 AP75/AP90, 说明损失函数是定位瓶颈.

对比: rf_heun_adaln.py (baseline, L1 loss)
依据: criterion.py 中 scale_aware 模式
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            scale_aware=True,
            scale_aware_mode='inverse',  # 按面积倒数加权
            scale_aware_min_weight=0.5,
            scale_aware_max_weight=3.0,
            scale_aware_alpha=0.15,
            scale_aware_giou=True,  # GIoU 也按尺度加权
        ),
    ),
)
