"""瓶颈消融实验 — 加大 GIoU 损失权重

实验目标: 验证加大 GIoU 损失权重是否能改善定位精度
假设: GIoU 损失直接优化框重叠度, 加大权重可能提升高 IoU 下的 AP.
      若 AP75/AP90 提升, 说明损失权重配置是定位瓶颈.

对比: rf_heun_adaln.py (baseline, loss_giou weight=2.0)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=5.0),
            loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=5.0),  # 2.0 -> 5.0
        ),
    ),
)
