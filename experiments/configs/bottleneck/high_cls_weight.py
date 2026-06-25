"""瓶颈消融实验 — 高 cls 权重 (loss_cls 2.0 -> 4.0)

实验目标: 验证加大分类损失权重是否能改善分类困难类
依据: 诊断显示 IoU=0.5 下 Cls 误差占 43.4%, 是低 IoU 下的主要瓶颈.
假设: 若加大 cls 权重提升整体 AP50 和 Y/G22 的 AP, 说明分类权重不足;
      若无提升或下降, 说明分类能力受限于特征表达, 而非损失权重.

对比: rf_heun_adaln.py (baseline, loss_cls weight=2.0)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=4.0),  # 2.0 -> 4.0
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=4.0),  # 2.0 -> 4.0
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                ],
            ),
        ),
    ),
)
