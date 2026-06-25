"""瓶颈消融实验 — Focal Loss gamma=1.5 (减少对困难样本的过度聚焦)

实验目标: 对比 gamma=1.5 是否比默认 2.0 更好
依据: gamma 过大可能导致模型过度关注极少数困难样本, 忽略整体优化.
假设: 若 gamma=1.5 整体 AP 提升, 说明默认 gamma=2.0 过于激进;
      若下降, 说明当前 gamma=2.0 是合理的.

对比: rf_heun_adaln.py (baseline, gamma=2.0)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0, gamma=1.5),
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0, gamma=1.5),
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                ],
            ),
        ),
    ),
)
