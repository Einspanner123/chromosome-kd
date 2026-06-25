"""瓶颈消融实验 — Focal Loss gamma=3.0 (更关注困难样本)

实验目标: 提高分类困难类 (Y/G22/G21) 的检测性能
依据: 诊断显示 Y/G22/G21 同时是分类最差和定位最差的类.
      gamma=2.0 是默认值, 增大 gamma 让损失更聚焦于困难样本.
假设: 若 gamma=3.0 提升 Y/G22 的 AP, 说明分类困难可通过损失调参缓解;
      若无提升, 说明需要更根本的改进 (如特征增强/数据增强).

对比: rf_heun_adaln.py (baseline, gamma=2.0)
"""

_base_ = ['../ldmdet/rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0, gamma=3.0),
            assigner=dict(
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0, gamma=3.0),
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                ],
            ),
        ),
    ),
)
