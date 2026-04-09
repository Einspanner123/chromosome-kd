_base_ = ["./ldmdet_flowdet_adaln.py"]

# Phase 2 消融: AdaLN-Zero + Sinkhorn OT (独立验证)
model = dict(
    bbox_head=dict(
        criterion=dict(
            type="PurePyTorchDiffusionDetCriterion",
            num_classes=24,
            assigner=dict(
                type="PurePyTorchSinkhornOTMatcher",
                epsilon=0.1,
                num_iters=50,
                dustbin_cost=10.0,
                match_costs=[
                    dict(type="PurePyTorchFocalLossCost", weight=2.0),
                    dict(type="PurePyTorchBBoxL1Cost", weight=5.0),
                    dict(type="PurePyTorchIoUCost", iou_mode="giou", weight=2.0),
                ],
            ),
            loss_cls=dict(type="PurePyTorchFocalLoss", loss_weight=2.0),
            loss_bbox=dict(type="PurePyTorchL1Loss", loss_weight=5.0),
            loss_giou=dict(type="PurePyTorchGIoULoss", loss_weight=2.0),
        ),
    ),
)
