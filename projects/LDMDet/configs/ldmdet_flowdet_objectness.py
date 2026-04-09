_base_ = ["./ldmdet_flowdet_adaln.py"]

# Phase 3B 消融: AdaLN-Zero + Objectness 头 (独立验证)
model = dict(
    bbox_head=dict(
        single_head=dict(
            use_objectness=True,
        ),
        criterion=dict(
            loss_objectness_weight=1.0,
        ),
    ),
)
