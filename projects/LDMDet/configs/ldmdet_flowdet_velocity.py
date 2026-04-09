_base_ = ["./ldmdet_flowdet_adaln.py"]

# Phase 4 消融: AdaLN-Zero + Velocity 辅助 loss (独立验证)
# velocity_head 仅作为辅助分支，预测扩散空间速度 v = x_0 - noise
# bbox 预测仍用 delta regression (不改变推理逻辑)
model = dict(
    bbox_head=dict(
        prediction_mode="velocity",
        single_head=dict(
            prediction_mode="velocity",
        ),
    ),
)
