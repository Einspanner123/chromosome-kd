_base_ = ["./ldmdet_flowdet_adaln.py"]

# AdaLN-Zero + Objectness 精简组合
# 两个已验证最稳定的组件
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
