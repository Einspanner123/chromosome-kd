_base_ = ["../ldmdet_flowdet_adaln.py"]

# 尺度自适应损失加权: 小染色体 → 更高的回归损失权重 → 更多梯度分配给定位
# loss_weight = (1/relative_size)^power, 归一化到 max=1.0
model = dict(
    bbox_head=dict(
        scale_adaptive_loss=True,
        scale_loss_power=0.5,
    ),
)
