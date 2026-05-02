_base_ = ["../ldmdet_flowdet_adaln.py"]

# Scale-Conditioned Flow Matching (完整版): 噪声缩放 + 回归损失加权
model = dict(
    bbox_head=dict(
        scale_conditioned_noise=True,
        scale_noise_power=0.5,
        scale_adaptive_loss=True,
        scale_loss_power=0.5,
    ),
)
