"""Velocity Loss only: 辅助速度场拟合监督

在 baseline 基础上添加 velocity loss。
匹配 RF 目标速度 v* = x_noise - x_start。
"""
_base_ = ['./ablation_baseline.py']

model = dict(
    bbox_head=dict(
        use_velocity_loss=True,
        velocity_loss_weight=1.0,
    ),
)
