"""Full: TRD + CAT (velocity_curvature) + Velocity Loss

在 baseline 基础上启用全部辅助 loss。
TRD: 速度分解 (解析传输速度)
CAT: 纯曲率正则化
Velocity: 辅助速度场监督
"""
_base_ = ['./ablation_baseline.py']

model = dict(
    bbox_head=dict(
        use_trd=True,
        trd_delta_t=0.05,
        trd_use_analytic_v=True,
        trd_weight=1.0,
        use_cat=True,
        cat_delta_t=0.02,
        cat_loss_type='velocity_curvature',
        cat_weight=1.0,
        use_velocity_loss=True,
        velocity_loss_weight=1.0,
    ),
)
