"""CAT velocity_curvature: Curvature-Aware Training (纯曲率模式)

在 baseline 基础上添加 CAT loss (velocity_curvature 模式)。
惩罚 |v(t+dt) - v(t)|^2，仅约束曲率，不约束速度大小。
比 x0_consistency 更温和的正则化。
"""
_base_ = ['./ablation_baseline.py']

model = dict(
    bbox_head=dict(
        use_cat=True,
        cat_delta_t=0.02,
        cat_loss_type='velocity_curvature',
        cat_weight=1.0,
    ),
)
