"""TRD + CAT (velocity_curvature): 组合实验

在 baseline 基础上同时启用 TRD 和纯曲率 CAT。
TRD 负责速度分解，CAT 负责曲率正则化，两者互补。
解耦步长: trd_delta_t=0.05, cat_delta_t=0.02。
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
    ),
)
