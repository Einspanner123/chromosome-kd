"""CAT x0_consistency: Curvature-Aware Training (x0 一致性模式)

在 baseline 基础上添加 CAT loss (x0_consistency 模式)。
惩罚 |x0_pred(t) - x0_pred(t+dt)|^2，同时约束速度大小和曲率。
"""
_base_ = ['./ablation_baseline.py']

model = dict(
    bbox_head=dict(
        use_cat=True,
        cat_delta_t=0.02,
        cat_loss_type='x0_consistency',
        cat_weight=1.0,
    ),
)
