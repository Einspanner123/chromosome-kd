"""TRD only: Transport-Refinement Decomposition

在 baseline 基础上添加 TRD loss。
使用解析传输速度 (v* = x_1 - x_0)，独立 trd_delta_t=0.05。
"""
_base_ = ['./ablation_baseline.py']

model = dict(
    bbox_head=dict(
        use_trd=True,
        trd_delta_t=0.05,
        trd_use_analytic_v=True,
        trd_weight=1.0,
    ),
)
