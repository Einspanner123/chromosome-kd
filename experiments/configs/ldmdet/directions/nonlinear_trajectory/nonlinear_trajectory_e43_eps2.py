"""LDMDet + Nonlinear Trajectory E4.3-tune: ε=2.0 调参

调参实验: 增大 Sinkhorn 熵正则化 ε=1.0→2.0, 使传输矩阵更平滑,
减少 argmax 耦合的训练波动 (E4.3 原始训练波动 ±4.5%).

对比:
- E4.3 (nonlinear_trajectory): ε=1.0, argmax → mAP=0.752, 波动大
- E4.3-tune (本配置): ε=2.0, argmax → 预期波动减小, mAP 可能提升
"""

_base_ = ['nonlinear_trajectory.py']

model = dict(
    bbox_head=dict(
        coupling=dict(
            type='ot_flow',
            epsilon=2.0,         # 1.0 → 2.0, 更平滑的传输矩阵
            num_iters=10,
            coupling_mode='argmax',
        ),
    ),
)
