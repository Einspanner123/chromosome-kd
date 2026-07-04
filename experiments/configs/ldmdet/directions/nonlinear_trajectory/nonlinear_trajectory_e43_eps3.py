"""LDMDet + Nonlinear Trajectory E4.3-tune: ε=3.0 调参

调参实验: 进一步增大 Sinkhorn 熵正则化 ε=2.0→3.0, 传输矩阵更接近均匀,
减少 argmax 耦合的训练波动. 与 ε=1.0(E4.3) 和 ε=2.0 对比, 寻找最优 ε.

对比:
- E4.3 (nonlinear_trajectory): ε=1.0, argmax → mAP=0.752, 波动 ±4.5%
- E4.3-tune ε=2.0: ε=2.0, argmax → 待测
- E4.3-tune ε=3.0 (本配置): ε=3.0, argmax → 预期波动进一步减小
"""

_base_ = ['nonlinear_trajectory.py']

model = dict(
    bbox_head=dict(
        coupling=dict(
            type='ot_flow',
            epsilon=3.0,         # 1.0 → 3.0, 更平滑
            num_iters=10,
            coupling_mode='argmax',
        ),
    ),
)
