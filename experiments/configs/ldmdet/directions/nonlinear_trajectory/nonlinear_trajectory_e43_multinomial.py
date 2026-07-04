"""LDMDet + Nonlinear Trajectory E4.3-tune: multinomial 耦合模式

调参实验: 将 OT 耦合从 argmax (确定性) 改为 multinomial (随机采样),
每个 proposal 按传输矩阵概率从多个 GT 中采样, 减少训练波动.

对比:
- E4.3 (nonlinear_trajectory): ε=1.0, argmax → mAP=0.752, 波动 ±4.5%
- E4.3-tune multinomial (本配置): ε=1.0, multinomial → 预期波动减小

理论: argmax 对 mini-batch 噪声敏感 (一个 epoch 内配对跳变);
      multinomial 允许软配对, 期望路径更平滑.
"""

_base_ = ['nonlinear_trajectory.py']

model = dict(
    bbox_head=dict(
        coupling=dict(
            type='ot_flow',
            epsilon=1.0,
            num_iters=10,
            coupling_mode='multinomial',  # argmax → multinomial
        ),
    ),
)
