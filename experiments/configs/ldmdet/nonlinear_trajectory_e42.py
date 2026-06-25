"""LDMDet + Nonlinear Trajectory E4.2: 仅 OT Flow 耦合

消融实验: 关闭尺度条件化 (lambda_mod=0), 仅保留 OT Flow 耦合.
验证 OT Flow 耦合的独立价值.

对比:
- E4.0 (rf_heun_adaln): 线性 RF + random coupling → baseline
- E4.2 (本配置): 线性 RF + OT Flow coupling → 验证 OT 独立价值
- E4.3 (nonlinear_trajectory): 尺度条件化 RF + OT Flow → 联合
"""

_base_ = ['nonlinear_trajectory.py']

model = dict(
    bbox_head=dict(
        # 关闭尺度条件化: lambda_mod=0 退化为标准线性 RF
        scale_conditioned_rf=dict(
            type='ScaleConditionedRF',
            lambda_mod=0.0,      # 无调制, κ(s)=1 对所有尺度
            s_max=0.15,
            snr_scale=2.0,
        ),
        # OT Flow 耦合保持不变 (epsilon=1.0, argmax)
    ),
)
