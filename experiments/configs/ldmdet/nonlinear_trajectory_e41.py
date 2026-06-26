"""LDMDet + Nonlinear Trajectory E4.1: 仅尺度条件化 RF

消融实验: 保留尺度条件化 (lambda_mod=0.5), 关闭 OT Flow (改用 random coupling).
验证尺度条件化调度的独立价值.

对比:
- E4.0 (rf_heun_adaln): 线性 RF + random coupling → baseline
- E4.1 (本配置): 尺度条件化 RF + random coupling → 验证尺度条件化独立价值
- E4.3 (nonlinear_trajectory): 尺度条件化 RF + OT Flow → 联合
"""

_base_ = ['nonlinear_trajectory.py']

model = dict(
    bbox_head=dict(
        # 尺度条件化保持不变 (lambda_mod=0.5)
        # scale_conditioned_rf 继承自 nonlinear_trajectory.py
        # 关闭 OT Flow: 改用随机耦合
        coupling=dict(_delete_=True, type='random'),
    ),
)
