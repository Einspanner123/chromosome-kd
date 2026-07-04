"""LDMDet + Nonlinear Trajectory E4.2: 仅 OT Flow 耦合

消融实验: 与 E4.3 (nonlinear_trajectory.py) 配置相同, 作为 OT Flow 的基准.
ScaleConditionedRF 已证伪并移除, 本配置现与 nonlinear_trajectory.py 等价.

历史: 原配置通过 lambda_mod=0 关闭 SCRF, 现 SCRF 已删除, 自然等价.

对比:
- E4.0 (rf_heun_adaln): 线性 RF + random coupling → baseline
- E4.2 (本配置): 线性 RF + OT Flow coupling → 验证 OT 独立价值
"""

_base_ = ['nonlinear_trajectory.py']

# OT Flow 耦合保持不变 (epsilon=1.0, argmax), 与 baseline nonlinear_trajectory.py 相同
