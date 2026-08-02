"""LDMDet RF+Heun+AdaLN + Hard OT Coupling

在 rf_heun_adaln (Random Coupling Baseline) 基础上, 将耦合策略改为 Hard OT
(确定性最优传输配对)。用于验证 OT Diversity Collapse 理论:
d=4 低维检测空间 + K≈46 时, Hard OT 导致多样性坍缩。

对比: Random (rf_heun_adaln) vs Hard OT (本配置) vs Stochastic (stochot_eps5)
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        coupling=dict(type='hard_ot'),
    ),
)
