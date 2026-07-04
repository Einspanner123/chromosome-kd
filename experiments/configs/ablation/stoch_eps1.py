"""ε 扫描 — Sinkhorn Stochastic (论文 Table 4)

Usage:
    bash train.sh experiments/configs/ablation/stoch_eps1.py
"""

_base_ = ['../ldmdet/sinkhorn_stochastic.py']

model = dict(
    bbox_head=dict(
        coupling=dict(type='sinkhorn_stochastic', epsilon=1.0, num_iters=20),
    ),
)
