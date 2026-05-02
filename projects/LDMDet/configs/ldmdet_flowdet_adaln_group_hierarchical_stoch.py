_base_ = ["./ldmdet_flowdet_adaln.py"]

# Group-Hierarchical + Stochastic Coupling (ε=5)
# 组合两个降低 ΔH 的机制:
#   1. 分组 OT: 加权 ΔH 从 log(46)=3.83 降至 ~1.21
#   2. 随机采样: 从传输矩阵采样, 保留更大多样性
# 预期效果: OT 效率 + 多样性保留 = 超越 baseline
model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher="sinkhorn",
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
        ot_group_hierarchical=True,
    ),
)
