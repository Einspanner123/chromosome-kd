_base_ = ["./ldmdet_flowdet_adaln.py"]

# Group-Hierarchical Coupling: 按染色体组分组 OT (A-G + Sex = 8 组)
# 每组独立运行 Sinkhorn, 有效 K 从 46 降至 max(K_g)=7
# 理论预测: 加权 ΔH 从 log(46)=3.83 降至 ~1.21
model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher="sinkhorn",
        ot_epsilon=1.0,
        ot_num_iters=20,
        ot_group_hierarchical=True,
    ),
)
