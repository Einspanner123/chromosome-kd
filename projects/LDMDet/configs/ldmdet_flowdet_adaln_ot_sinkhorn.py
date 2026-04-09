_base_ = ["./ldmdet_flowdet_adaln.py"]

# OT 改进方案 A: 放松 OT 约束
#
# 问题: 原始 ot_coupling 用 argmin 最近邻匹配，
# 导致所有噪声框都分配到最近的 GT，GT 分布不均匀。
#
# 改进: 使用 Sinkhorn OT 做噪声→GT 的均衡配对，
# 保证每个 GT 被大致相同数量的噪声框覆盖。
# 同时保留 SimOTA 做标签分配（不影响检测质量）。

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher="sinkhorn",
        ot_epsilon=1.0,
        ot_num_iters=20,
    ),
)
