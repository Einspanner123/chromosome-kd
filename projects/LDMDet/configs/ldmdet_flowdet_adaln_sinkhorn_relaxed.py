_base_ = ["./ldmdet_flowdet_adaln.py"]

# OT 改进方案 C: 调整 Sinkhorn 超参（用于标签分配器版本）
#
# 问题: ldmdet_flowdet_sinkhorn (0.720) 低于 baseline (0.725)
# 可能原因:
# 1. epsilon=0.1 太小 → 分配过于尖锐，接近硬分配
# 2. dustbin_cost=10.0 太大 → 太多 proposal 被分配到背景
# 3. 每个 GT 接收的 proposal 数量太少
#
# 改进: 放松 Sinkhorn 参数
# - epsilon: 0.1 → 1.0 (更平滑的分配)
# - dustbin_cost: 10.0 → 5.0 (减少背景分配)
# - num_iters: 50 → 30 (减少迭代，允许更模糊的分配)

model = dict(
    bbox_head=dict(
        criterion=dict(
            assigner=dict(
                type="PurePyTorchSinkhornOTMatcher",
                epsilon=1.0,
                num_iters=30,
                dustbin_cost=5.0,
                cost_class=2.0,
                cost_bbox=5.0,
                cost_giou=2.0,
            ),
        ),
    ),
)
