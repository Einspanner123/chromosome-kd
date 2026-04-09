_base_ = ["./ldmdet_flowdet_adaln.py"]

# 方向 2: OT-Guided 训练时噪声耦合
#
# 核心改进: 训练时 (noise, GT) 配对不再随机，而是用 OT 最优耦合
# 每个 noise 分配到距离最近的 GT → 传输路径更短、更直、更少交叉
#
# 理论依据:
# - Rectified Flow 的 OT 路径比随机配对的路径更直 (Liu et al. 2023)
# - 在 4D bbox 空间中 OT 收敛快 (维度低)
# - 等价于免费的 1 轮 Reflow (减少 path crossing)
#
# 实现: 用 torch.cdist + argmin 做贪心最近邻 OT (无需 Sinkhorn)
# 计算开销: 仅 O(N*K) 距离矩阵 (N=500, K≈46), 可忽略

model = dict(
    bbox_head=dict(
        ot_coupling=True,
    ),
)
