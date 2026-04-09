_base_ = ["./ldmdet_flowdet_adaln.py"]

# OT 改进方案 B: OT 仅用于噪声初始化（混合匹配）
#
# 核心思路:
# - 标签分配: 仍用 SimOTA（保证检测质量）
# - 噪声-GT 配对: 用 OT 找到最优 (noise, GT) 配对
#   但不是直接替换 x_start，而是用 OT 结果指导噪声框的初始化位置
#
# 具体做法:
# 1. 先随机采样噪声
# 2. 用 OT 找到每个噪声框应该匹配的 GT
# 3. 将噪声框平移到对应 GT 附近（加小扰动）
# 这样既减小了传输距离，又保持了噪声的随机性

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_init_mode="guided",
        ot_init_scale=0.5,
    ),
)
