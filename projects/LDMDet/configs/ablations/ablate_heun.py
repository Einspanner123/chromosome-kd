_base_ = ["../ldmdet_flowdet_adaln.py"]

# 消融: 移除 Heun 二阶求解器 → 回退到 Euler 一阶
model = dict(
    bbox_head=dict(
        solver_type="euler",
    ),
)
