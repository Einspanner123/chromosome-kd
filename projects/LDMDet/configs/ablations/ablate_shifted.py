_base_ = ["../ldmdet_flowdet_adaln.py"]

# 消融: 移除 Shifted Schedule → 回退到 uniform (linear) 时间采样
model = dict(
    bbox_head=dict(
        rf_schedule="linear",
    ),
)
