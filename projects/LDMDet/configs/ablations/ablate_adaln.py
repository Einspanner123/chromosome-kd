_base_ = ["../ldmdet_flowdet_adaln.py"]

# 消融: 移除 AdaLN-Zero → 回退到 scale-shift time conditioning
model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning="scale_shift",
        ),
    ),
)
