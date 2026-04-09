_base_ = ["./ldmdet_flowdet_adaln.py"]

# Phase 3A 消融: AdaLN-Zero + 结构化噪声 (独立验证)
# 结构化噪声仅影响推理时的初始噪声框，训练时仍用标准高斯
# noise_scale 需要适中：太大会淹没网格结构，太小会过于规则
model = dict(
    bbox_head=dict(
        noise_sampler=dict(
            type="PurePyTorchStructuredNoiseSampler",
            num_proposals=500,
            noise_scale=0.5,  # 比之前的 1.0 小，保留更多网格结构
            strategy="grid",
            snr_scale=2.0,
        ),
    ),
)
