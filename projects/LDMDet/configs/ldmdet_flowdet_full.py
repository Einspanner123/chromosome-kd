_base_ = ["./ldmdet_flowdet_adaln.py"]

# FlowDet 完整配置 v2:
# 仅组合已验证有效/无害的组件，去掉 Sinkhorn (单独跑 0.715 低于 baseline)
#
# ✅ Phase 1: AdaLN-Zero        (0.751, 超过 baseline)
# ✅ Phase 3A: 结构化噪声        (0.742, 接近 baseline, 仅推理时生效)
# ✅ Phase 3B: Objectness        (0.739, 接近 baseline)
# ✅ Phase 4: Velocity 辅助 loss  (0.738, 接近 baseline, 权重 1.0)
# ❌ Phase 2: Sinkhorn OT        (0.715, 拖后腿, 暂不包含)
#
# Matcher: 仍用 SimOTA (继承自 adaln baseline)

model = dict(
    bbox_head=dict(
        prediction_mode="velocity",
        velocity_loss_weight=1.0,
        single_head=dict(
            use_objectness=True,
            prediction_mode="velocity",
        ),
        criterion=dict(
            loss_objectness_weight=1.0,
        ),
        noise_sampler=dict(
            type="PurePyTorchStructuredNoiseSampler",
            num_proposals=500,
            noise_scale=0.5,
            strategy="grid",
            snr_scale=2.0,
        ),
    ),
)
