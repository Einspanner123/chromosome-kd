_base_ = ["./ldmdet_rf_heun_shifted_bs2.py"]

# Phase 1 消融: 仅 AdaLN-Zero (其他保持 RF baseline)
model = dict(
    bbox_head=dict(
        single_head=dict(
            type="PurePyTorchSingleDiffusionDetHead",
            num_classes=24,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            time_conditioning="adaln_zero",
        ),
    ),
)
