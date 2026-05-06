_base_ = ["./ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py"]

model = dict(
    backbone=dict(
        _delete_=True,
        type="TIMMBackbone",
        model_name="convnext_base",
        features_only=True,
        pretrained=False,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.4,
        frozen_stages=2,
        checkpoint_path="checkpoints/convnext_base_22k_1k_224.pth",
        feature_norm=True,
    ),
    neck=dict(
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4,
    ),
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.05),
)

max_epoch = 100
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

train_dataloader = dict(
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
)

val_dataloader = dict(
    num_workers=1,
    prefetch_factor=2,
    persistent_workers=True,
)
