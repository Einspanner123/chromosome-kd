_base_ = ["./ldmdet_flowdet_adaln_trd_full.py"]

model = dict(
    bbox_head=dict(
        use_consistency=True,
        consistency_ema_rate=0.999,
        consistency_num_timesteps=18,
        consistency_weight=1.0,
        sampling_timesteps=1,
    ),
)

max_epoch = 50
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=0.00002, weight_decay=0.0001))

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

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.001,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="EarlyStoppingHook",
        priority=50,
        patience=15,
        min_delta=0.001,
        monitor="coco/bbox_mAP",
        rule="greater",
    ),
    dict(type="CopyProjectHook", priority="VERY_LOW"),
]
