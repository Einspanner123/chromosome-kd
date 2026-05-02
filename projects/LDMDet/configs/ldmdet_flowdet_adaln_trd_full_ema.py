_base_ = ["./ldmdet_flowdet_adaln_trd_full.py"]

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="EarlyStoppingHook",
        priority=50,
        patience=20,
        min_delta=0.001,
        monitor="coco/bbox_mAP",
        rule="greater",
    ),
    dict(type="CopyProjectHook", priority="VERY_LOW"),
]
