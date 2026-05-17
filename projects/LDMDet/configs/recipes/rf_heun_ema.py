_base_ = ['../ldmdet_rf_heun_shifted_bs8.py']

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        momentum=0.0002,
        begin_epoch=10,
        priority='ABOVE_NORMAL',
    ),
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=15,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]
