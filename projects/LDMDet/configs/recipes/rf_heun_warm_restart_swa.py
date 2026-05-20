_base_ = ['../ldmdet_rf_heun_shifted_bs8.py']

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.0005, by_epoch=True, begin=0, end=10),
    dict(
        type='CosineAnnealingLR',
        T_max=65,
        eta_min=1e-6,
        begin=10,
        end=75,
        by_epoch=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=75,
        eta_min=1e-6,
        begin=75,
        end=max_epoch,
        by_epoch=True,
    ),
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='StochasticWeightAverage',
        interval=1,
        begin_epoch=75,
        priority='ABOVE_NORMAL',
    ),
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=30,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]
