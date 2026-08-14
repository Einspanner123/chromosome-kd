"""Frozen historical D2 DiffusionDet optimization policy."""

custom_hooks = [
    dict(
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        patience=30,
        priority=50,
        rule='greater',
        type='EarlyStoppingHook'),
    dict(priority='VERY_LOW', type='CopyProjectHook'),
]
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=0.0001))
param_scheduler = [
    dict(begin=0, by_epoch=True, end=5, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=150,
        begin=5,
        by_epoch=True,
        end=150,
        eta_min=0,
        type='CosineAnnealingLR'),
]
test_cfg = dict()
train_cfg = dict(max_epochs=150, type='EpochBasedTrainLoop')
val_cfg = dict()
