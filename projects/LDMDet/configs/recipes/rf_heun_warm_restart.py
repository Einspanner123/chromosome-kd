_base_ = ['../ldmdet_rf_heun_shifted_bs8.py']

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=75,
        eta_min=1e-6,
        begin=5,
        end=80,
        by_epoch=True,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=70,
        eta_min=1e-6,
        begin=80,
        end=max_epoch,
        by_epoch=True,
    ),
]
