_base_ = ['./ldmdet_baseline.py']
train_dataloader = dict(batch_size=8, num_workers=8)
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='heun',
        sampling_timesteps=4,
        rf_schedule='shifted',
        rf_shift=3.0,
        snr_scale=2.0,
    )
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.0005, by_epoch=True, begin=0, end=10),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=10,
        end=max_epoch,
        by_epoch=True,
    ),
]
