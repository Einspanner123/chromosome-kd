_base_ = ['../ldmdet_rf_heun_shifted_bs8.py']

model = dict(
    backbone=dict(
        _delete_=True,
        type='timm',
        model_name='convnextv2_base.fcmae_ft_in1k',
        pretrained=True,
        features_only=True,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.4,
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4,
    ),
    bbox_head=dict(
        use_flash_attn=True,
    ),
)

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
)
val_dataloader = dict(
    num_workers=1,
    prefetch_factor=2,
    persistent_workers=True,
)
test_dataloader = dict(
    num_workers=1,
    prefetch_factor=2,
    persistent_workers=True,
)
optim_wrapper = dict(optimizer=dict(lr=2.5e-5, weight_decay=0.05))

compile = False

experiment_name = 'arch_D_convnextv2_base_fcmae'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-arch',
                experiment_name=experiment_name,
                description='Arch D: ConvNeXtV2-Base + FCMAE pretrained | FPN | RF+Heun+Sinkhorn | bs1 lr2.5e-5',
            ),
        ),
    ]
)
