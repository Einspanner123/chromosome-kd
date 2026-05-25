_base_ = ['./scheme_A_finetune28.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            scale_aware=True,
            scale_aware_mode='sqrt_inverse',
            scale_aware_min_weight=0.5,
            scale_aware_max_weight=3.0,
        ),
    ),
)

experiment_name = 'smallobj_AB_finetune28_scaleaware'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj A+B: finest_scale=28 + scale-aware loss (sqrt_inverse, w∈[0.5,3.0]) | R50+RF+Heun+Sinkhorn bs2',
            ),
        ),
    ]
)
