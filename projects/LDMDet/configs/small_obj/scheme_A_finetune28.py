_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

model = dict(
    bbox_head=dict(
        roi_extractor=dict(
            finest_scale=28,
        ),
    ),
)

experiment_name = 'smallobj_A_finetune28'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj A: finest_scale 56->28, fix FPN degradation | R50+RF+Heun+Sinkhorn bs2',
            ),
        ),
    ]
)
