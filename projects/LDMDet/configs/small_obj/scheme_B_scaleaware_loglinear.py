_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

model = dict(
    bbox_head=dict(
        criterion=dict(
            scale_aware=True,
            scale_aware_mode='log_linear',
            scale_aware_alpha=0.15,
            scale_aware_giou=False,
        ),
    ),
)

experiment_name = 'smallobj_B_scaleaware_loglinear'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-smallobj',
                experiment_name=experiment_name,
                description='SmallObj B: scale-aware L1 loss (log_linear, alpha=0.15, GIoU unweighted) | R50+RF+Heun+Sinkhorn bs2',
            ),
        ),
    ]
)
