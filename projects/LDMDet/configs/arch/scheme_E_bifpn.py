_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        num_repeats=1,
    ),
)

experiment_name = 'arch_E_bifpn'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-arch',
                experiment_name=experiment_name,
                description='Arch E: ResNet-50 + BiFPN (1 repeat) | RF+Heun+Sinkhorn+Stochastic eps5 | baseline 0.753',
            ),
        ),
    ]
)
