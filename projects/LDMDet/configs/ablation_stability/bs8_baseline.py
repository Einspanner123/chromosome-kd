_base_ = ['../ldmdet_rf_heun_shifted_bs8.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-stability',
                experiment_name='bs8-baseline',
                description='Stability baseline: bs=8, lr=2e-4 | seed=1769925607',
            ),
        ),
    ],
)
