_base_ = ['../recipes/rf_heun_ema.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-stability',
                experiment_name='ema',
                description='Stability: EMA(momentum=0.0002, begin=ep10) | seed=1769925607',
            ),
        ),
    ],
)
