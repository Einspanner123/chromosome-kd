_base_ = ['../recipes/rf_heun_warm_restart.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-stability',
                experiment_name='bs8-warm-restart',
                description='Stability: CosineAnnealing warm-restart(ep80) bs=8 | seed=1769925607',
            ),
        ),
    ],
)
