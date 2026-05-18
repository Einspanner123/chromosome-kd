_base_ = ['../recipes/rf_heun_warm_restart_ema.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-stability',
                experiment_name='bs8-warm-restart-ema',
                description='Stability: WarmRestart(ep75) + EMA(mom=0.0002) bs=8 | seed=1769925607',
            ),
        ),
    ],
)
