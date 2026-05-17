_base_ = ['../recipes/rf_heun_ema_stratified.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-stability',
                experiment_name='ema-stratified',
                description='Stability: EMA + stratified t-sampling | seed=1769925607',
            ),
        ),
    ],
)
