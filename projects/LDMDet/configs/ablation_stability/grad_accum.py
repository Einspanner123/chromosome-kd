_base_ = ['../recipes/rf_heun_grad_accum.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-stability',
                experiment_name='grad-accum',
                description='Stability: gradient accumulation(accum=2, effective_bs=4) | seed=1769925607',
            ),
        ),
    ],
)
