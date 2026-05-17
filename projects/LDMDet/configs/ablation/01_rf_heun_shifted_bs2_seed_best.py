_base_ = ['../ldmdet_rf_heun_shifted_bs2.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='01_rf_heun_shifted_bs2',
                description='Ablation: RF+Heun+Shifted | seed=1769925607',
            ),
        ),
    ],
)
