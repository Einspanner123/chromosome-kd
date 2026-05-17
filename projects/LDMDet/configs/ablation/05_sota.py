_base_ = ['../recipes/rf_heun_adaln_stochot_grouphier.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='05-sota',
                description='Ablation: RF+Heun+Shifted+AdaLN+StochOT(eps5)+GroupHier=SOTA | seed=1769925607',
            ),
        ),
    ],
)
