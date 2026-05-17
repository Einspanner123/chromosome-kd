_base_ = ['../ldmdet_group_hierarchical_stoch.py']

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-ablation',
                experiment_name='sota_group_hier_stoch',
                description='Ablation: RF+Heun+Shifted+AdaLN+StochOT(eps5)+GroupHier | seed=1769925607',
            ),
        ),
    ],
)
