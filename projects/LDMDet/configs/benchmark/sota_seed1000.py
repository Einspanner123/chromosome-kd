_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

SEED = 1000

randomness = dict(seed=SEED, deterministic=False, diff_rank_seed=True)

experiment_name = f'sota_seed{SEED}'

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-multiseed',
                experiment_name=experiment_name,
                description=f'SOTA multi-seed: seed={SEED}, RF+Heun+Shifted+AdaLN+SinkhornStochastic eps=5',
                api_key='Huzvq1fnDeqOwgQo2AMAI',
            ),
        ),
    ]
)
