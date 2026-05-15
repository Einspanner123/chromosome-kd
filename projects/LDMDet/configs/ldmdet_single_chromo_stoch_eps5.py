"""LDMDet on single_chromosomes_object — Sinkhorn stochastic coupling eps=5."""

_base_ = ['./ldmdet_single_chromo_random.py']

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
    ),
)

work_dir = 'work_dirs/ldmdet_single_chromo_stoch_eps5'
