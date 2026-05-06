"""LDMDet on single_chromosomes_object -- Sinkhorn stochastic coupling eps=5.

Same as the random baseline config but with Sinkhorn OT + stochastic sampling
at epsilon=5, matching the best historical configuration.
"""
_base_ = ["./ldmdet_single_chromo_random.py"]

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning="adaln_zero",
        ),
        ot_coupling=True,
        ot_matcher="sinkhorn",
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
    ),
    num_classes=1,
)
