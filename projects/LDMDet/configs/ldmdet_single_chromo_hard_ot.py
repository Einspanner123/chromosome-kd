"""LDMDet on single_chromosomes_object -- Hard OT coupling.

Same as the random baseline config but with deterministic OT coupling enabled.
"""
_base_ = ["./ldmdet_single_chromo_random.py"]

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning="adaln_zero",
        ),
        ot_coupling=True,
        ot_matcher="hard",  # deterministic OT via torch.cdist argmin
        ot_epsilon=0.0,
        ot_num_iters=1,
        ot_sample=False,
    ),
    num_classes=1,
)
