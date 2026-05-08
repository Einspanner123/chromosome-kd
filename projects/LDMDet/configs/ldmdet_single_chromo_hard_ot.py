"""LDMDet on single_chromosomes_object — Hard OT coupling."""
_base_ = ["./ldmdet_single_chromo_random.py"]

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher="nearest",
        ot_epsilon=0.0,
        ot_num_iters=1,
        ot_sample=False,
    ),
)

work_dir = "work_dirs/ldmdet_single_chromo_hard_ot"
