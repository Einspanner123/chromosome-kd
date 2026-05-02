_base_ = ["./ldmdet_flowdet_adaln.py"]

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher="sinkhorn",
        ot_epsilon=2.0,
        ot_num_iters=20,
        ot_sample=True,
    ),
)
