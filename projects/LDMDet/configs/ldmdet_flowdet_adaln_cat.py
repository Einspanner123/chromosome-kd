_base_ = ["./ldmdet_flowdet_adaln.py"]

model = dict(
    bbox_head=dict(
        use_cat=True,
        cat_weight=1.0,
        cat_delta_t=0.05,
    ),
)
