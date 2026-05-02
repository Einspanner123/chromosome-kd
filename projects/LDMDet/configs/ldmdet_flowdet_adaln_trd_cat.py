_base_ = ["./ldmdet_flowdet_adaln.py"]

model = dict(
    bbox_head=dict(
        use_trd=True,
        trd_self_cond_prob=0.5,
        use_cat=True,
        cat_weight=1.0,
        cat_delta_t=0.05,
        prediction_mode="velocity",
        velocity_loss_weight=1.0,
    ),
)
