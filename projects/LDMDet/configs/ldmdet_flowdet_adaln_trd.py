_base_ = ["./ldmdet_flowdet_adaln.py"]

model = dict(
    bbox_head=dict(
        use_trd=True,
        trd_self_cond_prob=0.5,
        prediction_mode="velocity",
        velocity_loss_weight=1.0,
        ot_coupling=True,
        ot_matcher="nearest",
    ),
)
