_base_ = ['../ldmdet_rf_heun_shifted_bs2.py']

model = dict(
    bbox_head=dict(
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
    ),
)
