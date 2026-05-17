_base_ = ['../ldmdet_rf_heun_shifted_bs2.py']

train_dataloader = dict(batch_size=8, num_workers=8)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0002, weight_decay=0.0001, _delete_=True
    ),
)

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning='adaln_zero',
        ),
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=0.1,
        ot_num_iters=50,
        ot_sample=True,
        ot_kcec=True,
        kcec_morph_weight=0.25,
        kcec_group_weight=0.1,
        kcec_cls_weight=0.0,
        kcec_quota_strength=1.0,
        kcec_slack=0.05,
        kcec_log_interval=100,
    ),
)
