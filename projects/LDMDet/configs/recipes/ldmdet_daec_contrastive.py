_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

# ==============================================================================
# DAEC Contrastive Alignment Configuration
# ------------------------------------------------------------------------------
# 在 KCEC 的基础上，增加 Contrastive Representation Alignment，解决 Information Cutoff
# ==============================================================================

# 保持与 0.753 SOTA 基准完全一致的训练超参
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00005, weight_decay=0.0001, _delete_=True
    )
)

# Enable find_unused_parameters to avoid DDP errors
find_unused_parameters = True

model = dict(
    bbox_head=dict(
        ot_kcec=True,
        ot_epsilon=5.0,
        ot_num_iters=20,
        kcec_morph_weight=0.25,
        kcec_group_weight=0.1,
        kcec_cls_weight=0.0,
        kcec_quota_strength=1.0,
        kcec_slack=0.05,
        kcec_log_interval=100,
        daec_contrastive_weight=0.1,  # DAEC 对比损失，强制特征对齐
    )
)

# 监控与可视化
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-daec',
                experiment_name='daec_contrastive_kcec_eps5',
                description='DAEC Contrastive Alignment on top of KCEC eps=5',
            ),
        ),
    ],
)

custom_hooks = [
    dict(
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        patience=30,
        priority=50,
        rule='greater',
        type='EarlyStoppingHook',
    ),
    dict(priority='VERY_LOW', type='CopyProjectHook'),
    dict(
        type='WeightSummaryHook', interval=100, log_norm=True, log_hist=False
    ),
]

work_dir = 'work_dirs/ldmdet_daec_contrastive_kcec_eps5'
