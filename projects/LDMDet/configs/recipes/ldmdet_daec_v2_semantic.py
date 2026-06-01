_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

# ==============================================================================
# DAEC Phase 2: Semantic-Aware Coupling + Contrastive Alignment
# ------------------------------------------------------------------------------
# 在 Phase 1 的基础上，引入 kcec_cls_weight 实现双向对齐。
# 同时调低对比温度 daec_temperature 以增加排斥力，防止特征坍缩。
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
        kcec_cls_weight=0.5,  # 启用语义感知耦合，权重设为 0.5
        kcec_quota_strength=1.0,
        kcec_slack=0.05,
        kcec_log_interval=100,
        daec_contrastive_weight=0.1,
        daec_temperature=0.05,  # 降低温度，增加特征排斥力
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
                experiment_name='daec_v2_semantic_eps5',
                description='DAEC Phase 2: Semantic Coupling + Low-Temp Contrastive',
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
        type='WeightSummaryHook', interval=100, log_norm=True, log_heatmap=True
    ),
]

work_dir = 'work_dirs/ldmdet_daec_v2_semantic'
