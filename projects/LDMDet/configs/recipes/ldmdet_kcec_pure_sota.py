_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

# ==============================================================================
# KCEC Pure SOTA Configuration
# ------------------------------------------------------------------------------
# 此配置严格基于 0.753 mAP 的基准环境 (Stochastic OT eps=5, TF32 OFF, Heun Solver)
# 仅作为“干净加法”引入 KCEC 机制。
# ==============================================================================

# 保持与 0.753 SOTA 基准完全一致的训练超参，确保增益仅来源于 KCEC
train_dataloader = dict(batch_size=2)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.00005, weight_decay=0.0001, _delete_=True
    )
)

model = dict(
    bbox_head=dict(
        ot_kcec=True,
        ot_epsilon=5.0,
        ot_num_iters=20,  # 严格对齐 SOTA 的 20 次迭代
        kcec_morph_weight=0.25,  # 形态 (w,h) 匹配代价
        kcec_group_weight=0.1,  # A-G 组别先验
        kcec_cls_weight=0.0,  # 类别代价 (当前暂不启用)
        kcec_quota_strength=1.0,  # 软倍性配额强度
        kcec_slack=0.05,  # 允许异常核型的弹性边际质量
        kcec_log_interval=100,
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
                project='chromosome-kd-kcec',
                experiment_name='kcec_pure_sota_eps5',
                description='KCEC on top of 0.753 SOTA base (eps=5, TF32-OFF)',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_kcec_pure_sota_eps5'
