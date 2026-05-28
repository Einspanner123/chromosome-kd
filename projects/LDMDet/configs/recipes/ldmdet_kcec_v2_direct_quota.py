_base_ = ['./ldmdet_kcec_pure_sota.py']

# ==============================================================================
# KCEC V2: Direct GT Quota Configuration
# ------------------------------------------------------------------------------
# 理论进化：废除 Slot 代理，回归原子级 (Proposal-to-GT) 匹配。
# 通过 col_mass 直接为每个 GT 赋予基于类别的“倍性权重”。
# 旨在找回 SOTA (0.753) 的匹配多样性，同时保留核型配额约束。
# ==============================================================================

model = dict(
    bbox_head=dict(
        kcec_morph_weight=0.0,   # 初始设为 0，验证纯 V2 结构的有效性
        kcec_group_weight=0.0,
        kcec_quota_strength=1.0,
        kcec_slack=0.05,
    )
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-kcec',
                experiment_name='kcec_v2_direct_quota',
                description='KCEC V2: Direct Proposal-to-GT coupling with class-based quota weights',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_kcec_v2_direct_quota'
