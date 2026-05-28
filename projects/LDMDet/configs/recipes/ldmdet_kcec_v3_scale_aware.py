
_base_ = ['./ldmdet_kcec_v2_direct_quota.py']

# ==============================================================================
# KCEC V3: Scale-Aware & Gated Quota Configuration
# ------------------------------------------------------------------------------
# 理论进化：解决 V2 中“中小物体跌落”的信噪比问题。
# 1. kcec_scale_anneal=True: 根据物体面积动态调整配额强度。
#    大染色体 (1-5号) 维持强约束以保住 +3.7% 的增益。
#    小染色体 (D/E/F/G组) 自动减弱约束以填平 -3.1% 的坑。
# 2. kcec_conf_threshold=0.5: 门控机制，防止在没有 Proposal 的背景区域
#    强行注入额外质量导致假阳性 (False Positives)。
# ==============================================================================

model = dict(
    bbox_head=dict(
        ot_kcec=True,
        kcec_scale_anneal=True,     # 开启尺度感知
        kcec_conf_threshold=0.5,    # 开启代价门控 (阈值基于 cdist 欧氏距离)
        kcec_quota_strength=1.0,    # 基础强度
        kcec_morph_weight=0.0,      # 保持纯净 V3 结构验证
        kcec_group_weight=0.0,
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
                experiment_name='kcec_v3_scale_aware_gated',
                description='KCEC V3: Solving S/M performance drop with scale-aware strength and confidence gating',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_kcec_v3_scale_aware_gated'
