# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['./chromodet_baseline.py']

# HyperParam
use_morphology_aware = True

# model settings
model = dict(
    type='DiffusionDet',
    bbox_head=dict(
        type='ChromoDetDynamicHead',
        use_morphology_aware=use_morphology_aware,
        single_head=dict(type='ChromoDetSingleHead'),
        # criterion
        criterion=dict(
            type='ChromoDetCriterion',
            use_morphology_aware=use_morphology_aware,  # 形态感知
            assigner=dict(
                type='ChromoDetMatcher',
                use_morphology_aware=use_morphology_aware),
        ),
    ))

randomness = dict(
    deterministic=False,  # 使用确定性CUDA计算
)
