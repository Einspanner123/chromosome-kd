# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['./chromodet_baseline.py']

# HyperParam
use_length_prior = True

# model settings
model = dict(
    bbox_head=dict(
        type='ChromoDetDynamicHead',
        # 长度感知
        use_length_prior=use_length_prior,
        single_head=dict(
            type='ChromoDetSingleHead', use_length_prior=use_length_prior
        ),
        # criterion
        criterion=dict(
            type='ChromoDetCriterion',
            use_length_prior=use_length_prior,  # 长度先验
            assigner=dict(
                type='ChromoDetMatcher',
                use_length_prior=use_length_prior,
            ),
        ),
    ),
)
