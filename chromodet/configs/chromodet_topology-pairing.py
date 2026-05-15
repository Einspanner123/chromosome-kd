# Copyright (c) OpenMMLab. All rights reserved.
_base_ = ['./chromodet_baseline.py']

# HyperParam
use_topology_pairing = True
# 将匹配器的拓扑配对权重暴露为可调参数（保持与实现默认值一致）
topology_weight = 0.2

# model settings
model = dict(
    type='DiffusionDet',
    bbox_head=dict(
        type='ChromoDetDynamicHead',
        # 拓扑匹配
        use_topology_pairing=use_topology_pairing,
        single_head=dict(type='ChromoDetSingleHead'),
        # criterion
        criterion=dict(
            type='ChromoDetCriterion',
            use_topology_pairing=use_topology_pairing,  # 拓扑匹配
            assigner=dict(
                type='ChromoDetMatcher',
                use_topology_pairing=use_topology_pairing,
                topology_weight=topology_weight))))
