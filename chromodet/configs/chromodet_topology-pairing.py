_base_ = [
    'mmdet::_base_/datasets/chromo_coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]
custom_imports = dict(
    imports=[
        'chromodet.model',
        'projects.DiffusionDet.diffusiondet'],
    allow_failed_imports=False)

num_classes = 24

# HyperParam
use_morphology_aware = False
use_length_prior = False
length_priors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75,
                 0.70, 0.65, 0.60, 0.55, 0.50, 0.45,
                 0.40, 0.38, 0.36, 0.34, 0.32, 0.30,
                 0.28, 0.26, 0.24, 0.22, 0.20, 0.18]
use_topology_pairing = True


# model settings
model = dict(
    type='DiffusionDet',
    bbox_head=dict(
        type='ChromoDetDynamicHead',
        # 拓扑匹配
        use_topology_pairing=use_topology_pairing,
        single_head=dict(
            type='ChromoDetSingleHead'),
        # criterion
        criterion=dict(
            type='ChromoDetCriterion', # 保持原Criterion
            use_topology_pairing=use_topology_pairing, # 拓扑匹配
            assigner=dict(
                type='ChromoDetMatcher', # 保持原Assigner
                use_topology_pairing=use_topology_pairing)
            )
        )
    )
