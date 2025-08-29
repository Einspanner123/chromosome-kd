_base_ = ['./diffusiondet_baseline.py']

custom_imports = dict(
    imports=['chromodet.model'],
    allow_failed_imports=False)

num_classes = 24
# model settings
model = dict(
    bbox_head=dict(
        type='ChromoDetDynamicHead',
        num_classes=num_classes, 
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        aspect_ratio_gamma=10.0,
        single_head=dict(
            type='ChromoDetSingleHead',
            num_classes=num_classes,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # criterion
        criterion=dict(
            type='ChromoDetCriterion', # 保持原Criterion
            num_classes=num_classes,
            assigner=dict(
                type='ChromoDetMatcher', # 保持原Assigner
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
    )))
