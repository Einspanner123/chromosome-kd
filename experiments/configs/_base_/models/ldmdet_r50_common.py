"""Shared detector architecture; no dataset, schedule, or runtime settings."""

num_classes = 24

model = dict(
    type='LDMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='ResNet', depth=50, num_stages=4,
        out_indices=(0, 1, 2, 3), frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(
        type='FPN', in_channels=[256, 512, 1024, 2048],
        out_channels=256, num_outs=4,
    ),
    bbox_head=dict(
        type='PurePyTorchDiffusionDetHead',
        num_classes=num_classes, feat_channels=256,
        num_proposals=500, num_heads=6, deep_supervision=True,
        prior_prob=0.01, snr_scale=2.0,
        single_head=dict(
            type='PurePyTorchSingleDiffusionDetHead',
            num_classes=num_classes, feat_channels=256,
            num_cls_convs=1, num_reg_convs=3,
            dim_feedforward=2048, num_heads=8, dropout=0.0,
        ),
        roi_extractor=dict(
            type='PurePyTorchSingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256, featmap_strides=[4, 8, 16, 32],
        ),
        criterion=dict(
            type='PurePyTorchDiffusionDetCriterion',
            num_classes=num_classes,
            assigner=dict(
                type='PurePyTorchDiffusionDetMatcher',
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0),
                ],
                center_radius=2.5, candidate_topk=5,
            ),
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0),
            loss_bbox=dict(type='PurePyTorchL1Loss', loss_weight=5.0),
            loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=2.0),
        ),
    ),
    test_cfg=dict(
        use_nms=True, score_thr=0.5, min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.5),
    ),
)
