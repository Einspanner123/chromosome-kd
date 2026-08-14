"""Frozen historical D2 yolox model."""

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        out_indices=(
            2,
            3,
            4,
        ),
        type='CSPDarknet',
        widen_factor=0.5),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        feat_channels=128,
        in_channels=128,
        loss_bbox=dict(loss_weight=5.0, reduction='sum', type='IoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_classes=24,
        stacked_convs=2,
        strides=(
            8,
            16,
            32,
        ),
        type='YOLOXHead',
        use_depthwise=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            128,
            256,
            512,
        ],
        num_csp_blocks=1,
        out_channels=128,
        type='YOLOXPAFPN'),
    test_cfg=dict(nms=dict(iou_threshold=0.65, type='nms'), score_thr=0.01),
    train_cfg=dict(
        assigner=dict(
            candidate_topk=10, center_radius=2.5, type='SimOTAAssigner')),
    type='YOLOX')
