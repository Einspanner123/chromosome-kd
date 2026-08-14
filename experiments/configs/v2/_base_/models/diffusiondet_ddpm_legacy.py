"""Frozen historical D2 DiffusionDet DDPM model."""

model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        criterion=dict(
            assigner=dict(
                candidate_topk=5,
                center_radius=2.5,
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(type='PurePyTorchBBoxL1Cost', weight=5.0),
                    dict(
                        iou_mode='giou', type='PurePyTorchIoUCost',
                        weight=2.0),
                ]),
            deep_supervision=True,
            loss_bbox=dict(loss_weight=5.0, type='PurePyTorchL1Loss'),
            loss_cls=dict(loss_weight=2.0, type='PurePyTorchFocalLoss'),
            loss_giou=dict(loss_weight=2.0, type='PurePyTorchGIoULoss'),
            num_classes=24),
        deep_supervision=True,
        diffusion_type='ddpm',
        feat_channels=256,
        num_classes=24,
        num_heads=6,
        num_proposals=500,
        prior_prob=0.01,
        rf_schedule='shifted',
        rf_shift=3.0,
        roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign')),
        sampling_timesteps=1,
        single_head=dict(
            dim_feedforward=2048,
            dropout=0.0,
            feat_channels=256,
            num_classes=24,
            num_cls_convs=1,
            num_heads=8,
            num_reg_convs=3,
            time_conditioning='scale_shift'),
        snr_scale=2.0,
        solver_type='euler',
        use_ensemble=False),
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
            256,
            512,
            1024,
            2048,
        ],
        num_outs=4,
        out_channels=256,
        type='FPN'),
    type='LDMDet')
