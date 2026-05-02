_base_ = [
    "../../../chromodet/base/schedules/schedule_1x.py",
    "../../../chromodet/base/default_runtime.py",
]

custom_imports = dict(
    imports=["projects.LDMDet.model", "projects.LDMDet.hooks"],
    allow_failed_imports=False,
)

dataset_type = "CocoDataset"
data_root = "data/coco/"
num_classes = 80
batch_size = 2
num_workers = 4
prefetch_factor = 4

backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333),
                        (576, 1333), (608, 1333), (640, 1333),
                        (672, 1333), (704, 1333), (736, 1333),
                        (768, 1333), (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val2017.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

model = dict(
    type="LDMDet",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=4
    ),
    bbox_head=dict(
        type="PurePyTorchDiffusionDetHead",
        num_classes=num_classes,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=4,
        diffusion_type="rectified_flow",
        solver_type="heun",
        rf_schedule="shifted",
        rf_shift=3.0,
        prediction_mode="velocity",
        velocity_loss_weight=1.0,
        ot_coupling=True,
        ot_matcher="nearest",
        use_trd=True,
        trd_self_cond_prob=0.5,
        use_cat=True,
        cat_weight=1.0,
        cat_delta_t=0.05,
        use_lsas=True,
        lsas_num_bins=100,
        lsas_temp=1.0,
        single_head=dict(
            type="PurePyTorchSingleDiffusionDetHead",
            num_classes=num_classes,
            feat_channels=256,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            time_conditioning="adaln_zero",
        ),
        roi_extractor=dict(
            type="PurePyTorchSingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        criterion=dict(
            type="PurePyTorchDiffusionDetCriterion",
            num_classes=num_classes,
            assigner=dict(
                type="PurePyTorchDiffusionDetMatcher",
                match_costs=[
                    dict(type="PurePyTorchFocalLossCost", weight=2.0),
                    dict(type="PurePyTorchBBoxL1Cost", weight=5.0),
                    dict(type="PurePyTorchIoUCost", iou_mode="giou", weight=2.0),
                ],
                center_radius=2.5,
                candidate_topk=5,
            ),
            loss_cls=dict(type="PurePyTorchFocalLoss", loss_weight=2.0),
            loss_bbox=dict(type="PurePyTorchL1Loss", loss_weight=5.0),
            loss_giou=dict(type="PurePyTorchGIoULoss", loss_weight=2.0),
        ),
    ),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type="nms", iou_threshold=0.5),
    ),
)

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=0.00005, weight_decay=0.0001))

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="EarlyStoppingHook",
        priority=50,
        patience=20,
        min_delta=0.001,
        monitor="coco/bbox_mAP",
        rule="greater",
    ),
    dict(type="CopyProjectHook", priority="VERY_LOW"),
]

log_level = "INFO"
