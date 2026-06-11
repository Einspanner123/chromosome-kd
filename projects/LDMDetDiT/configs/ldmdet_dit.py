_base_ = [
    '../../../configs/_base_/datasets/chromo_coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        'projects.LDMDetDiT.model',
        'projects.LDMDetDiT.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# 模型设置
num_classes = 24  # A1-A3, B4-B5, C6-C12, D13-D15, E16-E18, F19-F20, G21-G22, X, Y
feat_channels = 384  # DINOv3-Small 原生维度
max_epoch = 150
batch_size = 4
num_workers = 4
prefetch_factor = 4

model = dict(
    type='PurePyTorchDiffusionDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    # Backbone 切换为 DINOv3-Small
    backbone=dict(
        type='timm_model',
        model_name='vit_small_patch16_dinov3.lvd1689m',
        pretrained=True,
        features_only=True,
        out_indices=(5, 8, 11),  # STA 3 级输入: 中层结构(B5) -> 深层语义(B8) -> 全局表征(B11)
    ),
    neck=dict(
        type='SpatialTuningAdapter',  # DEIMv2 STA: 双线性插值 + CNN 细节分支 + Bi-Fusion
        embed_dim=feat_channels,
        out_channels=feat_channels,
        conv_inplane=16,
        num_levels=3,
    ),
    bbox_head=dict(
        type='DiTDiffusionDetHead',
        num_classes=num_classes,
        feat_channels=feat_channels,
        num_proposals=100,
        num_heads=3,
        num_blocks=3,  # 每个 head 内 DiTBlock 堆叠数，增加模型深度
        share_heads=False,  # 不使用权重共享，每个 head 独立学习
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=6,
        diffusion_type='rectified_flow',
        solver_type='heun',
        rf_schedule='shifted',
        rf_shift=1.0,
        box_renewal=True,
        use_ensemble=True,
        prediction_mode='x0',
        adaln_params=9,
        regression_mode='direct',  # direct模式: reg_head输出RF velocity v (v=x_noise-x_start)，推理时x0=x_t-v*t，非sigmoid bbox
        use_adaln_zero=False,  # AdaLN-Zero 导致 3×3=9个DiTBlock退化为恒等函数，分类头无法学习
        train_noise_source='grid',  # 训练噪声也用网格初始化，与推理分布一致，消除训练-推理不匹配
        num_fpn_levels=3,
        num_ref_points=8,
        box_init_mode='spatial_prior',  # DAB-DETR: 均匀分布锚点位置编码，注入空间先验
        ot_coupling=True,  # OT 耦合: 将噪声最优分配到 GT，确保每个 proposal 都有有意义的回归目标
        ot_matcher='sinkhorn',  # Sinkhorn 匹配: 更均匀地分配噪声到各 GT
        single_head=dict(
            type='DiTSingleHead',
            num_classes=num_classes,
            feat_channels=feat_channels,
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=1536,  # 384 * 4
            num_heads=12,  # 384 / 12 = 32
            num_fpn_levels=3,
            num_ref_points=8,
            prediction_mode='x0',
            adaln_params=9,
            regression_mode='direct',
            use_adaln_zero=False,
        ),
        criterion=dict(
            type='PurePyTorchDiffusionDetCriterion',
            num_classes=num_classes,
            # assigner 保留接口兼容，实际使用 OT (Sinkhorn) 匹配，不再走 SimOTA
            assigner=dict(
                type='PurePyTorchDiffusionDetMatcher',
                match_costs=[
                    dict(type='PurePyTorchFocalLossCost', weight=2.0),
                    dict(type='PurePyTorchRelativeL1Cost', weight=5.0),
                    dict(
                        type='PurePyTorchIoUCost', iou_mode='giou', weight=2.0
                    ),
                ],
                center_radius=5.0,
                candidate_topk=12,
            ),
            # OT 匹配下: loss_cls (分类) + loss_giou (尺度敏感回归) + loss_vel (扩散核心)
            # loss_objectness 已移除 (OT 下全为 1, 无信息量)
            # loss_bbox 已移除 (与 loss_vel/displacement MSE 功能重叠)
            loss_cls=dict(type='PurePyTorchFocalLoss', loss_weight=2.0),
            loss_giou=dict(type='PurePyTorchGIoULoss', loss_weight=2.0),
            # OT 正例比例: 只保留传输概率 top-k 的 proposal 作为正例
            # 0.25 = 100 proposals 中 25 个正例，与 GT 数量 (~46) 匹配
            # 其余 proposal 标记为背景，学习抑制噪声框
            ot_pos_ratio=0.5,
        ),
    ),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.001,  # 降至此门槛以下：sigmoid(bias_init)≈0.01 远高于此，不会滤空初始预测
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.5),
    ),
)

backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                    backend=backend,
                ),
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                    backend=backend,
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type='RandomChoiceResize',
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    keep_ratio=True,
                    backend=backend,
                ),
            ],
        ],
    ),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend,
    ),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True, backend=backend),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

custom_hooks = [
    dict(type='CopyProjectHook', priority='VERY_LOW'),
    dict(
        type='WeightSummaryHook', interval=50, log_norm=True, log_heatmap=True
    ),
    dict(
        type='PredictionVisHook',
        num_images=4,
        score_thr=0.01,
    ),
]

# 优化器设置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, weight_decay=0.0001, _delete_=True
    ),
    # deep supervision 7 个 head 梯度叠加，grad_norm 约 250-320，需要适当放宽
    clip_grad=dict(max_norm=50.0, norm_type=2),
)

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch, val_interval=1)
# 修复 DDP 错误: 当某些 batch 为空或 contrastive loss 未触发时，防止梯度同步失败
find_unused_parameters = True

param_scheduler = [
    # AdaLN-Zero 初始化: 残差分支初始为零 (identity block)
    # 需要足够长的 warmup 让调制参数 (gamma, beta, alpha) 从零平滑过渡
    # 2000 iters ≈ 2 epochs (960 iters/epoch), 确保残差路径稳定建立
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=1e-6,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd',
                experiment_name='ldmdet_dinov3_small_384_ot_matching',
                description='Phase 5: DINOv3-Small out_indices=(2,5,8,11) + Neck LayerNorm + OT Matching (no SimOTA) + RF-Shifted3 + Heun',
                api_key='Huzvq1fnDeqOwgQo2AMAI',
            ),
        ),
    ],
    name='visualizer',
)

work_dir = 'work_dirs/ldmdet_dit_v9'

log_level = 'INFO'
