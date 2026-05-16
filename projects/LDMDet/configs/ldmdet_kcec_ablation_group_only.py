"""KCEC ablation: group-only prior (no morphology, no quota marginals)."""

_base_ = ['./ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

custom_imports = dict(
    imports=[
        'projects.LDMDet.model',
        'projects.LDMDet.hooks',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

data_root = '/data/linkst/datasets/Chromosome20240904_NoAug_NoResize_coco/'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
    ),
)
val_dataloader = dict(
    num_workers=4,
    prefetch_factor=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
    ),
)
test_dataloader = dict(
    num_workers=4,
    prefetch_factor=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
    ),
)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    classwise=True,
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
        'AR@100',
        'AR@300',
        'AR@1000',
        'AR_s@1000',
        'AR_m@1000',
        'AR_l@1000',
    ],
    format_only=False,
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox',
    classwise=True,
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
        'AR@100',
        'AR@300',
        'AR@1000',
        'AR_s@1000',
        'AR_m@1000',
        'AR_l@1000',
    ],
    format_only=False,
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0002, weight_decay=0.0001, _delete_=True
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

max_epoch = 150
train_cfg = dict(max_epochs=max_epoch)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

model = dict(
    bbox_head=dict(
        ot_kcec=True,
        ot_matcher='sinkhorn',
        ot_epsilon=0.1,
        ot_num_iters=50,
        ot_sample=True,
        kcec_morph_weight=0.0,
        kcec_group_weight=0.1,
        kcec_cls_weight=0.0,
        kcec_quota_strength=0.0,
        kcec_slack=0.0,
        kcec_log_interval=100,
    )
)

visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(_scope_='mmdet', type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-kcec',
                experiment_name='ablation-group-only',
                description='KCEC ablation: group-only prior',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_kcec_ablation_group_only'
