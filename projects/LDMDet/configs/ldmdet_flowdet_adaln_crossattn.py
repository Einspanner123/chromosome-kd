_base_ = ["./ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py"]

load_from = None

custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        priority=50,
        patience=20,
        min_delta=0.001,
        monitor="coco/bbox_mAP",
        rule="greater",
    ),
    dict(type="CopyProjectHook", priority="VERY_LOW"),
    dict(
        type="CheckpointMigrationHook",
        old_ckpt_path="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/best_coco_bbox_mAP_epoch_86.pth",
    ),
]

model = dict(
    bbox_head=dict(
        single_head=dict(
            interact_type="linear_cross_attn",
        ),
    ),
)

train_dataloader = dict(
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,
)

val_dataloader = dict(
    num_workers=1,
    prefetch_factor=2,
    persistent_workers=True,
)
