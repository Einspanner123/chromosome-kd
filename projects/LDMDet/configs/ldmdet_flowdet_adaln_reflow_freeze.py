_base_ = ["./ldmdet_flowdet_adaln_reflow.py"]

model = dict(
    bbox_head=dict(
        reflow_det_loss_scale=1.0,
        velocity_loss_weight=1.0,
        reflow_velocity_warmup_steps=0,
        freeze_shared=True,
    ),
)

load_from = "work_dirs/ldmdet_flowdet_adaln_reflow_v5/best_coco_bbox_mAP_epoch_1.pth"

max_epoch = 10
train_cfg = dict(max_epochs=max_epoch)

optim_wrapper = dict(optimizer=dict(type="AdamW", lr=0.000005, weight_decay=0.0001))

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=True, begin=0, end=1),
    dict(
        type="CosineAnnealingLR",
        T_max=max_epoch,
        eta_min=0,
        begin=1,
        end=max_epoch,
        by_epoch=True,
    ),
]

custom_hooks = [
    dict(type="CopyProjectHook", priority="VERY_LOW"),
]
