"""跨域 Zero-shot 评估: LDMDet +DPM-Solver++ (DPM-Solver++) on AutoKary

源域: 24obj (mAP=0.863)
目标域: AutoKary test set (无微调)
checkpoint: work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth

Usage:
    python experiments/runners/test.py \\
        experiments/configs/cross_domain/autokary/zero_shot_a4.py \\
        --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth
"""
_base_ = [
    '../../ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
]

# === 覆盖为 AutoKary 数据集 (dict merge: 保留 metainfo/pipeline, 仅覆盖路径) ===
data_root = 'data/AutoKary2022_coco/'

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))

# test_dataloader: 覆盖为 test/ 目录 (base 中 test_dataloader = val_dataloader 指向 valid/)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
    ),
)

val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    classwise=True,
)
test_evaluator = dict(
    ann_file=data_root + 'test/_annotations.coco.json',
    format_only=False,
    classwise=True,
)

# SwanLab
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='cross-domain-autokary',
            experiment_name='zero_shot_a4',
            description='Zero-shot: LDMDet +DPM-Solver++ (24obj-trained) on AutoKary test',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
