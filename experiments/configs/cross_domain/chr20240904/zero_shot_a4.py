"""跨域 Zero-shot 评估: LDMDet A4 (DPM-Solver++) on Chromosome20240904 (Dataset 1).

源域: Dataset 2 = 24 Chromosomes Object (mAP=0.863)
目标域: Dataset 1 = Chromosome20240904 test set (220 imgs, 10262 instances, 无微调)
checkpoint: work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth

注: 论文中 Dataset 1 = Chromosome20240904 (1540 imgs total, 220 test),
而非 AutoKary2022 (118 test imgs). 本配置修正了 3-D 阶段的标识错误。

Usage:
    /home/linkst/data/miniconda3/envs/chromo/bin/python experiments/runners/test.py \\
        experiments/configs/cross_domain/chr20240904/zero_shot_a4.py \\
        --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth \\
        --dataset test --seed 42 --gpu-id 0 --exp-name zero_shot_a4_chr20240904
"""
_base_ = [
    '../../ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py',
]

# === 覆盖为 Chromosome20240904 (Dataset 1) ===
data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'

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
            project='cross-domain-chr20240904',
            experiment_name='zero_shot_a4',
            description='Zero-shot: LDMDet A4 (24obj-trained) on Chromosome20240904 test (Dataset 1)',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
