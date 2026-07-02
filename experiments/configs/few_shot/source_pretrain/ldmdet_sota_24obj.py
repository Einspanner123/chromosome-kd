_base_ = [
    '../../_legacy/'
    'ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py'
]

# LDMDet SOTA baseline (无 feature_bridge)
custom_imports = dict(
    imports=[
        'experiments.mmdet_bridge.registry',
        'experiments.mmdet_bridge.detector',
        'experiments.mmdet_bridge.hooks',
        'experiments.mmdet_bridge.transforms',
        'swanlab.integration.mmengine',
    ],
    allow_failed_imports=False,
)

# === Override to 24obj dataset ===
# 24obj uses standard C-group order (C6,C7,C8,C9,C10,C11,C12)
# unlike the old chromo dataset (C10,C11,C12,C6,C7,C8,C9).
data_root = 'data/24_chromosomes_object/coco/'

classes = (
    'A1', 'A2', 'A3',
    'B4', 'B5',
    'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
    'D13', 'D14', 'D15',
    'E16', 'E17', 'E18',
    'F19', 'F20',
    'G21', 'G22',
    'X', 'Y',
)

METAINFO = dict(
    classes=classes,
    palette=[
        (220, 20, 60), (119, 11, 32), (0, 0, 142),
        (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192),
        (250, 170, 30), (100, 170, 30), (220, 220, 0),
        (175, 116, 175), (250, 0, 30), (165, 42, 42),
        (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255),
        (199, 100, 0), (72, 0, 118),
    ],
)

train_dataloader = dict(
    batch_size=2,
    dataset=dict(data_root=data_root, metainfo=METAINFO),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root, metainfo=METAINFO),
)
val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# SwanLab
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='few-shot-benchmark',
                experiment_name='source_pretrain_ldmdet_sota_24obj',
                description='Few-shot source pretrain: LDMDet SOTA baseline '
                             'on 24obj | bs=2, 150ep',
            ),
        ),
    ],
)
