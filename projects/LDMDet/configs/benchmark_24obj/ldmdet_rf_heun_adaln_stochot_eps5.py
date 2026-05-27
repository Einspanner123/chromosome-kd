_base_ = ['../recipes/rf_heun_adaln_stochot_eps5.py']

# === Override to benchmark_24obj dataset ===
# The 24obj dataset uses standard C-group order (C6,C7,C8,C9,C10,C11,C12)
# unlike the old dataset (C10,C11,C12,C6,C7,C8,C9).
# data_root also differs from the old chromo_coco_detection default.

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

# Update dataloader paths to 24obj data_root
train_dataloader = dict(
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root),
)
val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
)

# SwanLab project name for 24obj benchmark
visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-benchmark-24obj',
                experiment_name='ldmdet-rf-adaln-stochot-eps5',
                description='Benchmark 24obj: LDMDet RF+AdaLN+StochasticOT eps=5 | bs=8, 150ep',
            ),
        ),
    ],
)
