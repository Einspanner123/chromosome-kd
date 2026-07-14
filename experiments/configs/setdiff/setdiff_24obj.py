"""SetDiff 24obj 配置 — 24 chromosomes object detection

覆盖 setdiff_baseline.py 为 24obj 数据集:
  - data_root: data/24_chromosomes_object/coco/
  - classes: 24obj 类别顺序 (C6-C12 数值序)
  - val_evaluator: 启用 classwise 输出 24 per-class AP (项目硬约束)
  - SwanLab: project='setdiff-24obj'

参考: experiments/configs/ldmdet/directions/mainline_ablation_24obj/a3_full_sota_24obj.py
"""
_base_ = ['./setdiff_baseline.py']

# === 覆盖为 24obj 数据集 ===
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
    dataset=dict(data_root=data_root, metainfo=METAINFO),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root, metainfo=METAINFO),
)
test_dataloader = val_dataloader

# Val 评估器: 启用 classwise 输出 24 per-class AP (项目硬约束)
val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    classwise=True,
)
test_evaluator = val_evaluator

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_baseline',
            description='SetDiff baseline (Coupled State Diffusion) | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
