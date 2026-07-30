"""24obj 主路线消融实验 +AdaLN-Zero: +RF+Heun+AdaLN (验证 AdaLN-Zero 效果)

目的: 在 RF+Heun 基础上添加 AdaLN-Zero 时间条件
组件:
  + time_conditioning='adaln_zero' (AdaLN-Zero 时间条件)
  - 无 StochOT (默认 random coupling)

对照: RF+Heun (无 AdaLN) → 验证 AdaLN-Zero 的增益

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a2_rf_heun_adaln'
"""
_base_ = ['../../ldmdet_rf_heun_shifted_bs2.py']

# === 添加 AdaLN-Zero 时间条件 ===
model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning='adaln_zero',
        ),
    ),
)

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
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    classwise=True,
)
test_evaluator = val_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a2_rf_heun_adaln',
            description='24obj 主路线消融 +AdaLN-Zero: +RF+Heun+AdaLN-Zero | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
