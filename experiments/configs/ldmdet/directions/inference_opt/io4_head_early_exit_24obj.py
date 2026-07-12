"""IO4 级联头提前退出推理优化 — 24obj 数据集

目的: 验证级联头提前退出在 24obj 上的 mAP 影响和推理加速
方案:
  - 6 级级联头中, 从第 3 头开始检测收敛
  - 框对角线归一化 L2 变化 < threshold AND 分类 argmax 一致率 > threshold → 收敛
  - 收敛后提前退出, 剩余头用最后一头结果填充
  - 时间步感知: 早期步严格阈值(少退出), 后期步宽松阈值(多退出)
  - 仅推理时启用, 训练时保持完整 6 级 (deep supervision 不受影响)

理论依据:
  - Cascade R-CNN: 级联头逐级精化, 后期头冗余
  - BranchyNet: 分支提前退出
  - LDMDet: 后期时间步框已稳定, 前 3 头即收敛

SwanLab: 项目 'ldmdet-inference-opt-24obj', 实验 'io4_head_early_exit'
"""
_base_ = ['../../ldmdet_rf_heun_adaln_stochot_eps5.py']

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

# === IO4: 级联头提前退出 ===
model = dict(
    bbox_head=dict(
        head_early_exit_enabled=True,
        head_exit_box_threshold=0.005,
        head_exit_cls_threshold=0.98,
        head_exit_min_heads=3,
        head_exit_max_heads=6,
        head_exit_time_aware=True,
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference-opt-24obj',
            experiment_name='io4_head_early_exit',
            description='IO4 级联头提前退出 (min=3, box=0.005, cls=0.98, time_aware) | 24obj, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
