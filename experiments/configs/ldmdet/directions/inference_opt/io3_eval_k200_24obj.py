"""IO3 推理评估 — K=200 (在 SOTA checkpoint 上直接评估, 不训练)

目的: 测试 K=200 剪枝在 SOTA 模型上的 mAP 影响
方法: 加载 a3_full_sota checkpoint, 推理时启用 topk_pruning K=200
预期: mAP 下降 <0.005, 加速 ~37%
"""
_base_ = ['../../ldmdet_rf_heun_adaln_stochot_eps5.py']

# 24obj 数据集
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

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    classwise=True,
)
test_evaluator = val_evaluator

# IO3: Top-K 剪枝 K=200
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=200,
        topk_pruning_step=0,
    ),
)
