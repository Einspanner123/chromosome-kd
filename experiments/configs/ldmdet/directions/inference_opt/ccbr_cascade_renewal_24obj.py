"""方向U CCBR 跨级联框更新 — 24obj 数据集

目的: 验证 CCBR 级联间软 renewal + 后向置信度传播的 mAP 增益
方案:
  - 6级级联Head间引入软box_renewal (非纯噪声替换)
  - 利用上一时间步 Head6 置信度指导当前时间步级联间更新
  - 自适应阈值: 后面Head更自信 → 降低renewal阈值, 保留更多框
  - 保持 cascade_detach=True, 纯推理优化

理论依据:
  - box_renewal 是扩散检测最强特有机制 (+0.016 mAP)
  - 级联间 renewal 扩展探索维度到级联空间
  - 后向置信度传播解决级联"信息孤岛"问题

预期:
  - mAP gain: +0.005~+0.015
  - 推理时间: 几乎无增加 (仅在级联间加扰动)

SwanLab: 项目 'ldmdet-inference-opt-24obj', 实验 'ccbr_cascade_renewal'
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

# === CCBR: 跨级联框更新 ===
model = dict(
    bbox_head=dict(
        use_ccbr=True,
        ccbr_alpha=0.7,
        ccbr_sigma=0.1,
        ccbr_lambda=0.5,
        ccbr_beta=0.5,
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference-opt-24obj',
            experiment_name='ccbr_cascade_renewal',
            description='方向U CCBR 跨级联框更新 (α=0.7, σ=0.1, λ=0.5, β=0.5) | 24obj, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
