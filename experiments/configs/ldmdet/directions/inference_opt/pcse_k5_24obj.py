"""方向R PCSE 配对一致性随机集成评分 — 24obj 数据集

目的: 验证 PCSE 多假设集成评分在 24obj 上的 mAP 增益
方案:
  - 生成 K=5 个独立假设（不同随机种子）
  - 用核型先验（数量、配对、形态、性染色体）对假设评分
  - 选择评分最优的假设输出
  - 纯推理优化, 无需重训练, 不改训练动态

理论依据:
  - 扩散采样随机性产生不同质量的检测假设
  - 核型先验（46条、22对+XX/XY）提供强全局约束
  - 评分融合置信度 + 先验一致性, 比纯置信度更鲁棒

预期:
  - mAP gain: +0.005~+0.015 (保守估计)
  - 推理时间: ~2.5x (K=5, 特征复用)

SwanLab: 项目 'ldmdet-inference-opt-24obj', 实验 'pcse_k5'
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

# === PCSE: 配对一致性随机集成评分 ===
model = dict(
    bbox_head=dict(
        use_pcse=True,
        pcse_k=5,
        pcse_lambda=0.5,
        pcse_alphas=(1.0, 2.0, 1.5, 1.5),
        pcse_diversity='seed',
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference-opt-24obj',
            experiment_name='pcse_k5',
            description='方向R PCSE 配对一致性随机集成评分 (K=5, λ=0.5, α=(1,2,1.5,1.5)) | 24obj, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
