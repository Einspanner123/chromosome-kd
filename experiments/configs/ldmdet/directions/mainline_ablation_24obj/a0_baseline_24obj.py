"""24obj 主路线消融实验 DDPM baseline: Baseline (纯 DiffusionDet, 无 RF)

目的: 在 24obj 完整实例标注数据集上建立基线
背景: original 数据集使用 visible-only 标注 (48% 图片框数>46),
      count-prior 理论假设框数=46 不成立; 24obj 数据集 91.6% 图片框数=46,
      符合 count-prior 假设, 适合验证理论

组件: 无 RF, 无 Heun, 无 AdaLN, 无 StochOT
  - sampling_timesteps=1 (DDPM 单步)
  - 默认 Euler 采样
  - 默认 time_conditioning (无 AdaLN)
  - 默认 coupling (random)

对照:
  - RF+Heun: +RF+Heun (验证 RF+Heun 效果)
  - +AdaLN-Zero: +RF+Heun+AdaLN (验证 AdaLN 效果)
  - +Stoch. Coupling: +RF+Heun+AdaLN+StochOT eps5 (完整 SOTA)

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a0_baseline'
"""
_base_ = ['../../ldmdet_baseline.py']

# === 覆盖为 24obj 数据集 ===
# 24obj 使用标准 C-group 顺序 (C6,C7,C8,C9,C10,C11,C12),
# 不同于 original (C10,C11,C12,C6,C7,C8,C9)
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

# 覆盖 dataloader 路径 (保持 baseline 的多尺度 train_pipeline)
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(data_root=data_root),
)
val_dataloader = dict(
    dataset=dict(data_root=data_root),
)
test_dataloader = val_dataloader

# Val 评估器: 启用 classwise 输出 24 个 per-class AP (项目硬约束)
val_evaluator = dict(
    ann_file=data_root + 'valid/_annotations.coco.json',
    classwise=True,
)
test_evaluator = val_evaluator

# SwanLab
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a0_baseline',
            description='24obj 主路线消融 DDPM baseline: Baseline (无RF, Euler 1步) | bs=4, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
