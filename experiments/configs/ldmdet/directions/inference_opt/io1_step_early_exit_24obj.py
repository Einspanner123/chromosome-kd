"""IO1 自适应步数提前终止推理优化 — 24obj 数据集

目的: 验证步级提前终止在 24obj 上的 mAP 影响和推理加速
方案:
  - Heun 4步采样 (8 NFE) 中, 从第 2 步开始检测 x0_pred 收敛
  - 收敛判据: 相对 L2 变化 < 0.01 AND 高置信度框 argmax 一致率 > 0.95
  - 收敛后提前终止, 剩余步用最后结果填充 ensemble
  - 仅推理时启用, 训练不受影响
  - 最少执行 1 步 (不跳第 0 步), 最后一步不检测 (退出无意义)

理论依据:
  - Rectified Flow 直线路径: t→0 时 x_t → x_0, x0_pred 趋于稳定
  - DeepCache (CVPR 2024): 相邻步特征高度相似
  - DDIM (ICLR 2021): 少步采样在简单样本上已接近收敛

预期:
  - 平均 NFE: 8.0 → ~5.2 (省 ~35%)
  - mAP 下降 < 0.002

SwanLab: 项目 'ldmdet-inference-opt-24obj', 实验 'io1_step_early_exit'
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

# === IO1: 自适应步数提前终止 ===
model = dict(
    bbox_head=dict(
        step_early_exit_enabled=True,
        step_exit_threshold=0.01,
        step_exit_cls_threshold=0.95,
        step_exit_min_steps=1,
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference-opt-24obj',
            experiment_name='io1_step_early_exit',
            description='IO1 自适应步数提前终止 (thr=0.01, cls=0.95, min=1) | 24obj, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
