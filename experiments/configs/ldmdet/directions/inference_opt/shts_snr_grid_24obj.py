"""方向V SHTS SNR层级时间步采样 — 24obj 数据集

目的: 验证 SHTS 自适应 SNR 网格在 24obj 上的 mAP 增益
方案:
  - 基于 SNR 导数密度函数 rho(t) = |d ln SNR / dt| * w(t)
  - 在 SNR≈1 的临界区加密采样 (高斯窗口 w(t))
  - 在 t→0 和 t→1 两端也加密 (低SNR精细定位 + 高SNR分类)
  - 非均匀网格传递给 DPM-Solver++, 与 Heun 协同
  - 纯推理优化, 不改训练

理论依据:
  - RF SNR(t) = ((1-t)/t)², 密度与变化率成正比
  - 分类决策在 SNR≈1 处最关键
  - DPM-Solver++ 的差商公式天然兼容非均匀网格

预期:
  - mAP gain: +0.003~+0.010 (尤其在少步场景)
  - 推理时间: 无增加 (仅改时间步位置)

SwanLab: 项目 'ldmdet-inference-opt-24obj', 实验 'shts_snr_grid'
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

# === SHTS: SNR 层级时间步采样 ===
model = dict(
    bbox_head=dict(
        rf_schedule='shts',
        shts_alpha=1.0,
        shts_sigma=0.15,
        shts_shifted=False,
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference-opt-24obj',
            experiment_name='shts_snr_grid',
            description='方向V SHTS SNR层级时间步采样 (α=1.0, σ=0.15, shifted=False, 4步) | 24obj, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
