"""IO3 Top-K 框剪枝推理优化 — 24obj 数据集

目的: 验证 Top-K 框剪枝在 24obj 上的 mAP 影响和推理加速
方案:
  - 第 0 步后保留 Top-K=100 高置信框, 丢弃其余 400 个
  - 后续步 (1-3) 仅处理 100 个 proposal
  - RoIAlign (O(N)) 和 DynamicConv (O(N)) 计算量降低 5x
  - Box renewal 在剪枝后仍运行 (100 框中替换低置信为噪声)

理论依据:
  - 24obj 每图 ~46 GT, 500 proposal 中 90%+ 冗余
  - 第 0 步后分类置信度已具判别力, Top-K 可有效保留 GT 对应框
  - 实测预期: N=100 整体加速 ~1.8x, mAP 降 <0.003

SwanLab: 项目 'ldmdet-inference-opt-24obj', 实验 'io3_topk_pruning'
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

# === IO3: Top-K 框剪枝 ===
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=100,
        topk_pruning_step=0,
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-inference-opt-24obj',
            experiment_name='io3_topk_pruning',
            description='IO3 Top-K框剪枝 (K=100, step0后剪枝) | 24obj, bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
