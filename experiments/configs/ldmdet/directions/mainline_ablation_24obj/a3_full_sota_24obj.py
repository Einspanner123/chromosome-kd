"""24obj 主路线消融实验 +Stoch. Coupling: 完整 SOTA (RF+Heun+AdaLN+StochOT eps5)

目的: 在 +AdaLN-Zero 基础上添加 StochasticOT eps=5, 构成完整主路线 SOTA
组件:
  + coupling='ot_flow', epsilon=5.0, num_iters=20, coupling_mode='multinomial'

对照:
  - +AdaLN-Zero (无 StochOT) → 验证 StochOT 的增益
  - benchmark_24obj/ldmdet_rf_heun_adaln_stochot_eps5.py (同配置, 无 classwise)

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a3_full_sota'
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

# Val 评估器: 启用 classwise 输出 24 个 per-class AP (项目硬约束)
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
            experiment_name='a3_full_sota',
            description='24obj 主路线消融 +Stoch. Coupling: 完整SOTA (RF+Heun+AdaLN+StochOT eps5) | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
