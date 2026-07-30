"""D2 (24 Chromosomes Object) 跨数据集训练配置

基于 0.753 SOTA 模型配置 (ldmdet_rf_heun_adaln_stochot_eps5.py),
将数据集从 D1 (Chromosome20240904_NoAug_NoResize_coco) 切换到 D2 (24_chromosomes_object)。

模型配置与 0.753 SOTA 完全一致:
  - RF + Heun solver + Shifted schedule (shift=3.0)
  - AdaLN-Zero 时间条件
  - Stochastic OT coupling (ot_flow, epsilon=5, sinkhorn 20 iters, multinomial)
  - bs=2, AdamW lr=5e-5, 150 epochs, CosineAnnealing

种子: 2016452323 (从 0.753 原始 checkpoint 元数据恢复;
      原配置未设置 randomness, mmengine 默认 seed=None 触发
      sync_random_seed() 生成此随机种子)

D2 类别顺序 (数字序, 与 D1 的字母序不同):
  RF+Heun +AdaLN-Zero +Stoch. Coupling B4 B5 C6 C7 C8 C9 C10 C11 C12
  D13 D14 D15 E16 E17 E18 F19 F20 G21 G22 X Y

D2 无单独 val, 用 test 作为 val。
"""
_base_ = ['./ldmdet_rf_heun_adaln_stochot_eps5.py']

# === 随机种子 (从 0.753 checkpoint 元数据恢复) ===
# 注意: experiments/runners/train.py 的 --seed 只影响 work_dir 名,
# 并未实际设置 mmengine 训练种子 (RANDOM_SEED env var 未被消费)。
# 必须在配置中显式设置 randomness.seed 才能真正生效。
randomness = dict(seed=2016452323, deterministic=False, diff_rank_seed=False)

# === D2 数据集 ===
data_root = 'data/24_chromosomes_object/coco/'

# D2 类别顺序 (数字序)
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

# D2 无单独 val, 用 test 作为 val
test_ann_file = data_root + 'test/_annotations.coco.json'

# === 训练数据 ===
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=METAINFO,
    ),
)

# === Val 数据 (用 test 作为 val) ===
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
    ),
)

# === Test 数据 ===
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=METAINFO,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
    ),
)

# === Evaluator (format_only=False 以便看到 mAP) ===
val_evaluator = dict(
    ann_file=test_ann_file,
    format_only=False,
    classwise=True,
)

test_evaluator = dict(
    ann_file=test_ann_file,
    format_only=False,
    classwise=True,
)

# === SwanLab ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-cross-dataset',
            experiment_name='d2_0753_stochot_eps5_seed2016452323',
            description='D2 跨数据集: 0.753 SOTA 配置 (RF+Heun+AdaLN+StochOT eps5) | seed=2016452323 | bs=2, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
