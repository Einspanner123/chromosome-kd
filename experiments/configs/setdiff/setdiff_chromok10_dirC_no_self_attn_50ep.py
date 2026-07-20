"""SetDiff chromo k10 50ep — 方向 C: per-proposal 退化 (禁用 self-attention)

小数据集快速验证版 (2026-07-19):
  24obj 数据集 50ep ETA 约 2 天, warmup 5ep 需 3 小时才能看到 Epoch 6 val mAP
  信号, 太慢. 改用 chromo few-shot k=10 (222 images / 240 annotations) 验证:
    - 约 15x 加速 (111 iter/ep vs 1750 iter/ep)
    - 50ep ETA 约 2.3 小时
    - warmup 5ep 约 14 分钟, 1 小时内可见 Epoch 6-10 val mAP 信号

  ⚠️ 目的: 快速验证 SetDiff 训练管线能否产生 mAP>0 (区分"结构问题"vs"数据
  量问题"), 不是为了追求高 mAP. k=10 数据量适中 (每类 10 ann), 避免极端
  few-shot 的不稳定性 (memory: k=5 时 StochOT 噪声>收益).

  若 k10 验证 mAP>0 → 训练管线 OK, 可推广回 24obj 做正式结果
  若 k10 验证 mAP=0 → 结构问题确认, 不需在 24obj 上浪费时间

方向 C (诊断性消融 + 保底方案, 放弃 joint state 理论区分点):
  禁用 self-attention (用对角 mask 阻断 slot 间交互), 每个 slot 独立处理,
  退化为 per-proposal 范式 (对齐 DiffusionDet).

  ⚠️ 对角 mask ≠ 严格跳过 self-attn 层 (保留 value/out 投影, per-slot
  线性变换), 但满足 "slot 间无信息流动" 的核心需求.

继承 setdiff_baseline.py (而非 setdiff_24obj.py), 因为 baseline 默认就是
chromo 数据集 (C-group order: C10,C11,C12 在 C6 之前), 与 few-shot 同源.
覆盖:
  - train_dataloader.dataset.ann_file: train/_annotations.coco.json
    → train/few_shot_k10.json
  - max_epoch: 150 → 50
  - param_scheduler: T_max=50 (其他保留 baseline 的 warmup 5ep + cosine)
  - EarlyStopping: patience 30 → 15
  - bbox_head: enable_self_attn=False (方向 C 核心配置)
  - SwanLab: project='setdiff-chromok10', experiment_name='setdiff_k10_dirC_no_self_attn_50ep'
"""

_base_ = ['./setdiff_baseline.py']

# === 切换到 few-shot k=10 训练集 ===
# baseline 默认 data_root='data/Chromosome20240904_NoAug_NoResize_coco/'
# (与 few-shot 同源), 仅覆盖 ann_file 即可
train_dataloader = dict(
    dataset=dict(ann_file='train/few_shot_k10.json'),
)

# === 50 epoch 验证 ===
max_epoch = 50
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# 学习率调度: warmup 5ep (start_factor=0.001) + cosine 45ep
# baseline 已是 warmup 5ep + cosine, 这里仅覆盖 T_max=50
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=5,
        end=max_epoch,
        by_epoch=True,
    ),
]

# EarlyStopping: 50ep 内 patience=15 合理
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=15,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]

# === 方向 C: per-proposal 退化 (禁用 self-attention) ===
# enable_self_attn=False: 对角 mask 阻断 slot 间交互, 每个 slot 独立处理.
# matcher_type='hungarian' (默认): 保留原 coupling matching (对照方向 A).
# unmatched_strategy='noise' (默认): 保留原行为.
model = dict(
    bbox_head=dict(
        enable_self_attn=False,
        matcher_type='hungarian',
        unmatched_strategy='noise',
    ),
)

# === SwanLab: 方向 C 标记 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-chromok10',
            experiment_name='setdiff_k10_dirC_no_self_attn_50ep',
            description=(
                'SetDiff chromo k10 方向 C | per-proposal退化 (禁用self-attention) | '
                '放弃joint state理论区分点 | 对齐DiffusionDet | '
                '50ep warmup5ep | 快速验证 (15x加速) | GPU1'
            ),
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
