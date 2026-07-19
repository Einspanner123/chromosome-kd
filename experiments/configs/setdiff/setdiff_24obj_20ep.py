"""SetDiff 24obj 20ep 快速验证配置 — Box Renewal 实验

验证假设: unmatched slot 在推理时发散导致 mAP=0.
修复: 推理时每步 Euler 后, 低置信 slot 重置为 randn (box_renewal),
保留高置信 slot 继续迭代. 对齐 LDMDet apply_box_renewal.

继承 setdiff_24obj.py, 仅覆盖:
  - max_epoch: 150 → 20 (快速验证)
  - param_scheduler: cosine T_max 同步改为 20
  - EarlyStopping: patience 30 → 10 (20ep 内合理终止)
  - bbox_head: 显式启用 box_renewal (score_thr=0.3, min_keep=75)
  - SwanLab experiment_name: 标记为 box_renewal 验证
"""

_base_ = ['./setdiff_24obj.py']

# === 快速验证: 20 epoch ===
max_epoch = 20
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# 学习率调度同步调整 (warmup 5ep + cosine 15ep)
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

# EarlyStopping: 20ep 内 patience=10 合理
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        priority=50,
        patience=10,
        min_delta=0.001,
        monitor='coco/bbox_mAP',
        rule='greater',
    ),
    dict(type='CopyProjectHook', priority='VERY_LOW'),
]

# === Box Renewal: 显式配置 (验证 unmatched slot 发散假设) ===
# 核心修复: 推理时低置信 slot 重置为 randn, 回到训练分布 (noise),
# 避免发散到 OOD 后通过 self-attention 污染 matched slot.
model = dict(
    bbox_head=dict(
        box_renewal=True,
        score_thr=0.3,
        min_keep=75,  # num_queries//4 = 300//4
    ),
)

# === SwanLab: 标记为 box_renewal 验证实验 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_box_renewal_20ep',
            description='SetDiff Box Renewal 验证 | 修复: 推理时低置信slot重置为randn, 验证unmatched发散假设 | 20ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
