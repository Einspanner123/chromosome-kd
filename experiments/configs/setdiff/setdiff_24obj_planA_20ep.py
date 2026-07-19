"""SetDiff 24obj 20ep — 方案 A: RandomMatcher (所有 slot 随机分配 GT)

验证假设: mAP=0 根因是 unmatched slot 训练-推理分布不匹配.
方案 A: 放弃 Hungarian global coupled matching, 所有 N 个 slot 从 M 个 GT
中独立均匀采样 (允许重复), 对齐 LDMDet `_couple_single_image` 的
`torch.randint(0, num_gt, (num_proposals,))`.

核心区别 vs baseline:
  - matcher_type='random' (替代默认 'hungarian')
  - 所有 slot 在 [GT, noise] 插值轨迹上, box_head 对所有 slot 有监督
  - 训练-推理分布对齐 (推理时所有 slot 从 noise 出发, 与训练 unmatched 分布一致)

代价: 失去 global coupled matching (理论 v4 的核心区分点).
若方案 A 成功 (mAP > 0), 说明 mAP=0 根因确实是 unmatched slot 无监督,
后续可继续验证方案 B 保留 Hungarian 的 random_gt 策略.

继承 setdiff_24obj.py, 覆盖:
  - max_epoch: 150 → 20 (快速验证)
  - param_scheduler: cosine T_max 同步改为 20
  - EarlyStopping: patience 30 → 10
  - bbox_head: matcher_type='random'
  - SwanLab experiment_name: setdiff_planA_random_20ep
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

# === 方案 A: RandomMatcher ===
# 所有 slot 随机分配 GT, 对齐 LDMDet _couple_single_image.
# box_renewal 保持 True (推理时仍重置低置信 slot, 但训练时所有 slot 已有监督).
model = dict(
    bbox_head=dict(
        matcher_type='random',
    ),
)

# === SwanLab: 方案 A 标记 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_planA_random_20ep',
            description='SetDiff 方案 A | RandomMatcher: 所有slot随机分配GT, 对齐LDMDet | 20ep | GPU0',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
