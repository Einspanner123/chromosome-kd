"""SetDiff 24obj 20ep — 方案 B: Hungarian + random_gt unmatched strategy

验证假设: mAP=0 根因是 unmatched slot 训练-推理分布不匹配.
方案 B: 保留 Hungarian global coupled matching (matched slot 最优一对一),
unmatched slot 从 GT 集合随机采样 (允许重复), 保证所有 slot 有监督.

核心区别 vs baseline:
  - unmatched_strategy='random_gt' (替代默认 'noise')
  - matched slot (M 个): Hungarian 一对一最优匹配 (保留理论 v4 区分点)
  - unmatched slot (N-M 个): 随机 GT 分配 (保证 box_head 监督)
  - 训练-推理分布对齐

方案 B vs 方案 A:
  - 方案 A: 完全放弃 Hungarian, 所有 slot 随机分配
  - 方案 B: 保留 Hungarian 的 global coupled matching, 仅修复 unmatched
  - 若两者都成功, 方案 B 更优 (保留理论区分点)
  - 若方案 A 成功方案 B 失败, 说明 Hungarian 匹配本身有问题
  - 若两者都失败, 说明根因不在 unmatched slot (需要重新诊断)

继承 setdiff_24obj.py, 覆盖:
  - max_epoch: 150 → 20 (快速验证)
  - param_scheduler: cosine T_max 同步改为 20
  - EarlyStopping: patience 30 → 10
  - bbox_head: unmatched_strategy='random_gt'
  - SwanLab experiment_name: setdiff_planB_random_gt_20ep
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

# === 方案 B: Hungarian + random_gt unmatched strategy ===
# 保留 Hungarian global coupled matching, unmatched slot 分配随机 GT.
# box_renewal 保持 True (推理时仍重置低置信 slot, 但训练时所有 slot 已有监督).
model = dict(
    bbox_head=dict(
        matcher_type='hungarian',
        unmatched_strategy='random_gt',
    ),
)

# === SwanLab: 方案 B 标记 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_planB_random_gt_20ep',
            description='SetDiff 方案 B | Hungarian+random_gt: 保留global coupled matching, unmatched分配随机GT | 20ep | GPU1',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
