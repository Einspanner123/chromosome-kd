"""SetDiff 24obj 50ep — 方案 B + loss 归一化修复

验证修复 (2026-07-19):
  修复根因: 旧 planA/B SetCriterion 直接用 coupling 阶段的 matched_labels
  (num_pos=300), 导致 per-slot 梯度稀释 37.5 倍. 修复后 SetCriterion 内部
  用 Hungarian 重新匹配 pred_boxes 和 GT, num_pos=M (不是 N=300).

方案 B vs 方案 A:
  - 方案 A: 完全放弃 Hungarian coupling, 所有 slot 随机分配 (无理论区分点)
  - 方案 B: 保留 Hungarian global coupled matching (v4 理论区分点),
    unmatched slot 分配随机 GT (训练-推理分布对齐)
  - 两者并行训练, 比较哪个更优 (理论上方案 B 应保留 v4 贡献)

本配置调整 (与 planA_50ep_fix 一致):
  - max_epoch: 20 → 50
  - warmup: 5ep → 1ep (start_factor=0.1)
  - EarlyStopping: patience 10 → 15
  - SwanLab experiment_name: setdiff_planB_50ep_fix

继承 setdiff_24obj.py, 覆盖:
  - max_epoch: 150 → 50
  - param_scheduler: warmup 1ep (start_factor=0.1) + cosine 49ep
  - EarlyStopping: patience 10 → 15
  - bbox_head: matcher_type='hungarian', unmatched_strategy='random_gt' (方案 B)
  - SwanLab experiment_name: setdiff_planB_50ep_fix
"""

_base_ = ['./setdiff_24obj.py']

# === 50 epoch 验证 ===
max_epoch = 50
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# 学习率调度: warmup 1ep (start_factor=0.1) + cosine 49ep
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1),
    dict(
        type='CosineAnnealingLR',
        T_max=max_epoch,
        eta_min=0,
        begin=1,
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

# === 方案 B: Hungarian + random_gt unmatched strategy ===
# 保留 Hungarian global coupled matching (v4 理论区分点),
# unmatched slot 分配随机 GT (训练-推理分布对齐).
# loss 阶段 SetCriterion 内部 Hungarian 重新匹配 (num_pos=M, 不稀释).
model = dict(
    bbox_head=dict(
        matcher_type='hungarian',
        unmatched_strategy='random_gt',
    ),
)

# === SwanLab: 方案 B + 修复标记 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_planB_50ep_fix',
            description=(
                'SetDiff 方案 B | loss归一化修复 (num_pos=M, 不稀释) | '
                '50ep warmup1ep | Hungarian+random_gt (保留v4区分点)'
            ),
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
