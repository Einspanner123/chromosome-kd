"""SetDiff 24obj 50ep — 方案 A + loss 归一化修复

验证修复 (2026-07-19):
  修复根因: 旧 planA/B SetCriterion 直接用 coupling 阶段的 matched_labels
  (num_pos=300), 导致 per-slot 梯度稀释 37.5 倍 (0.000733 vs LDMDet
  baseline 0.025125). 修复后 SetCriterion 内部用 Hungarian 重新匹配
  pred_boxes 和 GT, num_pos=M (不是 N=300).

  诊断结论: 梯度稀释 + warmup 阶段 lr 太低 (5e-8) + clip_grad=1.0 严重裁剪
  导致 box_head 模式坍缩 (输出几乎相同的"中心大框"), hs 区分度近乎随机初始化.

本配置调整:
  - max_epoch: 20 → 50 (给 cosine annealing 更多收敛空间)
  - warmup: 5ep → 1ep (避免 lr 长期停留在 5e-8, 修复后梯度已健康可直接训)
  - start_factor: 0.001 → 0.1 (1ep warmup 内从 0.1*lr 升到 lr, 不再过度衰减)
  - EarlyStopping: patience 10 → 15 (50ep 内合理终止)
  - SwanLab experiment_name: setdiff_planA_50ep_fix

注: clip_grad=1.0 保留 (修复后梯度 norm 在 8-12 范围, clip 后 1.0 仍合理;
若 Epoch 1 仍无信号可考虑放宽到 5.0).

继承 setdiff_24obj.py, 覆盖:
  - max_epoch: 150 → 50
  - param_scheduler: warmup 1ep (start_factor=0.1) + cosine 49ep
  - EarlyStopping: patience 10 → 15
  - bbox_head: matcher_type='random' (方案 A)
  - SwanLab experiment_name: setdiff_planA_50ep_fix
"""

_base_ = ['./setdiff_24obj.py']

# === 50 epoch 验证 ===
max_epoch = 50
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# 学习率调度: warmup 1ep (start_factor=0.1) + cosine 49ep
# 修复后梯度健康 (box_head avg grad 1.27, 不再稀释), 可快速进入正常 lr.
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

# === 方案 A: RandomMatcher ===
# 所有 slot 随机分配 GT (训练-推理分布对齐), loss 阶段 Hungarian 重新匹配
# (num_pos=M, 不稀释).
model = dict(
    bbox_head=dict(
        matcher_type='random',
    ),
)

# === SwanLab: 方案 A + 修复标记 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_planA_50ep_fix',
            description=(
                'SetDiff 方案 A | loss归一化修复 (num_pos=M, 不稀释) | '
                '50ep warmup1ep | RandomMatcher'
            ),
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
