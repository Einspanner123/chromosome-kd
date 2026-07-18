"""SetDiff 24obj 20ep 快速验证配置

用于验证 snr_scale + 损失函数方案A + 时间嵌入缩放 修复后是否能产生正向 mAP 趋势.
继承 setdiff_24obj.py, 仅覆盖:
  - max_epoch: 150 → 20 (快速验证)
  - param_scheduler: cosine T_max 同步改为 20
  - EarlyStopping: patience 30 → 10 (20ep 内合理终止)
  - SwanLab experiment_name: 标记为 20ep 验证
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

# === SwanLab: 标记为全修复后的 20ep 验证 ===
# 修复清单: snr_scale + 方案A(2:5:2) + t*1000 + GIoU在[0,1]空间
#          + double-counting(移除loss key) + L1在[0,1]空间
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_full_fix_20ep',
            description='SetDiff 全修复 (snr_scale+方案A+t*1000+GIoU[0,1]+无double-counting+L1[0,1]) | 20ep 快速验证',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
