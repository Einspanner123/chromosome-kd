"""SetDiff 24obj 50ep — 方向 A: coupling 用 pred_boxes matching (DS 路径 A)

方向 A 背景 (2026-07-19, 论文调研结论):
  SetDiff 当前结构 (joint state + global coupled matching + self-attention)
  训练 3 epoch 后 val mAP 仍为 0 (planA/planB 均为 0.000), 梯度归一化修复后
  梯度幅度已恢复正常 (grad_norm=40-170), 但 mAP 不动.

  论文调研发现核心矛盾:
  - JRM (CVPR 2026) 是 joint diffusion 唯一直接先例, 但它是 reconstruction
    (K 已知、无需 matching、slot 天然有语义对应), SetDiff 三个条件都不满足.
  - DN-DETR 明确警告: bipartite matching 不稳定是 slow convergence 根因.
  - SetDiff 的 coupling 阶段每步重新 matching 正是极端版的不稳定.

方向 A (DS 路径 A, 保留 joint state + 理论区分点):
  coupling 阶段用 pred_boxes (第一次 encoder forward 的预测) 做 matching,
  而非 noise. matching 随训练稳定 (DETR 早期 matching 也不稳定, 但随训练收敛).

  实现新增:
  - HungarianMatcher.match_coupling_batch(proposals, noise, gt, labels)
  - JointDiffusionHead(coupling_source='pred_init'): _forward_train 先做
    no_grad forward (t=1.0) 得到 pred_init, 再用 pred_init 做 coupling matching.

  已知张力 (Plan agent A-3):
  coupling matching 用 t=1.0 的 pred_init, loss matching 用 sampled t 的
  pred_boxes, 两者天然不一致. 但比"完全随机(noise)"稳定.

  代价: 2 次 encoder forward, 训练时间约 1.3-1.6x.

对照方向 C (GPU1, setdiff_24obj_dirC_no_self_attn_50ep.py):
  方向 C 禁用 self-attention (per-proposal 退化). 若 A 成功 C 失败 →
  joint state 正确, matching 稳定是关键; 若 A 失败 C 成功 → self-attention
  是根因.

继承 setdiff_24obj.py, 覆盖:
  - max_epoch: 150 → 50
  - param_scheduler: warmup 5ep (start_factor=0.001) + cosine 45ep
    (对齐 LDMDet ldmdet_rf_heun_shifted_bs2.py, 修复 warmup 1ep 的
    scheduler 链 bug: LinearLR end=1 与 CosineAnnealingLR begin=1 重叠
    导致 cosine 错误用 warmup 起始值 5e-6 作为 base_lr, 实际 lr 低 10 倍)
  - EarlyStopping: patience 30 → 15
  - bbox_head: coupling_source='pred_init' (方向 A 核心配置)
  - SwanLab experiment_name: setdiff_dirA_pred_coupling_50ep
"""

_base_ = ['./setdiff_24obj.py']

# === 50 epoch 验证 ===
max_epoch = 50
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# 学习率调度: warmup 5ep (start_factor=0.001) + cosine 45ep
# 修复 scheduler 链 bug (2026-07-19): warmup 1ep 时 LinearLR end=1 与
# CosineAnnealingLR begin=1 重叠, cosine 错误用 warmup 起始值 5e-6 作为
# base_lr, 实际 lr 低 10 倍 (4.88e-6 vs 预期 4.4e-5). 对齐 LDMDet
# ldmdet_rf_heun_shifted_bs2.py 的 warmup 5ep 配置 (已验证 lr 正确过渡到 5e-5).
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

# === 方向 A: coupling 用 pred_boxes matching ===
# coupling_source='pred_init': 第一次 encoder forward (no_grad, t=1.0)
# 得到 pred_init, 用 pred_init 做 coupling matching.
# matcher_type='hungarian' (默认): 保留 global coupled matching 理论区分点.
# unmatched_strategy='noise' (默认): unmatched slot 的 x_start = noise.
model = dict(
    bbox_head=dict(
        coupling_source='pred_init',
        matcher_type='hungarian',
        unmatched_strategy='noise',
    ),
)

# === SwanLab: 方向 A 标记 ===
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='setdiff-24obj',
            experiment_name='setdiff_dirA_pred_coupling_50ep',
            description=(
                'SetDiff 方向 A | coupling用pred_boxes匹配 (DS路径A) | '
                '保留joint state+理论区分点 | 2次encoder forward | '
                '50ep warmup5ep (修复scheduler bug) | GPU0'
            ),
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
