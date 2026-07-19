"""SetDiff 24obj 50ep — 方向 C: per-proposal 退化 (禁用 self-attention)

方向 C 背景 (2026-07-19, 论文调研结论):
  SetDiff 当前结构 (joint state + global coupled matching + self-attention)
  训练 3 epoch 后 val mAP 仍为 0 (planA/planB 均为 0.000), 梯度归一化修复后
  梯度幅度已恢复正常 (grad_norm=40-170), 但 mAP 不动.

  论文调研发现核心矛盾:
  - joint state 要求 slot 语义稳定, 但 coupling 阶段每步重匹配导致 slot
    角色漂移. self-attention 在 slot 角色不稳定时可能引入 ROT (DS-Det 警告).
  - DiffusionDet 用 per-proposal 扩散 (每个 proposal 独立), 已验证能 work.

方向 C (诊断性消融 + 保底方案, 放弃 joint state 理论区分点):
  禁用 self-attention (用对角 mask 阻断 slot 间交互), 每个 slot 独立处理,
  退化为 per-proposal 范式 (对齐 DiffusionDet).

  实现新增:
  - SetEncoder(enable_self_attn=False): 对角 mask 阻断 inter-slot 交互.
  - JointDiffusionHead 透传 enable_self_attn 参数.

  ⚠️ 对角 mask ≠ 严格跳过 self-attn 层 (保留 value/out 投影, per-slot
  线性变换), 但满足 "slot 间无信息流动" 的核心需求.

  方向 C 下 box_renewal 无必要 (self-attention 污染机制不存在), 但保留不删
  (对照实验, 验证 box_renewal 是否仍有作用).

对照方向 A (GPU0, setdiff_24obj_dirA_pred_coupling_50ep.py):
  方向 A 保留 joint state, 用 pred_boxes 做 coupling matching. 若 A 成功
  C 失败 → joint state 正确, matching 稳定是关键; 若 A 失败 C 成功 →
  self-attention 是 mAP=0 根因, 需放弃 joint state.

继承 setdiff_24obj.py, 覆盖:
  - max_epoch: 150 → 50
  - param_scheduler: warmup 5ep (start_factor=0.001) + cosine 45ep
    (对齐 LDMDet ldmdet_rf_heun_shifted_bs2.py, 修复 warmup 1ep 的
    scheduler 链 bug: LinearLR end=1 与 CosineAnnealingLR begin=1 重叠
    导致 cosine 错误用 warmup 起始值 5e-6 作为 base_lr, 实际 lr 低 10 倍)
  - EarlyStopping: patience 30 → 15
  - bbox_head: enable_self_attn=False (方向 C 核心配置)
  - SwanLab experiment_name: setdiff_dirC_no_self_attn_50ep
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

# === 方向 C: per-proposal 退化 (禁用 self-attention) ===
# enable_self_attn=False: 对角 mask 阻断 slot 间交互, 每个 slot 独立处理.
# matcher_type='hungarian' (默认): 保留原 coupling matching (对照方向 A).
# unmatched_strategy='noise' (默认): 保留原行为.
# box_renewal=True (默认): 保留 (方向 C 下无必要, 但对照实验不删).
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
            project='setdiff-24obj',
            experiment_name='setdiff_dirC_no_self_attn_50ep',
            description=(
                'SetDiff 方向 C | per-proposal退化 (禁用self-attention) | '
                '放弃joint state理论区分点 | 对齐DiffusionDet | '
                '50ep warmup5ep (修复scheduler bug) | GPU1'
            ),
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
