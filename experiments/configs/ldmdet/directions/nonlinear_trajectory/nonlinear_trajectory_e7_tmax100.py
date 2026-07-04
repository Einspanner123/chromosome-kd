"""E7: 方向A — 余弦退火周期对齐实际训练长度

诊断发现:
  baseline max_epochs=150, T_max=150, 但所有实验在 98-124 epoch 早停.
  LR 从未衰减到 0, 最优解附近 LR 仍占 33% (Epoch 94), 导致无法精细调优.

改动 (仅此一项, 隔离变量):
  max_epochs: 150 → 100
  T_max:      150 → 100

预期:
  Epoch 94 时 LR ≈ 0 (vs 当前 33%), 进入精细调优阶段, 有望突破 0.752.
"""
_base_ = ['nonlinear_trajectory.py']

# ── 训练周期对齐 ──────────────────────────────────
max_epochs = 100
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs)

# ── 余弦退火: T_max 对齐实际训练长度 ──────────────
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=0, begin=5, end=max_epochs, by_epoch=True),
]
