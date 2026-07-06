"""ChromoGen Phase 1 v2 (SD-1.5 fine-tune): 24_chromosomes_object + 优化方案

针对染色体与自然图像域差异大的特性，采用分阶段训练 + 分组学习率 + EDM Karras + EMA Warmup。

主方案：分阶段训练（方案 2）
  - Stage 1（前 10 epoch）：冻结 UNet，仅训练 Condition Encoder（lr=5e-4）
    解决"UNet 已收敛但 CondEncoder 欠拟合"的根因
  - Stage 2（后 90 epoch）：解冻 UNet，分组 LR 联合训练
    UNet 5e-5（保护 SD-1.5 先验），CondEncoder 5e-4（继续学习）

附加优化项：
  - CFG Dropout 0.1 → 0.2（增强 conditional/unconditional 平衡）
  - AdamW β2 0.999 → 0.95（加速后期梯度收敛）
  - EDM Karras Schedule 替换 linear + epsilon（医学图像上表现更好）
  - EMA Warmup（decay 从 0.999 渐进到 0.9999，提升后期稳定性）

训练效率优化：
  - batch_size 8 → 16（RTX A6000 48GB 显存富余）
  - 降低 max_epochs 100 → 50（fine-tune 收敛更快）
"""

_base_ = ['./chromogen_phase1_sd15_24obj.py']

# ============================================================
# 主方案：分阶段训练
# ============================================================
# Stage 1: 前 10 epoch 冻结 UNet，仅训练 Condition Encoder
# Stage 2: 后续 epoch 解冻 UNet，联合 fine-tune
freeze_unet_epochs = 10
max_epochs = 50  # fine-tune 收敛更快，50 epoch 足够

# ============================================================
# 分组学习率
# ============================================================
# UNet: 5e-5（保护 SD-1.5 先验，相对 v1 提高 5x 加速域适应）
# Condition Encoder: 5e-4（10x UNet，加速从零训练的 CondEncoder 学习）
unet_learning_rate = 5e-5
cond_encoder_learning_rate = 5e-4

# ============================================================
# 附加优化项
# ============================================================
# CFG Dropout 提高到 0.2（增强 conditional/unconditional 平衡）
# 注意：cfg_dropout 是 classifier-free guidance dropout（训练时随机置零条件），
# 与 condition_dropout（encoder 内部 nn.Dropout）不同。
cfg_dropout = 0.2

# AdamW β2 0.999 → 0.95（加速后期梯度收敛，避免被旧梯度拖累）
adam_beta2 = 0.95

# EDM Karras Schedule（医学图像上比 linear 表现更好）
noise_schedule = 'karras'
prediction_type = 'v_prediction'  # EDM 推荐 v_prediction

# EMA Warmup（decay 从 0.999 渐进到 0.9999）
ema_decay = 0.9999
ema_decay_start = 0.999
ema_warmup_steps = 1000  # 约 2-3 epoch 完成 warmup

# ============================================================
# 训练效率优化
# ============================================================
# batch_size 8 → 16（RTX A6000 48GB 显存富余，GPU 利用率提高）
batch_size = 8
num_workers = 8  # 同步提高 worker 数

# 学习率 warmup 步数（与 EMA warmup 解耦）
lr_warmup_steps = 500

# ============================================================
# 输出目录（独立，避免覆盖 v1）
# ============================================================
output_dir = 'work_dirs/chromogen_phase1_sd15_24obj_v2'
