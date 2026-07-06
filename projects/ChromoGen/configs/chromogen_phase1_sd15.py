"""ChromoGen Phase 1 (SD-1.5 fine-tune): 仅图像生成

基于 Stable Diffusion v1.5 预训练 UNet 权重做 fine-tune，替代 Phase1 from-scratch 路线。

差异（相对 chromogen_phase1_imgonly.py）：
  - UNet 从 SD-1.5 预训练权重初始化（LAION-5B 强视觉先验）
  - 切换为 SD-1.5 兼容配置：epsilon prediction + linear schedule
  - 学习率降到 1e-5（fine-tune 社区惯例，避免破坏预训练先验）
  - 输出目录独立，避免覆盖原 Phase1 checkpoint
"""

_base_ = ['./chromogen_base.py']

# === SD-1.5 预训练 UNet ===
unet_pretrained_model = 'runwayml/stable-diffusion-v1-5'

# === SD-1.5 兼容配置 ===
# SD-1.5 原配为 linear + epsilon，切换以最大化先验利用
noise_schedule = 'linear'
prediction_type = 'epsilon'

# === Fine-tune 学习率 ===
# SD 社区惯例 1e-5 ~ 5e-6；from-scratch 才用 5e-5
learning_rate = 1e-5

# === 训练规模 ===
enable_bbox_head = False
max_epochs = 100  # fine-tune 收敛更快，100 epoch 足够

# === 输出目录（独立，避免覆盖原 Phase1）===
output_dir = 'work_dirs/chromogen_phase1_sd15'
