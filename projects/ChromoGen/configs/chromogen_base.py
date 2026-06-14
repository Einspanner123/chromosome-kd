"""ChromoGen基础配置

Phase 1: 仅图像生成 (enable_bbox_head=False)
Phase 2: 图像+BBox联合训练 (enable_bbox_head=True)
"""

# ============================================================
# 数据集配置
# ============================================================
data_root = 'data/Chromosome20240904_NoAug_NoResize_coco/'
train_ann_file = 'train/_annotations.coco.json'
train_img_dir = 'train'
val_ann_file = 'valid/_annotations.coco.json'
val_img_dir = 'valid'
image_size = 768

# 评估配置
eval_every_n_epochs = 10  # 每隔多少epoch计算FID/IS
num_eval_real_images = 256  # 用于FID计算的真实图像数量
num_eval_gen_images = 256  # 用于FID计算的生成图像数量
eval_gen_batch_size = 8  # 评估时每次生成的图像数

# ============================================================
# 模型配置
# ============================================================

# VAE
vae_model = 'stabilityai/sd-vae-ft-mse'

# UNet
sample_size = 96  # 768 / 8
unet_block_out_channels = (320, 640, 1280, 1280)
unet_attention_head_dim = 8
cross_attention_dim = 768
gradient_checkpointing = True

# Condition Encoder
condition_embed_dim = 768
condition_max_count = 50
condition_dropout = 0.1

# ============================================================
# BBox Head 配置 (通过enable_bbox_head控制是否启用)
# ============================================================
enable_bbox_head = False  # Phase 1: 关闭; Phase 2: 开启

bbox_feat_channels = 512
bbox_num_proposals = 100
bbox_num_heads = 8
bbox_num_layers = 3
bbox_snr_scale = 2.0

# ============================================================
# 扩散配置
# ============================================================
num_train_timesteps = 1000
noise_schedule = 'squaredcos_cap_v2'
prediction_type = 'v_prediction'

# ============================================================
# 损失权重
# ============================================================
lambda_img = 1.0
lambda_bbox = 0.5
lambda_cls = 0.5

# Classifier-free guidance
cfg_dropout = 0.1

# ============================================================
# 训练配置
# ============================================================
batch_size = 8
num_workers = 8
max_epochs = 100
learning_rate = 5e-5
weight_decay = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-8
ema_decay = 0.9999
gradient_accumulation_steps = 1
fp16 = True

# 学习率调度
lr_scheduler = 'cosine'
lr_warmup_steps = 1000

# ============================================================
# 推理配置
# ============================================================
num_inference_steps = 50
guidance_scale = 7.5

# ============================================================
# 日志与保存
# ============================================================
output_dir = 'work_dirs/chromogen'
save_every_n_epochs = 10
log_every_n_steps = 50
sample_every_n_epochs = 5
num_sample_images = 4
