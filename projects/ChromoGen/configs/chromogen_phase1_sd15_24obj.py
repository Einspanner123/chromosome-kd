"""ChromoGen Phase 1 (SD-1.5 fine-tune): 24_chromosomes_object 数据集

基于 chromogen_phase1_sd15.py，切换到 24_chromosomes_object 数据集重新完整训练。

差异（相对 chromogen_phase1_sd15.py）：
  - 数据集: 24_chromosomes_object/coco (train=3500 / valid=500 / test=1000)
  - 输出目录独立，避免覆盖原 SD-1.5 训练 checkpoint
  - 启动时通过 --swanlab_project 指定新 swanlab 项目
"""

_base_ = ['./chromogen_phase1_sd15.py']

# === 数据集：24_chromosomes_object ===
data_root = 'data/24_chromosomes_object/coco/'
train_ann_file = 'train/_annotations.coco.json'
train_img_dir = 'train'
val_ann_file = 'valid/_annotations.coco.json'
val_img_dir = 'valid'
image_size = 768

# === 输出目录（独立，避免覆盖原 SD-1.5 训练）===
output_dir = 'work_dirs/chromogen_phase1_sd15_24obj'

# === 训练规模 ===
# 24_chromosomes_object 训练集 3500 张（vs 原 1540 张），单 epoch step 数翻倍
# 100 epoch 足以收敛
max_epochs = 100
