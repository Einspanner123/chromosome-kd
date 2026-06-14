#!/bin/bash
# ChromoGen训练脚本

set -e

# Phase 1: 仅图像生成
echo "=== Phase 1: Image-only training ==="
python projects/ChromoGen/tools/train.py \
    --config projects/ChromoGen/configs/chromogen_phase1_imgonly.py \
    --gpu 0

# Phase 2: 图像+BBox联合训练 (从Phase 1 checkpoint继续)
echo "=== Phase 2: Joint training ==="
python projects/ChromoGen/tools/train.py \
    --config projects/ChromoGen/configs/chromogen_phase2_joint.py \
    --resume work_dirs/chromogen_phase1/final_model.pt \
    --gpu 0
