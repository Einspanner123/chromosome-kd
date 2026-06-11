#!/bin/bash
# Quick test: train DiT model with AdaLN gain=0.5 + delta regression
cd /home/linkst/workplace/chromo/chromosome-kd

# Find all experiments for this config
TRAIN_SCRIPT="projects/LDMDet/train.py"
CONFIG="projects/LDMDet/configs/train_dit_x0_epoch50.py"
WORK_DIR="work_dirs/ldmdet_dit_gain05"

echo "=== Starting training ==="
echo "Config: $CONFIG"
echo "Work dir: $WORK_DIR"
echo "Key changes: AdaLN gain=0.02→0.5, regression_mode='direct'→'delta'"
echo

# Run training (short test — 5 epochs, then check mAP)
/home/linkst/miniconda3/envs/chromo/bin/python $TRAIN_SCRIPT \
    --config $CONFIG \
    --work-dir $WORK_DIR \
    2>&1 | head -500