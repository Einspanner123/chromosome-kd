#!/bin/bash
# 重跑 seed42 方向 A + 方向 D (首次因 renewal_mask bug 失败, 代码已修复)
# 在 orchestration 完成后运行 (避免 GPU 抢占)
set -u
cd /media/ross/8TB/linkst/chromo/chromosome-kd || exit 1

PYTHON=/home/linkst/data/miniconda3/envs/chromo/bin/python
D1_CONFIG=experiments/configs/ldmdet/a4_dpm_pp_chr2024.py
D1_ANN=data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json
CKPT=work_dirs/a4_dpm_pp_chr2024_seed42/best_coco_bbox_mAP_epoch_49.pth
DIAG_DIR=work_dirs/diagnosis

LOG="$DIAG_DIR/rerun_seed42_dir_a_d_$(date +%Y%m%d_%H%M%S).log"
echo "=== 重跑 seed42 方向 A + D $(date) ===" | tee "$LOG"

echo "[$(date +%H:%M:%S)] >>> 方向A per-dim seed=42" | tee -a "$LOG"
"$PYTHON" experiments/analysis/direction_a_per_dim_comparison.py \
    --config "$D1_CONFIG" --checkpoint "$CKPT" --ann "$D1_ANN" --gpu 0 \
    --output "$DIAG_DIR/direction_a_per_dim_d1_seed42.json" >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] <<< 方向A [rc=$?]" | tee -a "$LOG"

echo "[$(date +%H:%M:%S)] >>> 方向D 自适应阶次 seed=42" | tee -a "$LOG"
"$PYTHON" experiments/analysis/direction_d_solver_comparison.py \
    --config "$D1_CONFIG" --checkpoint "$CKPT" --ann "$D1_ANN" --gpu 0 \
    --output "$DIAG_DIR/direction_d_comparison_d1_seed42.json" >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] <<< 方向D [rc=$?]" | tee -a "$LOG"

echo "=== 完成 $(date) ===" | tee -a "$LOG"
