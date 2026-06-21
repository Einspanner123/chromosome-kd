#!/bin/bash
# 24obj 消融实验 — 启动所有 coupling 策略
# 用法: bash tools/run_24obj_ablation.sh

BASE_DIR="work_dirs/24obj_ablation"
GPUS=(0 0 0)  # 3 个 seed 映射到 GPU 0

SEEDS=(42 123 789)
CONFIGS=(
  "experiments/configs/multiset/chromo_24obj.py"        # GHSS
  "experiments/configs/multiset/chromo_24obj_random.py"  # Random
  "experiments/configs/multiset/chromo_24obj_ddpm.py"    # DDPM
  "experiments/configs/multiset/chromo_24obj_sinkhorn.py" # Sinkhorn
  "experiments/configs/multiset/chromo_24obj_hard_ot.py" # Hard OT
)
NAMES=("ghss" "random" "ddpm" "sinkhorn" "hard_ot")

for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  NAME="${NAMES[$i]}"
  echo "========================================="
  echo "Starting $NAME on 24obj dataset"
  echo "========================================="
  for j in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$j]}"
    GPU="${GPUS[$j]}"
    WORK_DIR="${BASE_DIR}/${NAME}/seed_${SEED}"
    echo "  -> seed=$SEED gpu=$GPU work_dir=$WORK_DIR"
    if [ -f "${WORK_DIR}/last_checkpoint" ]; then
      echo "     [RESUME] checkpoint exists"
      CMD="python experiments/runners/train.py $CONFIG --work-dir $WORK_DIR --seed $SEED --gpu-id $GPU --resume"
    else
      CMD="python experiments/runners/train.py $CONFIG --work-dir $WORK_DIR --seed $SEED --gpu-id $GPU"
    fi
    # Run sequentially on single GPU
    eval "$CMD"
  done
done
