#!/bin/bash
# 等待 DINO R50 训练完成 (PID 3096621) 后自动启动 D1 A4 seed789 训练
# 在 ross A6000 上运行, 预计 DINO R50 早停后 ~5h 完成 seed789
#
# 用法: nohup bash experiments/runners/wait_and_start_seed789.sh > /tmp/wait_seed789.log 2>&1 &

set -e

DINO_PID=3096621
PROJECT_ROOT="/media/ross/8TB/linkst/chromo/chromosome-kd"
PYTHON="/home/linkst/data/miniconda3/envs/chromo/bin/python"
CONFIG="experiments/configs/ldmdet/a4_dpm_pp_chr2024.py"
SEED=789

cd "$PROJECT_ROOT"

echo "[$(date)] 等待 DINO R50 (PID $DINO_PID) 完成..."
echo "[$(date)] 当前 GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# 等待 DINO R50 进程退出
while kill -0 "$DINO_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] DINO R50 (PID $DINO_PID) 已退出!"
echo "[$(date)] 等待 30s 确保 GPU 内存释放..."
sleep 30

echo "[$(date)] GPU 状态 (DINO R50 退出后):"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# 确认 GPU 0 基本空闲 (used < 1000 MiB)
GPU0_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
if [ "$GPU0_USED" -gt 1000 ]; then
    echo "[$(date)] 警告: GPU 0 仍使用 ${GPU0_USED} MiB, 可能未完全释放"
    echo "[$(date)] 再等待 60s..."
    sleep 60
fi

echo "[$(date)] 启动 D1 A4 seed${SEED} 训练..."
echo "[$(date)] Config: $CONFIG"
echo "[$(date)] Python: $PYTHON"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/runners/train.py \
    "$CONFIG" \
    --seed "$SEED" \
    --gpu-id 0 \
    > "work_dirs/a4_dpm_pp_chr2024_seed${SEED}_train.log" 2>&1 < /dev/null &

TRAIN_PID=$!
echo "[$(date)] seed${SEED} 训练已启动, PID=$TRAIN_PID"
echo "[$(date)] 日志: work_dirs/a4_dpm_pp_chr2024_seed${SEED}_train.log"

# 等待训练启动并输出前几行日志
sleep 30
echo "[$(date)] 训练日志前 5 行:"
head -5 "work_dirs/a4_dpm_pp_chr2024_seed${SEED}_train.log" 2>/dev/null

echo "[$(date)] 脚本完成. seed${SEED} 训练在后台运行中 (PID=$TRAIN_PID)."
