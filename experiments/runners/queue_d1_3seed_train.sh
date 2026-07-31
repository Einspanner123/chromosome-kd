#!/bin/bash
# Dataset 1 双数据集缺口补全训练队列 (workstation GPU 1, A4000 16GB)
#
# 按优先级顺序运行 4 个训练 (GPU 1 一次只能跑 1 个):
#   1. r3_vpred_chr2024 seed42  (高优先 - 论文需要, v-prediction 对照 Dataset 1)
#   2. r3_vpred_chr2024 seed123 (高优先 - 3-seed)
#   3. r3_vpred_chr2024 seed789 (高优先 - 3-seed)
#   4. s1_h3_s4_chr2024 seed42  (中优先 - Cascade × Solver 解耦 Dataset 1)
#
# 用法 (workstation):
#   nohup bash experiments/runners/queue_d1_3seed_train.sh > work_dirs/queue_d1_3seed.log 2>&1 &
#
# 监控:
#   tail -f work_dirs/queue_d1_3seed.log
#   tail -f work_dirs/r3_vpred_chr2024_seed42_train.log

set -u  # 未定义变量报错 (不用 -e, 避免单个训练失败中断整个队列)

PROJECT_ROOT="/home/linkst/workplace/chromo/chromosome-kd"
PYTHON="/home/linkst/miniconda3/envs/chromo-new/bin/python"
GPU_ID=1

cd "$PROJECT_ROOT"

echo "========================================"
echo "[$(date)] Dataset 1 双数据集缺口补全训练队列启动"
echo "[$(date)] GPU: $GPU_ID (A4000 16GB)"
echo "[$(date)] Python: $PYTHON"
echo "========================================"

# 训练任务队列: <config> <seed> <work_dir_suffix> <log_file>
TASKS=(
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_chr2024.py|42|r3_vpred_chr2024_seed42|r3_vpred_chr2024_seed42_train.log"
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_chr2024.py|123|r3_vpred_chr2024_seed123|r3_vpred_chr2024_seed123_train.log"
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_chr2024.py|789|r3_vpred_chr2024_seed789|r3_vpred_chr2024_seed789_train.log"
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/s1_h3_s4_chr2024.py|42|s1_h3_s4_chr2024_seed42|s1_h3_s4_chr2024_seed42_train.log"
)

TASK_NUM=0
TOTAL=${#TASKS[@]}

for task in "${TASKS[@]}"; do
    TASK_NUM=$((TASK_NUM + 1))
    IFS='|' read -r CONFIG SEED WORK_DIR_SUFFIX LOG_FILE <<< "$task"
    WORK_DIR="work_dirs/${WORK_DIR_SUFFIX}"

    echo "========================================"
    echo "[$(date)] 任务 ${TASK_NUM}/${TOTAL}: $WORK_DIR_SUFFIX"
    echo "[$(date)] Config: $CONFIG"
    echo "[$(date)] Seed: $SEED"
    echo "[$(date)] Work dir: $WORK_DIR"
    echo "[$(date)] Log: $LOG_FILE"
    echo "========================================"

    # 检查 work_dir 是否已有 best checkpoint (跳过已完成的任务)
    if [ -d "$WORK_DIR" ] && ls "$WORK_DIR"/best_coco_bbox_mAP_*.pth 1>/dev/null 2>&1; then
        echo "[$(date)] 跳过: $WORK_DIR 已有 best checkpoint, 假定已完成"
        continue
    fi

    # 检查 GPU 1 是否空闲 (used < 2000 MiB)
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d ' ')
    if [ "$GPU_USED" -gt 2000 ]; then
        echo "[$(date)] 警告: GPU $GPU_ID 使用 ${GPU_USED} MiB, 等待释放..."
        while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d ' ')" -gt 2000 ]; do
            sleep 60
        done
        echo "[$(date)] GPU $GPU_ID 已空闲"
    fi

    echo "[$(date)] 启动训练..."
    "$PYTHON" experiments/runners/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --seed "$SEED" \
        --gpu-id "$GPU_ID" \
        > "$LOG_FILE" 2>&1 < /dev/null &

    TRAIN_PID=$!
    echo "[$(date)] 训练已启动, PID=$TRAIN_PID"

    # 等待训练启动并输出前几行日志
    sleep 30
    echo "[$(date)] 训练日志前 10 行:"
    head -10 "$LOG_FILE" 2>/dev/null || echo "(日志暂无输出)"

    # 等待训练进程退出
    echo "[$(date)] 等待训练完成 (PID=$TRAIN_PID)..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep 120
    done

    echo "[$(date)] 训练 $WORK_DIR_SUFFIX 已退出 (PID=$TRAIN_PID)"
    echo "[$(date)] 训练日志末尾 5 行:"
    tail -5 "$LOG_FILE" 2>/dev/null

    # 等待 GPU 内存释放
    echo "[$(date)] 等待 30s 确保 GPU 内存释放..."
    sleep 30
done

echo "========================================"
echo "[$(date)] 所有训练任务已完成!"
echo "[$(date)] 各任务日志:"
for task in "${TASKS[@]}"; do
    IFS='|' read -r CONFIG SEED WORK_DIR_SUFFIX LOG_FILE <<< "$task"
    echo "  - $WORK_DIR_SUFFIX: $LOG_FILE"
done
echo "========================================"
