#!/bin/bash
# Dataset 2 (24 Chromosomes Object) Hard OT 3-seed 训练
# 目的: 补全 OT Diversity Collapse 理论的双数据集验证缺口
#       (Dataset 2 Hard OT 实验此前从未运行, LINEAGE §二 标注为缺口)
# 关联: Dataset 1 Hard OT 3-seed (work_dirs/multi_seed/hard_ot/, 0.705±0.002 无aug)
#       Dataset 2 Random baseline (chromo_24obj_random.py, 0.860)
#       Dataset 2 Stoch OT (chromo_24obj_sinkhorn.py, 0.856)
# 预测: Dataset 2 Hard OT 应显示坍缩 (mAP < Random), 因 D2 K≈46 更大, 坍缩更严重
#
# Usage: bash experiments/runners/train_hard_ot_24obj_3seed.sh
# 日志: work_dirs/hard_ot_24obj_seed{42,123,789}/train.log
#
# GPU 分配: workstation A5000 (GPU 0, 24GB) + A4000 (GPU 1, 16GB)
# seed 42 → GPU 0, seed 123 → GPU 1 (并行), seed 789 → 先完成的 GPU

set -e
cd /home/linkst/workplace/chromo/chromosome-kd

source /home/linkst/miniconda3/etc/profile.d/conda.sh
conda activate chromo-new

CONFIG=experiments/configs/multiset/chromo_24obj_hard_ot.py

# ── seed 42 (GPU 0, A5000) ──
SEED=42
WORK_DIR=work_dirs/hard_ot_24obj_seed${SEED}
LOG=${WORK_DIR}/train.log
mkdir -p ${WORK_DIR}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting seed ${SEED} on GPU 0"
python experiments/runners/train.py ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --seed ${SEED} \
    --gpu-id 0 \
    2>&1 | tee ${LOG} &
PID_42=$!
echo "  PID: ${PID_42}, Log: ${LOG}"

# ── seed 123 (GPU 1, A4000) ──
SEED=123
WORK_DIR=work_dirs/hard_ot_24obj_seed${SEED}
LOG=${WORK_DIR}/train.log
mkdir -p ${WORK_DIR}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting seed ${SEED} on GPU 1"
python experiments/runners/train.py ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --seed ${SEED} \
    --gpu-id 1 \
    2>&1 | tee ${LOG} &
PID_123=$!
echo "  PID: ${PID_123}, Log: ${LOG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] seed 42 (GPU 0) and seed 123 (GPU 1) running in parallel"
echo "Waiting for first to finish..."

# ── 等待任一完成, 然后在空闲 GPU 上跑 seed 789 ──
# 临时禁用 set -e, 防止 wait -n 返回非零 (某 seed 失败) 时脚本提前退出
set +e
wait -n ${PID_42} ${PID_123}
EXIT_CODE=$?
set -e

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] One seed finished, checking which GPU is free"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] One seed exited with code ${EXIT_CODE}, continuing"
fi

# 检查哪块 GPU 空闲
GPU0_FREE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 2>/dev/null | wc -l)
if [ ${GPU0_FREE} -eq 0 ]; then
    FREE_GPU=0
    wait ${PID_123} 2>/dev/null
else
    FREE_GPU=1
    wait ${PID_42} 2>/dev/null
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU ${FREE_GPU} is free, starting seed 789"

SEED=789
WORK_DIR=work_dirs/hard_ot_24obj_seed${SEED}
LOG=${WORK_DIR}/train.log
mkdir -p ${WORK_DIR}
python experiments/runners/train.py ${CONFIG} \
    --work-dir ${WORK_DIR} \
    --seed ${SEED} \
    --gpu-id ${FREE_GPU} \
    2>&1 | tee ${LOG}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 3 seeds completed."

# ── 汇总结果 ──
echo ""
echo "========================================================"
echo "Hard OT Dataset 2 3-seed Results Summary"
echo "========================================================"
for SEED in 42 123 789; do
    WORK_DIR=work_dirs/hard_ot_24obj_seed${SEED}
    BEST=$(ls ${WORK_DIR}/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1)
    if [ -n "${BEST}" ]; then
        EPOCH=$(echo ${BEST} | grep -o 'epoch_[0-9]*' | grep -o '[0-9]*')
        echo "  seed ${SEED}: best@ep${EPOCH}"
    else
        echo "  seed ${SEED}: NO CHECKPOINT (training may have failed)"
    fi
done
echo "========================================================"
