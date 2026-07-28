#!/bin/bash
# Dataset 1 (Chromosome20240904) A4 DPM-Solver++ 3-seed 顺序训练
# 补全 Dataset 1 主消融链第 4 环 (RF+AdaLN+StochOT + DPM-Solver++)
# 关联: Dataset 2 A4 (a4_dpm_pp_24obj.py, 3-seed 0.859)
#
# Usage: bash experiments/runners/train_a4_chr2024_3seed.sh
# 日志: work_dirs/a4_dpm_pp_chr2024_seed{42,123,789}/train.log

set -e
cd /home/linkst/workspace/chromosome-kd

source /home/linkst/data/miniconda3/etc/profile.d/conda.sh
conda activate chromo

CONFIG=experiments/configs/ldmdet/a4_dpm_pp_chr2024.py
GPU=0

for SEED in 42 123 789; do
    WORK_DIR=work_dirs/a4_dpm_pp_chr2024_seed${SEED}
    LOG=${WORK_DIR}/train.log
    mkdir -p ${WORK_DIR}

    echo "========================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting seed ${SEED}"
    echo "  Config: ${CONFIG}"
    echo "  Work dir: ${WORK_DIR}"
    echo "  Log: ${LOG}"
    echo "========================================================"

    python experiments/runners/train.py ${CONFIG} \
        --work-dir ${WORK_DIR} \
        --seed ${SEED} \
        --gpu-id ${GPU} \
        2>&1 | tee ${LOG}

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished seed ${SEED}"
    echo ""
done

echo "========================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 3 seeds completed."
echo "========================================================"
