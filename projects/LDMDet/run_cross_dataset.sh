#!/bin/bash
# Cross-dataset validation experiments for single_chromosomes_object
# Runs 3 coupling strategies on the external chromosome dataset.
#
# Usage:
#   GPU 0 free:  ./run_cross_dataset.sh 0
#   GPU 1 free:  ./run_cross_dataset.sh 1
#
# Data: /data/linkst/datasets/single_chromosomes_object/
#  - 1200 train / 400 val / 400 test
#  - 1 class (chromosomes), ~46 instances/image
#
# Configs:
#   1. Random baseline:       ldmdet_single_chromo_random.py
#   2. Hard OT:                ldmdet_single_chromo_hard_ot.py
#   3. Sinkhorn Stochastic:   ldmdet_single_chromo_stoch_eps5.py
#
# Expected training time: ~3-5 days per config on RTX A5000

set -euo pipefail

GPU_ID="${1:-0}"
CONFIG_DIR="projects/LDMDet/configs"

CONFIGS=(
    "${CONFIG_DIR}/ldmdet_single_chromo_random.py"
    "${CONFIG_DIR}/ldmdet_single_chromo_hard_ot.py"
    "${CONFIG_DIR}/ldmdet_single_chromo_stoch_eps5.py"
)

echo "============================================"
echo "Cross-Dataset Validation Experiments"
echo "GPU: ${GPU_ID}"
echo "Configs to run: ${#CONFIGS[@]}"
echo "============================================"

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    echo ""
    echo "[$((i+1))/${#CONFIGS[@]}] Launching: ${cfg}"
    echo "----------------------------------------"

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python tools/train.py "${cfg}"

    echo "[$((i+1))/${#CONFIGS[@]}] Completed: ${cfg}"
done

echo ""
echo "All cross-dataset experiments completed."
