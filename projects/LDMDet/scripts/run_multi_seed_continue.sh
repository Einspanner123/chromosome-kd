#!/bin/bash
# SOTA Multi-Seed Continue Script
# Resumes seed123 from epoch 36, then runs seeds 456, 789, 1000 fresh.
#
# Usage:
#   bash projects/LDMDet/scripts/run_multi_seed_continue.sh [GPU_ID]
#
# Example:
#   bash projects/LDMDet/scripts/run_multi_seed_continue.sh 0

set -euo pipefail

GPU_ID="${1:-0}"
CONFIG_DIR="projects/LDMDet/configs/benchmark"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"
echo "============================================"
echo "  SOTA Multi-Seed Continue"
echo "  GPU: ${GPU_ID}"
echo "============================================"
echo ""

# =========================================================
# 1. Resume seed123 from epoch 36
# =========================================================
SEED=123
cfg="${CONFIG_DIR}/sota_seed${SEED}.py"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>> Resuming seed=${SEED} from last checkpoint..."
echo "----------------------------------------"

CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python tools/train.py "${cfg}" --resume

echo "[$(date +'%Y-%m-%d %H:%M:%S')] <<< Completed seed=${SEED}"
echo ""

# =========================================================
# 2. Fresh runs for seeds 456, 789, 1000
# =========================================================
for SEED in 456 789 1000; do
    cfg="${CONFIG_DIR}/sota_seed${SEED}.py"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] >>> Launching seed=${SEED}: ${cfg}"
    echo "----------------------------------------"

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python tools/train.py "${cfg}"

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] <<< Completed seed=${SEED}"
    echo ""
done

echo "============================================"
echo "  All remaining experiments completed!"
echo "  Seeds: 123 (resumed), 456, 789, 1000"
echo "============================================"
