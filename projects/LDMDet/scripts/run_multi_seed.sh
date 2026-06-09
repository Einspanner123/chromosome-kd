#!/bin/bash
set -euo pipefail
GPU_ID="${1:-0}"
CONFIG_DIR="projects/LDMDet/configs/benchmark"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_DIR}"
SEEDS=(42 123 456 789 1000)
echo "============================================"
echo "  SOTA Multi-Seed Benchmark"
echo "  GPU: ${GPU_ID}"
echo "  Seeds: ${SEEDS[*]}"
echo "  Total: ${#SEEDS[@]} runs (sequential)"
echo "============================================"
echo ""
for seed in "${SEEDS[@]}"; do
    cfg="${CONFIG_DIR}/sota_seed${seed}.py"
    echo "[$(date +\"%Y-%m-%d %H:%M:%S\")] >>> Launching seed=${seed}: ${cfg}"
    echo "----------------------------------------"
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${GPU_ID}" python tools/train.py "${cfg}"
    echo "[$(date +\"%Y-%m-%d %H:%M:%S\")] <<< Completed seed=${seed}"
    echo ""
done
echo "============================================"
echo "  All ${#SEEDS[@]} multi-seed experiments completed!"
echo "============================================"
