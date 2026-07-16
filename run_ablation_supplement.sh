#!/bin/bash
# LDMDet 消融补充实验串行执行脚本
# 用途: DeepSeek Major Concern 1 (shifted schedule ablation) + Concern 2 (A0 multi-seed)
#
# 实验列表 (串行执行):
#   1. A1 RF+Heun unshifted schedule (seed 42, 单次训练, ~10-16h)
#   2. A0 baseline seed 123 (~5-6h, 早停极早)
#   3. A0 baseline seed 789 (~5-6h, 早停极早)
#
# Usage:
#   bash run_ablation_supplement.sh [GPU_ID]
#   默认 GPU_ID=0
#
# 在 ross 服务器上执行:
#   cd /media/ross/8TB/linkst/chromo/chromosome-kd
#   conda activate chromo
#   bash run_ablation_supplement.sh 0

set -euo pipefail
cd "$(dirname "$0")"

GPU_ID="${1:-0}"
CONFIG_DIR="experiments/configs/ldmdet/directions/mainline_ablation_24obj"
WORK_DIR_BASE="work_dirs/ablation_supplement"

echo "================================================"
echo "LDMDet 消融补充实验 (串行执行)"
echo "GPU: $GPU_ID"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"

# ================================================
# 实验 1: A1 RF+Heun unshifted schedule (seed 42)
# 目的: 消融 shifted schedule 贡献 (DeepSeek Concern 1)
# 预计耗时: ~10-16h
# ================================================
EXP1_CONFIG="$CONFIG_DIR/a1_rf_heun_unshifted_24obj.py"
EXP1_WORKDIR="$WORK_DIR_BASE/a1_rf_heun_unshifted_24obj"
EXP1_SEED=42

echo ""
echo "[实验 1/3] A1 RF+Heun unshifted schedule (seed=$EXP1_SEED)"
echo "Config: $EXP1_CONFIG"
echo "WorkDir: $EXP1_WORKDIR"
echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"

python -m experiments.runners.train \
    "$EXP1_CONFIG" \
    --work-dir "$EXP1_WORKDIR" \
    --seed "$EXP1_SEED" \
    --gpu-id "$GPU_ID"

echo "实验 1 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ================================================
# 实验 2: A0 baseline multi-seed (seed 123, 789)
# 目的: 统计显著性验证 (DeepSeek Concern 2)
# 使用 train_multi_seed.py 串行执行 2 个 seed
# 预计耗时: ~10-12h (A0 早停极早, ~ep42)
# ================================================
EXP2_CONFIG="$CONFIG_DIR/a0_baseline_24obj_multiseed.py"
EXP2_WORKDIR="$WORK_DIR_BASE/a0_baseline_24obj_multiseed"
EXP2_SEEDS="123,789"

echo ""
echo "[实验 2/3] A0 baseline multi-seed (seeds=$EXP2_SEEDS)"
echo "Config: $EXP2_CONFIG"
echo "WorkDir: $EXP2_WORKDIR"
echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"

python -m experiments.runners.train_multi_seed \
    "$EXP2_CONFIG" \
    --seeds "$EXP2_SEEDS" \
    --gpus "$GPU_ID,$GPU_ID"

echo "实验 2 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ================================================
# 汇总
# ================================================
echo ""
echo "================================================"
echo "所有实验完成!"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================"
echo ""
echo "实验结果汇总:"
echo "  实验 1 (A1 unshifted): $EXP1_WORKDIR"
echo "  实验 2 (A0 multi-seed): $EXP2_WORKDIR"
echo ""
echo "下一步: 提取 best mAP 并更新论文"
echo "  - A1 unshifted mAP vs A1 shifted mAP (0.856) → shifted schedule 贡献"
echo "  - A0 3-seed mean±std vs A1 3-seed (复用 24obj_ablation/random) → 统计显著性"
