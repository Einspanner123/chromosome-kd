#!/bin/bash
# LDMDet 消融补充实验串行执行脚本
# 用途: 消融缺口补充 (shifted schedule / multi-seed / shift参数扫描)
#
# 实验列表 (串行执行):
#   1. A1 RF+Heun unshifted schedule (seed 42, ~10-16h)
#   2. A0 baseline multi-seed (seed 123, 789, ~10-12h)
#   3. A1 RF+Heun multi-seed (seed 123, 789, ~20-30h)
#   4. shifted schedule 参数扫描 shift=[2, 5] (seed 42, ~20-30h)
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
# 目的: 消融 shifted schedule 贡献
# 对照: A1 shifted (mAP=0.856) vs A1 unshifted → shifted schedule 贡献
# 预计耗时: ~10-16h (参考 A1 best@ep62)
# ================================================
EXP1_CONFIG="$CONFIG_DIR/a1_rf_heun_unshifted_24obj.py"
EXP1_WORKDIR="$WORK_DIR_BASE/a1_rf_heun_unshifted_24obj"
EXP1_SEED=42

echo ""
echo "[实验 1/4] A1 RF+Heun unshifted schedule (seed=$EXP1_SEED)"
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
# 目的: A0 统计显著性验证 (当前仅 seed42 mAP=0.774)
# 使用 train_multi_seed.py 串行执行 2 个 seed
# 预计耗时: ~10-12h (A0 早停极早, best@ep12)
# ================================================
EXP2_CONFIG="$CONFIG_DIR/a0_baseline_24obj_multiseed.py"
EXP2_WORKDIR="$WORK_DIR_BASE/a0_baseline_24obj_multiseed"
EXP2_SEEDS="123,789"

echo ""
echo "[实验 2/4] A0 baseline multi-seed (seeds=$EXP2_SEEDS)"
echo "Config: $EXP2_CONFIG"
echo "WorkDir: $EXP2_WORKDIR"
echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"

python -m experiments.runners.train_multi_seed \
    "$EXP2_CONFIG" \
    --seeds "$EXP2_SEEDS" \
    --gpus "$GPU_ID,$GPU_ID"

echo "实验 2 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ================================================
# 实验 3: A1 RF+Heun multi-seed (seed 123, 789)
# 目的: A1 统计显著性验证 (当前仅 seed42 mAP=0.856)
# 配合 A0 multi-seed → A0 vs A1 的 3-seed t-test
# 预计耗时: ~20-30h (A1 best@ep62, 每个 seed ~10-15h)
# ================================================
EXP3_CONFIG="$CONFIG_DIR/a1_rf_heun_24obj_multiseed.py"
EXP3_WORKDIR="$WORK_DIR_BASE/a1_rf_heun_24obj_multiseed"
EXP3_SEEDS="123,789"

echo ""
echo "[实验 3/4] A1 RF+Heun multi-seed (seeds=$EXP3_SEEDS)"
echo "Config: $EXP3_CONFIG"
echo "WorkDir: $EXP3_WORKDIR"
echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"

python -m experiments.runners.train_multi_seed \
    "$EXP3_CONFIG" \
    --seeds "$EXP3_SEEDS" \
    --gpus "$GPU_ID,$GPU_ID"

echo "实验 3 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# ================================================
# 实验 4: shifted schedule 参数扫描 (shift=2, 5)
# 目的: 验证 shift=3.0 是否为最优值, 或 shift 参数不敏感
# 对照: shift=1 (unshifted, 实验1) vs shift=2 vs shift=3 (A1, 0.856) vs shift=5
# 预计耗时: ~20-30h (2 个实验, 每个 ~10-15h)
# ================================================
echo ""
echo "[实验 4/4] shifted schedule 参数扫描 (shift=2, 5)"

# 4a: shift=2
EXP4A_CONFIG="$CONFIG_DIR/a1_rf_heun_shift2_24obj.py"
EXP4A_WORKDIR="$WORK_DIR_BASE/a1_rf_heun_shift2_24obj"
EXP4A_SEED=42

echo ""
echo "[实验 4a] A1 RF+Heun shift=2 (seed=$EXP4A_SEED)"
echo "Config: $EXP4A_CONFIG"
echo "WorkDir: $EXP4A_WORKDIR"
echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"

python -m experiments.runners.train \
    "$EXP4A_CONFIG" \
    --work-dir "$EXP4A_WORKDIR" \
    --seed "$EXP4A_SEED" \
    --gpu-id "$GPU_ID"

echo "实验 4a 完成: $(date '+%Y-%m-%d %H:%M:%S')"

# 4b: shift=5
EXP4B_CONFIG="$CONFIG_DIR/a1_rf_heun_shift5_24obj.py"
EXP4B_WORKDIR="$WORK_DIR_BASE/a1_rf_heun_shift5_24obj"
EXP4B_SEED=42

echo ""
echo "[实验 4b] A1 RF+Heun shift=5 (seed=$EXP4B_SEED)"
echo "Config: $EXP4B_CONFIG"
echo "WorkDir: $EXP4B_WORKDIR"
echo "开始: $(date '+%Y-%m-%d %H:%M:%S')"

python -m experiments.runners.train \
    "$EXP4B_CONFIG" \
    --work-dir "$EXP4B_WORKDIR" \
    --seed "$EXP4B_SEED" \
    --gpu-id "$GPU_ID"

echo "实验 4b 完成: $(date '+%Y-%m-%d %H:%M:%S')"

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
echo "  实验 1 (A1 unshifted shift=1): $EXP1_WORKDIR"
echo "  实验 2 (A0 multi-seed):        $EXP2_WORKDIR"
echo "  实验 3 (A1 multi-seed):        $EXP3_WORKDIR"
echo "  实验 4a (A1 shift=2):          $EXP4A_WORKDIR"
echo "  实验 4b (A1 shift=5):          $EXP4B_WORKDIR"
echo ""
echo "关键对照:"
echo "  shifted schedule 贡献:"
echo "    A1 shift=1 (实验1) vs A1 shift=3 (0.856) → shift 本身贡献"
echo "    A1 shift=2 (实验4a) / shift=5 (实验4b) → shift 参数敏感性"
echo "  统计显著性:"
echo "    A0 3-seed (seed42=0.774 + 实验2) vs A1 3-seed (seed42=0.856 + 实验3) → t-test"
