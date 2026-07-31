#!/bin/bash
# Dataset 1 零成本推理补跑 (ross A6000, GPU 0)
# 闭合 4 个双数据集验证缺口:
#   1. 方向 A: per-dim solver mAP 对比 (LINEAGE §九)
#   2. 方向 D: 自适应阶次 DPM-Solver++ (LINEAGE §十)
#   3. dim_d1_mask 4 配置 (LINEAGE §六.6)
#   4. K=100 Top-K pruning 3-seed (LINEAGE §四/§六)
#
# 复用 a4_dpm_pp_chr2024_seed{42,789,123} checkpoints (无需重训)
# 用法: bash experiments/runners/queue_d1_zero_cost_inference.sh
set -u

PROJECT_ROOT=/media/ross/8TB/linkst/chromo/chromosome-kd
cd "$PROJECT_ROOT" || exit 1

PYTHON=/home/linkst/data/miniconda3/envs/chromo/bin/python
D1_CONFIG=experiments/configs/ldmdet/a4_dpm_pp_chr2024.py
D1_ANN=data/Chromosome20240904_NoAug_NoResize_coco/valid/_annotations.coco.json
DIAG_DIR=work_dirs/diagnosis
mkdir -p "$DIAG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$DIAG_DIR/d1_zero_cost_inference_${TS}.log"

echo "=== Dataset 1 零成本推理补跑 ===" | tee "$LOG"
echo "开始: $(date)" | tee -a "$LOG"
echo "Python: $PYTHON" | tee -a "$LOG"
echo "Config: $D1_CONFIG" | tee -a "$LOG"
echo "Ann: $D1_ANN" | tee -a "$LOG"
echo "" | tee -a "$LOG"

ckpt_for() {
    local seed=$1
    ls work_dirs/a4_dpm_pp_chr2024_seed${seed}/best_coco_bbox_mAP_*.pth 2>/dev/null | head -1
}

run_step() {
    # run_step <label> <cmd...>
    local label=$1; shift
    echo "[$(date +%H:%M:%S)] >>> $label" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] <<< $label [OK]" | tee -a "$LOG"
    else
        echo "[$(date +%H:%M:%S)] <<< $label [FAIL rc=$rc]" | tee -a "$LOG"
    fi
    return $rc
}

for seed in 42 789 123; do
    CKPT=$(ckpt_for "$seed")
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "" | tee -a "$LOG"
        echo "[SKIP] seed=$seed checkpoint 不存在, 跳过该 seed" | tee -a "$LOG"
        continue
    fi

    echo "" | tee -a "$LOG"
    echo "########## seed=$seed  ckpt=$CKPT ##########" | tee -a "$LOG"

    # 1. 方向 A: per-dim solver (2 solvers × 500 图)
    run_step "方向A per-dim seed=$seed" \
        "$PYTHON" experiments/analysis/direction_a_per_dim_comparison.py \
        --config "$D1_CONFIG" --checkpoint "$CKPT" --ann "$D1_ANN" --gpu 0 \
        --output "$DIAG_DIR/direction_a_per_dim_d1_seed${seed}.json"

    # 2. 方向 D: 自适应阶次 (3 solvers × 500 图)
    run_step "方向D 自适应阶次 seed=$seed" \
        "$PYTHON" experiments/analysis/direction_d_solver_comparison.py \
        --config "$D1_CONFIG" --checkpoint "$CKPT" --ann "$D1_ANN" --gpu 0 \
        --output "$DIAG_DIR/direction_d_comparison_d1_seed${seed}.json"

    # 3. dim_d1_mask 4 配置 (renewal OFF, 匹配 D2 clean repro 方法论)
    run_step "dim_d1_mask 4 configs (renewal OFF) seed=$seed" \
        "$PYTHON" experiments/analysis/per_dim_d1_clean_repro.py \
        --config "$D1_CONFIG" --checkpoint "$CKPT" --ann "$D1_ANN" --gpu 0 \
        --no-box-renewal \
        --output "$DIAG_DIR/per_dim_d1_clean_repro_d1_seed${seed}.json"

    # 4. K=100/200/300 Top-K pruning + renewal on/off (脚本自动 seed-suffix 输出)
    run_step "Top-K pruning 全 K 矩阵 seed=$seed" \
        "$PYTHON" experiments/analysis/d1_topk_validation.py \
        --gpu 0 --seed "$seed"
done

echo "" | tee -a "$LOG"
echo "=== 全部完成 $(date) ===" | tee -a "$LOG"
echo "输出 JSON 文件清单:" | tee -a "$LOG"
ls -la "$DIAG_DIR"/*_d1_seed*.json 2>/dev/null | tee -a "$LOG"
ls -la "$DIAG_DIR"/d1_topk_validation_seed*.json 2>/dev/null | tee -a "$LOG"
echo "日志: $LOG" | tee -a "$LOG"
