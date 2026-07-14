#!/bin/bash
# ============================================================
# 需要训练的实验脚本 (AAAI 论文补充)
#
# 实验列表:
#   1. 24obj A4 DPM-Solver++ 多种子 (seed 123, 789)
#   2. 旧数据集 StochOT ε=5 多种子 (seed 42, 123, 789)
#   3. 旧数据集 StochOT ε=1 消融 (seed 42)
#   4. 旧数据集 StochOT ε=2 消融 (seed 42)
#
# 用法:
#   bash results/run_train_commands.sh [实验序号] [GPU_ID]
#
# 参数:
#   实验序号: 1-4 指定单个实验, all 运行全部 (默认 all)
#   GPU_ID:   GPU 编号 (默认 0)
#
# 示例:
#   bash results/run_train_commands.sh all 0     # 在 GPU 0 上运行全部
#   bash results/run_train_commands.sh 1 0       # 在 GPU 0 上运行实验 1
#   bash results/run_train_commands.sh 2 1       # 在 GPU 1 上运行实验 2
#   bash results/run_train_commands.sh 3 2       # 在 GPU 2 上运行实验 3
#   bash results/run_train_commands.sh 4 3       # 在 GPU 3 上运行实验 4
#
# 多服务器并行示例:
#   服务器 A: bash results/run_train_commands.sh 1 0
#   服务器 B: bash results/run_train_commands.sh 2 0
#   服务器 C: bash results/run_train_commands.sh 3 0
#   服务器 D: bash results/run_train_commands.sh 4 0
#
# 注意:
#   - 旧数据集 seed 42 原使用 sinkhorn_stochastic (旧 API), 现已改为 ot_flow (新 API)
#   - 两种 API 算法等价 (Sinkhorn OT + multinomial 采样), 但建议重新运行保证一致性
#   - 24obj A4 seed 42 已有结果 (a4_dpm_pp), 此脚本仅训练 seed 123/789
# ============================================================

set -e

EXP=${1:-all}
GPU=${2:-0}
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 实验描述
declare -A EXP_DESC
EXP_DESC[1]="24obj A4 DPM-Solver++ 多种子 (seed 123, 789)"
EXP_DESC[2]="旧数据集 StochOT ε=5 多种子 (seed 42, 123, 789)"
EXP_DESC[3]="旧数据集 StochOT ε=1 消融 (seed 42)"
EXP_DESC[4]="旧数据集 StochOT ε=2 消融 (seed 42)"

run_exp1() {
    echo "[实验 1] ${EXP_DESC[1]}"
    echo "配置: a4_dpm_pp_24obj_multiseed.py"
    echo "预计时间: ~10h (2 seeds × ~5h)"
    echo "------------------------------------------------------------"
    python experiments/runners/train_multi_seed.py \
        experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj_multiseed.py \
        --seeds 123,789 \
        --gpus $GPU,$GPU \
        --base-dir work_dirs/multi_seed/a4_dpm_pp_24obj
}

run_exp2() {
    echo "[实验 2] ${EXP_DESC[2]}"
    echo "配置: stochot_eps5_old_multiseed.py"
    echo "预计时间: ~15h (3 seeds × ~5h)"
    echo "注意: seed 42 重新运行以保证 API 一致性 (旧 sinkhorn_stochastic → 新 ot_flow)"
    echo "------------------------------------------------------------"
    python experiments/runners/train_multi_seed.py \
        experiments/configs/ldmdet/directions/mainline_ablation_old/stochot_eps5_old_multiseed.py \
        --seeds 42,123,789 \
        --gpus $GPU,$GPU,$GPU \
        --base-dir work_dirs/multi_seed/stochot_eps5_old
}

run_exp3() {
    echo "[实验 3] ${EXP_DESC[3]}"
    echo "配置: stochot_eps1_old.py"
    echo "预计时间: ~5h"
    echo "------------------------------------------------------------"
    python experiments/runners/train.py \
        experiments/configs/ldmdet/directions/mainline_ablation_old/stochot_eps1_old.py \
        --work-dir work_dirs/ablation_old/stochot_eps1 \
        --seed 42 \
        --gpu-id $GPU
}

run_exp4() {
    echo "[实验 4] ${EXP_DESC[4]}"
    echo "配置: stochot_eps2_old.py"
    echo "预计时间: ~5h"
    echo "------------------------------------------------------------"
    python experiments/runners/train.py \
        experiments/configs/ldmdet/directions/mainline_ablation_old/stochot_eps2_old.py \
        --work-dir work_dirs/ablation_old/stochot_eps2 \
        --seed 42 \
        --gpu-id $GPU
}

# 打印实验列表
echo "============================================================"
echo "训练脚本 | 实验=$EXP | GPU=$GPU"
echo "项目根目录: $PROJECT_ROOT"
echo "============================================================"
echo ""
echo "可用实验:"
for i in 1 2 3 4; do
    echo "  $i. ${EXP_DESC[$i]}"
done
echo ""

case $EXP in
    1)   run_exp1 ;;
    2)   run_exp2 ;;
    3)   run_exp3 ;;
    4)   run_exp4 ;;
    all)
        run_exp1
        echo ""
        run_exp2
        echo ""
        run_exp3
        echo ""
        run_exp4
        ;;
    *)
        echo "错误: 未知实验序号 '$EXP'"
        echo "用法: bash $0 [1-4|all] [GPU_ID]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "实验 $EXP 完成!"
echo "============================================================"
echo ""
echo "结果目录:"
echo "  24obj A4 多种子:    work_dirs/multi_seed/a4_dpm_pp_24obj/"
echo "  旧数据集 StochOT:   work_dirs/multi_seed/stochot_eps5_old/"
echo "  旧数据集 ε=1:       work_dirs/ablation_old/stochot_eps1/"
echo "  旧数据集 ε=2:       work_dirs/ablation_old/stochot_eps2/"
echo ""
echo "SwanLab 项目:"
echo "  24obj: ldmdet-mainline-ablation-24obj"
echo "  旧数据集: ldmdet-mainline-ablation-old"
