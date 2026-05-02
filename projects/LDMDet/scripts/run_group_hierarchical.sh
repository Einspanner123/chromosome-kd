#!/bin/bash
# Group-Hierarchical Coupling 实验
# 目标: 验证染色体分组 OT 降低 ΔH 从 log(46)≈3.83 到 ~1.90
set -e
cd /home/linkst/workplace/chromo/chromosome-kd

PYTHON="/home/linkst/miniconda3/envs/chromo/bin/python"
CONFIG_DIR="projects/LDMDet/configs"

# ============================================================
# Experiment 1: Group-Hierarchical argmax (no stochastic)
# 单纯分组效果: 降低有效 K → 降低 ΔH
# ============================================================
echo "=========================================="
echo "Experiment 1: Group-Hierarchical (argmax)"
echo "=========================================="
$PYTHON tools/train.py \
    ${CONFIG_DIR}/ldmdet_flowdet_adaln_group_hierarchical.py \
    --work-dir work_dirs/ldmdet_flowdet_adaln_group_hierarchical

# ============================================================
# Experiment 2: Group-Hierarchical + Stochastic ε=5 (推荐)
# 组合分组降 ΔH + 随机采样多样性保留
# ============================================================
echo "=========================================="
echo "Experiment 2: Group-Hierarchical + Stochastic ε=5"
echo "=========================================="
$PYTHON tools/train.py \
    ${CONFIG_DIR}/ldmdet_flowdet_adaln_group_hierarchical_stoch.py \
    --work-dir work_dirs/ldmdet_flowdet_adaln_group_hierarchical_stoch

echo ""
echo "=========================================="
echo "All experiments complete."
echo ""
echo "Expected results (theory):"
echo "  Experiment                    | Predicted mAP | ΔH_eff"
echo "  ------------------------------|---------------|--------"
echo "  AdaLN baseline (random)       | 0.751         | 0 (no OT)"
echo "  Hard OT (full)                | 0.735         | 3.83"
echo "  Stochastic ε=5 (full)         | 0.751         | 3.83 (sampling)"
echo "  Group-Hierarchical (argmax)   | 0.743-0.748   | 1.90"
echo "  Group-Hierarchical + Stoch ε5 | 0.750-0.755   | 1.90 (sampling)"
echo "=========================================="
