#!/bin/bash
# ============================================================
# Scale-Conditioned Flow Matching 实验
# 目标: 验证染色体尺寸条件化对小染色体检测的改善
#
# 基线: ldmdet_flowdet_adaln (RF+Heun+shifted+AdaLN, 0.751)
# 核心假设: 染色体物理尺寸 (r=0.74) → 决定检测难度
#           → 按尺寸缩放噪声 + 损失可改善小染色体定位
# ============================================================
set -e
cd /home/linkst/workplace/chromo/chromosome-kd

source /home/linkst/miniconda3/etc/profile.d/conda.sh
conda activate chromo

PYTHON="python"
CONFIG_DIR="projects/LDMDet/configs/scale_conditioned"
GPU=0

EXPERIMENTS=(
    "sc_noise:仅噪声缩放(小染色体用更小噪声)"
    "sc_loss:仅损失加权(小染色体用更高回归权重)"
    "sc_combined:噪声缩放+损失加权(完整Scale-Conditioned)"
)

echo "=========================================="
echo "  Scale-Conditioned Flow Matching"
echo "  Baseline: ldmdet_flowdet_adaln (0.751)"
echo "  GPU: $GPU"
echo "=========================================="
echo ""

for exp_info in "${EXPERIMENTS[@]}"; do
    exp_name="${exp_info%%:*}"
    exp_desc="${exp_info##*:}"

    echo "=========================================="
    echo "  [$exp_name] $exp_desc"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON tools/train.py \
        "${CONFIG_DIR}/${exp_name}.py" \
        --work-dir "work_dirs/scale_conditioned_${exp_name}"

    echo "  [$exp_name] Done: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "=========================================="
echo "  All scale-conditioned experiments complete!"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Expected results:"
echo "  | Experiment          | Predicted mAP | Key Effect              |"
echo "  |--------------------|---------------|-------------------------|"
echo "  | Baseline (AdaLN)    | 0.751         | --                      |"
echo "  | SC-Noise only       | 0.755-0.760   | 小染色体流路径更紧致    |"
echo "  | SC-Loss only        | 0.753-0.757   | 小染色体更多梯度分配    |"
echo "  | SC-Combined         | 0.758-0.765   | 协同效应                |"
echo ""
echo "  Key evaluation: per-class AP of G21/G22/Y should improve most"
echo "=========================================="
