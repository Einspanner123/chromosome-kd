#!/bin/bash
# ============================================================
# LDMDet 完整消融实验: Remove-One from Best Model
# 最佳模型: ldmdet_flowdet_adaln (RF + Heun + shifted + AdaLN, bs=2, 0.751)
#
# 已知数据:
#   ldmdet_baseline       (DDPM)                    → 0.725
#   ldmdet_rf             (+RF)                     → 0.733
#   ldmdet_rf_shifted     (+shifted)                → 0.747
#   ldmdet_rf_heun_shifted(+Heun, bs=4)             → 0.749
#   ldmdet_flowdet_adaln  (+AdaLN, bs=2, **BEST**)  → 0.751
#
# 3 个消融实验 (每个从最佳模型移除一个组件):
#   ablate_adaln:  移除 AdaLN → scale-shift   (验证 AdaLN 贡献)
#   ablate_heun:   移除 Heun  → Euler 一阶    (验证 Heun 贡献)
#   ablate_shifted: 移除 shifted → linear     (验证 shifted 贡献)
# ============================================================
set -e
cd /home/linkst/workplace/chromo/chromosome-kd

source /home/linkst/miniconda3/etc/profile.d/conda.sh
conda activate chromo

PYTHON="python"
CONFIG_DIR="projects/LDMDet/configs/ablations"
GPU=0  # 使用 A5000 (24GB)

EXPERIMENTS=(
    "ablate_adaln:移除AdaLN回退scale-shift"
    "ablate_heun:移除Heun回退Euler一阶"
    "ablate_shifted:移除Shifted回退linear均匀采样"
)

echo "=========================================="
echo "  LDMDet Remove-One Ablation Study"
echo "  Best model: ldmdet_flowdet_adaln (0.751)"
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
        --work-dir "work_dirs/ablations/${exp_name}"

    echo "  [$exp_name] Done: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
done

echo "=========================================="
echo "  All ablation experiments complete!"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Expected results table:"
echo "  | Config               | Components                     | Best mAP | Δ     |"
echo "  |----------------------|--------------------------------|----------|-------|"
echo "  | ldmdet_baseline      | DDPM+Euler+linear+scale-shift | 0.725    | —     |"
echo "  | ldmdet_rf            | RF+Euler+linear+scale-shift   | 0.733    | +0.8% |"
echo "  | ldmdet_rf_shifted    | RF+Euler+shifted+scale-shift  | 0.747    | +1.4% |"
echo "  | ldmdet_rf_heun_shift | RF+Heun+shifted+scale-shift   | 0.749    | +0.2% |"
echo "  | ablate_adaln         | RF+Heun+shifted+scale-shift*  | 0.748†   | −0.3% |"
echo "  | ablate_heun          | RF+Euler+shifted+AdaLN        | ???      | ???   |"
echo "  | ablate_shifted       | RF+Heun+linear+AdaLN          | ???      | ???   |"
echo "  | ldmdet_flowdet_adaln | RF+Heun+shifted+AdaLN (BEST)  | 0.751    | —     |"
echo ""
echo "  † ablate_adaln ≈ ldmdet_rf_heun_shifted_bs2 (已训练, 0.748)"
echo "  * bs=2 版本"
echo "=========================================="
