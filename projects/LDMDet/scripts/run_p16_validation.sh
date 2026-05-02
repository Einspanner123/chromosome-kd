#!/bin/bash
# P1-6 最优 ε 验证实验
# 验证三区间模型预测: ε* ≈ 1-3
#
# 使用方法:
#   bash scripts/run_p16_validation.sh [priority]
#   priority: p1 (仅 ε=1.0) | p2 (ε=1.0,2.0) | all (ε=0.5,1.0,2.0,3.0,10.0)
#
# 已有数据:
#   Hard OT (ε=0):    mAP=0.735
#   Stochastic ε=5:   mAP=0.751
#   Stochastic ε=50:  mAP=0.736 (peak) → 0.696 (degraded)
#   Random (ε=∞):     mAP=0.751
#
# 预测:
#   ε=0.5: mAP 0.745-0.750  (区间1→2边界, ρ=0.951)
#   ε=1.0: mAP 0.750-0.752  (★最强验证点, ρ=0.986)
#   ε=2.0: mAP 0.749-0.751  (区间2核心, ρ≈0.999)
#   ε=3.0: mAP 0.748-0.751  (区间2核心)
#   ε=10:  mAP 0.740-0.748  (区间2→3边界)

set -e
cd /home/linkst/workplace/chromo/chromosome-kd

PYTHON="/home/linkst/miniconda3/envs/chromo/bin/python"
CONFIG_DIR="projects/LDMDet/configs"

PRIORITY="${1:-p1}"

echo "=========================================="
echo "P1-6: Optimal Epsilon Validation"
echo "Priority: ${PRIORITY}"
echo "=========================================="

case "$PRIORITY" in
    p1)
        EPS_LIST=(1.0)
        ;;
    p2)
        EPS_LIST=(1.0 2.0)
        ;;
    all)
        EPS_LIST=(0.5 1.0 2.0 3.0 10.0)
        ;;
    *)
        echo "Usage: bash scripts/run_p16_validation.sh [p1|p2|all]"
        exit 1
        ;;
esac

for EPS in "${EPS_LIST[@]}"; do
    if [ "$EPS" == "0.5" ]; then
        CONFIG="${CONFIG_DIR}/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05.py"
        WORK_DIR="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05"
    else
        CONFIG="${CONFIG_DIR}/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps${EPS}.py"
        WORK_DIR="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps${EPS}"
    fi

    if [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Config not found: $CONFIG"
        continue
    fi

    echo ""
    echo "------------------------------------------"
    echo "[P1-6] Training Stochastic ε=${EPS}..."
    echo "  Config:   $CONFIG"
    echo "  Work dir: $WORK_DIR"
    echo "  Predicted mAP: $(python -c "
eps = ${EPS}
if eps < 0.5: print('0.735-0.745')
elif eps < 1.0: print('0.745-0.750')
elif eps < 3.0: print('0.750-0.752')
elif eps < 5.0: print('0.748-0.751')
else: print('0.740-0.748')
")"
    echo "------------------------------------------"

    $PYTHON tools/train.py "$CONFIG" --work-dir "$WORK_DIR"
done

echo ""
echo "=========================================="
echo "P1-6: Results Summary"
echo "=========================================="
echo ""
echo "  ε     | mAP (实测) | mAP (预测)  | 区间"
echo "  ------|-----------|-------------|----------"
echo "  0     | 0.735     | —           | 区间1 (OT主导)"
echo "  0.5   | —         | 0.745-0.750 | 区间1→2边界"
echo "  1.0   | —         | 0.750-0.752 | 区间2 (★最优点)"
echo "  2.0   | —         | 0.749-0.751 | 区间2"
echo "  3.0   | —         | 0.748-0.751 | 区间2"
echo "  5.0   | 0.751     | —           | 区间2 (已验证)"
echo "  10.0  | —         | 0.740-0.748 | 区间2→3边界"
echo "  50.0  | 0.736     | —           | 区间3 (退化)"
echo "  ∞     | 0.751     | —           | Random baseline"
echo ""

for EPS in "${EPS_LIST[@]}"; do
    if [ "$EPS" == "0.5" ]; then
        WORK_DIR="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05"
    else
        WORK_DIR="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps${EPS}"
    fi

    if [ -d "$WORK_DIR" ]; then
        LOG=$(find "$WORK_DIR" -name "*.log" -type f | head -1)
        if [ -n "$LOG" ]; then
            BEST_MAP=$(grep "coco/bbox_mAP" "$LOG" | grep -oP "coco/bbox_mAP: [0-9.]+" | sort -t: -k2 -rn | head -1 | grep -oP "[0-9.]+$")
            FINAL_MAP=$(grep "coco/bbox_mAP" "$LOG" | tail -1 | grep -oP "coco/bbox_mAP: [0-9.]+" | grep -oP "[0-9.]+$")
            echo "  ε=${EPS}: best_mAP=${BEST_MAP:-N/A}, final_mAP=${FINAL_MAP:-N/A}"
        else
            echo "  ε=${EPS}: training log not found"
        fi
    else
        echo "  ε=${EPS}: work_dir not found"
    fi
done

echo ""
echo "验证标准:"
echo "  ✅ 如果 ε=1.0 mAP ≥ 0.750 → 三区间模型验证通过"
echo "  ✅ 如果 ε=1.0 ≥ ε=5.0 → ε*≈1-3 预测正确"
echo "  ❌ 如果 ε=1.0 < 0.745 → 模型需要修正"
echo "  ❌ 如果 ε=0.5 ≥ ε=1.0 → 多样性收益被高估"
