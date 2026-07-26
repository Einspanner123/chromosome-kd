#!/bin/bash
# ==============================================================================
# 批量 test set 评估脚本 — 用于 val/test mAP 一致性核对
# 生成日期: 2026-07-26
# 用途: 在 24obj test set (1000 张图) 上评估 Table 10 的 9 个模型
# 输出: results/test_eval_<timestamp>/ 目录下的日志和汇总 JSON
# ==============================================================================

PYTHON=/home/linkst/data/miniconda3/envs/chromo/bin/python
PROJECT=/home/linkst/workspace/projects/chromosome-kd
cd "$PROJECT"

RESULTS_DIR="results/test_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "=== Test Set Evaluation Summary ===" > "$SUMMARY_FILE"
echo "Timestamp: $(date)" >> "$SUMMARY_FILE"
echo "Test set: 1000 images, 45980 annotations" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

run_eval() {
    local idx=$1
    local total=$2
    local name=$3
    local config=$4
    local ckpt=$5
    local dataset=${6:-test}

    echo ""
    echo "============================================================"
    echo "=== $idx/$total $name ==="
    echo "=== config: $config"
    echo "=== ckpt: $ckpt"
    echo "=== dataset: $dataset"
    echo "============================================================"

    $PYTHON experiments/runners/test.py \
        "$config" \
        --checkpoint "$ckpt" \
        --dataset "$dataset" \
        --gpu-id 0 \
        --exp-name "$name" \
        2>&1 | tee "$RESULTS_DIR/${name}.log"

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[OK] $name completed" >> "$SUMMARY_FILE"
    else
        echo "[FAIL] $name (exit $status)" >> "$SUMMARY_FILE"
    fi

    # 提取 mAP
    grep -E "coco/bbox_mAP" "$RESULTS_DIR/${name}.log" | tail -10 >> "$SUMMARY_FILE"
    echo "---" >> "$SUMMARY_FILE"
}

# === ldmdet 系列 (使用 test_eval 配置, --dataset test) ===

run_eval 1 9 "a1_rf_heun" \
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a1_test_eval_24obj.py" \
    "work_dirs/a1_rf_heun_24obj/best_coco_bbox_mAP_epoch_62.pth"

run_eval 2 9 "a2_stochot_heun" \
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a2_test_eval_24obj.py" \
    "work_dirs/a3_full_sota_24obj/best_coco_bbox_mAP_epoch_114.pth"

run_eval 3 9 "a3_dpm_pp" \
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_test_eval_24obj.py" \
    "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth"

run_eval 4 9 "a3_io3_k300" \
    "experiments/configs/ldmdet/directions/inference_opt/a4_io3_k300_test_eval_24obj.py" \
    "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth"

run_eval 5 9 "a3_io3_k200" \
    "experiments/configs/ldmdet/directions/inference_opt/a4_io3_k200_test_eval_24obj.py" \
    "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth"

run_eval 6 9 "a3_io3_k100" \
    "experiments/configs/ldmdet/directions/inference_opt/a4_io3_k100_test_eval_24obj.py" \
    "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth"

# === baseline 系列 ===

run_eval 7 9 "cascade_rcnn" \
    "experiments/configs/baselines/benchmark_24obj/cascade_rcnn_r50.py" \
    "work_dirs/baselines/cascade_rcnn_r50/best_coco_bbox_mAP_epoch_72.pth" \
    test

run_eval 8 9 "yolox_s" \
    "experiments/configs/baselines/benchmark_24obj/yolox_s.py" \
    "work_dirs/baselines/yolox_s/best_coco_bbox_mAP_epoch_200.pth" \
    test

run_eval 9 9 "diffusiondet" \
    "experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm_test_eval.py" \
    "work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth"

echo ""
echo "============================================================"
echo "=== ALL DONE. Summary at: $SUMMARY_FILE ==="
echo "============================================================"
cat "$SUMMARY_FILE"
