#!/bin/bash
# LDMDet NeurIPS 冲刺实验脚本
# 使用方法: bash run_all_experiments.sh [experiment_group]
# experiment_group: p0_eval | p1_train | p1_ot | all

set -e
cd /home/linkst/workplace/chromo/chromosome-kd

PYTHON="/home/linkst/miniconda3/envs/chromo/bin/python"
CONFIG_DIR="projects/LDMDet/configs"

# ============================================================
# P0: 多步推理评估实验 (不需要训练，约 30 分钟)
# ============================================================
run_p0_eval() {
    echo "=========================================="
    echo "P0: Multi-step Inference Evaluation"
    echo "=========================================="

    # --- 最佳模型: ldmdet_flowdet_adaln (RF + AdaLN-Zero) ---
    echo "[P0] Evaluating ldmdet_flowdet_adaln (best RF model)..."
    for STEPS in 1 2 4 8; do
        echo "  [P0] adaln_steps${STEPS}..."
        $PYTHON tools/test.py \
            ${CONFIG_DIR}/ldmdet_flowdet_adaln.py \
            work_dirs/ldmdet_flowdet_adaln/best_coco_bbox_mAP_epoch_56.pth \
            --cfg-options model.bbox_head.sampling_timesteps=${STEPS} \
            --work-dir work_dirs/eval_multistep/adaln_steps${STEPS}
    done

    # --- DDPM Baseline ---
    echo "[P0] Evaluating ldmdet_baseline (DDPM)..."
    for STEPS in 1 2 4 8; do
        echo "  [P0] baseline_steps${STEPS}..."
        $PYTHON tools/test.py \
            ${CONFIG_DIR}/ldmdet_baseline.py \
            work_dirs/ldmdet_baseline/best_coco_bbox_mAP_epoch_68.pth \
            --cfg-options model.bbox_head.sampling_timesteps=${STEPS} \
            --work-dir work_dirs/eval_multistep/baseline_steps${STEPS}
    done

    echo "[P0] Done! Results summary:"
    echo "Model | 1-step | 2-step | 4-step | 8-step"
    echo "------|--------|--------|--------|-------"
    for model in adaln baseline; do
        row="$model"
        for steps in 1 2 4 8; do
            log="work_dirs/eval_multistep/${model}_steps${steps}/$(ls work_dirs/eval_multistep/${model}_steps${steps}/ 2>/dev/null | grep '.log$' | head -1)"
            if [ -f "$log" ]; then
                map=$(grep "coco/bbox_mAP" "$log" | grep -oP "coco/bbox_mAP: [0-9.]+" | head -1 | grep -oP "[0-9.]+$")
                row="$row | $map"
            else
                row="$row | N/A"
            fi
        done
        echo "$row"
    done
}

# ============================================================
# P1: 两阶段 Reflow 训练 - Stage 2
# ============================================================
run_p1_train() {
    echo "=========================================="
    echo "P1: Two-Stage Reflow Training - Stage 2"
    echo "=========================================="

    # Stage 1 已完成: ldmdet_flowdet_adaln_reflow_det_only_lr1e6
    # Best checkpoint: best_coco_bbox_mAP_epoch_3.pth (mAP=0.741)

    # Stage 2: Freeze shared layers, train velocity_head only
    echo "[P1] Starting Stage 2: freeze shared + velocity training..."
    $PYTHON tools/train.py \
        ${CONFIG_DIR}/ldmdet_flowdet_adaln_reflow_freeze_stage2.py \
        --work-dir work_dirs/ldmdet_flowdet_adaln_reflow_freeze_stage2

    echo "[P1] Stage 2 training complete."
    echo "[P1] Evaluating Stage 2 with multi-step inference..."
    CKPT=$(ls -t work_dirs/ldmdet_flowdet_adaln_reflow_freeze_stage2/best_*.pth 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        for STEPS in 1 2 4 8; do
            echo "  [P1] stage2_steps${STEPS}..."
            $PYTHON tools/test.py \
                ${CONFIG_DIR}/ldmdet_flowdet_adaln_reflow_freeze_stage2.py \
                $CKPT \
                --cfg-options model.bbox_head.sampling_timesteps=${STEPS} \
                --work-dir work_dirs/eval_multistep/stage2_steps${STEPS}
        done
    fi
}

# ============================================================
# P1: OT Diversity Paradox 验证 - Sinkhorn ε 扫描
# ============================================================
run_p1_ot() {
    echo "=========================================="
    echo "P1: OT Diversity Paradox - Sinkhorn ε Scan"
    echo "=========================================="

    # 已有: eps=1.0 (ldmdet_flowdet_adaln_ot_sinkhorn, mAP=0.720)
    # 新增: eps=5, 10, 50, 100
    for EPS in 5 10 50 100; do
        echo "[P1] Training with Sinkhorn OT ε=${EPS}..."
        $PYTHON tools/train.py \
            ${CONFIG_DIR}/ldmdet_flowdet_adaln_ot_sinkhorn_eps${EPS}.py \
            --work-dir work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_eps${EPS}
    done

    echo "[P1] OT ε scan complete. Results summary:"
    echo "ε | Best mAP"
    echo "--|--------"
    for EPS in 1 5 10 50 100; do
        dir="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_eps${EPS}"
        if [ -d "$dir" ]; then
            log=$(find "$dir" -name "*.log" | head -1)
            if [ -n "$log" ]; then
                map=$(grep "coco/bbox_mAP" "$log" | grep -oP "coco/bbox_mAP: [0-9.]+" | sort -t: -k2 -rn | head -1 | grep -oP "[0-9.]+$")
                echo "$EPS | $map"
            else
                echo "$EPS | N/A"
            fi
        else
            echo "$EPS | not trained"
        fi
    done
    echo "∞ (random) | 0.751"
}

# ============================================================
# ============================================================
# P1-6: 最优 ε 验证 — Stochastic Coupling ε 扫描
# ============================================================
run_p16_validation() {
    echo "=========================================="
    echo "P1-6: Optimal Epsilon Validation"
    echo "=========================================="

    for EPS in 0.5 1.0 2.0 3.0 10.0; do
        if [ "$EPS" == "0.5" ]; then
            CONFIG="${CONFIG_DIR}/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05.py"
            WORK_DIR="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05"
        else
            CONFIG="${CONFIG_DIR}/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps${EPS}.py"
            WORK_DIR="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps${EPS}"
        fi

        if [ ! -f "$CONFIG" ]; then
            echo "[P1-6] Config not found: $CONFIG, skipping"
            continue
        fi

        echo "[P1-6] Training Stochastic ε=${EPS}..."
        $PYTHON tools/train.py "$CONFIG" --work-dir "$WORK_DIR"
    done

    echo "[P1-6] Validation complete. Results:"
    echo "  ε | Best mAP | Predicted"
    echo "  --|----------|----------"
    for EPS in 0.5 1.0 2.0 3.0 5.0 10.0 50.0; do
        if [ "$EPS" == "0.5" ]; then
            dir="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05"
        else
            dir="work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps${EPS}"
        fi
        if [ -d "$dir" ]; then
            log=$(find "$dir" -name "*.log" | head -1)
            if [ -n "$log" ]; then
                map=$(grep "coco/bbox_mAP" "$log" | grep -oP "coco/bbox_mAP: [0-9.]+" | sort -t: -k2 -rn | head -1 | grep -oP "[0-9.]+$")
                echo "  $EPS | $map | —"
            else
                echo "  $EPS | N/A | —"
            fi
        else
            echo "  $EPS | not trained | —"
        fi
    done
    echo "  ∞ (random) | 0.751 | —"
}

# Main
# ============================================================
case "${1:-all}" in
    p0_eval)
        run_p0_eval
        ;;
    p1_train)
        run_p1_train
        ;;
    p1_ot)
        run_p1_ot
        ;;
    p16_validation)
        run_p16_validation
        ;;
    all)
        run_p0_eval
        run_p1_train
        run_p1_ot
        ;;
    *)
        echo "Usage: bash run_all_experiments.sh [p0_eval|p1_train|p1_ot|p16_validation|all]"
        exit 1
        ;;
esac
