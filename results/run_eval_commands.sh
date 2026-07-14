#!/bin/bash
# ==============================================================================
# 不需要训练的评估实验命令脚本
# 生成日期: 2026-07-15
# 用法: 在项目根目录执行 bash results/run_eval_commands.sh
# 注意: 需要 GPU, sandbox 内无法执行, 请在终端中手动运行
# ==============================================================================

set -e  # 任一命令失败则停止

A4_CONFIG="experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py"
A4_CKPT="work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth"
A4_TEST_24OBJ="experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_test_eval_24obj.py"
A4_TEST_CHROMO="experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_test_eval_chromo.py"

echo "============================================================"
echo "1. DPM-Solver++ 步数消融 (val set, A4 checkpoint)"
echo "============================================================"
# 1步
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 1 --solver-type dpm_solver_pp \
    --exp-name dpm_pp_1step

# 2步
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 2 --solver-type dpm_solver_pp \
    --exp-name dpm_pp_2step

# 3步
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 3 --solver-type dpm_solver_pp \
    --exp-name dpm_pp_3step

# 8步
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 8 --solver-type dpm_solver_pp \
    --exp-name dpm_pp_8step

echo "============================================================"
echo "2. DPM-Solver++ vs Heun 公平对比 (val set, A4 checkpoint)"
echo "============================================================"
# Heun 2步 (4 NFE) — 与 DPM++ 4步 (4 NFE) 公平对比
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 2 --solver-type heun \
    --exp-name heun_2step

# Heun 4步 (7 NFE) — 原始 A3 配置
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 4 --solver-type heun \
    --exp-name heun_4step

# DPM++ 4步 (4 NFE) — 已有结果 mAP=0.863, 此处确认
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 4 --solver-type dpm_solver_pp \
    --exp-name dpm_pp_4step

echo "============================================================"
echo "3. DPM-Solver++ 3阶 (val set, A4 checkpoint)"
echo "============================================================"
python experiments/runners/test.py $A4_CONFIG \
    --checkpoint $A4_CKPT --dataset val \
    --sampling-steps 4 --solver-type dpm_solver_pp_3 \
    --exp-name dpm_pp3_4step

echo "============================================================"
echo "4. Test set 评估 (24obj, A4 checkpoint)"
echo "============================================================"
python experiments/runners/test.py $A4_TEST_24OBJ \
    --checkpoint $A4_CKPT --dataset test \
    --exp-name a4_dpm_pp_test_24obj

echo "============================================================"
echo "5. Test set 评估 (旧数据集 Chromosome20240904)"
echo "注意: A4 checkpoint 在 24obj 上训练, 此为跨数据集评估"
echo "如需旧数据集训练的 checkpoint 评估, 请使用 phase9_dpm checkpoint"
echo "============================================================"
python experiments/runners/test.py $A4_TEST_CHROMO \
    --checkpoint $A4_CKPT --dataset test \
    --exp-name a4_dpm_pp_cross_test_chromo

echo "============================================================"
echo "6. FPS Benchmark (已有, 无需重复)"
echo "原始数据: results/benchmark_fps_20260714_231841.md"
echo "============================================================"

echo "============================================================"
echo "全部评估完成!"
echo "SwanLab 项目: ldmdet-inference"
echo "============================================================"
