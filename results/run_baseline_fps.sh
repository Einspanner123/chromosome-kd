#!/bin/bash
# ==============================================================================
# Baseline FPS Benchmark 命令脚本
# 生成日期: 2026-07-15
# 用途: 测量 Cascade R-CNN / YOLOX-S / DiffusionDet 的端到端 FPS
# 注意: 无本地 checkpoint, 使用 random weights (延迟测量仍然有效)
# 需 GPU, 请在终端中手动运行
# ==============================================================================

set -e

cd /home/linkst/workspace/projects/chromosome-kd

echo "============================================================"
echo "Baseline FPS Benchmark (Cascade / YOLOX / DiffusionDet)"
echo "GPU: 自动选择 GPU 0"
echo "Image: 512x512, Batch: 1, Warmup: 10, Iters: 100"
echo "============================================================"

# 运行 3 个 baseline (mAP < A4=0.863, 论文中保留)
python ldmdet/tools/benchmark_fps.py \
    --models cascade_rcnn yolox_s diffusiondet \
    --img-size 512 --iters 100 --gpu 0

echo "============================================================"
echo "Baseline FPS benchmark 完成!"
echo "结果保存在 results/benchmark_fps_*.md"
echo "============================================================"
