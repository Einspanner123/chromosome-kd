#!/bin/bash
# 前后加速比对比 Benchmark
#
# 验证优化是否带来真实加速 (而非反向优化)
# 方法: 在优化前 (git stash) 和优化后分别运行 benchmark_inference.py, 对比延迟
#
# Usage:
#   bash benchmark_before_after.sh [GPU_ID]
#
# 输出:
#   results/benchmark_before_after_<timestamp>.txt

set -e

GPU_ID=${1:-0}
PROJECT_ROOT="/home/linkst/workspace/projects/chromosome-kd"
cd "$PROJECT_ROOT"

TS=$(date +%Y%m%d_%H%M%S)
OUT_FILE="results/benchmark_before_after_${TS}.txt"
mkdir -p results

# benchmark 参数 (与论文 A3 SOTA 配置一致)
BENCH_ARGS="--solvers heun dpm_solver_pp --steps 4 --bs 1 --num-proposals 500 \
    --warmup 20 --iters 100 --no-profile --gpu ${GPU_ID}"

echo "========================================" | tee "$OUT_FILE"
echo "前后加速比对比 Benchmark" | tee -a "$OUT_FILE"
echo "GPU: ${GPU_ID}" | tee -a "$OUT_FILE"
echo "时间: ${TS}" | tee -a "$OUT_FILE"
echo "========================================" | tee -a "$OUT_FILE"

# ============================================================
# 1. 优化后 (当前代码)
# ============================================================
echo "" | tee -a "$OUT_FILE"
echo "=== [1/2] 优化后 (当前代码) ===" | tee -a "$OUT_FILE"
echo "Commit: $(git rev-parse --short HEAD)" | tee -a "$OUT_FILE"
echo "" | tee -a "$OUT_FILE"

python ldmdet/tools/benchmark_inference.py $BENCH_ARGS 2>&1 | tee -a "$OUT_FILE"

# ============================================================
# 2. 优化前 (git stash 回退)
# ============================================================
echo "" | tee -a "$OUT_FILE"
echo "=== [2/2] 优化前 (git stash 回退到 ccbccdd6) ===" | tee -a "$OUT_FILE"

STASHED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Stashing uncommitted changes..." | tee -a "$OUT_FILE"
    git stash | tee -a "$OUT_FILE"
    STASHED=true
fi

# 回退到优化前的 commit
git checkout ccbccdd6 -- \
    projects/LDMDetDiT/mods/rectified_flow.py \
    projects/LDMDetDiT/mods/dit_head.py \
    experiments/configs/_base_/default_runtime.py \
    experiments/configs/_base_/datasets/autokary_coco_detection.py \
    experiments/configs/_base_/datasets/chromo_24obj_coco_detection.py \
    experiments/configs/_base_/datasets/chromo_coco_detection.py \
    experiments/configs/_base_/datasets/chromo_kary_coco_detection.py \
    2>&1 | tee -a "$OUT_FILE" || echo "checkout failed, using stash only"

echo "Code version: $(git rev-parse --short HEAD) (优化前)" | tee -a "$OUT_FILE"
echo "" | tee -a "$OUT_FILE"

python ldmdet/tools/benchmark_inference.py $BENCH_ARGS 2>&1 | tee -a "$OUT_FILE"

# ============================================================
# 3. 恢复优化后代码
# ============================================================
echo "" | tee -a "$OUT_FILE"
echo "=== 恢复优化后代码 ===" | tee -a "$OUT_FILE"
git checkout HEAD -- \
    projects/LDMDetDiT/mods/rectified_flow.py \
    projects/LDMDetDiT/mods/dit_head.py \
    experiments/configs/_base_/default_runtime.py \
    experiments/configs/_base_/datasets/autokary_coco_detection.py \
    experiments/configs/_base_/datasets/chromo_24obj_coco_detection.py \
    experiments/configs/_base_/datasets/chromo_coco_detection.py \
    experiments/configs/_base_/datasets/chromo_kary_coco_detection.py \
    2>&1 | tee -a "$OUT_FILE"

if [ "$STASHED" = true ]; then
    git stash pop 2>&1 | tee -a "$OUT_FILE" || true
fi

echo "" | tee -a "$OUT_FILE"
echo "=== 完成 ===" | tee -a "$OUT_FILE"
echo "结果已保存到: $OUT_FILE" | tee -a "$OUT_FILE"
echo "" | tee -a "$OUT_FILE"
echo "请手动对比 [1/2] 和 [2/2] 的 Latency (ms) 和 FPS:"
echo "  - 若优化后 latency < 优化前: 优化有效"
echo "  - 若优化后 latency >= 优化前: 反向优化, 需回退该项"
