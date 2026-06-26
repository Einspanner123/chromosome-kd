#!/bin/bash
# 瓶颈消融实验 — 逐一运行，每次只改一个变量
# 用法: bash run_bottleneck_ablations.sh
# 每个实验 ~17h，串行执行
# 完成后自动启动下一个，也可 Ctrl+C 中断后手动 --resume

BASE_DIR="work_dirs/bottleneck_ablation"
GPU=0

EXPERIMENTS=(
  "bottleneck/ddpm_baseline:ddpm_vs_rf"
  "bottleneck/no_ensemble:no_ensemble"
  "bottleneck/single_step_sampling:single_step"
)

echo "=============================================="
echo "瓶颈消融实验 — 3个实验串行"
echo "每个约 17h，总计 ~51h"
echo "=============================================="
echo ""

for item in "${EXPERIMENTS[@]}"; do
  CONFIG="${item%%:*}"
  NAME="${item##*:}"
  WORK_DIR="${BASE_DIR}/${NAME}"

  echo "----------------------------------------------"
  echo "启动: $NAME  (config: $CONFIG)"
  echo "输出: $WORK_DIR"
  echo "----------------------------------------------"

  CMD="python experiments/runners/train.py experiments/configs/${CONFIG}.py --seed 42 --gpu-id ${GPU} --work-dir ${WORK_DIR}"

  if [ -f "${WORK_DIR}/last_checkpoint" ]; then
    echo "  检测到已有 checkpoint，添加 --resume"
    CMD="${CMD} --resume"
  fi

  echo "  $CMD"
  echo ""
  eval "$CMD"

  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 137 ] && [ $EXIT_CODE -ne -15 ]; then
    echo "错误: $NAME 异常退出 (code=$EXIT_CODE)，停止后续实验"
    exit $EXIT_CODE
  fi

  echo ""
  echo "完成: $NAME"
  echo ""
done

echo "=============================================="
echo "所有瓶颈消融实验已完成"
echo "=============================================="
