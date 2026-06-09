#!/bin/bash
# docker-train.sh — 后台启动训练（适合 kiki 通过 QQ 调用）
# 用法:
#   ./docker-train.sh <config_path> [gpu_id] [container_name_suffix]
#
# 示例:
#   ./docker-train.sh projects/LDMDet/configs/sota_seed42.py 0 sota42
#   ./docker-train.sh projects/LDMDet/configs/sota_seed42.py 0
#
# 返回容器名，用于后续 stop/logs

set -euo pipefail

CONFIG="${1:?错误: 请指定配置文件路径}"
GPU_ID="${2:-0}"
SUFFIX="${3:-$(basename "${CONFIG%.py}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')}"

CONTAINER_NAME="train-${SUFFIX}"
PROJECT_DIR="/media/ross/8TB/linkst/chromo/chromosome-kd"
PROJECT_DIR_M="/workspace/project"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "错误: 容器 ${CONTAINER_NAME} 已在运行中"
    echo "使用 ./docker-train-stop.sh ${CONTAINER_NAME} 停止它"
    exit 1
fi

CONFIG_HOST="${PROJECT_DIR}/${CONFIG}"
if [ ! -f "${CONFIG_HOST}" ]; then
    echo "错误: 找不到配置文件 ${CONFIG_HOST}"
    exit 1
fi

echo "启动训练容器: ${CONTAINER_NAME}"
echo "  Config: ${CONFIG}"
echo "  GPU:    ${GPU_ID}"

docker run --gpus all -d --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    -v "${PROJECT_DIR}:${PROJECT_DIR_M}" \
    -w "${PROJECT_DIR_M}" \
    -e CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    -e CUBLAS_WORKSPACE_CONFIG=":4096:8" \
    chromo-train:latest \
    sh -c "pip install -e . --no-build-isolation -q 2>/dev/null && python tools/train.py ${CONFIG}"

echo "容器 ${CONTAINER_NAME} 已启动"
echo ""
echo "后续命令:"
echo "  ./docker-train-logs.sh   ${CONTAINER_NAME}  查看日志"
echo "  ./docker-train-list.sh                      列出运行中的训练"
echo "  ./docker-train-stop.sh  ${CONTAINER_NAME}   停止训练"
