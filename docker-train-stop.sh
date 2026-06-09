#!/bin/bash
# docker-train-stop.sh — 停止训练容器
# 用法: ./docker-train-stop.sh <container_name>

set -euo pipefail

NAME="${1:?错误: 请指定容器名 (例如 train-sota_seed42)}"

if ! docker ps --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "容器 ${NAME} 未在运行中"
    docker ps --format 'table {{.Names}}\t{{.Status}}' | grep "^train-" || true
    exit 1
fi

echo "正在停止 ${NAME}..."
docker stop "${NAME}"
echo "已停止 ${NAME}"
