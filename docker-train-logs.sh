#!/bin/bash
# docker-train-logs.sh — 查看训练容器日志
# 用法:
#   ./docker-train-logs.sh <container_name>      # 查看最近的日志
#   ./docker-train-logs.sh <container_name> -f    # 持续追踪日志

set -euo pipefail

NAME="${1:?错误: 请指定容器名 (例如 train-sota_seed42)}"
shift || true

if ! docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "错误: 找不到容器 ${NAME}"
    docker ps -a --format 'table {{.Names}}\t{{.Status}}' | grep "^train-" || true
    exit 1
fi

docker logs "$@" "${NAME}"
