#!/bin/bash
# docker-train-list.sh — 列出所有运行中的训练容器
# 用法: ./docker-train-list.sh

set -euo pipefail

RUNNING=$(docker ps --filter "name=train-" --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}')

if [ -z "$RUNNING" ] || [ "$(echo "$RUNNING" | wc -l)" -le 1 ]; then
    echo "当前没有运行中的训练任务"
else
    echo "$RUNNING"
    echo ""
    docker ps --filter "name=train-" --format '{{.Names}}' | while read name; do
        GPU=$(docker inspect "$name" --format '{{range .Config.Env}}{{if contains "CUDA_VISIBLE_DEVICES" .}}{{.}}{{end}}{{end}}' 2>/dev/null || echo "N/A")
        echo "  $name → GPU: ${GPU#CUDA_VISIBLE_DEVICES=}"
    done
fi
