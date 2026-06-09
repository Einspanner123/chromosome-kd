#!/bin/bash
# build-train-image.sh — 构建 chromo-train 训练镜像
# 使用 buildx + --network host 来绕过 Clash TUN DNS 问题
set -euo pipefail
cd "$(dirname "$0")"
echo "构建 chromo-train:latest ..."
docker buildx build \
  --network host \
  --build-arg HTTP_PROXY=http://127.0.0.1:7890 \
  --build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
  --build-arg http_proxy=http://127.0.0.1:7890 \
  --build-arg https_proxy=http://127.0.0.1:7890 \
  -t chromo-train:latest \
  -f Dockerfile.train \
  --load .
echo ""
echo "✅ 构建完成！"
docker images chromo-train:latest
