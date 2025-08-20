#!/bin/bash

# 用法:
# 默认使用GPU0: >> ./test.sh config checkpoint
# 使用指定GPU: >> ./test.sh 1 config checkpoint

# 检查第一个参数是否为数字
if [[ $1 =~ ^[0-9]+$ ]]; then
    GPU_ID=$1
    # 移除第一个参数，剩余参数传递给test.py
    shift
else
    # 若第一个参数非数字或无参数，设置默认GPU_ID为0
    GPU_ID=0
fi

# 设置CUDA_VISIBLE_DEVICES并运行测试脚本
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/test.py "$@" --show-dir ./show_preds