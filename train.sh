#!/bin/bash

# 用法:
# 默认使用GPU0: >> ./train.sh config
# 使用第二块GPU: >> ./train.sh 1 config

# 检查第一个参数是否为数字
if [[ $1 =~ ^[0-9]+$ ]]; then
    GPU_ID=$1
    # 移除第一个参数，剩余参数传递给train.py
    shift
else
    # 若第一个参数非数字或无参数，设置默认GPU_ID（如0，根据实际需求调整）
    GPU_ID=0
fi

output_notes(){
    if [ ! -z "$USER_NOTES" ]; then
    echo ""
    echo "$USER_NOTES"
    echo ""
    fi
}

read -p "(Notes)" USER_NOTES
trap 'output_notes' EXIT


# 设置gpu id并运行训练脚本
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train.py "$@"