#!/bin/bash
# LDMDet 多 seed 训练
# Usage:
#   bash multi_seed.sh experiments/configs/ldmdet/sinkhorn_stochastic.py --seeds 42,123,456 --gpus 0,0,1

cd "$(dirname "$0")"

exec python experiments/runners/train_multi_seed.py "$@"
