#!/bin/bash
# LDMDet 训练 CLI 入口
# Usage:
#   bash tools/train.sh experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py
#   bash tools/train.sh experiments/configs/ldmdet/sinkhorn_stochastic.py --seed 42

cd "$(dirname "$0")/.."

exec python experiments/runners/train.py "$@"
