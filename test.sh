#!/bin/bash
# LDMDet 推理测试
# Usage:
#   bash test.sh experiments/configs/ldmdet/sinkhorn_stochastic.py --checkpoint work_dirs/xxx/best.pth

cd "$(dirname "$0")"

exec python experiments/runners/test.py "$@"
