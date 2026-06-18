#!/bin/bash
# Chromosome v2 数据集验证 — 在另一台服务器执行
cd /root/chromosome-kd && git pull origin refactor/ldmdet && pip install -e .

echo "=== Chromosome v2 validation (GHSS, 1 seed) ==="
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2.py --seeds 42 --gpus 0 --base-dir work_dirs/multi_dataset/chromo_v2

echo "=== Chromosome v2 done ==="
