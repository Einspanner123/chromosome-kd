#!/bin/bash
# Chromosome v2 数据集验证 — 串行执行
cd /root/chromosome-kd && git pull origin refactor/ldmdet && pip install -e .

echo "[1/2] Chromosome v2 — Random baseline (1 seed)"
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2_random.py --seeds 42 --gpus 0 --base-dir work_dirs/multi_dataset/chromo_v2_random

echo "[2/2] Chromosome v2 — GHSS (1 seed)"
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2.py --seeds 42 --gpus 0 --base-dir work_dirs/multi_dataset/chromo_v2_ghss

echo "=== Chromosome v2 ALL DONE ==="
