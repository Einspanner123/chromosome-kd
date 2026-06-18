#!/bin/bash
# A6000 Tier 1 — 核心消融串行执行
cd /media/ross/8TB/linkst/chromo/chromosome-kd

echo "=== Tier 1 Core Ablation ==="

echo "[1/4] Random baseline (should finish on its own)"
# Already running via multi_seed, wait for it
# multi_seed handles 3 seeds sequentially

echo "[2/4] Hard OT"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/hard_ot.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/hard_ot

echo "[3/4] Sinkhorn stochastic eps=5"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/sinkhorn_stochastic.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/sinkhorn_stochastic

echo "[4/4] GHSS eps=5"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/ghss.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/ghss

echo "=== Tier 1 done ==="
echo "Final report in work_dirs/multi_seed_aug/*/report.json"
