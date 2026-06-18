#!/bin/bash
# A6000 消融 — 从 DDPM 开始（Random 已完成）
cd /media/ross/8TB/linkst/chromo/chromosome-kd

echo "[2/5] DDPM baseline"
python experiments/runners/train_multi_seed.py experiments/configs/baselines/diffusiondet_ddpm.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/ddpm

echo "[3/5] Hard OT"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/hard_ot.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/hard_ot

echo "[4/5] Sinkhorn stochastic eps=5"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/sinkhorn_stochastic.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/sinkhorn_stochastic

echo "[5/5] GHSS eps=5"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/ghss.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/ghss

echo "=== ALL DONE ==="
