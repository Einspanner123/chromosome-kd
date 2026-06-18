#!/bin/bash
# A6000 完整消融 — 串行执行
cd /media/ross/8TB/linkst/chromo/chromosome-kd

echo "[1/5] Random baseline"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/rf_heun_adaln.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/rf_heun_adaln

echo "[2/5] DDPM baseline"
python experiments/runners/train_multi_seed.py experiments/configs/baselines/diffusiondet_ddpm.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/ddpm

echo "[3/5] Hard OT"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/hard_ot.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/hard_ot

echo "[4/5] Sinkhorn stochastic eps=5"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/sinkhorn_stochastic.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/sinkhorn_stochastic

echo "[5/5] GHSS eps=5"
python experiments/runners/train_multi_seed.py experiments/configs/ldmdet/ghss.py --seeds 42,123,789 --gpus 0 --base-dir work_dirs/multi_seed_aug/ghss

echo "=== ALL DONE ==="
