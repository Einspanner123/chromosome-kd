#!/bin/bash
# Chromosome v2 完整消融 — 串行执行 (Random → DDPM → HardOT → Sinkhorn → GHSS)
set -euo pipefail
cd /media/ross/8TB/linkst/chromo/chromosome-kd

SEEDS="42,123,789"
GPU=0
OUTDIR="work_dirs/multi_dataset"

echo "[1/5] Random baseline"
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2_random.py \
    --seeds "$SEEDS" --gpus "$GPU" --base-dir "$OUTDIR/chromo_v2_random"

echo "[2/5] DDPM baseline"
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2_ddpm.py \
    --seeds "$SEEDS" --gpus "$GPU" --base-dir "$OUTDIR/chromo_v2_ddpm"

echo "[3/5] Hard OT"
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2_hard_ot.py \
    --seeds "$SEEDS" --gpus "$GPU" --base-dir "$OUTDIR/chromo_v2_hard_ot"

echo "[4/5] Sinkhorn stochastic ε=5"
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2_sinkhorn.py \
    --seeds "$SEEDS" --gpus "$GPU" --base-dir "$OUTDIR/chromo_v2_sinkhorn"

echo "[5/5] GHSS ε=5"
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2.py \
    --seeds "$SEEDS" --gpus "$GPU" --base-dir "$OUTDIR/chromo_v2_ghss"

echo "=== ALL DONE ==="
