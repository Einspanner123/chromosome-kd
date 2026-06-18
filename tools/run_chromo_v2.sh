#!/bin/bash
# Chromosome v2 — Random baseline (GHSS already running/done)
cd /root/chromosome-kd && git pull origin refactor/ldmdet && pip install -e .
python experiments/runners/train_multi_seed.py experiments/configs/multiset/chromo_v2_random.py --seeds 42 --gpus 0 --base-dir work_dirs/multi_dataset/chromo_v2_random
