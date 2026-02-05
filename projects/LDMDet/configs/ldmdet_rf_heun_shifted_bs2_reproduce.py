_base_ = ["./ldmdet_rf_heun_shifted_bs2.py"]

# 继承自 ldmdet_rf_heun_shifted_bs2.py
# 显式固定随机种子以复现 20260127_145108 的实验结果

randomness = dict(seed=1769925607, deterministic=False, diff_rank_seed=True)

# 可以在这里根据需要微调其他参数
# 目前保持与 ldmdet_rf_heun_shifted_bs2.py 一致
