_base_ = ['../ldmdet_rf_heun_shifted_bs2.py']

optim_wrapper = dict(
    accumulative_counts=2,
)
