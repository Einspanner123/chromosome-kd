_base_ = ['../ldmdet_rf_heun_shifted_bs2.py']

model = dict(
    bbox_head=dict(
        t_sampling='stratified',
        t_sampling_bins=8,
    ),
)
