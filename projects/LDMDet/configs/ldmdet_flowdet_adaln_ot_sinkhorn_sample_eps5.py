_base_ = ['./ldmdet_rf_heun_shifted_bs2.py']

# Reconstructed from the historical best run stored under:
# work_dirs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/20260423_060304
#
# Key ingredients:
# - Rectified Flow + Heun + shifted schedule
# - AdaLN-Zero time conditioning
# - Sinkhorn OT coupling with stochastic sampling
model = dict(
    bbox_head=dict(
        single_head=dict(time_conditioning='adaln_zero', ),
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
    ))
