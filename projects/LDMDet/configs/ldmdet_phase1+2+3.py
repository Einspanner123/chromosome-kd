_base_ = ['./ldmdet_phase1+2.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            interact_type='linear_cross_attn',
            self_attn_type='linear',
        ), ))
