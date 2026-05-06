_base_ = ["./ldmdet_phase1+2.py"]

model = dict(
    bbox_head=dict(
        domain_adaptive_prior=True,
    )
)
