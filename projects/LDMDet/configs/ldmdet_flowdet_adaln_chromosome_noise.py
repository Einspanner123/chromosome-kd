_base_ = ["./ldmdet_flowdet_adaln.py"]

model = dict(
    bbox_head=dict(
        noise_sampler=dict(
            type="PurePyTorchStructuredNoiseSampler",
            num_proposals=500,
            noise_scale=0.5,
            strategy="chromosome",
            snr_scale=2.0,
        ),
    ),
)
