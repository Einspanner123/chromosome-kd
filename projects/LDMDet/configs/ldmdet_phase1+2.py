_base_ = ["./ldmdet_phase1.py"]

model = dict(
    bbox_head=dict(
        num_proposals=300,
        num_heads=4,
        proposal_prune_ratios=[1.0, 1.0, 0.8, 0.6],
        single_head=dict(
            dynamic_dim=32,
            pooler_resolution=5,
        ),
        roi_extractor=dict(
            roi_layer=dict(type="RoIAlign", output_size=5, sampling_ratio=2),
        ),
    )
)
