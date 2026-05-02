_base_ = ["./ldmdet_flowdet_adaln.py"]

model = dict(
    bbox_head=dict(
        use_lsas=True,
        lsas_num_bins=100,
        lsas_temp=1.0,
    ),
)
