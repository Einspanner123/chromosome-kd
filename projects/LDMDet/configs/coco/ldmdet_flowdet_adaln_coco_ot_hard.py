_base_ = ["./ldmdet_flowdet_adaln_coco.py"]

model = dict(
    bbox_head=dict(
        ot_coupling=True,
    ),
)
