"""Group-Hierarchical Sinkhorn OT — current SOTA at 0.752 mAP.

Matches the historical best run at:
  work_dirs/ldmdet_flowdet_adaln_group_hierarchical_stoch/20260429_100047
"""
_base_ = ["./ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py"]

model = dict(
    bbox_head=dict(
        ot_group_hierarchical=True,
    )
)

work_dir = "work_dirs/ldmdet_group_hierarchical_stoch"
