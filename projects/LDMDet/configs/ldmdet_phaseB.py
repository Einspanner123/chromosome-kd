_base_ = ['./ldmdet_phase1+2.py']

# Phase B: Phase 1+2 + GO-LSD iterative head self-distillation
# Inspired by D-FINE (ICLR 2025). Last head distills to earlier heads.
# Training-only, zero inference cost.

model = dict(bbox_head=dict(
    go_lsd=True,
    go_lsd_weight=1.0,
))
