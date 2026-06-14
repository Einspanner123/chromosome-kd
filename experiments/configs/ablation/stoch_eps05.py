"""ε 扫描 — Sinkhorn Stochastic ε=0.5 (OT-dominated regime)"""
_base_ = ['../ldmdet/sinkhorn_stochastic.py']
model = dict(bbox_head=dict(coupling=dict(type='sinkhorn_stochastic', epsilon=0.5)))
