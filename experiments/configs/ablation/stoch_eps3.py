"""ε 扫描 — Sinkhorn Stochastic ε=3"""
_base_ = ['../ldmdet/sinkhorn_stochastic.py']
model = dict(bbox_head=dict(coupling=dict(type='sinkhorn_stochastic', epsilon=3.0)))
