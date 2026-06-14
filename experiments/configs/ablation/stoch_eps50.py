"""ε 扫描 — Sinkhorn Stochastic ε=50"""
_base_ = ['../ldmdet/sinkhorn_stochastic.py']
model = dict(bbox_head=dict(coupling=dict(type='sinkhorn_stochastic', epsilon=50.0)))
