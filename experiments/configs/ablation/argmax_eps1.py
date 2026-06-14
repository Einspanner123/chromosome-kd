"""ε 扫描 — Sinkhorn Argmax ε=1"""
_base_ = ['../ldmdet/sinkhorn_argmax.py']
model = dict(bbox_head=dict(coupling=dict(type='sinkhorn_argmax', epsilon=1.0)))
