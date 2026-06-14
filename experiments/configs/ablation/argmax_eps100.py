"""ε 扫描 — Sinkhorn Argmax ε=100 (退化到硬 OT)"""
_base_ = ['../ldmdet/sinkhorn_argmax.py']
model = dict(bbox_head=dict(coupling=dict(type='sinkhorn_argmax', epsilon=100.0)))
