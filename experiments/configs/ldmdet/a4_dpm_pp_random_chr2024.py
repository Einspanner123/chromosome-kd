"""Dataset1 clean RF + DPM-Solver++ baseline with random coupling.

This configuration intentionally removes the historical stochastic Sinkhorn
coupling inherited by ``a4_dpm_pp_chr2024.py``.  It keeps every other A4
setting unchanged so paired OCGR runs isolate the structural intervention.

Training and validation seeds must be supplied by ``experiments/runners/train.py``::

    --seed <training-seed> --val-seed 42
"""

_base_ = ['./a4_dpm_pp_chr2024.py']

model = dict(
    bbox_head=dict(
        coupling=dict(
            _delete_=True,
            type='random',
        ),
    ),
)
