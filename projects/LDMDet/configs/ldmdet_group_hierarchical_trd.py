"""Experiment: Group-Hierarchical Sinkhorn OT + TRD + Heun

Predicted mAP: 0.753-0.754

Combines:
- Group-Hierarchical Sinkhorn OT (stochastic, eps=5) from group_hierarchical_stoch (0.752)
- Transport-Refinement Decomposition (TRD) from trd_full (0.752)
- Heun second-order solver

Rationale: group-hierarchical OT provides better training couplings by preventing
cross-group mismatches. TRD adds self-conditioned refinement during inference.
These two mechanisms are orthogonal — OT improves the training signal, TRD improves
how the model uses that signal at inference time.
"""

_base_ = ['./ldmdet_rf_heun_shifted_bs2.py']

model = dict(
    bbox_head=dict(
        single_head=dict(time_conditioning='adaln_zero', ),
        # Group-Hierarchical Sinkhorn OT
        ot_coupling=True,
        ot_matcher='sinkhorn',
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
        ot_group_hierarchical=True,
        # Transport-Refinement Decomposition
        use_trd=True,
        trd_self_cond_prob=0.5,
        # Solver
        solver_type='heun',
    ))

work_dir = 'work_dirs/ldmdet_group_hierarchical_trd'
