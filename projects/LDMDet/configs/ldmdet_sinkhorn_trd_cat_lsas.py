"""Experiment: Stochastic Sinkhorn + TRD + CAT + LSAS + Velocity Prediction + Heun

Predicted mAP: 0.753-0.754

Combines:
- Stochastic Sinkhorn OT (eps=5, ot_sample=True) from sinkhorn_sample_eps5 (0.751)
- TRD + CAT + LSAS + velocity prediction from trd_full (0.752)
- Heun second-order solver

Rationale: trd_full already achieves 0.752 using NEAREST OT (simple argmin).
Replacing nearest OT with stochastic Sinkhorn OT should provide:
- Better training diversity (stochastic coupling preserves entropy)
- Epsilon-controlled trade-off between structure and diversity
- Column-marginal constraints preventing target starvation

The TRD+CAT+LSAS suite improves ODE path quality and inference-time refinement,
while stochastic Sinkhorn improves training coupling diversity. These are
orthogonal gain sources that should compound.
"""

_base_ = ["./ldmdet_rf_heun_shifted_bs2.py"]

model = dict(
    bbox_head=dict(
        single_head=dict(
            time_conditioning="adaln_zero",
        ),
        # Stochastic Sinkhorn OT (NO group hierarchy — pure Sinkhorn)
        ot_coupling=True,
        ot_matcher="sinkhorn",
        ot_epsilon=5.0,
        ot_num_iters=20,
        ot_sample=True,
        ot_group_hierarchical=False,
        # Transport-Refinement Decomposition
        use_trd=True,
        trd_self_cond_prob=0.5,
        # Curvature-Aware Training
        use_cat=True,
        cat_weight=1.0,
        cat_delta_t=0.05,
        # Loss-Sensitive Adaptive Scheduling
        use_lsas=True,
        lsas_num_bins=100,
        lsas_temp=1.0,
        # Velocity prediction
        prediction_mode="velocity",
        velocity_loss_weight=1.0,
        # Solver
        solver_type="heun",
    )
)

work_dir = "work_dirs/ldmdet_sinkhorn_trd_cat_lsas"
