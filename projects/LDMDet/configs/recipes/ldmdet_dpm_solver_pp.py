_base_ = ['../_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py']

# ==============================================================================
# DPM-Solver++ 多步法配置 (Phase 1: 2阶, 6步, 7 NFE)
# ------------------------------------------------------------------------------
# 基于 0.753 SOTA 配置，仅替换求解器。
# - solver_type='dpm_solver_pp': t 空间直接插值，精确积分半线性 ODE
# - sampling_timesteps=6: 6 步 = 7 NFE (起步 1 + 后续 5×1 + 终点 1)
# - 对比基准: Heun 4 步 = 8 NFE
# ==============================================================================

model = dict(
    bbox_head=dict(
        solver_type='dpm_solver_pp',
        sampling_timesteps=6,
    )
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(
            type='SwanlabVisBackend',
            init_kwargs=dict(
                project='chromosome-kd-dpm',
                experiment_name='dpm_solver_pp_order2_s6',
                description='DPM-Solver++ order=2, 6 steps (7 NFE) on SOTA base',
            ),
        ),
    ],
)

work_dir = 'work_dirs/ldmdet_dpm_solver_pp_o2_s6'
