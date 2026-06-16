"""LDMDet + DPM-Solver++ (二阶, 6步) — 高阶 ODE 求解器"""
_base_ = ['rf_heun_adaln.py']
model = dict(bbox_head=dict(
    solver_type='dpm_solver_pp',
    sampling_timesteps=6,
))
