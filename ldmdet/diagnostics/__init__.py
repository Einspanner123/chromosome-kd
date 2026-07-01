"""ldmdet.diagnostics — 训练诊断工具包.

提供权重/梯度/激活统计、数值健康度检查、模块分组等工具,
供 TrainingDiagnosticsHook 和方向特定诊断模块使用.

模块划分:
- stats.py: 公共统计工具 (所有方向共享)
- hooks.py: TrainingDiagnosticsHook (基础监控)
- trajectory_diag.py: 方向四非线性轨迹诊断
- feature_bridge_diag.py: 方向六 FBM 诊断
"""

from ldmdet.diagnostics.trajectory_diag import (
    TrajectoryDiagnosticsCallback,
    compute_ot_coupling_stats,
    compute_path_curvature,
    compute_scale_conditioned_stats,
)

__all__ = [
    # 方向四: 非线性轨迹
    'TrajectoryDiagnosticsCallback',
    'compute_scale_conditioned_stats',
    'compute_ot_coupling_stats',
    'compute_path_curvature',
]
