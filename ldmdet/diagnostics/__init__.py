"""ldmdet.diagnostics — 训练诊断工具包.

提供权重/梯度/激活统计、数值健康度检查、模块分组等工具,
供 TrainingDiagnosticsHook 和方向特定诊断模块使用.

模块划分:
- stats.py: 公共统计工具 (所有方向共享)
- hooks.py: TrainingDiagnosticsHook (基础监控)
- coupling_diag.py: 方向一耦合诊断 (可独立清理)
- count_diag.py: 方向二计数诊断 (可独立清理)
- snr_diag.py: 方向三 SNR 诊断 (可独立清理)
- trajectory_diag.py: 方向四非线性轨迹诊断 (可独立清理)
- hierarchical_diag.py: 方向五分层分类诊断 (可独立清理)
- feature_bridge_diag.py: 方向六 FBM 诊断 (可独立清理)
"""

from ldmdet.diagnostics.trajectory_diag import (
    TrajectoryDiagnosticsCallback,
    compute_ot_coupling_stats,
    compute_path_curvature,
    compute_scale_conditioned_stats,
)
from ldmdet.diagnostics.hierarchical_diag import (
    HierarchicalDiagnosticsCallback,
    compute_group_classification_stats,
    compute_group_confusion,
    compute_intra_group_classification_stats,
    compute_information_stats,
    compute_group_conditioning_stats,
)
from ldmdet.diagnostics.feature_bridge_diag import (
    FeatureBridgeDiagnosticsHook,
)

__all__ = [
    # 方向四: 非线性轨迹
    'TrajectoryDiagnosticsCallback',
    'compute_scale_conditioned_stats',
    'compute_ot_coupling_stats',
    'compute_path_curvature',
    # 方向五: 分层分类
    'HierarchicalDiagnosticsCallback',
    'compute_group_classification_stats',
    'compute_group_confusion',
    'compute_intra_group_classification_stats',
    'compute_information_stats',
    'compute_group_conditioning_stats',
    # 方向六: FBM 诊断
    'FeatureBridgeDiagnosticsHook',
]

