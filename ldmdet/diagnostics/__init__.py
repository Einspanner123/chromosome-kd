"""ldmdet.diagnostics — 训练诊断工具包.

提供权重/梯度/激活统计、数值健康度检查、模块分组等工具,
供 TrainingDiagnosticsHook 使用.

模块划分:
- stats.py: 公共统计工具
- hooks.py: TrainingDiagnosticsHook (基础监控)
- feature_bridge_diag.py: 方向六 FBM 诊断
"""

__all__ = []
