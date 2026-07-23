"""ldmdet.diagnostics — 训练诊断工具包.

提供权重/梯度/激活统计、数值健康度检查、模块分组等工具,
供 TrainingDiagnosticsHook 使用.

模块划分:
- stats.py: 公共统计工具
- hooks.py: TrainingDiagnosticsHook (基础监控 + Probe 生命周期管理)
- instrumentation.py: Probe 运行时探针 (模型插桩核心)
- feature_bridge_diag.py: 方向六 FBM 诊断
"""

__all__ = ['probe']

from ldmdet.diagnostics.instrumentation import probe
