"""mmdet ↔ ldmdet 桥接层

唯一依赖 mmdet/mmengine 的代码集中于此。
将 mmdet 配置解析为 ldmdet 纯 PyTorch 对象。
"""

from experiments.mmdet_bridge.detector import LDMDetDetector  # noqa: F401
from experiments.mmdet_bridge.hooks import (  # noqa: F401
    AsyncCheckpointHook,
    CopyProjectHook,
    PredictionVisHook,
    WeightSummaryHook,
)
from experiments.mmdet_bridge.registry import register_all  # noqa: F401

# SetDiff detector — setdiff 包已归档, 跳过导入
try:
    from experiments.mmdet_bridge.setdiff_detector import SetDiffDetector  # noqa: F401
except ImportError:
    pass  # setdiff 已归档, 不影响主路线推理
