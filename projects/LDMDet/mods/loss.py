"""损失函数、匹配代价、匹配器和准则 - 向后兼容的重导出

所有组件已拆分到独立模块中，此文件仅保留重导出以兼容旧 import 路径。
新代码应直接从子模块导入。
"""

from .costs import BBoxL1Cost, FocalLossCost, IoUCost, RelativeL1Cost
from .criterion import DiffusionDetCriterion
from .losses import (
    FlowMatchingVelocityLoss,
    FocalLoss,
    GIoULoss,
    L1Loss,
    sigmoid_focal_loss,
)
from .matcher import DiffusionDetMatcher

__all__ = [
    'BBoxL1Cost',
    'DiffusionDetCriterion',
    'DiffusionDetMatcher',
    'FlowMatchingVelocityLoss',
    'FocalLoss',
    'FocalLossCost',
    'GIoULoss',
    'IoUCost',
    'L1Loss',
    'RelativeL1Cost',
    'sigmoid_focal_loss',
]
