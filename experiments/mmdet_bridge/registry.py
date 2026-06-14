"""集中注册所有 ldmdet 模块到 mmdet MODELS 注册表

导入此模块即自动完成注册，之后 mmdet config 可以通过 type='...' 引用。
"""

from mmdet.registry import MODELS

from ldmdet.core import DiffusionDetHead, DynamicConv, SingleDiffusionDetHead, SingleRoIExtractor
from ldmdet.criterion import (
    BBoxL1Cost,
    DiffusionDetCriterion,
    DiffusionDetMatcher,
    FocalLoss,
    FocalLossCost,
    GIoULoss,
    IoUCost,
    L1Loss,
)

# 注册所有纯 PyTorch 组件

MODELS.register_module(name='PurePyTorchDiffusionDetHead', module=DiffusionDetHead)
MODELS.register_module(name='PurePyTorchSingleDiffusionDetHead', module=SingleDiffusionDetHead)
MODELS.register_module(name='PurePyTorchSingleRoIExtractor', module=SingleRoIExtractor)
MODELS.register_module(name='PurePyTorchDiffusionDetCriterion', module=DiffusionDetCriterion)
MODELS.register_module(name='PurePyTorchDiffusionDetMatcher', module=DiffusionDetMatcher)
MODELS.register_module(name='PurePyTorchFocalLoss', module=FocalLoss)
MODELS.register_module(name='PurePyTorchL1Loss', module=L1Loss)
MODELS.register_module(name='PurePyTorchGIoULoss', module=GIoULoss)
MODELS.register_module(name='PurePyTorchFocalLossCost', module=FocalLossCost)
MODELS.register_module(name='PurePyTorchBBoxL1Cost', module=BBoxL1Cost)
MODELS.register_module(name='PurePyTorchIoUCost', module=IoUCost)

# 注册 bridge 层的 detector 和 hooks (导入即触发注册)
register_all = lambda: None  # 占位，实际注册在 import 时完成

# 注册 transforms
from experiments.mmdet_bridge.transforms import CLAHE, SmallObjectCopyPaste  # noqa: F401
