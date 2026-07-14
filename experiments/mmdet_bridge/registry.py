"""集中注册所有 ldmdet 模块到 mmdet MODELS 注册表 (force=True 覆盖旧注册)"""

from mmdet.registry import MODELS
from mmengine.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS

from ldmdet.core import (
    DiffusionDetHead, DynamicConv, NormalizedLinear, SingleDiffusionDetHead,
    SingleRoIExtractor,
)
from ldmdet.criterion import (
    BBoxL1Cost, DiffusionDetCriterion, DiffusionDetMatcher,
    FocalLoss, FocalLossCost, GIoULoss, IoUCost, L1Loss, SeesawLoss,
)

# 触发 TrainingDiagnosticsHook 注册到 HOOKS
from ldmdet.diagnostics import hooks as _diag_hooks  # noqa: F401

# Muon 混合优化器和构造器
from ldmdet.optim.hybrid_optimizer import MuonHybrid
from ldmdet.optim.constructor import MuonHybridConstructor

OPTIMIZERS.register_module(name='MuonHybrid', module=MuonHybrid, force=True)
OPTIM_WRAPPER_CONSTRUCTORS.register_module(
    name='MuonHybridConstructor', module=MuonHybridConstructor, force=True
)

MODELS.register_module(name='PurePyTorchDiffusionDetHead', module=DiffusionDetHead, force=True)
MODELS.register_module(name='PurePyTorchSingleDiffusionDetHead', module=SingleDiffusionDetHead, force=True)
MODELS.register_module(name='PurePyTorchSingleRoIExtractor', module=SingleRoIExtractor, force=True)
MODELS.register_module(name='PurePyTorchDiffusionDetCriterion', module=DiffusionDetCriterion, force=True)
MODELS.register_module(name='PurePyTorchDiffusionDetMatcher', module=DiffusionDetMatcher, force=True)
MODELS.register_module(name='PurePyTorchFocalLoss', module=FocalLoss, force=True)
MODELS.register_module(name='PurePyTorchSeesawLoss', module=SeesawLoss, force=True)
MODELS.register_module(name='PurePyTorchL1Loss', module=L1Loss, force=True)
MODELS.register_module(name='PurePyTorchGIoULoss', module=GIoULoss, force=True)
MODELS.register_module(name='PurePyTorchNormalizedLinear', module=NormalizedLinear, force=True)
MODELS.register_module(name='PurePyTorchFocalLossCost', module=FocalLossCost, force=True)
MODELS.register_module(name='PurePyTorchBBoxL1Cost', module=BBoxL1Cost, force=True)
MODELS.register_module(name='PurePyTorchIoUCost', module=IoUCost, force=True)

from experiments.mmdet_bridge.transforms import CLAHE, SmallObjectCopyPaste  # noqa: F401

# SetDiff 注册 — 仅注册需要从 config 构建的核心模块
from setdiff.models.set_head import JointDiffusionHead  # noqa: E402

MODELS.register_module(name='SetDiffJointDiffusionHead', module=JointDiffusionHead, force=True)

register_all = lambda: None
