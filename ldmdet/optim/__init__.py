"""优化器改进 — 方向六

Muon 优化器、混合 OptimWrapper 和自定义构造器,
用于改进训练稳定性。
"""

from ldmdet.optim.hybrid_wrapper import (  # noqa: F401
    MuonHybridOptimWrapper,
    classify_parameters,
)
from ldmdet.optim.muon import Muon, zeropower_via_newtonschulz5  # noqa: F401
from ldmdet.optim.hybrid_optimizer import MuonHybrid  # noqa: F401
from ldmdet.optim.constructor import MuonHybridConstructor  # noqa: F401
