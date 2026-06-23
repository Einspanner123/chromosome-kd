"""耦合策略基类 + 工厂函数"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor


class CouplingStrategy(ABC, nn.Module):
    """耦合策略抽象基类。

    每种策略实现 couple() 方法，将噪声提案与 GT 框配对。
    """

    @abstractmethod
    def couple(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """将噪声提案与 GT 框配对。

        Args:
            noise: [N, 4] 噪声提案 (扩散空间)
            gt_diffusion: [M, 4] GT 框 (扩散空间)
            gt_labels: [M] GT 类别索引
            device: 计算设备

        Returns:
            x_start: [N, 4] 配对的 GT 框
            matched_gt_idx: [N] 匹配的 GT 索引
        """
        ...


# 注册表：耦合策略名 → 类 (延迟加载)
_COUPLING_REGISTRY: Dict[str, Type[CouplingStrategy]] = {}


def register_coupling(name: str):
    """注册耦合策略的装饰器"""

    def decorator(cls: Type[CouplingStrategy]) -> Type[CouplingStrategy]:
        _COUPLING_REGISTRY[name] = cls
        return cls

    return decorator


def build_coupling(name: str, **kwargs) -> CouplingStrategy:
    """工厂函数：按名称构建耦合策略。

    Examples:
        >>> build_coupling('random')
        >>> build_coupling('hard_ot')
        >>> build_coupling('sinkhorn_stochastic', epsilon=5.0, num_iters=20)
        >>> build_coupling('ghss', epsilon=5.0)
    """
    if name not in _COUPLING_REGISTRY:
        # 触发延迟导入
        _import_all_strategies()
    if name not in _COUPLING_REGISTRY:
        raise ValueError(
            f"Unknown coupling strategy: {name}. "
            f"Known: {list(_COUPLING_REGISTRY.keys())}"
        )
    return _COUPLING_REGISTRY[name](**kwargs)


def _import_all_strategies():
    """延迟导入所有耦合策略，填充注册表"""
    try:
        from ldmdet.coupling.random import RandomCoupling  # noqa: F401
        from ldmdet.coupling.hard_ot import HardOTCoupling  # noqa: F401
        from ldmdet.coupling.sinkhorn_argmax import SinkhornArgmaxCoupling  # noqa: F401
        from ldmdet.coupling.sinkhorn_stochastic import SinkhornStochasticCoupling  # noqa: F401
        from ldmdet.coupling.ghss import GHSSCoupling  # noqa: F401
        from ldmdet.coupling.unbalanced_ghss import UnbalancedGHSSCoupling  # noqa: F401
        from ldmdet.coupling.ot_flow_coupling import OTFlowCoupling  # noqa: F401
    except ImportError:
        pass
