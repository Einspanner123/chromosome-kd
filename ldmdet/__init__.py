"""LDMDet — 纯 PyTorch 扩散目标检测库

零 mmdet/mmengine 依赖。可独立安装和测试。

Usage:
    from ldmdet.core import DiffusionDetHead
    from ldmdet.coupling import build_coupling
    from ldmdet.diffusion.sampling import DiffusionSampler
"""

__version__ = '0.1.0'

from ldmdet.core.head import DiffusionDetHead  # noqa: F401
