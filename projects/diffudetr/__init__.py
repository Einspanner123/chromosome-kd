"""DiffuDETR 复现项目 — DiffuDETR (ICLR 2026) 移植到 mmdet 框架

将 DiffuDETR 的核心扩散逻辑移植到 mmdet (PyTorch 2.1 + cu118),
不依赖 detrex/detectron2.

使用方式:
    custom_imports = dict(
        imports=['projects.diffudetr'],
        allow_failed_imports=False,
    )
    model = dict(type='DiffuDETR', ...)
"""

from .models import (
    DiffuDETRDetector,
    DiffuDETRHead,
    DiffuDETRTransformer,
    DiffusionScheduler,
)

__all__ = [
    'DiffuDETRDetector',
    'DiffuDETRHead',
    'DiffuDETRTransformer',
    'DiffusionScheduler',
]
