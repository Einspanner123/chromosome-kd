"""纯 PyTorch 数据结构 (dataclass)。

所有结构仅依赖 torch.Tensor，无 mmdet/mmengine 引用。
"""

from ldmdet.data.structures import DetectionResult, ImageMeta, InstanceData, ModelOutput  # noqa: F401
