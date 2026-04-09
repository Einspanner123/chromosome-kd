from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from torch import Tensor


@dataclass
class InstanceData:
    """单个图像的真值数据 (纯 PyTorch 结构)"""

    bboxes: Tensor  # [M, 4] normalized xyxy
    labels: Tensor  # [M]
    img_shape: Tuple[int, int]  # (h, w)


@dataclass
class ModelOutput:
    """模型预测输出 (纯 PyTorch 结构)"""

    pred_logits: Tensor  # [B, N, C]
    pred_boxes: Tensor  # [B, N, 4] normalized xyxy
    pred_objectness: Optional[Tensor] = None  # [B, N, 1] objectness logits
    pred_count: Optional[Tensor] = None  # [B, 1] 全局计数预测
    aux_outputs: Optional[List["ModelOutput"]] = None


@dataclass
class ImageMeta:
    """图像元信息 (纯 PyTorch 结构)"""

    img_shape: Tuple[int, int]  # (h, w)
    ori_shape: Optional[Tuple[int, int]] = None
    scale_factor: Optional[Union[List[float], Tensor]] = None


@dataclass
class DetectionResult:
    """模型推理结果 (纯 PyTorch 结构)"""

    bboxes: Tensor  # [K, 4]
    scores: Tensor  # [K]
    labels: Tensor  # [K]
