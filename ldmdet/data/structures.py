"""纯 PyTorch 数据结构 (dataclass)

这些结构是 ldmdet 库的内外交互格式，与 mmdet 的 DetDataSample / InstanceData 解耦。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from torch import Tensor


@dataclass
class InstanceData:
    """单个图像的真值数据。

    Attributes:
        bboxes: [M, 4] 归一化 xyxy 格式
        labels: [M] 类别索引
        img_shape: (h, w) 图像尺寸
    """

    bboxes: Tensor
    labels: Tensor
    img_shape: Tuple[int, int]


@dataclass
class ModelOutput:
    """模型预测输出。

    Attributes:
        pred_logits: [B, N, C] 分类 logits
        pred_boxes: [B, N, 4] 预测框 (xyxy)
        pred_count: [B, 1] 可选全局计数
        pred_quality: [B, N, 1] 可选 IoU quality logits
        aux_outputs: 深度监督的辅助输出
    """

    pred_logits: Tensor
    pred_boxes: Tensor
    pred_count: Optional[Tensor] = None
    pred_quality: Optional[Tensor] = None
    aux_outputs: Optional[List['ModelOutput']] = None


@dataclass
class ImageMeta:
    """图像元信息。

    Attributes:
        img_shape: (h, w) resize 后的尺寸
        pad_shape: (h, w) padding 后的尺寸
        ori_shape: 原始图像尺寸
        scale_factor: 缩放因子
        img_id: 图像唯一标识
    """

    img_shape: Tuple[int, int]
    pad_shape: Optional[Tuple[int, int]] = None
    ori_shape: Optional[Tuple[int, int]] = None
    scale_factor: Optional[Union[List[float], Tensor]] = None
    img_id: Optional[int] = None


@dataclass
class DetectionResult:
    """模型推理结果 (NMS 后)。

    Attributes:
        bboxes: [K, 4] 检测框 (xyxy 原始坐标)
        scores: [K] 置信度
        labels: [K] 类别索引
    """

    bboxes: Tensor
    scores: Tensor
    labels: Tensor
