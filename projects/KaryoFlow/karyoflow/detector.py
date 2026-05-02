"""
检测器抽象接口 + 适配器

KaryoFlow 通过此接口获取染色体检测结果，不绑定任何具体检测器实现。

适配器:
- GTDetector: 训练时用 GT 标注 (消除检测噪声)
- COCOResultDetector: 读取离线检测结果 JSON (任意检测器输出)
- CropOnlyDetector: 直接接收预裁剪 crops (最快)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DetectionResult:
    """检测结果数据结构"""

    bboxes: Tensor     # (N, 4) xyxy 格式, 图像坐标
    labels: Tensor     # (N,) 类别 ID (1-based, 与 COCO 一致)
    scores: Tensor     # (N,) 置信度 [0, 1]
    crops: Tensor      # (N, 3, crop_H, crop_W) 裁剪并 resize 的染色体图像
    num_detections: int  # 有效检测数 (= N)


class BaseChromosomeDetector(ABC):
    """检测器抽象接口"""

    @abstractmethod
    def detect(self, image: Tensor, **kwargs) -> DetectionResult:
        """
        Args:
            image: (3, H, W) 单张图像

        Returns:
            DetectionResult
        """
        ...


def crop_and_resize(
    image: Tensor, bboxes: Tensor, crop_size: int = 64
) -> Tensor:
    """从图像中裁剪 bbox 区域并 resize 到统一大小

    Args:
        image: (3, H, W)
        bboxes: (N, 4) xyxy 格式
        crop_size: 输出尺寸

    Returns:
        crops: (N, 3, crop_size, crop_size)
    """
    N = bboxes.shape[0]
    if N == 0:
        return torch.zeros(0, 3, crop_size, crop_size, device=image.device)

    _, H, W = image.shape
    crops = []

    for i in range(N):
        x1, y1, x2, y2 = bboxes[i].int().tolist()
        # 裁剪边界保护
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, max(x1 + 1, x2))
        y2 = min(H, max(y1 + 1, y2))

        crop = image[:, y1:y2, x1:x2]  # (3, h, w)
        # resize 到统一大小
        crop = F.interpolate(
            crop.unsqueeze(0), size=(crop_size, crop_size), mode="bilinear", align_corners=False
        ).squeeze(0)  # (3, crop_size, crop_size)
        crops.append(crop)

    return torch.stack(crops)  # (N, 3, crop_size, crop_size)


class GTDetector(BaseChromosomeDetector):
    """使用 GT 标注作为检测结果

    训练阶段使用，消除检测噪声，专注验证排列模块。
    直接从数据集提供的 bbox + label 裁剪图像。
    """

    def __init__(self, crop_size: int = 64):
        self.crop_size = crop_size

    def detect(
        self,
        image: Tensor,
        gt_bboxes: Optional[Tensor] = None,
        gt_labels: Optional[Tensor] = None,
        **kwargs,
    ) -> DetectionResult:
        """
        Args:
            image: (3, H, W)
            gt_bboxes: (N, 4) xyxy 格式
            gt_labels: (N,) 类别 ID

        Returns:
            DetectionResult with confidence=1.0 for all
        """
        assert gt_bboxes is not None and gt_labels is not None, \
            "GTDetector requires gt_bboxes and gt_labels"

        N = gt_bboxes.shape[0]
        crops = crop_and_resize(image, gt_bboxes, self.crop_size)

        return DetectionResult(
            bboxes=gt_bboxes,
            labels=gt_labels,
            scores=torch.ones(N, device=image.device),
            crops=crops,
            num_detections=N,
        )


class COCOResultDetector(BaseChromosomeDetector):
    """读取离线检测结果 JSON

    先用任意检测器跑一遍数据集，保存为 COCO results format:
    [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]

    然后用此适配器读取结果，裁剪图像。
    """

    def __init__(
        self,
        results_json: str,
        score_threshold: float = 0.3,
        max_detections: int = 50,
        crop_size: int = 64,
    ):
        import json

        self.crop_size = crop_size
        self.score_threshold = score_threshold
        self.max_detections = max_detections

        # 加载检测结果
        with open(results_json, "r") as f:
            results = json.load(f)

        # 按 image_id 分组
        self.results_by_image: Dict[int, List[dict]] = {}
        for r in results:
            img_id = r["image_id"]
            self.results_by_image.setdefault(img_id, []).append(r)

    def detect(
        self,
        image: Tensor,
        image_id: Optional[int] = None,
        **kwargs,
    ) -> DetectionResult:
        """
        Args:
            image: (3, H, W)
            image_id: COCO image_id (用于查找对应的检测结果)

        Returns:
            DetectionResult
        """
        assert image_id is not None, "COCOResultDetector requires image_id"

        device = image.device
        results = self.results_by_image.get(image_id, [])

        # 过滤低置信度
        results = [r for r in results if r["score"] >= self.score_threshold]
        # 按置信度排序，取 top-k
        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[: self.max_detections]

        if len(results) == 0:
            return DetectionResult(
                bboxes=torch.zeros(0, 4, device=device),
                labels=torch.zeros(0, dtype=torch.long, device=device),
                scores=torch.zeros(0, device=device),
                crops=torch.zeros(0, 3, self.crop_size, self.crop_size, device=device),
                num_detections=0,
            )

        # COCO bbox = [x, y, w, h] → xyxy
        bboxes = []
        labels = []
        scores = []
        for r in results:
            x, y, w, h = r["bbox"]
            bboxes.append([x, y, x + w, y + h])
            labels.append(r["category_id"])
            scores.append(r["score"])

        bboxes = torch.tensor(bboxes, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=device)

        crops = crop_and_resize(image, bboxes, self.crop_size)

        return DetectionResult(
            bboxes=bboxes,
            labels=labels,
            scores=scores_t,
            crops=crops,
            num_detections=len(results),
        )


class CropOnlyDetector(BaseChromosomeDetector):
    """最简模式: 直接接收预裁剪的 crops + metadata

    用于预计算特征缓存后的快速训练。
    """

    def detect(
        self,
        image: Tensor = None,
        crops: Optional[Tensor] = None,
        bboxes: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        scores: Optional[Tensor] = None,
        **kwargs,
    ) -> DetectionResult:
        assert crops is not None, "CropOnlyDetector requires crops"
        N = crops.shape[0]
        device = crops.device

        return DetectionResult(
            bboxes=bboxes if bboxes is not None else torch.zeros(N, 4, device=device),
            labels=labels if labels is not None else torch.zeros(N, dtype=torch.long, device=device),
            scores=scores if scores is not None else torch.ones(N, device=device),
            crops=crops,
            num_detections=N,
        )
