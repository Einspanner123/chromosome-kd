"""Bounding box 坐标转换工具"""

from typing import List

import torch
from torch import Tensor


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """(x1, y1, x2, y2) → (cx, cy, w, h)"""
    xy1, xy2 = bbox.split((2, 2), dim=-1)
    return torch.cat([(xy1 + xy2) / 2, xy2 - xy1], dim=-1)


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2)"""
    cxy, wh = bbox.split((2, 2), dim=-1)
    return torch.cat([cxy - wh / 2, cxy + wh / 2], dim=-1)


def bbox2roi(bbox_list: List[Tensor]) -> Tensor:
    """将 batch bbox 列表转换为 RoI 格式。

    Returns:
        [N, 5] tensor: [batch_idx, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        img_idxs = bboxes.new_full((bboxes.shape[0], 1), img_id)
        rois = torch.cat([img_idxs, bboxes], dim=-1)
        rois_list.append(rois)
    return torch.cat(rois_list, dim=0)
