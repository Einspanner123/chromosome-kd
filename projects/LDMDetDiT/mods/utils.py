from typing import List

import torch
from torch import Tensor


def bbox2roi(bbox_list: List[Tensor]) -> Tensor:
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (List[Union[Tensor, :obj:`BaseBoxes`]): a list of bboxes
            corresponding to a batch of images.

    Returns:
        Tensor: shape (n, box_dim + 1),
        [bs, x1, y1, x2, y2] of where on last dim.
    """
    rois_list = []
    for id, bboxes in enumerate(bbox_list):
        img_idxs = bboxes.new_full((bboxes.shape[0], 1), id)
        rois = torch.cat([img_idxs, bboxes], dim=-1)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    xy1, xy2 = bbox.split((2, 2), dim=-1)
    return torch.cat([(xy1 + xy2) / 2, xy2 - xy1], dim=-1)


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cxy, wh = bbox.split((2, 2), dim=-1)
    return torch.cat([cxy - wh / 2, cxy + wh / 2], dim=-1)


def sanitize_bboxes(bboxes: Tensor) -> Tensor:
    """清洗 bbox 坐标中的 NaN/Inf 并 clamp 到 [0,1]"""
    return torch.nan_to_num(bboxes, nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)


def sanitize_features(features: Tensor) -> Tensor:
    """清洗特征/Token 中的 NaN/Inf，替换为 0"""
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == '__main__':
    a = torch.randn((3, 3))
    print(a.shape)
