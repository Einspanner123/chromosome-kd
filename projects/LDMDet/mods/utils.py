from typing import List, Union

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


def apply_box_renewal(
    x_raw: Tensor,
    cls_logits: Tensor,
    min_keep: int,
    score_thr: float,
) -> Tensor:
    """向量化的框更新策略，消除 Python 循环"""
    num_proposals = x_raw.shape[1]

    # 1. 计算每个框的最大得分
    scores = torch.sigmoid(cls_logits).max(-1)[0]  # [bs, num_proposals]

    # 2. 确定哪些框需要保留
    # 策略：分数高于阈值 OR 属于每张图的前 min_keep 名
    topk_mask = torch.zeros_like(scores, dtype=torch.bool)
    k = min(min_keep, num_proposals)
    topk_idx = torch.topk(scores, k, dim=1).indices
    topk_mask.scatter_(1, topk_idx, True)

    keep = (scores > score_thr) | topk_mask  # [bs, num_proposals]

    # 3. 向量化更新：对于不保留的框，替换为新的随机噪声
    noise = torch.randn_like(x_raw)
    x_raw_new = torch.where(keep.unsqueeze(-1), x_raw, noise)

    return x_raw_new


def xyxy_to_raw(
    bboxes: Tensor,
    img_metas: Union[List, Tensor],
    snr_scale: float,
) -> Tensor:
    """向量化的坐标转换：图像空间 xyxy -> 扩散空间 raw (cxcywh)"""
    # 1. 提取批量图像尺寸并构造 scale Tensor
    # img_shape: (h, w) -> scale: (w, h, w, h)
    if isinstance(img_metas, Tensor):
        # img_metas: [bs, 6] (h, w, s1, s2, s3, s4)
        img_shapes = img_metas[:, :2]
    else:
        img_shapes = torch.stack(
            [bboxes.new_tensor(m.img_shape[:2]) for m in img_metas]
        )
    scales = img_shapes[:, [1, 0, 1, 0]].unsqueeze(1)  # [bs, 1, 4]

    # 2. 归一化并转换格式
    x0 = bboxes / scales
    x0 = bbox_xyxy_to_cxcywh(x0)

    # 3. 映射到扩散空间 [-snr, snr]
    x0 = (x0 * 2 - 1) * snr_scale
    return x0


def raw_to_xyxy(
    raw_bboxes: Tensor,
    img_metas: Union[List, Tensor],
    snr_scale: float,
) -> Tensor:
    """向量化的坐标转换：扩散空间 raw -> 图像空间 xyxy"""
    # 1. 扩散空间 [-snr, snr] -> 归一化空间 [0, 1]
    bboxes = ((raw_bboxes.clamp(-snr_scale, snr_scale) / snr_scale) + 1) / 2
    # cxcywh -> xyxy
    bboxes = bbox_cxcywh_to_xyxy(bboxes)

    # 2. 提取批量图像尺寸并映射
    if isinstance(img_metas, Tensor):
        # img_metas: [bs, 6] (h, w, s1, s2, s3, s4)
        img_shapes = img_metas[:, :2]
    else:
        img_shapes = torch.stack(
            [bboxes.new_tensor(m.img_shape[:2]) for m in img_metas]
        )
    scales = img_shapes[:, [1, 0, 1, 0]].unsqueeze(1)  # [bs, 1, 4]
    bboxes = bboxes * scales

    return bboxes
