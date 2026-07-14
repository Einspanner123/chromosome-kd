"""SetDiffusionDetector — top-level detector.

Combines backbone + FPN + JointDiffusionHead. For now, this is a simple
wrapper. In practice, backbone and FPN will come from mmdet/torchvision
(ResNet, Swin, etc.).
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from setdiff.models.set_head import JointDiffusionHead


class SetDiffusionDetector(nn.Module):
    """SetDiff Detector: backbone + FPN + JointDiffusionHead.

    For now, this is a simple wrapper. In practice, backbone and FPN will
    come from mmdet/torchvision (ResNet, Swin, etc.).
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module],
        head: JointDiffusionHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        images: Tensor,
        gt_boxes: Optional[List[Tensor]] = None,
        gt_labels: Optional[List[Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Args:
            images: [B, C, H, W] input images.
            gt_boxes: list of [M_i, 4] GT boxes (cxcywh, diffusion space),
                None for inference.
            gt_labels: list of [M_i] GT labels, None for inference.

        Returns:
            Training: dict with loss terms.
            Inference: dict with 'pred_logits' and 'pred_boxes'.
        """
        features = self.backbone(images)
        if self.neck is not None:
            features = self.neck(features)
        # Flatten multi-scale features: tuple of [B, C, H_i, W_i] →
        # [B, sum(H_i*W_i), C]
        flat_features = self._flatten_features(features)
        return self.head(flat_features, gt_boxes, gt_labels)

    @staticmethod
    def _flatten_features(
        features: Union[Tensor, tuple, list],
    ) -> Tensor:
        """Flatten multi-scale features to [B, HW, C].

        Args:
            features: tuple/list of [B, C, H_i, W_i] tensors, or a single
                [B, C, H, W] tensor.

        Returns:
            flat: [B, sum(H_i*W_i), C]
        """
        if isinstance(features, Tensor):
            features = [features]
        flat_list = []
        for feat in features:
            B, C, H, W = feat.shape
            # [B, C, H, W] → [B, H*W, C]
            flat_list.append(feat.flatten(2).transpose(1, 2))
        return torch.cat(flat_list, dim=1)
