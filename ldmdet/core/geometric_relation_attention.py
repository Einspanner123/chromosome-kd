"""Overlap-conditional geometric relation bias for proposal attention."""

import torch
import torch.nn as nn


class OverlapConditionalGeometryBias(nn.Module):
    """Predict a per-head attention bias from relative box geometry.

    The descriptor is invariant to joint translation and isotropic scaling.
    A smooth proximity envelope restricts the intervention to spatially close
    proposals, where the Phase-0 diagnostic found missing conditional
    information.  The final projection is zero initialized, making the module
    an exact identity intervention at initialization.
    """

    def __init__(self, num_heads, hidden_channels=32, proximity_radius=3.0):
        super().__init__()
        if num_heads <= 0 or hidden_channels <= 0:
            raise ValueError('num_heads and hidden_channels must be positive')
        if proximity_radius <= 0:
            raise ValueError('proximity_radius must be positive')
        self.num_heads = int(num_heads)
        self.proximity_radius = float(proximity_radius)
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, boxes):
        """Return additive attention bias with shape [B, H, N, N]."""
        wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=1e-4)
        centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
        delta = centers[:, :, None, :] - centers[:, None, :, :]
        scale = torch.sqrt(
            wh[:, :, None, :] * wh[:, None, :, :]).clamp(min=1e-4)
        normalized_delta = delta / scale
        log_ratio = torch.log(
            wh[:, :, None, :] / wh[:, None, :, :]).clamp(-5.0, 5.0)

        lt = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
        rb = torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
        intersection_wh = (rb - lt).clamp(min=0)
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
        area = wh[..., 0] * wh[..., 1]
        union = area[:, :, None] + area[:, None, :] - intersection
        iou = intersection / union.clamp(min=1e-6)
        distance = torch.sqrt(
            normalized_delta.square().sum(-1) + 1e-8)
        log_area_ratio = torch.log(
            area[:, :, None] / area[:, None, :]).clamp(-10.0, 10.0)
        descriptor = torch.cat((
            normalized_delta,
            log_ratio,
            iou.unsqueeze(-1),
            distance.unsqueeze(-1),
            log_area_ratio.unsqueeze(-1),
        ), dim=-1)
        bias = self.mlp(descriptor)
        envelope = torch.exp(
            -0.5 * (distance / self.proximity_radius).square())
        bias = bias * envelope.unsqueeze(-1)
        return bias.permute(0, 3, 1, 2).contiguous()
