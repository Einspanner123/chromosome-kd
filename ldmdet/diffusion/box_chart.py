"""Smooth coordinates for valid axis-aligned bounding boxes.

A normalized box is represented by the horizontal and vertical gap
compositions ``(left, width, right)`` and ``(top, height, bottom)``.  An
isometric log-ratio (ILR) transform maps the interiors of the two 2-simplexes
to R^4.  Decoding uses softmax, so every finite chart point produces a box
strictly inside the image with positive width and height.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ValidBoxChart(nn.Module):
    """ILR chart and optional moment calibration for normalized ``xyxy`` boxes.

    Args:
        eps: Positive gap floor used only while encoding boxes on the boundary.
        mean: Optional chart-space mean, shape ``(4,)``.
        covariance: Optional chart-space covariance, shape ``(4, 4)``.
        eigenvalue_floor: Minimum eigenvalue used to construct whitening maps.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        mean: Tensor | None = None,
        covariance: Tensor | None = None,
        eigenvalue_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps}')
        self.eps = float(eps)
        self.eigenvalue_floor = float(eigenvalue_floor)

        helmert = torch.tensor(
            [
                [2.0 ** -0.5, -(2.0 ** -0.5), 0.0],
                [6.0 ** -0.5, 6.0 ** -0.5, -2.0 * (6.0 ** -0.5)],
            ],
            dtype=torch.float64,
        )
        self.register_buffer('helmert', helmert)
        self.register_buffer('mean', torch.zeros(4, dtype=torch.float64))
        self.register_buffer('whitening', torch.eye(4, dtype=torch.float64))
        self.register_buffer('coloring', torch.eye(4, dtype=torch.float64))

        if (mean is None) != (covariance is None):
            raise ValueError('mean and covariance must be provided together')
        if mean is not None:
            self.set_moments(mean, covariance)

    def _basis(self, reference: Tensor) -> Tensor:
        return self.helmert.to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _check_last_dim(tensor: Tensor) -> None:
        if tensor.shape[-1] != 4:
            raise ValueError(f'expected last dimension 4, got {tensor.shape}')

    def _xyxy_to_gaps(self, boxes: Tensor) -> tuple[Tensor, Tensor]:
        self._check_last_dim(boxes)
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        gaps_x = torch.stack((x1, x2 - x1, 1.0 - x2), dim=-1)
        gaps_y = torch.stack((y1, y2 - y1, 1.0 - y2), dim=-1)
        # Boundary boxes have zero gaps.  Flooring followed by closure is the
        # standard stable extension of log-ratio coordinates to such boxes.
        gaps_x = gaps_x.clamp_min(self.eps)
        gaps_y = gaps_y.clamp_min(self.eps)
        gaps_x = gaps_x / gaps_x.sum(dim=-1, keepdim=True)
        gaps_y = gaps_y / gaps_y.sum(dim=-1, keepdim=True)
        return gaps_x, gaps_y

    def to_chart(self, boxes: Tensor) -> Tensor:
        """Map normalized ``xyxy`` boxes to uncalibrated ILR coordinates."""
        gaps_x, gaps_y = self._xyxy_to_gaps(boxes)
        basis = self._basis(boxes)
        ux = torch.matmul(gaps_x.log(), basis.transpose(0, 1))
        uy = torch.matmul(gaps_y.log(), basis.transpose(0, 1))
        return torch.cat((ux, uy), dim=-1)

    def from_chart(self, chart: Tensor) -> Tensor:
        """Decode ILR coordinates to strictly valid normalized ``xyxy`` boxes."""
        self._check_last_dim(chart)
        basis = self._basis(chart)
        ux, uy = chart[..., :2], chart[..., 2:]
        # Softmax is positive over the reals, but a very small component can
        # underflow or disappear when adjacent coordinates are subtracted in
        # finite precision.  A dtype-aware interior floor keeps the numerical
        # decoder on the open box manifold as well.
        gap_floor = max(self.eps, 4.0 * torch.finfo(chart.dtype).eps)
        if 3.0 * gap_floor >= 1.0:
            raise ValueError(f'gap floor is too large for dtype {chart.dtype}')
        gaps_x = torch.softmax(torch.matmul(ux, basis), dim=-1)
        gaps_y = torch.softmax(torch.matmul(uy, basis), dim=-1)
        gaps_x = gaps_x * (1.0 - 3.0 * gap_floor) + gap_floor
        gaps_y = gaps_y * (1.0 - 3.0 * gap_floor) + gap_floor
        x1 = gaps_x[..., 0]
        # Use the complementary right/bottom gaps instead of a two-term sum.
        # The expressions are identical in exact arithmetic, while this form
        # preserves the upper image bound under floating-point roundoff.
        x2 = 1.0 - gaps_x[..., 2]
        y1 = gaps_y[..., 0]
        y2 = 1.0 - gaps_y[..., 2]
        return torch.stack((x1, y1, x2, y2), dim=-1)

    def encode(self, boxes: Tensor) -> Tensor:
        """Encode boxes and whiten their chart coordinates."""
        chart = self.to_chart(boxes)
        mean = self.mean.to(device=chart.device, dtype=chart.dtype)
        whitening = self.whitening.to(device=chart.device, dtype=chart.dtype)
        return torch.matmul(chart - mean, whitening.transpose(0, 1))

    def decode(self, latent: Tensor) -> Tensor:
        """Color latent coordinates and decode them to valid boxes."""
        self._check_last_dim(latent)
        mean = self.mean.to(device=latent.device, dtype=latent.dtype)
        coloring = self.coloring.to(device=latent.device, dtype=latent.dtype)
        chart = torch.matmul(latent, coloring.transpose(0, 1)) + mean
        return self.from_chart(chart)

    @torch.no_grad()
    def set_moments(self, mean: Tensor, covariance: Tensor) -> None:
        """Set calibration from chart-space population moments."""
        mean = torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device)
        covariance = torch.as_tensor(
            covariance, dtype=self.mean.dtype, device=self.mean.device)
        if mean.shape != (4,) or covariance.shape != (4, 4):
            raise ValueError(
                f'expected mean (4,) and covariance (4,4), got '
                f'{mean.shape} and {covariance.shape}')
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp_min(self.eigenvalue_floor)
        whitening = (
            eigenvectors @ torch.diag(eigenvalues.rsqrt()) @
            eigenvectors.transpose(0, 1))
        coloring = (
            eigenvectors @ torch.diag(eigenvalues.sqrt()) @
            eigenvectors.transpose(0, 1))
        self.mean.copy_(mean)
        self.whitening.copy_(whitening)
        self.coloring.copy_(coloring)

    @torch.no_grad()
    def fit(self, boxes: Tensor) -> tuple[Tensor, Tensor]:
        """Estimate population moments from normalized boxes and store them."""
        chart = self.to_chart(boxes).to(dtype=self.mean.dtype)
        flat = chart.reshape(-1, 4)
        if flat.shape[0] < 2:
            raise ValueError('at least two boxes are required to fit moments')
        mean = flat.mean(dim=0)
        centered = flat - mean
        covariance = centered.transpose(0, 1) @ centered / flat.shape[0]
        self.set_moments(mean, covariance)
        return mean.clone(), covariance.clone()
