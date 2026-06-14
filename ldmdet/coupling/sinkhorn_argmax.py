"""Sinkhorn OT + argmax 解码"""

import torch
from torch import Tensor

from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport
from ldmdet.coupling.base import CouplingStrategy, register_coupling


@register_coupling('sinkhorn_argmax')
class SinkhornArgmaxCoupling(CouplingStrategy):
    """Sinkhorn OT 计算传输矩阵，argmax 解码每行。

    注意：argmax 破坏了 ε 的多样性控制 —— 见论文 CAM 命题。
    """

    def __init__(self, epsilon: float = 1.0, num_iters: int = 20):
        super().__init__()
        self.epsilon = epsilon
        self.num_iters = num_iters

    def couple(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        N = noise.shape[0]
        if gt_diffusion.shape[0] == 0:
            return noise, torch.zeros(N, dtype=torch.long, device=device)

        cost = torch.cdist(noise, gt_diffusion, p=2)
        transport = sinkhorn_transport(
            cost, epsilon=self.epsilon, num_iters=self.num_iters
        )
        idx = transport.argmax(dim=1)
        return gt_diffusion[idx], idx
