"""随机耦合 — 每个噪声提案随机匹配一个 GT 框"""

import torch
from torch import Tensor

from ldmdet.coupling.base import CouplingStrategy, register_coupling


@register_coupling('random')
class RandomCoupling(CouplingStrategy):
    """均匀随机耦合。最大化训练信号多样性，无空间结构。"""

    def couple(
        self,
        noise: Tensor,
        gt_diffusion: Tensor,
        gt_labels: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        N = noise.shape[0]
        M = gt_diffusion.shape[0]
        if M == 0:
            return noise, torch.zeros(N, dtype=torch.long, device=device)
        idx = torch.randint(0, M, (N,), device=device)
        return gt_diffusion[idx], idx
