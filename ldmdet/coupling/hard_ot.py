"""硬 OT 耦合 — 最近邻确定性匹配"""

import torch
from torch import Tensor

from ldmdet.coupling.base import CouplingStrategy, register_coupling


@register_coupling('hard_ot')
class HardOTCoupling(CouplingStrategy):
    """确定性最近邻 OT: argmin L2 距离。传输效率最高，多样性为零。"""

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
        idx = cost.argmin(dim=1)
        return gt_diffusion[idx], idx
