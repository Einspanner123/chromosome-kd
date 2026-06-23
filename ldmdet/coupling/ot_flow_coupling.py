"""OT Flow Matching 耦合策略适配器 — 方向四

将 OTFlowMatching 包装为 CouplingStrategy, 使其可通过 coupling 参数配置.
同时暴露 epsilon/num_iters 属性供诊断使用.

若方向四废弃, 删除本文件 + tests/unit/test_nonlinear_trajectory.py 即可回滚.
"""

import torch
from torch import Tensor

from ldmdet.coupling._sinkhorn_ops import sinkhorn_transport
from ldmdet.coupling.base import CouplingStrategy, register_coupling
from ldmdet.diffusion.ot_flow_matching import OTFlowMatching


@register_coupling('ot_flow')
class OTFlowCoupling(CouplingStrategy):
    """OT Flow Matching 耦合策略.

    使用 Sinkhorn OT 计算传输矩阵, 支持 argmax 或 multinomial 解码.

    Args:
        epsilon: Sinkhorn 熵正则化
        num_iters: Sinkhorn 迭代次数
        coupling_mode: 'argmax' (确定性) 或 'multinomial' (随机采样)
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        num_iters: int = 10,
        coupling_mode: str = 'argmax',
    ):
        super().__init__()
        self._ot = OTFlowMatching(
            epsilon=epsilon,
            num_iters=num_iters,
            coupling_mode=coupling_mode,
        )
        self.epsilon = epsilon
        self.num_iters = num_iters

    @property
    def ot_module(self) -> OTFlowMatching:
        """暴露内部 OTFlowMatching 供诊断使用."""
        return self._ot

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

        # 复用 OTFlowMatching.compute_ot_coupling
        # 注意: OTFlowMatching.compute_ot_coupling 接收 (x_start, x_noise)
        # 这里 noise 是 x_noise, gt_diffusion 是 x_start
        x_start_c, x_noise_c = self._ot.compute_ot_coupling(gt_diffusion, noise)
        # 计算匹配索引 (用于诊断)
        cost = torch.cdist(gt_diffusion, noise, p=2)
        transport = sinkhorn_transport(
            cost, epsilon=self.epsilon, num_iters=self.num_iters
        )
        idx = transport.argmax(dim=1) if self._ot.coupling_mode == 'argmax' else \
            torch.multinomial(transport.clamp_min(1e-10), 1).squeeze(-1)
        return x_start_c, idx

    def compute_coupling_cost(self, x_start: Tensor, x_noise: Tensor) -> Tensor:
        """计算代价矩阵 (供诊断使用)."""
        return self._ot.compute_coupling_cost(x_start, x_noise)
