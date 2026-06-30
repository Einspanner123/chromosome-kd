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

        # 诊断缓存: couple() 调用后可被 CouplingDiagnostics 读取
        self.last_transport = None  # [M, N] 传输矩阵
        self.last_cost = None  # [M, N] cost 矩阵
        self.last_coupling_mode = 'none'

    @property
    def ot_epsilon(self) -> float:
        """兼容诊断器读取 (legacy OTCoupling 用 ot_epsilon 属性)."""
        return self.epsilon

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
        """将噪声提案与 GT 框通过 OT 配对.

        传输矩阵 T shape [M, N] (M=num_gt, N=num_proposals).
        列归一化后, 每个 proposal 按概率选择一个 GT,
        返回 [N, 4] 的 x_start 和 [N] 的 matched_idx.

        Args:
            noise: [N, 4] 噪声提案
            gt_diffusion: [M, 4] GT 框 (扩散空间)
            gt_labels: [M] GT 类别
            device: 设备

        Returns:
            x_start: [N, 4] 每个 proposal 对应的 GT 框
            matched_idx: [N] 每个 proposal 对应的 GT 索引
        """
        N = noise.shape[0]
        M = gt_diffusion.shape[0]
        if M == 0:
            return noise, torch.zeros(N, dtype=torch.long, device=device)

        # 代价矩阵: [M, N]
        cost = torch.cdist(gt_diffusion, noise, p=2)
        # 传输矩阵: [M, N]
        transport = sinkhorn_transport(
            cost, epsilon=self.epsilon, num_iters=self.num_iters
        )

        # 缓存供 CouplingDiagnostics 使用
        self.last_cost = cost.detach()
        self.last_transport = transport.detach()
        self.last_coupling_mode = self._ot.coupling_mode  # 'argmax' 或 'multinomial'

        # 列归一化: 每个 proposal (列) 从 M 个 GT 中选一个
        col_sums = transport.sum(dim=0, keepdim=True).clamp_min(1e-10)  # [1, N]
        transport_col = transport / col_sums  # [M, N], 每列和为 1

        if self._ot.coupling_mode == 'argmax':
            # 每个 proposal 选概率最大的 GT
            matched_idx = transport_col.argmax(dim=0)  # [N]
        else:  # multinomial
            # 每个 proposal 按概率采样一个 GT
            matched_idx = torch.multinomial(
                transport_col.t().clamp_min(1e-10), 1
            ).squeeze(-1)  # [N]

        # 扩展 GT 到 proposal 数量
        x_start = gt_diffusion[matched_idx]  # [N, 4]
        return x_start, matched_idx

    def compute_coupling_cost(self, x_start: Tensor, x_noise: Tensor) -> Tensor:
        """计算代价矩阵 (供诊断使用)."""
        return self._ot.compute_coupling_cost(x_start, x_noise)
