"""Muon 优化器 — MomentUm Orthogonalized by Newton-Schulz

方向六优化器改进实验: 对 bbox_head 的 2D 矩阵参数使用 Newton-Schulz 正交化动量。

核心思想:
    1. 累积梯度动量 (类似 SGD momentum)
    2. 对 2D 矩阵参数, 用 Newton-Schulz 迭代正交化动量
    3. 正交化后的更新方向谱归一化, 提高训练稳定性
    4. 对 1D 参数 (bias/LayerNorm), 退化为标准 momentum SGD

参考: Jordan et al. 2024 "Muon: Momentum + Orthogonalization"
      系数 (3.4445, -4.7750, 2.0315) 来自 Keller Jordan 的优化
"""

import math
from typing import Iterable, List, Optional

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz 迭代计算矩阵的零次幂 (正交极因子)

    通过 5 次迭代逼近 X = U V^T (其中 G = U Σ V^T 是 SVD),
    使 X 的奇异值全部为 1, 保持方向但归一化幅度。

    对非方阵 (m != n), 统一处理为 "高瘦" 形式 (m >= n) 后迭代,
    最后恢复原形状, 保证 X^T X ≈ I (列正交) 或 X X^T ≈ I (行正交)。

    Args:
        G: 输入矩阵 (m, n), m >= n 或 m < n 均可
        steps: Newton-Schulz 迭代步数 (默认 5)

    Returns:
        正交化后的矩阵 (与 G 形状一致)
    """
    assert G.ndim >= 2, f'G must be 2D+, got {G.ndim}D'

    # 系数 (来自 Keller Jordan 的优化, 针对 m >= n 的矩阵)
    a, b, c = (3.4445, -4.7750, 2.0315)

    # 对宽矩阵 (m < n) 转置为高瘦矩阵处理, 保证 NS 迭代收敛
    needs_transpose = G.shape[-2] < G.shape[-1]
    X = G.mT if needs_transpose else G
    X = X.bfloat16() if X.is_cuda else X.float()

    # 归一化: 确保 X 的谱范数 <= 1 (NS 迭代收敛条件)
    # 用 Frobenius norm (谱范数的上界) 归一化, 保证 NS 迭代必定收敛
    # Frobenius norm 略保守但鲁棒; 幂迭代可能低估谱范数导致 NS 发散产生 NaN
    frob_norm = X.norm()
    if frob_norm > 1e-7:
        X = X / frob_norm
    else:
        # 极小矩阵, 直接返回原始 G (避免除零, 保持形状不变)
        return G

    for _ in range(steps):
        A = X @ X.mT  # (m, m)
        B = b * A + c * A @ A
        X = a * X + B @ X

    X = X.to(G.dtype)
    return X.mT if needs_transpose else X


class Muon(Optimizer):
    """Muon 优化器

    对 2D 矩阵参数使用 Newton-Schulz 正交化动量;
    对 1D 参数 (bias, LayerNorm) 退化为 momentum SGD。

    Args:
        params: 参数列表
        lr: 学习率 (默认 0.02, 经验值)
        momentum: 动量系数 (默认 0.95)
        weight_decay: 权重衰减 (默认 0)
        ns_steps: Newton-Schulz 迭代步数 (默认 5)
        nesterov: 是否使用 Nesterov 动量 (默认 True)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f'lr must be positive, got {lr}')
        if momentum < 0 or momentum >= 1:
            raise ValueError(f'momentum must be in [0, 1), got {momentum}')
        if weight_decay < 0:
            raise ValueError(f'weight_decay must be non-negative, got {weight_decay}')
        if ns_steps < 1:
            raise ValueError(f'ns_steps must be >= 1, got {ns_steps}')

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')

                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']

                # Weight decay (decoupled, 类似 AdamW, 独立于梯度更新)
                # 在梯度累积之前应用, 确保 weight_decay 不被正交化影响
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                # 累积动量
                buf.mul_(momentum).add_(grad)

                # Nesterov: 用 buf + momentum * grad 作为更新方向
                update = buf.add(grad, alpha=momentum) if nesterov else buf

                if p.ndim >= 2:
                    # 2D 参数: Newton-Schulz 正交化
                    # 对大矩阵, NS 迭代中间矩阵可能 OOM, 移到 CPU 计算
                    m_dim, n_dim = update.shape[-2], update.shape[-1]
                    max_dim = max(m_dim, n_dim)
                    if max_dim > 4096 and update.is_cuda:
                        update_cpu = update.detach().cpu()
                        update_ortho = zeropower_via_newtonschulz5(update_cpu, steps=ns_steps)
                        update_ortho = update_ortho.to(p.device, dtype=p.dtype)
                    else:
                        update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # RMS-aligned 缩放: 匹配 AdamW 更新幅度
                    rms_scale = max(1.0, 0.2 * math.sqrt(max(m_dim, n_dim) / min(m_dim, n_dim)))
                    update_scaled = update_ortho * rms_scale
                else:
                    # 1D 参数: 退化为 momentum SGD
                    update_scaled = update

                # 参数更新
                p.add_(update_scaled, alpha=-lr)

        return loss
