"""MuonHybrid — 单一 Optimizer 子类 (mmengine 兼容)

将 Muon (2D 矩阵参数) 和 AdamW (1D 参数/backbone/neck) 集成到一个
torch.optim.Optimizer 子类中, 通过 param_group 的 'use_muon' 标志路由更新。

mmengine 的 OptimWrapper 要求 optimizer 是 torch.optim.Optimizer 子类,
并提供 step/zero_grad/param_groups/defaults/state 接口。本类满足这些要求,
可直接用于 mmengine 训练流程。

参数组配置示例:
    optim_wrapper = dict(
        optimizer=dict(
            type='MuonHybrid',
            lr=5e-5,                 # AdamW 学习率 (Muon 用 muon_lr)
            muon_lr=0.02,            # Muon 学习率
            weight_decay=1e-4,
            momentum=0.95,           # Muon momentum
            betas=(0.9, 0.999),      # AdamW betas
            ns_steps=5,
        ),
        paramwise_cfg=dict(
            custom_keys={
                'backbone': dict(use_muon=False),  # backbone 用 AdamW
                'neck': dict(use_muon=False),      # neck 用 AdamW
            },
        ),
        clip_grad=dict(max_norm=1.0, norm_type=2),
    )

默认规则 (无 paramwise_cfg):
    - 2D 参数 (Linear/Conv2d weight) → Muon
    - 1D 参数 (bias/LayerNorm) → AdamW
"""

import math
from typing import Iterable, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.adamw import adamw  # 复用 PyTorch 的 AdamW 核心算法

from ldmdet.optim.muon import zeropower_via_newtonschulz5


class MuonHybrid(Optimizer):
    """Muon + AdamW 混合优化器 (单一 Optimizer 子类)

    通过 param_group 的 'use_muon' 标志路由:
    - use_muon=True: Muon 更新 (Newton-Schulz 正交化动量)
    - use_muon=False: AdamW 更新 (标准自适应梯度)

    Args:
        params: 参数或参数组列表
        lr: AdamW 学习率 (默认 5e-5)
        muon_lr: Muon 学习率 (默认 0.02)
        weight_decay: 权重衰减 (默认 1e-4, 对 Muon 和 AdamW 都生效)
        momentum: Muon 动量系数 (默认 0.95)
        betas: AdamW 的 beta1, beta2 (默认 (0.9, 0.999))
        eps: AdamW 的 eps (默认 1e-8)
        ns_steps: Newton-Schulz 迭代步数 (默认 5)
        nesterov: Muon 是否使用 Nesterov 动量 (默认 True)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 5e-5,
        muon_lr: float = 0.02,
        weight_decay: float = 1e-4,
        momentum: float = 0.95,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f'lr must be positive, got {lr}')
        if muon_lr <= 0:
            raise ValueError(f'muon_lr must be positive, got {muon_lr}')
        if weight_decay < 0:
            raise ValueError(f'weight_decay must be non-negative, got {weight_decay}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'beta1 must be in [0, 1), got {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'beta2 must be in [0, 1), got {betas[1]}')
        if eps <= 0:
            raise ValueError(f'eps must be positive, got {eps}')
        if momentum < 0 or momentum >= 1:
            raise ValueError(f'momentum must be in [0, 1), got {momentum}')
        if ns_steps < 1:
            raise ValueError(f'ns_steps must be >= 1, got {ns_steps}')

        defaults = dict(
            lr=lr,
            muon_lr=muon_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,
            nesterov=nesterov,
            use_muon=False,  # 默认值, param_group 可覆盖
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步优化: 对每个 param_group 路由到 Muon 或 AdamW 更新"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get('use_muon', False)
            if use_muon:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        """Muon 更新 (Newton-Schulz 正交化动量)

        LR 策略: mmengine 的 LR scheduler 只调整 group['lr'] (AdamW LR),
        不调整 group['muon_lr']. 为使 Muon LR 也跟随 warmup/decay,
        实际 Muon LR = group['lr'] * muon_lr_mult,
        其中 muon_lr_mult = muon_lr / base_lr (初始化时计算).
        """
        # 计算 Muon LR: 跟随 scheduler 的 lr, 乘以固定倍率
        # muon_lr_mult = muon_lr / lr (初始化时的比值)
        base_lr = self.defaults['lr']
        muon_lr_base = self.defaults['muon_lr']
        muon_lr_mult = muon_lr_base / base_lr if base_lr > 0 else 1.0
        lr = group['lr'] * muon_lr_mult

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

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(p)

            buf = state['momentum_buffer']

            # Decoupled weight decay (AdamW 风格)
            if weight_decay > 0:
                p.mul_(1 - lr * weight_decay)

            # 动量累积
            buf.mul_(momentum).add_(grad)

            # Nesterov: 用 buf + momentum * grad 作为更新方向
            update = buf.add(grad, alpha=momentum) if nesterov else buf

            if p.ndim >= 2:
                # 2D+ 参数: Newton-Schulz 正交化
                # 对大矩阵, NS 迭代的中间矩阵 A=X@X.mT 可能很大 (m,m),
                # 在 16GB GPU 上可能 OOM. 将正交化计算移到 CPU.
                m_dim, n_dim = update.shape[-2], update.shape[-1]
                max_dim = max(m_dim, n_dim)
                use_cpu = max_dim > 4096  # 仅极大矩阵在 CPU 上正交化

                if use_cpu:
                    update_cpu = update.detach().cpu()
                    update_ortho = zeropower_via_newtonschulz5(update_cpu, steps=ns_steps)
                    update_ortho = update_ortho.to(p.device, dtype=p.dtype)
                else:
                    update_ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)

                # 安全检查: 确保正交化后形状一致
                if update_ortho.shape != update.shape:
                    raise RuntimeError(
                        f'Muon shape mismatch: p={tuple(p.shape)}, '
                        f'update={tuple(update.shape)}, '
                        f'ortho={tuple(update_ortho.shape)}'
                    )
                rms_scale = max(1.0, 0.2 * math.sqrt(max(m_dim, n_dim) / min(m_dim, n_dim)))
                update_scaled = update_ortho * rms_scale
            else:
                # 1D 参数 (理论不应进入 Muon 组, 但容错): 退化为 momentum SGD
                update_scaled = update

            p.add_(update_scaled, alpha=-lr)

    def _adamw_step(self, group):
        """AdamW 更新 (复用 PyTorch 的 adamw 核心算法)

        为每个参数维护 exp_avg (一阶矩) 和 exp_avg_sq (二阶矩),
        并执行 decoupled weight decay。
        """
        lr = group['lr']
        beta1, beta2 = group['betas']
        weight_decay = group['weight_decay']
        eps = group['eps']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError('AdamW does not support sparse gradients')

            state = self.state[p]

            # 初始化 AdamW 状态
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # 调用 PyTorch 的 adamw 核心函数 (内部处理 bias correction + decay)
            # amsgrad=False 时 max_exp_avg_sqs 必须为空列表 (非 [None])
            adamw(
                [p],
                grads=[grad],
                exp_avgs=[exp_avg],
                exp_avg_sqs=[exp_avg_sq],
                max_exp_avg_sqs=[],
                state_steps=[state['step']],
                amsgrad=False,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                maximize=False,
            )
