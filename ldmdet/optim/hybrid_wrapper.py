"""MuonHybridOptimWrapper — 混合优化器 wrapper

方向六优化器改进: 对不同参数组使用不同优化器
- bbox_head 的 2D 矩阵参数 (transformer attention, FFN, dynamic conv) → Muon
- backbone/neck (ResNet/FPN conv) → AdamW
- 偏置/LayerNorm (1D) → AdamW

兼容 mmengine OptimWrapper 接口 (step, zero_grad, clip_grad, state_dict)。
"""

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.optim import Optimizer


def classify_parameters(
    model: torch.nn.Module,
    muon_patterns: Optional[List[str]] = None,
    adam_patterns: Optional[List[str]] = None,
    name_resolver: Optional[Callable] = None,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """分类模型参数为 Muon 组和 AdamW 组

    分类规则 (优先级从高到低):
    1. 若参数名匹配 adam_patterns → AdamW
    2. 若参数名匹配 muon_patterns → Muon
    3. 默认: 2D 参数 → Muon, 1D 参数 → AdamW

    Args:
        model: 待分类的模型
        muon_patterns: 强制使用 Muon 的参数名子串列表 (如 ['head_series', 'time_mlp'])
        adam_patterns: 强制使用 AdamW 的参数名子串列表 (如 ['backbone', 'neck'])
        name_resolver: 自定义参数名解析函数 (module, param) -> str
                       默认使用参数名

    Returns:
        (muon_params, adam_params): 两个参数列表
    """
    muon_params: List[torch.nn.Parameter] = []
    adam_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        # 决定参数名 (允许自定义)
        resolved_name = name
        if name_resolver is not None:
            try:
                resolved_name = name_resolver(param, name) or name
            except Exception:
                resolved_name = name

        # 分类
        use_adam = False
        use_muon = False

        if adam_patterns:
            for pat in adam_patterns:
                if pat in resolved_name:
                    use_adam = True
                    break

        if muon_patterns and not use_adam:
            for pat in muon_patterns:
                if pat in resolved_name:
                    use_muon = True
                    break

        # 默认规则: 2D → Muon, 1D → AdamW
        if not use_adam and not use_muon:
            if param.ndim >= 2:
                use_muon = True
            else:
                use_adam = True

        if use_adam:
            adam_params.append(param)
        elif use_muon:
            muon_params.append(param)

    return muon_params, adam_params


class MuonHybridOptimWrapper:
    """混合优化器 wrapper

    管理两个优化器 (Muon + AdamW), 提供 mmengine OptimWrapper 兼容接口。

    Args:
        muon_optimizer: Muon 优化器实例
        adam_optimizer: AdamW 优化器实例
        muon_param_ids: Muon 管理的参数 id 集合 (用于梯度路由)
    """

    def __init__(
        self,
        muon_optimizer: Optimizer,
        adam_optimizer: Optimizer,
        muon_param_ids: Set[int],
    ):
        self.muon_optimizer = muon_optimizer
        self.adam_optimizer = adam_optimizer
        self.muon_param_ids = muon_param_ids

        # mmengine 兼容字段
        self._param_groups = (
            list(self.muon_optimizer.param_groups)
            + list(self.adam_optimizer.param_groups)
        )
        self._max_iters = None

    # ── 核心接口 ──────────────────────────────────────

    def step(self, closure=None):
        """执行一步优化"""
        # 两个优化器共享同一批参数的梯度, 各自路由到自己的参数组
        loss = self.muon_optimizer.step(closure)
        self.adam_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """清空所有参数梯度"""
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adam_optimizer.zero_grad(set_to_none=set_to_none)

    def clip_grad(self, max_norm: float, norm_type: int = 2):
        """对所有参数进行梯度裁剪"""
        all_params = []
        for group in self.muon_optimizer.param_groups:
            all_params.extend(group['params'])
        for group in self.adam_optimizer.param_groups:
            all_params.extend(group['params'])

        # 过滤出有梯度的参数
        params_with_grad = [p for p in all_params if p.grad is not None]
        if not params_with_grad:
            return

        torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=max_norm, norm_type=norm_type)

    # ── state_dict 接口 (mmengine 兼容) ─────────────────

    def state_dict(self) -> Dict:
        """返回可序列化的状态字典"""
        return {
            'muon': self.muon_optimizer.state_dict(),
            'adam': self.adam_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        """加载状态字典"""
        if 'muon' in state_dict:
            self.muon_optimizer.load_state_dict(state_dict['muon'])
        if 'adam' in state_dict:
            self.adam_optimizer.load_state_dict(state_dict['adam'])

    # ── mmengine OptimWrapper 兼容属性 ─────────────────

    @property
    def param_groups(self):
        return self._param_groups

    @property
    def optimizer(self):
        """mmengine 期望的 optimizer 属性 (返回主优化器)"""
        return self.muon_optimizer

    def state_dict_for_checkpoint(self):
        """mmengine 检查点保存用"""
        return self.state_dict()

    def get_lr(self):
        """获取学习率"""
        lrs = []
        for group in self._param_groups:
            lrs.append(group.get('lr', 0))
        return lrs

    def set_lr(self, lr):
        """设置所有参数组学习率"""
        for group in self._param_groups:
            group['lr'] = lr

    # ── 反向传播接口 (mmengine 期望) ─────────────────

    def backward(self, loss, scaler=None):
        """反向传播"""
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def scale_loss(self, loss):
        """返回 loss (无缩放)"""
        return loss

    def unscale_(self, scaler=None):
        """反缩放 (无操作)"""
        pass

    def update_params(self, loss):
        """mmengine 接口: backward + step"""
        self.backward(loss)
        self.step()
        self.zero_grad()
