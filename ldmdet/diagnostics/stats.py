"""公共统计工具 — 权重/梯度/激活统计、数值健康度、模块分组.

所有函数均为纯函数, 不依赖外部状态, 便于测试和复用.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable

import torch
import torch.nn as nn
from torch import Tensor


# ──────────────────────────────────────────────
# 模块分组
# ──────────────────────────────────────────────

@dataclass
class ModuleGroup:
    """一组命名模块, 用于聚合统计."""
    name: str
    modules: List[nn.Module] = field(default_factory=list)


def group_modules_by_prefix(
    model: nn.Module,
    prefixes: Dict[str, str],
) -> Dict[str, ModuleGroup]:
    """按名称前缀将模型模块分组.

    Args:
        model: 待分组的模型
        prefixes: {前缀: 组名} 映射, 如 {'backbone.': 'backbone', 'head_series.': 'head'}

    Returns:
        {组名: ModuleGroup} 字典
    """
    groups: Dict[str, ModuleGroup] = {}
    for name, module in model.named_modules():
        if name == '':
            continue
        for prefix, group_name in prefixes.items():
            if name.startswith(prefix) or name == prefix.rstrip('.'):
                if group_name not in groups:
                    groups[group_name] = ModuleGroup(name=group_name)
                groups[group_name].modules.append(module)
                break
    return groups


# ──────────────────────────────────────────────
# 权重统计
# ──────────────────────────────────────────────

def compute_weight_stats(
    module: nn.Module,
) -> Dict[str, float]:
    """计算模块所有参数的权重统计.

    统计内容:
    - mean, std, norm: 权重分布
    - numel: 参数总数
    - dead_ratio: 死神经元比例 (某输出维度权重全零)
    - has_nan, has_inf: 数值健康度

    Args:
        module: 待统计的模块

    Returns:
        统计字典
    """
    params = [p.data for p in module.parameters()]
    if not params:
        return {
            'mean': 0.0, 'std': 0.0, 'norm': 0.0, 'numel': 0,
            'dead_ratio': 0.0, 'has_nan': False, 'has_inf': False,
        }

    all_w = torch.cat([p.reshape(-1) for p in params])
    has_nan = bool(torch.isnan(all_w).any().item())
    has_inf = bool(torch.isinf(all_w).any().item())

    # 死神经元检测: 仅检查 weight (dim>=2) 的输出维度是否全零
    # bias 不单独算输出维度, 因为 bias[i]=0 不代表神经元死 (weight 可能非零)
    dead_count = 0
    total_out_dims = 0
    for p in module.parameters():
        if p.dim() >= 2:
            # weight: [out, in] 或 [out, in, k, k]
            # 检查每个输出维度是否全零
            out_dims = p.shape[0]
            for i in range(out_dims):
                if p.data[i].abs().sum().item() == 0.0:
                    dead_count += 1
                total_out_dims += 1

    dead_ratio = dead_count / max(total_out_dims, 1)

    return {
        'mean': float(all_w.mean().item()),
        'std': float(all_w.std().item()) if all_w.numel() > 1 else 0.0,
        'norm': float(all_w.norm(2).item()),
        'numel': int(all_w.numel()),
        'dead_ratio': float(dead_ratio),
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


# ──────────────────────────────────────────────
# 梯度统计
# ──────────────────────────────────────────────

def compute_grad_stats(
    module: nn.Module,
    weight_data: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算模块所有参数的梯度统计.

    统计内容:
    - mean, std, norm: 梯度分布
    - numel: 参数总数
    - zero_ratio: 零梯度比例
    - grad_to_weight_ratio: 梯度均值/权重均值 (学习效率指标)
    - has_nan, has_inf: 数值健康度

    Args:
        module: 待统计的模块
        weight_data: 可选的权重张量, 用于计算 grad_to_weight_ratio

    Returns:
        统计字典
    """
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return {
            'mean': 0.0, 'std': 0.0, 'norm': 0.0, 'numel': 0,
            'zero_ratio': 1.0, 'grad_to_weight_ratio': 0.0,
            'has_nan': False, 'has_inf': False,
        }

    all_g = torch.cat([g.reshape(-1) for g in grads])
    zero_ratio = float((all_g == 0.0).float().mean().item())
    has_nan = bool(torch.isnan(all_g).any().item())
    has_inf = bool(torch.isinf(all_g).any().item())

    grad_to_weight_ratio = 0.0
    if weight_data is not None and weight_data.numel() > 0:
        weight_mean = float(weight_data.mean().item())
        grad_mean = float(all_g.mean().item())
        if abs(weight_mean) > 1e-10:
            grad_to_weight_ratio = grad_mean / weight_mean

    return {
        'mean': float(all_g.mean().item()),
        'std': float(all_g.std().item()) if all_g.numel() > 1 else 0.0,
        'norm': float(all_g.norm(2).item()),
        'numel': int(all_g.numel()),
        'zero_ratio': zero_ratio,
        'grad_to_weight_ratio': grad_to_weight_ratio,
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


# ──────────────────────────────────────────────
# 激活统计
# ──────────────────────────────────────────────

def compute_activation_stats(
    activation: Tensor,
    saturation_threshold: float = 5.0,
) -> Dict[str, float]:
    """计算激活张量的统计.

    统计内容:
    - mean, std, norm: 激活分布
    - numel: 元素总数
    - dead_ratio: 死激活比例 (值为 0)
    - saturation: 饱和度 (|x| > threshold)
    - has_nan, has_inf: 数值健康度

    Args:
        activation: 激活张量
        saturation_threshold: 饱和度阈值

    Returns:
        统计字典
    """
    if activation.numel() == 0:
        return {
            'mean': 0.0, 'std': 0.0, 'norm': 0.0, 'numel': 0,
            'dead_ratio': 0.0, 'saturation': 0.0,
            'has_nan': False, 'has_inf': False,
        }

    flat = activation.detach().reshape(-1)
    dead_ratio = float((flat == 0.0).float().mean().item())
    saturation = float((flat.abs() > saturation_threshold).float().mean().item())
    has_nan = bool(torch.isnan(flat).any().item())
    has_inf = bool(torch.isinf(flat).any().item())

    return {
        'mean': float(flat.mean().item()),
        'std': float(flat.std().item()) if flat.numel() > 1 else 0.0,
        'norm': float(flat.norm(2).item()),
        'numel': int(flat.numel()),
        'dead_ratio': dead_ratio,
        'saturation': saturation,
        'has_nan': has_nan,
        'has_inf': has_inf,
    }


# ──────────────────────────────────────────────
# 数值健康度
# ──────────────────────────────────────────────

def check_numerical_health(
    tensor: Tensor,
    extreme_threshold: float = 1e6,
) -> Dict[str, bool]:
    """检查张量的数值健康度.

    Args:
        tensor: 待检查的张量
        extreme_threshold: 极端值阈值

    Returns:
        健康度字典:
        - has_nan: 是否含 NaN
        - has_inf: 是否含 Inf
        - has_extreme: 是否含极端值 (|x| > threshold)
        - is_healthy: 是否健康 (无 NaN/Inf/极端值)
    """
    flat = tensor.detach().reshape(-1)
    has_nan = bool(torch.isnan(flat).any().item())
    has_inf = bool(torch.isinf(flat).any().item())
    has_extreme = bool((flat.abs() > extreme_threshold).any().item())
    is_healthy = not (has_nan or has_inf or has_extreme)

    return {
        'has_nan': has_nan,
        'has_inf': has_inf,
        'has_extreme': has_extreme,
        'is_healthy': is_healthy,
    }
