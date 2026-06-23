"""方向四轨迹诊断 — 验证非线性轨迹是否按预期工作.

若方向四废弃, 删除本文件 + tests/unit/test_diagnostics_trajectory.py 即可清理.

诊断内容:
1. 尺度条件化统计 (κ 分布、t_eff 分布、尺度-时间相关性)
2. OT 耦合质量 (配对代价、传输矩阵边缘)
3. 路径曲率 (速度场变化率, 用于评估 2-RectFlow 收益)

使用方式:
- 在 head.py 的 loss() 中调用 TrajectoryDiagnosticsCallback.update_scale()
  / update_ot() / update_curvature()
- TrainingDiagnosticsHook 通过 diagnostics_callback 自动采集
"""

from typing import Dict, Optional

import torch
from torch import Tensor


# ──────────────────────────────────────────────
# 尺度条件化统计
# ──────────────────────────────────────────────

def compute_scale_conditioned_stats(
    scales: Tensor,
    t: Tensor,
    t_eff: Tensor,
    kappa: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算尺度条件化 RF 的统计.

    Args:
        scales: [bs, N] 框尺度 sqrt(area)
        t: [bs] 原始时间
        t_eff: [bs, N] 有效时间 t^{1/κ(s)}
        kappa: [bs, N] 调制系数 κ(s) (可选, 若提供则统计)

    Returns:
        统计字典:
        - scale_mean/std/min/max: 尺度分布
        - t_eff_mean/std: 有效时间分布
        - t_eff_t_correlation: t_eff 与 t 的相关系数 (越接近 1 越接近线性)
        - scale_t_eff_correlation: 尺度与 t_eff 的相关性 (负相关表示小尺度→小 t_eff)
        - kappa_mean/std (若提供): κ 分布
    """
    s = scales.float()
    t_eff_flat = t_eff.float().reshape(-1)
    s_flat = s.float().reshape(-1)

    stats: Dict[str, float] = {
        'scale_mean': float(s_flat.mean().item()),
        'scale_std': float(s_flat.std().item()) if s_flat.numel() > 1 else 0.0,
        'scale_min': float(s_flat.min().item()),
        'scale_max': float(s_flat.max().item()),
        't_eff_mean': float(t_eff_flat.mean().item()),
        't_eff_std': float(t_eff_flat.std().item()) if t_eff_flat.numel() > 1 else 0.0,
    }

    # t_eff 与 t 的相关性 (广播 t 到 [bs, N])
    t_float = t.float()
    if t.dim() == 1 and s.dim() == 2:
        t_expanded = t_float.unsqueeze(1).expand_as(s).reshape(-1)
    else:
        t_expanded = t_float.reshape(-1)

    if t_eff_flat.numel() > 1 and t_eff_flat.std() > 0 and t_expanded.std() > 0:
        corr_t = torch.corrcoef(torch.stack([t_eff_flat, t_expanded]))[0, 1]
        stats['t_eff_t_correlation'] = float(corr_t.item())
    else:
        stats['t_eff_t_correlation'] = 1.0  # 退化为线性

    # 尺度与 t_eff 的相关性
    if s_flat.numel() > 1 and s_flat.std() > 0 and t_eff_flat.std() > 0:
        corr_s = torch.corrcoef(torch.stack([s_flat, t_eff_flat]))[0, 1]
        stats['scale_t_eff_correlation'] = float(corr_s.item())
    else:
        stats['scale_t_eff_correlation'] = 0.0

    # κ 统计
    if kappa is not None:
        k_flat = kappa.float().reshape(-1)
        stats['kappa_mean'] = float(k_flat.mean().item())
        stats['kappa_std'] = float(k_flat.std().item()) if k_flat.numel() > 1 else 0.0
        stats['kappa_min'] = float(k_flat.min().item())
        stats['kappa_max'] = float(k_flat.max().item())

    return stats


# ──────────────────────────────────────────────
# OT 耦合质量统计
# ──────────────────────────────────────────────

def compute_ot_coupling_stats(
    transport: Tensor,
    cost: Optional[Tensor] = None,
) -> Dict[str, float]:
    """计算 OT 传输矩阵的质量统计.

    Args:
        transport: [N, K] 传输矩阵
        cost: [N, K] 代价矩阵 (可选, 用于计算配对代价)

    Returns:
        统计字典:
        - transport_entropy: 传输矩阵熵 (越高越均匀)
        - transport_concentration: 集中度 (每行最大值均值, 越高越集中)
        - row_marginal_std: 行边缘标准差 (越接近 0 越均匀)
        - col_marginal_std: 列边缘标准差
        - paired_cost_mean/max (若提供 cost): 配对代价
    """
    P = transport.float().clamp_min(1e-10)
    N, K = P.shape

    # 熵 (归一化)
    P_norm = P / P.sum().clamp_min(1e-10)
    entropy = float(-(P_norm * P_norm.log()).sum().item())
    max_entropy = float(torch.log(torch.tensor(float(N * K))).item())
    stats: Dict[str, float] = {
        'transport_entropy': entropy / max(max_entropy, 1e-10),
        'transport_concentration': float(P.max(dim=1).values.mean().item()),
    }

    # 边缘标准差
    row_marginal = P.sum(dim=1)
    col_marginal = P.sum(dim=0)
    stats['row_marginal_std'] = float(row_marginal.std().item()) if N > 1 else 0.0
    stats['col_marginal_std'] = float(col_marginal.std().item()) if K > 1 else 0.0

    # 配对代价
    if cost is not None:
        coupled_idx = P.argmax(dim=1)
        paired_cost = cost.float().gather(1, coupled_idx.unsqueeze(1)).squeeze(1)
        stats['paired_cost_mean'] = float(paired_cost.mean().item())
        stats['paired_cost_max'] = float(paired_cost.max().item())

    return stats


# ──────────────────────────────────────────────
# 路径曲率统计
# ──────────────────────────────────────────────

def compute_path_curvature(
    velocity_t: Tensor,
    velocity_t_next: Tensor,
    dt: float = 1.0,
) -> Dict[str, float]:
    """计算路径曲率 κ(t) = ||dv/dt||.

    用于评估 2-RectFlow 收益: κ 越小, 路径越直, 少步采样精度越高.

    Args:
        velocity_t: [bs, N, D] 时刻 t 的速度场
        velocity_t_next: [bs, N, D] 时刻 t+dt 的速度场
        dt: 时间间隔

    Returns:
        统计字典:
        - curvature_mean/std/max: 曲率统计
        - curvature_relative: 相对曲率 (曲率/速度 norm, 无量纲)
    """
    dv = (velocity_t_next - velocity_t).float()
    curvature = dv.norm(2, dim=-1) / max(dt, 1e-8)  # [bs, N]

    v_norm = velocity_t.float().norm(2, dim=-1).clamp_min(1e-8)
    curvature_relative = curvature / v_norm

    return {
        'curvature_mean': float(curvature.mean().item()),
        'curvature_std': float(curvature.std().item()) if curvature.numel() > 1 else 0.0,
        'curvature_max': float(curvature.max().item()),
        'curvature_relative': float(curvature_relative.mean().item()),
    }


def compute_velocity_stats(
    velocity: Tensor,
) -> Dict[str, float]:
    """计算速度场统计 (用于诊断路径质量).

    Args:
        velocity: [bs, N, D] 速度场

    Returns:
        统计字典:
        - velocity_norm_mean/std: 速度 norm 统计
        - velocity_norm_min/max
        - velocity_direction_consistency: 速度方向一致性
            (衡量 batch 内速度方向是否一致, 越高越接近直线)
    """
    v = velocity.float()
    v_norm = v.norm(2, dim=-1)  # [bs, N]

    stats: Dict[str, float] = {
        'velocity_norm_mean': float(v_norm.mean().item()),
        'velocity_norm_std': float(v_norm.std().item()) if v_norm.numel() > 1 else 0.0,
        'velocity_norm_min': float(v_norm.min().item()),
        'velocity_norm_max': float(v_norm.max().item()),
    }

    # 方向一致性: 计算每个样本内速度方向的平均余弦相似度
    # v: [bs, N, D] → 单位向量 [bs, N, D]
    v_unit = v / v_norm.unsqueeze(-1).clamp_min(1e-8)
    # 平均单位向量: [bs, D]
    v_mean = v_unit.mean(dim=1)
    v_mean_norm = v_mean.norm(2, dim=-1).clamp_min(1e-8)
    # 一致性 = ||mean(v_unit)|| ∈ [0, 1]
    consistency = float(v_mean_norm.mean().item())
    stats['velocity_direction_consistency'] = consistency

    return stats


# ──────────────────────────────────────────────
# 诊断回调
# ──────────────────────────────────────────────

class TrajectoryDiagnosticsCallback:
    """轨迹诊断回调, 集成到 TrainingDiagnosticsHook.

    用法:
        1. 在 head.py 的 loss() 中调用:
           - callback.update_scale(scales, t, t_eff, kappa) (尺度条件化)
           - callback.update_ot(transport, cost) (OT 耦合)
           - callback.update_curvature(v_t, v_t_next, dt) (路径曲率)
        2. TrainingDiagnosticsHook 通过 diagnostics_callback 自动调用 collect()

    Args:
        interval: 采样间隔 (iter), 默认 100
    """

    def __init__(self, interval: int = 100):
        self.interval = interval
        self.last_scales: Optional[Tensor] = None
        self.last_t: Optional[Tensor] = None
        self.last_t_eff: Optional[Tensor] = None
        self.last_kappa: Optional[Tensor] = None
        self.last_transport: Optional[Tensor] = None
        self.last_cost: Optional[Tensor] = None
        self.last_velocity: Optional[Tensor] = None
        self.last_velocity_next: Optional[Tensor] = None

    def update_scale(
        self,
        scales: Tensor,
        t: Tensor,
        t_eff: Tensor,
        kappa: Optional[Tensor] = None,
    ) -> None:
        """更新尺度条件化数据."""
        self.last_scales = scales.detach().cpu()
        self.last_t = t.detach().cpu()
        self.last_t_eff = t_eff.detach().cpu()
        if kappa is not None:
            self.last_kappa = kappa.detach().cpu()

    def update_ot(
        self,
        transport: Tensor,
        cost: Optional[Tensor] = None,
    ) -> None:
        """更新 OT 耦合数据."""
        self.last_transport = transport.detach().cpu()
        if cost is not None:
            self.last_cost = cost.detach().cpu()

    def update_curvature(
        self,
        velocity_t: Tensor,
        velocity_t_next: Tensor,
        dt: float = 1.0,
    ) -> None:
        """更新路径曲率数据."""
        self.last_velocity = velocity_t.detach().cpu()
        self.last_velocity_next = velocity_t_next.detach().cpu()
        self._curvature_dt = float(dt)

    def collect(self, step: int) -> Dict[str, float]:
        """采集诊断数据 (由 TrainingDiagnosticsHook 调用).

        Returns:
            诊断数据字典, 若非采样点则返回空字典
        """
        if step % self.interval != 0 or step == 0:
            return {}

        data: Dict[str, float] = {}

        # 尺度条件化统计
        if (self.last_scales is not None
                and self.last_t is not None
                and self.last_t_eff is not None):
            stats = compute_scale_conditioned_stats(
                self.last_scales,
                self.last_t,
                self.last_t_eff,
                self.last_kappa,
            )
            for k, v in stats.items():
                data[f'trajectory/{k}'] = v

        # OT 耦合统计
        if self.last_transport is not None:
            ostats = compute_ot_coupling_stats(
                self.last_transport, self.last_cost,
            )
            for k, v in ostats.items():
                data[f'trajectory/ot_{k}'] = v

        # 路径曲率统计
        if (self.last_velocity is not None
                and self.last_velocity_next is not None):
            cstats = compute_path_curvature(
                self.last_velocity,
                self.last_velocity_next,
                getattr(self, '_curvature_dt', 1.0),
            )
            for k, v in cstats.items():
                data[f'trajectory/{k}'] = v

            # 速度场统计
            vstats = compute_velocity_stats(self.last_velocity)
            for k, v in vstats.items():
                data[f'trajectory/{k}'] = v

        return data
