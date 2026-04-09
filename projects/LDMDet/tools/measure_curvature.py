"""
路径曲率测量工具 — FlowDet

量化传输路径的直线性，用于:
1. 验证 Reflow 的效果
2. 对比 DDPM vs Flow Matching 的路径差异
3. 论文 Figure 数据

曲率定义:
    curvature = E[ max_t ||φ_actual(t) - φ_linear(t)|| / ||φ(1) - φ(0)|| ]
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@torch.no_grad()
def measure_path_curvature(
    model: nn.Module,
    features: Tensor,
    img_metas: List,
    num_t_points: int = 100,
    noise_source: Optional[Tensor] = None,
) -> dict:
    """测量去噪路径的曲率

    Args:
        model: DiffusionDetHead 模型
        features: FPN 特征
        img_metas: 图像元信息
        num_t_points: 路径采样点数
        noise_source: 可选的固定噪声源

    Returns:
        dict with:
            'mean_curvature': 平均曲率
            'max_curvature': 最大曲率
            'per_box_curvature': (bs, N) 每个框的曲率
            'path_points': (T+1, bs, N, 4) 路径点
            'linear_path': (T+1, bs, N, 4) 直线路径
    """
    device = features[0].device
    bs = len(img_metas)

    # 采样初始噪声
    if noise_source is not None:
        z = noise_source
    else:
        z = model._sample_noise(bs, device)

    # 多步求解真实路径
    path_points = [z.clone()]
    b = z.clone()
    times = torch.linspace(1.0, 0.0, num_t_points + 1, device=device)

    for i in range(num_t_points):
        t_curr = times[i].item()
        t_next = times[i + 1].item()

        _, _, x0_raw, _ = model._forward_at_t(features, b, t_curr, img_metas)
        b = model.rf.step(b, x0_raw, t_curr, t_next)
        path_points.append(b.clone())

    path_points = torch.stack(path_points)  # (T+1, bs, N, 4)

    # 直线路径 (从起点到终点的线性插值)
    start = path_points[0]  # (bs, N, 4) — 噪声端
    end = path_points[-1]  # (bs, N, 4) — 数据端
    alphas = torch.linspace(0, 1, num_t_points + 1, device=device).view(-1, 1, 1, 1)
    linear_path = start.unsqueeze(0) * (1 - alphas) + end.unsqueeze(0) * alphas
    # (T+1, bs, N, 4)

    # 计算偏离度
    deviation = (path_points - linear_path).norm(dim=-1)  # (T+1, bs, N)
    max_deviation = deviation.max(dim=0).values  # (bs, N) — 每条路径的最大偏离

    # 总位移
    displacement = (end - start).norm(dim=-1)  # (bs, N)

    # 曲率 = 最大偏离 / 总位移
    per_box_curvature = max_deviation / displacement.clamp(min=1e-6)

    return {
        "mean_curvature": per_box_curvature.mean().item(),
        "max_curvature": per_box_curvature.max().item(),
        "median_curvature": per_box_curvature.median().item(),
        "per_box_curvature": per_box_curvature,
        "path_points": path_points,
        "linear_path": linear_path,
    }


@torch.no_grad()
def compare_path_straightness(
    model_ddpm: nn.Module,
    model_rf: nn.Module,
    features: Tensor,
    img_metas: List,
    num_t_points: int = 50,
) -> dict:
    """对比 DDPM 和 RF 模型的路径直线性

    Returns:
        dict with curvature comparison
    """
    # 使用相同的噪声源
    device = features[0].device
    bs = len(img_metas)
    shared_noise = torch.randn(bs, model_rf.num_proposals, 4, device=device)

    results = {}

    # RF 模型
    if hasattr(model_rf, "rf"):
        rf_curvature = measure_path_curvature(
            model_rf, features, img_metas, num_t_points, noise_source=shared_noise
        )
        results["rf_mean_curvature"] = rf_curvature["mean_curvature"]
        results["rf_max_curvature"] = rf_curvature["max_curvature"]

    return results
