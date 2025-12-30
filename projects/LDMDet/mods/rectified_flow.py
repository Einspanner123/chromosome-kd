import torch
from torch import Tensor
from typing import List, Optional, Tuple

class RectifiedFlow:
    """
    Rectified Flow (1-RectFlow) 模块。
    实现了直线路径的前向扩散和采样逻辑。
    $x_t = (1-t)x_0 + t x_1$
    其中 $x_0$ 是数据 (GT)，$x_1$ 是噪声。
    """

    def __init__(self, snr_scale: float = 2.0):
        self.snr_scale = snr_scale

    def q_sample(
        self, x_start: Tensor, x_noise: Optional[Tensor] = None, t: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        前向采样 (加噪过程): x_t = (1-t)x_0 + t * x_1
        
        Args:
            x_start (Tensor): 原始数据 x_0, shape [N, 4]
            x_noise (Tensor, optional): 噪声 x_1. 默认为标准正态分布。
            t (Tensor): 时间步, 范围 [0, 1], shape [N] or [1]
            
        Returns:
            x_t (Tensor): 加噪后的样本
            velocity (Tensor): 目标速度 x_1 - x_0
        """
        if x_noise is None:
            x_noise = torch.randn_like(x_start)
        
        if t is None:
            # 随机采样 t 
            t = torch.rand((x_start.shape[0],), device=x_start.device)
            
        # 扩展 t 的维度以便广播
        t_view = t.view(-1, *([1] * (x_start.dim() - 1)))
        
        x_t = (1.0 - t_view) * x_start + t_view * x_noise
        velocity = x_noise - x_start
        
        return x_t, velocity

    def get_velocity(self, x_t: Tensor, x_0_pred: Tensor, t: Tensor) -> Tensor:
        """
        根据预测的 x_0 计算当前的速度 v_t。
        在 Rectified Flow 中, v_t = (x_t - x_0_pred) / t (当 t > 0 时)
        或者更直接地，如果模型预测了 x_0，且我们知道当前的 x_t 和 t，
        那么从 x_t = (1-t)x_0 + t x_1 可以推导出 x_1 = (x_t - (1-t)x_0) / t
        则 v = x_1 - x_0 = (x_t - x_0) / t
        """
        t_view = t.view(-1, *([1] * (x_t.dim() - 1)))
        # 避免除以 0
        v_t = (x_t - x_0_pred) / torch.clamp(t_view, min=1e-5)
        return v_t

    def step(
        self, 
        x_t: Tensor, 
        x_0_pred: Tensor, 
        t_curr: float, 
        t_next: float
    ) -> Tensor:
        """
        ODE 采样的一步 (Euler Step): x_{t_next} = x_t + (t_next - t_curr) * v_t
        """
        dt = t_next - t_curr
        v_t = self.get_velocity(x_t, x_0_pred, torch.tensor([t_curr], device=x_t.device))
        x_next = x_t + dt * v_t
        return x_next
