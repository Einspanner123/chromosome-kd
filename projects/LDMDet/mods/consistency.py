"""
一致性蒸馏 — FlowDet

将多步 FlowDet teacher 蒸馏为单步 student，
使用一致性模型 (Consistency Model) 约束。

核心约束: f(b_t, t) 在同一 ODE 轨迹上的任意两点应映射到相同的数据点。
即 f_student(b_tn, tn) ≈ f_ema(b_{tn-Δt}, tn-Δt)

参考:
- Song et al. (2023). Consistency Models.
"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConsistencyDetDistillation(nn.Module):
    """一致性蒸馏: 从多步 teacher 到单步 student

    使用方法:
    1. 训练好多步 FlowDet (teacher)
    2. 初始化 student = copy(teacher)
    3. 迭代训练: 使用 consistency loss 约束 student

    约束:
    f_student(b_tn, tn) = f_ema(b_{tn+1}, tn+1)
    其中 b_{tn+1} = b_tn + Δt * v_teacher(b_tn, tn) (teacher 的一步更新)
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        ema_rate: float = 0.999,
        num_timesteps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ):
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = student
        self.ema_student = deepcopy(student)
        self.ema_student.eval()
        for p in self.ema_student.parameters():
            p.requires_grad = False

        self.ema_rate = ema_rate
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @torch.no_grad()
    def update_ema(self):
        """更新 EMA student"""
        for p_ema, p_student in zip(
            self.ema_student.parameters(), self.student.parameters()
        ):
            p_ema.data.mul_(self.ema_rate).add_(p_student.data, alpha=1 - self.ema_rate)

    def consistency_loss(
        self,
        features: Tensor,
        img_metas: List,
        x_raw: Tensor,
    ) -> Tensor:
        """计算一致性蒸馏损失

        Args:
            features: FPN 特征
            img_metas: 图像元信息
            x_raw: (bs, N, 4) 噪声框 (扩散空间)

        Returns:
            consistency loss
        """
        device = features[0].device
        bs = x_raw.shape[0]

        # 采样两个相邻时间步
        # t_n > t_{n-1}, 越大越接近噪声
        indices = torch.randint(1, self.num_timesteps, (bs,), device=device)
        t_n = indices.float() / self.num_timesteps
        t_n_minus_1 = (indices - 1).float() / self.num_timesteps

        # 1. Teacher: 在 t_n 处预测 x0, 做一步 ODE 到 t_{n-1}
        with torch.no_grad():
            _, _, x0_teacher, _ = self.teacher._forward_at_t(
                features, x_raw, t_n[0].item(), img_metas
            )
            # Euler step: x_{t_{n-1}} = x_{t_n} + (t_{n-1} - t_n) * v_t
            x_at_t_n_minus_1 = self.teacher.rf.step(
                x_raw, x0_teacher, t_n[0].item(), t_n_minus_1[0].item()
            )

        # 2. Student: 在 t_n 处预测 x0
        _, _, x0_student_n, _ = self.student._forward_at_t(
            features, x_raw, t_n[0].item(), img_metas
        )

        # 3. EMA Student: 在 t_{n-1} 处预测 x0
        with torch.no_grad():
            _, _, x0_ema_n_minus_1, _ = self.ema_student._forward_at_t(
                features, x_at_t_n_minus_1, t_n_minus_1[0].item(), img_metas
            )

        # 4. Consistency loss: student(x_tn, tn) ≈ ema_student(x_{tn-1}, tn-1)
        loss = F.mse_loss(x0_student_n, x0_ema_n_minus_1)

        return loss

    def train_step(
        self,
        features: Tensor,
        img_metas: List,
    ) -> Tensor:
        """一步训练"""
        device = features[0].device
        bs = len(img_metas)

        # 采样噪声
        x_raw = self.student._sample_noise(bs, device)

        loss = self.consistency_loss(features, img_metas, x_raw)

        return loss
