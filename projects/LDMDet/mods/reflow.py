"""
Reflow 机制 — FlowDet

迭代拉直传输路径，使单步推理接近多步质量。

算法:
1. 用训练好的模型多步 ODE 生成 (noise, detection) 配对
2. 将配对存储后重新训练模型
3. 新模型的传输路径更直，单步质量更高
4. 可重复迭代 (Reflow ×1, ×2, ...)

参考:
- Liu et al. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data
  with Rectified Flow.
"""

import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class DetectionReflow:
    """检测 Reflow: 生成 (noise, detection) 配对并拉直路径

    使用方法:
    1. 先训练一个标准 FlowDet 模型 (Generation 0)
    2. 调用 generate_pairs() 生成新的配对数据
    3. 用新数据调用 reflow_train_step() 重新训练
    """

    def __init__(
        self,
        model: nn.Module,
        num_ode_steps: int = 10,
        snr_scale: float = 2.0,
    ):
        """
        Args:
            model: 训练好的 FlowDet 模型 (DiffusionDetHead)
            num_ode_steps: 用于生成高质量检测结果的 ODE 步数
            snr_scale: 信噪比缩放
        """
        self.model = model
        self.num_ode_steps = num_ode_steps
        self.snr_scale = snr_scale

    @torch.no_grad()
    def generate_pairs(
        self,
        features: Tensor,
        img_metas: List,
        noise_source: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """用当前模型多步 ODE 生成 (noise, detection) 配对

        Args:
            features: FPN 特征 (已提取)
            img_metas: 图像元信息
            noise_source: 可选的指定噪声源

        Returns:
            z: (bs, N, 4) 原始噪声 (扩散空间)
            b_pred: (bs, N, 4) 多步推理结果 (扩散空间)
        """
        device = features[0].device
        bs = len(img_metas)

        # 采样初始噪声
        if noise_source is not None:
            z = noise_source
        else:
            z = self.model._sample_noise(bs, device)

        # 多步 ODE 求解
        b = z.clone()
        times = torch.linspace(1.0, 0.0, self.num_ode_steps + 1, device=device)

        for i in range(self.num_ode_steps):
            t_curr = times[i].item()
            t_next = times[i + 1].item()

            # 获取预测的 x0
            _, _, x0_raw, _ = self.model._forward_at_t(
                features, b, t_curr, img_metas
            )

            # Euler step
            b = self.model.rf.step(b, x0_raw, t_curr, t_next)

        return z, b  # (noise, detection) 配对

    @staticmethod
    def reflow_loss(
        model: nn.Module,
        features: Tensor,
        z: Tensor,
        b_pred: Tensor,
        img_metas: List,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Reflow 训练损失

        在 (z, b_pred) 配对上训练，target 是直线路径的速度 v = b_pred - z

        Args:
            model: 要训练的模型
            features: FPN 特征
            z: (bs, N, 4) 噪声源 (扩散空间)
            b_pred: (bs, N, 4) 检测结果 (扩散空间)
            img_metas: 图像元信息
            t: (bs,) 可选指定时间步

        Returns:
            MSE loss
        """
        device = features[0].device
        bs = z.shape[0]

        if t is None:
            t = torch.rand(bs, device=device)

        # 插值: b_t = (1-t)z + t*b_pred (注意方向：t=0 是噪声端，t=1 是数据端)
        # 但在我们的 RF 约定中 t=0 是数据, t=1 是噪声
        # 所以: b_t = (1-t)*b_pred + t*z
        t_view = t.view(bs, 1, 1)
        b_t = (1.0 - t_view) * b_pred + t_view * z

        # 目标速度: z - b_pred (从数据到噪声的方向, 与 RF 约定一致)
        v_target = z - b_pred

        # 将 b_t 转到图像空间，过模型
        curr_bboxes = model._raw_to_xyxy(b_t, img_metas)
        t_input = t * model.timesteps
        cls_seq, bbox_seq, _, _ = model(features, curr_bboxes, t_input)

        # 将模型预测的 x0 转回扩散空间，计算隐含的速度
        x0_pred = model._xyxy_to_raw(bbox_seq[-1], img_metas)
        # 隐含速度: v_pred = (b_t - x0_pred) / t
        t_safe = torch.clamp(t_view, min=1e-5)
        v_pred = (b_t - x0_pred) / t_safe

        # MSE loss on velocity
        loss = F.mse_loss(v_pred, v_target)
        return loss

    def save_pairs(
        self,
        save_dir: str,
        z: Tensor,
        b_pred: Tensor,
        batch_idx: int = 0,
    ):
        """保存 reflow 配对到磁盘"""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {"z": z.cpu(), "b_pred": b_pred.cpu()},
            os.path.join(save_dir, f"reflow_pairs_{batch_idx:06d}.pt"),
        )

    @staticmethod
    def load_pairs(save_dir: str) -> Tuple[Tensor, Tensor]:
        """从磁盘加载所有 reflow 配对"""
        all_z = []
        all_b = []
        for fname in sorted(os.listdir(save_dir)):
            if fname.startswith("reflow_pairs_"):
                data = torch.load(os.path.join(save_dir, fname), map_location="cpu")
                all_z.append(data["z"])
                all_b.append(data["b_pred"])
        return torch.cat(all_z, dim=0), torch.cat(all_b, dim=0)
