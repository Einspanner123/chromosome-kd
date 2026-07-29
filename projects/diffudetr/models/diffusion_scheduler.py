"""扩散调度器 — DiffuDETR 核心扩散逻辑移植

从原仓库 (MBadran2000/DiffuDETR) 的 dino_diffu_det_noise.py 移植:
  - cosine_beta_schedule (1000 timesteps, s=0.008)
  - register_schedule: 注册 alphas_cumprod 等缓冲区 + SNR loss_weight
  - q_sample: 前向扩散 x_t = sqrt(acp_t)*x_0 + sqrt(1-acp_t)*noise
  - predict_noise_from_start: 由 x0 反推 noise (parameterization="x0")
  - DDIM 采样: 25 步, eta=0 (确定性)

关键设计决策:
  - parameterization="x0": 模型直接预测 x0 (干净框), 而非预测 noise
  - scale=2: box [0,1] → [-2,2] 扩散空间, 匹配 N(0,1) 噪声尺度
  - loss_weight (SNR/VLB 加权): 0.5*sqrt(acp)/(2-acp), 低 timestep (高 SNR, acp≈1)
    权重高 (~0.5), 高 timestep (低 SNR, acp≈0) 权重低 (~0).
    语义: 高噪声时 x0 预测不可靠 → 降权, 抑制噪声大时的损失.
    注: 原仓库 50ep use_vlb=False, 此权重实际未参与训练; 公式仅注册对齐.
    对齐原仓库 register_schedule_old 中 lvlb_weights 公式 (line 278-279).
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Improved DDPM, https://openreview.net/forum?id=-NEXDKk8gZ).

    对齐原仓库 cosine_beta_schedule:
        alphas_cumprod = cos(((x/T)+s)/(1+s) * pi/2)^2
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]

    Args:
        timesteps: 扩散总步数 (默认 1000).
        s: 偏移量, 防止 t=0 时 beta 过小.

    Returns:
        betas: [T] beta 值, clip 到 [0, 0.999].
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(
    a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]
) -> torch.Tensor:
    """从序列 a 中按时间步索引 t 提取值, 并 reshape 以广播到 x_shape.

    对齐原仓库 extract 函数.

    Args:
        a: [T] 参数序列 (如 alphas_cumprod).
        t: [B] 时间步索引.
        x_shape: 目标张量形状, 用于推断广播维度.

    Returns:
        [B, 1, 1, ...] 与 x_shape 维度数对齐的张量.
    """
    batch_size = t.shape[0]
    if len(t.shape) > 1:
        t = t.squeeze()
    out = a.gather(-1, t.long())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    repeat_only: bool = False,
) -> torch.Tensor:
    """正弦时间步嵌入 (Sinusoidal timestep embedding).

    对齐原仓库 ldm/modules/diffusionmodules/util.py timestep_embedding.
    将标量时间步编码为 [B, dim] 的正弦向量, 供 TimeStepBlock 使用.

    Args:
        timesteps: [B] 时间步.
        dim: 输出维度 (embed_dim).
        max_period: 角度最大周期.
        repeat_only: 若 True 只重复时间步 (不使用正弦).

    Returns:
        [B, dim] 时间步嵌入.
    """
    if repeat_only:
        return timesteps.view(-1, 1).repeat(1, dim)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(
            start=0, end=half, dtype=torch.float32, device=timesteps.device
        )
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def make_ddim_timesteps(
    num_ddim_timesteps: int, num_ddpm_timesteps: int, verbose: bool = False
) -> np.ndarray:
    """从 DDPM 时间步中均匀采样 DDIM 时间步.

    对齐原仓库 make_ddim_timesteps.
    返回的索引为降序 (从大到小, 反向扩散用).

    Args:
        num_ddim_timesteps: DDIM 采样步数 (如 25).
        num_ddpm_timesteps: DDPM 总步数 (如 1000).

    Returns:
        [num_ddim_timesteps] 时间步索引数组 (降序).
    """
    step_ratio = num_ddpm_timesteps // num_ddim_timesteps
    timesteps = (
        (np.arange(0, num_ddim_timesteps) * step_ratio)
        .round()[::-1]
        .astype(int)
    )
    if verbose:
        print(f'DDIM timesteps: {timesteps}')
    return timesteps


class DiffusionScheduler(nn.Module):
    """扩散调度器 — 管理 beta/alpha 缓冲区及 q_sample/DDIM 采样.

    对齐原仓库 DIFFUSION_DINO.register_schedule_old + q_sample + DDIM 逻辑.
    所有缓冲区随模型自动迁移设备.

    Args:
        timesteps: DDPM 总步数 (默认 1000).
        sampling_timesteps: DDIM 采样步数 (默认 25).
        ddim_eta: DDIM 随机性参数, 0=确定性 (默认 0).
        scale: 扩散空间缩放, box [0,1] → [-scale, scale] (默认 2).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        sampling_timesteps: int = 25,
        ddim_eta: float = 0.0,
        scale: float = 2.0,
    ):
        super().__init__()
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_eta
        self.scale = scale
        # parameterization="x0": 模型直接预测 x0 (干净框)
        self.parameterization = 'x0'
        self.register_schedule(timesteps)

    def register_schedule(self, timesteps: int) -> None:
        """注册扩散调度缓冲区 + SNR loss_weight.

        对齐原仓库 register_schedule_old:
          - cosine_beta_schedule 生成 betas
          - alphas = 1 - betas, alphas_cumprod = cumprod(alphas)
          - loss_weight (SNR 加权): 0.5*sqrt(acp)/(2-acp)
        """
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas.float())
        self.register_buffer('alphas_cumprod', alphas_cumprod.float())
        self.register_buffer(
            'alphas_cumprod_prev', alphas_cumprod_prev.float()
        )

        # 前向扩散 q(x_t|x_0) 所需
        self.register_buffer(
            'sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float()
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - alphas_cumprod).float(),
        )
        self.register_buffer(
            'log_one_minus_alphas_cumprod',
            torch.log(1.0 - alphas_cumprod).float(),
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod',
            torch.sqrt(1.0 / alphas_cumprod).float(),
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod',
            torch.sqrt(1.0 / alphas_cumprod - 1).float(),
        )

        # SNR/VLB 加权损失权重 (对齐原仓库 lvlb_weights, line 278-279)
        # 原公式: 0.5 * sqrt(alphas_cumprod) / (2 - alphas_cumprod)
        # 趋势: 低 t (acp≈1) → 权重 ~0.5; 高 t (acp≈0) → 权重 ~0
        # 语义: 高噪声时 x0 预测不可靠 → 降权 (抑制噪声大时的损失)
        acp = alphas_cumprod.double()
        lvlb_weights = 0.5 * torch.sqrt(acp) / (2.0 - acp)
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer(
            'loss_weight', lvlb_weights.float(), persistent=False
        )

        # 后验 q(x_{t-1}|x_t, x_0) 所需 (DDPM posterior, DDIM 不直接用但保留)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance.float())
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)).float(),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            (
                betas
                * torch.sqrt(alphas_cumprod_prev)
                / (1.0 - alphas_cumprod)
            ).float(),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (
                (1.0 - alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - alphas_cumprod)
            ).float(),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向扩散: x_t = sqrt(acp_t)*x_0 + sqrt(1-acp_t)*noise.

        对齐原仓库 q_sample. x_start 在扩散空间 [-scale, scale].

        Args:
            x_start: [B, N, 4] 干净框 (扩散空间).
            t: [B] 时间步.
            noise: [B, N, 4] 噪声, None 则随机生成.

        Returns:
            x_t: [B, N, 4] 加噪框.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_acp_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_acp_t * x_start + sqrt_one_minus_acp_t * noise

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """由预测的 x0 反推 noise (parameterization="x0").

        对齐原仓库 predict_start_from_noise 的逆运算:
            noise = (sqrt(1/acp)*x_t - x0) / sqrt(1/acp - 1)

        用于 DDIM 采样: 先预测 x0, 再反推 noise, 再做 DDIM 更新.

        Args:
            x_t: [B, N, 4] 当前加噪框.
            t: [B] 时间步.
            x0: [B, N, 4] 模型预测的干净框.

        Returns:
            [B, N, 4] 推导出的噪声.
        """
        sqrt_recip_acp_t = extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_acp_t = extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )
        return (sqrt_recip_acp_t * x_t - x0) / sqrt_recipm1_acp_t

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_next: int,
        x0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """单步 DDIM 采样 (eta=0 时为确定性).

        对齐 DiffusionDet/DDIM 公式:
            sigma = eta * sqrt((1-alpha/alpha_next)*(1-alpha_next)/(1-alpha))
            c = sqrt(1 - alpha_next - sigma^2)
            x_{t-1} = sqrt(alpha_next)*x0 + c*pred_noise + sigma*noise

        当 eta=0: sigma=0, c=sqrt(1-alpha_next),
            x_{t-1} = sqrt(alpha_next)*x0 + sqrt(1-alpha_next)*pred_noise

        Args:
            x_t: [B, N, 4] 当前加噪框 (用于反推 noise).
            t: 当前时间步 (标量).
            t_next: 下一时间步 (标量, <0 表示最后一步).
            x0: [B, N, 4] 模型预测的干净框.
            noise: 可选随机噪声, eta>0 时使用.

        Returns:
            x_{t-1}: [B, N, 4] 去噪一步后的框.
        """
        if t_next < 0:
            # 最后一步直接返回 x0
            return x0

        device = x_t.device
        B = x_t.shape[0]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        pred_noise = self.predict_noise_from_start(x_t, t_batch, x0)

        alpha = self.alphas_cumprod[t]
        alpha_next = self.alphas_cumprod[t_next]
        sigma = (
            self.ddim_sampling_eta
            * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        if noise is None:
            noise = torch.randn_like(x_t)
        return x0 * alpha_next.sqrt() + c * pred_noise + sigma * noise

    def get_time_pairs(self) -> list:
        """生成 DDIM 反向采样时间对.

        对齐 DiffusionDet prepare_testing_targets:
            times = linspace(-1, T-1, steps=sampling+1)
            times = reversed(int(times))
            time_pairs = zip(times[:-1], times[1:])

        返回 [(T-1, T-2), ..., (1, 0), (0, -1)] 共 sampling_timesteps 对.

        Returns:
            时间对列表, 每对 (time, time_next).
        """
        times = torch.linspace(
            -1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        return list(zip(times[:-1], times[1:]))

    def get_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """获取时间步 t 对应的 SNR 损失权重.

        Args:
            t: [B] 时间步.

        Returns:
            [B, 1, 1] 损失权重 (用于加权回归损失).
        """
        return extract(self.loss_weight, t, (t.shape[0], 1, 1))
