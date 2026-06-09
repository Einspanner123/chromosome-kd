import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class RoPE1D(nn.Module):
    """1D 旋转位置编码 (Rotary Positional Embedding)

    用于对 Box 坐标 (x, y, w, h) 进行编码。
    相比于简单的线性映射，RoPE 能够保持相对位置的旋转不变性。
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (bs, N, 4) 归一化坐标
        Returns:
            pe: (bs, N, 4, dim) 坐标嵌入
        """
        # 将坐标映射到不同频率
        # x: (bs, N, 4, 1), inv_freq: (dim/2)
        # out: (bs, N, 4, dim/2)
        sinusoid_inp = torch.einsum('bni,j->bnij', x, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


def rotate_half(x: Tensor) -> Tensor:
    """旋转一半的维度，用于 RoPE 计算"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> Tuple[Tensor, Tensor]:
    """将旋转编码应用到 Query 和 Key 上"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def cosine_noise_schedule(T, s=0.008, device='cpu'):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64, device=device)
    alphas_t = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_t = alphas_t / alphas_t[0]
    betas_t = 1 - (alphas_t[1:] / alphas_t[:-1])
    return torch.clamp(betas_t, 0, 0.999)


def load_buffer(arr: Tensor, steps: Tensor, x_shape: List[int]):
    """
    从序列a中提取时间步t对应的值,并调整形状以匹配x_shape
    a: 序列参数,如alphas_cumprod等,shape: [timesteps]
    t: 时间步索引,shape: [batch_size]
    x_shape: 目标形状,用于reshape
    """
    step_arr = arr.gather(-1, steps)
    return step_arr.reshape(
        -1, *[1] * (len(x_shape) - 1)
    )  # shape: [bs, 1, 1, 1, ..., 1(输入的维度数-1)]


class PurePyTorchSimpleFeatureFusion(nn.Module):
    """简单的特征融合模块，用于 ViT 输出的多级特征对齐

    每个 ViT block 输出先经过 LayerNorm 归一化，再进入上/下采样分支。
    LayerNorm 解决深层 block (如 Block 11) 数值爆炸问题 (std 从 1 增长到 27)。
    """

    def __init__(
        self, in_channels: List[int], out_channels: int, num_outs: int = 4
    ):
        super().__init__()
        # 每个 ViT block 输出先做 LayerNorm，消除层间数值尺度差异
        # ViT 输出格式: (B, C, H, W)，LayerNorm 在 C 维度上归一化
        self.ln_p2 = nn.LayerNorm(in_channels[0])
        self.ln_p3 = nn.LayerNorm(in_channels[1])
        self.ln_p4 = nn.LayerNorm(in_channels[2])
        self.ln_p5 = nn.LayerNorm(in_channels[3])

        # 对应 ViT 默认输出 (H/16, W/16)，我们需要通过反卷积或上采样还原多尺度
        # P2: 1/4, P3: 1/8, P4: 1/16, P5: 1/32
        self.upsample_p2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels[0], out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.upsample_p3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels[1], out_channels, kernel_size=2, stride=2
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.p4_conv = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.downsample_p5 = nn.Sequential(
            nn.Conv2d(in_channels[3], out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.num_outs = num_outs

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        # x[i] shape: (B, 384, H/16, W/16)
        # LayerNorm 需要 (B, ..., C) 格式，先 permute 再归一化再 permute 回来
        x0 = self.ln_p2(x[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x1 = self.ln_p3(x[1].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = self.ln_p4(x[2].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x3 = self.ln_p5(x[3].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        p2 = self.upsample_p2(x0)
        p3 = self.upsample_p3(x1)
        p4 = self.p4_conv(x2)
        p5 = self.downsample_p5(x3)
        return [p2, p3, p4, p5]


class SpatialPriorModule(nn.Module):
    """轻量 CNN 分支，从原图提取多尺度细粒度细节特征

    参考 DEIMv2 (https://arxiv.org/abs/2509.20787) 的 SpatialPriorModulev2。
    输出 3 级特征: 1/8, 1/16, 1/32，与 DINOv3 语义特征互补。
    """

    def __init__(self, inplanes: int = 16):
        super().__init__()
        # 1/4
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # 1/8
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
        )
        # 1/16
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )
        # 1/32
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        c1 = self.stem(x)
        c2 = self.conv2(c1)   # 1/8
        c3 = self.conv3(c2)   # 1/16
        c4 = self.conv4(c3)   # 1/32
        return [c2, c3, c4]


class SpatialTuningAdapter(nn.Module):
    """Spatial Tuning Adapter (DEIMv2 STA)

    参考: Real-Time Object Detection Meets DINOv3 (https://arxiv.org/abs/2509.20787)

    核心设计:
    1. 从 DINOv3 多个中间层取特征 → 双线性插值调整尺度 (无参数)
    2. 轻量 CNN (SpatialPriorModule) 从原图提取多尺度细粒度细节
    3. Bi-Fusion: concat 语义 + 细节 → 1×1 Conv + BN 融合
    4. 输出 3 级特征 (P3: 1/8, P4: 1/16, P5: 1/32)

    相比 PurePyTorchSimpleFeatureFusion 的改进:
    - 用无参数双线性插值替代 ConvTranspose2d (无棋盘格伪影)
    - 引入 CNN 细节分支补充 DINOv3 缺失的细粒度空间信息
    - BN 替代 LayerNorm + GroupNorm (训练更稳定)
    - 跨层融合 (语义 + 细节 concat) 替代独立上采样
    """

    def __init__(
        self,
        embed_dim: int = 384,
        out_channels: int = 256,
        conv_inplane: int = 16,
        num_levels: int = 3,
    ):
        """
        Args:
            embed_dim: DINOv3 输出特征维度 (ViT-Small: 384)
            out_channels: 输出特征通道数
            conv_inplane: CNN 细节分支基础通道数, 0 表示不使用 CNN 分支
            num_levels: 输出尺度数 (默认 3: P3, P4, P5)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.use_cnn = conv_inplane > 0

        # 轻量 CNN 细节分支
        if self.use_cnn:
            self.spatial_prior = SpatialPriorModule(inplanes=conv_inplane)

        # Bi-Fusion: 1×1 Conv 将 concat(语义 + 细节) 投影到 out_channels
        # 每级 concat 后的通道数: embed_dim + conv_inplane * k
        # P3(1/8):  embed_dim + conv_inplane*2, P4(1/16): embed_dim + conv_inplane*4, P5(1/32): embed_dim + conv_inplane*4
        if self.use_cnn:
            in_ch_list = [
                embed_dim + conv_inplane * 2,
                embed_dim + conv_inplane * 4,
                embed_dim + conv_inplane * 4,
            ]
        else:
            in_ch_list = [embed_dim] * num_levels

        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            for in_ch in in_ch_list
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in range(num_levels)
        ])

    def forward(
        self, vit_features: List[Tensor], raw_image: Tensor
    ) -> List[Tensor]:
        """
        Args:
            vit_features: DINOv3 中间层特征列表, 每个元素 shape (B, embed_dim, H/16, W/16)
            raw_image: 原始输入图像 (B, 3, H, W)
        Returns:
            3 级特征: [P3(1/8), P4(1/16), P5(1/32)]
        """
        bs = vit_features[0].shape[0]
        H_feat, W_feat = vit_features[0].shape[2], vit_features[0].shape[3]
        H_img = raw_image.shape[2]
        num_scales = len(vit_features) - 1  # 用于计算每级目标尺度

        # 1. 双线性插值: 将 ViT 特征调整到目标尺度 (无参数)
        # 3 个 ViT 特征 (均 H/16) → P3(1/8), P4(1/16), P5(1/32)
        # P3: 上采样 2x, P4: 保持, P5: 下采样 2x
        target_scales = [2, 1, 0.5]  # 相对于 H/16 的缩放因子
        sem_feats = []
        for i, feat in enumerate(vit_features):
            scale = target_scales[i] if i < len(target_scales) else target_scales[-1]
            resize_H = int(H_feat * scale)
            resize_W = int(W_feat * scale)
            if resize_H != H_feat or resize_W != W_feat:
                resized = F.interpolate(
                    feat, size=[resize_H, resize_W], mode='bilinear', align_corners=False
                )
            else:
                resized = feat
            sem_feats.append(resized)

        # 2. CNN 细节分支: 从原图提取多尺度细粒度特征
        if self.use_cnn:
            detail_feats = self.spatial_prior(raw_image)
            # 确保细节特征与语义特征尺度匹配
            fused_feats = []
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                if detail_feat.shape[2:] != sem_feat.shape[2:]:
                    detail_feat = F.interpolate(
                        detail_feat, size=sem_feat.shape[2:],
                        mode='bilinear', align_corners=False,
                    )
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        # 3. 1×1 Conv + BN 融合
        outputs = []
        for i, (feat, conv, norm) in enumerate(zip(fused_feats, self.convs, self.norms)):
            out = norm(conv(feat))
            outputs.append(out)

        return outputs


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码模块,用于时间步编码"""

    def __init__(self, dim: int):
        """
        初始化正弦位置编码
        dim: 编码维度
        """
        super().__init__()
        self.dim = dim  # 位置编码的维度

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: [batch_size] (float or int)
        return: [batch_size, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2

        # exp(-log(10000) * i / (half_dim - 1))
        exponent = -math.log(10000) / (half_dim - 1)
        i = torch.arange(half_dim, device=device, dtype=timesteps.dtype)
        freqs = torch.exp(i * exponent)

        # args = timesteps[:, None] * freqs[None, :]  # 元素乘法, 自动广播
        args = torch.outer(timesteps, freqs)  # 效率更高
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return emb


if __name__ == '__main__':
    x = torch.randn((1, 3, 256, 256))
    x = x.permute(0, 2, 3, 1).reshape(1, -1, 3)
    attn = nn.MultiheadAttention(3, 3)
    out = attn(x, x, x)

    print(out[0].shape)
    print(out[1].shape)
