import torch
import torch.nn as nn
from typing import List

class STVAEDownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, tempc_kernel_size=3):
        super().__init__()
        # 时间维度卷积（只对时间轴做卷积，保持空间维度不变）
        self.temporal_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(tempc_kernel_size, 1, 1),  # 时间核大小3，空间核1x1
            padding=(tempc_kernel_size//2, 0, 0)    # 时间维度Padding，保持T不变
        )
        # 空间维度卷积（只对空间轴做卷积，保持时间维度不变）
        self.spatial_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3),  # 空间核3x3，时间核1
            padding=(0, 1, 1)       # 空间Padding，保持H/W不变
        )
        # 归一化与激活
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act = nn.SiLU()
        
        # 下采样层（通过步长实现时空压缩）
        self.downsample = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),  # 时间步长2（T/2），空间步长2（H/2, W/2）
            padding=1
        )
        
        # 残差连接的通道匹配（若输入输出通道不同）
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: [B, in_channels, T, H, W]
        residual = self.residual_conv(x)  # 残差分支：[B, out_channels, T, H, W]
        
        # 时空分离卷积
        x = self.temporal_conv(x)  # 时间卷积：[B, in_channels, T, H, W]（T不变）
        x = self.spatial_conv(x)   # 空间卷积：[B, out_channels, T, H, W]（H/W不变）
        x = self.norm(x)
        x = self.act(x)
        
        # 下采样（时空压缩）
        x = self.downsample(x)  # [B, out_channels, T/2, H/2, W/2]
        
        # 残差连接（注意：残差需先下采样以匹配维度）
        residual = nn.functional.interpolate(
            residual, 
            size=x.shape[2:],  # 匹配下采样后的T/2, H/2, W/2
            mode='trilinear', 
            align_corners=False
        )
        return x + residual  # [B, out_channels, T/2, H/2, W/2]

class STVAEUpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, tempc_kernel_size=3):
        super().__init__()
        # 上采样层（恢复时空维度）
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),  # 时间×2，空间×2
            padding=1,
            output_padding=1
        )  # [B, in_channels, T, H, W] → [B, in_channels, T×2, H×2, W×2]
        
        # 时空分离卷积（同下采样块）
        self.temporal_conv = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=(tempc_kernel_size, 1, 1),
            padding=(tempc_kernel_size//2, 0, 0)
        )
        self.spatial_conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.act = nn.SiLU()
        
        # 残差连接的通道匹配
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: [B, in_channels, T, H, W]
        # 上采样（恢复时空维度）
        x = self.upsample(x)  # [B, in_channels, T×2, H×2, W×2]
        
        residual = self.residual_conv(x)  # 残差分支：[B, out_channels, T×2, H×2, W×2]
        
        # 时空分离卷积
        x = self.temporal_conv(x)  # [B, in_channels, T×2, H×2, W×2]
        x = self.spatial_conv(x)   # [B, out_channels, T×2, H×2, W×2]
        x = self.norm(x)
        x = self.act(x)
        
        return x + residual  # [B, out_channels, T×2, H×2, W×2]

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, kv_group=4, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_group = kv_group  # 分组数（32头→4组，每组8头共享K/V）
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.kv_group == 0, "num_heads must be divisible by kv_group"
        self.head_dim = hidden_size // num_heads
        
        # 多头注意力（GQA）
        self.norm1 = nn.LayerNorm(hidden_size)  # 注意力前的层归一化
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim*self.kv_group)
        self.v_proj = nn.Linear(hidden_size, self.head_dim*self.kv_group)
        # MLP
        self.norm2 = nn.LayerNorm(hidden_size)  # MLP前的层归一化
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),  # 扩展4倍
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)   # 压缩回原维度
        )

    def forward(self, x, q_rot=None, k_rot=None):
        # x: [B, seq_len, hidden_size]
        B, N, C = x.shape
        residual = x
        xn = self.norm1(x)
        # 生成 Q/K/V
        q = self.q_proj(xn).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]
        k = self.k_proj(xn).view(B, N, self.kv_group, self.head_dim).permute(0, 2, 1, 3)   # [B, g, N, d]
        v = self.v_proj(xn).view(B, N, self.kv_group, self.head_dim).permute(0, 2, 1, 3)   # [B, g, N, d]
        # 可选：应用旋转位置编码（RoPE），应基于投影后的 q/k
        if q_rot is not None:
            q = q_rot  # 期望形状 [B, h, N, d]
        if k_rot is not None:
            k = k_rot  # 期望形状 [B, g, N, d]
        # 分组查询：每组K/V对应 (h/g) 个Q头
        q = q.view(B, self.kv_group, self.num_heads // self.kv_group, N, self.head_dim)  # [B, g, h/g, N, d]
        k = k.unsqueeze(2)  # [B, g, 1, N, d]
        v = v.unsqueeze(2)  # [B, g, 1, N, d]
        # 注意力
        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, g, h/g, N, N]
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).view(B, self.num_heads, N, self.head_dim).transpose(1, 2).contiguous().view(B, N, C)
        x = residual + attn_output
        # MLP 残差
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class RotaryEmbedding3D(nn.Module):
    def __init__(self, dim: List[int], base=10000.0):
        super().__init__()
        self.dim_t, self.dim_h, self.dim_w = dim  # 三个维度的嵌入维度（需为偶数）
        self.base = base
        
        # 计算三个维度的逆频率（控制位置编码周期）
        self.inv_freqs_t = 1.0 / (base ** (torch.arange(0, self.dim_t, 2).float() / self.dim_t))
        self.inv_freqs_h = 1.0 / (base ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        self.inv_freqs_w = 1.0 / (base ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))

    def compute_freqs(self, coords_t, coords_h, coords_w):
        """计算三个维度的正弦/余弦频率（缓存以加速）"""
        # coords_t: [B, T], coords_h: [B, H, W], coords_w: [B, H, W]
        freqs_t = torch.einsum('bt,d->btd', coords_t, self.inv_freqs_t)  # [B, T, dim_t/2]
        freqs_h = torch.einsum('bhw,d->bhwd', coords_h, self.inv_freqs_h)  # [B, H, W, dim_h/2]
        freqs_w = torch.einsum('bhw,d->bhwd', coords_w, self.inv_freqs_w)  # [B, H, W, dim_w/2]
        
        # 扩展为复数形式（sin和cos作为实部和虚部）
        freqs_t = torch.stack([freqs_t.sin(), freqs_t.cos()], dim=-1).view(*freqs_t.shape[:2], -1)  # [B, T, dim_t]
        freqs_h = torch.stack([freqs_h.sin(), freqs_h.cos()], dim=-1).view(*freqs_h.shape[:3], -1)  # [B, H, W, dim_h]
        freqs_w = torch.stack([freqs_w.sin(), freqs_w.cos()], dim=-1).view(*freqs_w.shape[:3], -1)  # [B, H, W, dim_w]
        return freqs_t, freqs_h, freqs_w

    def apply_rotary_emb(self, q, k, freqs):
        """将频率应用到Q和K（通过旋转矩阵）"""
        freqs_t, freqs_h, freqs_w = freqs
        
        # 1. 时间维度旋转
        q_t, q_rest = q.split(self.dim_t, dim=-1)  # Q拆分为时间部分和剩余部分
        k_t, k_rest = k.split(self.dim_t, dim=-1)
        q_t_rot = self.rotate(q_t, freqs_t)  # 旋转时间部分
        k_t_rot = self.rotate(k_t, freqs_t)
        
        # 2. 高度维度旋转
        q_h, q_rest = q_rest.split(self.dim_h, dim=-1)
        k_h, k_rest = k_rest.split(self.dim_h, dim=-1)
        q_h_rot = self.rotate(q_h, freqs_h)  # 旋转高度部分
        k_h_rot = self.rotate(k_h, freqs_h)
        
        # 3. 宽度维度旋转
        q_w, q_rest = q_rest.split(self.dim_w, dim=-1)
        k_w, k_rest = k_rest.split(self.dim_w, dim=-1)
        q_w_rot = self.rotate(q_w, freqs_w)  # 旋转宽度部分
        k_w_rot = self.rotate(k_w, freqs_w)
        
        # 拼接旋转后的部分
        q_rot = torch.cat([q_t_rot, q_h_rot, q_w_rot, q_rest], dim=-1)
        k_rot = torch.cat([k_t_rot, k_h_rot, k_w_rot, k_rest], dim=-1)
        return q_rot, k_rot

    @staticmethod
    def rotate(x, freqs):
        """旋转操作：x * cosθ - y * sinθ, x * sinθ + y * cosθ"""
        x1, x2 = x.chunk(2, dim=-1)  # 按最后一维拆分为两半
        cos, sin = freqs.chunk(2, dim=-1)  # 频率拆分为cos和sin
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # 旋转后拼接
    
if __name__ == "__main__":
    block = DiTBlock(
        256,
        8
    )
    dummy_input = torch.randn(2, 1000, 256)
    out = block(dummy_input)