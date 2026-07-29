"""TimeStepBlock — FiLM 时间步调制模块

从原仓库 (MBadran2000/DiffuDETR) 的 bbox_embedd.py 移植:
  - TimeStepBlock: FiLM (Feature-wise Linear Modulation) 调制 + 残差

关键设计决策:
  - use_scale_shift_norm=True: emb_layers 输出 2*out_ch, 拆分为 (scale, shift)
  - 调制: h = x * (1 + scale) + shift  (FiLM 核心)
  - 残差: return x + h  (保留原始特征信息)
  - 输入 time_embed 维度 = 4*embed_dim (1024), 输出 out_channels (256)
  - emb_layers: SiLU → Linear(emb_ch → 2*out_ch)

与原仓库差异:
  - 原仓库使用 ldm.util.linear (带 zero_module 的 Linear), 这里用标准 nn.Linear
    (zero_module 对扩散模型输出初始化为 0 有意义, 但 TimeStepBlock 的 emb_layers
    非最终输出层, 标准 init 即可)
"""

from typing import Optional

import torch
import torch.nn as nn


class TimeStepBlock(nn.Module):
    """FiLM 时间步调制块 (Feature-wise Linear Modulation).

    对齐原仓库 bbox_embedd.TimeStepBlock:
      1. emb_out = emb_layers(time_embed)  # SiLU → Linear → [B, 2*out_ch]
      2. 拆分 emb_out → (scale, shift)
      3. h = x * (1 + scale) + shift       # FiLM 调制
      4. return x + h                       # 残差连接

    用于在 transformer decoder 每层注入时间步信息.

    Args:
        channels: 输入特征通道数 (保留参数, 原仓库有但未实际使用).
        emb_channels: 时间步嵌入输入维度 (4*embed_dim = 1024).
        out_channels: 调制输出通道数 (embed_dim = 256).
        dims: 卷积维度 (保留参数, 原仓库支持 1D/2D/3D, 这里只用 1D 即逐元素).
        dropout: dropout 概率 (保留参数).
        use_scale_shift_norm: 是否使用 scale-shift (FiLM), False 则只加 shift.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int = 256,
        dims: int = 1,
        dropout: float = 0.2,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        # emb_layers: SiLU → Linear(emb_ch → 2*out_ch if scale_shift else out_ch)
        # 对齐原仓库: 将时间步嵌入投影到调制参数
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm
                else self.out_channels,
            ),
        )

    def forward(
        self, x: torch.Tensor, time_embed: torch.Tensor
    ) -> torch.Tensor:
        """FiLM 调制前向.

        Args:
            x: [B, N, C] 输入特征 (decoder query 特征).
            time_embed: [B, emb_ch] 时间步嵌入.

        Returns:
            [B, N, C] 调制后特征 (x + FiLM(x, time_embed)).
        """
        # [B, emb_ch] → [B, 2*out_ch]
        emb_out = self.emb_layers(time_embed).type(x.dtype)
        # 广播到 x 的维度: [B, 2*out_ch] → [B, 1, 2*out_ch]
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[:, None]

        if self.use_scale_shift_norm:
            # FiLM: 拆分为 scale 和 shift, 逐元素调制
            scale, shift = emb_out.chunk(2, dim=-1)
            h = x * (1 + scale) + shift
        else:
            # 仅 shift 模制
            h = x + emb_out

        # 残差连接: 保留原始特征, 叠加时间步调制
        return x + h


class BBoxEmbed(nn.Module):
    """边界框预测头 (带 TimeStepBlock, 对齐原仓库 BBoxEmbed).

    原仓库中 TimeStepBlock 在 forward 中被注释掉, 这里保留接口但默认不启用,
    时间步注入主要在 decoder 层完成.

    Args:
        embed_dim: 特征维度 (256).
        time_embed_channels: 时间步嵌入维度 (4*embed_dim = 1024).
    """

    def __init__(self, embed_dim: int, time_embed_channels: int):
        super().__init__()
        from .transformer import MLP

        self.pred = MLP(embed_dim, embed_dim, 4, 3)
        self.norm = nn.LayerNorm(embed_dim)
        self.time_step_embed = TimeStepBlock(
            channels=embed_dim,
            emb_channels=time_embed_channels,
        )

    def forward(
        self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """预测边界框偏移.

        Args:
            x: [B, N, embed_dim] decoder 输出特征.
            time_embed: 未使用 (对齐原仓库, 时间步注入在 decoder 层完成).

        Returns:
            [B, N, 4] 预测框偏移.
        """
        # 对齐原仓库: time_step_embed 被注释, 直接 pred
        x = self.pred(x)
        return x


class ClassEmbed(nn.Module):
    """分类预测头 (带 TimeStepBlock, 对齐原仓库 ClassEmbed).

    与 BBoxEmbed 不同, ClassEmbed 在 forward 中启用了 TimeStepBlock,
    在分类前注入时间步信息.

    Args:
        embed_dim: 特征维度 (256).
        time_embed_channels: 时间步嵌入维度 (4*embed_dim = 1024).
        num_classes: 类别数 (不含背景).
    """

    def __init__(
        self, embed_dim: int, time_embed_channels: int, num_classes: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pred = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)
        self.time_step_embed = TimeStepBlock(
            channels=embed_dim,
            emb_channels=4 * embed_dim,
        )

    def forward(
        self, x: torch.Tensor, time_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """预测分类 logits.

        Args:
            x: [B, N, embed_dim] decoder 输出特征.
            time_embed: [B, emb_ch] 时间步嵌入, 启用时注入.

        Returns:
            [B, N, num_classes] 分类 logits.
        """
        if time_embed is not None:
            x = self.time_step_embed(x, time_embed)
        x = self.pred(x)
        return x
