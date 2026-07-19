# Diffusion Transformers (DiT) 详解

## 1. 什么是 DiT？

**DiT (Diffusion Transformers)** 是由伯克利大学的 William Peebles 和纽约大学的谢赛宁（Saining Xie）在 2023 年提出的架构。
它的核心贡献是：**彻底抛弃了扩散模型中统治已久的 U-Net，改用 Transformer 作为去噪主干网络。**

如果你关注最近的生成 AI 新闻，你会发现：

- **Sora** (OpenAI 的视频生成模型) 的核心架构是 DiT。
- **Stable Diffusion 3 (SD3)** 转向了 DiT。
- **Flux.1** 使用的是 DiT。

______________________________________________________________________

## 2. 为什么要抛弃 U-Net？

长期以来，[DDPM](DDPM推导.md) 和 [LDM](Latent_Diffusion详解.md) 都使用带有注意力机制的 U-Net。但 U-Net 有几个局限性：

1. **缩放性 (Scalability) 差：** 当你增加参数量时，U-Net 的性能提升并不像 Transformer 那样遵循“缩放定律 (Scaling Laws)”。
2. **灵活性差：** U-Net 的归纳偏置（如卷积的局部性）虽然在图像早期很有用，但在处理更复杂、更高维的数据（如视频）时，反而成了束缚。

**DiT 的逻辑：** 直接用 Transformer 处理图像 Patch，享受 Transformer 强大的全局建模能力和完美的扩展性。

______________________________________________________________________

## 3. DiT 的核心流程

DiT 的工作流可以概括为：**Latent Space + Patchify + Transformer Blocks**。

### 3.1 Patchify (切片化)

DiT 并不直接处理像素，也不直接处理 Latent 矩阵。它将 [VAE](Latent_Diffusion详解.md) 输出的 Latent 特征图（例如 $32 \\times 32$）切成一个个小方块（Patch）。

- 每个 Patch 被线性映射为一个向量（Token）。
- 这与 ViT (Vision Transformer) 的处理方式完全一致。

### 3.2 融入时间与条件

在扩散模型中，模型必须知道当前处于哪个时间步 $t$。DiT 引入了 **adaLN-Zero (Adaptive Layer Norm)**。

______________________________________________________________________

## 4. 核心组件：adaLN-Zero

这是 DiT 论文中最精妙的设计。
传统的 Transformer 使用普通的 Layer Norm，而 DiT 改造了它：

1. **计算调制参数：** 根据时间 $t$ 和条件 $c$（如文本嵌入），通过一个 MLP 预测出 6 个参数：$\\gamma, \\beta, \\alpha$（针对缩放和平移）。
2. **应用调制：**
   $$adaLN(x, t, c) = \\gamma(t,c) \\cdot LN(x) + \\beta(t,c)$$
3. **Zero 初始化：** 在初始化时，将 MLP 的最后一层设为 0，这使得模型在训练初期表现得像个恒等函数，极大地增强了训练稳定性。

______________________________________________________________________

## 5. DiT 的四种变体

DiT 论文探讨了四种融入条件的方式，最终证明 **adaLN-Zero** 效果最好：

- **In-context conditioning:** 把条件 $c$ 当作额外的 Token 拼在后面（类似视觉 Token）。
- **Cross-attention:** 像 LDM 那样使用交叉注意力（SDXL 的做法）。
- **Adaptive Layer Norm (adaLN):** 使用调制参数。
- **adaLN-Zero:** 在 adaLN 基础上增加零初始化技巧。

______________________________________________________________________

## 6. 为什么 DiT 引发了视频生成的革命？

视频可以看作是“带时间轴的图像”。

- **U-Net 处理视频：** 需要复杂的 3D 卷积或交替的空域/时域注意力，极其臃肿。
- **DiT 处理视频：** 只需要把视频切成 3D 的小方块（Spatiotemporal Patches），丢进 Transformer 就行了。Transformer 不在乎你的 Token 是来自图像还是视频，它只管计算 Token 之间的关系。

这就是为什么 **Sora** 能够生成长达一分钟的高质量视频的原因。

______________________________________________________________________

## 7. 简易代码实现 (架构逻辑)

```python
import torch
import torch.nn as nn

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        # 用于预测 adaLN-Zero 参数的 MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, c):
        # c 是由时间 t 和 label/text 融合后的特征
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # 1. Attention 部分 (带调制和门控)
        x_norm = self.norm1(x)
        x_mod = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. MLP 部分 (带调制和门控)
        x_norm = self.norm2(x)
        x_mod = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_mod)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x
```

______________________________________________________________________

## 8. 总结

DiT 的出现标志着 **CV (计算机视觉) 与 NLP (自然语言处理) 在架构上的彻底大一统**。
现在，不管是对话（GPT）、图像生成（SD3/Flux）、视频生成（Sora），底层代码几乎都是一模一样的 Transformer Blocks。

**学习建议：**
理解了 DiT 之后，你可以去看看 [Flow Matching](Flow_Matching详解.md)。因为现在的顶级模型（如 Flux.1）就是 **DiT 架构 + Flow Matching 训练目标** 的完美结合。
