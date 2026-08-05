# 方向 2: Geometric Relation Attention (GeoRelAttn) — 几何关系感知注意力

> **方向类别**: 中等激进 (部分重构 / 增强现有模块)
> **目标**: 在 cascade head 的 self_attn 中注入 4D 相对几何偏置, 使提案间注意力感知染色体空间布局, 抑制重叠染色体框漂移, 同时兼容 SDPA 加速与 box_renewal 机制
> **当前 SOTA 基线**: KaryoFlow +DPM-Solver++ (RF, mAP=0.863, 24obj, checkpoint best_epoch_117)
> **预期增益**: +0.005 ~ +0.015 mAP (主攻 G21/Y 等 mAP 瓶颈类别)
> **文档状态**: 设计完成, 待 TDD 实施
> **创建日期**: 2026-07-27
>
> **核心约束** (来自项目 memory):
> - 必须与染色体检测任务特性深度结合 (密集排列 + 重叠 + 24 类细粒度)
> - 不能与 [FALSIFIED_DIRECTIONS.md](../FALSIFIED_DIRECTIONS.md) 中已证伪方向重复
> - 文献依据需经 WebSearch 验证 (arxiv ID / 标题 / 作者 / 发表状态)
> - 论文中使用 "KaryoFlow" 名称, 内部代号 "LDMDet" 不出现于论文

---

## 1. 动机: 实测诊断暴露的 self_attn 缺陷

### 1.1 D5 实测: self_attn 在关键 cascade head 上近乎均匀

来自 [STRUCTURAL_IMPROVEMENT_ANALYSIS.md §零·D5](./STRUCTURAL_IMPROVEMENT_ANALYSIS.md) 的 PyTorch forward hooks 提取的真实注意力熵:

| Cascade Head | entropy | topk10_cov | 判读 |
|--------------|---------|------------|------|
| head0 | 6.084 | 0.075 | 近乎均匀 (max=log(500)=6.21) |
| head1 | 5.444 | 0.249 | 中度发散 |
| head2 | 5.028 | 0.375 | 最聚焦 |
| head3 | 4.973 | 0.392 | 最聚焦 |
| head4 | 5.544 | 0.198 | 中度发散 |
| head5 | 5.387 | 0.218 | 中度发散 |

**关键发现**: head0 (第一级, 处理纯噪声框) 的注意力熵 6.084 接近最大值 6.21, top-10 覆盖率仅 7.5% (均匀分布应为 10/500=2%), 说明 self_attn 在处理噪声 proposals 时**没有学到任何有意义的提案关系**。head2-3 虽有聚焦 (top-10 覆盖 37-39%), 但仍相当发散。

### 1.2 染色体检测的空间结构先验未被利用

当前 self_attn 是标准 `nn.MultiheadAttention`, query/key/value 均为 256 维 proposal 特征, **完全不包含提案的空间位置信息**。这导致:

- **重叠染色体的提案无法"协商"像素归属**: 两条染色体的 RoI 有共享像素, 标准 self_attn 让提案交互但没有空间信息, 无法识别"哪些提案在物理上接近/重叠"。
- **box_renewal 后的新框无空间记忆**: 低置信度框被替换为 randn, 新框的空间位置完全随机, self_attn 无法利用空间邻域关系快速恢复有意义的内容。
- **C 组 7 条亚中着丝粒染色体空间布局被忽略**: C6-C12 形态相似, 但在中期相铺展上有典型的空间排布 (避免缠绕), 标准 self_attn 无法利用此布局先验。

### 1.3 与已失败/在研方向的正交性

| 已有方向 | 与 GeoRelAttn 的关系 |
|---------|---------------------|
| [FALSIFIED §六 Decoupled Head](../FALSIFIED_DIRECTIONS.md) | **正交** — Decoupled Head 解耦 cls/reg 路径, GeoRelAttn 增强 self_attn 内部, 不改 cls/reg 分支 |
| [FALSIFIED §一 ScaleConditionedRF](../FALSIFIED_DIRECTIONS.md) | **正交** — SCRF 修改噪声调度, GeoRelAttn 修改注意力, 不动 RF 路径 |
| [archived/形态感知分类](./archived/形态感知分类.md) | **互补** — 方向C 在 RoI 内部做形态注意力, GeoRelAttn 在提案间做几何注意力, 两者作用层次不同 |
| [archived/解耦分类定位](./archived/解耦分类定位.md) | **正交** — 方向B 解耦分支, GeoRelAttn 增强共享路径上的注意力 |
| [TODO §四 速度引导自适应 Renewal](../TODO_DIRECTIONS.md) | **正交** — VGAR 修改 box_renewal, GeoRelAttn 修改 self_attn, 可叠加 |
| [TODO §一 级联头角色分化](../TODO_DIRECTIONS.md) | **互补** — 角色分化调整 deep_supervision 权重, GeoRelAttn 增强单头内部, 可叠加 |

**结论**: GeoRelAttn 与所有已失败/在研方向正交或互补, 不存在重复。

---

## 2. 文献依据 (已 WebSearch 验证)

### 2.1 主参考文献

| 文献 | arxiv | 发表 | 与本方向的关系 |
|------|-------|------|---------------|
| **Relation Network** (Hu et al.) | 1711.08043 | CVPR 2018 | 首次在目标检测中用几何偏置替代位置编码, relation_aware 模块启发本设计 |
| **DETR with Relational Priors** | 2304.08014 | ICCV 2023 | 在 DETR cross-attention 中注入 4D 相对几何偏置, 与本方向公式形式最接近 |
| **LP-DETR** (Liu et al.) | 2407.18762 | arXiv 2024 | 在 query-key 注意力中加入学习式 4D 相对位置偏置 (log-scale), 直接验证 log 几何特征的有效性 |
| **Vision Transformer with Relative Position** (Swin) | 2103.14030 | ICCV 2021 | 相对位置偏置 (RPB) 在视觉 transformer 中的标准做法, 本方向借鉴其加性偏置形式 |

### 2.2 已排除的不当引用 (经 subagent 二号检验)

| 文献 | 排除原因 |
|------|---------|
| ~~URoPE (Universal RoPE)~~ | URoPE 是跨视角几何嵌入 (多视图对齐), 与单图像 proposal 间的相对几何偏置无关, 引用不当 |
| ~~DiffusionDet~~ | DiffusionDet 是本方法的范式祖先, 但其 self_attn 是标准 MHA, 无几何偏置设计, 不能作为几何偏置依据 |

### 2.3 与染色体任务结合的差异化论证

现有几何偏置方法 (Relation Network / LP-DETR / DETR-RP) 均针对 COCO/general detection (K≈7, sparse layout), **未考虑以下染色体检测特性**:

1. **t≈1 时 proposals 是纯噪声框** (RF 起点), 几何偏置需 time-aware 衰减以避免误导 → 本方向引入 `time_proj` 调制 alpha
2. **K≈46 高密度 + 97.8% 图含重叠** (vs COCO K≈7), 几何偏置的边际价值更高, 但重叠框的 log 几何特征可能饱和 → 本方向引入 `geom_encoder` MLP 学习非线性映射
3. **6 级 cascade head 角色分化** (D3 实测: head0 粗定位 reg_std=0.94, head5 精细分类 cls_std=1.41), 不同 head 应有不同 alpha → 本方向支持 per-head alpha 参数

---

## 3. 核心设计

### 3.1 数学形式

标准 MHA 的注意力分数:
$$A_{ij}^{(h)} = \frac{(W_Q^{(h)} q_i)^\top (W_K^{(h)} k_j)}{\sqrt{d_k}}$$

GeoRelAttn 注入 4D 相对几何偏置 $g_{ij}^{(h)}$:
$$A_{ij}^{(h)} = \frac{(W_Q^{(h)} q_i)^\top (W_K^{(h)} k_j)}{\sqrt{d_k}} + \alpha^{(h)}(t) \cdot g_{ij}^{(h)}$$

其中:
- $g_{ij}^{(h)} = \text{MLP}_g^{(h)}(\phi_{ij}) \in \mathbb{R}$ 是 per-head 标量偏置
- $\phi_{ij} \in \mathbb{R}^4$ 是 4D 相对几何特征 (见 §3.2)
- $\alpha^{(h)}(t) \in \mathbb{R}$ 是 time-aware 调制系数 (见 §3.3), 初始化为 0

### 3.2 4D 相对几何特征 $\phi_{ij}$

对两个提案框 $b_i = (x_i, y_i, w_i, h_i)$ 和 $b_j = (x_j, y_j, w_j, h_j)$ (cxcywh 格式, 归一化到 [0,1]):

$$\phi_{ij} = \left[\log\frac{\Delta x_{ij}}{\bar{w}_{ij}}, \log\frac{\Delta y_{ij}}{\bar{h}_{ij}}, \log\frac{w_j}{w_i}, \log\frac{h_j}{h_i}\right] \in \mathbb{R}^4$$

其中:
- $\Delta x_{ij} = x_j - x_i$, $\Delta y_{ij} = y_j - y_i$ (中心坐标差)
- $\bar{w}_{ij} = (w_i + w_j) / 2$, $\bar{h}_{ij} = (h_i + h_j) / 2$ (平均尺寸归一化, 使偏置尺度不变)
- log 变换压缩长尾 (远距离提案对的 $\Delta x$ 可能很大)

**不变性论证**:
- **平移不变性**: $\phi_{ij}$ 仅依赖 $\Delta x, \Delta y$, 整图平移不变 ✓
- **尺度不变性**: 用 $\bar{w}, \bar{h}$ 归一化, 整图缩放不变 ✓ (但 $w_j/w_i, h_j/h_i$ 保留相对尺寸差异)
- **旋转不变性**: ✗ (染色体有方向性, 不需要旋转不变; 方向信息对区分着丝粒位置有用)

### 3.3 Time-aware alpha 调制 $\alpha^{(h)}(t)$

**问题**: t≈1 时 proposals 是纯噪声框 (randn), 此时 $\phi_{ij}$ 的几何信息完全无意义, 强行注入会误导注意力。t→0 时 proposals 接近 GT, 几何偏置才可靠。

**方案**: 引入 time-aware 调制:
$$\alpha^{(h)}(t) = \alpha_0^{(h)} \cdot \sigma(\text{time\_proj}^{(h)}(\text{time\_emb}))$$

其中:
- $\alpha_0^{(h)} \in \mathbb{R}$ 是可学习标量 (per-head), 初始化为 0 (零初始化恒等性, 见 §3.5)
- $\text{time\_proj}^{(h)}: \mathbb{R}^{1024} \to \mathbb{R}$ 是 time_emb → 标量的 MLP
- $\sigma$ 是 sigmoid, 保证 $\alpha^{(h)}(t) \in [0, |\alpha_0^{(h)}|]$

**预期行为**:
- t≈1: $\sigma(\cdot) \approx 0$ (time_proj 学到 t 大时输出大负数), $\alpha \approx 0$, GeoRelAttn 退化为标准 MHA (保护噪声阶段)
- t→0: $\sigma(\cdot) \approx 1$, $\alpha \approx \alpha_0$, 几何偏置完全生效

### 3.4 与 SDPA 兼容性

PyTorch `F.scaled_dot_product_attention` (SDPA) 不支持任意加性偏置 (仅支持 attn_mask 二值掩码)。两种实现路径:

**路径 A (推荐, 兼容 SDPA)**: 当 $\alpha^{(h)}(t) < \epsilon$ (如 0.01), 走标准 SDPA 路径; 否则走显式 attention 计算。由于 t→0 时才生效, 大多数 cascade head (head0-1 处理噪声框) 仍走 SDPA, 仅 head2-5 走显式路径, 速度影响有限。

**路径 B (全显式)**: 始终走 `softmax(QK^T/sqrt(d) + alpha * g)` 显式计算。实现简单, 但放弃 SDPA 加速。

**决策**: 采用路径 A, 保留 SDPA 兼容性。在 §5 实验设计中加入 latency 对比。

### 3.5 零初始化恒等性 (安全保障)

为避免破坏 +DPM-Solver++ checkpoint 的预训练表示, 所有新增参数零初始化:

| 参数 | 初始化 | 训练后行为 |
|------|--------|-----------|
| `alpha_init` | 0.0 | 通过 time_proj 渐进激活 |
| `geom_encoder` 最后一层 weight | zeros | 初始 $g_{ij}=0$, 注意力恒等于标准 MHA |
| `geom_encoder` 最后一层 bias | zeros | 同上 |
| `time_proj` 最后一层 weight | zeros | 初始 $\alpha=0$ |
| `time_proj` 最后一层 bias | zeros | 同上 |

**保证**: 训练步 0 时, GeoRelAttn 的 forward 输出**逐数值等于**标准 MHA 的 forward 输出。从 +DPM-Solver++ checkpoint 微调时, 不会因为架构改变而破坏已有表示。此性质由单元测试 `test_zero_init_equivalent_to_mha` 强制保证 (见 §6.3)。

### 3.6 与 box_renewal 的兼容性

box_renewal 在推理时将低置信度框替换为 randn, 新框的 $\phi_{ij}$ 计算依赖新框的 $(x, y, w, h)$。两种处理:

- **新框作为 j (key)**: $\phi_{ij}$ 中 $b_j$ 是噪声框, $\Delta x, \Delta y$ 可能很大, $\log(\Delta x / \bar{w})$ 可能饱和。但因 $\alpha(t)$ 在 t 大时为 0, 噪声阶段不受影响。
- **新框作为 i (query)**: 同上, 但 query 是主动方, 几何偏置影响该 query 对所有 key 的注意力分布。

**决策**: 不做特殊处理, 依赖 time-aware alpha 自动处理。在 §5 实验中验证 box_renewal 开/关时的 GeoRelAttn 行为。

---

## 4. 代码蓝图 (待 TDD 实施)

> **状态**: 设计完成, 文件尚未创建。以下为完整代码蓝图, 实施时按 TDD red-green-refactor 流程: 先写测试 (§6.3), 再实现, 再重构。

### 4.1 新建 `ldmdet/core/geometric_relation_attention.py`

```python
"""Geometric Relation Attention (GeoRelAttn).

在标准 MultiheadAttention 基础上注入 4D 相对几何偏置, 通过 time-aware alpha
调制处理 t≈1 噪声框问题。零初始化保证训练步 0 与标准 MHA 数值恒等。

文献依据:
- Relation Network (Hu et al., CVPR 2018, arxiv 1711.08043)
- DETR with Relational Priors (ICCV 2023, arxiv 2304.08014)
- LP-DETR (Liu et al., arXiv 2024, arxiv 2407.18762)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricRelationAttention(nn.Module):
    """几何关系感知注意力.

    Args:
        embed_dim: 输入特征维度 (默认 256, 对齐 single_head 的 feat_channels)
        num_heads: 注意力头数 (默认 8)
        geom_dim: 几何特征维度 (默认 4: [log_dx/w_bar, log_dy/h_bar, log_wj/wi, log_hj/hi])
        geom_hidden: 几何 MLP 隐藏层维度 (默认 64)
        alpha_init: alpha_0 初始值 (默认 0.0, 零初始化恒等性)
        time_aware: 是否启用 time-aware alpha 调制 (默认 True)
        sdpa_threshold: alpha 低于此值时走 SDPA 路径 (默认 0.01)

    Forward:
        query: [N, bs, embed_dim] (proposals, 与 nn.MultiheadAttention 接口一致)
        key:   [N, bs, embed_dim] (同 query, 自注意力)
        value: [N, bs, embed_dim] (同 query)
        bboxes: [bs, N, 4] cxcywh 归一化框 (用于计算 phi_ij)
        time_emb: [bs, 1024] 时间嵌入 (用于 time-aware alpha, 可选)

    Returns:
        attn_output: [N, bs, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        geom_dim: int = 4,
        geom_hidden: int = 64,
        alpha_init: float = 0.0,
        time_aware: bool = True,
        sdpa_threshold: float = 0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"
        self.sdpa_threshold = sdpa_threshold

        # 标准 MHA 投影 (与 nn.MultiheadAttention 等价)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 几何偏置 MLP: phi_ij -> per-head scalar bias
        self.geom_encoder = nn.Sequential(
            nn.Linear(geom_dim, geom_hidden),
            nn.ReLU(),
            nn.Linear(geom_hidden, geom_hidden),
            nn.ReLU(),
            nn.Linear(geom_hidden, num_heads),
        )
        # 零初始化最后一层 (恒等性保障)
        nn.init.zeros_(self.geom_encoder[-1].weight)
        nn.init.zeros_(self.geom_encoder[-1].bias)

        # Time-aware alpha 调制
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.time_aware = time_aware
        if time_aware:
            self.time_proj = nn.Sequential(
                nn.Linear(1024, geom_hidden),
                nn.SiLU(),
                nn.Linear(geom_hidden, 1),
            )
            nn.init.zeros_(self.time_proj[-1].weight)
            nn.init.zeros_(self.time_proj[-1].bias)

    def compute_geometric_features(self, bboxes: torch.Tensor) -> torch.Tensor:
        """计算 4D 相对几何特征 phi_ij.

        Args:
            bboxes: [bs, N, 4] cxcywh 归一化框

        Returns:
            phi: [bs, N, N, 4] 相对几何特征
        """
        bs, n, _ = bboxes.shape
        cx, cy, w, h = bboxes.unbind(dim=-1)  # 各 [bs, N]

        # 中心差值 [bs, N, N]
        dx = cx.unsqueeze(2) - cx.unsqueeze(1)  # b_j - b_i
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)

        # 平均尺寸 [bs, N, N]
        w_bar = (w.unsqueeze(2) + w.unsqueeze(1)) / 2
        h_bar = (h.unsqueeze(2) + h.unsqueeze(1)) / 2

        # 尺寸比 [bs, N, N]
        w_ratio = w.unsqueeze(2) / (w.unsqueeze(1) + 1e-6)
        h_ratio = h.unsqueeze(2) / (h.unsqueeze(1) + 1e-6)

        # log 变换 (clamp 防止 log(0) 和过大值)
        eps = 1e-6
        phi = torch.stack([
            torch.log((dx.abs() + eps) / (w_bar + eps)),
            torch.log((dy.abs() + eps) / (h_bar + eps)),
            torch.log(w_ratio.clamp(min=eps)),
            torch.log(h_ratio.clamp(min=eps)),
        ], dim=-1)  # [bs, N, N, 4]

        return phi

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播.

        自注意力模式: query=key=value (cascade head 的标准用法)
        """
        if key is None:
            key = query
        if value is None:
            value = query

        n, bs, _ = query.shape

        # 标准投影
        q = self.q_proj(query).reshape(n, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(key).reshape(n, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value).reshape(n, bs * self.num_heads, self.head_dim).transpose(0, 1)

        # 计算 alpha(t)
        if self.time_aware and time_emb is not None:
            # time_emb: [bs, 1024] -> [bs, 1] -> sigmoid
            alpha_t = torch.sigmoid(self.time_proj(time_emb))  # [bs, 1]
            alpha_t = alpha_t * self.alpha  # 标量广播
        else:
            alpha_t = self.alpha

        # 判断是否走 SDPA 路径 (alpha 接近 0 时)
        max_alpha = alpha_t.abs().max().item() if alpha_t.numel() > 0 else 0.0
        if max_alpha < self.sdpa_threshold or bboxes is None:
            # 标准 SDPA 路径
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = attn_output.transpose(0, 1).reshape(n, bs, self.embed_dim)
            return self.out_proj(attn_output)

        # 显式路径 (含几何偏置)
        # phi: [bs, N, N, 4] -> geom_encoder -> [bs, N, N, num_heads]
        phi = self.compute_geometric_features(bboxes)
        geom_bias = self.geom_encoder(phi)  # [bs, N, N, num_heads]

        # 标准注意力分数 [bs*num_heads, N, N]
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        # 重塑 geom_bias 并加到 attn_scores
        # geom_bias: [bs, N, N, num_heads] -> [num_heads, bs, N, N] -> [bs*num_heads, N, N]
        geom_bias = geom_bias.permute(3, 0, 1, 2).reshape(self.num_heads * bs, n, n)
        # alpha_t: [bs, 1] -> 广播到 [bs*num_heads, 1, 1]
        alpha_t_expanded = alpha_t.unsqueeze(1).repeat(1, self.num_heads).reshape(-1, 1, 1)

        attn_scores = attn_scores + alpha_t_expanded * geom_bias
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).reshape(n, bs, self.embed_dim)
        return self.out_proj(attn_output)
```

### 4.2 修改 `ldmdet/core/single_head.py` 集成点

当前 self_attn 实例化位置 (待实施时定位精确行号):

```python
# 当前 (single_head.py 现状):
self.self_attn = nn.MultiheadAttention(feat_channels, num_heads, ...)

# 修改为:
if use_geom_rel_attn:
    from ldmdet.core.geometric_relation_attention import GeometricRelationAttention
    self.self_attn = GeometricRelationAttention(
        embed_dim=feat_channels,
        num_heads=num_heads,
        geom_dim=geom_dim,
        geom_hidden=geom_hidden,
        alpha_init=geom_alpha_init,
        time_aware=geom_time_aware,
    )
else:
    self.self_attn = nn.MultiheadAttention(feat_channels, num_heads, ...)
```

forward 路径修改 (传递 bboxes 和 time_emb):

```python
# 当前:
attn_output = self.self_attn(proposals, proposals, proposals)[0]

# 修改为:
if isinstance(self.self_attn, GeometricRelationAttention):
    # bboxes: [bs, N, 4] cxcywh 归一化 (从 curr_bboxes 转换)
    bboxes_norm = self._normalize_bboxes(curr_bboxes)  # 新增辅助方法
    attn_output = self.self_attn(
        proposals, proposals, proposals,
        bboxes=bboxes_norm,
        time_emb=time_emb,
    )
else:
    attn_output = self.self_attn(proposals, proposals, proposals)[0]
```

### 4.3 实验配置 `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_geom_rel_attn_24obj.py`

```python
"""GeoRelAttn 实验配置 (从 +DPM-Solver++ checkpoint 微调).

基线: +DPM-Solver++ (RF, mAP=0.863)
策略: 从 +DPM-Solver++ best checkpoint 加载, 仅微调新增的 geom_encoder + alpha + time_proj,
      backbone 和已有 self_attn 投影层保持低 lr 微调, 30 epoch 快速验证.
"""
_base_ = ['./a4_dpm_pp_24obj.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            use_geom_rel_attn=True,
            geom_dim=4,
            geom_hidden=64,
            geom_alpha_init=0.0,
            geom_time_aware=True,
        ),
    ),
)

# 从 +DPM-Solver++ best checkpoint 加载 (新增参数随机/零初始化, 已有参数从 +DPM-Solver++ 加载)
load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'

# 微调超参
max_epochs = 30
optim_wrapper = dict(optimizer=dict(lr=1e-5))  # 比 baseline 5e-5 低 5x, 保护 +DPM-Solver++ 表示
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=3),  # warmup
    dict(type='CosineAnnealingLR', T_max=27, eta_min=1e-6, by_epoch=True, begin=3, end=30),
]

# SwanLab
swanlab = dict(
    project='ldmdet-mainline-ablation-24obj',
    experiment_name='a4_geom_rel_attn',
)
```

---

## 5. 实验设计

### 5.1 主实验: +DPM-Solver++ + GeoRelAttn vs +DPM-Solver++ baseline

| 实验 | 配置 | 训练 | 数据集 | 预期 |
|------|------|------|--------|------|
| +DPM-Solver++ baseline | a4_dpm_pp_24obj | 已完成 (150 ep) | 24obj | mAP=0.863 (3-seed 0.859±0.003) |
| +DPM-Solver++ + GeoRelAttn | a4_geom_rel_attn_24obj | 30 ep 微调 | 24obj | mAP 0.868-0.878 (+0.005~+0.015) |

**成功判据**:
- ✅ mAP ≥ 0.868 (+0.005, 超出 3-seed noise ±0.003)
- ✅ G21/Y per-class AP 提升 ≥ +0.01 (主攻瓶颈类别)
- ✅ box_renewal 开/关时均不退化 (兼容性验证)

### 5.2 Ablation: 分量贡献

| Ablation | geom_encoder | alpha (time-aware) | 预期 |
|----------|--------------|--------------------|---------|
| +DPM-Solver++ baseline | ✗ | ✗ | 0.863 |
| +geom (固定 alpha=1) | ✓ | ✗ (固定 1) | 测试纯几何偏置效果, 验证 time-aware 必要性 |
| +alpha (随机 geom) | ✗ (随机) | ✓ | 测试 time-aware 调制本身的效果 |
| full GeoRelAttn | ✓ | ✓ | 完整设计 |
| full + 不归一化 bboxes | ✓ | ✓ | 用原始像素坐标 (非 [0,1] 归一化), 验证尺度不变性 |

### 5.3 Per-cascade-head 分析

逐 head 分析 alpha_0 学到的值 (训练后):

| Head | 角色 (D3 实测) | 预期 alpha_0 | 解释 |
|------|---------------|-------------|------|
| head0 | 粗定位 (reg_std=0.94, 处理噪声) | 接近 0 | time_proj 应在 t 大时压制 |
| head2-3 | 中级精化 (熵最低, 已聚焦) | 中等 | 几何偏置增强已有聚焦 |
| head5 | 精细分类 (cls_std=1.41) | 中等 | 几何偏置帮助区分 C 组近邻 |

### 5.4 Latency 对比 (路径 A SDPA 兼容性验证)

| 配置 | 推理延迟 (ms/img) | 备注 |
|------|------------------|------|
| +DPM-Solver++ baseline | 155.4 | 已测 |
| +DPM-Solver++ + GeoRelAttn (路径 A) | 预期 < 165 | head0-1 走 SDPA, head2-5 走显式 |
| +DPM-Solver++ + GeoRelAttn (路径 B 全显式) | 预期 170-180 | 全程显式, 放弃 SDPA |

**判据**: 路径 A 延迟增加 ≤ 10ms (≤ 6.4%) 视为可接受。

### 5.5 3-seed 验证 (若 1-seed 成功)

若 +DPM-Solver++ + GeoRelAttn 1-seed mAP ≥ 0.868, 启动 3-seed 完整验证 (seed 42/789/123):
- 3-seed 均值 ≥ 0.865 (vs +DPM-Solver++ 3-seed 0.859)
- 3-seed std ≤ 0.003 (与 +DPM-Solver++ 相当)

---

## 6. TDD 实施计划

### 6.1 Red-Green-Refactor 流程

1. **Red**: 先写 `ldmdet/tests/test_geometric_relation_attention.py`, 全部失败
2. **Green**: 实现 `ldmdet/core/geometric_relation_attention.py`, 测试全过
3. **Refactor**: 集成到 `single_head.py`, 跑 +DPM-Solver++ checkpoint 数值一致性验证

### 6.2 实施步骤 (预估 2-3 天)

| Day | 任务 |
|-----|------|
| 1 上午 | 写测试 (§6.3), 验证 Red |
| 1 下午 | 实现 GeometricRelationAttention, 验证 Green |
| 2 上午 | 集成到 single_head.py, 数值一致性验证 |
| 2 下午 | 写配置 + 跑 1-seed 30 epoch 微调 |
| 3 | 评估 + ablation + 文档化结果 |

### 6.3 单元测试用例 `ldmdet/tests/test_geometric_relation_attention.py`

```python
"""GeoRelAttn 单元测试. TDD Red 阶段: 全部应先失败."""
import torch
import pytest
from ldmdet.core.geometric_relation_attention import GeometricRelationAttention


class TestGeometricRelationAttention:
    def test_zero_init_equivalent_to_mha(self):
        """零初始化时, GeoRelAttn 输出应数值等于标准 MHA (允许 fp32 容差)."""
        torch.manual_seed(42)
        geo_attn = GeometricRelationAttention(embed_dim=256, num_heads=8, alpha_init=0.0)
        geo_attn.eval()

        # 同样初始化 q/k/v/out_proj 与标准 MHA
        mha = torch.nn.MultiheadAttention(256, 8, batch_first=False)
        # ... 复制权重 ...

        query = torch.randn(500, 2, 256)
        bboxes = torch.rand(2, 500, 4)  # cxcywh in [0,1]
        time_emb = torch.randn(2, 1024)

        out_geo = geo_attn(query, bboxes=bboxes, time_emb=time_emb)
        out_mha, _ = mha(query, query, query)
        assert torch.allclose(out_geo, out_mha, atol=1e-5), "零初始化恒等性破坏"

    def test_translation_invariance(self):
        """phi_ij 应平移不变: 整体平移 bboxes 不改变注意力输出."""
        geo_attn = GeometricRelationAttention(embed_dim=256, num_heads=8, alpha_init=1.0)
        geo_attn.eval()
        query = torch.randn(50, 1, 256)
        bboxes1 = torch.rand(1, 50, 4)
        bboxes2 = bboxes1.clone()
        bboxes2[..., :2] += 0.1  # 平移 cx, cy

        out1 = geo_attn(query, bboxes=bboxes1)
        out2 = geo_attn(query, bboxes=bboxes2)
        assert torch.allclose(out1, out2, atol=1e-6), "平移不变性破坏"

    def test_scale_invariance(self):
        """phi_ij 应尺度不变: 整体缩放 bboxes 不改变注意力输出."""
        geo_attn = GeometricRelationAttention(embed_dim=256, num_heads=8, alpha_init=1.0)
        geo_attn.eval()
        query = torch.randn(50, 1, 256)
        bboxes1 = torch.rand(1, 50, 4) * 0.5 + 0.25  # 集中在中心
        bboxes2 = bboxes1 * 2.0  # 整体放大 2x (注意不能超出 [0,1])

        out1 = geo_attn(query, bboxes=bboxes1)
        out2 = geo_attn(query, bboxes=bboxes2)
        assert torch.allclose(out1, out2, atol=1e-6), "尺度不变性破坏"

    def test_time_aware_alpha_zero_at_t1(self):
        """t≈1 时 (噪声框阶段), alpha(t) 应接近 0, 输出接近标准 MHA."""
        geo_attn = GeometricRelationAttention(embed_dim=256, num_heads=8, alpha_init=1.0, time_aware=True)
        geo_attn.eval()
        # ... 模拟 t=1 的 time_emb, 验证 alpha_t ≈ 0 ...

    def test_sdpa_path_when_alpha_small(self):
        """alpha < sdpa_threshold 时, 应走 SDPA 路径 (输出数值等于 SDPA)."""
        # ...

    def test_gradient_flow_to_geom_encoder(self):
        """梯度应回传到 geom_encoder (确保几何偏置可学习)."""
        # ...

    def test_box_renewal_compatibility(self):
        """box_renewal 替换部分框后, GeoRelAttn 应正常 forward 不崩溃."""
        # ...
```

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 几何偏置在 head0 (噪声框) 反而有害 | 中 | time-aware alpha 在 t 大时压制为 0; §5.2 '+geom (固定 alpha=1)' ablation 验证 |
| geom_encoder 过拟合 (24 类小数据) | 低 | geom_hidden=64 限制容量; 30 epoch 微调; 3-seed 验证 |
| SDPA 兼容路径延迟增加 > 10% | 低 | §5.4 latency 对比; 若超 10% 切回路径 B 全显式 |
| +DPM-Solver++ checkpoint 加载后新增参数破坏表示 | 低 | 零初始化恒等性 (§3.5) + 单元测试强制保证 |
| 与 box_renewal 冲突 (新框 phi 饱和) | 低 | log 变换压缩长尾; §6.3 test_box_renewal_compatibility 验证 |
| G21/Y 提升不显著 (瓶颈在别处) | 中 | §5.3 per-head alpha 分析定位瓶颈; 若无效归档为 FALSIFIED |

---

## 8. 与论文叙事的对接

GeoRelAttn 对应论文 §4.4 "架构改进" 的一个子方向, 叙事要点:

1. **任务结合**: "染色体中期相铺展图像 97.8% 含框重叠, 标准 self_attn 在 head0 注意力熵 6.08 (近乎均匀), 表明提案间空间关系未被利用。GeoRelAttn 注入 4D 相对几何偏置, 使注意力感知染色体空间布局。"

2. **理论新颖性**: "不同于 Relation Network / LP-DETR 针对稀疏布局 (COCO K≈7), GeoRelAttn 引入 time-aware alpha 调制, 在 RF 的 t≈1 噪声框阶段自动衰减几何偏置, 解决扩散检测特有的'噪声框几何无意义'问题。"

3. **可扩展性**: "GeoRelAttn 的 4D 相对几何偏置 + time-aware 调制可推广到任何扩散检测任务 (细胞检测、病灶检测), 只要该任务存在密集目标和噪声框阶段。"

---

## 9. 实施触发条件

本方向的实施优先级: **高** (主推方向)。

**触发条件** (满足任一即可启动):
- ReFlow 2-Rectification 重试结果出炉 (无论成功/失败), 释放 ross GPU
- 用户明确指示启动 GeoRelAttn 实施
- M4 (级联头角色分化) 在 workstation 跑完, GPU 空闲

**实施顺序建议**:
1. 先启动 M4 (零代码改动, workstation 0.5 天) 占用 GPU
2. 同步实施 GeoRelAttn TDD (本方向, 2-3 天)
3. GeoRelAttn 1-seed 结果出来后决定是否启动 3-seed

---

## 10. 参考文献

- **Relation Network**: Hu et al., "Relation Networks for Object Detection", CVPR 2018, [arxiv 1711.08043](https://arxiv.org/abs/1711.08043)
- **DETR with Relational Priors**: "DETR with Relational Priors", ICCV 2023, [arxiv 2304.08014](https://arxiv.org/abs/2304.08014)
- **LP-DETR**: Liu et al., "LP-DETR: Layer-wise Position DEtR with Relative Position Bias", arXiv 2024, [arxiv 2407.18762](https://arxiv.org/abs/2407.18762)
- **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021, [arxiv 2103.14030](https://arxiv.org/abs/2103.14030)
- **结构诊断数据**: [STRUCTURAL_IMPROVEMENT_ANALYSIS.md §零·D5](./STRUCTURAL_IMPROVEMENT_ANALYSIS.md)
