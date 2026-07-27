# 方向 1: KaryoSetDiff — 集合级扩散检测 (Set-Level Diffusion Detection)

> **方向类别**: 激进 (完全重构 self_attn 子模块 + 新增 count head 辅助监督)
> **目标**: 用 Set Transformer 的 ISAB 模块替换标准 self_attn, 实现 proposal 集合的诱导点级上下文建模; 新增 PMA count head 作为辅助监督, 显式约束 proposal 集合的基数一致性
> **当前 SOTA 基线**: KaryoFlow A4 (RF + DPM-Solver++, mAP=0.863, 24obj)
> **预期增益**: +0.005 ~ +0.020 mAP (定位为辅助 ablation, 非主推)
> **文档状态**: 设计完成, 待主推 GeoRelAttn 验证后再决定是否实施
> **创建日期**: 2026-07-27
>
> **⚠️ 重要声明**: 本方向经历了从 **KaryoLatent (潜在向量压缩)** 到 **KaryoSetDiff (保留 per-proposal 扩散)** 的降级。原 KaryoLatent 设计存在 5 处致命问题 (见 §2), 已废弃。本文档记录最终 KaryoSetDiff 设计。
>
> **⚠️ 与已失败方向区分**: 本方向**不是** [breakthrough_directions/方向二_计数先验约束的扩散生成](./breakthrough_directions/方向二_计数先验约束的扩散生成.md) 的复活。方向二的核心是"计数约束推理 (路径 B) + 计数分支辅助监督 (路径 C)", 在 Dataset 1 上已失败 (mAP=0.726 vs 0.753)。KaryoSetDiff 的核心是 **ISAB 集合级上下文建模**, PMA count head 仅是 ISAB 的副产品辅助监督, 两者方法学本质不同 (见 §1.4)。

---

## 1. 动机与定位

### 1.1 标准 self_attn 的集合建模局限

KaryoFlow cascade head 的 self_attn 是 `nn.MultiheadAttention(256, 8)`, 在 500 个 proposals 间做全连接注意力。来自 [STRUCTURAL_IMPROVEMENT_ANALYSIS.md §零·D5](./STRUCTURAL_IMPROVEMENT_ANALYSIS.md) 的实测诊断:

- **head0 注意力熵 6.084 (近乎均匀)**: 第一级处理纯噪声框时, self_attn 没有学到有意义的提案关系
- **head2-3 熵 ~5.0, top-10 覆盖 37-39%**: 中级 head 有聚焦但仍发散
- **O(N²) 复杂度**: N=500 proposals, attention matrix 250000 元素, 计算密集

**核心问题**: 标准 self_attn 把 500 个 proposals 视为**无结构的独立 token 集合**, 每个提案独立 query 所有其他提案。但染色体中期相铺展图像中, 46 条染色体的 proposals 集合有**强集合结构**:
- 计数先验 (正常 46 条, 异常 45/47)
- 分组结构 (A/B/C/D/E/F/G 七组, 每组 2-7 条)
- 空间布局 (避免缠绕的典型排布)

标准 self_attn 无法显式建模这种集合级结构。

### 1.2 Set Transformer 的诱导点优势

Set Transformer (Lee et al., ICML 2019) 提出 ISAB (Induced Set Attention Block), 用 M 个**学习式诱导点** (inducing points) 作为中介:

```
标准 self_attn:   N proposals ↔ N proposals        O(N²)
ISAB:             N proposals ↔ M inducing points  O(NM)
                  M inducing points ↔ N proposals  O(NM)
                  (M << N, 如 M=32)
```

**优势**:
1. **复杂度降低**: O(N²) → O(NM), N=500, M=32 时 8x 加速
2. **集合级抽象**: inducing points 学到"典型提案模式", 提供 per-proposal 无法捕获的集合级上下文
3. **基数感知**: inducing points 数量 M 是全局信息, 隐式编码集合规模

### 1.3 与染色体检测任务特性的结合

| 任务特性 | KaryoSetDiff 的应对 |
|---------|---------------------|
| K≈46 高密度 (vs COCO K≈7) | M=32 inducing points 编码 46 个 GT 的典型模式 |
| 24 类细粒度 + 7 组分组 | inducing points 可隐式学到组级别表示 |
| 97.8% 图含重叠 | 集合级上下文帮助提案"协商"像素归属 |
| Y 染色体数据稀缺 (n=397) | 集合级先验降低对单实例特征的依赖 |
| 500 proposals 冗余 | ISAB 的 O(NM) 复杂度更适合大 N |

### 1.4 与已失败方向二的本质区别

| 维度 | 方向二 (已失败) | KaryoSetDiff (本方向) |
|------|----------------|----------------------|
| **核心创新** | 计数约束 NMS (路径 B) + 计数分支辅助 (路径 C) | ISAB 集合级上下文建模 |
| **count head 角色** | 主创新之一 (路径 C 从 FPN 预测 c) | 辅助监督 (PMA head, ISAB 的副产品) |
| **作用位置** | 推理后处理 (B) + FPN 特征 (C) | cascade head 内部 self_attn 替换 |
| **失败原因** | 计数约束与检测质量冲突 + 辅助损失干扰主任务 | 不适用 (方法学不同) |
| **数据集** | Dataset 1 (1540 张, mAP≈0.75) | Dataset 2 (5000 张, mAP≈0.86) |
| **基线** | rf_heun_adaln (0.746) | A4 DPM-Solver++ (0.863) |

**关键区分**: 方向二把计数作为**约束/条件**强加于检测, KaryoSetDiff 把集合结构作为**注意力归纳偏置**融入特征学习, 两者方法学本质不同。方向二的失败不构成 KaryoSetDiff 的反证。

### 1.5 与 GeoRelAttn 的关系

| 维度 | GeoRelAttn (方向 2, 主推) | KaryoSetDiff (方向 1, 辅助) |
|------|--------------------------|----------------------------|
| 改动范围 | self_attn 内部 (加几何偏置) | self_attn 整体替换 (ISAB) |
| 归纳偏置 | 4D 相对几何 (空间布局) | 集合级抽象 (inducing points) |
| 复杂度 | O(N²) (保留全连接) | O(NM) (降低) |
| 风险 | 低 (零初始化恒等) | 中 (架构改变较大) |
| 实施优先级 | **高** (主推) | 低 (待 GeoRelAttn 验证后) |

**关系**: 两者**互斥** (都改 self_attn), 不可叠加。定位为"同一瓶颈 (self_attn 集合建模) 的两种正交解法", 若 GeoRelAttn 成功则 KaryoSetDiff 不必实施; 若 GeoRelAttn 失败, KaryoSetDiff 作为替代方案。

---

## 2. KaryoLatent → KaryoSetDiff 降级记录

### 2.1 原 KaryoLatent 设计 (已废弃)

原设想: 将 500 个 proposals 的 4D 框压缩到 256 维潜在向量, 在潜在空间做扩散, 解码回框集合。灵感来自 LDM (Latent Diffusion, arxiv 2112.10752)。

### 2.2 KaryoLatent 的 5 处致命问题 (subagent 二号评估发现)

| 问题 | 描述 | 严重性 |
|------|------|--------|
| 1. DPM-Solver++ 不兼容 | DPM-Solver++ 要求 data-prediction 形式, 潜在空间扩散的 x0 是潜在向量, 无法直接用框的 DPM-Solver++ 更新公式 | 致命 |
| 2. box_renewal 失效 | box_renewal 依赖单框级操作 (低置信度框替换为 randn), 潜在向量压缩后无法定位"哪个框" | 致命 |
| 3. 解码器训练困难 | 潜在向量 → 500 个框的解码器需要训练, 24 类小数据集难以学到稳定解码 | 严重 |
| 4. OT Coupling 失效 | OT coupling 依赖框级别的 (x0, x1) 配对, 潜在向量级别 OT 无意义 (与论文 §3.3 Stochastic Coupling 核心创新冲突) | 致命 |
| 5. Stochastic Coupling 理论失效 | 论文命题 1-3 基于 per-proposal OT 多样性坍缩分析, 潜在向量压缩使该理论框架不适用 | 致命 |

### 2.3 降级决策: 策略 B (保留 per-proposal 扩散)

**降级原则**: 放弃潜在向量压缩, 保留 per-proposal 扩散框架 (与 DPM-Solver++ / box_renewal / Stochastic Coupling 全部兼容), 仅用 ISAB 替换 self_attn 实现集合级上下文建模。

**降级后的 KaryoSetDiff**:
- ✅ 保留 per-proposal 扩散 (DPM-Solver++ 兼容)
- ✅ 保留 box_renewal (单框级操作不变)
- ✅ 保留 OT Coupling / Stochastic Coupling (per-proposal 配对不变)
- ✅ ISAB 仅替换 self_attn 子模块, 其余架构不变
- ✅ PMA count head 作为辅助监督, 不影响主扩散路径

---

## 3. 文献依据 (已 WebSearch 验证)

### 3.1 主参考文献

| 文献 | arxiv | 发表 | 与本方向的关系 |
|------|-------|------|---------------|
| **Set Transformer** (Lee et al.) | 1810.00825 | ICML 2019 | ISAB + PMA 模块的原始出处, 本方向直接采用其架构 |
| **Point Set Diffusion** | 2410.02560 | ICLR 2025 | 在点集上做扩散, 用 set-level 上下文建模, 启发"集合级扩散检测"思路 |
| **LayoutFlow** | 2407.18041 | ECCV 2024 | 在布局元素集 (框集) 上做 flow matching, 验证 set-level flow 的可行性 |
| **LDM** (Rombach et al.) | 2112.10752 | CVPR 2022 | 潜在扩散的原始出处, 本方向原 KaryoLatent 的灵感来源 (降级后仅参考其"分阶段训练"思路) |

### 3.2 文献修正记录

| 原引用 | 问题 | 修正 |
|--------|------|------|
| ~~LDM arxiv 2112.05259~~ | arxiv ID 错误 | 修正为 **2112.10752** |
| ~~Point Set Diffusion (2024 未确认发表)~~ | 发表状态未确认 | 已确认 **ICLR 2025 接收** (WebSearch 验证) |

### 3.3 与 Point Set Diffusion / LayoutFlow 的差异化

| 文献 | 任务 | 与 KaryoSetDiff 的区别 |
|------|------|----------------------|
| Point Set Diffusion | 点集生成 (无类别) | KaryoSetDiff 是**检测** (有类别 cls + 框 reg), 且保留 per-proposal 扩散而非全集合扩散 |
| LayoutFlow | 布局元素 flow matching | LayoutFlow 在**完整布局集**上做 flow, KaryoSetDiff 保留 per-proposal RF, 仅在 attention 层引入集合上下文 |
| Set Transformer | 集合输入的通用 transformer | 本方向将其 ISAB/PMA 模块适配到扩散检测 cascade head |

---

## 4. 核心设计

### 4.1 整体架构 (KaryoSetDiff 替换 self_attn)

```
原 cascade head 流程:
  proposals → [self_attn (MHA)] → +residual → norm1 → [DynamicConv] → [FFN] → ...

KaryoSetDiff 流程:
  proposals → [ISAB (Induced Set Attention Block)] → +residual → norm1 → [DynamicConv] → [FFN] → ...
                    │
                    └─ 内部:
                       proposals ↔ M inducing points ↔ proposals
                       (PMA count head 从最终 inducing points 预测集合基数)
```

### 4.2 ISAB (Induced Set Attention Block) 数学形式

给定 proposal 集合 $X \in \mathbb{R}^{N \times d}$ (N=500, d=256) 和 M 个学习式 inducing points $I \in \mathbb{R}^{M \times d}$ (M=32):

**Step 1**: inducing points 作为 query, proposals 作为 key/value
$$H = \text{MHA}(Q=I, K=X, V=X) \in \mathbb{R}^{M \times d}$$

**Step 2**: proposals 作为 query, H 作为 key/value
$$Y = \text{MHA}(Q=X, K=H, V=H) \in \mathbb{R}^{N \times d}$$

**复杂度**: O(NMd) + O(M²d) + O(NMd) = O(NMd), 当 M=32, N=500 时约 8x 加速 vs O(N²d)。

### 4.3 PMA count head (辅助监督)

PMA (Pooling by Multihead Attention) 从 inducing points 池化出集合级表示, 用于预测染色体计数:

$$z = \text{PMA}(H) = \text{MHA}(Q=s, K=H, V=H) \in \mathbb{R}^{d}$$

其中 $s \in \mathbb{R}^{d}$ 是学习式 seed vector。然后:

$$\hat{c} = \text{MLP}_{count}(z) \in \mathbb{R}^{1}$$

**训练损失**:
$$\mathcal{L}_{count} = |\hat{c} - c_{gt}|_1$$

其中 $c_{gt}$ 是该图的 GT 染色体数 (通常 46, 异常 45/47)。$\mathcal{L}_{count}$ 作为辅助损失, 权重 $\lambda_{count} = 0.1$ (小权重, 避免干扰主任务)。

**与方向二的关键区别**:
- 方向二: count head 从 **FPN 特征**预测 c, 独立于检测头
- KaryoSetDiff: count head 从 **ISAB 的 inducing points**预测 c, 是集合级上下文建模的副产品, 与检测共享表示

### 4.4 零初始化恒等性 (安全保障)

为兼容 A4 checkpoint 微调, ISAB 的输出投影层零初始化, 保证训练步 0 时:
$$\text{ISAB}(X) \approx \text{MHA}(X) \text{ (在零初始化近似下)}$$

具体:
- ISAB 内部两个 MHA 的 out_proj 零初始化
- PMA count head 的 MLP 最后一层零初始化 ($\hat{c}=0$ 初始)
- $\lambda_{count}$ 从 0 渐进 ramp-up 到 0.1 (前 5 epoch 为 0)

**注意**: 严格数值恒等性较难保证 (ISAB 有 inducing points 中介), 改为"零初始化下不破坏 A4 表示"的弱保证, 由单元测试 `test_zero_init_preserves_a4_representation` 验证 (前向输出 L2 距离 < 阈值)。

### 4.5 与 box_renewal / DPM-Solver++ 的兼容性

| 机制 | 兼容性 | 说明 |
|------|--------|------|
| box_renewal | ✅ 完全兼容 | ISAB 输入输出都是 [N, d] proposal 序列, 单框级 box_renewal 在 ISAB 外部操作 |
| DPM-Solver++ | ✅ 完全兼容 | DPM-Solver++ 作用于 x_t 序列 (per-proposal), ISAB 作用于 proposal 特征, 两者正交 |
| Stochastic Coupling | ✅ 完全兼容 | Coupling 在 (x0, x1) 配对层操作, ISAB 在特征层操作 |
| deep_supervision | ✅ 完全兼容 | 各 cascade head 的 ISAB 独立, deep_supervision 权重不变 |

---

## 5. 代码蓝图 (待实施)

### 5.1 新建 `ldmdet/core/set_attention.py`

```python
"""Set-level Attention 模块 (ISAB + PMA).

文献依据:
- Set Transformer (Lee et al., ICML 2019, arxiv 1810.00825)
- Point Set Diffusion (ICLR 2025, arxiv 2410.02560)
- LayoutFlow (ECCV 2024, arxiv 2407.18041)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InducedSetAttentionBlock(nn.Module):
    """ISAB: 用 M 个 inducing points 作为中介的集合注意力.

    复杂度 O(NMd) vs 标准 MHA 的 O(N²d), M << N 时显著加速.

    Args:
        embed_dim: 特征维度 (默认 256)
        num_heads: 注意力头数 (默认 8)
        num_inducing: inducing points 数量 (默认 32)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_inducing: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_inducing = num_inducing

        # 学习式 inducing points
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, embed_dim))
        nn.init.xavier_uniform_(self.inducing_points)

        # 两个 MHA: I→X 和 X→H
        self.mha_ix = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.mha_xh = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # 零初始化 out_proj (弱恒等性保障)
        nn.init.zeros_(self.mha_ix.out_proj.weight)
        nn.init.zeros_(self.mha_ix.out_proj.bias)
        nn.init.zeros_(self.mha_xh.out_proj.weight)
        nn.init.zeros_(self.mha_xh.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, bs, embed_dim] proposals (与 nn.MultiheadAttention 接口一致)

        Returns:
            y: [N, bs, embed_dim]
        """
        n, bs, d = x.shape

        # 扩展 inducing points 到 batch 维度: [M, bs, d]
        inducing = self.inducing_points.unsqueeze(1).expand(self.num_inducing, bs, d)

        # Step 1: I 作为 query, X 作为 key/value -> H [M, bs, d]
        h, _ = self.mha_ix(inducing, x, x)

        # Step 2: X 作为 query, H 作为 key/value -> Y [N, bs, d]
        y, _ = self.mha_xh(x, h, h)

        return y


class PMACountHead(nn.Module):
    """PMA count head: 从 inducing points 池化出集合表示, 预测染色体计数.

    作为 ISAB 的副产品辅助监督, 与方向二 (从 FPN 预测) 方法学不同.

    Args:
        embed_dim: 特征维度 (默认 256)
        num_heads: 注意力头数 (默认 8)
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        # 学习式 seed vector
        self.seed = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.seed)

        self.pma = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.count_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        # 零初始化 count_mlp 最后一层 (初始 c_hat=0)
        nn.init.zeros_(self.count_mlp[-1].weight)
        nn.init.zeros_(self.count_mlp[-1].bias)

    def forward(self, inducing_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inducing_output: [M, bs, embed_dim] ISAB Step 1 输出

        Returns:
            count_pred: [bs, 1] 预测染色体计数
        """
        m, bs, d = inducing_output.shape
        seed = self.seed.expand(1, bs, d)  # [1, bs, d]

        # PMA: seed 作为 query, inducing 作为 key/value
        z, _ = self.pma(seed, inducing_output, inducing_output)  # [1, bs, d]
        z = z.squeeze(0)  # [bs, d]

        count_pred = self.count_mlp(z)  # [bs, 1]
        return count_pred
```

### 5.2 修改 `ldmdet/core/single_head.py` 集成

```python
# 当前:
self.self_attn = nn.MultiheadAttention(feat_channels, num_heads, ...)

# 修改为:
if use_set_attention:
    from ldmdet.core.set_attention import InducedSetAttentionBlock, PMACountHead
    self.self_attn = InducedSetAttentionBlock(
        embed_dim=feat_channels,
        num_heads=num_heads,
        num_inducing=num_inducing,  # 默认 32
    )
    if use_count_head:
        self.count_head = PMACountHead(embed_dim=feat_channels, num_heads=num_heads)
else:
    self.self_attn = nn.MultiheadAttention(feat_channels, num_heads, ...)
```

forward 路径 (返回 count_pred 用于辅助损失):

```python
# self_attn 调用
if isinstance(self.self_attn, InducedSetAttentionBlock):
    # ISAB 内部需要先算 inducing_output 给 count_head
    # 修改 ISAB.forward 使其可选返回 inducing_output
    attn_output, inducing_output = self.self_attn(proposals, return_inducing=True)
    count_pred = self.count_head(inducing_output) if self.use_count_head else None
else:
    attn_output = self.self_attn(proposals, proposals, proposals)[0]
    count_pred = None
```

### 5.3 修改 `ldmdet/core/head.py` 损失计算

```python
# 在 loss() 中新增 count 辅助损失
def loss(self, ...):
    # ... 现有主损失计算 ...

    # count 辅助损失
    if any(r.get('count_pred') is not None for r in all_results):
        count_preds = [r['count_pred'] for r in all_results if r.get('count_pred') is not None]
        # 取最后一级 cascade head 的 count_pred
        count_pred = count_preds[-1]
        # c_gt: 该图的 GT 染色体数 (从 targets 计算)
        c_gt = torch.tensor([len(t['bboxes']) for t in targets], device=count_pred.device, dtype=torch.float32)
        c_gt = c_gt.unsqueeze(1)  # [bs, 1]
        loss_count = F.l1_loss(count_pred, c_gt) * self.count_loss_weight
        losses['loss_count'] = loss_count
    return losses
```

### 5.4 实验配置

```python
"""KaryoSetDiff 实验配置."""
_base_ = ['./a4_dpm_pp_24obj.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            use_set_attention=True,
            num_inducing=32,
            use_count_head=True,
        ),
        count_loss_weight=0.1,  # 辅助损失权重
        count_loss_warmup_epochs=5,  # 前 5 epoch 权重为 0
    ),
)

load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'
max_epochs = 30
optim_wrapper = dict(optimizer=dict(lr=1e-5))

swanlab = dict(
    project='ldmdet-mainline-ablation-24obj',
    experiment_name='a4_setdiff',
)
```

---

## 6. 实验设计

### 6.1 主实验

| 实验 | 配置 | 训练 | 预期 |
|------|------|------|------|
| A4 baseline | a4_dpm_pp_24obj | 已完成 | mAP=0.863 |
| A4 + ISAB only | a4_setdiff (无 count head) | 30 ep 微调 | mAP 0.865-0.875 |
| A4 + ISAB + PMA count | a4_setdiff (含 count head) | 30 ep 微调 | mAP 0.868-0.880 |

### 6.2 Ablation

| Ablation | ISAB | PMA count | num_inducing | 预期 |
|----------|------|-----------|--------------|------|
| A0: baseline | ✗ | ✗ | — | 0.863 |
| A1: ISAB only | ✓ | ✗ | 32 | 测试纯集合上下文效果 |
| A2: ISAB + count | ✓ | ✓ | 32 | 完整设计 |
| A3: num_inducing=16 | ✓ | ✓ | 16 | 诱导点数量扫描 |
| A4: num_inducing=64 | ✓ | ✓ | 64 | 同上 |
| A5: count_loss_weight=0 | ✓ | ✓ (但权重 0) | 32 | count head 是否有害 (与方向二对比) |

### 6.3 与方向二的直接对比

为彻底回应"是否是方向二的复活", 设计直接对比实验:

| 实验 | count head 来源 | count head 权重 | mAP | 说明 |
|------|----------------|----------------|-----|------|
| 方向二 (已失败) | FPN 特征 (独立分支) | 1.0 | 0.726 (Dataset 1) | 计数约束干扰主任务 |
| KaryoSetDiff A5 | ISAB inducing (共享) | 0.0 | 待测 | 权重 0, 等效无 count head |
| KaryoSetDiff A2 | ISAB inducing (共享) | 0.1 | 待测 | 小权重 + 共享表示 |

**判据**: 若 KaryoSetDiff A2 mAP ≥ A4 baseline, 且 A5 (权重 0) ≈ A2, 则 count head 无害且可能有益, 证明与方向二方法学不同。

### 6.4 Latency 对比

| 配置 | 推理延迟 (ms/img) | 加速比 |
|------|------------------|--------|
| A4 baseline (MHA) | 155.4 | 1.0x |
| A4 + ISAB (M=32) | 预期 140-150 | 1.04-1.11x |
| A4 + ISAB (M=64) | 预期 145-155 | 1.0-1.07x |

**预期**: ISAB 应带来轻微加速 (O(N²) → O(NM)), 这是相对 GeoRelAttn (保留 O(N²)) 的一个优势。

---

## 7. TDD 单元测试 (待实施)

```python
class TestInducedSetAttentionBlock:
    def test_output_shape(self):
        """ISAB 输出形状应与输入一致 [N, bs, d]."""

    def test_zero_init_preserves_a4_representation(self):
        """零初始化下, ISAB(X) 与 MHA(X) 的 L2 距离 < 阈值 (弱恒等性)."""

    def test_complexity_reduction(self):
        """ISAB 的 FLOPs 应低于标准 MHA (M < N)."""

    def test_inducing_points_learned(self):
        """inducing points 应在训练后更新 (梯度非零)."""

class TestPMACountHead:
    def test_output_shape(self):
        """count_pred 形状应为 [bs, 1]."""

    def test_zero_init_output_zero(self):
        """零初始化下 count_pred 应为 0."""

    def test_gradient_flow(self):
        """count 损失梯度应回传到 ISAB 的 inducing points."""

class TestIntegrationWithCascadeHead:
    def test_box_renewal_compatibility(self):
        """box_renewal 开启时 ISAB 应正常工作."""

    def test_dpm_solver_compatibility(self):
        """DPM-Solver++ 推理时 ISAB 应正常工作."""

    def test_count_loss_warmup(self):
        """前 5 epoch count_loss_weight 应为 0."""
```

---

## 8. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| ISAB 集合抽象损失细粒度信息 | 中 | num_inducing=32 足够大; Ablation A3/A4 扫描 |
| count head 与方向二同样有害 | 低 | 共享表示 (非独立分支) + 小权重 0.1 + warmup; §6.3 直接对比 |
| 零初始化弱恒等性破坏 A4 表示 | 中 | §4.4 弱保证 + 单元测试; 30 ep 微调可恢复 |
| inducing points 在 24 类小数据上过拟合 | 中 | num_inducing 限制容量; 3-seed 验证 |
| 与 GeoRelAttn 互斥, 实施机会成本 | 高 | 定位为 GeoRelAttn 的替代方案, 待主推验证后再决定 |

---

## 9. 实施触发条件

本方向的实施优先级: **低** (辅助 ablation, 待 GeoRelAttn 验证后再决定)。

**触发条件** (满足任一):
- GeoRelAttn 实施后 mAP 无提升 (< +0.005), 需要替代方案
- GeoRelAttn 实施后 latency 增加 > 10%, 需要复杂度更低的替代
- 用户明确指示启动 KaryoSetDiff

**不实施条件**:
- GeoRelAttn 实施后 mAP ≥ 0.868 且 latency 可接受 → KaryoSetDiff 不必实施, 仅作为论文 §6 未来工作提及

---

## 10. 与论文叙事的对接

若实施成功, KaryoSetDiff 对应论文 §4.4 "架构改进" 的集合级上下文建模子方向, 叙事要点:

1. **任务结合**: "染色体检测的 K≈46 高密度 (vs COCO K≈7) 使标准 self_attn 的 O(N²) 全连接注意力难以学到集合级结构。KaryoSetDiff 用 Set Transformer 的 ISAB 模块, 通过 M=32 学习式 inducing points 中介, 实现集合级上下文建模。"

2. **理论新颖性**: "不同于 Point Set Diffusion / LayoutFlow 在完整集合上做扩散, KaryoSetDiff 保留 per-proposal RF 扩散 (与 DPM-Solver++ / Stochastic Coupling 兼容), 仅在 attention 层引入集合归纳偏置。"

3. **与方向二的区分**: "PMA count head 作为 ISAB 的副产品辅助监督 (共享表示, 小权重 0.1), 与独立的计数约束 (方向二, 已失败) 方法学本质不同。"

4. **可扩展性**: "ISAB 的 O(NM) 复杂度更适合高密度检测任务 (细胞计数、遥感密集目标), M 可根据任务 K 调整。"

---

## 11. 参考文献

- **Set Transformer**: Lee et al., "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks", ICML 2019, [arxiv 1810.00825](https://arxiv.org/abs/1810.00825)
- **Point Set Diffusion**: "Point Set Diffusion for Object Detection", ICLR 2025, [arxiv 2410.02560](https://arxiv.org/abs/2410.02560)
- **LayoutFlow**: "LayoutFlow: Rectified Flow for Layout Generation", ECCV 2024, [arxiv 2407.18041](https://arxiv.org/abs/2407.18041)
- **LDM**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022, [arxiv 2112.10752](https://arxiv.org/abs/2112.10752)
- **结构诊断数据**: [STRUCTURAL_IMPROVEMENT_ANALYSIS.md §零·D5](./STRUCTURAL_IMPROVEMENT_ANALYSIS.md)
- **已失败方向二**: [breakthrough_directions/方向二_计数先验约束的扩散生成](./breakthrough_directions/方向二_计数先验约束的扩散生成.md)
