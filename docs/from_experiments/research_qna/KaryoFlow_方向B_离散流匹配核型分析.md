# KaryoFlow: 基于离散流匹配的端到端染色体核型分析

## 一、动机与定位

### 1.1 为什么不能继续做纯检测

| 方法 | 染色体 mAP | FPS | 状态 |
|------|-----------|-----|------|
| YOLOv8-L | ~0.85+ | 80+ | 工程 SOTA |
| DINO-DETR | ~0.82 | 20+ | 学术 SOTA |
| FlowDet (ours) | 0.751 | ~5 | 速度和精度都输 |

**结论**：扩散/流匹配做 bbox 回归没有竞争力。检测这一步用 YOLO 即可。

### 1.2 真正的学术空白在哪

自动核型分析的完整流程：

```
图像 → ①检测 → ②分类 → ③配对 → ④排列 → 核型图
         ✅        ✅       ❌        ❌
       (已解决)   (已解决) (未解决)  (未解决)
```

**③配对**和**④排列**至今没有端到端的深度学习方案。现有系统全部是启发式后处理：
- 先按分类结果分组
- 组内按长度排序
- 手动规则配对

这是一个**受约束的组合优化问题**：
- 输入：46 个检测到的染色体 (bbox + 类别 + 特征)
- 输出：一个**排列矩阵** P ∈ {0,1}^{46×46}，将检测结果映射到标准核型图的 46 个槽位
- 约束：每个槽位恰好一个染色体，每对同源染色体正确配对，按 Denver 分类排列

**这正是离散流匹配 (Discrete Flow Matching) 的理想应用场景。**

### 1.3 核心 Claim

> 我们将染色体核型分析重新表述为**受约束排列生成问题**，
> 并提出 KaryoFlow —— 首个基于离散流匹配的端到端核型分析框架，
> 在一个统一模型中同时完成检测、分类、配对和排列。

---

## 二、理论框架

### 2.1 问题形式化

**定义 1 (核型排列)**：给定一幅中期分裂相图像 I 中检测到的 N 个染色体
$\mathcal{C} = \{c_1, c_2, \ldots, c_N\}$，其中 $c_i = (b_i, f_i)$ 包含边界框 $b_i \in \mathbb{R}^4$
和视觉特征 $f_i \in \mathbb{R}^d$。核型分析的目标是找到一个排列 $\sigma \in S_N$，将每个
染色体映射到标准核型图中的正确位置。

**标准核型图的 46 个槽位** (正常人类)：

```
Slot 1-2:   A1 pair    (大, 中着丝粒)
Slot 3-4:   A2 pair
Slot 5-6:   A3 pair
Slot 7-8:   B4 pair    (大, 亚中着丝粒)
Slot 9-10:  B5 pair
Slot 11-12: C6 pair    (中, 亚中着丝粒)
...
Slot 43-44: G22 pair   (小, 近端着丝粒)
Slot 45:    X
Slot 46:    Y (or X)
```

**定义 2 (排列矩阵)**：排列 $\sigma$ 可表示为双随机矩阵 $P \in \{0,1\}^{N \times N}$，
其中 $P_{ij} = 1$ 表示检测 $c_i$ 被分配到槽位 $j$。

**定义 3 (核型分析 = 受约束排列生成)**：

$$\sigma^* = \arg\min_{\sigma \in S_N} \sum_{i=1}^{N} \text{cost}(c_i, \text{slot}_{\sigma(i)})$$

约束条件：
- 每个槽位恰好分配一个染色体
- 同一对中的两个染色体属于同一类别
- 对内按长度降序排列 (p arm 朝上)

### 2.2 离散流匹配用于排列生成

**灵感来源**：
- Campbell et al. (ICML 2024): 离散状态空间上的生成流
- Gat et al. (NeurIPS 2024 Spotlight): 任意源-目标耦合的离散流匹配
- SymmetricDiffusers (2024): 对称群上的扩散模型

**核心思想**：在排列空间 $S_N$ 上定义一个连续时间马尔可夫链 (CTMC)，
从随机排列 $\sigma_0 \sim \text{Uniform}(S_N)$ 流向正确排列 $\sigma_1 = \sigma^*$。

**数学形式化**：

定义概率速度场 $u_t: S_N \to \mathbb{R}^{S_N}$，其中 $u_t(\sigma' | \sigma)$ 是
从排列 $\sigma$ 转移到 $\sigma'$ 的速率。

流匹配目标：

$$\mathcal{L}_{\text{KaryoFlow}} = \mathbb{E}_{t \sim U[0,1]} \mathbb{E}_{\sigma_t \sim p_t} \left[ D_{\text{KL}}(u_t^{\text{target}}(\cdot | \sigma_t) \| u_t^{\theta}(\cdot | \sigma_t, I)) \right]$$

其中 $u_t^{\text{target}}$ 是条件概率速度场 (已知起点和终点的最优路径)。

### 2.3 实际操作：排列的因式分解表示

直接在 $S_N$ ($N!=46!$ 个元素) 上工作不可行。使用**因式分解表示**
(参考 "Learning Distributions over Permutations", 2025)：

**Lehmer Code 表示**：排列 $\sigma \in S_N$ 可唯一编码为向量
$L = (l_1, l_2, \ldots, l_N)$，其中 $l_i \in \{0, 1, \ldots, N-i\}$。

这将排列生成问题转化为**独立离散变量的联合生成**：
- $l_1 \in \{0, \ldots, 45\}$ (46 个选择)
- $l_2 \in \{0, \ldots, 44\}$ (45 个选择)
- ...
- $l_{46} \in \{0\}$ (1 个选择)

**关键优势**：
1. 每个 $l_i$ 是有限离散变量 → 可用 MDLM/SEDD 的 masking 策略
2. 任何 $(l_1, \ldots, l_N)$ 组合都对应一个合法排列 → 无需额外约束投影
3. 可并行生成所有 $l_i$ → 非自回归

---

## 三、模型架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    KaryoFlow Architecture                        │
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │  Image I  │──→│  Detection   │──→│  N Chromosome Instances  │ │
│  │ (H×W×3)  │   │  (YOLO/DETR) │   │  {(bbox, crop, feat)}   │ │
│  └──────────┘   └──────────────┘   └───────────┬──────────────┘ │
│                                                 │                │
│                                     ┌───────────▼──────────────┐ │
│                                     │  Chromosome Encoder      │ │
│                                     │  (ViT on each crop)      │ │
│                                     │  → f_i ∈ R^d per chrom   │ │
│                                     └───────────┬──────────────┘ │
│                                                 │                │
│  ┌──────────────────────────────────────────────▼──────────────┐ │
│  │              Discrete Flow Matching Module                   │ │
│  │                                                              │ │
│  │  Input: N chromosome features + noisy Lehmer code L_t        │ │
│  │                                                              │ │
│  │  ┌─────────────────────────────────────────────────────┐    │ │
│  │  │  Transformer Decoder (Bidirectional)                 │    │ │
│  │  │                                                      │    │ │
│  │  │  Query: Lehmer code embeddings (N tokens)            │    │ │
│  │  │  Key/Value: Chromosome features (N tokens)           │    │ │
│  │  │  + Time embedding (AdaLN-Zero)                       │    │ │
│  │  │  + Slot template embeddings (prior knowledge)        │    │ │
│  │  │                                                      │    │ │
│  │  │  Output: denoised probability over each l_i          │    │ │
│  │  └─────────────────────────────────────────────────────┘    │ │
│  │                                                              │ │
│  │  Training: Masked Discrete FM (MDLM-style)                  │ │
│  │    t~U[0,1], mask each l_i with prob t                       │ │
│  │    Loss = Σ_i CE(predicted l_i, true l_i) for masked i      │ │
│  │                                                              │ │
│  │  Inference: Iterative unmasking (few steps)                  │ │
│  │    Start: all [MASK] → gradually reveal → final permutation  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Output: σ* (permutation) → Karyotype                           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 模块详解

#### A. 检测模块 (Stage 1 — 冻结)

直接用 YOLOv8 或当前最好的检测器，**不做训练**。输出 46 个染色体检测。

```python
class ChromosomeDetector:
    """冻结的检测器，仅提供 bbox 和 crop"""
    def __init__(self, checkpoint="yolov8l_chromo.pth"):
        self.model = load_yolo(checkpoint)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, image):
        # Returns: List[(bbox, crop, confidence)]
        detections = self.model(image)
        crops = [crop_and_resize(image, det.bbox, size=64) for det in detections]
        return detections, crops
```

#### B. 染色体编码器 (Stage 2)

将每个染色体 crop 编码为特征向量。

```python
class ChromosomeEncoder(nn.Module):
    """轻量 ViT 编码器，将 64×64 crop → d 维特征"""
    def __init__(self, d_model=256, patch_size=8, num_layers=4):
        super().__init__()
        self.patch_embed = PatchEmbed(64, patch_size, 3, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=num_layers,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, crops):
        # crops: (N, 3, 64, 64) — N个染色体
        patches = self.patch_embed(crops)  # (N, num_patches, d)
        cls = self.cls_token.expand(crops.size(0), -1, -1)
        x = torch.cat([cls, patches], dim=1)
        x = self.transformer(x)
        return x[:, 0]  # (N, d) — 每个染色体的全局特征
```

#### C. 离散流匹配核型模块 (Stage 3 — 核心)

```python
class KaryoFlowModule(nn.Module):
    """离散流匹配排列生成器

    将 N 个染色体特征映射到 Lehmer code 表示的排列。
    使用 MDLM 风格的 masking + 去噪。
    """
    def __init__(self, d_model=256, num_layers=6, num_slots=46):
        super().__init__()
        self.num_slots = num_slots  # 核型图中的槽位数

        # Lehmer code 每个位置的词表大小 (l_i ∈ {0, ..., N-i})
        # 简化: 统一用 num_slots 大小的词表，无效位置 mask 掉
        self.lehmer_embed = nn.Embedding(num_slots + 1, d_model)  # +1 for [MASK]
        self.mask_token_id = num_slots

        # 槽位模板嵌入 (编码先验知识: 每个槽位期望什么类型的染色体)
        self.slot_embed = nn.Embedding(num_slots, d_model)

        # 时间嵌入 (AdaLN-Zero)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
        )

        # Transformer 解码器 (双向注意力)
        self.layers = nn.ModuleList([
            KaryoFlowDecoderLayer(d_model, nhead=8)
            for _ in range(num_layers)
        ])

        # 输出头: 预测每个位置的 Lehmer code 值
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, num_slots - i) for i in range(num_slots)
        ])

    def forward(self, chrom_features, lehmer_noisy, t):
        """
        Args:
            chrom_features: (B, N, d) 染色体特征
            lehmer_noisy: (B, N) 带噪声的 Lehmer code (部分为 [MASK])
            t: (B,) 时间步

        Returns:
            logits: list of (B, vocab_i) — 每个位置的预测分布
        """
        B, N, d = chrom_features.shape

        # 嵌入 Lehmer code
        lehmer_emb = self.lehmer_embed(lehmer_noisy)  # (B, N, d)

        # 融合: 染色体特征 + Lehmer 嵌入 + 槽位先验
        slot_ids = torch.arange(N, device=chrom_features.device)
        slot_emb = self.slot_embed(slot_ids).unsqueeze(0)  # (1, N, d)

        query = lehmer_emb + slot_emb  # (B, N, d)
        memory = chrom_features        # (B, N, d)

        # 时间条件化
        time_emb = self.time_mlp(t)  # (B, 4d)

        for layer in self.layers:
            query = layer(query, memory, time_emb)

        # 分位置预测
        logits = []
        for i in range(N):
            logits_i = self.output_heads[i](query[:, i])  # (B, N-i)
            logits.append(logits_i)

        return logits


class KaryoFlowDecoderLayer(nn.Module):
    """带 AdaLN-Zero 的双向解码层"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        # AdaLN-Zero: 6 组调制参数
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 6),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query, memory, time_emb):
        # AdaLN 参数
        params = self.adaln(time_emb).unsqueeze(1)  # (B, 1, 6d)
        g1, b1, a1, g2, b2, a2 = params.chunk(6, dim=-1)

        # Self-Attention (排列 tokens 之间交互)
        q = self.norm1(query) * (1 + g1) + b1
        sa_out, _ = self.self_attn(q, q, q)
        query = query + a1 * sa_out

        # Cross-Attention (排列 tokens 关注染色体特征)
        q = self.norm2(query) * (1 + g2) + b2
        ca_out, _ = self.cross_attn(q, memory, memory)
        query = query + a2 * ca_out

        # FFN
        query = query + self.ffn(self.norm3(query))

        return query
```

### 3.3 训练流程

```python
class KaryoFlowTrainer:
    """MDLM 风格的训练: 随机 mask Lehmer code → 预测被 mask 的位置"""

    def train_step(self, image, gt_lehmer_code):
        """
        gt_lehmer_code: (B, 46) — 真值 Lehmer code 表示的排列
        """
        # 1. 检测 + 编码 (冻结)
        detections, crops = self.detector(image)
        chrom_features = self.encoder(crops)  # (B, N, d)

        # 2. 随机 mask
        t = torch.rand(B, device=device)  # (B,)
        mask = torch.rand(B, N, device=device) < t.unsqueeze(1)  # (B, N) bool
        lehmer_noisy = gt_lehmer_code.clone()
        lehmer_noisy[mask] = self.model.mask_token_id

        # 3. 前向预测
        logits = self.model(chrom_features, lehmer_noisy, t)

        # 4. 计算 loss (仅在 masked 位置)
        loss = 0
        num_masked = 0
        for i in range(N):
            if mask[:, i].any():
                target_i = gt_lehmer_code[:, i][mask[:, i]]
                pred_i = logits[i][mask[:, i]]
                loss += F.cross_entropy(pred_i, target_i, reduction='sum')
                num_masked += mask[:, i].sum()

        loss = loss / max(num_masked, 1)
        return loss

    @torch.no_grad()
    def inference(self, image, num_steps=10):
        """迭代去噪推理"""
        detections, crops = self.detector(image)
        chrom_features = self.encoder(crops)
        B, N = 1, len(detections)

        # 初始化: 全部 masked
        lehmer = torch.full((B, N), self.model.mask_token_id, device=device)

        for step in range(num_steps):
            t = torch.tensor([(num_steps - step) / num_steps], device=device)
            logits = self.model(chrom_features, lehmer, t)

            # 对每个 masked 位置，取概率最高的值
            confidence = []
            for i in range(N):
                if lehmer[0, i] == self.model.mask_token_id:
                    probs = F.softmax(logits[i][0], dim=-1)
                    conf, pred = probs.max(dim=-1)
                    confidence.append((conf.item(), i, pred.item()))

            # 按置信度从高到低 unmask
            confidence.sort(reverse=True)
            num_unmask = max(1, len(confidence) // num_steps)
            for _, pos, val in confidence[:num_unmask]:
                lehmer[0, pos] = val

        # Lehmer code → 排列
        permutation = lehmer_to_permutation(lehmer[0])
        return permutation
```

---

## 四、数据准备

### 4.1 从检测标注生成排列标注

当前 COCO 标注只有 (bbox, class_id)。需要生成 Lehmer code 标注。

```python
def generate_lehmer_annotation(annotations):
    """
    从 COCO 标注生成 Lehmer code 排列标注

    标准核型图排列规则 (Denver 分类):
    1. 按组排列: A(1-3) → B(4-5) → C(6-12) → D(13-15) → E(16-18) → F(19-20) → G(21-22) → X → Y
    2. 组内按类别编号排列
    3. 同类别两条按长度降序排列 (较大的在左)

    Input: List of {bbox, category_id} for one image
    Output: Lehmer code (N,) — 排列的因式分解表示
    """
    # 按标准顺序排列 slot
    SLOT_ORDER = [
        'A1', 'A1', 'A2', 'A2', 'A3', 'A3',           # Group A
        'B4', 'B4', 'B5', 'B5',                         # Group B
        'C6', 'C6', 'C7', 'C7', 'C8', 'C8', 'C9', 'C9',
        'C10', 'C10', 'C11', 'C11', 'C12', 'C12',      # Group C
        'D13', 'D13', 'D14', 'D14', 'D15', 'D15',      # Group D
        'E16', 'E16', 'E17', 'E17', 'E18', 'E18',      # Group E
        'F19', 'F19', 'F20', 'F20',                     # Group F
        'G21', 'G21', 'G22', 'G22',                     # Group G
        'X', 'Y'                                          # Sex
    ]

    # 1. 按类别分组
    groups = defaultdict(list)
    for ann in annotations:
        cls_name = id_to_name[ann['category_id']]
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]  # w * h
        groups[cls_name].append((ann, area))

    # 2. 组内按面积降序 (大的在前)
    for cls_name in groups:
        groups[cls_name].sort(key=lambda x: -x[1])

    # 3. 按标准顺序生成排列
    sorted_anns = []
    for slot_class in SLOT_ORDER:
        if groups[slot_class]:
            ann, area = groups[slot_class].pop(0)
            sorted_anns.append(ann)

    # 4. 生成排列: sorted_anns[i] 应该放在 slot i
    # 原始顺序 → 标准顺序的映射
    original_order = list(range(len(annotations)))
    permutation = [annotations.index(ann) for ann in sorted_anns]

    # 5. 排列 → Lehmer code
    lehmer = permutation_to_lehmer(permutation)
    return lehmer
```

### 4.2 数据增强

```python
# 排列空间的数据增强:
# 1. 同源对交换 (swap pair): 每对中的两条染色体顺序互换
#    → 产生合法的新排列 (大小相近的同源染色体无确定顺序)
# 2. 检测噪声: 随机微扰 bbox → 模拟检测器的不完美输出
# 3. 缺失/多余处理: 随机删除 1-2 个检测或添加假检测
```

---

## 五、与现有工作的对比

### 5.1 与现有核型分析系统

| | KaryoXpert | GRC-Net | Integral System | **KaryoFlow (ours)** |
|---|---|---|---|---|
| 检测 | DL | DL | DL | YOLO (冻结) |
| 分类 | Metric Learning | Multi-task | DL | **隐式 (排列预测包含分类)** |
| 配对 | 规则 | 规则 | 规则 | **离散流匹配 (学习)** |
| 排列 | 规则 | 规则 | 规则 | **离散流匹配 (学习)** |
| 端到端 | ❌ | ❌ | ❌ | **✅** |
| 不确定性 | ❌ | ❌ | ❌ | **✅ (生成模型天然支持)** |

### 5.2 与扩散/流匹配方法

| | DiffusionDet | FlowDet | DIFUSCO | **KaryoFlow (ours)** |
|---|---|---|---|---|
| 域 | 连续 (bbox) | 连续 (bbox) | 离散 (0/1) | **离散 (排列)** |
| 任务 | 检测 | 检测 | TSP/CO | **核型排列** |
| 优势 | 生成式检测 | 快速推理 | 并行优化 | **端到端结构化预测** |
| 约束处理 | 无 | 无 | 后处理 | **天然 (Lehmer code)** |

---

## 六、实验设计

### 6.1 评价指标

```python
# 1. 排列准确率 (Permutation Accuracy)
#    整个排列完全正确的比例
perm_acc = (predicted_perm == gt_perm).all(dim=1).float().mean()

# 2. 位置准确率 (Position Accuracy)
#    每个染色体被分配到正确位置的比例
pos_acc = (predicted_perm == gt_perm).float().mean()

# 3. 配对准确率 (Pairing Accuracy)
#    同源对正确配对的比例 (不考虑对内顺序)
pair_acc = compute_pair_accuracy(predicted_perm, gt_perm)

# 4. 组准确率 (Group Accuracy)
#    按 Denver 分组 (A-G, Sex) 的分组正确率
group_acc = compute_group_accuracy(predicted_perm, gt_perm)

# 5. Kendall Tau 距离
#    排列之间的距离 (归一化到 [0, 1])
kendall_tau = kendalltau(predicted_perm, gt_perm)
```

### 6.2 消融实验

| 实验 | 说明 |
|------|------|
| Random baseline | 随机排列 |
| Greedy matching | 按分类置信度贪心分配 |
| Hungarian matching | 最优匹配 (非学习) |
| AR 排列生成 | 自回归逐位置生成 Lehmer code |
| **KaryoFlow (MDLM)** | 离散流匹配 (masked) |
| **KaryoFlow (SEDD)** | 离散流匹配 (score entropy) |
| KaryoFlow + OT coupling | 加 OT 训练耦合 |

### 6.3 推理步数 vs 准确率

| Steps | 位置准确率 | 推理时间 |
|-------|-----------|---------|
| 1 | ? | ~5ms |
| 4 | ? | ~20ms |
| 10 | ? | ~50ms |
| 46 (AR) | ? | ~100ms |

---

## 七、实现路线图

### Phase 0: 数据准备 (1 周)

```
tasks:
  - [ ] 编写 Lehmer code 标注生成脚本 (从 COCO 标注)
  - [ ] 验证标注正确性 (可视化核型图)
  - [ ] 数据增强: 同源对交换
  - [ ] 处理异常: N≠46 的图像
```

### Phase 1: 检测 + 编码 (1 周)

```
tasks:
  - [ ] 训练/选用最好的检测器 (YOLO 或现有模型)
  - [ ] 实现 ChromosomeEncoder (ViT on crops)
  - [ ] 冻结检测器，微调编码器
  - [ ] 验证: 分类准确率 (应 > 95%)
```

### Phase 2: KaryoFlow 核心 (2 周)

```
tasks:
  - [ ] 实现 KaryoFlowModule (Transformer + AdaLN-Zero)
  - [ ] 实现 MDLM 训练循环 (mask + predict)
  - [ ] 实现迭代去噪推理
  - [ ] 实现 Lehmer code ↔ 排列 转换
  - [ ] 验证: 在小数据集上过拟合
```

### Phase 3: 实验 + 优化 (2 周)

```
tasks:
  - [ ] 完整数据集训练
  - [ ] 消融: MDLM vs SEDD vs AR
  - [ ] 消融: 推理步数 vs 准确率
  - [ ] 对比: KaryoFlow vs 规则方法 vs Hungarian
  - [ ] 可视化: 去噪过程动画
```

### Phase 4: 论文 (2 周)

```
tasks:
  - [ ] 完整实验表格
  - [ ] 可视化图表
  - [ ] 写作: MICCAI / NeurIPS 格式
```

---

## 八、论文结构

```
Title: KaryoFlow: Discrete Flow Matching for End-to-End Chromosome Karyotyping

Abstract (15 lines)

1. Introduction
   - 核型分析的临床重要性
   - 现有方法: pipeline 分离, 配对/排列靠规则
   - 我们的洞察: 核型分析 = 受约束排列生成
   - KaryoFlow: 首个离散流匹配端到端核型分析

2. Related Work
   - 2.1 自动核型分析 (KaryoXpert, GRC-Net, ...)
   - 2.2 离散扩散/流匹配 (MDLM, SEDD, Campbell, Gat)
   - 2.3 排列学习 (SymmetricDiffusers, Sinkhorn)

3. Method
   - 3.1 问题形式化: 核型分析 = 排列生成
   - 3.2 Lehmer Code 因式分解
   - 3.3 KaryoFlow 架构
   - 3.4 MDLM 训练 + 迭代推理

4. Experiments
   - 4.1 数据集和评价指标
   - 4.2 主实验: KaryoFlow vs baselines
   - 4.3 消融: 架构/训练策略/推理步数
   - 4.4 不确定性估计 (临床应用)
   - 4.5 可视化

5. Discussion & Limitations

6. Conclusion
```

---

## 九、代码复用

从 LDMDet 可以直接复用的组件:

| 组件 | 来源 | 用途 |
|------|------|------|
| AdaLN-Zero | `single_head.py` | KaryoFlowDecoderLayer |
| SinusoidalPositionEmbeddings | `modules.py` | 时间嵌入 |
| Sinkhorn | `sinkhorn.py` | (可选) OT 耦合 |
| MMDet 框架 | `model.py` | 数据加载、训练循环 |
| 检测模型 | 现有 checkpoint | 冻结的检测器 |

需要新实现的:
- `karyoflow/lehmer.py`: Lehmer code 编码/解码
- `karyoflow/encoder.py`: 染色体 ViT 编码器
- `karyoflow/flow_module.py`: 离散流匹配排列生成
- `karyoflow/trainer.py`: MDLM 训练循环
- `karyoflow/dataset.py`: 排列标注数据集
- `tools/generate_lehmer_annotations.py`: 标注生成

---

## 十、风险与缓解

| 风险 | 概率 | 缓解策略 |
|------|------|---------|
| N≠46 (异常核型) | 低 | 支持可变 N; 添加 padding slot |
| 检测器漏检/误检 | 中 | 训练时随机 drop/add 检测作为数据增强 |
| 排列空间太大 (46!) | 中 | Lehmer code 因式分解 + MDLM 并行生成 |
| 同组内类别难区分 | 高 | ChromosomeEncoder 需要足够强; 加对比学习 |
| 审稿人认为问题太小 | 中 | 强调通用性: 框架可迁移到其他组合优化问题 |
| 数据量不够 | 中 | 1540 张图 × 数据增强 (同源对交换) → ~50K 排列 |

---

## 十一、为什么这个方向值得做

1. **学术新颖性**: 首个将离散流匹配应用于医学图像结构化预测的工作
2. **实际价值**: 核型分析是遗传学诊断的刚需，现有系统配对/排列步骤仍需人工
3. **理论深度**: Lehmer code + MDLM 的组合提供了优雅的数学框架
4. **可扩展性**: 框架可直接迁移到:
   - 蛋白质组装 (离散结构预测)
   - 文档版面排列 (页面布局生成)
   - 调度问题 (受约束排列优化)
5. **与热点结合**: 离散扩散/流匹配是 2024-2025 最热的方向之一
6. **不与 YOLO 竞争**: 检测用 YOLO，KaryoFlow 做 YOLO 做不了的事
