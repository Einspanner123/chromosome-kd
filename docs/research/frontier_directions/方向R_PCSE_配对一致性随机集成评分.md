# 方向 R：PCSE 配对一致性随机集成评分 (Pairwise Consistency Stochastic Ensemble)

> **核心思想**: 利用扩散采样的随机性生成 K 组候选检测假设, 用核型先验 (配对一致性) 对假设集合评分, 选择核型最合法的假设作为最终输出。纯推理期优化, 无需重训练。
>
> **理论依据**:
> - 多假设检验 (MHT): Reid, "An Algorithm for Tracking Multiple Targets" (IEEE TAC 1979)
> - 贝叶斯模型平均: Hoeting et al. (Statistical Science 1999)
> - 扩散采样随机性: DDPM (NeurIPS 2020), Rectified Flow (ICLR 2023)
> - 染色体核型学: ISCN 国际标准, Denver 分组
>
> **当前代码位置**:
> - [ldmdet/diffusion/sampling.py](../../ldmdet/diffusion/sampling.py) — 采样器与 `post_process` (现有时间维 ensemble)
> - [ldmdet/core/head.py](../../ldmdet/core/head.py) — `predict` 方法 (单次随机初始化采样循环)
> - [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) — RF Euler/Heun 步进
> - [experiments/configs/ldmdet/ldmdet_baseline.py](../../experiments/configs/ldmdet/ldmdet_baseline.py) — `use_ensemble` 配置

---

## 1. 背景与动机

### 1.1 染色体检测的独特优势: 核型约束

染色体检测是**少数存在强全局离散约束**的检测任务。普通目标检测 (COCO) 中, 一张图出现多少个目标、哪些类别组合, 没有固定模式; 但正常人类核型严格满足:

```
正常核型 = 46 条染色体 = 23 对
  ├── 22 对常染色体 (同源对): A1×2, A2×2, A3×2, B4×2, B5×2, ..., G21×2, G22×2
  └── 1 对性染色体: XX (女性) 或 XY (男性)
```

这个约束提供了**远超一般检测任务的先验信息**:
1. **数量先验**: 检测框总数应 ≈ 46 (允许 ±2 容差应对分割/标注噪声)
2. **配对先验**: 同源染色体对大小、形态、带型高度相似
3. **类别先验**: 1-22 类各应出现 2 次, X/Y 出现 1 或 2 次

**当前模型完全未利用这些先验**。`post_process` 仅做 NMS 去重 + 置信度排序, 不检查"是否凑齐 23 对"。

### 1.2 现有 Ensemble 机制的局限

阅读 [sampling.py](../../ldmdet/diffusion/sampling.py) 第 194-264 行的 `post_process` 与 [head.py](../../ldmdet/core/head.py) 第 317-395 行的 `predict` 可知, 当前 `use_ensemble=True` 的实际行为是:

```python
# head.py:324 — 单次随机初始化
x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

# head.py:332-339 — 同一轨迹内, 每个时间步的 (cls_logits, pred_bboxes) 都收集
for step_idx, (t_curr, t_next) in enumerate(time_pairs):
    cls_logits, pred_bboxes, x0_raw, velocity = self._forward_at_t(...)
    if self.use_ensemble:
        ensemble_results.append((cls_logits, pred_bboxes))  # 时间维 ensemble

# sampling.py:209-218 — 所有时间步结果拼接后做一次 NMS
for cls_logits, pred_bboxes in ensemble_results:
    all_scores.append(...); all_bboxes.append(...); all_labels.append(...)
final_scores = torch.cat(all_scores)  # T × num_proposals 个框
# → NMS 去重
```

**本质**: 这是**时间维 ensemble** (temporal ensemble), 聚合同一随机种子、同一轨迹在不同时间步的预测。它的作用类似"多级 cascade 的投票", 但**无法探索不同随机初始化产生的假设空间**。

**局限**:
- 单条轨迹的误差具有**系统性偏差**: 若初始噪声导致某条染色体被某个高置信度框"抢占", 后续时间步难以纠正 (box_renewal 只替换低置信度框, 不改变已收敛的高置信度框)
- NMS 是**局部贪心**操作: 只看 IoU 和分数, 不看全局核型结构。一个"凑不齐 23 对"的预测集合, NMS 不会主动修复
- 1→8 步采样增益递减 (0.740→0.755, +0.015/8× 计算量) 说明: 在同一轨迹上增加步数已接近饱和

### 1.3 PCSE 的核心洞察

扩散采样的随机性是一把双刃剑:
- **坏处**: 单次采样的结果有方差, 可能恰好"采歪了"
- **好处**: 多次采样会覆盖不同的假设空间, 提供选择余地

PCSE 的核心: **不再用 NMS 贪心去重, 而是用核型先验做"裁判", 从 K 个候选假设中选最合法的**。

```
当前:  单次采样 → 时间维 ensemble → NMS → 输出
PCSE:  K 次采样 → 每次独立 NMS → 核型评分 → 选最优假设 → 输出
```

这与人类核型分析师的工作流程一致: 先找到所有染色体, 再尝试配对和排序, 若配不上则回头复查是否有遗漏或误检。

---

## 2. 理论依据

### 2.1 多假设采样理论

扩散检测模型的推理过程可形式化为从后验分布 $p_\theta(\mathbf{Y} | \mathbf{X})$ 中采样, 其中 $\mathbf{Y} = \{(b_i, c_i)\}_{i=1}^{N}$ 是检测集合, $\mathbf{X}$ 是图像。由于扩散采样的随机性 (初始噪声 $z \sim \mathcal{N}(0, I)$ 和 box_renewal 注入的噪声), 不同的随机种子 $z_k$ 会产生不同的 $\mathbf{Y}_k$。

理想情况下, 边缘化随机性可得最优估计:
$$p(\mathbf{Y} | \mathbf{X}) = \int p(\mathbf{Y} | \mathbf{X}, z) p(z) \, dz \approx \frac{1}{K} \sum_{k=1}^{K} p(\mathbf{Y} | \mathbf{X}, z_k)$$

但直接平均 (如现有 NMS 拼接) 会混合不同假设, 破坏每个假设的内部一致性。PCSE 采用**选择**而非**平均**:
$$\mathbf{Y}^* = \arg\max_{k \in \{1, \dots, K\}} p(\mathbf{Y}_k | \mathbf{X})$$

### 2.2 贝叶斯后验选择

由贝叶斯定理:
$$p(\mathbf{Y}_k | \mathbf{X}) \propto p(\mathbf{X} | \mathbf{Y}_k) \cdot p(\mathbf{Y}_k)$$

- **似然项** $p(\mathbf{X} | \mathbf{Y}_k)$: 检测置信度, 已由模型编码。用假设 $\mathbf{Y}_k$ 内所有框的平均/中位置信度 $\bar{s}_k$ 近似
- **先验项** $p(\mathbf{Y}_k)$: **核型先验**, 即该检测集合作为"合法核型"的概率。这是 PCSE 引入的新信息

对数评分函数:
$$\log p(\mathbf{Y}_k | \mathbf{X}) = \underbrace{\log p(\mathbf{X} | \mathbf{Y}_k)}_{\text{似然 (已有)}} + \underbrace{\log p(\mathbf{Y}_k)}_{\text{核型先验 (新增)}}$$

PCSE 的核心贡献: 用一个可计算的**配对一致性评分** $S_{\text{karyo}}(\mathbf{Y}_k)$ 显式估计 $\log p(\mathbf{Y}_k)$。

### 2.3 与现有时间维 Ensemble 的理论区别

| 维度 | 时间维 Ensemble (现有) | PCSE (本方向) |
|------|----------------------|---------------|
| 采样空间 | 同一轨迹不同时间步 $\{t_1, \dots, t_T\}$ | 不同随机种子 $\{z_1, \dots, z_K\}$ |
| 假设关系 | 强相关 (同一轨迹演化) | 近似独立 (不同初始噪声) |
| 聚合方式 | 拼接 + NMS (平均) | 评分 + 选择 (argmax) |
| 利用先验 | 无 (仅 IoU/分数) | 核型先验 (配对一致性) |
| 误差模式 | 系统偏差无法纠正 | 不同假设可互补 |

---

## 3. 详细方案设计

### 3.1 整体流程

```
输入图像 X
    │
    ├──→ 采样 1 (seed=z_1) → NMS → 假设 Y_1 → 评分 S_1
    ├──→ 采样 2 (seed=z_2) → NMS → 假设 Y_2 → 评分 S_2
    ├──→ ...
    └──→ 采样 K (seed=z_K) → NMS → 假设 Y_K → 评分 S_K
                                         │
                          选 argmax S_k → 输出 Y*
```

### 3.2 多假设生成策略

#### 3.2.1 基础策略: 独立多种子采样

对同一张图, 用 K 个不同的随机种子独立运行完整的扩散采样流程:

```python
def pcse_sample(model, features, img_metas, K=5):
    """PCSE 多假设采样"""
    hypotheses = []
    for k in range(K):
        # 关键: 每次用不同的随机种子
        torch.manual_seed(base_seed + k)
        # 复用现有 predict 逻辑, 但禁用时间维 ensemble (每个假设应是独立完整的结果)
        result = model.bbox_head.predict_single_hypothesis(
            features, img_metas, use_ensemble=False
        )
        hypotheses.append(result)
    return hypotheses
```

**实现要点**:
- 现有 `predict` 方法 (head.py:317) 的 `x_raw = torch.randn(...)` 在调用前设置 `torch.manual_seed` 即可控制随机性
- 每个假设内部仍可用 box_renewal (保持与训练一致的随机性)
- 每个假设独立做 NMS (复用 `post_process`)

#### 3.2.2 进阶策略: 种子多样性增强

单纯换种子可能因为 box_renewal 的随机性, 不同假设间差异不够大。可选增强:

**策略 A: 初始噪声扰动幅度**
```python
# 不用纯 randn, 而是对一个基础噪声做扰动
base_noise = torch.randn(bs, num_proposals, 4)
for k in range(K):
    perturbation = sigma_k * torch.randn_like(base_noise)
    x_raw_k = base_noise + perturbation  # sigma_k 控制多样性
```

**策略 B: box_renewal 阈值扰动**
不同假设用不同的 `score_thr` (如 0.3/0.4/0.5/0.6/0.7), 使 box_renewal 保留/替换不同的框子集, 增加假设间多样性。

**策略 C: 时间步数扰动**
不同假设用不同的采样步数 (如 K=5 时用 2/3/4/5/6 步), 利用步数-质量曲线的不同工作点。但需注意: 步数过少会降低单假设质量。

**推荐**: Phase 1 先用基础策略 (独立多种子), Phase 2 再评估是否需要多样性增强。

### 3.3 配对一致性评分函数 (核心)

这是 PCSE 的数学核心。给定一个检测假设 $\mathbf{Y} = \{(b_i, c_i, s_i)\}_{i=1}^{N}$, 定义评分:

$$S_{\text{karyo}}(\mathbf{Y}) = \alpha \cdot S_{\text{count}} + \beta \cdot S_{\text{pair}} + \gamma \cdot S_{\text{morph}} + \delta \cdot S_{\text{sex}}$$

其中 $\alpha, \beta, \gamma, \delta$ 为权重 (Phase 1 用网格搜索确定, 初值 $\alpha=1.0, \beta=2.0, \gamma=1.5, \delta=1.5$)。

#### 3.3.1 数量一致性 $S_{\text{count}}$

染色体总数应接近 46:

$$S_{\text{count}}(\mathbf{Y}) = \exp\left(-\frac{(N - 46)^2}{2 \sigma_c^2}\right), \quad \sigma_c = 2.0$$

- $N = 46$: $S_{\text{count}} = 1.0$ (满分)
- $N = 44$ 或 $48$: $S_{\text{count}} \approx 0.61$
- $N = 42$ 或 $50$: $S_{\text{count}} \approx 0.14$

**实现**:
```python
def score_count(num_detections, target=46, sigma=2.0):
    return math.exp(-((num_detections - target) ** 2) / (2 * sigma ** 2))
```

**复杂度**: $O(1)$

#### 3.3.2 类别配对一致性 $S_{\text{pair}}$

每个常染色体类 (1-22) 应恰好出现 2 次:

$$S_{\text{pair}}(\mathbf{Y}) = \frac{1}{22} \sum_{k=1}^{22} f(n_k), \quad f(n) = \begin{cases} 1.0 & n = 2 \\ 0.5 & n = 1 \text{ 或 } n = 3 \\ 0.0 & n = 0 \text{ 或 } n \geq 4 \end{cases}$$

其中 $n_k = |\{i : c_i = k\}|$ 是类别 $k$ 的检测数。

**实现**:
```python
def score_pair(labels, num_autosome_classes=22):
    counts = torch.bincount(labels, minlength=num_autosome_classes + 2)
    scores = torch.zeros(num_autosome_classes)
    for k in range(num_autosome_classes):
        n = counts[k].item()
        if n == 2:
            scores[k] = 1.0
        elif n in (1, 3):
            scores[k] = 0.5
        else:
            scores[k] = 0.0
    return scores.mean().item()
```

**复杂度**: $O(N)$ (bincount) + $O(22)$ (查表) = $O(N)$

#### 3.3.3 形态配对一致性 $S_{\text{morph}}$

对每个恰好有 2 条检测的常染色体类, 检查这两条染色体的形态相似度。同源染色体应对大小、长宽比相似。

对框 $b = (x_1, y_1, x_2, y_2)$, 定义:
- 面积 $A = (x_2 - x_1) \cdot (y_2 - y_1)$
- 长宽比 $r = \frac{\max(w, h)}{\min(w, h)}$, 其中 $w = x_2 - x_1, h = y_2 - y_1$

对类别 $k$ 的同源对 $(b_i, b_j)$:

$$\text{sim}_{\text{size}}(i, j) = 1 - \frac{|A_i - A_j|}{\max(A_i, A_j)}$$

$$\text{sim}_{\text{aspect}}(i, j) = 1 - \frac{|r_i - r_j|}{\max(r_i, r_j)}$$

$$S_{\text{morph}} = \frac{1}{|P|} \sum_{k \in P} \frac{1}{2} \left( \text{sim}_{\text{size}}(i_k, j_k) + \text{sim}_{\text{aspect}}(i_k, j_k) \right)$$

其中 $P$ 是 $n_k = 2$ 的类别集合, $i_k, j_k$ 是类别 $k$ 的两条检测的索引。若 $|P| = 0$, $S_{\text{morph}} = 0$。

**实现**:
```python
def score_morph(bboxes, labels, num_autosome_classes=22):
    total_sim = 0.0
    num_pairs = 0
    for k in range(num_autosome_classes):
        idx = (labels == k).nonzero(as_tuple=True)[0]
        if len(idx) != 2:
            continue
        b1, b2 = bboxes[idx[0]], bboxes[idx[1]]
        w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
        w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
        A1, A2 = w1 * h1, w2 * h2
        r1 = max(w1, h1) / max(min(w1, h1), 1e-6)
        r2 = max(w2, h2) / max(min(w2, h2), 1e-6)
        sim_size = 1 - abs(A1 - A2) / max(A1, A2)
        sim_aspect = 1 - abs(r1 - r2) / max(r1, r2)
        total_sim += 0.5 * (sim_size + sim_aspect)
        num_pairs += 1
    return total_sim / max(num_pairs, 1)
```

**复杂度**: $O(N)$ (按类分组) + $O(22)$ (对每个类检查至多 1 对) = $O(N)$

#### 3.3.4 性染色体一致性 $S_{\text{sex}}$

性染色体应为 XX (女) 或 XY (男):

$$S_{\text{sex}}(\mathbf{Y}) = \begin{cases} 1.0 & (n_X, n_Y) \in \{(2, 0), (1, 1)\} \\ 0.5 & (n_X, n_Y) \in \{(1, 0), (2, 1), (3, 0)\} \text{ (可能漏检/多检)} \\ 0.0 & \text{其他} \end{cases}$$

**实现**:
```python
def score_sex(labels, x_class_idx=23, y_class_idx=24):
    n_x = (labels == x_class_idx).sum().item()
    n_y = (labels == y_class_idx).sum().item()
    if (n_x, n_y) in [(2, 0), (1, 1)]:
        return 1.0
    elif (n_x, n_y) in [(1, 0), (2, 1), (3, 0)]:
        return 0.5
    return 0.0
```

**复杂度**: $O(N)$

#### 3.3.5 综合评分与似然融合

最终选择函数融合核型先验与检测置信度:

$$S_{\text{total}}(\mathbf{Y}_k) = \lambda \cdot \bar{s}_k + (1 - \lambda) \cdot S_{\text{karyo}}(\mathbf{Y}_k)$$

其中 $\bar{s}_k = \frac{1}{N_k} \sum_i s_i^{(k)}$ 是假设 $k$ 的平均检测置信度, $\lambda \in [0, 1]$ 平衡似然与先验 (初值 $\lambda = 0.5$)。

选择:
$$\mathbf{Y}^* = \arg\max_{k} S_{\text{total}}(\mathbf{Y}_k)$$

**为什么用线性加权而非对数乘积**: 对数乘积 $\log \bar{s} + \log S_{\text{karyo}}$ 在 $S_{\text{karyo}} = 0$ 时数值不稳定, 线性加权更鲁棒且易调参。

### 3.4 选择算法

#### 3.4.1 基础算法: Argmax 选择

```python
def pcse_select(hypotheses, alphas=(1.0, 2.0, 1.5, 1.5), lam=0.5):
    """从 K 个假设中选择核型一致性最优的"""
    best_score = -float('inf')
    best_hypothesis = None
    for hyp in hypotheses:
        s_karyo = compute_karyo_score(hyp, alphas)  # 3.3 节定义
        s_conf = hyp.scores.mean().item()
        s_total = lam * s_conf + (1 - lam) * s_karyo
        if s_total > best_score:
            best_score = s_total
            best_hypothesis = hyp
    return best_hypothesis
```

**复杂度**: $O(K \cdot N)$, 其中 $N$ 为每个假设的检测数 (NMS 后 ≈ 46-100)

#### 3.4.2 进阶算法: 跨假设融合 (Optional)

某些情况下, 单个假设都不完美, 但不同假设互补 (假设 1 缺 A1, 假设 2 有 A1 但多了个假阳)。可做跨假设融合:

```python
def pcse_fuse(hypotheses):
    """跨假设融合: 对每个类, 从所有假设中选最配对的两条"""
    fused = []
    for k in range(24):
        # 收集所有假设中类别为 k 的检测
        candidates = collect_all(hypotheses, class=k)
        if len(candidates) < 2:
            fused.extend(candidates)
            continue
        # 用形态相似度找最佳配对
        best_pair = find_best_pair(candidates, criterion='morph_sim')
        fused.extend(best_pair)
    return fused
```

**注意**: 跨假设融合复杂度更高 ($O(K^2 \cdot N^2)$ 配对搜索), 且可能引入不一致。**Phase 1 不实现, 仅作未来扩展**。

### 3.5 与现有代码的集成方案

PCSE 是纯推理期优化, 改动集中在 `DiffusionSampler` 和 `predict` 方法:

**改动清单**:
1. `sampling.py`: 新增 `KaryotypeScorer` 类 (3.3 节评分函数)
2. `sampling.py`: 新增 `pcse_post_process` 方法 (替代 `post_process` 当 `use_pcse=True`)
3. `head.py`: `predict` 方法新增 `pcse_k` 参数, 循环调用 K 次采样
4. `head.py` / `sampling.py`: 新增 `predict_single_hypothesis` 方法 (单次采样, 不做时间维 ensemble)
5. 配置文件: 新增 `pcse_k`, `pcse_lambda`, `pcse_alphas` 参数

**配置示例**:
```python
# experiments/configs/ldmdet/r_pcse.py
_base_ = ['./a3_full_sota.py']
model = dict(
    bbox_head=dict(
        use_pcse=True,
        pcse_k=5,                          # 5 个候选假设
        pcse_lambda=0.5,                   # 似然-先验平衡
        pcse_alphas=(1.0, 2.0, 1.5, 1.5),  # (count, pair, morph, sex) 权重
        pcse_diversity='seed',             # 'seed' | 'threshold' | 'steps'
    ),
)
```

---

## 4. 深入可行性分析

### 4.1 计算开销评估

#### 4.1.1 单次采样开销

当前基线 (a3_full_sota) 配置: Heun 4 步采样。
- Heun 是二阶求解器, 每步需 2 次模型前向 (head.py:362-374 的 `model_fn` 回调)
- 单次采样: $4 \times 2 = 8$ 次模型前向
- 每次前向: 6 级 cascade head × 500 proposals, ResNet-50 + FPN 特征提取 (特征可缓存, 只算 head)

**特征提取**: 1 次 (可跨假设复用)
**单假设 head 前向**: 8 次 (Heun 4 步 × 2)

#### 4.1.2 PCSE 总开销

PCSE 的关键优化: **特征提取只做一次, K 个假设共享 backbone+FPN 特征**。

| 组件 | 单次采样 | PCSE (K=5) | 增量 |
|------|---------|------------|------|
| backbone+FPN 特征 | 1× | 1× (复用) | 0% |
| Head 前向 (8次/假设) | 8× | 40× | +400% |
| NMS | 1× | 5× | +400% |
| 核型评分 | 0 | 5× | 可忽略 |
| **总推理时间** | $T_0$ | $\approx 1.8 T_0 \sim 3.5 T_0$ | **+80% ~ +250%** |

**估算依据**:
- 若 backbone+FPN 占 50% 推理时间 (常见于 ResNet-50 + FPN), Head 占 50%:
  - PCSE: $0.5 T_0 + 5 \times 0.5 T_0 = 3.0 T_0$ (3× 总时间)
- 若 backbone+FPN 占 70% (大输入分辨率场景):
  - PCSE: $0.7 T_0 + 5 \times 0.3 T_0 = 2.2 T_0$ (2.2× 总时间)
- 若用 batch 并行 (K 个假设同时前向): 接近 $1.0 T_0$ (仅需更大显存)

#### 4.1.3 核型评分开销 (可忽略)

每个假设评分:
- `score_count`: $O(1)$, ~0.001 ms
- `score_pair`: $O(N)$ bincount, $N \approx 46$, ~0.01 ms
- `score_morph`: $O(N)$ + 22 对比较, ~0.05 ms
- `score_sex`: $O(N)$, ~0.01 ms
- **单假设评分**: ~0.07 ms
- **K=5 假设评分**: ~0.35 ms

相比单次模型前向 (~50-100 ms), 评分开销 **< 0.5%**, 完全可忽略。

#### 4.1.4 显存开销

K 个假设的中间张量: $K \times (\text{bs} \times 500 \times 4)$ = $5 \times 1 \times 500 \times 4 \times 4 \text{B} = 40 \text{KB}$

**显存增量可忽略**。若用 batch 并行, 需 $K \times$ Head 激活值, 约 $5 \times 2 \text{GB} = 10 \text{GB}$ (6 级 cascade × 500 proposals × 256 channels), 单卡 A100 (80GB) 可承受。

### 4.2 与现有 Ensemble 的区别 (深入对比)

阅读代码后确认, 现有 `use_ensemble` 与 PCSE 在**假设生成方式**和**聚合方式**上完全不同:

#### 4.2.1 假设生成: 同轨迹 vs 跨轨迹

```python
# 现有 ensemble (head.py:332-343) — 同一轨迹, 不同时间步
x_raw = torch.randn(...)  # 单次随机初始化
for t_curr, t_next in time_pairs:
    cls_logits, pred_bboxes, ... = self._forward_at_t(features, x_raw, t_curr, ...)
    ensemble_results.append((cls_logits, pred_bboxes))  # ← 同一轨迹的 T 个快照
    x_raw = solver.step(x_raw, x0_raw, t_curr, t_next)  # 轨迹演化

# PCSE — 不同轨迹, 完整采样
for k in range(K):
    torch.manual_seed(seed + k)
    x_raw_k = torch.randn(...)  # ← 每次新的随机初始化
    for t_curr, t_next in time_pairs:
        cls_logits, pred_bboxes, ... = self._forward_at_t(features, x_raw_k, t_curr, ...)
        x_raw_k = solver.step(x_raw_k, x0_raw, t_curr, t_next)
    hypotheses.append(final_result_of_trajectory_k)  # ← 完整轨迹的最终结果
```

**关键区别**:
- 现有 ensemble 的 T 个快照**强相关** (同一轨迹的相邻时间步预测高度相似), 拼接后 NMS 主要起去重作用
- PCSE 的 K 个假设**近似独立** (不同初始噪声 → 不同收敛点), 提供真正的多样性

#### 4.2.2 聚合方式: NMS 平均 vs 评分选择

```python
# 现有 post_process (sampling.py:194-264) — 拼接 + NMS
all_scores = torch.cat([hyp.scores for hyp in ensemble_results])  # T × N
all_bboxes = torch.cat([hyp.bboxes for hyp in ensemble_results])
keep = batched_nms(all_bboxes, all_scores, all_labels, nms_thr)    # 贪心去重
# 问题: 高置信度假阳会"压"过低置信度真阳

# PCSE — 评分 + 选择
scores = [compute_total_score(hyp) for hyp in hypotheses]
best_idx = argmax(scores)
return hypotheses[best_idx]  # 选整个假设, 保留内部一致性
```

**关键区别**:
- NMS 是**框级**贪心: 每个框独立决定去留, 不考虑全局核型结构
- PCSE 是**集合级**选择: 整个检测集合作为一个假设被评估, 保留"46 条互相配对"的内部一致性

#### 4.2.3 正交性: 可组合

PCSE 与现有时间维 ensemble **不冲突**, 可组合使用:
- **方案 A**: 每个假设内部禁用时间维 ensemble (只用最后一步), K 个假设做 PCSE 选择
- **方案 B**: 每个假设内部保留时间维 ensemble (NMS 去重), K 个去重后的假设再做 PCSE 选择

推荐 **方案 B**: 时间维 ensemble 先提升单假设质量, PCSE 再选最优。

### 4.3 实时性分析

#### 4.3.1 离线核型分析场景

染色体核型分析是**离线诊断任务**, 非实时。临床流程:
1. 培养细胞 → 显微镜扫描 → 获取中期分裂相图像 (数小时)
2. 核型分析 (人工或自动) → 出报告 (数分钟到数小时)

**结论**: 推理时间从 100ms 增加到 300ms (3×), 在临床流程中**完全可接受**。

#### 4.3.2 并行加速潜力

PCSE 的 K 个假设天然可并行:
- **Batch 并行**: K 个假设作为 batch 维度同时前向 (需调整 `predict` 的 batch 处理)
- **多卡并行**: K 个假设分到 K 张卡 (若可用)
- **流并行**: 单卡用 CUDA streams 重叠计算

Batch 并行可把 K=5 的开销降到接近 1× (受限于显存), 是最实际的加速方案。

### 4.4 评分函数的数学性质分析

#### 4.4.1 评分函数的有界性

$S_{\text{karyo}} \in [0, S_{\max}]$, 其中:
- $S_{\text{count}} \in [0, 1]$
- $S_{\text{pair}} \in [0, 1]$
- $S_{\text{morph}} \in [0, 1]$
- $S_{\text{sex}} \in [0, 1]$

$S_{\max} = \alpha + \beta + \gamma + \delta$ (当所有子项满分时)

**归一化**: 可除以 $S_{\max}$ 使 $S_{\text{karyo}} \in [0, 1]$, 便于与 $\bar{s}_k \in [0, 1]$ 加权。

#### 4.4.2 评分函数的单调性

- $S_{\text{count}}$: 当 $N \to 46$ 时单调递增, $N \to \infty$ 或 $N \to 0$ 时递减 ✓
- $S_{\text{pair}}$: 当更多类满足 $n_k = 2$ 时单调递增 ✓
- $S_{\text{morph}}$: 当同源对形态更相似时单调递增 ✓
- $S_{\text{sex}}$: 在合法核型 (XX/XY) 时最大 ✓

**整体单调性**: 在固定其他子项时, 改善任一子项都会提升总分, 符合直觉。

#### 4.4.3 评分函数的鲁棒性

**潜在问题**: 评分函数依赖 NMS 后的类别标签。若 NMS 阈值过严, 可能误删真阳, 使 $n_k$ 偏离 2。

**缓解**:
- 评分时用**软类别概率**而非硬标签: $n_k = \sum_i p(c_i = k)$, 允许模糊贡献
- 或对 NMS 阈值做扰动 (3.2.2 策略 B), 让不同假设用不同阈值

#### 4.4.4 计算复杂度汇总

| 子函数 | 复杂度 | 单次耗时 (N≈46) |
|--------|--------|----------------|
| `score_count` | $O(1)$ | ~0.001 ms |
| `score_pair` | $O(N)$ | ~0.01 ms |
| `score_morph` | $O(N)$ | ~0.05 ms |
| `score_sex` | $O(N)$ | ~0.01 ms |
| **单假设总评分** | $O(N)$ | **~0.07 ms** |
| **K=5 假设评分** | $O(K \cdot N)$ | **~0.35 ms** |

相比模型前向 (~50-100 ms), 评分开销 **< 0.5%**, 完全可忽略。

---

## 5. 风险评估

### 5.1 风险一: 多假设多样性不足 (假设空间覆盖不够)

**风险描述**: 若不同随机种子产生的假设高度相似 (扩散模型在低步数时可能收敛到相近的局部最优), PCSE 退化为"重复计算 + 选自己", 无增益。

**量化指标**: 定义假设间多样性 $D = \frac{1}{K(K-1)} \sum_{i \neq j} \text{IoU}(\mathbf{Y}_i, \mathbf{Y}_j)$。若 $D > 0.9$ (假设高度重叠), PCSE 增益有限。

**缓解方案**:
1. **种子扰动幅度调大**: 3.2.2 策略 A, 用 $\sigma_k = 0.5 \sim 1.0$ 增加初始噪声差异
2. **box_renewal 阈值扰动**: 3.2.2 策略 B, 不同假设用不同 `score_thr`, 强制保留/替换不同框子集
3. **前期诊断实验**: Phase 1 先测量 K=5 时的假设多样性 $D$, 若 $D < 0.7$ 再考虑进阶策略
4. **最坏情况兜底**: 若多样性不足, PCSE 退化为"多次采样取最高置信度", 仍不低于基线 (无负增益)

### 5.2 风险二: 评分函数权重 $\alpha, \beta, \gamma, \delta, \lambda$ 难调

**风险描述**: 5 个超参数的搜索空间大, 若权重设置不当 (如 $\beta$ 过大导致模型只追求"配对数对"而忽略置信度), 可能选出"核型合法但检测质量差"的假设。

**缓解方案**:
1. **归一化后等权起步**: 所有子项归一化到 $[0, 1]$, 初始 $\alpha = \beta = \gamma = \delta = 1.0, \lambda = 0.5$ (似然与先验等权)
2. **网格搜索**: 在验证集上对 $(\lambda, \beta)$ 做粗网格 (5×5=25 组), 其余固定。验证集 ~500 张, 25 组评分 ~1 分钟 (评分本身极快)
3. **退化为置信度选择**: 若所有核型权重的组合都不优于纯置信度选择 ($\lambda = 1.0$), 说明核型先验无增益, 此时 PCSE 退化为"多次采样取最高置信度", 仍不低于基线
4. **学习排序 (Phase 3)**: 用验证集的 (假设, GT mAP) 对训练一个轻量排序模型 (如 LightGBM), 自动学习权重

### 5.3 风险三: 异常核型样本被错误惩罚

**风险描述**: 24obj 数据集可能包含异常核型样本 (如三体 21、缺失、易位), 这些样本不满足"46 条 + 各类 2 个"约束, PCSE 会错误惩罚它们, 选择"凑齐 46 条但实际错误"的假设。

**量化估计**: 若数据集中异常核型比例 < 5% (常见临床比例), PCSE 对 95% 正常样本有益, 对 5% 异常样本可能略降。净收益仍为正, 但需控制异常样本上的退化幅度。

**缓解方案**:
1. **容差机制**: $S_{\text{count}}$ 用 $\sigma_c = 2.0$ 的高斯 (非硬阈值), 允许 ±2 偏差; $S_{\text{pair}}$ 对 $n_k = 3$ 给 0.5 分 (非 0), 容忍三体
2. **分数截断**: $S_{\text{total}} = \max(\lambda \bar{s}_k, \text{baseline})$, 保证 PCSE 不会选一个置信度极低的假设
3. **异常检测后处理**: 若所有假设的 $S_{\text{karyo}}$ 都很低 (< 0.3), 判定为异常样本, 退化为纯置信度选择
4. **数据集审计**: Phase 0 先统计 24obj 数据集中异常核型的比例和类型, 量化风险

### 5.4 风险四 (附加): 计算开销超标

**风险描述**: K=5 时推理时间增加 2-3×, 若部署环境有延迟约束 (如交互式诊断), 可能不可接受。

**缓解方案**:
1. **自适应 K**: 先用 K=1 快速出结果, 若 $S_{\text{karyo}} < 0.5$ (核型不合法) 再增加 K 到 5。大部分样本 K=1 即可, 仅难样本增加计算
2. **Batch 并行**: K 个假设 batch 化, 显存允许时开销接近 1×
3. **K=3 折中**: K=3 已能覆盖大部分假设空间 (类似 Top-3 选择), 开销 1.5-2×, 性价比最高

---

## 6. 预期收益分析

### 6.1 基于 1→8 步增益递减数据的定量估计

**已知数据**: 1→8 步采样, mAP 0.740→0.755, 增益 +0.015, 但 8× 计算量。

**增益来源分解**:
- 步数增加 → 单轨迹精度提升 (但递减): 1→2 步约 +0.008, 2→4 步约 +0.005, 4→8 步约 +0.002
- 这说明: **同一轨迹上增加计算量, 边际收益急剧递减**

**PCSE 的增益逻辑不同**:
- PCSE 不是"在同一轨迹上走更远", 而是"探索不同轨迹"
- 不同随机种子产生的假设, 误差模式不同 → 选择能规避系统性误差
- 核型先验提供了**正交于模型置信度的新信号**

**定量估计**:

| 增益来源 | 机制 | 预估 Δ mAP | 依据 |
|---------|------|-----------|------|
| 多假设选择 | 避免单次采样的系统性偏差 | +0.003 ~ +0.008 | 类比 box_renewal (+0.016) 的随机性收益, PCSE 是"选择性利用随机性" |
| 核型数量约束 | 修正漏检/多检 | +0.002 ~ +0.005 | 数量错误约占 5-10% 样本, 修正一半 → +0.003 |
| 配对一致性 | 修正同组混淆 (如 C9↔C10) | +0.001 ~ +0.003 | 同组混淆是 43.4% 分类误差的子集, PCSE 不直接改分类, 仅选分类更一致的假设 |
| **合计** | | **+0.006 ~ +0.016** | 取中位数 **+0.010** |

**保守估计**: +0.005 mAP (0.858 → 0.863)
**中位估计**: +0.010 mAP (0.858 → 0.868)
**乐观估计**: +0.015 mAP (0.858 → 0.873)

### 6.2 与现有组件的增益对比

| 组件 | 增益 (Δ mAP) | 计算开销 | 风险 |
|------|-------------|---------|------|
| box_renewal | +0.016 | 1× (无额外) | 已验证 |
| 1→8 步采样 | +0.015 | 8× | 已验证 (递减) |
| 时间维 ensemble | 已包含在基线 | 1× | 已验证 |
| **PCSE (K=5)** | **+0.006 ~ +0.016** | **2-3×** | 中 |
| CFM (已证伪) | -0.002 | 1× | 已失败 |

**关键观察**: PCSE 的增益预估与 box_renewal 相当 (两者都利用随机性), 但机制不同:
- box_renewal 是**训练时**注入随机性提升泛化
- PCSE 是**推理时**利用随机性做选择
- 两者可叠加 (box_renewal 已在基线中, PCSE 在其上再增加选择收益)

### 6.3 增益上限分析

PCSE 的增益上限受限于:
1. **假设质量**: 若所有 K 个假设都有同样的错误 (如都漏检了某条染色体), PCSE 无法修复。上限: 不同假设的误差重叠率
2. **评分准确性**: 若核型评分与真实 mAP 不相关, 选择无效。上限: 评分与 mAP 的相关系数

**理论上限**: 假设 K 个假设的错误率独立, 选最优可降低错误率至 $\epsilon^K$ (其中 $\epsilon$ 是单假设错误率)。但实际假设非独立, 增益低于此理论值。

**实际预期**: +0.010 mAP (中位估计), 不超过 +0.020 (乐观上限)。

---

## 7. 实现路线图

### Phase 0: 前置诊断 (1 周)

**目标**: 量化假设多样性和异常核型比例, 决定是否推进。

**任务**:
1. 用现有基线模型 (a3_full_sota), 对验证集 100 张图各运行 K=5 次不同种子的采样
2. 统计假设间多样性 $D$ (IoU 均值), 确认 $D < 0.8$ (有足够多样性)
3. 统计 24obj 数据集中异常核型 (非 46 条、非各类 2 个) 的比例
4. 统计当前基线的数量错误率 (预测 ≠ 46 的比例)

**决策点**:
- 若 $D > 0.9$ (假设高度相似) → 暂停 PCSE, 先研究多样性增强
- 若异常核型比例 > 15% → 调整评分函数容差
- 若数量错误率 < 2% → PCSE 增益有限, 重新评估优先级

### Phase 1: 核心实现与验证 (2 周)

**目标**: 实现 PCSE 基础版, 验证纯推理增益。

**任务**:
1. 实现 `KaryotypeScorer` 类 (3.3 节, 4 个子评分函数)
2. 实现 `predict_single_hypothesis` 方法 (单次采样, 不做时间维 ensemble)
3. 实现 `pcse_post_process` (K 个假设评分 + argmax 选择)
4. 修改 `predict` 方法支持 `pcse_k` 参数
5. 新增配置 `r_pcse.py`, 在 a3_full_sota 基础上启用 PCSE
6. 在验证集上跑 K=3, 5, 7, 对比 mAP
7. 网格搜索 $(\lambda, \beta)$ 权重

**预期产出**:
- PCSE (K=5) 在验证集上的 mAP 报告
- 最优权重配置
- 假设多样性分析图

### Phase 2: 多样性增强与优化 (2 周)

**目标**: 若 Phase 1 增益 > +0.005, 优化多样性策略和计算效率。

**任务**:
1. 实现 3.2.2 策略 B (box_renewal 阈值扰动), 对比多样性提升
2. 实现 batch 并行 (K 个假设同时前向), 测量加速比
3. 实现自适应 K (低 $S_{\text{karyo}}$ 时增加 K)
4. 在测试集上最终评估

**预期产出**:
- 最优多样性策略
- 并行加速 benchmark
- 测试集最终 mAP

### Phase 3 (可选): 跨假设融合 (3 周)

**目标**: 若 Phase 2 增益饱和但仍有提升空间, 探索跨假设融合。

**任务**:
1. 实现 3.4.2 跨假设融合算法
2. 用学习排序替代手工权重
3. 论文级实验: 消融、可视化、案例分析

**预期产出**:
- 跨假设融合的增益评估
- 学习排序模型
- 论文初稿素材

---

## 8. 与已证伪方向的对比: 为什么 PCSE 不会重蹈 CFM 覆辙

### 8.1 CFM 失败原因复盘

阅读 [方向H文档](../archived/FlowMatching检测.md) 可知, CFM (方向 H) 失败的核心原因:

1. **改变训练目标**: CFM 要求模型预测速度 $v$ 而非 $x_0$, 需修改 loss 和 head 输出语义
2. **权重过保守**: `velocity_loss_weight=0.1`, velocity loss 仅占总 loss < 3%, 不足以改变训练动态
3. **不引入新信息**: 速度一致性损失仅对齐训练-推理目标, **没有引入模型未知的先验**
4. **训练不稳定风险**: 改训练目标可能导致梯度尺度变化, 需仔细调参
5. **结果**: mAP=0.854 vs 基线 0.856, **-0.002** (噪声范围内, 无增益)

### 8.2 PCSE 与 CFM 的本质区别

| 维度 | CFM (已证伪) | PCSE (本方向) |
|------|-------------|---------------|
| **是否改训练** | ✅ 改 (loss + head) | ❌ 不改 (纯推理) |
| **是否引入新信息** | ❌ 不引入 (仅对齐目标) | ✅ 引入核型先验 (模型未知) |
| **失败模式** | 训练动态不变, 无增益 | 最坏退化为基线 (无负增益) |
| **可调参数** | 训练权重 (需重训验证) | 推理权重 (验证集秒级搜索) |
| **计算开销** | 1× (训练 1.5×) | 2-3× (推理, 可并行) |
| **可回退性** | 差 (需重训) | 好 (关掉 `use_pcse` 即可) |

### 8.3 为什么 PCSE 能避免 CFM 的三个陷阱

**陷阱 1: "改训练目标但权重太保守"**
- PCSE **不改训练目标**, 完全复用 a3_full_sota 的训练权重
- 推理期的 $\lambda$ 权重可在验证集上直接搜索, 无需重训
- 即使 $\lambda = 1.0$ (纯置信度选择), PCSE 仍不低于基线 (多次采样取最优)

**陷阱 2: "不引入新信息"**
- CFM 的速度损失只是把模型已有的 $x_0$ 预测"换个形式"监督, 没有新信号
- PCSE 的核型评分是**模型完全未知的全局约束**:
  - 模型训练时从未见过"46 条"的数量约束
  - 模型训练时从未见过"同源对形态相似"的配对约束
  - 这些是纯粹的领域先验, 提供了正交于模型置信度的新信号

**陷阱 3: "训练不稳定"**
- PCSE 无训练过程, 无梯度, 无不稳定风险
- 最坏情况: 评分函数权重全错 → 退化为随机选一个假设 → 期望性能等于单次采样 (基线)
- **PCSE 有下界保证**: $mAP_{\text{PCSE}} \geq mAP_{\text{baseline}} - \epsilon$ (其中 $\epsilon$ 是评分噪声, 实际可忽略)

### 8.4 PCSE 的失败模式分析

即使 PCSE 增益不如预期, 其失败模式是**可控的**:

| 失败模式 | 概率 | 影响 | 兜底 |
|---------|------|------|------|
| 假设多样性不足 | 中 | 增益 ≈ 0 | 退化为基线, 无负增益 |
| 评分权重错误 | 低 | 增益 ≈ 0 | 网格搜索修正, 或退化为纯置信度选择 |
| 异常核型被误惩罚 | 低 | 异常样本 mAP 略降 | 容差机制 + 异常检测退路 |
| 计算开销超标 | 低 | 部署受限 | 自适应 K + batch 并行 |

**关键结论**: PCSE 是**非负收益方向** (最坏退化为基线), 而 CFM 是**可能负收益方向** (改训练有风险)。这是 PCSE 相对 CFM 的根本优势。

---

## 9. 参考文献

### 9.1 多假设与集成方法
- Reid, D. B. "An Algorithm for Tracking Multiple Targets." IEEE Transactions on Automatic Control, 1979. — 多假设检验 (MHT) 经典
- Hoeting, J. A. et al. "Bayesian Model Averaging: A Tutorial." Statistical Science, 1999. — 贝叶斯模型平均
- Gal, Y. & Ghahramani, Z. "Dropout as a Bayesian Approximation." ICML, 2016. — MC Dropout (多次随机前向)
- Lakshminarayanan, B. et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS, 2017. — 深度集成

### 9.2 扩散检测与采样
- DiffusionDet: Chi et al. "DiffusionDet: Diffusion Model for Object Detection." ICCV, 2023. — 扩散检测基线
- DDPM: Ho, J. et al. "Denoising Diffusion Probabilistic Models." NeurIPS, 2020. — 扩散采样随机性
- Rectified Flow: Liu, X. et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR, 2023. — 当前所用 RF 采样
- DPM-Solver++: Lu, C. et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models." NeurIPS, 2022.

### 9.3 染色体核型学
- ISCN: International System for Human Cytogenomic Nomenclature (2020). — 核型国际标准
- Denver Classification: 1960. — A-G 分组标准
- Sharma, T. "Chromosome Techniques: Theory and Practice." 3rd Ed. — 核型分析流程

### 9.4 相关检测后处理
- Soft-NMS: Bodla, N. et al. "Soft-NMS -- Improving Object Detection with One Line of Code." ICCV, 2017. — NMS 改进
- DETR: Carion, N. et al. "End-to-End Object Detection with Transformers." ECCV, 2020. — 集合预测
- Relation Networks: Hu, H. et al. "Relation Networks for Object Detection." CVPR, 2018. — 框间关系

### 9.5 项目内相关方向
- [方向K: 核型结构化生成](方向K_核型结构化生成.md) — 训练期注入核型约束 (GNN)
- [方向O3: PKEC 概率核型熵冷却](方向O3_PKEC_概率核型熵冷却.md) — 推理期梯度引导
- [方向H: Flow Matching 检测 (已证伪)](../archived/FlowMatching检测.md) — CFM 失败教训

---

## 附录 A: 与方向 O3 (PKEC) 的对比

PCSE 与 PKEC 都在推理期利用核型约束, 但机制不同:

| 维度 | PKEC (梯度引导) | PCSE (假设选择) |
|------|----------------|----------------|
| **约束注入方式** | 梯度引导修改 $x_t$ | 离散选择完整假设 |
| **是否可微** | 是 (需 autograd) | 否 (纯评分) |
| **数值稳定性** | 风险 (梯度可能爆炸/消失) | 稳定 (无梯度) |
| **计算开销** | 每步额外反向传播 | 每假设额外评分 (可忽略) |
| **与现有采样器的兼容** | 需改采样循环 | 只改后处理 |
| **失败模式** | 梯度断路/局部最优 | 退化为基线 |

**协同可能**: PKEC 和 PCSE 可组合 — PKEC 引导每个假设向合法核型靠拢, PCSE 再从中选最优。但 Phase 1 建议单独验证 PCSE。

## 附录 B: 评分函数完整实现伪代码

```python
class KaryotypeScorer:
    """核型配对一致性评分器 (PCSE 核心)"""

    def __init__(
        self,
        num_classes=24,
        num_autosome=22,
        x_class_idx=23,
        y_class_idx=24,
        target_count=46,
        count_sigma=2.0,
        alphas=(1.0, 2.0, 1.5, 1.5),
        lam=0.5,
    ):
        self.num_autosome = num_autosome
        self.x_class_idx = x_class_idx
        self.y_class_idx = y_class_idx
        self.target_count = target_count
        self.count_sigma = count_sigma
        self.alphas = alphas  # (alpha_count, beta_pair, gamma_morph, delta_sex)
        self.lam = lam

    def score(self, bboxes, scores, labels):
        """计算单个假设的综合评分"""
        s_count = self._score_count(labels)
        s_pair = self._score_pair(labels)
        s_morph = self._score_morph(bboxes, labels)
        s_sex = self._score_sex(labels)
        s_karyo = (
            self.alphas[0] * s_count
            + self.alphas[1] * s_pair
            + self.alphas[2] * s_morph
            + self.alphas[3] * s_sex
        )
        s_karyo /= sum(self.alphas)  # 归一化到 [0, 1]
        s_conf = scores.mean().item() if len(scores) > 0 else 0.0
        return self.lam * s_conf + (1 - self.lam) * s_karyo

    def _score_count(self, labels):
        n = len(labels)
        return math.exp(-((n - self.target_count) ** 2) / (2 * self.count_sigma ** 2))

    def _score_pair(self, labels):
        counts = torch.bincount(labels, minlength=self.num_autosome + 2)
        total = 0.0
        for k in range(self.num_autosome):
            n = counts[k].item()
            if n == 2:
                total += 1.0
            elif n in (1, 3):
                total += 0.5
        return total / self.num_autosome

    def _score_morph(self, bboxes, labels):
        total_sim = 0.0
        num_pairs = 0
        for k in range(self.num_autosome):
            idx = (labels == k).nonzero(as_tuple=True)[0]
            if len(idx) != 2:
                continue
            b1, b2 = bboxes[idx[0]], bboxes[idx[1]]
            w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
            w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
            A1, A2 = w1 * h1, w2 * h2
            r1 = max(w1, h1) / max(min(w1, h1), 1e-6)
            r2 = max(w2, h2) / max(min(w2, h2), 1e-6)
            sim_size = 1 - abs(A1 - A2) / max(A1, A2, 1e-6)
            sim_aspect = 1 - abs(r1 - r2) / max(r1, r2, 1e-6)
            total_sim += 0.5 * (sim_size + sim_aspect)
            num_pairs += 1
        return total_sim / max(num_pairs, 1)

    def _score_sex(self, labels):
        n_x = (labels == self.x_class_idx).sum().item()
        n_y = (labels == self.y_class_idx).sum().item()
        if (n_x, n_y) in [(2, 0), (1, 1)]:
            return 1.0
        elif (n_x, n_y) in [(1, 0), (2, 1), (3, 0)]:
            return 0.5
        return 0.0


def pcse_inference(model, features, img_metas, scorer, K=5):
    """PCSE 推理流程"""
    hypotheses = []
    for k in range(K):
        torch.manual_seed(2024 + k)  # 可复现的不同种子
        result = model.bbox_head.predict_single_hypothesis(features, img_metas)
        hypotheses.append(result)

    scores = [scorer.score(h.bboxes, h.scores, h.labels) for h in hypotheses]
    best_idx = int(np.argmax(scores))
    return hypotheses[best_idx]
```
