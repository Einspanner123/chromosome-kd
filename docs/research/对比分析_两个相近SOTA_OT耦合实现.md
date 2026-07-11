# 两个相近 SOTA 的深度对比分析报告

> **对比对象**
> - **SOTA**: `reproduce_0751_stochot_eps5_v2` — Stochastic OT (ε=5, multinomial) 主线 SOTA, mAP=0.753
> - **近 SOTA-1**: `nonlinear_trajectory` (E4.3 eps1) — OT Flow (ε=1.0, argmax) + 尺度条件化 RF, mAP=0.752
> - **近 SOTA-2**: `nonlinear_trajectory_e43_eps2` (E4.3 eps2) — OT Flow (ε=2.0, argmax) + 尺度条件化 RF, mAP=0.752
>
> **核心问题**: 三者在 COCO mAP 上仅相差 0.001–0.002, 是否代表等价的工程实现？还是通过不同的理论路径抵达同一精度水平？
>
> **统计严谨性声明**: 以下所有 mAP 比较均为**单种子**结果，未报告置信区间。CV 领域 run-to-run 方差通常 0.3-0.8%（0.003-0.008 mAP），因此 0.001-0.002 mAP 量级的差异**在统计噪声内**，不足以做"等价"或"优劣"的因果归因。本文的"打平""上限""等价"等表述均在此限制下理解，需多种子 paired t-test（见[理论验证与修正实验方案](理论验证与修正实验方案.md) §9.3）才能做最终判定。

> ⚠️ **暂时废弃（实验数值部分）**：以下全部对比分析中的实验结果（SOTA mAP=0.753、E4.3 0.752、ε 扫描 0.735-0.755 等）均基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新（mAP 量级 0.77-0.87）。代码实现对比（OT 模块、Sinkhorn 迭代、解码策略）与数学等价性推导不受影响，但所有 mAP 数值与训练曲线不可直接引用。

---

## 1. 摘要 (Executive Summary)

三组实验在 mAP 上近乎打平（0.752 vs 0.753，Δ=0.001，**单种子，在统计噪声内**），但**底层实现与理论机制存在本质差异**：

| 维度 | SOTA (0.753) | E4.3 eps1 (0.752) | E4.3 eps2 (0.752) |
|------|:---:|:---:|:---:|
| OT 模块类 | `OTCoupling` (legacy) | `OTFlowCoupling` (方向四) | `OTFlowCoupling` (方向四) |
| 解码策略 | **stochastic multinomial** | **deterministic argmax** | **deterministic argmax** |
| Sinkhorn ε | 5.0 | 1.0 | 2.0 |
| Sinkhorn 迭代 | 20 | 10 | 10 |
| 代价矩阵朝向 | [N_prop, K_GT] | [M_GT, N_prop] (转置) | [M_GT, N_prop] (转置) |
| 尺度条件化 RF | ✗ (标准线性 RF) | ✓ (λ=0.5) | ✓ (λ=0.5) |
| Best epoch | 59 | 94 | 100 |
| 训练总 epochs | 132 (手动停) | 124 (early stop) | 130 (early stop) |
| 训练时间 | 较短 (59 ep 到 best) | 较长 (94 ep 到 best) | 最长 (100 ep 到 best) |

**核心结论**:
1. 三者求解的是**同一个平衡 OT 问题**（均匀边缘约束），但**解码策略与熵正则化强度不同**，由此产生不同的训练动力学。
2. SOTA 通过 **stochastic multinomial + 高熵 (ε=5)** 在 epoch 59 快速收敛到 0.753；E4.3 系列通过 **argmax + 低熵 (ε=1–2)** 在 epoch 94–100 收敛到 0.752。
3. 尺度条件化 RF（λ=0.5）对最终精度贡献**几乎为零**（E4.1 消融实验证明：单独使用尺度条件化反而使 mAP 下降到 0.743），E4.3 系列的精度几乎完全来自 OT Flow 耦合。
4. **0.752-0.753 是当前架构 (argmax/multinomial + AdaLN-Zero + Heun) 下反复触及的精度水平**，Δ=0.001 在单种子噪声内，**不构成 stochastic multinomial 优于 argmax 的统计证据**。需多种子 paired t-test 才能判定优劣。

---

## 2. 实验背景与定位

### 2.1 SOTA 的来源

`reproduce_0751_stochot_eps5_v2` 是 LDMDet 项目的**主线 SOTA**，继承自 Phase 5 的 Stochastic OT 研究。其关键创新点是：

- **Stochastic Coupling**: 用 Sinkhorn OT 传输矩阵的**随机采样**（而非 argmax）来匹配 proposal 与 GT, 解决了 Hard OT 的多样性崩塌问题（Phase 4 发现）。
- **CAM 定理验证**: Phase 5 证明了 argmax 会消除 ε 对多样性的控制（多样性常数 ≈2.80），而 stochastic 采样恢复 ε-多样性单调性（2.81→5.31）。
- **完整技术栈**: ResNet50 + FPN + 6×SingleHead (AdaLN-Zero) + Rectified Flow + Heun + Shifted Schedule + Stochastic Sinkhorn OT (ε=5)。

### 2.2 E4.3 系列的来源

`nonlinear_trajectory` 系列来自**方向四：流匹配的非线性轨迹**研究，目标是突破 1-RectFlow 直线路径的尺度不敏感问题。其两大组件：

- **ScaleConditionedRF**: 尺度条件化噪声调度，κ(s) = 1 + λ·(s_max − s)/s_max, 使小目标在更早的 t 处去噪。
- **OTFlowCoupling**: Mini-batch OT Flow 耦合，用 Sinkhorn 传输矩阵的 argmax/multinomial 解码配对 proposal 与 GT。

E4.3 系列是**尺度条件化 + OT Flow** 的联合实验。但消融实验（E4.1/E4.2）最终证明：
- **OT Flow 是核心价值**（E4.2 单独使用 = 0.751）
- **尺度条件化是无效组件**（E4.1 单独使用 = 0.743，反而下降）
- E4.3 联合的 0.752 ≈ E4.2 单独 OT Flow 的 0.751（尺度条件化无净增量）

### 2.3 为什么三者值得对比？

三者在 mAP 上仅相差 0.001–0.002，但：
- SOTA 用 **stochastic + 高熵 + 20 迭代**
- E4.3 eps1 用 **argmax + 低熵 (ε=1) + 10 迭代**
- E4.3 eps2 用 **argmax + 中熵 (ε=2) + 10 迭代**

这种"不同路径、相同终点"的现象，是检验 OT 耦合理论（特别是 CAM 定理、ε-多样性理论）的绝佳案例。

---

## 3. 代码实现深度对比

### 3.1 架构继承链对比

**SOTA 继承链**:
```
ldmdet_baseline.py
  └─ ldmdet_rf_heun_shifted_bs2.py    [RF + Heun + Shifted + bs=2]
      └─ ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py
            [ot_coupling=True, ot_matcher='sinkhorn', ot_sample=True, ε=5, iters=20]
```

**E4.3 eps2 继承链**:
```
../_base_/default_runtime.py + ../_base_/datasets/chromo_coco_detection.py
  └─ rf_heun_adaln.py    [RF + Heun + Shifted + AdaLN-Zero + bs=4]
      └─ nonlinear_trajectory.py
            [scale_conditioned_rf=dict(λ=0.5), coupling=dict(type='ot_flow', ε=1, iters=10, argmax)]
          └─ nonlinear_trajectory_e43_eps2.py
                [coupling.epsilon: 1.0 → 2.0]
```

**关键差异**:
- SOTA 的 `batch_size=2`，E4.3 的 `batch_size=4`（[rf_heun_adaln.py:13](../../experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py#L13)）
- SOTA 通过 `ot_coupling=True` + `ot_matcher='sinkhorn'` + `ot_sample=True` 三个布尔/字符串参数启用 OT
- E4.3 通过 `coupling=dict(type='ot_flow', ...)` 注册式策略启用 OT（更解耦的架构）

### 3.2 OT 耦合模块对比

#### 3.2.1 SOTA: `OTCoupling` (legacy 路径)

文件: [ot_coupling.py](../../ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/code/projects/LDMDet/mods/ot_coupling.py)

```python
class OTCoupling(nn.Module):
    def __init__(self, ot_coupling=False, ot_epsilon=1.0, ot_num_iters=20,
                 ot_sample_seed=None, ot_group_hierarchical=False):
        ...

    def couple(self, noise, gt_diffusion, gt_labels, device):
        # 1. 代价矩阵 [N=500 proposals, K=num_gt]
        cost = torch.cdist(noise, gt_diffusion, p=2)
        # 2. Sinkhorn 传输矩阵 [N, K]
        transport = self.sinkhorn_transport(cost)
        # 3. 行归一化（每个 proposal 行和=1）
        row_probs = transport / transport.sum(dim=1, keepdim=True).clamp_min(1e-10)
        # 4. 随机采样 1 个 GT per proposal
        return self._ot_multinomial(row_probs)
```

**关键实现特征**:
- 代价矩阵 `cost` 形状: `[N_proposals=500, K_GT]`
- 默认 `row_mass = 1/N` (proposals 均匀), `col_mass = proposals_per_gt/N = 1/K` (GTs 均匀)
- **行归一化** 后做 **multinomial 采样**
- 支持 `ot_group_hierarchical` 分组 OT（按染色体 A/B/C/D/E/F/G/X/Y 组分别 OT）

#### 3.2.2 E4.3: `OTFlowCoupling` (方向四)

文件: [ot_flow_coupling.py](../../ldmdet/coupling/ot_flow_coupling.py)

```python
@register_coupling('ot_flow')
class OTFlowCoupling(CouplingStrategy):
    def __init__(self, epsilon=1.0, num_iters=10, coupling_mode='argmax'):
        ...

    def couple(self, noise, gt_diffusion, gt_labels, device):
        N = noise.shape[0]; M = gt_diffusion.shape[0]
        # 1. 代价矩阵 [M=num_gt, N=500] (与 SOTA 转置!)
        cost = torch.cdist(gt_diffusion, noise, p=2)
        # 2. Sinkhorn 传输矩阵 [M, N]
        transport = sinkhorn_transport(cost, epsilon=self.epsilon, num_iters=self.num_iters)
        # 3. 列归一化（每个 proposal 列和=1）
        col_sums = transport.sum(dim=0, keepdim=True).clamp_min(1e-10)
        transport_col = transport / col_sums  # [M, N]
        # 4. argmax 选 GT per proposal
        if coupling_mode == 'argmax':
            matched_idx = transport_col.argmax(dim=0)  # [N]
        else:  # multinomial
            matched_idx = torch.multinomial(transport_col.t().clamp_min(1e-10), 1).squeeze(-1)
        return gt_diffusion[matched_idx], matched_idx
```

**关键实现特征**:
- 代价矩阵 `cost` 形状: `[M_GT, N_proposals=500]` — **与 SOTA 转置**
- 默认 `row_mass = 1/M` (GTs 均匀), `col_mass = 1/N` (proposals 均匀)
- **列归一化** 后做 **argmax** (或 multinomial)
- 不支持分组 OT（无 `ot_group_hierarchical` 选项）

#### 3.2.3 数学等价性分析（严格推导）

##### 3.2.3.1 问题形式化

设 $N$ = proposals 数（=500），$K$ = GT 数（=10，记 $M \equiv K$）。两实现的代价矩阵互为转置：

- **Legacy** ([ot_coupling.py:126](../../ldmdet/coupling/ot_flow_coupling.py#L126)): $C^{\text{leg}} = \text{cdist}(\text{noise}, \text{gt}) \in \mathbb{R}^{N \times K}$
- **New** ([ot_flow_coupling.py:88](../../ldmdet/coupling/ot_flow_coupling.py#L88)): $C^{\text{new}} = \text{cdist}(\text{gt}, \text{noise}) \in \mathbb{R}^{K \times N}$

显然 $C^{\text{new}} = (C^{\text{leg}})^T$。

两实现求解的熵正则化 OT 问题：

$$P^* = \arg\min_{P \in \Pi(a, b)} \langle P, C \rangle - \varepsilon H(P), \quad H(P) = -\sum_{ij} P_{ij} \log P_{ij}$$

其中 $\Pi(a, b) = \{P : P\mathbf{1} = a,\; P^T\mathbf{1} = b\}$ 为运输多面体。

##### 3.2.3.2 边缘约束的对称性

从代码逐行核对两实现的边缘约束：

**Legacy** ([ot_coupling.py:101-106](../../ldmdet/coupling/ot_flow_coupling.py#L101-L106)):
```python
row_mass = ones(N) / N                     # a_i = 1/N       (proposals)
proposals_per_gt = max(N // K, 1)           # = 50
col_mass = full(K, 50/500)                  # b_j = 0.1       (GTs)
col_mass = col_mass / col_mass.sum()        # 归一化后仍 0.1
```

**New** ([_sinkhorn_ops.py:39-44](../../ldmdet/coupling/_sinkhorn_ops.py#L39-L44), cost shape $[K, N]$):
```python
N_, K_ = cost.shape                          # N_=K=10, K_=N=500
row_mass = ones(N_) / N_                     # a'_j = 1/K = 0.1    (GTs)
proposals_per_gt = max(N_ // K_, 1)          # = max(10//500, 1) = 1
col_mass = full(K_, 1/10)                    # b'_i = 0.1           (proposals)
col_mass = col_mass / col_mass.sum()         # 归一化: 0.1/50 = 1/500 = 0.002
```

汇总：

| | 行边缘 $a$ | 列边缘 $b$ | 代价 $C$ |
|---|---|---|---|
| Legacy | $a_i = 1/N$ (proposals) | $b_j = 1/K$ (GTs) | $C \in \mathbb{R}^{N \times K}$ |
| New | $a'_j = 1/K$ (GTs) | $b'_i = 1/N$ (proposals) | $C' = C^T \in \mathbb{R}^{K \times N}$ |

**关键观察**: $a' = b$, $b' = a$, $C' = C^T$ — 边缘约束与代价矩阵同时转置。

##### 3.2.3.3 定理：两 OT 解互为转置

**定理**. 设 $P^* = \arg\min_{P \in \Pi(a,b)} \langle P, C \rangle - \varepsilon H(P)$，$Q^* = \arg\min_{Q \in \Pi(a',b')} \langle Q, C^T \rangle - \varepsilon H(Q)$，其中 $a' = b$, $b' = a$。则 $Q^* = (P^*)^T$。

**证明**. 对任意 $Q \in \Pi(a', b')$，令 $P = Q^T$。则：

1. **目标函数不变**:
   $$\langle Q, C^T \rangle = \sum_{ji} Q_{ji} C^T_{ji} = \sum_{ij} Q_{ji} C_{ij} = \sum_{ij} P_{ij} C_{ij} = \langle P, C \rangle$$
   $$H(Q) = -\sum_{ji} Q_{ji} \log Q_{ji} = -\sum_{ij} P_{ij} \log P_{ij} = H(P)$$

2. **约束集等价**:
   $$Q \in \Pi(a', b') \iff Q\mathbf{1} = a',\; Q^T\mathbf{1} = b' \iff P^T\mathbf{1} = b,\; P\mathbf{1} = a \iff P \in \Pi(a, b)$$

因此 $\min_{Q \in \Pi(a',b')} \langle Q, C^T \rangle - \varepsilon H(Q) = \min_{P \in \Pi(a,b)} \langle P, C \rangle - \varepsilon H(P)$。

由 $\varepsilon > 0$ 时熵正则化 OT 解的唯一性（目标函数严格凸），$Q^* = (P^*)^T$。$\square$

##### 3.2.3.4 归一化等价性

Legacy 在 $P^*$ 上做**行归一化**（每 proposal 对 GTs 的分布）：

$$\tilde{P}^{\text{leg}}_{ij} = \frac{P^*_{ij}}{\sum_{j'} P^*_{ij'}} = \frac{P^*_{ij}}{a_i}$$

New 在 $Q^* = (P^*)^T$ 上做**列归一化**（每 proposal 对 GTs 的分布）：

$$\tilde{Q}^{\text{new}}_{ji} = \frac{Q^*_{ji}}{\sum_{j'} Q^*_{j'i}} = \frac{(P^*)^T_{ji}}{\sum_{j'} (P^*)^T_{j'i}} = \frac{P^*_{ij}}{\sum_{j'} P^*_{ij'}} = \frac{P^*_{ij}}{a_i}$$

**两者得到的 proposal→GT 概率分布完全相同**：$\tilde{Q}^{\text{new}}_{ji} = \tilde{P}^{\text{leg}}_{ij}$。

##### 3.2.3.5 解码等价性

**argmax 解码**（对 proposal $i$，选 GT）:
- Legacy: $j^*_{\text{leg}}(i) = \arg\max_j \tilde{P}^{\text{leg}}_{ij} = \arg\max_j \frac{P^*_{ij}}{a_i} = \arg\max_j P^*_{ij}$（$a_i$ 是 $j$ 的常数）
- New: $j^*_{\text{new}}(i) = \arg\max_j \tilde{Q}^{\text{new}}_{ji} = \arg\max_j \frac{P^*_{ij}}{a_i} = \arg\max_j P^*_{ij}$

**完全相同**。

**multinomial 采样**（对 proposal $i$，按概率采 GT）:
- Legacy: $j \sim \text{Categorical}\left(\frac{P^*_{i,:}}{a_i}\right)$
- New: $j \sim \text{Categorical}\left(\frac{P^*_{i,:}}{a_i}\right)$

**完全相同**（Categorical 分布对归一化常数不变，$\frac{P^*_{i,:}}{a_i}$ 与 $\frac{P^*_{i,:}}{\sum_j P^*_{ij}}$ 定义相同的分布）。

##### 3.2.3.6 数学结论：配置相同时两实现严格等价

> **推论**. 在相同的 $(\varepsilon, \text{num\_iters}, \text{coupling\_mode}, \text{batch\_size})$ 下，Legacy OTCoupling（行归一化）与 New OTFlowCoupling（列归一化）产生**完全相同的 matched_idx**（argmax 情况）或**完全相同的采样分布**（multinomial 情况）。
>
> 因此，C4 矛盾（"sample > argmax" 在 SOTA 成立但在 E4.3 反转）**不可能由 OT 实现的矩阵朝向或归一化方向差异导致**——这些差异在数学上被转置对称性完全抵消。

##### 3.2.3.7 等价性在实践中的破坏因素

数学等价性成立的前提是"相同配置"。实际 SOTA 与 E4.3 的配置差异如下表，每个差异都是潜在的破坏因素：

| 因素 | SOTA (Legacy) | E4.3 (New) | 数学后果 |
|:----:|:---:|:---:|:---|
| **ε** | 5.0 | 1.0 / 2.0 | 改变 $P^*$ 本身（$\varepsilon \to \infty$ 时 $P^* \to ab^T$；$\varepsilon \to 0$ 时 $P^* \to$ 硬 OT） |
| **num_iters** | 20 | 10 | 有限迭代下 Sinkhorn 未完全收敛，$P^{(T)} \neq P^*$ |
| **coupling_mode** | multinomial | argmax | 采样 vs 确定性解码，多样性 $H(V\|X_t)$ 从 $>0$ 变 $\approx 0$ |
| **batch_size** | 2 | 4 | 改变梯度噪声结构，影响训练动力学（不影响 $P^*$ 本身） |
| **尺度条件化 RF** | 无 | 有 ($\lambda=0.5$) | 改变 $x_t$ 的分布，间接改变 $\text{cdist}(\text{noise}, \text{gt})$ 的代价矩阵 |

##### 3.2.3.8 一个隐蔽的数值差异：Sinkhorn 迭代顺序

除上述配置差异外，代码实现中还存在一个**数学上对称但数值上可能不等价**的因素——**迭代更新顺序**。

Sinkhorn 迭代（log-domain）:
```
log_u ← log_a - logsumexp(log_K + log_v, dim=1)   # 更新 u
log_v ← log_b - logsumexp(log_K + log_u, dim=0)   # 更新 v
```

- **Legacy** ([ot_coupling.py:113-119](../../ldmdet/coupling/ot_flow_coupling.py#L113-L119)): 先更新 $u$（proposal 维度，N=500），再更新 $v$（GT 维度，K=10）
- **New** ([_sinkhorn_ops.py:57-63](../../ldmdet/coupling/_sinkhorn_ops.py#L57-L63)): 先更新 $u$（GT 维度，K=10），再更新 $v$（proposal 维度，N=500）

由定理 3.2.3.3，New 的 $u$ 对应 Legacy 的 $v$，New 的 $v$ 对应 Legacy 的 $u$。因此：

- Legacy 迭代顺序：$u_{\text{prop}} \to v_{\text{GT}}$
- New 迭代顺序：$u_{\text{GT}} \to v_{\text{prop}}$，即 $v_{\text{GT}}^{\text{legacy}} \to u_{\text{prop}}^{\text{legacy}}$

**两实现的迭代顺序相反**。

**数学性质**: Sinkhorn 迭代在收敛时（$T \to \infty$）与更新顺序无关，收敛到唯一不动点 $P^*$。但在**有限迭代次数**下，不同顺序产生的 $P^{(T)}$ 可能不同——尤其当 $\varepsilon$ 较小（收敛慢）且 $T$ 较少时。

**量化影响**: SOTA 用 $\varepsilon=5, T=20$（快速收敛，顺序差异可忽略）；E4.3 用 $\varepsilon=1, T=10$（慢收敛 + 少迭代，顺序差异可能显著）。这是 V1 实验组需统一 $(\varepsilon, T)$ 的数学依据。

##### 3.2.3.9 对 V1 实验组的指导意义

上述推导明确了 V1 实验组的设计原则：

1. **V1-A vs V1-E**（legacy vs new, 同 sample+ε5+iter20）: 由定理 3.2.3.3，两应在数学上等价。若结果不同，只能归因于 §3.2.3.8 的迭代顺序数值差异或 batch_size 差异。
2. **V1-E vs V1-F**（同 new+ε5+iter20, sample vs argmax）: 隔离解码策略，这是 C4 矛盾的核心变量。
3. **V1-G vs V1-H**（同 legacy+ε1+iter10, sample vs argmax）: 在 E4.3 配置下隔离解码策略。
4. **V1-E vs V1-G**（同 sample, new+ε5 vs legacy+ε1）: 隔离 ε 的影响。

**关键预测**: 若 V1-E ≈ V1-A（在统一 batch_size 下），则证实定理 3.2.3.3——OT 实现方向不影响结果。若 V1-E ≠ V1-A，则需调查 §3.2.3.8 的迭代顺序数值差异。

### 3.3 Sinkhorn 迭代实现对比

两者的 Sinkhorn 核心循环在数学上等价（log-domain 数值稳定）：

**SOTA 的 `sinkhorn_transport`** ([ot_coupling.py:79-107](../../ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/code/projects/LDMDet/mods/ot_coupling.py#L79-L107)):
```python
log_K_mat = -cost / max(self.ot_epsilon, 1e-6)
log_u = torch.zeros(N, device=device)
log_v = torch.zeros(K, device=device)
for _ in range(self.ot_num_iters):  # 20 iterations
    log_u = torch.log(row_mass + 1e-10) - torch.logsumexp(log_K_mat + log_v.unsqueeze(0), dim=1)
    log_v = torch.log(col_mass + 1e-10) - torch.logsumexp(log_K_mat + log_u.unsqueeze(1), dim=0)
return torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))
```

**E4.3 的 `sinkhorn_transport`** ([_sinkhorn_ops.py:15-65](../../ldmdet/coupling/_sinkhorn_ops.py#L15-L65)):
```python
eps = max(epsilon, 1e-6)
log_K_mat = -cost / eps
log_u = torch.zeros(N, device=device)
log_v = torch.zeros(K, device=device)
log_row_mass = torch.log(row_mass + 1e-10)  # 预计算
log_col_mass = torch.log(col_mass + 1e-10)
for _ in range(num_iters):  # 10 iterations
    log_u = log_row_mass - torch.logsumexp(log_K_mat + log_v.unsqueeze(0), dim=1)
    log_v = log_col_mass - torch.logsumexp(log_K_mat + log_u.unsqueeze(1), dim=0)
return torch.exp(log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0))
```

**差异**:
- E4.3 预计算 `log_row_mass`/`log_col_mass`（微优化，对结果无影响）
- SOTA 20 次迭代，E4.3 10 次迭代
- Sinkhorn 在 ε=5 时收敛更快（更平滑），所以 SOTA 的 20 次已充分收敛；E4.3 的 ε=1/2 需要更多迭代，但只用 10 次（可能未完全收敛，但已足够接近）

### 3.4 解码策略对比：multinomial vs argmax

这是三者间**最核心的差异**。

**SOTA (stochastic multinomial)**:
```python
# 每次前向都随机采样
matched_idx = torch.multinomial(row_probs, 1).squeeze(-1)
```
- 同一 proposal 在不同 epoch 可能匹配不同 GT
- **多样性**: H(V|X_t) > 0, 随 ε 单调递增
- **训练信号**: 软性，每个 proposal 看到多个 GT 的混合梯度

**E4.3 (deterministic argmax)**:
```python
# 每次前向都选最大概率的 GT
matched_idx = transport_col.argmax(dim=0)
```
- 同一 proposal 在不同 epoch 倾向于匹配同一个 GT（除非 Sinkhorn 解发生跳变）
- **多样性**: H(V|X_t) ≈ 0（确定性分配）
- **训练信号**: 硬性，每个 proposal 只看到一个 GT 的梯度

### 3.5 尺度条件化 RF (仅 E4.3)

文件: [scale_conditioned_rf.py](../../ldmdet/diffusion/scale_conditioned_rf.py)

E4.3 的 RF 路径不再是常数速度，而是尺度依赖的：

$$
x_t = \alpha(t, s) x_0 + \sigma(t, s) x_1, \quad \alpha = 1 - t^{1/\kappa(s)}, \sigma = t^{1/\kappa(s)}
$$

$$
\kappa(s) = 1 + \lambda \cdot \frac{s_{\max} - s}{s_{\max}}, \quad \lambda = 0.5, s_{\max} = 0.15
$$

**关键代码** ([scale_conditioned_rf.py:54-82](../../ldmdet/diffusion/scale_conditioned_rf.py#L54-L82)):
```python
def compute_kappa(self, scales):
    return 1.0 + self.lambda_mod * (self.s_max - scales) / self.s_max.clamp(min=1e-6)

def compute_t_eff(self, t, scales):
    kappa = self.compute_kappa(scales)
    return t.clamp(min=0.0, max=1.0).pow(1.0 / kappa)
```

**性质**:
- s = s_max (大目标): κ=1, t_eff = t, 退化为标准线性 RF
- s → 0 (小目标): κ→1.5, t_eff = t^(2/3), 噪声衰减更慢

**消融结论** (来自 [方向四文档 §9](breakthrough_directions/方向四_流匹配的非线性轨迹.md)):
- E4.1 (仅尺度条件化, 无 OT Flow): mAP=0.743, **比 baseline 低 0.010**
- E4.2 (仅 OT Flow, 无尺度条件化): mAP=0.751
- E4.3 (联合): mAP=0.752
- **尺度条件化对最终精度贡献 ≈ +0.001 (在 OT Flow 基础上), 且单独使用是负面的**

---

## 4. 理论推导对比

### 4.1 平衡 OT 问题的数学等价性

**定理**: 三者求解的 OT 问题在数学上等价。

证明：设 $N$ = num_proposals, $K$ = num_gt。三者的边缘约束均为：
- Proposal 边缘: $a_i = 1/N$ (均匀)
- GT 边缘: $b_j = 1/K$ (均匀)

代价矩阵 $C_{ij} = \|noise_i - gt_j\|_2$。Sinkhorn 解为：
$$
P^*_\varepsilon = \arg\min_{P: P\mathbf{1}=a, P^T\mathbf{1}=b} \langle P, C \rangle - \varepsilon H(P)
$$

无论 $C$ 是 $[N, K]$ 还是 $[K, N]$ 朝向，得到的传输矩阵互为转置。因此**三者求解的是同一个 OT 问题**，差异仅在 ε 值与解码方式。

### 4.2 熵正则化 ε 的作用

ε 控制传输矩阵的"尖锐度":
- ε → 0: 退化为硬 OT（Hungarian），传输矩阵为置换矩阵
- ε → ∞: 退化为均匀耦合（随机配对），$P_{ij} = 1/(NK)$

**三者的 ε 选择**:
- SOTA: ε=5.0 — 较高熵，传输矩阵较平滑
- E4.3 eps1: ε=1.0 — 较低熵，传输矩阵较尖锐
- E4.3 eps2: ε=2.0 — 中等熵

**ε 与传输矩阵熵的关系** (Cuturi, 2013):
$$
H(P^*_\varepsilon) \approx H(P^*_0) + d \log \varepsilon + O(1)
$$

其中 $d$ 是数据维度（这里是 4, 检测框坐标）。因此 ε 从 1→2→5 会让传输矩阵的熵增加约 $4 \log 2 \approx 2.77$ 到 $4 \log 5 \approx 6.44$ nats。

### 4.3 CAM 定理与多样性分析

根据 Phase 5 的 **CAM 定理**（[OT_DIVERSITY_COLLAPSE_PROOF.md §12](../theory/THEORY_FRAMEWORK.md)):

> **Argmax 操作会消除 ε 对多样性的控制**。Argmax 多样性常数 ≈ 2.80, 跨所有 ε; 而 Stochastic 多样性随 ε 单调递增 (2.81 → 5.31)。

**对三者的含义**:
- **SOTA (multinomial, ε=5)**: 多样性 ≈ 5.31, 高于随机耦合 (log K)
- **E4.3 eps1 (argmax, ε=1)**: 多样性 ≈ 2.80 (常数)
- **E4.3 eps2 (argmax, ε=2)**: 多样性 ≈ 2.80 (常数, 与 eps1 相同!)

**关键悖论**: E4.3 eps1 和 eps2 的理论多样性相同 (CAM 定理), 但实际 mAP 也相同 (0.752)。这印证了 CAM 定理的预测——**argmax 下 ε 的变化不改变多样性, 因此也不改变最终精度**。

而 SOTA 的 multinomial 采样利用了 ε=5 的高多样性, 但最终精度仅比 argmax 高 0.001, 说明**在当前架构下, 多样性已不再是瓶颈**。

### 4.4 速度场与路径对比

**SOTA (标准线性 RF)**:
$$
x_t = (1-t) x_0 + t x_1, \quad v(t) = x_1 - x_0 = \text{const}
$$

**E4.3 (尺度条件化 RF)**:
$$
x_t = (1 - t^{1/\kappa(s)}) x_0 + t^{1/\kappa(s)} x_1
$$
$$
v(t, s) = \frac{t^{1/\kappa(s) - 1}}{\kappa(s)} (x_1 - x_0)
$$

速度不再是常数, 而是 $t$ 与 $s$ 的函数。**采样时需数值积分** (Euler 在 t_eff 空间)。

**理论上**: 尺度条件化应让小目标 (s 小, κ 大) 在更早的 t_eff 处去噪, 提升小目标精度。

**实际上**: E4.1 消融显示 APs 反而下降 (0.513 vs 0.521), 因为:
1. λ=0.5 调制力度不足: $t_{eff}$ 在 $t=0.5$ 时差异仅约 0.06
2. 尺度条件化干扰了 RF 的标准训练动态, 在随机耦合下路径优化更困难
3. 小目标的尺度差异本身不大 (染色体小目标面积差异 < 2x)

---

## 5. 任务层面对比

### 5.1 训练曲线关键节点

| Epoch | SOTA mAP | E4.3 eps1 mAP | E4.3 eps2 mAP |
|:-----:|:--------:|:-------------:|:-------------:|
| 1     | 0.000    | 0.000         | 0.000         |
| 10    | 0.565    | 0.577         | 0.616         |
| 20    | 0.693    | 0.698         | 0.683         |
| 40    | 0.715    | 0.733         | 0.724         |
| 59    | **0.753** | 0.738        | 0.722         |
| 80    | 0.685    | 0.744         | 0.747         |
| 94    | 0.716    | **0.752**     | 0.751         |
| 100   | 0.719    | 0.746         | **0.752**     |
| 130   | 0.715    | —             | 0.732         |
| 132   | 0.726    | —             | —             |

**观察**:
1. **SOTA 收敛最快**: epoch 59 即达 best, 比 E4.3 eps2 (epoch 100) 早 41 epochs
2. **E4.3 系列收敛较慢但波动更平缓**: 中后期 (ep 40-100) mAP 在 0.72-0.75 间小幅波动
3. **三者后期都出现过拟合**: best 后 mAP 下降 2-3 个百分点

### 5.2 Best checkpoint 精度分解

| 指标 | SOTA (ep59) | E4.3 eps1 (ep94) | E4.3 eps2 (ep100) | Δ (SOTA - E4.3 eps2) |
|------|:-----------:|:----------------:|:-----------------:|:--------------------:|
| **mAP**  | **0.753** | 0.752 | 0.752 | +0.001 |
| AP50 | 0.943 | 0.942 | 0.942 | +0.001 |
| AP75 | **0.844** | 0.839 | 0.841 | +0.003 |
| APs  | 0.521 | 0.520 | **0.523** | −0.002 |
| APm  | **0.745** | 0.744 | 0.742 | +0.003 |
| APl  | 0.642 | **0.656** | 0.634 | +0.008 |

**关键发现**:

1. **mAP 差异在统计噪声内**: 0.001 的差异远小于训练波动 (±2-4%), 三者可视为等价精度。

2. **SOTA 在 AP75/APm 上略优**: 高 IoU 阈值下的精细匹配, SOTA 的 stochastic multinomial 提供更平滑的梯度, 有利于精化。

3. **E4.3 eps1 在 APl 上最优** (0.656 vs SOTA 0.642, +0.014): 大目标从 argmax OT 中受益——大目标之间的 OT 配对更确定, 减少路径交叉。

4. **E4.3 eps2 在 APs 上最优** (0.523 vs SOTA 0.521, +0.002): 小目标从尺度条件化中略有受益, 但增量微小。

5. **ε=1→2 的影响**: E4.3 eps1→eps2, AP75 +0.002 (0.839→0.841), APl −0.022 (0.656→0.634)。增大 ε 让传输矩阵更平滑, 精细匹配略好但大目标配对质量略降。

### 5.3 训练稳定性分析

**末尾 5 epoch 波动幅度**:

| 实验 | 末尾 5 epoch mAP | 波动幅度 |
|------|:----------------:|:--------:|
| SOTA (ep 128-132) | 0.726 / 0.725 / 0.726 / 0.715 / 0.726 | ±0.6% |
| E4.3 eps1 (ep 120-124) | 0.725 / 0.720 / 0.741 / 0.741 | ±1.0% |
| E4.3 eps2 (ep 126-130) | 0.734 / 0.725 / 0.723 / 0.728 / 0.732 | ±0.5% |

**观察**:
- E4.3 eps2 末尾波动最小 (±0.5%), 与 SOTA 相当
- E4.3 eps1 末尾波动略大 (±1.0%), 但已在可接受范围
- 文档中提到的 "E4.3 波动 ±4.5%" 是指**中后期** (epoch 40-80) 的波动, 末尾已收敛

### 5.4 收敛速度与计算成本

| 指标 | SOTA | E4.3 eps1 | E4.3 eps2 |
|------|:----:|:---------:|:---------:|
| Best epoch | 59 | 94 | 100 |
| Best 时间 (相对) | 1.0× | 1.59× | 1.69× |
| Sinkhorn 迭代/前向 | 20 | 10 | 10 |
| 单步 OT 计算成本 | 2.0× | 1.0× | 1.0× |
| 总训练成本 (best 前) | 1.0× | 0.80× | 0.85× |

**结论**: E4.3 系列虽然 Sinkhorn 迭代少 (10 vs 20), 但收敛慢 (best epoch 94-100 vs 59), 总训练成本与 SOTA 相当。

---

## 6. 综合分析与结论

### 6.1 为什么三者精度近乎相同？

**根本原因**: 在 LDMDet 当前架构 (ResNet50 + FPN + 6×AdaLN-Zero SingleHead + Heun + Shifted Schedule) 下, **0.752-0.753 是当前架构下反复触及的精度水平**, 不同 OT 实现都已触及该水平。注意：这并非严格的"经验上限"——DPM-Solver++ 8-step 在 SOTA 权重上达 0.755，表明推理端仍有挖掘空间（见 [对照分析](论文草稿理论主张vs实验结果对照分析.md) §4 C11）。

具体机制:
1. **OT 问题的等价性**: 三者求解同一平衡 OT, 差异仅在解码与 ε
2. **CAM 定理的预测**: argmax 下 ε 不影响多样性, 所以 E4.3 eps1=eps2 (都 0.752)
3. **多样性已饱和（待验证假说）**: SOTA 的高多样性 (5.31) 仅比 argmax (2.80) 多 0.001 mAP, **在单种子噪声内**，说明多样性可能不再是瓶颈——但"多样性饱和"需多种子验证后才能定论
4. **尺度条件化无效**: λ=0.5 调制力度不足, 对最终精度无贡献

### 6.2 SOTA (0.753) 的微小优势来自哪里？

> **统计限制**: +0.001 mAP **在单种子噪声内**（典型方差 0.003-0.008），以下"可能来源"是**机制假说而非统计结论**。要证实任一机制为真，需多种子 paired t-test 显示 Δ > 0 显著。

**+0.001 mAP 的可能来源（机制假说）**:
1. **stochastic multinomial 的正则化效应**: 软配对提供隐式正则化, 减少过拟合
2. **ε=5 的更平滑传输矩阵**: 20 次 Sinkhorn 迭代 + 高熵, OT 解更稳定
3. **更快的收敛**: best@59 vs 100, 在过拟合前抓住更好的局部最优
4. **AP75/APm 优势**: stochastic 采样让模型看到更多 GT 配对变体, 有利于精细匹配

### 6.3 E4.3 系列 (0.752) 的价值

尽管精度持平, E4.3 系列有独立价值:
1. **代码架构更清晰**: `coupling=dict(type='ot_flow', ...)` 注册式策略, 易于扩展
2. **计算效率更高**: 10 次 Sinkhorn 迭代 vs 20 次 (单步成本减半)
3. **可解释性更强**: argmax 是确定性解码, 易于调试与可视化
4. **APl 优势**: E4.3 eps1 的 APl=0.656 是所有实验中最高的 (+0.014 vs SOTA)

### 6.4 风险与限制

**三者共同的风险**:
1. **过拟合**: best 后 mAP 下降 2-3%, 需要早停或 SWA
2. **训练精度水平**: 0.752-0.753 是当前架构**训练阶段**反复触及的精度水平（非"硬上限"——推理端 DPM-Solver++ 已达 0.755），进一步提升训练精度需架构级改进
3. **OT 计算开销**: 每次前向需 O(N·K) 的 Sinkhorn 迭代, 在大 batch 下成为瓶颈
4. **单种子限制**: 本报告所有 mAP 比较为单种子结果，0.001-0.002 差异在统计噪声内，**不可做"等价"或"优劣"的因果归因**

**E4.3 系列的额外风险**:
1. **尺度条件化代码复杂度**: 修改了 RF 路径、采样器、损失多处, 但收益为零
2. **argmax 的训练不稳定风险**: 虽然末尾稳定, 但中后期波动大于 stochastic

---

## 7. 建议与后续方向

### 7.1 短期建议

1. **保留 SOTA (stochastic multinomial, ε=5) 作为主线**: 收敛快, 精度最优, 稳定性好
2. **E4.3 eps2 作为对照基线**: argmax 路径的代表, 用于消融研究
3. **考虑移除尺度条件化 RF**: E4.1 消融已证明其无效, 移除可简化代码

### 7.2 中期方向

1. **multinomial + ε=2 的 E4.3 变体**: 结合 SOTA 的 stochastic 优势与 E4.3 的低熵精确性
2. **SWA / EMA**: 三者都存在过拟合, 权重平均可能进一步提升 0.001-0.003
3. **Group-Hierarchical OT**: SOTA_ANALYSIS.md 提到 group_hierarchical_stoch 达到 0.752, 可与 E4.3 架构结合

### 7.3 长期方向

1. **2-RectFlow (E4.4)**: reflow 后路径更直, 少步采样精度提升, 可能突破 0.753 训练精度水平
2. **架构级改进**: 更强的 backbone (ConvNeXt-V2 + MAE 预训练)、更好的 neck (BiFPN)、更深的 head
3. **DPM-Solver++ 推理**: 离线实验已证明 DPM-2 8-step 在 SOTA 权重上达 0.755（已突破训练端的 0.753 水平，来自推理端改进）, 可与训练改进叠加

---

## 附录 A: 关键代码位置索引

| 组件 | 文件 | 关键行 |
|------|------|--------|
| SOTA OT 耦合 | [ot_coupling.py](../../ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/code/projects/LDMDet/mods/ot_coupling.py) | L21-L196 |
| SOTA Sinkhorn | 同上 | L79-L107 |
| SOTA multinomial 采样 | 同上 | L182-L196 |
| E4.3 OT Flow 耦合 | [ot_flow_coupling.py](../../ldmdet/coupling/ot_flow_coupling.py) | L18-L99 |
| E4.3 Sinkhorn (共享) | [_sinkhorn_ops.py](../../ldmdet/coupling/_sinkhorn_ops.py) | L15-L65 |
| 尺度条件化 RF | [scale_conditioned_rf.py](../../ldmdet/diffusion/scale_conditioned_rf.py) | L29-L194 |
| SOTA 配置 | [ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py](../../ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/code/projects/LDMDet/configs/_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py) | L11-L22 |
| E4.3 eps1 配置 | [nonlinear_trajectory.py](../../experiments/configs/ldmdet/directions/nonlinear_trajectory/nonlinear_trajectory.py) | L23-L40 |
| E4.3 eps2 配置 | [nonlinear_trajectory_e43_eps2.py](../../experiments/configs/ldmdet/directions/nonlinear_trajectory/nonlinear_trajectory_e43_eps2.py) | L13-L21 |
| SOTA 训练曲线 | [metrics.json](../../ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/metrics.json) | — |
| E4.3 eps2 训练日志 | [train.log](../../work_dirs/nonlinear_trajectory_e43_eps2/train.log) | — |
| 方向四研究文档 | [方向四_流匹配的非线性轨迹.md](breakthrough_directions/方向四_流匹配的非线性轨迹.md) | — |

## 附录 B: 完整 ε 扫描数据

| 实验 | ε | 解码 | 尺度条件化 | Best mAP | APs | APm | APl | Best Epoch |
|------|:-:|:----:|:----------:|:--------:|:---:|:---:|:---:|:----------:|
| SOTA | 5.0 | multinomial | ✗ | **0.753** | 0.521 | **0.745** | 0.642 | 59 |
| E4.3 eps1 | 1.0 | argmax | ✓ (λ=0.5) | 0.752 | 0.520 | 0.744 | **0.656** | 94 |
| E4.3 eps2 | 2.0 | argmax | ✓ (λ=0.5) | 0.752 | **0.523** | 0.742 | 0.634 | 100 |
| E4.3 eps3 | 3.0 | argmax | ✓ (λ=0.5) | 0.750 | 0.505 | 0.742 | 0.625 | 70 |
| E4.3 multinomial | 1.0 | multinomial | ✓ (λ=0.5) | 0.748 | — | — | — | 72 |
| E4.2 (仅 OT Flow) | 1.0 | argmax | ✗ | 0.751 | 0.516 | 0.741 | 0.666 | 81 |
| E4.1 (仅尺度条件化) | — | random | ✓ (λ=0.5) | 0.743 | 0.513 | 0.737 | 0.664 | 85 |

**关键趋势（均基于单种子，0.001-0.004 mAP 差异在统计噪声内）**:
- argmax 路径: ε=1→2→3, mAP = 0.752→0.752→0.750 (CAM 定理预测的 ε-不变性, ε=3 时略降 0.002——**轻微反驳但未达统计显著**)
- argmax vs multinomial (ε=1): 0.752 vs 0.748 (**方向与 SOTA 的 multinomial 优势相反, Δ=0.004 在单种子噪声内, 不能"推翻"论文主张, 仅表明 E4.3 架构下未复现**)
- 尺度条件化: 单独使用 (E4.1) 反而下降, 联合 OT Flow (E4.3) 无净增量
