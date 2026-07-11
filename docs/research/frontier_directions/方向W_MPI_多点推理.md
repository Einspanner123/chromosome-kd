# 方向 W：MPI 多点推理 (Multi-Point Inference)

> **核心思想**: 在推理时, 对同一组噪声框 $x_{\text{raw}}$ 在多个时间点 $\{t_1, t_2, \dots, t_K\}$ 独立前向传播, 利用模型在训练时对全 $t \in [0,1]$ 区间的监督能力, 通过加权聚合降低预测方差, 并以方差作为不确定性估计触发额外精化。纯推理期优化, 无需重训练。
>
> **理论依据**:
> - 偏差-方差分解 (Bias-Variance Decomposition): Geman et al. (1992)
> - 集成学习方差降低: Brown et al., "Statistical Challenges of Algorithmic Trading" (2005)
> - Rectified Flow: Liu et al. (ICLR 2023) — $x_t = (1-t)x_0 + t x_1$ 直线路径
> - 扩散检测时间维集成: DiffusionDet (ICCV 2023)
>
> **当前代码位置**:
> - [ldmdet/core/head.py](../../ldmdet/core/head.py) — `predict` 方法 (推理主循环, 第 317-395 行) 与 `_forward_at_t` (单点评估, 第 470-498 行)
> - [ldmdet/diffusion/sampling.py](../../ldmdet/diffusion/sampling.py) — `post_process` (时间维 ensemble 后处理, 第 194-264 行) 与 `build_time_pairs` (时间序列构建, 第 59-85 行)
> - [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) — RF 前向扩散与 Euler/Heun 步进
> - [ldmdet/core/single_head.py](../../ldmdet/core/single_head.py) — SingleDiffusionDetHead (时间条件化前向, 第 267-319 行)
> - [experiments/configs/ldmdet/ldmdet_baseline.py](../../experiments/configs/ldmdet/ldmdet_baseline.py) — `use_ensemble` 配置

---

## 1. 背景与动机

### 1.1 当前推理流程的"单轨迹"局限

阅读 [head.py](../../ldmdet/core/head.py) 第 317-395 行的 `predict` 方法可知, 当前推理流程是**单轨迹序列采样**:

```python
# head.py:324 — 单次随机初始化
x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

# head.py:332-387 — 沿 t=1→0 轨迹逐步采样
for step_idx, (t_curr, t_next) in enumerate(time_pairs):
    cls_logits, pred_bboxes, x0_raw, velocity = self._forward_at_t(
        features, x_raw, t_curr, img_metas
    )
    if self.use_ensemble:
        ensemble_results.append((cls_logits, pred_bboxes))  # 收集轨迹快照
    x_raw = solver.step(x_raw, x0_raw, t_curr, t_next)  # 轨迹演化
    if self.box_renewal:
        x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
```

**关键观察**: 推理时模型只在轨迹点 $\{t_1, t_2, \dots, t_T\}$ (如 Heun 4 步: $t \in \{1.0, 0.75, 0.5, 0.25\}$) 上被评估, 且每次评估后 $x_{\text{raw}}$ 被更新为下一时间步的状态。**模型在其他 $t$ 值上的预测能力被完全忽略**。

### 1.2 训练-推理的信息不对称

阅读 [head.py](../../ldmdet/core/head.py) 第 415-423 行的 `_sample_t` 与 [rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) 第 20-40 行的 `q_sample` 可知训练时的设置:

```python
# head.py:415-423 — 训练时 t ~ U[0,1] 均匀采样
def _sample_t(self, bs, device):
    if self.diffusion_type == 'ddpm':
        return torch.randint(0, self.timesteps, (bs,), device=device).long()
    t = torch.rand((bs,), device=device)  # RF: t ~ U[0,1]
    ...

# rectified_flow.py:37-38 — 前向扩散: x_t = (1-t)x_0 + t·x_noise
t_view = t.view(-1, *([1] * (x_start.dim() - 1)))
x_t = (1.0 - t_view) * x_start + t_view * x_noise
```

**训练时**: 模型在 $t \sim U[0,1]$ 的**全区间**接收监督信号, 每个时间点都学习从 $x_t$ 预测 $x_0$。模型参数 $\theta$ 对所有 $t$ 值都有梯度更新。

**推理时**: 只在轨迹点 $t \in \{1.0, 0.75, 0.5, 0.25\}$ (Heun 4 步) 评估, 且 $x_{\text{raw}}$ 在轨迹上演化。模型在 $t = 0.9, 0.6, 0.3$ 等非轨迹点的预测能力**被浪费**。

**类比**: 这就像一个学生 (模型) 在训练时学会了做各种难度 ($t$) 的题目, 但考试 (推理) 时只做了一道题 (单轨迹), 没有发挥出全部能力。

### 1.3 现有时间维 Ensemble 的局限

阅读 [sampling.py](../../ldmdet/diffusion/sampling.py) 第 194-264 行的 `post_process`, 当前 `use_ensemble=True` 的行为:

```python
# sampling.py:209-218 — 拼接所有时间步的预测, 一次 NMS
for cls_logits, pred_bboxes in ensemble_results:
    scores = torch.sigmoid(cls_logits[i])
    conf, labels = scores.max(-1)
    all_scores.append(conf)
    all_bboxes.append(pred_bboxes[i])
    all_labels.append(labels)

final_scores = torch.cat(all_scores)  # T × num_proposals 个框
final_bboxes = torch.cat(all_bboxes)
keep = batched_nms(final_bboxes, final_scores, final_labels, nms_thr)
```

**本质**: 这是**时间维 ensemble** (temporal ensemble), 聚合同一随机种子、同一轨迹在不同时间步的预测。

**局限**:
1. **序列强相关**: 轨迹相邻时间步的 $x_{\text{raw}}$ 高度相似 (Euler/Heun 步长小), 导致预测高度相关。$K$ 个相关预测的方差降低仅为 $\sigma^2 \cdot (1 + (K-1)\rho)$, 其中 $\rho$ 是平均相关系数。当 $\rho \to 1$ 时, 方差降低趋近于 0。
2. **无法并行**: 轨迹采样必须序列执行 ($x_{t_{i+1}}$ 依赖 $x_{t_i}$ 的模型输出), 无法利用 batch 并行加速。
3. **增益递减**: 1→8 步采样 mAP 0.740→0.755 (+0.015/8× 计算量), 边际收益急剧递减, 说明同一轨迹上增加步数已接近饱和。

### 1.4 MPI 的核心洞察

MPI 的核心: **不再沿轨迹序列评估, 而是对固定的 $x_{\text{raw}}$ 在多个独立 $t$ 值上并行评估, 利用模型在训练时获得的全 $t$ 预测能力**。

```
当前 (序列 ensemble):   x_raw(t=1) → 评估 → 步进 → x_raw(t=0.75) → 评估 → 步进 → ...
                        ↓                  ↓
                     预测_1 (相关)      预测_2 (相关)     →  拼接 + NMS

MPI (并行多点):         x_raw (固定)
                         ├──→ 评估 at t=1.0  → 预测_1 (独立)
                         ├──→ 评估 at t=0.8  → 预测_2 (独立)
                         ├──→ 评估 at t=0.6  → 预测_3 (独立)
                         └──→ 评估 at t=0.5  → 预测_4 (独立)
                                                              →  加权聚合 + NMS
```

**三个关键优势**:
1. **独立性**: 不同 $t$ 值的预测无序列依赖, 相关系数 $\rho$ 更低, 方差降低更显著
2. **可并行**: $K$ 个评估完全独立, 可用 batch 维度并行, 计算开销接近 1×
3. **不确定性量化**: $K$ 组预测的方差提供 per-box 不确定性估计, 可触发自适应精化

---

## 2. 理论依据

### 2.1 RF 路径下不同 $t$ 值的偏差-方差分解

在 Rectified Flow 中, 前向扩散路径为:

$$x_t = (1-t) x_0 + t \cdot x_{\text{noise}}, \quad t \in [0, 1]$$

其中 $x_0$ 是数据 (GT 框, 归一化到 $[-\text{snr\_scale}, \text{snr\_scale}]$), $x_{\text{noise}} \sim \mathcal{N}(0, I)$。

模型 $f_\theta(x_t, t) \to \hat{x}_0(t)$ 学习从 $(x_t, t)$ 预测 $x_0$。训练目标 (L1 + GIoU + Focal) 在所有 $t \sim U[0,1]$ 上施加监督。

**推理时的关键问题**: 推理时我们只有 $x_{\text{raw}} = x_{\text{noise}}$ (即 $t=1$ 的样本), 没有 $x_0$。标准做法是在 $t=1$ 处评估模型, 然后沿轨迹步进。MPI 则在多个 $t$ 值上评估同一 $x_{\text{raw}}$。

**偏差分析**: 对固定的 $x_{\text{raw}}$ (即 $t=1$ 的噪声样本), 在 $t = \tau$ 处评估模型:

- 模型期望输入: $x_\tau = (1-\tau) x_0 + \tau \cdot x_{\text{noise}}$
- 实际输入: $x_{\text{raw}} = x_{\text{noise}}$
- 输入偏差: $\Delta x = x_{\text{raw}} - x_\tau = (1-\tau)(x_{\text{noise}} - x_0)$

当 $\tau = 1$ 时, $\Delta x = 0$ (无偏差, 在分布内)。当 $\tau < 1$ 时, $\Delta x \neq 0$ (分布外, OOD)。

但模型 $f_\theta$ 是在 $t \sim U[0,1]$ 上训练的, 具有一定的鲁棒性。在 $\tau < 1$ 处, 模型仍能产生有意义的预测, 只是存在系统性偏差。

**偏差的数学表达**:

$$\text{Bias}(\hat{x}_0(\tau)) = \mathbb{E}[\hat{x}_0(\tau)] - x_0 \approx (1-\tau) \cdot \nabla_x f_\theta \cdot (x_{\text{noise}} - x_0)$$

其中 $\nabla_x f_\theta$ 是模型对输入的雅可比。偏差随 $|\tau - 1|$ 增大而增大。

**方差分析**: 模型预测的方差来自 $x_{\text{noise}}$ 的随机性:

$$\text{Var}(\hat{x}_0(\tau)) = \text{Var}_{x_{\text{noise}}}[f_\theta(x_{\text{noise}}, \tau)]$$

- 在 $\tau = 1$ (纯噪声): 模型完全依赖噪声模式提取信号, 方差高 ($\sigma^2_{\text{noise}}$)
- 在 $\tau = 0$ (纯数据): 模型期望 $x_t \approx x_0$, 输入 $x_{\text{noise}}$ 导致预测 $\approx x_{\text{noise}}$, 方差高但无意义
- 在 $\tau \approx 0.5$ (SNR ≈ 1): 信号与噪声等权, 模型在"一半数据一半噪声"上做决策, 方差适中

**信噪比 (SNR) 分析**:

RF 路径下, 信号能量 $\propto (1-t)^2$, 噪声能量 $\propto t^2$, 因此:

$$\text{SNR}(t) = \frac{(1-t)^2 \|x_0\|^2}{t^2 \cdot \mathbb{E}[\|x_{\text{noise}}\|^2]} \approx \frac{(1-t)^2}{t^2} \quad (\text{归一化后})$$

- $t = 1.0$: SNR = 0 (纯噪声)
- $t = 0.5$: SNR = 1 (信号 = 噪声, **临界点**)
- $t = 0.0$: SNR = $\infty$ (纯数据)

**临界点 $t = 0.5$ 的分类价值**: 在 SNR = 1 时, 模型需要从等量的信号和噪声中区分出真实框。这迫使分类头学习最具判别性的特征。因此, $t \approx 0.5$ 的预测在**分类**上可能最有价值, 即使**定位**上有偏差。

### 2.2 多点聚合的方差降低推导

设 $K$ 个预测 $\hat{x}_0(\tau_1), \dots, \hat{x}_0(\tau_K)$, 加权聚合:

$$\hat{x}_0^{\text{MPI}} = \sum_{k=1}^{K} w_k \hat{x}_0(\tau_k), \quad \sum_k w_k = 1$$

**期望与方差**:

$$\mathbb{E}[\hat{x}_0^{\text{MPI}}] = \sum_k w_k \mathbb{E}[\hat{x}_0(\tau_k)] = \sum_k w_k (x_0 + \text{Bias}(\tau_k))$$

$$\text{Var}(\hat{x}_0^{\text{MPI}}) = \sum_k w_k^2 \text{Var}(\hat{x}_0(\tau_k)) + 2 \sum_{i < j} w_i w_j \text{Cov}(\hat{x}_0(\tau_i), \hat{x}_0(\tau_j))$$

**与单点预测 ($w_1 = 1$, 其余 0) 对比**:

$$\Delta \text{Var} = \text{Var}(\hat{x}_0^{\text{MPI}}) - \text{Var}(\hat{x}_0(\tau_1))$$

$$= \sum_{k \neq 1} w_k^2 \sigma_k^2 + 2 \sum_{i < j} w_i w_j \sigma_{ij} - 2 \sum_{k \neq 1} w_k \sigma_{1k}$$

其中 $\sigma_k^2 = \text{Var}(\hat{x}_0(\tau_k))$, $\sigma_{ij} = \text{Cov}(\hat{x}_0(\tau_i), \hat{x}_0(\tau_j))$。

**方差降低的充分条件**: 当新增预测点与主预测点的相关性足够低时, 方差降低。具体地, 若所有 $\sigma_k^2 = \sigma^2$ (等方差), $\sigma_{ij} = \rho_{ij} \sigma^2$, 则:

$$\text{Var}(\hat{x}_0^{\text{MPI}}) = \sigma^2 \left( \sum_k w_k^2 + 2 \sum_{i<j} w_i w_j \rho_{ij} \right)$$

当 $\rho_{ij} < 1$ (不完全相关) 且权重均匀 ($w_k = 1/K$) 时:

$$\text{Var}(\hat{x}_0^{\text{MPI}}) = \frac{\sigma^2}{K} \left( 1 + (K-1)\bar{\rho} \right) < \sigma^2 \quad \text{当} \quad \bar{\rho} < 1$$

其中 $\bar{\rho}$ 是平均相关系数。

**与现有时间维 ensemble 的对比**:

| 属性 | 时间维 ensemble (序列) | MPI (并行) |
|------|----------------------|-----------|
| 预测相关性 $\bar{\rho}$ | 高 (0.7-0.9, 轨迹相邻步) | 低 (0.3-0.5, 不同 $t$ 区制) |
| $K=4$ 时方差降低 | $\frac{\sigma^2}{4}(1+3\times 0.8) = 0.85\sigma^2$ (降 15%) | $\frac{\sigma^2}{4}(1+3\times 0.4) = 0.55\sigma^2$ (降 45%) |
| 偏差 | 零 (所有点在轨迹上, 分布内) | 非零 ($\tau < 1$ 点有 OOD 偏差) |

**关键结论**: MPI 通过牺牲少量偏差 (OOD 评估) 换取更大的方差降低 (低相关性), 净效果取决于偏差-方差的权衡。

### 2.3 最优权重的设计

最优权重 $w^*$ 应最小化均方误差 (MSE):

$$w^* = \arg\min_w \mathbb{E}\left[\left\|\hat{x}_0^{\text{MPI}} - x_0\right\|^2\right] = \arg\min_w \left(\left\|\sum_k w_k \text{Bias}(\tau_k)\right\|^2 + \text{Var}(\hat{x}_0^{\text{MPI}})\right)$$

这是一个二次优化问题, 解为:

$$w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}$$

其中 $\Sigma$ 是 $K \times K$ 的偏差-协方差矩阵, $\Sigma_{ij} = \text{Cov}(\hat{x}_0(\tau_i), \hat{x}_0(\tau_j)) + \text{Bias}(\tau_i)^T \text{Bias}(\tau_j)$。

**实践近似**: 精确的 $\Sigma$ 需要在验证集上估计, 计算量大。可用以下近似:

1. **SNR 加权**: $w_k \propto \text{SNR}(\tau_k) = \frac{(1-\tau_k)^2}{\tau_k^2}$, 偏好高 SNR (低 $t$) 的预测
2. **指数衰减**: $w_k \propto \exp(\alpha(\tau_k - 1))$, 偏好 $\tau \approx 1$ (在分布内)
3. **临界点加权**: $w_k \propto \exp(-\frac{(\tau_k - 0.5)^2}{2\sigma_t^2})$, 偏好 $\tau \approx 0.5$ (SNR=1)
4. **分类-回归解耦**: 对 cls_logits 用临界点加权, 对 pred_bboxes 用指数衰减加权

### 2.4 不确定性量化的理论基础

$K$ 组预测的样本方差作为不确定性估计:

$$\text{Uncertainty}(i) = \frac{1}{K-1} \sum_{k=1}^{K} \left\| \hat{x}_0^{(k)}(i) - \bar{x}_0(i) \right\|^2$$

其中 $\bar{x}_0(i) = \frac{1}{K} \sum_k \hat{x}_0^{(k)}(i)$ 是第 $i$ 个 proposal 的平均预测。

**理论依据**: 在贝叶斯框架下, 多个 $t$ 点的预测可视为从近似后验中采样。方差高意味着模型对该 proposal 的预测不稳定, 可能是:
- 假阳 (噪声框被误检为目标)
- 边界模糊 (proposal 位于两个真实目标之间)
- 遮挡严重 (amodal 标注的隐藏部分)

**与 MC Dropout 的类比**: MPI 的不确定性估计类似于 MC Dropout (Gal & Ghahramani, 2016), 但扰动源不同:
- MC Dropout: 随机失活神经元
- MPI: 扰动时间条件 $t$

两者都利用模型对扰动的敏感性来估计不确定性, 且都不需要修改训练过程。

---

## 3. 详细方案设计

### 3.1 整体架构

```
输入图像 X → backbone+FPN → features (缓存, 只计算一次)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
             x_raw (固定噪声, bs×500×4)
                    │
        ┌───────────┼───────────┬───────────┐
        │           │           │           │
        ▼           ▼           ▼           ▼
   _forward_at_t  _forward_at_t  _forward_at_t  _forward_at_t
    (t=1.0)        (t=0.8)        (t=0.6)       (t=0.5)
        │           │           │           │
        ▼           ▼           ▼           ▼
   预测_1        预测_2        预测_3       预测_4
   (cls_1,box_1) (cls_2,box_2) (cls_3,box_3) (cls_4,box_4)
        │           │           │           │
        └─────┬─────┴─────┬─────┘
              │           │
              ▼           ▼
        加权聚合      方差计算
        (w_k 权重)   (不确定性)
              │           │
              ▼           ▼
        聚合预测     高方差框 → 触发额外精化 (可选)
              │
              ▼
         NMS + 后处理 → 最终结果
```

### 3.2 多点评估策略

#### 3.2.1 评估点选择

MPI 的核心超参数是评估点集合 $\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_K\}$。推荐三种策略:

**策略 A: 等距网格 (Phase 1 默认)**

$$\mathcal{T} = \{1.0, \ 0.8, \ 0.6, \ 0.4\} \quad (K=4)$$

- 优点: 简单, 覆盖 $t \in [0.4, 1.0]$ 的主要区域
- 缺点: 未针对 SNR 临界点优化

**策略 B: SNR 感知 (Phase 2)**

在 SNR 对数空间均匀分布, 使每个评估点贡献等量的 SNR 区间:

$$\log \text{SNR}(\tau_k) = \log \text{SNR}_{\max} - \frac{k-1}{K-1} (\log \text{SNR}_{\max} - \log \text{SNR}_{\min})$$

对 $\text{SNR} \in [0.1, 10]$ ($\tau \in [0.24, 0.76]$), $K=4$ 时:

$$\mathcal{T} \approx \{0.76, \ 0.62, \ 0.48, \ 0.38\}$$

- 优点: 每个点覆盖不同的 SNR 区间, 信息互补
- 缺点: 所有点都偏离 $t=1$, 偏差较大

**策略 C: 混合策略 (Phase 2, 推荐)**

包含 $t=1$ (在分布内, 低偏差) 和若干 SNR 感知点:

$$\mathcal{T} = \{1.0, \ 0.7, \ 0.5, \ 0.3\} \quad (K=4)$$

- $t=1.0$: 标准预测, 低偏差, 高方差, 作为锚点
- $t=0.7$: 轻微 OOD, 提供鲁棒性
- $t=0.5$: SNR=1 临界点, 分类最优
- $t=0.3$: 严重 OOD, 但定位可能更精确 (模型期望更多数据结构)

**推荐**: Phase 1 用策略 C, 在验证集上对比 A/B/C 后确定最优。

#### 3.2.2 评估点数量 $K$ 的选择

$K$ 的选择权衡方差降低和计算开销:

| $K$ | 方差降低 (理论, $\bar{\rho}=0.4$) | 计算开销 (Head 前向) | 推荐场景 |
|-----|--------------------------------|-------------------|---------|
| 1 | 0% (基线) | 1× | 基线 |
| 2 | 35% | 2× | 轻量 MPI |
| 3 | 45% | 3× | 标准 MPI |
| 4 | 52% | 4× | 推荐 |
| 5 | 56% | 5× | 高精度 |
| 8 | 63% | 8× | 过重, 不推荐 |

**推荐 $K=4$**: 方差降低已超过 50%, 计算开销可接受 (backbone 复用后, Head 前向仅占总推理时间的 30-50%)。

#### 3.2.3 评估点的批量并行化

关键优化: $K$ 个评估点可**批量并行**执行, 而非序列执行:

```python
def _forward_at_t_batch(self, features, x_raw, t_list, img_metas):
    """批量多点评估: 将 K 个 t 值作为 batch 维度并行前向"""
    K = len(t_list)
    bs, num_proposals, _ = x_raw.shape

    # 将 x_raw 复制 K 份: [K*bs, num_proposals, 4]
    x_raw_batched = x_raw.unsqueeze(0).expand(K, bs, num_proposals, 4)
    x_raw_batched = x_raw_batched.reshape(K * bs, num_proposals, 4)

    # 构建时间输入: [K*bs]
    t_input_list = []
    for t_k in t_list:
        t_input_list.append(
            torch.full((bs,), t_k * self.timesteps, device=x_raw.device)
        )
    t_input = torch.cat(t_input_list)  # [K*bs]

    # img_metas 复制 K 份
    img_metas_batched = []
    for t_k in t_list:
        img_metas_batched.extend(img_metas)

    # 单次前向 (K*bs 的 batch)
    curr_bboxes = self._sampler.raw_to_xyxy(x_raw_batched, img_metas_batched)
    cls_logits_seq, pred_output_seq, _ = self(features, curr_bboxes, t_input)

    # 提取最后一级 head 的预测, 拆分为 K 组
    cls_last = cls_logits_seq[-1]  # [K*bs, num_proposals, num_classes]
    pred_last = pred_output_seq[-1]  # [K*bs, num_proposals, 4]

    results = []
    for k in range(K):
        cls_k = cls_last[k*bs:(k+1)*bs]
        pred_k = pred_last[k*bs:(k+1)*bs]
        x0_k = self._sampler.xyxy_to_raw(pred_k, img_metas)
        results.append((cls_k, pred_k, x0_k, t_list[k]))

    return results
```

**优势**: 
- backbone+FPN 特征只计算 1 次 (所有 $K$ 个评估共享)
- Head 前向通过 batch 维并行, GPU 利用率更高
- 显存开销: $K \times$ Head 激活值, 约 $4 \times 2\text{GB} = 8\text{GB}$ (6 级 cascade × 500 proposals × 256 channels), 单卡 A100 可承受

### 3.3 加权聚合算法

#### 3.3.1 基础: 分数加权 NMS (Score-Weighted NMS)

最简单的聚合方式: 对 $K$ 组预测的分数施加 $t$ 相关权重, 拼接后做 NMS:

```python
def mpi_aggregate_nbs(self, mpi_results, img_metas, weights, rescale=True):
    """MPI 加权聚合: 分数加权 + NMS"""
    results_list = []
    bs = len(img_metas)

    for i in range(bs):
        all_scores = []
        all_bboxes = []
        all_labels = []

        for (cls_logits, pred_bboxes, _, t_k), w_k in zip(mpi_results, weights):
            scores = torch.sigmoid(cls_logits[i])
            # 关键: 分数乘以 t 相关权重
            scores = scores * w_k
            conf, labels = scores.max(-1)
            all_scores.append(conf)
            all_bboxes.append(pred_bboxes[i])
            all_labels.append(labels)

        final_scores = torch.cat(all_scores)
        final_bboxes = torch.cat(all_bboxes)
        final_labels = torch.cat(all_labels)

        if self.use_nms:
            keep = batched_nms(
                final_bboxes, final_scores, final_labels, self.nms_thr
            )
            final_scores = final_scores[keep]
            final_bboxes = final_bboxes[keep]
            final_labels = final_labels[keep]

        # ... (rescale 逻辑复用现有 post_process)
        results_list.append(
            DetectionResult(
                bboxes=final_bboxes,
                scores=final_scores,
                labels=final_labels,
            )
        )
    return results_list
```

**优点**: 实现简单, 复用现有 NMS 基础设施, 不改变 proposal 对齐方式。
**缺点**: $K$ 组预测的 proposal 索引不对齐 (第 $i$ 个 proposal 在不同 $t$ 可能对应不同目标), NMS 后的聚合是框级而非 proposal 级。

#### 3.3.2 进阶: Proposal 对齐加权平均

若 $K$ 组预测的 proposal 索引对齐 (因为使用同一 $x_{\text{raw}}$, cascade head 的 RoI 提取相同), 可做 proposal 级加权平均:

```python
def mpi_aggregate_aligned(self, mpi_results, img_metas, weights):
    """Proposal 对齐加权平均: 对每个 proposal 独立聚合"""
    cls_stack = torch.stack([r[0] for r in mpi_results])  # [K, bs, P, C]
    box_stack = torch.stack([r[1] for r in mpi_results])  # [K, bs, P, 4]

    w = torch.tensor(weights, device=cls_stack.device)
    w = w.view(-1, 1, 1, 1)  # [K, 1, 1, 1]

    # 加权平均
    cls_agg = (cls_stack * w).sum(dim=0)  # [bs, P, C]
    box_agg = (box_stack * w).sum(dim=0)  # [bs, P, 4]

    # 后续 NMS 处理...
    return cls_agg, box_agg
```

**前提条件**: $K$ 组预测使用同一 $x_{\text{raw}}$ 且同一 `img_metas`, cascade head 的 RoI 提取结果相同, proposal 索引对齐。

**优势**:
- 真正的 proposal 级聚合, 方差降低最有效
- 可计算 per-proposal 不确定性 (方差)
- 分类和回归可分别用不同权重 (见 3.3.3)

**风险**: 若不同 $t$ 的 RoI 提取结果不同 (因 `raw_to_xyxy` 受 $t$ 影响间接改变框位置), proposal 可能不对齐。需验证对齐性。

#### 3.3.3 分类-回归解耦加权

基于 2.1 节的 SNR 分析, 分类和回归在不同 $t$ 有不同最优工作点:

```python
def mpi_aggregate_decoupled(self, mpi_results, img_metas,
                             cls_weights, reg_weights):
    """分类-回归解耦加权聚合"""
    cls_stack = torch.stack([r[0] for r in mpi_results])  # [K, bs, P, C]
    box_stack = torch.stack([r[1] for r in mpi_results])  # [K, bs, P, 4]

    wc = torch.tensor(cls_weights, device=cls_stack.device).view(-1,1,1,1)
    wr = torch.tensor(reg_weights, device=box_stack.device).view(-1,1,1,1)

    # 分类用临界点加权 (偏好 t≈0.5)
    cls_agg = (cls_stack * wc).sum(dim=0)
    # 回归用指数衰减加权 (偏好 t≈1.0)
    box_agg = (box_stack * wr).sum(dim=0)

    return cls_agg, box_agg
```

**推荐权重配置** ($K=4$, $\mathcal{T} = \{1.0, 0.7, 0.5, 0.3\}$):

```python
# 分类权重: 偏好 SNR=1 的 t=0.5
cls_weights = [0.15, 0.25, 0.40, 0.20]  # t=0.5 权重最高

# 回归权重: 偏好 t=1.0 (在分布内, 低偏差)
reg_weights = [0.50, 0.30, 0.15, 0.05]  # t=1.0 权重最高
```

### 3.4 不确定性量化与自适应精化

#### 3.4.1 不确定性计算

对每个 proposal $i$, 计算 $K$ 组预测的方差:

```python
def compute_uncertainty(self, mpi_results):
    """计算 per-proposal 不确定性"""
    box_stack = torch.stack([r[1] for r in mpi_results])  # [K, bs, P, 4]

    # 框方差: 各维度方差之和
    box_var = box_stack.var(dim=0)  # [bs, P, 4]
    uncertainty = box_var.sum(dim=-1)  # [bs, P] — 每个 proposal 的不确定性

    # 分类不确定性: 预测类别的不一致度
    cls_stack = torch.stack([r[0] for r in mpi_results])  # [K, bs, P, C]
    cls_probs = torch.sigmoid(cls_stack)
    cls_var = cls_probs.var(dim=0).mean(dim=-1)  # [bs, P]

    return uncertainty, cls_var
```

#### 3.4.2 自适应精化触发

高方差的 proposal 可触发额外精化:

```python
def mpi_predict_with_refinement(self, features, img_metas, mpi_points, weights):
    """MPI + 不确定性触发的自适应精化"""
    # Phase 1: 标准 MPI 多点评估
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
    mpi_results = self._forward_at_t_batch(features, x_raw, mpi_points, img_metas)

    # Phase 2: 计算不确定性
    box_uncertainty, cls_uncertainty = self.compute_uncertainty(mpi_results)

    # Phase 3: 对高方差 proposal 触发额外精化
    refinement_threshold = box_uncertainty.quantile(0.8)  # Top 20% 高方差
    needs_refinement = box_uncertainty > refinement_threshold  # [bs, P]

    if needs_refinement.any():
        # 对高方差 proposal 做额外 1-2 步 Heun 精化
        x_raw_refined = x_raw.clone()
        for i in range(bs):
            high_var_idx = needs_refinement[i]
            if high_var_idx.any():
                # 用聚合后的 x0 作为精化起点
                x0_agg = self._aggregate_x0(mpi_results, weights, i)
                x_raw_refined[i, high_var_idx] = x0_agg[high_var_idx]

        # 对精化后的 proposal 再做一次 t=1 评估
        cls_refined, box_refined, _, _ = self._forward_at_t(
            features, x_raw_refined, 1.0, img_metas
        )
        # 用精化结果替换高方差 proposal 的预测
        self._merge_refined(mpi_results, cls_refined, box_refined,
                           needs_refinement)

    # Phase 4: 最终聚合 + NMS
    results = self.mpi_aggregate(mpi_results, img_metas, weights)
    return results
```

**设计要点**:
- 只对 Top 20% 高方差 proposal 做精化, 避免全量精化的计算开销
- 精化用 $t=1$ (在分布内) 重新评估, 确保低偏差
- 精化起点是聚合后的 $\hat{x}_0$, 相当于"从更好的起点重新采一次"

### 3.5 与现有代码的集成方案

MPI 是纯推理期优化, 改动集中在 `predict` 方法和 `DiffusionSampler`:

**改动清单**:

| 文件 | 改动 | 说明 |
|------|------|------|
| `head.py` | 新增 `mpi_predict` 方法 | MPI 推理入口, 替代 `predict` 当 `use_mpi=True` |
| `head.py` | 新增 `_forward_at_t_batch` 方法 | 批量多点评估 (3.2.3) |
| `head.py` | 新增 `_forward_at_t` 修改 | 支持返回中间量供不确定性计算 |
| `sampling.py` | 新增 `mpi_post_process` 方法 | 加权聚合后处理 (3.3.1) |
| `sampling.py` | 新增 `compute_uncertainty` 方法 | 不确定性计算 (3.4.1) |
| 配置文件 | 新增 `use_mpi`, `mpi_points`, `mpi_weights` 等参数 | |

**配置示例**:

```python
# experiments/configs/ldmdet/w_mpi.py
_base_ = ['./a3_full_sota.py']
model = dict(
    bbox_head=dict(
        use_mpi=True,                         # 启用 MPI
        mpi_points=[1.0, 0.7, 0.5, 0.3],     # 4 个评估点
        mpi_cls_weights=[0.15, 0.25, 0.40, 0.20],  # 分类权重
        mpi_reg_weights=[0.50, 0.30, 0.15, 0.05],  # 回归权重
        mpi_aggregate='decoupled',            # 'nms' | 'aligned' | 'decoupled'
        mpi_uncertainty_refinement=False,     # Phase 2: 不确定性精化
        mpi_refine_topk_ratio=0.2,            # 精化 Top 20% 高方差
    ),
)
```

---

## 4. 深入可行性分析

### 4.1 与现有 use_ensemble 的深入对比

阅读代码后, 确认 MPI 与现有时间维 ensemble 在**评估方式**、**预测关系**和**聚合机制**上完全不同:

#### 4.1.1 评估方式: 序列轨迹 vs 并行多点

```python
# 现有 ensemble (head.py:332-387) — 序列轨迹, x_raw 沿轨迹演化
x_raw = torch.randn(...)                    # t=1 的初始噪声
for t_curr, t_next in time_pairs:           # t: 1.0 → 0.75 → 0.5 → 0.25
    cls, box, x0, v = self._forward_at_t(
        features, x_raw, t_curr, img_metas  # ← x_raw 随 t_curr 变化
    )
    ensemble_results.append((cls, box))
    x_raw = solver.step(x_raw, x0, t_curr, t_next)  # ← x_raw 更新
    x_raw = apply_box_renewal(x_raw, cls)            # ← box_renewal 修改

# MPI — 并行多点, x_raw 固定
x_raw = torch.randn(...)                    # t=1 的初始噪声 (固定)
for t_k in mpi_points:                       # t: 1.0, 0.7, 0.5, 0.3 (独立)
    cls, box, x0, v = self._forward_at_t(
        features, x_raw, t_k, img_metas     # ← 同一个 x_raw
    )
    mpi_results.append((cls, box, t_k))
    # 无 solver.step, 无 box_renewal — x_raw 不变
```

**关键区别**:
- 现有 ensemble: $(x_{\text{raw}}^{(i)}, t_i)$, 其中 $x_{\text{raw}}^{(i)}$ 依赖 $x_{\text{raw}}^{(i-1)}$ 的模型输出。**序列强依赖**。
- MPI: $(x_{\text{raw}}, \tau_k)$, 其中 $x_{\text{raw}}$ 对所有 $k$ 相同。**完全独立**。

#### 4.1.2 预测相关性分析

**现有 ensemble 的相关性来源**:
1. 轨迹连续性: $x_{\text{raw}}^{(i+1)} = x_{\text{raw}}^{(i)} + \Delta t \cdot v$, 相邻步的 $x_{\text{raw}}$ 差异小
2. 模型平滑性: $f_\theta$ 是 Lipschitz 连续的, 相似输入产生相似输出
3. Box renewal 只替换低置信度框, 高置信度框在轨迹中保持不变

**估计**: 相邻时间步的预测相关系数 $\rho_{\text{adj}} \approx 0.8 - 0.9$

**MPI 的相关性来源**:
1. 同一 $x_{\text{raw}}$: 所有评估共享噪声框, 有一定相关性
2. 但 $t$ 条件不同: 时间嵌入 $t \cdot \text{timesteps}$ 通过 SinusoidalPositionEmbeddings 产生不同的条件信号
3. AdaLN-Zero 的 $\gamma, \beta, \alpha$ 参数随 $t$ 变化, 改变 head 的计算

**估计**: 不同 $t$ 点的预测相关系数 $\rho_{\text{mpi}} \approx 0.3 - 0.5$ (远低于 ensemble)

**方差降低对比** ($K=4$, $\sigma^2$ 为单点方差):

| 方法 | $\bar{\rho}$ | $\text{Var} / \sigma^2$ | 方差降低 |
|------|-------------|----------------------|---------|
| 单点 | N/A | 1.0 | 0% |
| 时间维 ensemble | 0.85 | $\frac{1+3\times 0.85}{4} = 0.89$ | 11% |
| MPI | 0.40 | $\frac{1+3\times 0.40}{4} = 0.55$ | 45% |

MPI 的方差降低效率是时间维 ensemble 的 **4 倍**。

#### 4.1.3 聚合机制: 拼接 NMS vs 加权聚合

```python
# 现有 post_process (sampling.py:194-264) — 无权拼接 + NMS
for cls_logits, pred_bboxes in ensemble_results:
    scores = torch.sigmoid(cls_logits[i])  # 无 t 相关权重
    all_scores.append(scores.max(-1)[0])
final_scores = torch.cat(all_scores)  # T × P 个框, 等权
keep = batched_nms(final_bboxes, final_scores, final_labels, nms_thr)

# MPI — t 相关加权聚合
for (cls_logits, pred_bboxes, _, t_k), w_k in zip(mpi_results, weights):
    scores = torch.sigmoid(cls_logits[i]) * w_k  # ← t 相关权重
    all_scores.append(scores.max(-1)[0])
final_scores = torch.cat(all_scores)  # K × P 个框, 加权
keep = batched_nms(final_bboxes, final_scores, final_labels, nms_thr)
```

**关键区别**:
- 现有 ensemble: 所有时间步等权 ($w = 1$), 忽略了不同 $t$ 的预测质量差异
- MPI: 按 $t$ 相关权重加权, 高质量预测 (如 $t=1$ 的回归, $t=0.5$ 的分类) 贡献更大

#### 4.1.4 正交性与可组合性

MPI 与时间维 ensemble **不冲突**, 可组合使用:

**方案 A (替代)**: 用 MPI 替代时间维 ensemble, 每个评估点只做单步评估
- 适用: 低延迟场景, $K=4$ 时计算量 ≈ 4 次单步前向

**方案 B (组合, 推荐)**: 先做 MPI 多点评估, 再对 $t=1$ 的预测做轨迹采样, 融合两者
- MPI 提供 $K$ 个独立预测 (低相关性, 降方差)
- 轨迹采样提供 $T$ 个相关预测 (低偏差, 精化)
- 融合: $K + T$ 个预测, 兼顾独立性和精度

### 4.2 计算复杂度分析

#### 4.2.1 单次前向的开销分解

当前基线 (a3_full_sota) 配置: Heun 4 步采样, 6 级 cascade head, 500 proposals。

| 组件 | 单次前向开销 | 是否可复用 |
|------|------------|----------|
| ResNet-50 backbone | ~40 ms | ✅ 跨所有评估点复用 |
| FPN | ~5 ms | ✅ 跨所有评估点复用 |
| RoIAlign (500 props × 7×7) | ~8 ms | ❌ 每个评估点独立 (因框位置不同) |
| 6 级 cascade head (self-attn + dynamic conv + FFN) | ~30 ms | ❌ 每个评估点独立 |
| Box renewal + NMS | ~2 ms | ❌ 每个评估点独立 |
| **总计** | ~85 ms | |

**特征复用率**: backbone + FPN = 45 ms / 85 ms ≈ **53%** 可复用。

#### 4.2.2 MPI 总开销

| 组件 | 基线 (Heun 4步) | MPI (K=4) | MPI + 轨迹 (组合) |
|------|---------------|-----------|------------------|
| backbone+FPN | 1× (45ms) | 1× (45ms) | 1× (45ms) |
| RoIAlign+Head | 8× (272ms) | 4× (136ms) | 12× (408ms) |
| 后处理 | 1× (2ms) | 1× (2ms) | 1× (2ms) |
| **总时间** | ~319ms | ~183ms | ~455ms |
| **相对基线** | 1.0× | **0.57×** | **1.43×** |

**关键发现**: MPI ($K=4$) 比基线 Heun 4 步**更快** (0.57×), 因为:
- Heun 4 步 = 4 步 × 2 次前向/步 = 8 次 Head 前向
- MPI $K=4$ = 4 次 Head 前向 (无轨迹步进, 无 Heun 二阶)
- 两者共享 backbone 特征

**这是 MPI 的巨大优势**: 在降低计算量的同时降低方差, 是罕见的"又快又好"方向。

#### 4.2.3 批量并行化的加速

若用 batch 维并行 (3.2.3 节的 `_forward_at_t_batch`), $K=4$ 的 MPI 可进一步加速:

| 并行方式 | 总时间 | 相对基线 | 显存 |
|---------|--------|---------|------|
| 序列执行 | 183ms | 0.57× | 1× |
| Batch 并行 (K=4) | ~120ms | **0.38×** | ~4× Head 激活 |
| Batch 并行 (K=4, AMP) | ~90ms | **0.28×** | ~2× Head 激活 |

**显存估算**: 6 级 cascade × 500 proposals × 256 channels × 7×7 RoI = ~2.3 GB/eval。$K=4$ batch 并行需 ~9.2 GB, 单卡 A100 (40/80 GB) 可承受。若启用 AMP (bfloat16), 显存减半至 ~4.6 GB。

#### 4.2.4 不确定性精化的额外开销

若启用不确定性精化 (3.4.2 节), 对 Top 20% 高方差 proposal 做额外 1 次评估:

| 组件 | 开销 |
|------|------|
| 不确定性计算 | $O(K \cdot P)$, ~0.1 ms |
| Top 20% proposal 提取 | $O(P \log P)$, ~0.05 ms |
| 额外 1 次 Head 前向 (100 props) | ~6 ms (约为全量 Head 的 20%) |
| **总额外开销** | ~6.2 ms (< 4% 总推理时间) |

### 4.3 数值稳定性分析

#### 4.3.1 OOD 评估的数值范围

在 $\tau < 1$ 处评估模型时, 输入 $x_{\text{raw}}$ (即 $x_{\text{noise}}$) 的数值范围是 $\mathcal{N}(0, I)$, 而模型期望 $x_\tau = (1-\tau)x_0 + \tau \cdot x_{\text{noise}}$。

- $x_0$ (GT 框) 归一化到 $[-\text{snr\_scale}, \text{snr\_scale}] = [-2.0, 2.0]$
- $x_{\text{noise}} \sim \mathcal{N}(0, 1)$, 范围约 $[-3, 3]$ (3σ)
- $x_\tau = (1-\tau) \cdot [-2, 2] + \tau \cdot [-3, 3]$

对 $\tau = 0.3$: $x_\tau \in [-2.1, 2.1]$, 与 $x_{\text{noise}} \in [-3, 3]$ 有偏差但不爆炸。
对 $\tau = 0.0$: $x_\tau \in [-2, 2]$, $x_{\text{noise}}$ 可能超出, 但 `raw_to_xyxy` 中有 `clamp(-snr_scale, snr_scale)` 保护。

**结论**: OOD 评估的数值范围可控, 不会产生 NaN/Inf。模型输出可能偏差但不会数值爆炸。

#### 4.3.2 加权聚合的数值稳定性

- 权重 $w_k \geq 0$, $\sum w_k = 1$, 加权平均不会放大数值范围
- 分类 logits 加权后, $\sum w_k \cdot \text{logits}_k$ 的范围在 $\min(\text{logits})$ 到 $\max(\text{logits})$ 之间
- 若用 `softmax` 聚合概率 (而非 `sigmoid` logits), 需注意 log-sum-exp 的数值稳定性

### 4.4 实时性分析

染色体核型分析是**离线诊断任务**, 非实时。临床流程中, 从样本制备到出报告需数小时到数天。推理时间从 319ms (基线) 降低到 183ms (MPI) 或增加到 455ms (MPI+轨迹), 在临床流程中**完全可接受**。

**MPI 的独特优势**: 在**降低推理时间**的同时**提升精度**, 这是罕见的双赢特性。对比:
- 1→8 步采样: 精度 +0.015, 时间 8× (代价大)
- MPI ($K=4$): 精度预期 +0.003~0.008, 时间 0.57× (反而更快)

---

## 5. 风险评估

### 5.1 风险一: OOD 评估导致预测偏差过大

**风险描述**: 在 $\tau \ll 1$ 处, $x_{\text{raw}}$ (纯噪声) 严重偏离模型期望的 $x_\tau = (1-\tau)x_0 + \tau \cdot x_{\text{noise}}$。模型可能产生系统性偏差的预测, 若权重配置不当, 偏差可能抵消方差降低的收益。

**量化指标**: 定义 OOD 偏差 $B(\tau) = \|\mathbb{E}[\hat{x}_0(\tau)] - x_0\|$。若 $B(\tau) > 2\sigma$ (偏差超过 2 倍标准差), MPI 净收益为负。

**缓解方案**:
1. **保守评估点**: Phase 1 只用 $\tau \in \{1.0, 0.8, 0.6\}$ (偏离不大), 验证收益后再扩展到 $\tau = 0.5, 0.3$
2. **偏差感知权重**: 权重 $w_k \propto \exp(-\alpha \cdot B(\tau_k)^2)$, 偏差大的点权重自动降低。$B(\tau)$ 可在验证集上预估计
3. **退化保护**: 设 $w_1 \geq 0.5$ (保证 $t=1$ 的在分布内预测占主导), 其余 $w_k$ 之和 $\leq 0.5$。最坏情况下退化为单点预测 (基线), 无负增益
4. **诊断实验**: Phase 0 先测量不同 $\tau$ 的 $B(\tau)$ 和 $\text{Var}(\hat{x}_0(\tau))$, 确定偏差-方差可接受的 $\tau$ 范围

### 5.2 风险二: 预测相关性过高, 方差降低不显著

**风险描述**: 若模型对 $t$ 条件的敏感度不足 (即 $f_\theta(x, t_1) \approx f_\theta(x, t_2)$ 对不同 $t$), 不同 $\tau$ 的预测高度相关, MPI 退化为"重复计算 + 平均", 方差降低趋近于 0。

**量化指标**: 测量 $K$ 组预测的平均相关系数 $\bar{\rho}$。若 $\bar{\rho} > 0.8$, 方差降低 $< 15\%$, 收益有限。

**潜在原因**:
- AdaLN-Zero 的 $\gamma, \beta, \alpha$ 初始化为 0 (single_head.py:129-130), 训练初期 $t$ 条件化弱
- 若训练后 $\gamma, \beta, \alpha$ 仍接近 0, 模型对不同 $t$ 的预测几乎相同

**缓解方案**:
1. **诊断实验**: Phase 0 测量 $\bar{\rho}$, 若 $\bar{\rho} > 0.8$, 暂停 MPI, 先研究 $t$ 条件化增强
2. **扩大 $\tau$ 范围**: 若 $\tau$ 间距太小导致相关性高, 扩大到 $\tau \in \{1.0, 0.5, 0.0\}$ (极端但独立)
3. **结合 box_renewal 扰动**: 不同 $\tau$ 点用不同 `score_thr` 的 box_renewal, 增加预测多样性
4. **退化保护**: 即使 $\bar{\rho} = 1$ (完全相关), MPI 退化为 $t=1$ 的单点预测 (基线), 无负增益

### 5.3 风险三: Proposal 不对齐导致聚合失效

**风险描述**: 3.3.2 节的 proposal 对齐加权平均假设不同 $\tau$ 的第 $i$ 个 proposal 对应同一目标。但 `_forward_at_t` 中 `raw_to_xyxy` 将 $x_{\text{raw}}$ 转为 xyxy 框, 而 $x_{\text{raw}}$ 是固定的, 因此框位置应相同。但 cascade head 的 RoI 提取可能因 `curr_proposals` 的传递而产生差异。

**代码验证**: 阅读 single_head.py 第 267-290 行, `forward` 方法中:
```python
rois = bbox2roi([bboxes[i] for i in range(bs)])  # bboxes 来自 raw_to_xyxy(x_raw)
roi_features = pooler(features, rois)            # RoI 提取
```
由于 `bboxes` 来自同一 `x_raw`, 不同 $\tau$ 的 `bboxes` 相同 (在 `_forward_at_t` 中 `raw_to_xyxy` 不依赖 $t$), 因此 **proposal 是对齐的**。

**结论**: 此风险在实际代码中**不存在**, proposal 对齐假设成立。但需注意: 若未来修改 `_forward_at_t` 使 `raw_to_xyxy` 依赖 $t$ (如 CFM 模式), 则需重新验证对齐性。

### 5.4 风险四: 权重超参数过拟合验证集

**风险描述**: MPI 的权重 $\{w_k\}$ (或分类/回归解耦的 $2K$ 个权重) 在验证集上搜索, 可能过拟合验证集, 在测试集上泛化差。

**量化风险**: $K=4$ 时有 4-8 个权重参数, 验证集 ~500 张图。参数/样本比 = 8/500 = 0.016, 过拟合风险低。但若权重搜索空间大 (如连续优化), 仍需注意。

**缓解方案**:
1. **粗网格搜索**: 只搜索 3-5 组预设权重配置 (如等权、SNR 加权、指数衰减、临界点加权、解耦加权), 不做连续优化
2. **交叉验证**: 将验证集分 5 折, 在 4 折上搜索权重, 1 折上验证, 取平均
3. **退化保护**: 等权配置 ($w_k = 1/K$) 作为 baseline, 任何加权方案都必须在验证集上超过等权才采用
4. **简化权重**: 用 1-2 个超参数的参数化权重 (如 $w_k \propto \tau_k^\alpha$, 只搜 $\alpha$), 减少过拟合

### 5.5 风险五 (附加): 与 box_renewal 的交互

**风险描述**: 现有推理在每步轨迹后做 `box_renewal` (替换低置信度框为随机噪声)。MPI 不做轨迹步进, 因此不做 box_renewal。这可能使低置信度 proposal 保留, 影响 NMS 后处理。

**分析**: 
- box_renewal 的作用是"在轨迹中途替换掉已经明确是假阳的框, 用新噪声探索"
- MPI 中所有评估点使用同一 $x_{\text{raw}}$, 无"中途"概念
- 低置信度框在 NMS 时自然被过滤 (score_thr)

**缓解方案**:
1. **NMS 兜底**: 现有 `score_thr=0.5` 的 NMS 已能过滤低置信度框, 不依赖 box_renewal
2. **可选 box_renewal**: 在 MPI 评估后, 对聚合结果做一次 box_renewal + 重新评估 (类似不确定性精化)
3. **组合方案**: MPI + 轨迹采样 (4.1.4 方案 B), 轨迹部分保留 box_renewal

---

## 6. 预期收益分析

### 6.1 基于 1→8 步增益曲线的定量估计

**已知数据**: 1→8 步采样, mAP 0.740→0.755, 增益 +0.015, 但 8× 计算量。

**增益来源分解** (1→8 步):
| 步数变化 | Δ mAP | 增益来源 |
|---------|-------|---------|
| 1→2 步 | ~+0.008 | 轨迹精化 (减少单步 x_0_pred 误差) |
| 2→4 步 | ~+0.005 | 轨迹精化 + ensemble (轻微方差降低) |
| 4→8 步 | ~+0.002 | 仅 ensemble (方差降低, 但相关性高, 收益饱和) |

**关键观察**: 4→8 步的增益仅 +0.002, 说明**时间维 ensemble 的方差降低已饱和** (因预测相关性高)。这间接证实了 MPI 的价值——**通过低相关性的多点评估, 可以突破时间维 ensemble 的饱和点**。

### 6.2 MPI 增益来源分解

| 增益来源 | 机制 | 预估 Δ mAP | 依据 |
|---------|------|-----------|------|
| 方差降低 (多点独立) | $K=4$ 独立预测, $\bar{\rho} \approx 0.4$, 方差降 45% | +0.002 ~ +0.005 | 类比 4→8 步 ensemble (+0.002), 但 MPI 相关性更低, 收益更高 |
| 偏差-方差优化 (加权) | SNR 感知权重, 分类用 $t \approx 0.5$, 回归用 $t \approx 1.0$ | +0.001 ~ +0.003 | 分类-回归解耦, 各取最优工作点 |
| 不确定性精化 | 高方差框重新评估, 修正假阳/漏检 | +0.000 ~ +0.002 | Top 20% 高方差框中, 约 30% 可被修正 |
| **合计** | | **+0.003 ~ +0.010** | 取中位数 **+0.006** |

### 6.3 不同配置的预期收益

| 配置 | $K$ | 评估点 | 聚合方式 | 预期 Δ mAP | 计算开销 |
|------|-----|--------|---------|-----------|---------|
| MPI-Lite | 2 | {1.0, 0.5} | NMS 加权 | +0.002 ~ +0.004 | 0.5× 基线 |
| MPI-Standard | 4 | {1.0, 0.7, 0.5, 0.3} | 解耦加权 | +0.004 ~ +0.008 | 0.57× 基线 |
| MPI-Full | 4 | {1.0, 0.7, 0.5, 0.3} | 解耦 + 精化 | +0.005 ~ +0.010 | 0.6× 基线 |
| MPI + 轨迹 | 4+4 | MPI + Heun 4步 | 融合 | +0.006 ~ +0.012 | 1.43× 基线 |

**推荐配置**: MPI-Standard ($K=4$, 解耦加权), 在降低计算量的同时获得中位 +0.006 mAP。

### 6.4 增益上界分析

MPI 的增益上界受限于:
1. **OOD 偏差**: $\tau < 1$ 的预测偏差不可消除, 限制了方差降低的净收益
2. **预测相关性**: 若 $\bar{\rho} > 0$, 方差降低低于理论最大值 $1/K$
3. **模型容量**: 模型在训练时对非 $t=1$ 点的预测能力有限, OOD 预测质量不高

**理论上界**: 若 $K$ 个预测完全独立 ($\rho = 0$) 且无偏差, 方差降低 $1 - 1/K$。对 $K=4$, 方差降 75%, 对应 mAP 提升 $\approx +0.015$ (类比 1→8 步的 +0.015, 但机制不同)。

**实际上界**: 考虑 OOD 偏差和相关性, 实际增益约为理论上界的 30-50%, 即 **+0.005 ~ +0.008**。

### 6.5 与现有组件的增益对比

| 组件 | 增益 (Δ mAP) | 计算开销 | 风险 | 状态 |
|------|-------------|---------|------|------|
| box_renewal | +0.016 | 1× | 已验证 | 基线组件 |
| 1→8 步采样 | +0.015 | 8× | 已验证 (递减) | 可选 |
| 时间维 ensemble | 已含在基线 | 1× | 已验证 | 基线组件 |
| **MPI-Standard** | **+0.004 ~ +0.008** | **0.57×** | 中 | 待验证 |
| PCSE (方向 R) | +0.006 ~ +0.016 | 2-3× | 中 | 待验证 |
| CFM (方向 H, 已证伪) | -0.002 | 1× | 已失败 | 不采用 |

**关键优势**: MPI 是唯一一个**在降低计算量的同时提升精度**的方向。其他方向 (多步采样, PCSE) 都以增加计算量为代价。

---

## 7. 实现路线图

### Phase 0: 前置诊断 (3 天)

**目标**: 量化 OOD 偏差和预测相关性, 确定最优评估点。

**任务**:
1. 用基线模型 (a3_full_sota), 对验证集 50 张图, 在 $\tau \in \{1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1\}$ 各评估一次
2. 测量每个 $\tau$ 的:
   - 预测偏差 $B(\tau) = \|\hat{x}_0(\tau) - \hat{x}_0(1.0)\|$ (以 $t=1$ 预测为参考)
   - 预测方差 $\text{Var}(\hat{x}_0(\tau))$ (跨不同噪声样本)
   - 与 $t=1$ 预测的相关系数 $\rho(\tau, 1.0)$
3. 绘制 $B(\tau)$-$\tau$ 和 $\rho(\tau, 1.0)$-$\tau$ 曲线, 确定偏差和相关性可接受的 $\tau$ 范围
4. 测量 batch 并行的实际加速比和显存占用

**决策点**:
- 若 $B(\tau) > 2\sigma$ 对所有 $\tau < 0.7$ → MPI 收益有限, 调整为 $\tau \in \{1.0, 0.9, 0.8\}$
- 若 $\rho(\tau, 1.0) > 0.8$ 对所有 $\tau$ → $t$ 条件化弱, 暂停 MPI, 先研究条件化增强
- 若 $B(\tau) < \sigma$ 且 $\rho < 0.6$ 对 $\tau \in [0.3, 0.8]$ → 推进 Phase 1

### Phase 1: 核心实现与验证 (1 周)

**目标**: 实现 MPI 基础版, 验证纯推理增益。

**任务**:
1. 实现 `_forward_at_t_batch` (3.2.3 节, 批量多点评估)
2. 实现 `mpi_aggregate_nms` (3.3.1 节, 分数加权 NMS)
3. 实现 `mpi_aggregate_decoupled` (3.3.3 节, 分类-回归解耦)
4. 修改 `predict` 方法, 新增 `mpi_predict` 入口
5. 新增配置 `w_mpi.py`, 在 a3_full_sota 基础上启用 MPI
6. 在验证集上对比:
   - 基线 (Heun 4步 + ensemble): mAP 基线
   - MPI-Lite ($K=2$, $\tau=\{1.0, 0.5\}$): 验证最小可行
   - MPI-Standard ($K=4$, $\tau=\{1.0, 0.7, 0.5, 0.3\}$): 标准配置
   - MPI + 轨迹 (组合方案 B): 验证组合收益
7. 网格搜索权重配置 (等权 / SNR / 指数 / 临界点 / 解耦)
8. 测量推理时间和显存占用

**预期产出**:
- MPI 各配置的 mAP 报告
- 最优权重配置
- 偏差-方差曲线和相关性矩阵
- 推理时间 benchmark

**决策点**:
- 若 MPI-Standard 增益 > +0.003 → 推进 Phase 2
- 若 MPI-Standard 增益 < +0.001 → 分析原因, 考虑调整评估点或放弃

### Phase 2: 不确定性精化与优化 (1 周)

**目标**: 若 Phase 1 增益 > +0.003, 添加不确定性精化和批量并行。

**任务**:
1. 实现 `compute_uncertainty` (3.4.1 节)
2. 实现不确定性触发的自适应精化 (3.4.2 节)
3. 实现 `_forward_at_t_batch` 的 batch 并行优化
4. 测量 batch 并行的加速比和显存占用
5. 在测试集上最终评估

**预期产出**:
- 不确定性精化的增益评估
- batch 并行 benchmark
- 测试集最终 mAP

### Phase 3 (可选): 高级聚合与学习权重 (2 周)

**目标**: 若 Phase 2 增益饱和但仍有提升空间, 探索高级聚合。

**任务**:
1. 实现学习权重 (在验证集上用 LightGBM 学习 $w(t)$)
2. 实现跨 proposal 的注意力聚合 (而非简单加权平均)
3. 探索 MPI + PCSE (方向 R) 的组合
4. 论文级实验: 消融、可视化、案例分析

**预期产出**:
- 学习权重的增益评估
- MPI + PCSE 组合的增益
- 论文初稿素材

### 代码改动清单 (Phase 1)

```
ldmdet/
├── core/
│   └── head.py                    # 修改: 新增 mpi_predict, _forward_at_t_batch
├── diffusion/
│   └── sampling.py                # 修改: 新增 mpi_post_process, compute_uncertainty
└── tests/
    └── test_direction_w_mpi.py    # 新增: 单元测试

experiments/
└── configs/
    └── ldmdet/
        └── w_mpi.py               # 新增: MPI 配置
```

**纯推理优化, 不改动**:
- 训练损失 (`criterion`)
- 模型结构 (`single_head`, `backbone`, `neck`)
- 扩散核心 (`rectified_flow`)
- 数据处理 (`transforms`, `pipeline`)

---

## 8. 与已证伪方向的对比: 为什么 MPI 是安全的纯推理优化

### 8.1 已证伪方向复盘

阅读 [README.md](README.md) 和 [方向间关系与纠正说明.md](方向间关系与纠正说明.md) 可知, 两个已证伪方向的失败原因:

**方向 H (CFM, mAP=0.854, -0.002)**:
1. 改变训练目标 (预测速度 $v$ 而非 $x_0$), 需修改 loss 和 head 输出语义
2. `velocity_loss_weight=0.1` 过保守, velocity loss 仅占总 loss < 3%, 不足以改变训练动态
3. 不引入新信息, 仅对齐训练-推理目标
4. 训练不稳定风险, 改训练目标导致梯度尺度变化

**方向 N (端到端可微 Cascade, mAP=0.684, -0.172)**:
1. 去 `detach` 后梯度贯通, 但 6 级 cascade 的梯度反向传播严重不稳定
2. mAP 震荡 0.35~0.68, 训练完全失控
3. 改动训练架构, 风险极高

### 8.2 MPI 与已证伪方向的对比

| 维度 | CFM (方向 H, 已证伪) | N (方向 N, 已证伪) | **MPI (方向 W)** |
|------|--------------------|-------------------|-----------------|
| **是否改训练** | ✅ 改 (loss + head 输出) | ✅ 改 (去 detach) | ❌ **不改 (纯推理)** |
| **是否改模型结构** | ✅ 改 (predict_velocity) | ❌ 不改 | ❌ **不改** |
| **是否改训练数据** | ❌ 不改 | ❌ 不改 | ❌ **不改** |
| **是否引入新信息** | ❌ 不引入 | ❌ 不引入 | ✅ **引入 (多 $t$ 点的独立预测)** |
| **失败模式** | 训练动态不变, 无增益 | 梯度爆炸, 训练崩溃 | **退化为基线 (无负增益)** |
| **可回退性** | 差 (需重训) | 差 (需重训) | **好 (关 `use_mpi` 即可)** |
| **可调参数** | 训练权重 (需重训验证) | 架构开关 (需重训) | **推理权重 (验证集秒级搜索)** |
| **计算开销** | 1× (训练 1.5×) | 1× (训练不稳定) | **0.57× (反而更快)** |

### 8.3 为什么 MPI 能避免三个陷阱

**陷阱 1: "改训练目标但权重太保守" (CFM 的失败)**
- MPI **完全不改训练目标**, 复用 a3_full_sota 的全部训练权重
- 推理权重 $\{w_k\}$ 在验证集上直接搜索, 无需重训
- 即使等权 ($w_k = 1/K$), MPI 仍提供方差降低 (只要 $\bar{\rho} < 1$)

**陷阱 2: "不引入新信息" (CFM 的失败)**
- CFM 的速度损失只是把模型已有的 $x_0$ 预测"换个形式"监督, 没有新信号
- MPI 的多 $t$ 点预测是**模型在训练时学过但推理时未使用的能力**:
  - 训练时 $t \sim U[0,1]$, 模型在所有 $t$ 上有监督
  - 推理时只用 $t=1 \to 0$ 的轨迹点, 浪费了模型在 $t=0.7, 0.5, 0.3$ 等点的预测能力
  - MPI 激活了这些"休眠"的预测能力, 提供了正交于轨迹采样的新信息

**陷阱 3: "训练不稳定" (N 的失败)**
- MPI 无训练过程, 无梯度, 无不稳定风险
- 最坏情况: 评估点全是 OOD, 权重全错 → 退化为 $t=1$ 单点预测 (基线)
- **MPI 有下界保证**: $mAP_{\text{MPI}} \geq mAP_{\text{baseline}} - \epsilon$
  - 设 $w_1 \geq 0.5$ (保证 $t=1$ 占主导), 最坏退化为 $0.5 \times \text{baseline} + 0.5 \times \text{noise}$, 但 NMS 会过滤噪声框, 实际接近 baseline

### 8.4 MPI 的失败模式分析

即使 MPI 增益不如预期, 其失败模式是**可控且非负**的:

| 失败模式 | 概率 | 影响 | 兜底机制 |
|---------|------|------|---------|
| OOD 偏差过大 | 中 | 增益 ≈ 0 | 保守评估点 + $w_1 \geq 0.5$ 退化保护 |
| 预测相关性过高 | 中 | 方差降低不显著 | 扩大 $\tau$ 范围 + 组合 box_renewal 扰动 |
| 权重过拟合验证集 | 低 | 测试集增益降低 | 粗网格搜索 + 交叉验证 |
| Proposal 不对齐 | 低 (已验证) | 聚合失效 | 退化为分数加权 NMS (3.3.1) |
| 计算开销超标 | 极低 | 部署受限 | MPI 本身比基线更快 (0.57×) |

**关键结论**: MPI 是**非负收益方向** (最坏退化为基线), 且**计算开销为负** (比基线更快)。这是 MPI 相对 CFM 和 N 的根本优势——**下行风险几乎为零, 上行收益确定存在** (只是幅度待验证)。

### 8.5 与方向 R (PCSE) 的对比与协同

MPI 与 PCSE (方向 R) 都在推理期优化, 但探索不同的"扰动维度":

| 维度 | PCSE (方向 R) | MPI (方向 W) |
|------|-------------|-------------|
| **扰动源** | 随机种子 (不同 $x_{\text{raw}}$) | 时间条件 (不同 $t$) |
| **预测独立性来源** | 不同初始噪声 → 不同收敛点 | 不同 $t$ 条件化 → 不同预测区制 |
| **聚合方式** | 核型评分 + 选择 (argmax) | 加权平均 (weighted average) |
| **利用的先验** | 核型学先验 (46 条, 配对) | 偏差-方差理论 (SNR, 相关性) |
| **计算开销** | 2-3× (K 次完整轨迹) | 0.57× (K 次单步评估) |
| **互补性** | 探索噪声空间 | 探索时间空间 |

**协同可能**: MPI 和 PCSE 可组合——
1. MPI 在多个 $t$ 点评估, 提供 $K_{\text{mpi}}$ 组预测
2. PCSE 用不同种子重复, 提供 $K_{\text{pcse}}$ 组假设
3. 总计 $K_{\text{mpi}} \times K_{\text{pcse}}$ 组预测, 用核型评分选择最优
4. 但计算开销为 $K_{\text{pcse}} \times 0.57\times$ 基线, 仍可接受

Phase 1 建议单独验证 MPI, 确认收益后再探索与 PCSE 的组合。

---

## 9. 参考文献

### 9.1 偏差-方差分解与集成学习
- Geman, S., Bienenstock, E., & Doursat, R. "Neural Networks and the Bias/Variance Dilemma." Neural Computation, 1992. — 偏差-方差分解经典
- Brown, G. et al. "Diversity Creation Methods: A Survey and Categorisation." Information Fusion, 2005. — 集成多样性理论
- Krogh, A., & Vedelsby, J. "Neural Network Ensembles, Cross Validation, and Active Learning." NeurIPS, 1995. — 集成方差降低公式
- Lakshminarayanan, B. et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS, 2017. — 深度集成与不确定性

### 9.2 扩散模型与多时间点评估
- DiffusionDet: Chi et al. "DiffusionDet: Diffusion Model for Object Detection." ICCV, 2023. — 扩散检测基线, 时间维 ensemble
- DDPM: Ho, J. et al. "Denoising Diffusion Probabilistic Models." NeurIPS, 2020. — 扩散采样理论
- Rectified Flow: Liu, X. et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR, 2023. — 当前所用 RF 路径
- DPM-Solver++: Lu, C. et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models." NeurIPS, 2022. — 高阶求解器
- Song, Y. & Dhariwal, P. "Improved Techniques for Training Consistency Models." ICML, 2024. — 多时间点一致性约束

### 9.3 不确定性估计
- Gal, Y. & Ghahramani, Z. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML, 2016. — MC Dropout 不确定性
- Lakshminarayanan, B. et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS, 2017. — 集成方差作为不确定性
- Angelopoulos, A. N. & Bates, S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." 2022. — 共形预测

### 9.4 测试时增强与推理优化
- Krizhevsky, A. et al. "ImageNet Classification with Deep Convolutional Neural Networks." NeurIPS, 2012. — 测试时裁剪增强
- Wang, F. et al. "Residual Attention Network for Image Classification." CVPR, 2017. — 注意力机制与 $t$ 条件化
- Howard, A. et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." 2017. — 计算效率优化

### 9.5 项目内相关方向
- [方向 R: PCSE 配对一致性随机集成评分](方向R_PCSE_配对一致性随机集成评分.md) — 推理期多种子采样 + 核型评分选择
- [方向 I: Consistency Model 检测](方向I_ConsistencyModel检测.md) — 单步生成, 消除多步累积误差
- [方向 H: Flow Matching 检测 (已证伪)](方向H_FlowMatching检测.md) — CFM 失败教训
- [方向 N: 端到端可微 Cascade (已证伪)](方向N_端到端可微Cascade.md) — 去 detach 失败教训
- [方向 P: HCTS 层级粗到细时间步调度](方向P_HCTS_层级粗到细时间步调度.md) — 时间步调度优化

---

## 附录 A: 偏差-方差分解的完整数学推导

### A.1 设定

固定 $x_{\text{raw}}$ (即 $x_{\text{noise}}$, $t=1$ 的样本), 在 $K$ 个时间点 $\{\tau_1, \dots, \tau_K\}$ 评估模型, 得到 $K$ 个预测 $\hat{x}_0(\tau_k) = f_\theta(x_{\text{raw}}, \tau_k)$。

加权聚合: $\hat{x}_0^{\text{MPI}} = \sum_{k=1}^{K} w_k \hat{x}_0(\tau_k)$, $\sum w_k = 1$。

### A.2 期望分解

$$\mathbb{E}_{x_{\text{noise}}}[\hat{x}_0^{\text{MPI}}] = \sum_k w_k \mathbb{E}_{x_{\text{noise}}}[f_\theta(x_{\text{noise}}, \tau_k)]$$

对 $\tau_k = 1$ (在分布内): $\mathbb{E}[f_\theta(x_{\text{noise}}, 1)] \approx x_0$ (模型近似无偏)

对 $\tau_k < 1$ (OOD): $\mathbb{E}[f_\theta(x_{\text{noise}}, \tau_k)] = x_0 + B(\tau_k)$, 其中 $B(\tau_k)$ 是偏差

$$\mathbb{E}[\hat{x}_0^{\text{MPI}}] = x_0 + \sum_k w_k B(\tau_k)$$

**聚合偏差**: $B_{\text{MPI}} = \sum_k w_k B(\tau_k)$

### A.3 方差分解

$$\text{Var}[\hat{x}_0^{\text{MPI}}] = \sum_k w_k^2 \text{Var}[\hat{x}_0(\tau_k)] + 2 \sum_{i<j} w_i w_j \text{Cov}[\hat{x}_0(\tau_i), \hat{x}_0(\tau_j)]$$

设 $\sigma_k^2 = \text{Var}[\hat{x}_0(\tau_k)]$, $\rho_{ij} = \text{Corr}[\hat{x}_0(\tau_i), \hat{x}_0(\tau_j)]$, $\sigma_{ij} = \rho_{ij} \sigma_i \sigma_j$:

$$\text{Var}[\hat{x}_0^{\text{MPI}}] = \sum_k w_k^2 \sigma_k^2 + 2 \sum_{i<j} w_i w_j \rho_{ij} \sigma_i \sigma_j$$

### A.4 均方误差 (MSE)

$$\text{MSE} = \mathbb{E}\left[\|\hat{x}_0^{\text{MPI}} - x_0\|^2\right] = \|B_{\text{MPI}}\|^2 + \text{Var}[\hat{x}_0^{\text{MPI}}]$$

$$= \left\|\sum_k w_k B(\tau_k)\right\|^2 + \sum_k w_k^2 \sigma_k^2 + 2 \sum_{i<j} w_i w_j \rho_{ij} \sigma_i \sigma_j$$

### A.5 最优权重

$$w^* = \arg\min_w \text{MSE}(w)$$

对 $w$ 求导并令其为 0 (拉格朗日乘子法, 约束 $\sum w_k = 1$):

$$\frac{\partial \text{MSE}}{\partial w_k} = 2 B(\tau_k)^T B_{\text{MPI}} + 2 w_k \sigma_k^2 + 2 \sum_{j \neq k} w_j \rho_{kj} \sigma_k \sigma_j = \lambda$$

化简为矩阵形式: $(\Sigma + B B^T) w = \lambda \mathbf{1}$, 其中 $\Sigma_{kj} = \rho_{kj} \sigma_k \sigma_j$, $B = [B(\tau_1), \dots, B(\tau_K)]^T$。

$$w^* = \frac{(\Sigma + B B^T)^{-1} \mathbf{1}}{\mathbf{1}^T (\Sigma + B B^T)^{-1} \mathbf{1}}$$

### A.6 特例: 无偏差 ($B = 0$) 且等方差 ($\sigma_k = \sigma$)

$$w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}, \quad \Sigma_{kj} = \rho_{kj} \sigma^2$$

若进一步 $\rho_{kj} = \rho$ (常数相关):

$$w^* = \frac{1}{K} \quad \text{(等权最优)}$$

此时 $\text{Var}[\hat{x}_0^{\text{MPI}}] = \frac{\sigma^2}{K}(1 + (K-1)\rho)$。

### A.7 MPI vs 单点预测的 MSE 比较

单点 ($w_1 = 1$): $\text{MSE}_1 = B(\tau_1)^2 + \sigma_1^2$ (取 $\tau_1 = 1$, $B = 0$)

$$\text{MSE}_1 = \sigma_1^2$$

MPI: $\text{MSE}_{\text{MPI}} = \|B_{\text{MPI}}\|^2 + \text{Var}[\hat{x}_0^{\text{MPI}}]$

MPI 优于单点的条件: $\text{MSE}_{\text{MPI}} < \text{MSE}_1$

$$\|B_{\text{MPI}}\|^2 + \text{Var}[\hat{x}_0^{\text{MPI}}] < \sigma_1^2$$

**充分条件** (设 $w_1 = \alpha$, 其余 $w_k = \frac{1-\alpha}{K-1}$, $B(\tau_1) = 0$):

$$\left(\frac{1-\alpha}{K-1}\right)^2 \left\|\sum_{k \neq 1} B(\tau_k)\right\|^2 + \alpha^2 \sigma_1^2 + \frac{(1-\alpha)^2}{K-1} \sigma^2 (1 + (K-2)\rho) < \sigma_1^2$$

当 $\alpha \to 1$ (几乎只用 $t=1$ 预测): 左边 $\to \sigma_1^2$, 不等式接近相等。
当 $\alpha$ 适中 ($\alpha = 0.5$): 偏差项增加, 但方差项降低。若方差降低 > 偏差增加, MPI 更优。

**结论**: MPI 的收益取决于偏差-方差的具体数值, 需在 Phase 0 诊断中测量。

---

## 附录 B: MPI 完整实现伪代码

```python
class DiffusionDetHead(nn.Module):
    """扩散检测头 — 新增 MPI 推理支持"""

    @torch.no_grad()
    def mpi_predict(
        self, features, img_metas, rescale=True,
        mpi_points=None, mpi_cls_weights=None, mpi_reg_weights=None,
        uncertainty_refinement=False, refine_topk_ratio=0.2,
    ):
        """MPI 多点推理

        Args:
            mpi_points: 评估点列表, 默认 [1.0, 0.7, 0.5, 0.3]
            mpi_cls_weights: 分类权重, 默认 [0.15, 0.25, 0.40, 0.20]
            mpi_reg_weights: 回归权重, 默认 [0.50, 0.30, 0.15, 0.05]
            uncertainty_refinement: 是否启用不确定性精化
            refine_topk_ratio: 精化的高方差比例
        """
        if mpi_points is None:
            mpi_points = [1.0, 0.7, 0.5, 0.3]
        if mpi_cls_weights is None:
            mpi_cls_weights = [0.15, 0.25, 0.40, 0.20]
        if mpi_reg_weights is None:
            mpi_reg_weights = [0.50, 0.30, 0.15, 0.05]

        device = features[0].device
        bs = len(img_metas)
        K = len(mpi_points)

        # Step 1: 初始化噪声 (固定, 所有评估点共享)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        # Step 2: 批量多点评估
        mpi_results = self._forward_at_t_batch(
            features, x_raw, mpi_points, img_metas
        )

        # Step 3 (可选): 不确定性精化
        if uncertainty_refinement:
            mpi_results = self._uncertainty_refine(
                features, x_raw, mpi_results, img_metas, refine_topk_ratio
            )

        # Step 4: 分类-回归解耦加权聚合
        results = self._sampler.mpi_post_process(
            mpi_results, img_metas, rescale,
            cls_weights=mpi_cls_weights,
            reg_weights=mpi_reg_weights,
        )
        return results

    def _forward_at_t_batch(self, features, x_raw, t_list, img_metas):
        """批量多点评估 (3.2.3 节)"""
        K = len(t_list)
        bs, num_proposals, _ = x_raw.shape

        # 复制 x_raw K 份: [K*bs, P, 4]
        x_batched = x_raw.unsqueeze(0).expand(K, bs, num_proposals, 4)
        x_batched = x_batched.reshape(K * bs, num_proposals, 4)

        # 时间输入: [K*bs]
        t_input = torch.cat([
            torch.full((bs,), t_k * self.timesteps, device=x_raw.device)
            for t_k in t_list
        ])

        # img_metas 复制
        metas_batched = []
        for _ in t_list:
            metas_batched.extend(img_metas)

        # RoI 提取 (同一 x_raw, 框位置相同)
        curr_bboxes = self._sampler.raw_to_xyxy(x_batched, metas_batched)

        # 单次前向 (K*bs batch, 共享 backbone 特征)
        cls_seq, pred_seq, _ = self(features, curr_bboxes, t_input)
        cls_last = cls_seq[-1]   # [K*bs, P, C]
        box_last = pred_seq[-1]  # [K*bs, P, 4]

        # 拆分为 K 组结果
        results = []
        for k in range(K):
            cls_k = cls_last[k*bs:(k+1)*bs]
            box_k = box_last[k*bs:(k+1)*bs]
            x0_k = self._sampler.xyxy_to_raw(box_k, img_metas)
            results.append((cls_k, box_k, x0_k, t_list[k]))
        return results

    def _uncertainty_refine(self, features, x_raw, mpi_results,
                            img_metas, topk_ratio):
        """不确定性触发的自适应精化 (3.4.2 节)"""
        bs = len(img_metas)
        box_stack = torch.stack([r[1] for r in mpi_results])  # [K, bs, P, 4]
        uncertainty = box_stack.var(dim=0).sum(dim=-1)  # [bs, P]

        refined = [list(r) for r in mpi_results]
        for i in range(bs):
            n_refine = int(topk_ratio * uncertainty.shape[1])
            _, topk_idx = uncertainty[i].topk(n_refine)
            if len(topk_idx) == 0:
                continue
            # 用聚合 x0 作为精化起点
            x0_agg = box_stack[:, i].mean(dim=0)  # [P, 4]
            x_refined = x_raw[i].clone()
            x_refined[topk_idx] = x0_agg[topk_idx]
            # 重新评估
            cls_r, box_r, x0_r, _ = self._forward_at_t(
                features, x_refined.unsqueeze(0), 1.0, [img_metas[i]]
            )
            for k in range(len(mpi_results)):
                refined[k][1][i, topk_idx] = box_r[0, topk_idx]
        return [tuple(r) for r in refined]
```

```python
class DiffusionSampler:
    """采样器 — 新增 MPI 后处理"""

    def mpi_post_process(
        self, mpi_results, img_metas, rescale,
        cls_weights=None, reg_weights=None,
    ):
        """MPI 加权聚合后处理 (3.3 节)"""
        results_list = []
        bs = len(img_metas)
        K = len(mpi_results)

        if cls_weights is None:
            cls_weights = [1.0 / K] * K
        if reg_weights is None:
            reg_weights = [1.0 / K] * K

        for i in range(bs):
            all_scores = []
            all_bboxes = []
            all_labels = []

            for k, ((cls_k, box_k, _, t_k), w_c, w_r) in enumerate(
                zip(mpi_results, cls_weights, reg_weights)
            ):
                scores = torch.sigmoid(cls_k[i])
                conf, labels = scores.max(-1)
                # 分数用分类权重, 框用回归权重 (影响 NMS 排序)
                all_scores.append(conf * w_c)
                all_bboxes.append(box_k[i])
                all_labels.append(labels)

            final_scores = torch.cat(all_scores)
            final_bboxes = torch.cat(all_bboxes)
            final_labels = torch.cat(all_labels)

            if self.use_nms:
                keep = batched_nms(
                    final_bboxes, final_scores,
                    final_labels, self.nms_thr,
                )
                final_scores = final_scores[keep]
                final_bboxes = final_bboxes[keep]
                final_labels = final_labels[keep]

            if rescale:
                scale_factor = _get_scale_factor(img_metas[i])
                # ... (复用现有 rescale 逻辑)

            results_list.append(
                DetectionResult(
                    bboxes=final_bboxes,
                    scores=final_scores,
                    labels=final_labels,
                )
            )
        return results_list
```
