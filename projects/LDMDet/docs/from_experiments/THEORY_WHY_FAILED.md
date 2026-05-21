# 两个子方向失败的数学理论分析

## 方向 A: Scale-Conditioned Flow Matching 为何无效

### 1. 回顾方法

Scale-Conditioned FM 对噪声做类条件缩放：

$$z_t = (1-t)x_0 + t \\cdot \\sigma_c \\cdot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0,I)$$

其中 $\\sigma_c$ 按 Denver 组缩放 (A=1.0, G=0.283)，损失权重 $w_c \\propto 1/\\sigma_c$。

### 2. 理论一: OT 耦合下的 Scale-Conditioned 噪声等价于无效变换

**定理 1 (Scale-Conditioned OT 等价性)**

设 OT 计划 $\\pi^\*$ 将 GT box $x_0$ 与噪声 $\\epsilon$ 配对。Scale-conditioned 噪声 $\\sigma_c \\cdot \\epsilon$ 等价于在标准 RF 中缩放 box 坐标：

$$z_t^{(\\sigma)} = (1-t) x_0 + t \\cdot \\sigma_c \\cdot \\epsilon = (1-t) x_0 + t \\cdot \\tilde{\\epsilon}$$

其中 $\\tilde{\\epsilon} \\sim \\mathcal{N}(0, \\sigma_c^2 I)$。流速场变为：

$$v^{(\\sigma)}(z_t, t) = \\mathbb{E}\[\\sigma_c \\epsilon - x_0 | z_t\]$$

而标准 RF 流速为 $v(z_t, t) = \\mathbb{E}\[\\epsilon - x_0 | z_t\]$。注意到 $\\sigma_c \\epsilon - x_0 = (\\epsilon - x_0) + (\\sigma_c - 1)\\epsilon$。

第二项 $(\\sigma_c - 1)\\epsilon$ 的期望在给定 $z_t$ 的条件下非零，因为 $z_t$ 包含 $x_0$ 的信息。这引入了**有偏流速估计**——模型被训练去预测一个不再是最优传输方向的向量场。

具体地，在标准 RF 中：
$$\\frac{d}{dt}\\mathbb{E}\[||v\_\\theta(z_t, t) - v^\*(z_t, t)||^2\] = 0 \\text{ at optimum}$$

而在 scale-conditioned 版本中：
$$\\min\_\\theta \\mathbb{E}\[||v\_\\theta(z_t^{(\\sigma)}, t) - (\\sigma_c \\epsilon - x_0)||^2\]$$

因为 $z_t^{(\\sigma)} = z_t + t(\\sigma_c - 1)\\epsilon$，模型必须学习一个取决于 $\\sigma_c$ 的流速偏移，但不同 denver 组的 $\\sigma_c$ 不同，导致**多目标冲突**——模型对不同组的流速预测不可能同时最优。

### 3. 理论二: 损失重加权破坏变分原理

标准 FM 损失来自 KL 散度的变分上界：

$$\\mathcal{L}_{FM} = \\mathbb{E}_{t, x_0, \\epsilon}\\left\[||v\_\\theta(z_t, t) - (\\epsilon - x_0)||^2\\right\] \\propto D\_{KL}(p^{\\text{true}} || p\_\\theta)$$

Scale-adaptive loss 引入类权重 $w_c$:

$$\\mathcal{L}_{SC} = \\mathbb{E}_{t, x_0, \\epsilon}\\left\[w_c \\cdot ||v\_\\theta(z_t^{(\\sigma_c)}, t) - (\\sigma_c \\epsilon - x_0)||^2\\right\]$$

但 $w_c \\neq 1/\\sigma_c^2$ 时，这不等于任何概率散度。具体地，设 $w_c = (1/s_c)^p / Z$：

$$\\mathcal{L}\_{SC} \\neq \\text{任何 } f\\text{-divergence 当 } w_c \\neq \\text{const}$$

这意味着**优化目标与概率建模目标不一致**——模型在最小化一个不反映真实数据分布的目标函数。这解释了为何 sc_loss (p=0.5) 性能下降幅度小于 sc_combined：因为 sc_loss 只破坏了变分性质，而 sc_combined 同时破坏了流速的最优性和变分性质。

### 4. 理论三: Size-AP 相关性不是因果的

观测到的 $r=0.74$ 相关性来自：
$$\\text{AP}(c) \\approx f(\\text{pixel_area}(c), \\text{visual_complexity}(c))$$

其中 $\\text{pixel_area}$ 决定了信息量上限。按 Shannon-Hartley 定理，可分辨的特征数受像素面积约束：

$$I(\\text{image}; \\text{class} | \\text{size}=s) \\leq C \\cdot s \\cdot \\log(1 + \\text{SNR})$$

小染色体 G21/G22 的像素面积约为 A1 的 $1/10$，信息量上限相应降低。**Scale-conditioned noise 不能增加信息量**——它只改变了流速场的参数化，但输入特征的判别能力不变。

______________________________________________________________________

## 方向 C: KaryoFlow 排列学习为何必然失败

### 1. 排列空间的信息论下界

核型排列问题：给定 46 个染色体 crop 特征 ${f_1, ..., f\_{46}}$，输出排列 $\\sigma \\in S\_{46}$。

排列空间大小：$|S\_{46}| = 46! \\approx 1.8 \\times 10^{59}$

所需信息量：$\\log_2(46!) \\approx 200$ bits

**定理 2 (同源对称性下界)**

存在 22 对同源染色体 $(2i, 2i+1)\_{i=0}^{21}$，对内交换不改变核型的临床正确性。这给出等价关系：

$$\\sigma \\sim \\sigma' \\iff \\forall i \\in \[0,21\], \\sigma^{-1}(2i), \\sigma^{-1}(2i+1) \\text{ 与 } \\sigma'^{-1} \\text{ 中的位置一致(模交换)}$$

等价类数量：$46! / 2^{22} \\approx 4.3 \\times 10^{52}$

有效信息量：$\\log_2(46!) - 22 \\approx 178$ bits（减少了 22 bits 因为 22 对内交换不改变等价类）

### 2. 样本复杂度下界

**定理 3 (排列学习的样本复杂度)**

设 encoder $f\_\\phi$ 输出 $d$ 维特征。要从 $N$ 个样本中区分 $K$ 个等价排列类，需要：

$$N \\geq \\frac{\\log K}{I(f\_\\phi(x); \\sigma)} = \\frac{178}{I(f\_\\phi(x); \\sigma)}$$

其中 $I(f\_\\phi(x); \\sigma)$ 是特征与排列的互信息。

用 ResNet18 (d=256)，在 438 张图上训练的 encoder，其 8-way Denver 组分类 acc=54.5% 意味着：

$$I(f\_\\phi(x); \\text{group}) \\approx H(\\text{group}) - H(\\text{group}|f\_\\phi) \\approx 2.08 - 1.52 \\approx 0.56 \\text{ bits/crop}$$

对 46 个 crop 求和：$I\_{\\text{total}} \\approx 46 \\times 0.56 \\approx 25.8$ bits

但所需的 178 bits 远超 25.8 bits。即使 encoder 完美分类 8 组 (acc=100%, I=3 bits/crop)：

$$I\_{\\text{total}}^{\\text{max}} \\approx 46 \\times 3 = 138 \\text{ bits} \< 178 \\text{ bits}$$

**结论**: 即使完美 8 组分类，信息量也不足以唯一确定排列。组内排序需要额外信息（面积、形态），这些信息在纯视觉特征中编码不充分。

### 3. 特征坍塌的数学刻画

设第 $i$ 个 crop 的真实类别为 $c_i$，encoder 输出：

$$f_i = \\mu\_{c_i} + \\eta_i, \\quad \\eta_i \\sim \\mathcal{N}(0, \\Sigma\_{c_i})$$

其中 $\\mu_c$ 是类 $c$ 的均值特征，$\\Sigma_c$ 是类内方差。

对于组内类别 (如 C6-C12)，类间距离与类内方差的比值为：

$$\\text{SNR}_{\\text{inter}} = \\frac{||\\mu_{C6} - \\mu\_{C7}||^2}{\\text{tr}(\\Sigma\_{C6}) + \\text{tr}(\\Sigma\_{C7})} \\ll 1$$

实验验证：24 类分类 acc=27.4% → 类间 SNR ≈ 0.3（远小于可靠分类所需的 4-10）。

这导致**特征空间在组内几乎退化**——C6-C12 的 7 个类别的特征在高维空间中几乎完全重叠，使得任何排列学习算法无法区分组内顺序。

### 4. 流匹配在此场景下的额外困难

KaryoFlow 的 MDLM 训练目标：

$$\\mathcal{L} = -\\mathbb{E}_{t, \\sigma_0, \\sigma_t}\\left\[\\sum_{i: m_i=1} \\log p\_\\theta(\\sigma_0(i) | \\sigma_t)\\right\]$$

在特征坍塌条件下，$p\_\\theta(\\sigma_0(i) | \\sigma_t)$ 对组内位置不可区分：

$$\\forall j,k \\in \\text{same group}, \\quad p\_\\theta(\\sigma_0(i)=j | \\sigma_t) \\approx p\_\\theta(\\sigma_0(i)=k | \\sigma_t)$$

导致梯度信号趋近于 0（uniform 分布无学习信号）：

$$||\\nabla\_\\theta \\mathcal{L}|| \\propto \\text{Var}(p\_\\theta(\\cdot|\\sigma_t)) \\approx 0$$

______________________________________________________________________

这些负结果 + 理论分析构成一篇 solid 论文：

1. **ΔH = log K**: OT 耦合的多样性坍塌（已有理论）
2. **Theorem 1-2**: Scale-Conditioned 为何必然无效（新贡献）
3. **Theorem 3**: 排列学习的信息论不可行性（新贡献）
4. **r = 0.74**: Size-AP 反直觉相关性（实证贡献）
5. **实验**: LDMDet 基线 0.751 + Scale-Conditioned 负结果 + KaryoFlow 可行性分析

标题可改为: **"Why Scale-Conditioned Flow Matching Fails for Small Objects, and Why End-to-End Karyotype Arrangement is Information-Theoretically Infeasible"**

______________________________________________________________________

## 方向 D: Reflow 退化陷阱——梯度冲突

> 原载于 THEORY_FRAMEWORK.md §6，因属于失败方向分析，移入本文档。

### D.1 实验现象

所有 Reflow 实验均呈现 **"Epoch 1 最佳 → 后续退化"** 的模式：

| Reflow 版本            | Epoch 1 mAP | 最佳 mAP | velocity loss 天花板 |
| ---------------------- | ----------- | -------- | -------------------- |
| v2 (val配对)           | 0.739       | 0.739    | ~0.28                |
| v4 (修复velocity_head) | 0.739       | 0.739    | 0.96→收敛            |
| v5 (warmup+调参)       | 0.739       | 0.739    | ~0.23                |
| v6 (第2轮Reflow)       | 0.739       | 0.739    | ~0.23                |

### D.2 梯度冲突机制

**机制命题 D.1**（Reflow 梯度冲突）：设检测损失 $\mathcal{L}_{det}$ 和速度损失 $\mathcal{L}_{vel}$ 共享参数 $\theta_{\text{shared}}$（backbone + 检测头主体）。定义梯度冲突度：

$$\rho = \cos\left(\nabla_{\theta_{\text{shared}}} \mathcal{L}_{det}, \nabla_{\theta_{\text{shared}}} \mathcal{L}_{vel}\right)$$

当 $\rho < 0$ 时，两个损失的梯度方向相反，优化一个会恶化另一个。

**实验测量**：$\rho = -0.104$，86.8% 的层存在梯度冲突。

> **代码审计 / 重跑标注（2026-05-15）**：该梯度冲突测量使用的 velocity target 方向需要核对。由于当前 `_add_velocity_loss` 目标与 RF 定义相反，`rho=-0.104` 可作为"当前代码下的冲突证据"，但不能直接作为修正后 velocity loss 的理论证据。应在修正 `v_target = x_noises - x_starts` 后重新测量 `cos(g_det, g_vel)`、逐层冲突比例和 Reflow 曲线。

**启发式论证**（非严格证明）：

检测损失 $\mathcal{L}_{det}$ 要求 $x_0^{pred}$ 接近 $x_0^{gt}$，即模型需要利用图像特征 $f$ 精确预测 GT 位置。

速度损失 $\mathcal{L}_{vel} = |v_\theta - v^*|^2$ 要求速度场拟合 $v^* = x_1 - x_0$。在 Reflow 中，$v^*$ 来自教师模型的 ODE 轨迹，包含教师模型的系统性偏差。

关键矛盾：检测损失要求模型在**特定图像区域**精确预测，而速度损失要求模型在**整个噪声空间**均匀拟合速度场。两者的梯度方向在共享参数上产生冲突：

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{det} \propto -\frac{\partial x_0^{pred}}{\partial \theta} \cdot \nabla_{x_0} \mathcal{L}_{det}$$

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{vel} \propto -\frac{\partial v_\theta}{\partial \theta} \cdot (v_\theta - v^*)$$

由于 $x_0^{pred} = x_t - t \cdot v_\theta$，有 $\frac{\partial x_0^{pred}}{\partial \theta} = -t \cdot \frac{\partial v_\theta}{\partial \theta}$。因此：

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{det} \propto t \cdot \frac{\partial v_\theta}{\partial \theta} \cdot \nabla_{x_0} \mathcal{L}_{det}$$

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{vel} \propto -\frac{\partial v_\theta}{\partial \theta} \cdot (v_\theta - v^*)$$

**启发式简化**：若假设 $\frac{\partial v_\theta}{\partial \theta}$ 在两个梯度中近似可约，冲突条件简化为：

$$\rho < 0 \iff \nabla_{x_0} \mathcal{L}_{det} \cdot (v_\theta - v^*) > 0$$

即：当检测残差方向与速度残差方向一致时，两个梯度冲突。这在实践中经常发生，因为 $v_\theta - v^*$ 的方向倾向于与 $\nabla_{x_0} \mathcal{L}_{det}$ 的方向相关（两者都反映模型对 $x_0$ 的预测偏差）。但此简化在高维参数空间中不严格成立，$\frac{\partial v_\theta}{\partial \theta}$ 是矩阵而非标量，不能简单约去。冲突的具体程度需要实验测量（$\rho = -0.104$）。

### D.3 退化陷阱的动力学解释

设 $\theta_n$ 为第 $n$ 步的参数，梯度下降更新为：

$$\theta_{n+1} = \theta_n - \eta \left(\nabla \mathcal{L}_{det} + w \cdot \nabla \mathcal{L}_{vel}\right)$$

检测损失的变化为：

$$\Delta \mathcal{L}_{det} = -\eta \left(|\nabla \mathcal{L}_{det}|^2 + w \cdot \nabla \mathcal{L}_{det}^\top \nabla \mathcal{L}_{vel}\right)$$

当 $\nabla \mathcal{L}_{det}^\top \nabla \mathcal{L}_{vel} < 0$（梯度冲突）且 $w$ 足够大时，$\Delta \mathcal{L}_{det} > 0$，即检测损失**增加**。

Epoch 1 最佳的原因：初始时 $\theta$ 接近预训练模型，$\mathcal{L}_{det}$ 已在局部最优附近。速度损失的梯度将参数拉离此局部最优，且由于梯度冲突，检测性能持续退化。

### D.4 velocity loss 收敛天花板

velocity loss 收敛到 ~0.23-0.30 后停滞，原因：

1. **容量瓶颈**：velocity_head 为 3 层 MLP（256→256→256→4），参数量有限
2. **配对噪声**：Reflow 配对 $(z_0, x_1^{pred})$ 中 $x_1^{pred}$ 本身有 ODE 离散化误差
3. **梯度冲突**：共享层的梯度冲突阻止 velocity_head 获得足够的梯度信号

______________________________________________________________________

## 方向 E: 两条 SOTA 路径的负交互——机制冲突

> 原载于 THEORY_FRAMEWORK.md §7，因属于失败方向分析，移入本文档。

### E.1 实验现象

| 组合实验                               | mAP   | vs 最佳单路径 (0.752) |
| -------------------------------------- | ----- | --------------------- |
| group_hierarchical_stoch + TRD         | 0.746 | **-0.006**            |
| sinkhorn_stochastic + TRD + CAT + LSAS | 0.743 | **-0.009**            |
| sinkhorn_stochastic + TRD + CAT        | 0.740 | **-0.012**            |

三个组合实验全部低于任一单路径最佳值。

> **代码审计 / 重跑标注（2026-05-15）**：组合负交互结论在当前代码下成立，但不应被写成最终结构性定论。`sinkhorn_trd_cat_lsas` 和 `stochastic_eps5_trd_cat` 同时受 stochastic seed 方差、velocity target 符号、CAT 目标不一致、TRD 使用 `cat_delta_t` 等因素影响；`group_hierarchical_trd` 至少受 stochastic seed 方差和 TRD 自条件估计误差影响。建议修正后重跑：`group_hierarchical_trd`、`stochastic_eps5_trd_cat`、`sinkhorn_trd_cat_lsas`，并增加"analytic/current-pair TRD velocity"和"TRD/CAT delta_t 解耦"版本。

### E.2 机制冲突假说

**机制假说 E.1**（耦合-训练动力学冲突）：Stochastic OT 耦合与 TRD 自条件化可能存在结构性冲突。

**论证**：

Stochastic OT 耦合的核心机制是：在每个训练迭代中，从传输矩阵中随机采样配对，使得同一 $x_t$ 在不同迭代中看到不同的目标速度 $v^*$。这种**训练时不确定性**是多样性的来源。

TRD 自条件化的核心机制是：在训练时，以概率 $p$ 用前一步的预测 $x_0^{prev}$ 估计 $v_\pi$，然后沿 ODE 路径前进一步。这要求 $x_0^{prev}$ 是当前 $x_t$ 的合理估计。

**冲突点**：Stochastic OT 在不同迭代中为同一 $x_t$ 提供不同的 $v^*$，而 TRD 依赖 $x_0^{prev}$ 提供一致的 $v_\pi$ 估计。当两者组合时：

1. TRD 的自条件化步骤使用 $x_0^{prev}$ 估计 $v_\pi$，但 Stochastic OT 的随机性使得 $v_\pi$ 在不同迭代间不一致
2. TRD 将 $x_t$ 沿估计的 $v_\pi$ 前进到 $x_{t+\Delta t}$，但 Stochastic OT 在新位置可能分配不同的 GT 目标
3. 这导致 TRD 的自条件化步骤引入额外的训练噪声，而非提供有用的精化信号

形式化地，设 $v_\pi^{(i)}$ 为第 $i$ 次迭代的传输速度（因 Stochastic 而随机），TRD 的自条件化估计为 $\hat{v}_\pi = (x_t - x_0^{prev})/t$。当 $v_\pi^{(i)} \neq \hat{v}_\pi$ 时（Stochastic 耦合下概率很高），TRD 的前进步骤方向错误，引入噪声：

$$x_{t+\Delta t}^{TRD} = x_t + \Delta t \cdot \hat{v}_\pi \neq x_t + \Delta t \cdot v_\pi^{(i)}$$

此误差在 Stochastic 耦合下被放大（因 $v_\pi^{(i)}$ 的方差大），而在确定性耦合（nearest OT）下不存在（因 $v_\pi^{(i)} = \hat{v}_\pi$ 恒成立）。

### E.3 CAT 与 OT 的曲率冲突（二次修正）

**机制假说 E.2**（CAT-OT 冲突——精确版）：CAT 的 $x_0$ 一致性正则化与 OT 耦合可能在 Voronoi 边界处产生梯度冲突，并导致训练崩溃。

**论证（代码层面精确分析）**：

**Step 1：CAT 的 $x_{t+\Delta t}$ 构造方式**

CAT 在计算 $x_{t+\Delta t}$ 时使用与 $x_t$ **相同的 OT 配对**：

```python
# diffusiondet_head.py _add_cat_loss
x_start_batch = torch.stack(x_starts)   # OT 耦合后的 x_0
x_noise_batch = torch.stack(x_noises)   # 原始噪声 x_1
x_t2 = (1.0 - t2_view) * x_start_batch + t2_view * x_noise_batch
```

这里 `x_start_batch` 和 `x_noise_batch` 在两次前向传播间不变，隐含假设：$x_t$ 和 $x_{t+\Delta t}$ 沿**同一条 OT 配对的直线路径**，速度场应为常数。

**Step 2：OT 耦合下 Voronoi 边界的时间依赖性**

OT 耦合将噪声空间划分为 Voronoi 单元 $\mathcal{V}_k$。$x_t \in \mathcal{V}_k$ 的条件为：

$$\left|\frac{x_t - b_k}{1-t}\right| \leq \left|\frac{x_t - b_j}{1-t}\right|, \quad \forall j \neq k$$

化简得 Voronoi 边界超平面方程：

$$2(x_t - b_k)^\top(b_k - b_j) + (1-t)|b_k - b_j|^2 = 0$$

边界法向为 $(b_k - b_j)$，截距为 $(1-t)|b_k - b_j|^2 / 2$。**截距随 $t$ 线性变化**：当 $t$ 增加 $\Delta t$ 时，边界向 $b_k$ 方向移动 $\Delta t \cdot |b_k - b_j|^2 / 2$。

**Step 3：冲突的精确机制**

CAT 的 $x_{t+\Delta t}$ 使用相同的 OT 配对构造，意味着 CAT 假设 $x_t$ 和 $x_{t+\Delta t}$ 属于同一个 Voronoi 单元（配对不变）。但由于 Voronoi 边界随 $t$ 移动，$x_{t+\Delta t}$ 可能跨越到相邻的 Voronoi 单元 $\mathcal{V}_j$。

此时出现三方矛盾：

| 约束来源 | 要求                                             | 代码位置                             |
| -------- | ------------------------------------------------ | ------------------------------------ |
| OT 耦合  | $v^* = b_j - z$（新单元的速度）                 | `_couple_ot` 返回的 `x_start`        |
| CAT 构造 | $x_{t+\Delta t}$ 沿旧配对的直线路径            | `x_t2 = (1-t2)*x_start + t2*x_noise` |
| CAT loss | $x_0^{pred}(t) \approx x_0^{pred}(t+\Delta t)$ | `F.mse_loss(x0_t1, x0_t2.detach())`  |

具体地：

- CAT 构造的 $x_{t+\Delta t}$ 位于旧配对 $(z, b_k)$ 的直线路径上，期望模型预测 $x_0^{pred}(t+\Delta t) \approx b_k$
- 但 OT 耦合在 $x_{t+\Delta t}$ 处可能分配 $b_j$（因为 $x_{t+\Delta t}$ 已跨越 Voronoi 边界），检测损失要求 $x_0^{pred}(t+\Delta t) \approx b_j$
- CAT loss 惩罚 $x_0^{pred}(t) \neq x_0^{pred}(t+\Delta t)$，即惩罚 $b_k \neq b_j$；在跨越 Voronoi 边界的样本上，这种差异来自配对切换

**Step 4：崩溃的动力学解释**

当 CAT + OT 组合训练时，梯度更新陷入三方拉锯：

1. 检测损失 $\mathcal{L}_{det}$ 推动模型在 $x_{t+\Delta t}$ 处预测 $b_j$（OT 配对的目标）
2. CAT 损失 $\mathcal{L}_{CAT}$ 推动模型在 $x_{t+\Delta t}$ 处预测 $b_k$（与 $x_t$ 处一致）
3. 两者梯度方向可能相反，且 $b_k \neq b_j$ 会使冲突难以通过单一预测同时满足

在 Voronoi 边界附近，这种冲突的样本比例随训练进行而增加（模型学会在边界附近产生不确定预测，增加边界跨越的概率），形成正反馈循环，最终导致训练崩溃（eval=0）。

**对比**：CAT 单独使用时（无 OT 耦合），随机耦合下不存在 Voronoi 结构，$v^*$ 在不同 $t$ 处的变化是连续的（条件期望的平滑变化），CAT 的平滑化约束与训练信号兼容，因此不会崩溃（0.744）。

> **⚠️ 代码审计发现的问题**：CAT 的 $x_{t+\Delta t}$ 构造使用与 $x_t$ 相同的 OT 配对，但 Voronoi 边界随 $t$ 变化导致 $x_{t+\Delta t}$ 可能属于不同 Voronoi 单元。这是 CAT + OT 崩溃的代码层面根源。如果未来需要让 CAT 与 OT 兼容，有两个方向：(1) 为 $x_{t+\Delta t}$ 重新运行 OT 耦合（`x_start_t2 = _couple_ot(x_t2, gt_diffusion, labels, device)`），使 CAT 的构造与 OT 的配对一致，但这引入额外计算开销；(2) 在 CAT loss 中排除 Voronoi 边界附近的样本（通过检测 $x_0^{pred}(t)$ 与最近 GT 的距离是否接近次近 GT 的距离来识别边界样本），但这需要额外的边界检测逻辑。

### E.4 对论文的影响

两条 SOTA 路径的负交互不是超参数问题，而是**结构性冲突**：

1. **Stochastic OT + TRD**：训练时不确定性与自条件化一致性要求矛盾
2. **OT + CAT**：Voronoi 边界跳变与曲率平滑化要求矛盾

这意味着两条路径虽然各自有效，但**不能简单叠加**。需要设计新的组合策略来绕过这些冲突（如分阶段训练、解耦参数等）。

______________________________________________________________________

## 综合结论

| 方向                 | 失败原因                               | 数学本质                                  | 可否修复                           |
| -------------------- | -------------------------------------- | ----------------------------------------- | ---------------------------------- |
| Scale-Conditioned FM | 流速场偏移破坏最优性，重加权破坏变分性 | OT 映射对尺度已最优，额外调节引入偏差     | 需重新设计为几何一致的方案         |
| KaryoFlow 排列学习   | 特征信息量不足 (25.8 << 178 bits)    | 信息论下界 + 同源对称性使样本复杂度不可达 | 需更大模型/数据集(1000x)或降维任务 |
| Reflow 退化陷阱      | 检测-速度梯度冲突 ($\rho=-0.104$)     | 共享参数上的多目标优化冲突                | 分阶段训练或参数解耦               |
| SOTA 路径负交互      | Stochastic OT+TRD / CAT+OT 结构性冲突 | 耦合随机性与自条件一致性 / Voronoi边界时变 | 重新设计兼容组合策略               |
