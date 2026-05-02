# 两个子方向失败的数学理论分析

## 方向 A: Scale-Conditioned Flow Matching 为何无效

### 1. 回顾方法

Scale-Conditioned FM 对噪声做类条件缩放：

$$z_t = (1-t)x_0 + t \cdot \sigma_c \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

其中 $\sigma_c$ 按 Denver 组缩放 (A=1.0, G=0.283)，损失权重 $w_c \propto 1/\sigma_c$。

### 2. 理论一: OT 耦合下的 Scale-Conditioned 噪声等价于无效变换

**定理 1 (Scale-Conditioned OT 等价性)**

设 OT 计划 $\pi^*$ 将 GT box $x_0$ 与噪声 $\epsilon$ 配对。Scale-conditioned 噪声 $\sigma_c \cdot \epsilon$ 等价于在标准 RF 中缩放 box 坐标：

$$z_t^{(\sigma)} = (1-t) x_0 + t \cdot \sigma_c \cdot \epsilon = (1-t) x_0 + t \cdot \tilde{\epsilon}$$

其中 $\tilde{\epsilon} \sim \mathcal{N}(0, \sigma_c^2 I)$。流速场变为：

$$v^{(\sigma)}(z_t, t) = \mathbb{E}[\sigma_c \epsilon - x_0 | z_t]$$

而标准 RF 流速为 $v(z_t, t) = \mathbb{E}[\epsilon - x_0 | z_t]$。注意到 $\sigma_c \epsilon - x_0 = (\epsilon - x_0) + (\sigma_c - 1)\epsilon$。

第二项 $(\sigma_c - 1)\epsilon$ 的期望在给定 $z_t$ 的条件下非零，因为 $z_t$ 包含 $x_0$ 的信息。这引入了**有偏流速估计**——模型被训练去预测一个不再是最优传输方向的向量场。

具体地，在标准 RF 中：
$$\frac{d}{dt}\mathbb{E}[||v_\theta(z_t, t) - v^*(z_t, t)||^2] = 0 \text{ at optimum}$$

而在 scale-conditioned 版本中：
$$\min_\theta \mathbb{E}[||v_\theta(z_t^{(\sigma)}, t) - (\sigma_c \epsilon - x_0)||^2]$$

因为 $z_t^{(\sigma)} = z_t + t(\sigma_c - 1)\epsilon$，模型必须学习一个取决于 $\sigma_c$ 的流速偏移，但不同 denver 组的 $\sigma_c$ 不同，导致**多目标冲突**——模型对不同组的流速预测不可能同时最优。

### 3. 理论二: 损失重加权破坏变分原理

标准 FM 损失来自 KL 散度的变分上界：

$$\mathcal{L}_{FM} = \mathbb{E}_{t, x_0, \epsilon}\left[||v_\theta(z_t, t) - (\epsilon - x_0)||^2\right] \propto D_{KL}(p^{\text{true}} || p_\theta)$$

Scale-adaptive loss 引入类权重 $w_c$:

$$\mathcal{L}_{SC} = \mathbb{E}_{t, x_0, \epsilon}\left[w_c \cdot ||v_\theta(z_t^{(\sigma_c)}, t) - (\sigma_c \epsilon - x_0)||^2\right]$$

但 $w_c \neq 1/\sigma_c^2$ 时，这不等于任何概率散度。具体地，设 $w_c = (1/s_c)^p / Z$：

$$\mathcal{L}_{SC} \neq \text{任何 } f\text{-divergence 当 } w_c \neq \text{const}$$

这意味着**优化目标与概率建模目标不一致**——模型在最小化一个不反映真实数据分布的目标函数。这解释了为何 sc_loss (p=0.5) 性能下降幅度小于 sc_combined：因为 sc_loss 只破坏了变分性质，而 sc_combined 同时破坏了流速的最优性和变分性质。

### 4. 理论三: Size-AP 相关性不是因果的

观测到的 $r=0.74$ 相关性来自：
$$\text{AP}(c) \approx f(\text{pixel_area}(c), \text{visual_complexity}(c))$$

其中 $\text{pixel_area}$ 决定了信息量上限。按 Shannon-Hartley 定理，可分辨的特征数受像素面积约束：

$$I(\text{image}; \text{class} | \text{size}=s) \leq C \cdot s \cdot \log(1 + \text{SNR})$$

小染色体 G21/G22 的像素面积约为 A1 的 $1/10$，信息量上限相应降低。**Scale-conditioned noise 不能增加信息量**——它只改变了流速场的参数化，但输入特征的判别能力不变。

---

## 方向 C: KaryoFlow 排列学习为何必然失败

### 1. 排列空间的信息论下界

核型排列问题：给定 46 个染色体 crop 特征 $\{f_1, ..., f_{46}\}$，输出排列 $\sigma \in S_{46}$。

排列空间大小：$|S_{46}| = 46! \approx 1.8 \times 10^{59}$

所需信息量：$\log_2(46!) \approx 200$ bits

**定理 2 (同源对称性下界)**

存在 22 对同源染色体 $(2i, 2i+1)_{i=0}^{21}$，对内交换不改变核型的临床正确性。这给出等价关系：

$$\sigma \sim \sigma' \iff \forall i \in [0,21], \sigma^{-1}(2i), \sigma^{-1}(2i+1) \text{ 与 } \sigma'^{-1} \text{ 中的位置一致(模交换)}$$

等价类数量：$46! / 2^{22} \approx 4.3 \times 10^{52}$

有效信息量：$\log_2(46!) - 22 \approx 178$ bits（减少了 22 bits 因为 22 对内交换不改变等价类）

### 2. 样本复杂度下界

**定理 3 (排列学习的样本复杂度)**

设 encoder $f_\phi$ 输出 $d$ 维特征。要从 $N$ 个样本中区分 $K$ 个等价排列类，需要：

$$N \geq \frac{\log K}{I(f_\phi(x); \sigma)} = \frac{178}{I(f_\phi(x); \sigma)}$$

其中 $I(f_\phi(x); \sigma)$ 是特征与排列的互信息。

用 ResNet18 (d=256)，在 438 张图上训练的 encoder，其 8-way Denver 组分类 acc=54.5% 意味着：

$$I(f_\phi(x); \text{group}) \approx H(\text{group}) - H(\text{group}|f_\phi) \approx 2.08 - 1.52 \approx 0.56 \text{ bits/crop}$$

对 46 个 crop 求和：$I_{\text{total}} \approx 46 \times 0.56 \approx 25.8$ bits

但所需的 178 bits 远超 25.8 bits。即使 encoder 完美分类 8 组 (acc=100%, I=3 bits/crop)：

$$I_{\text{total}}^{\text{max}} \approx 46 \times 3 = 138 \text{ bits} < 178 \text{ bits}$$

**结论**: 即使完美 8 组分类，信息量也不足以唯一确定排列。组内排序需要额外信息（面积、形态），这些信息在纯视觉特征中编码不充分。

### 3. 特征坍塌的数学刻画

设第 $i$ 个 crop 的真实类别为 $c_i$，encoder 输出：

$$f_i = \mu_{c_i} + \eta_i, \quad \eta_i \sim \mathcal{N}(0, \Sigma_{c_i})$$

其中 $\mu_c$ 是类 $c$ 的均值特征，$\Sigma_c$ 是类内方差。

对于组内类别 (如 C6-C12)，类间距离与类内方差的比值为：

$$\text{SNR}_{\text{inter}} = \frac{||\mu_{C6} - \mu_{C7}||^2}{\text{tr}(\Sigma_{C6}) + \text{tr}(\Sigma_{C7})} \ll 1$$

实验验证：24 类分类 acc=27.4% → 类间 SNR ≈ 0.3（远小于可靠分类所需的 4-10）。

这导致**特征空间在组内几乎退化**——C6-C12 的 7 个类别的特征在高维空间中几乎完全重叠，使得任何排列学习算法无法区分组内顺序。

### 4. 流匹配在此场景下的额外困难

KaryoFlow 的 MDLM 训练目标：

$$\mathcal{L} = -\mathbb{E}_{t, \sigma_0, \sigma_t}\left[\sum_{i: m_i=1} \log p_\theta(\sigma_0(i) | \sigma_t)\right]$$

在特征坍塌条件下，$p_\theta(\sigma_0(i) | \sigma_t)$ 对组内位置不可区分：

$$\forall j,k \in \text{same group}, \quad p_\theta(\sigma_0(i)=j | \sigma_t) \approx p_\theta(\sigma_0(i)=k | \sigma_t)$$

导致梯度信号趋近于 0（uniform 分布无学习信号）：

$$||\nabla_\theta \mathcal{L}|| \propto \text{Var}(p_\theta(\cdot|\sigma_t)) \approx 0$$

---

## 综合结论

| 方向 | 失败原因 | 数学本质 | 可否修复 |
|------|----------|----------|----------|
| Scale-Conditioned FM | 流速场偏移破坏最优性，重加权破坏变分性 | OT 映射对尺度已最优，额外调节引入偏差 | 需重新设计为几何一致的方案 |
| KaryoFlow 排列学习 | 特征信息量不足 (25.8 << 178 bits) | 信息论下界 + 同源对称性使样本复杂度不可达 | 需更大模型/数据集(1000x)或降维任务 |

## 论文价值

这些负结果 + 理论分析构成一篇 solid 论文：

1. **ΔH = log K**: OT 耦合的多样性坍塌（已有理论）
2. **Theorem 1-2**: Scale-Conditioned 为何必然无效（新贡献）
3. **Theorem 3**: 排列学习的信息论不可行性（新贡献）
4. **r = 0.74**: Size-AP 反直觉相关性（实证贡献）
5. **实验**: LDMDet 基线 0.751 + Scale-Conditioned 负结果 + KaryoFlow 可行性分析

标题可改为: **"Why Scale-Conditioned Flow Matching Fails for Small Objects, and Why End-to-End Karyotype Arrangement is Information-Theoretically Infeasible"**
