# LDMDet 统一数学理论框架

## 从工程改进到数学原理的统一

---

## 1. 工程改进的数学原理

### 1.1 DDPM → RF：路径直度与传输代价

**DDPM 前向过程**（代码 `diffusiondet_head.py` `q_sample`）：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

路径速度：$v_{DDPM}(t) = \frac{d\sqrt{\bar{\alpha}_t}}{dt} x_0 + \frac{d\sqrt{1-\bar{\alpha}_t}}{dt} \epsilon$

由于 $\sqrt{\bar{\alpha}_t}$, $\sqrt{1-\bar{\alpha}_t}$ 对 $t$ 非线性，**路径弯曲**。

**RF 前向过程**（代码 `rectified_flow.py` `q_sample`）：

$$x_t = (1-t) x_0 + t \cdot x_1, \quad t \in [0,1]$$

路径速度：$v_{RF}(t) = x_1 - x_0 = \text{const}$，**路径为直线**。

**命题 1.1**（直线最小化传输代价）：对任意连接 $x_0$ 和 $x_1$ 的可微路径 $\gamma$，传输代价 $\mathcal{A}[\gamma] = \int_0^1 \|\dot{\gamma}(t)\|^2 dt$ 满足：

$$\mathcal{A}[\gamma] \geq \|x_1 - x_0\|^2 = \mathcal{A}[\gamma_{\text{straight}}]$$

等号当且仅当路径为直线时成立。

**证明**：由 Cauchy-Schwarz 不等式：

$$\|x_1 - x_0\|^2 = \left\|\int_0^1 v(t) dt\right\|^2 \leq \left(\int_0^1 \|v(t)\| dt\right)^2 \leq \int_0^1 \|v(t)\|^2 dt$$

最后一个等号成立当且仅当 $\|v(t)\|$ 为常数。$\square$

**推论**：DDPM 路径传输代价严格大于 RF，因此 RF 需要更少的采样步数。

---

### 1.2 Linear → Shifted Schedule：重要性采样

**代码实现**（`diffusiondet_head.py`）：

```python
t = torch.rand((bs,), device=device))  # t ~ U(0,1)
t = s * t / (1 + (s - 1) * t)          # shifted, s=3.0
```

映射 $g: t \mapsto t' = \frac{st}{1+(s-1)t}$ 将均匀分布变换为密度：

$$p(t') = \frac{s}{(s - (s-1)t')^2}$$

对 $s=3$：$p(0) = 1/3$, $p(1) = 3$，噪声端采样密度是数据端的 9 倍。

**命题 1.2**（检测损失的时间敏感度）：模型预测 $x_0^{pred} = x_t - t \cdot v_\theta$，速度误差对检测损失的影响为：

$$\frac{\partial \mathcal{L}_{det}}{\partial v_\theta} = -t \cdot \nabla_{x_0} \mathcal{L}_{det}$$

损失对速度误差的敏感度与 $t$ 成正比。Shifted schedule 在敏感度最高的区域（$t \approx 1$）分配最多训练资源。

---

### 1.3 Scale-shift → AdaLN-Zero：零初始化与恒等映射

**Scale-shift**：$h' = (1 + \gamma(t)) \cdot \text{LN}(h) + \beta(t)$，$\gamma, \beta$ 随机初始化（非零）。

**AdaLN-Zero**：$h' = h + \alpha(t) \cdot \text{SubLayer}((1+\gamma(t)) \cdot \text{LN}(h) + \beta(t))$

MLP 最后一层零初始化 → $\alpha(0) = 0$ → 初始 $h' = h$（恒等映射）。

**命题 1.3**：AdaLN-Zero 零初始化保证 $v_\theta|_{\text{init}} = 0$，即初始 ODE 路径曲率为零。训练从最直的路径出发，逐步学习必要的弯曲。Scale-shift 的随机初始化导致初始路径有随机曲率。

---

### 1.4 Random → OT Coupling：最优传输

**随机耦合**：每个噪声框随机分配 GT 框。

**OT-nearest**：$\pi_{ij} = \mathbb{1}[j = \arg\min_k \|z_i - b_k^{gt}\|]$

**OT-Sinkhorn**：$\pi^* = \arg\min_\pi \sum_{i,j} \pi_{ij} \|z_i - b_j^{gt}\|^2 - \epsilon H(\pi)$

**命题 1.4**：OT 耦合最小化期望传输代价 $\mathbb{E}[\|z - b_0\|^2] = W_2^2(\mu_z, \mu_{gt})$。由于 RF 目标速度 $v^* = z - b_0$，OT 直接最小化 $\mathbb{E}[\|v^*\|^2]$，使路径更短、速度更均匀、Euler 误差更小。

---

### 1.5 Gaussian → Structured Noise：源分布优化

**纯高斯**：$z \sim \mathcal{N}(0, I_4)$

**网格噪声**：$z = z_{\text{grid}} + \sigma \cdot \epsilon$

**命题 1.5**：GT 框中心在 $[0,1]$ 范围内，网格噪声中心也在 $[0,1]$ 附近，而高斯噪声中心在 $(-\infty, +\infty)$。因此 $W_2^2(\mu_{grid}, \mu_{gt}) < W_2^2(\mathcal{N}(0,I), \mu_{gt})$。

---

## 2. 统一数学框架：条件最优传输

### 2.1 问题定义

- 源分布：$\mu_z$（噪声框，$\mathbb{R}^4$）
- 目标分布：$\mu_{gt} = \frac{1}{K}\sum_{i=1}^K \delta_{b_i^{gt}}$（GT 框，离散）
- 条件信息：$f$（图像特征）
- 传输映射：$T_f: \mu_z \to \mu_{gt}$

目标：找到条件传输映射 $T_f$，使传输代价最小、检测精度最高、ODE 求解高效。

### 2.2 速度场分解定理

**定理 2.1**：RF 速度场可唯一分解为：

$$v_\theta(x_t, t, f) = \underbrace{v_{OT}(x_t, t)}_{\text{OT 传输分量}} + \underbrace{\delta v_\theta(x_t, t, f)}_{\text{特征修正分量}}$$

其中：
- $v_{OT}(x_t, t) = \mathbb{E}_{(z, b_0) \sim \pi^*}[z - b_0 \mid x_t]$：仅依赖 OT 耦合，与图像特征无关
- $\delta v_\theta(x_t, t, f) = \mathbb{E}[z - b_0 \mid x_t, f] - \mathbb{E}[z - b_0 \mid x_t]$：依赖图像特征的修正

**证明**：由条件期望的分解：

$$\mathbb{E}[z - b_0 \mid x_t, f] = \underbrace{\mathbb{E}[z - b_0 \mid x_t]}_{v_{OT}} + \underbrace{(\mathbb{E}[z - b_0 \mid x_t, f] - \mathbb{E}[z - b_0 \mid x_t])}_{\delta v_\theta}$$

$\square$

**五个改进的统一**：

| 改进 | 优化的分量 | 数学效果 |
|---|---|---|
| DDPM → RF | $v_{OT}$ 的路径形式 | 弯曲路径 → 直线，$v_{OT}$ 变为常数 |
| OT Coupling | $v_{OT}$ 的耦合 $\pi^*$ | 最小化 $\|v_{OT}\|^2 = W_2^2$ |
| Structured Noise | $v_{OT}$ 的源分布 $\mu_z$ | 减小 $W_2^2(\mu_z, \mu_{gt})$ |
| Shifted Schedule | 训练资源在 $t$ 上的分配 | 集中优化 $\delta v_\theta$ 影响最大处 |
| AdaLN-Zero | $\delta v_\theta$ 的初始化 | 初始 $\delta v_\theta = 0$，路径零曲率 |

### 2.3 采样误差与路径曲率

**定理 2.2**：$K$ 步 Euler 采样的全局误差满足：

$$\|e_K\| \leq \frac{1}{2K} \int_0^1 \left\|\frac{dv_\theta}{dt}\right\| dt \cdot e^{L/K}$$

其中 $L$ 是 $v_\theta$ 的 Lipschitz 常数。

**推论 2.2**：由于 $v_{OT}$ 是常数（直线路径），$\frac{dv_\theta}{dt} = \frac{d\delta v_\theta}{dt}$，因此：

$$\epsilon_K \leq \frac{C}{K} \cdot \text{Curv}(\delta v_\theta)$$

**采样误差完全由修正分量 $\delta v_\theta$ 的曲率决定。**

---

## 3. 目标检测的特殊性质

### 3.1 低维 OT 的精确可解性

**定理 3.1**：在 $\mathbb{R}^4$ 中，$N$ 个噪声框与 $K$ 个 GT 框的 OT 问题可在 $O(NK \log(NK))$ 时间内精确求解（Hungarian 算法）。由 Brenier 定理，最优传输映射唯一且为凸函数的梯度。

**关键推论**：OT 传输分量 $v_{OT}$ 可以解析计算，不需要神经网络学习。网络只需学习 $\delta v_\theta$。

### 3.2 检测损失的时间敏感度

**定理 3.2**：速度误差对检测损失的影响：

$$\frac{\partial \mathcal{L}_{det}}{\partial (\delta v)} = -t \cdot \nabla_{x_0} \mathcal{L}_{det}(x_0^{pred})$$

损失敏感度与 $t$ 成正比。$t \approx 1$ 处的误差影响是 $t \approx 0$ 处的 $1/t^2$ 倍以上。

### 3.3 离散目标分布的结构

$\mu_{gt}$ 是有限支撑的离散测度，OT 映射是分片常数的。当 $N \gg K$（500 proposals vs ~10 GT），nearest 耦合已近似最优。Sinkhorn 的价值在于均衡分配，避免某些 GT 被忽略。

---

## 4. 新方法：从理论推导的改进

### 4.1 Transport-Refinement Decomposition (TRD)

**核心思想**：将速度场显式分解为解析的 OT 传输分量和学习的特征修正分量。

**参数化**：

$$v_\theta(x_t, t, f) = v_{OT}(x_t, t) + \delta v_\phi(x_t, t, f)$$

**训练**：
1. 计算 OT 耦合 $\pi^*$，解析计算 $v_{OT} = z - b_0^{OT}$
2. 网络只学习残差：$\mathcal{L} = \|\delta v_\phi - (v^* - v_{OT})\|^2$
3. $\delta v_\phi$ 的目标值更小、更平滑，更容易学习

**推理**（自条件化）：
1. 估计 $\hat{v}_{OT} = (x_t - x_0^{prev}) / t$
2. 网络输出修正 $\delta v_\phi(x_t, t, f, x_0^{prev})$
3. 总速度 $v = \hat{v}_{OT} + \delta v_\phi$

**理论保证**：由推论 2.2，$\epsilon_K \leq \frac{C}{K} \cdot \text{Curv}(\delta v_\phi)$。由于 $\delta v_\phi$ 的目标值比 $v_\theta$ 更小更平滑，$\text{Curv}(\delta v_\phi) < \text{Curv}(v_\theta)$，采样误差更小。

### 4.2 Curvature-Aware Training (CAT)

**目标**：直接最小化路径曲率，降低采样误差上界。

**正则化项**：

$$\mathcal{L}_{curv} = \mathbb{E}_t \left[\left\|\frac{\partial v_\theta}{\partial t}\right\|^2\right] \approx \mathbb{E}_t \left[\left\|\frac{v_\theta(x_{t+\Delta t}, t+\Delta t) - v_\theta(x_t, t)}{\Delta t}\right\|^2\right]$$

**与 TRD 协同**：使用 TRD 后，只需正则化 $\delta v_\phi$ 的曲率（$v_{OT}$ 是常数，曲率为零）。

### 4.3 Loss-Sensitive Adaptive Scheduling (LSAS)

**目标**：从检测损失敏感度推导最优时间采样分布。

**理论最优**：

$$p^*(t) \propto t \cdot \sqrt{\mathbb{E}\left[\left\|\nabla_{x_0} \mathcal{L}_{det}\right\|^2\right]}$$

**实现**：可学习的时间分布 $p_\phi(t) = \text{softmax}(\text{MLP}_\phi(t))$，交替优化主模型和时间分布。

---

## 5. 核心命题

> **检测最优传输原理**：在基于扩散的目标检测中，检测框的生成质量由 ODE 路径的直度决定，而路径直度由三个因素控制：
> 1. **耦合质量** $\pi$：决定传输分量 $v_{OT}$ 的大小（$W_2$ 距离）
> 2. **修正曲率** $\text{Curv}(\delta v)$：决定采样误差的上界
> 3. **时间分配** $p(t)$：决定训练资源在损失敏感度上的分配效率
>
> 优化任一因素均可提升检测性能，三者联合优化可达到最优的速度-精度 trade-off。

**级联效果**：

```
显式分解 v = v_OT + δv
    ↓
δv 的目标值更小、更平滑
    ↓
网络更容易学习（更小的函数复杂度）
    ↓
学到的 δv 曲率更小
    ↓
采样误差更小 (ε_K ∝ Curv(δv))
    ↓
更少的步数达到同样的精度
    ↓
推理速度提升
```

---

## 6. 实验验证计划

### Phase 1：理论验证

1. **曲率测量**：在代码中添加 $\text{Curv}(v_\theta)$ 和 $\text{Curv}(\delta v_\theta)$ 的计算，量化 DDPM vs RF 的路径直度差异
2. **信息增益分布**：计算不同 $t$ 下的 $\|\nabla_{x_0} \mathcal{L}_{det}\|$，验证 shifted schedule 的理论依据
3. **ODE 轨迹可视化**：画出 DDPM vs RF 的 2D 轨迹投影

### Phase 2：TRD 实现

4. 修改 `diffusiondet_head.py`，添加 TRD 参数化
5. 修改 `rectified_flow.py`，支持自条件化推理
6. 在染色体数据集上验证

### Phase 3：CAT + LSAS

7. 实现曲率正则化
8. 实现自适应时间调度
9. 联合训练验证

### Phase 4：COCO 实验

10. COCO 上的完整对比实验
11. 与 DiffusionDet 原论文对齐
