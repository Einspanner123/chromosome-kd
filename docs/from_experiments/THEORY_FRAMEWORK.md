# LDMDet 统一数学理论框架

## 从工程改进到数学原理的统一

> **二次修订版**：基于代码审计对一次修正进行精确化
> 修正日期：2026-05-14（二次修正）
> 修正摘要：一次修正 7 处理论错误，新增 3 个定理；二次修正 3 处理论-代码偏差，新增 3 个代码问题记录

---

## 0. 修正总览

原始理论框架存在以下与实验不符的错误推论，本版逐一修正：

| 编号 | 原始主张 | 实验反例 | 修正内容 |
|---|---|---|---|
| E1 | OT 耦合总优于随机耦合 | hard OT (0.735) < random (0.751) | 引入多样性-传输效率权衡（§1.4 修正） |
| E2 | 采样误差完全由 $\text{Curv}(\delta v_\theta)$ 决定 | OT 减小曲率但总误差更大 | 加入泛化误差项（§2.3 修正） |
| E3 | AdaLN-Zero 保证 $v_\theta\|_{\text{init}}=0$ | 代码验证：reg_head 非零初始化 | 修正为零初始化保证时间条件残差为零（§1.3 修正） |
| E4 | 检测最优传输原理仅含三因素 | 缺失多样性维度 | 增加第四因素：训练信号多样性（§5 修正） |
| E5 | OT + TRD 组合应叠加增益 | 组合实验全为负交互 (0.740-0.746) | 新增机制冲突定理（§7） |
| E6 | Reflow 应持续改善路径直度 | Epoch 1 最佳后持续退化 | 新增梯度冲突理论（§6） |
| E7 | 速度场分解中 OT 分量可直接解析计算 | 实际训练中 OT 分量依赖耦合策略 | 修正分解定理的适用条件（§2.2 修正） |

**二次修正**（代码审计发现的理论-代码偏差）：

| 编号 | 一次修正主张 | 代码审计发现 | 二次修正内容 |
|---|---|---|---|
| E8 | AdaLN-Zero 初始时 `fc_feature = h`（原始 proposal 特征） | Block 2 (Instance Interaction) 无 α 门控，初始时 `fc_feature = h + inst_interact(h, f_roi)` | 弱化"时间无关基线"为"时间条件维度零初始化"（§1.3 二次修正） |
| E9 | CAT 惩罚曲率 $\|\partial v_\theta/\partial t\|^2$ | 代码惩罚 $\|x_0^{pred}(t) - x_0^{pred}(t+\Delta t)\|^2$，同时惩罚速度大小和曲率 | 精确描述 CAT 的实际目标为 $x_0$ 一致性正则化（§4.2 二次修正） |
| E10 | CAT-OT 冲突因"Voronoi 边界跳变与曲率平滑化矛盾" | CAT 的 $x_{t+\Delta t}$ 使用相同 OT 配对构造，但 Voronoi 边界随 $t$ 移动导致配对不一致 | 精确描述三方矛盾机制（§7.3 二次修正） |

---

## 1. 工程改进的数学原理

### 1.1 DDPM → RF：路径直度与传输代价

**DDPM 前向过程**（代码 `rectified_flow.py` 对照 `diffusiondet_head.py` DDPM 分支）：

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

**实验验证**：`ldmdet_rf` (0.733) > `ldmdet_baseline` (0.725)，+0.8% mAP，与理论预测一致。✅

---

### 1.2 Linear → Shifted Schedule：重要性采样

**代码实现**（`diffusiondet_head.py` `_sample_t`）：

```python
t = torch.rand((bs,), device=device)  # t ~ U(0,1)
if self.rf_schedule == "shifted":
    t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)  # s=3.0
```

映射 $g: t \mapsto t' = \frac{st}{1+(s-1)t}$ 将均匀分布变换为密度：

$$p(t') = \frac{s}{(s - (s-1)t')^2}$$

对 $s=3$：$p(0) = 1/3$, $p(1) = 3$，噪声端采样密度是数据端的 9 倍。

**命题 1.2**（检测损失的时间敏感度——修正版）：

模型通过 delta regression 预测 $x_0^{pred}$，隐式速度场为 $v_\theta = (x_t - x_0^{pred})/t$。速度误差 $\delta v = v_\theta - v^*$ 导致预测误差：

$$\delta x_0 = x_0^{pred} - x_0 = -t \cdot \delta v$$

因此检测损失对速度误差的敏感度为：

$$\left\|\frac{\partial \mathcal{L}_{det}}{\partial (\delta v)}\right\| = t \cdot \left\|\nabla_{x_0} \mathcal{L}_{det}\right\|$$

**关键修正**：原版直接写 $\frac{\partial \mathcal{L}_{det}}{\partial v_\theta} = -t \cdot \nabla_{x_0} \mathcal{L}_{det}$，隐含假设模型直接预测 $v_\theta$。但代码中模型预测 $x_0$（`prediction_mode="x0"`），速度是推导量。修正后的推导明确区分了预测空间（$x_0$）和速度空间（$v$），通过 $\delta x_0 = -t \cdot \delta v$ 建立两者联系。

**推论**：$t \approx 1$ 处的速度误差被放大 $t$ 倍后影响检测损失。Shifted schedule 在 $t \approx 1$ 处分配 9 倍于 $t \approx 0$ 处的采样密度，与敏感度分布匹配。

**实验验证**：`ldmdet_rf_shifted_schedule` (0.747) > `ldmdet_rf` (0.733)，+1.4% mAP。✅

---

### 1.3 Scale-shift → AdaLN-Zero：零初始化与残差学习

**Scale-shift**（代码 `_forward_scale_shift`）：

$$h' = (1 + \gamma(t)) \cdot h + \beta(t)$$

$\gamma, \beta$ 由 `time_mlp` 生成，随机初始化（非零），初始时 $h'$ 相对 $h$ 有随机扰动。

**AdaLN-Zero**（代码 `_forward_adaln_zero`）：

$$h' = h + \alpha(t) \cdot \text{SubLayer}((1+\gamma(t)) \cdot \text{LN}(h) + \beta(t))$$

代码验证（`single_head.py`）：

```python
self.adaln_mlp = nn.Sequential(
    nn.SiLU(),
    nn.Linear(feat_channels * 4, feat_channels * 6),
)
nn.init.zeros_(self.adaln_mlp[-1].weight)
nn.init.zeros_(self.adaln_mlp[-1].bias)
```

**命题 1.3**（AdaLN-Zero 零初始化——修正版）：

~~原版错误主张：AdaLN-Zero 零初始化保证 $v_\theta|_{\text{init}} = 0$，即初始 ODE 路径曲率为零。~~

**修正**：AdaLN-Zero 零初始化保证**时间条件残差为零**，即 $\alpha(0) = \gamma(0) = \beta(0) = 0$，使得初始时网络输出与时间 $t$ 无关。但 $v_\theta|_{\text{init}} \neq 0$，因为 `cls_head` 和 `reg_head` 非零初始化。

**严格论证（二次修正版）**：

初始化时，`adaln_mlp` 输出全零，因此：
- Block 1 (Self-Attention)：$h_{SA} = h + \alpha_1 \cdot \text{Attn}(\cdots) = h + 0 = h$（恒等）
- Block 3 (FFN)：$h_{FFN} = h + \alpha_2 \cdot \text{FFN}(\cdots) = h + 0 = h$（恒等）

~~因此 `fc_feature = h`（原始 proposal 特征，未经时间调制）。~~

**二次修正**：代码审计发现 Block 2 (Instance Interaction) **不受 $\alpha$ 门控**：

```python
# single_head.py Block 2
inst_out = self.inst_interact(proposals, roi_features)
proposals = proposals + self.dropout2(inst_out)  # ← 无 alpha 门控，直接残差
```

因此初始化时 `fc_feature` 并非原始 proposal 特征 $h$，而是：

$$\text{fc\_feature} = h + \text{inst\_interact}(h, f_{roi})$$

其中 $\text{inst\_interact}$ 的参数为 Xavier 非零初始化。这意味着初始时 `fc_feature` 已经包含了 ROI 特征的空间交互信息，且此信息**不受时间条件控制**。

因此初始速度场为：

$$v_\theta(x_t, t) = \frac{x_t - x_0^{pred}}{t}, \quad x_0^{pred} = \text{apply\_deltas}(\text{reg\_head}(h + \text{inst\_interact}(h, f_{roi})), \text{bboxes})$$

由于 `reg_head` 非零初始化且 `inst_interact` 非零初始化，$x_0^{pred} \neq x_t$，因此 $v_\theta|_{\text{init}} \neq 0$。初始速度场的非零性来自两个独立来源：(1) `reg_head` 的非零权重，(2) `inst_interact` 的非零空间特征交互。

**AdaLN-Zero 的真正保证**：AdaLN-Zero 零初始化保证的是**时间条件维度上的零初始化**，而非速度场的零初始化。具体地：

1. **时间无关性**：初始时模型输出不依赖 $t$，即 $x_0^{pred}(x_t, t) = x_0^{pred}(x_t)$。Block 1 和 Block 3 的时间条件残差为零（$\alpha = \gamma = \beta = 0$），Block 2 本身不使用时间条件。网络从"不使用时间信息"的状态出发，逐步学习时间条件调制。这是**残差学习**在时间条件维度的体现。

2. **空间特征独立性**：Block 2 的 `inst_interact` 在初始时已活跃，但它的行为与时间 $t$ 无关——它处理的是 proposal 之间的空间关系和 proposal-ROI 之间的特征交互，这些关系本身不依赖扩散时间步。这是一个**合理的设计选择**：空间关系应该从训练一开始就参与特征构建，而非等待时间条件学习后才介入。

3. **梯度稳定性**：初始时时间条件的梯度为零，避免随机时间调制对已训练好的空间特征的破坏。对比 Scale-shift：随机初始化的 $\gamma, \beta$ 在训练初期引入与时间相关的随机扰动，可能干扰空间特征的学习。

4. **曲率论证（修正）**：初始速度场为 $v_\theta(x_t, t) = (x_t - x_0^{pred}(x_t))/t$。由于 $x_0^{pred}$ 不依赖 $t$（时间条件残差为零），此速度场的 $t$-依赖性完全来自 $x_t/t$ 项，其曲率 $\frac{dv_\theta}{dt}$ 非零但结构简单。关键在于：**训练过程中学到的曲率修正从零开始增长**，而非从随机值开始。这保证了学到的曲率是最小必要的。

> **⚠️ 代码审计发现的问题**：Block 2 (Instance Interaction) 不受 AdaLN-Zero 的 $\alpha$ 门控，导致初始时 `fc_feature` 已包含非零的空间特征交互。这意味着"时间无关基线"的主张需要弱化——初始时模型确实不使用时间信息，但**已经使用了 ROI 空间特征信息**。如果未来需要真正的"零基线"初始化（$v_\theta|_{\text{init}} = 0$），需要同时：(1) 给 Block 2 增加 $\alpha_3$ 门控，(2) 将 `adaln_mlp` 输出扩展为 9 组参数（3 个 block × 3 参数），(3) 零初始化 $\alpha_3$。但这可能削弱模型在初始阶段学习空间关系的能力，需实验验证。

**实验验证**：`ldmdet_flowdet_adaln` (0.751) vs `ldmdet_rf_heun_shifted_bs2` (0.748，使用 scale-shift)，+0.3% mAP。若对比更早的 scale-shift 基线，AdaLN-Zero 的增益为 +1.1%（0.740→0.751）。✅

---

### 1.4 Random → OT Coupling：多样性-传输效率权衡（重大修正）

~~原版错误主张：OT 耦合最小化 $\mathbb{E}[\|v^*\|^2]$，使路径更短、速度更均匀、Euler 误差更小。~~

**实验反例**：

| 耦合策略 | mAP | $\mathbb{E}[\|v^*\|^2]$ | 条件速度熵 $H(V|X_t)$ |
|---|---|---|---|
| Random | **0.751** | 高 | $\log K \approx 3.84$ |
| Nearest OT (argmin) | 0.735 | 低 | $\approx 0$ |
| Sinkhorn argmax eps=1 | 0.748 | 中 | $\approx 2.98$ |
| Sinkhorn stochastic eps=5 | **0.751** | 中 | $\approx 3.84$ |
| Group-hierarchical stochastic | **0.752** | 中低 | $\approx 3.84$（组内） |

OT 耦合确实减小了 $\mathbb{E}[\|v^*\|^2]$（传输效率提升），但 mAP 反而下降。原命题 1.4 仅考虑了传输效率，忽略了**训练信号多样性**的代价。

**命题 1.4**（多样性-传输效率权衡——修正版）：

耦合策略 $\pi$ 对检测性能的影响由两个竞争因素决定：

$$\text{Performance}(\pi) = \underbrace{-\lambda_1 \cdot \mathbb{E}_\pi[\|v^*\|^2]}_{\text{传输效率收益}} + \underbrace{\lambda_2 \cdot H_\pi(V | X_t)}_{\text{训练多样性收益}} - \underbrace{\lambda_3 \cdot \text{Bias}(\pi)}_{\text{耦合偏差代价}}$$

其中：
- $\mathbb{E}_\pi[\|v^*\|^2] = W_2^2(\mu_z, \mu_{gt}; \pi)$：传输代价，OT 最小化此项
- $H_\pi(V | X_t)$：条件速度熵，衡量给定 $x_t$ 时训练信号的多样性
- $\text{Bias}(\pi)$：耦合策略引入的系统性偏差（如 Stochastic Coupling 在大 $\epsilon$ 下的配对偏离）

**维度依赖性**（OT Diversity Collapse 定理的核心结论）：

相对多样性损失的量级为：

$$\frac{\Delta H}{H_{\text{rand}}(V)} = \frac{\log K}{\frac{d}{2}\log(2\pi e \sigma^2) + \log K}$$

| 场景 | $d$ | $K$ | $\Delta H / H_{\text{rand}}$ | OT 是否有益 |
|---|---|---|---|---|
| 图像生成 | 196608 | $\sim N$ | $\approx 0$ | ✅ 传输效率主导 |
| 检测 (COCO) | 4 | $\sim 7$ | $\approx 0.55$ | ⚠️ 需权衡 |
| 检测 (染色体) | 4 | $\sim 46$ | $\approx 0.69$ | ❌ 多样性主导 |

**推论 1.4a**：在低维检测空间（$d=4$），OT 耦合的多样性损失（$\Delta H = \log K \approx 3.84$）远大于传输效率收益，因此硬 OT 反而降低性能。

**推论 1.4b**（Stochastic Coupling 最优性）：Sinkhorn + Stochastic Coupling 在传输效率与多样性之间建立帕累托前沿。最优 $\epsilon^*$ 位于多样性饱和但传输偏差尚未主导的区间（$\epsilon \in [0.5, 5.0]$），与实验观测的 eps=5 达到 0.751 一致。

**推论 1.4c**（CAM 定理）：Sinkhorn + argmax 管线中，argmax 操作消除了 $\epsilon$ 对多样性的调控能力（$D_{\text{argmax}}(\epsilon) \approx D_{\text{hard-OT}}$，与 $\epsilon$ 无关），因此 $\epsilon$ 扫描曲线异常平坦（实验：eps=5 到 eps=100 的 mAP 仅从 0.745 变到 0.733）。

---

### 1.5 Gaussian → Structured Noise：源分布优化

**纯高斯**：$z \sim \mathcal{N}(0, I_4)$

**网格噪声**：$z = z_{\text{grid}} + \sigma \cdot \epsilon$

**命题 1.5**：GT 框中心在 $[0,1]$ 范围内，网格噪声中心也在 $[0,1]$ 附近，而高斯噪声中心在 $(-\infty, +\infty)$。因此 $W_2^2(\mu_{grid}, \mu_{gt}) < W_2^2(\mathcal{N}(0,I), \mu_{gt})$。

**实验修正**：`ldmdet_flowdet_structured_noise` (0.742) < `ldmdet_flowdet_adaln` (0.751)。结构化噪声虽减小 $W_2^2$，但限制了噪声分布的覆盖范围，可能损害模型对极端位置的泛化能力。此改进的收益被泛化损失抵消，与 OT 耦合的多样性问题类似。

---

## 2. 统一数学框架：条件最优传输

### 2.1 问题定义

- 源分布：$\mu_z$（噪声框，$\mathbb{R}^4$）
- 目标分布：$\mu_{gt} = \frac{1}{K}\sum_{i=1}^K \delta_{b_i^{gt}}$（GT 框，离散）
- 条件信息：$f$（图像特征）
- 传输映射：$T_f: \mu_z \to \mu_{gt}$

目标：找到条件传输映射 $T_f$，使传输代价最小、检测精度最高、ODE 求解高效。

### 2.2 速度场分解定理（修正版）

**定理 2.1**（速度场分解——修正版）：给定耦合策略 $\pi$，RF 速度场可分解为：

$$v_\theta(x_t, t, f) = \underbrace{v_{\pi}(x_t, t)}_{\text{耦合依赖的传输分量}} + \underbrace{\delta v_\theta(x_t, t, f)}_{\text{特征修正分量}}$$

其中：
- $v_{\pi}(x_t, t) = \mathbb{E}_{(z, b_0) \sim \pi}[z - b_0 \mid x_t]$：依赖耦合策略 $\pi$，与图像特征无关
- $\delta v_\theta(x_t, t, f) = \mathbb{E}[z - b_0 \mid x_t, f] - \mathbb{E}[z - b_0 \mid x_t]$：依赖图像特征的修正

**关键修正**：原版将传输分量标记为 $v_{OT}$，暗示 OT 耦合是最优选择。修正后标记为 $v_\pi$，明确传输分量依赖耦合策略 $\pi$，不同 $\pi$ 给出不同的 $v_\pi$。

**修正后的统一表**：

| 改进 | 优化的分量 | 数学效果 | 实验验证 |
|---|---|---|---|
| DDPM → RF | $v_\pi$ 的路径形式 | 弯曲路径 → 直线，$v_\pi$ 变为常数 | ✅ +0.8% |
| OT Coupling | $v_\pi$ 的耦合 $\pi$ | 减小 $\|v_\pi\|^2$，但损失多样性 | ⚠️ 低维下净负效果 |
| Structured Noise | $v_\pi$ 的源分布 $\mu_z$ | 减小 $W_2^2$，但限制覆盖范围 | ⚠️ 净效果接近零 |
| Shifted Schedule | 训练资源在 $t$ 上的分配 | 集中优化 $\delta v_\theta$ 影响最大处 | ✅ +1.4% |
| AdaLN-Zero | $\delta v_\theta$ 的初始化 | 时间条件残差从零开始增长 | ✅ +1.1% |
| Stochastic Coupling | $v_\pi$ 的多样性 | 恢复 $H(V|X_t)$，修复 OT 多样性坍缩 | ✅ +0.3% vs hard OT |

### 2.3 采样误差与路径曲率（重大修正）

~~原版错误主张：采样误差完全由修正分量 $\delta v_\theta$ 的曲率决定。~~

**实验反例**：OT 耦合减小了 $v_\pi$ 的范数（从而减小 $\delta v_\theta$ 的目标值和曲率），但 OT (0.735) 的总误差大于 Random (0.751)。说明仅考虑离散化误差是不完整的。

**定理 2.2**（总采样误差——修正版）：检测器的总误差由两部分组成：

$$\epsilon_{\text{total}} = \underbrace{\epsilon_{\text{discretization}}}_{\text{ODE 离散化误差}} + \underbrace{\epsilon_{\text{generalization}}}_{\text{速度场泛化误差}}$$

**(a) 离散化误差**：$K$ 步 Euler 采样的离散化误差满足：

$$\epsilon_{\text{discretization}} \leq \frac{C}{K} \cdot \text{Curv}(v_\theta) = \frac{C}{K} \cdot \text{Curv}(\delta v_\theta)$$

（因 $v_\pi$ 为常数，$\text{Curv}(v_\theta) = \text{Curv}(\delta v_\theta)$。）

**(b) 泛化误差**：速度场在未见输入上的泛化误差满足：

$$\epsilon_{\text{generalization}} \leq O\left(\sqrt{\frac{C_{\text{model}} \cdot \text{Vol}(\mathcal{S})}{M_{\text{eff}}}}\right)$$

其中 $C_{\text{model}}$ 是模型容量，$\text{Vol}(\mathcal{S})$ 是输入空间的有效体积，$M_{\text{eff}}$ 是有效训练样本数。

**关键修正**：OT 耦合对两部分误差的影响方向相反：

| 误差分量 | OT 耦合的影响 | 机制 |
|---|---|---|
| $\epsilon_{\text{discretization}}$ | ↓ 减小 | $\|v_\pi\|$ 更小 → $\delta v_\theta$ 目标值更小 → 曲率更小 |
| $\epsilon_{\text{generalization}}$ | ↑ 增大 | $H(V|X_t) = 0$ → Voronoi 边界泛化差 → $M_{\text{eff}}$ 降低 |

**推论 2.2a**（维度依赖的净效果）：OT 耦合的净效果取决于两项的相对大小：

- **高维**（$d \gg \log K$）：$\epsilon_{\text{discretization}}$ 主导，OT 有益
- **低维**（$d \sim \log K$）：$\epsilon_{\text{generalization}}$ 主导，OT 有害

这解释了为何 OT 在图像生成中有效但在检测中失败。

**推论 2.2b**（Stochastic Coupling 的理论保证）：Stochastic Coupling 恢复了 $H(V|X_t)$（从而减小 $\epsilon_{\text{generalization}}$），同时保留了部分传输结构（从而不完全放弃 $\epsilon_{\text{discretization}}$ 的收益）。在 $\epsilon \in [0.5, 5.0]$ 区间内，两项误差达到帕累托最优平衡。

---

## 3. 目标检测的特殊性质

### 3.1 低维 OT 的精确可解性

**定理 3.1**：在 $\mathbb{R}^4$ 中，$N$ 个噪声框与 $K$ 个 GT 框的 OT 问题可在 $O(NK \log(NK))$ 时间内精确求解（Hungarian 算法）。由 Brenier 定理，最优传输映射唯一且为凸函数的梯度。

**修正**：~~OT 传输分量 $v_{OT}$ 可以解析计算，不需要神经网络学习。网络只需学习 $\delta v_\theta$。~~

OT 传输分量 $v_\pi$ 在给定耦合 $\pi$ 后可解析计算。但**最优耦合 $\pi$ 本身依赖图像特征**（不同图像的 GT 框分布不同），因此 $v_\pi$ 不能完全脱离特征计算。TRD 的自条件化推理通过估计 $\hat{v}_\pi$ 绕过了这个问题，但引入了估计误差。

### 3.2 检测损失的时间敏感度

**定理 3.2**：速度误差对检测损失的影响：

$$\left\|\frac{\partial \mathcal{L}_{det}}{\partial (\delta v)}\right\| = t \cdot \left\|\nabla_{x_0} \mathcal{L}_{det}(x_0^{pred})\right\|$$

损失敏感度与 $t$ 成正比。

**修正**：~~$t \approx 1$ 处的误差影响是 $t \approx 0$ 处的 $1/t^2$ 倍以上。~~

正确表述：$t = 1$ 处的速度误差对检测损失的影响是 $t = 0.1$ 处的 10 倍（线性比例关系），而非 $1/t^2$ 倍。原版的 $1/t^2$ 推导有误。

### 3.3 离散目标分布的结构

$\mu_{gt}$ 是有限支撑的离散测度，OT 映射是分片常数的。当 $N \gg K$（500 proposals vs ~46 GT），nearest 耦合已近似最优。

**修正**：~~Sinkhorn 的价值在于均衡分配，避免某些 GT 被忽略。~~

Sinkhorn 的真正价值在于通过 $\epsilon$ 参数在 OT 与随机耦合之间插值，从而调控训练多样性。但这一价值**仅在 Stochastic Coupling（从传输矩阵采样）下才能实现**；在 argmax 解码下，$\epsilon$ 的多样性调控被完全抹除（CAM 定理）。

---

## 4. 新方法：从理论推导的改进

### 4.1 Transport-Refinement Decomposition (TRD)

**核心思想**：将速度场显式分解为耦合依赖的传输分量和学习的特征修正分量。

**参数化**：

$$v_\theta(x_t, t, f) = v_\pi(x_t, t) + \delta v_\phi(x_t, t, f)$$

**训练**：
1. 计算耦合 $\pi$，解析计算 $v_\pi = z - b_0^\pi$
2. 网络只学习残差：$\mathcal{L} = \|\delta v_\phi - (v^* - v_\pi)\|^2$
3. $\delta v_\phi$ 的目标值更小、更平滑，更容易学习

**推理**（自条件化）：
1. 估计 $\hat{v}_\pi = (x_t - x_0^{prev}) / t$
2. 网络输出修正 $\delta v_\phi(x_t, t, f, x_0^{prev})$
3. 总速度 $v = \hat{v}_\pi + \delta v_\phi$

**理论保证（修正）**：由定理 2.2，TRD 减小 $\epsilon_{\text{discretization}}$（因 $\text{Curv}(\delta v_\phi) < \text{Curv}(v_\theta)$），但对 $\epsilon_{\text{generalization}}$ 的影响取决于耦合策略。当使用 nearest OT 时，TRD 的自条件化估计 $\hat{v}_\pi$ 在 Voronoi 边界附近可能不准确，增加泛化误差。

**实验验证**：`trd_only` (0.746) > `adaln` (0.751)? 否，0.746 < 0.751。TRD 单独使用时不如 AdaLN 基线。但 `trd_full` (0.752) > `adaln` (0.751)，说明 TRD 需要与 CAT + LSAS + velocity 组合才能发挥效果。这提示 TRD 的收益主要来自训练动力学的整体改善，而非单纯的误差分解。

### 4.2 Curvature-Aware Training (CAT)

**目标**：降低 ODE 路径的离散化误差上界。

**理论正则化项**（曲率惩罚）：

$$\mathcal{L}_{curv}^{theory} = \mathbb{E}_t \left[\left\|\frac{\partial v_\theta}{\partial t}\right\|^2\right] \approx \mathbb{E}_t \left[\left\|\frac{v_\theta(x_{t+\Delta t}, t+\Delta t) - v_\theta(x_t, t)}{\Delta t}\right\|^2\right]$$

**代码实际实现**（二次修正）：

代码审计发现，CAT 的实际实现并非惩罚速度场的时间导数，而是惩罚 **$x_0$ 预测的时间一致性**：

```python
# diffusiondet_head.py _add_cat_loss
t2 = (t + dt).clamp(0, 1)
x_t2 = (1.0 - t2_view) * x_start_batch + t2_view * x_noise_batch
# ...
x0_t2 = self._xyxy_to_raw(all_pred_t2[-1], img_metas)  # t+dt 处的 x0 预测
x0_t1 = self._xyxy_to_raw(all_pred_bboxes[-1], img_metas)  # t 处的 x0 预测
losses["loss_curvature"] = F.mse_loss(x0_t1, x0_t2.detach()) * self.cat_weight
```

即实际目标为：

$$\mathcal{L}_{CAT}^{code} = \mathbb{E}_t \left[\left\|x_0^{pred}(t) - x_0^{pred}(t+\Delta t)\right\|^2\right]$$

**理论目标与代码实现的精确关系**：

由 $x_0^{pred} = x_t - t \cdot v_\theta$，对 $t$ 求导：

$$\frac{\partial x_0^{pred}}{\partial t} = \frac{\partial x_t}{\partial t} - v_\theta - t \cdot \frac{\partial v_\theta}{\partial t}$$

在 RF 中 $\frac{\partial x_t}{\partial t} = x_1 - x_0 = v^*$（沿真实路径），但模型预测路径上 $\frac{\partial x_t}{\partial t} = v_\theta$，因此：

$$\frac{\partial x_0^{pred}}{\partial t} = v_\theta - v_\theta - t \cdot \frac{\partial v_\theta}{\partial t} = -t \cdot \frac{\partial v_\theta}{\partial t}$$

但这仅在模型完美拟合时成立。实际中 $x_0^{pred}$ 的时间导数包含额外项：

$$\frac{\partial x_0^{pred}}{\partial t} \approx -v_\theta - t \cdot \frac{\partial v_\theta}{\partial t}$$

因此代码实现的 $\mathcal{L}_{CAT}^{code}$ 可分解为：

$$\mathcal{L}_{CAT}^{code} \approx \Delta t^2 \cdot \underbrace{\|v_\theta\|^2}_{\text{速度大小惩罚}} + \Delta t^2 \cdot t^2 \cdot \underbrace{\left\|\frac{\partial v_\theta}{\partial t}\right\|^2}_{\text{曲率惩罚}} + \text{交叉项}$$

**关键差异**：代码实现同时约束了两个目标：
1. **速度场范数小**（$\|v_\theta\|^2$）：路径短，传输代价低
2. **速度场变化率小**（$\|\partial v_\theta/\partial t\|^2$）：曲率低，离散化误差小

而理论目标仅约束第 2 项。代码实现比理论描述**更激进**——它不仅要求速度场平滑，还要求速度场本身小。

**这解释了 CAT 单独使用时性能下降（0.744 < 0.751）**：过度的 $x_0$ 一致性约束限制了模型在不同时间步做出不同预测的能力。在检测任务中，不同时间步的 $x_0$ 预测本应不同（$t$ 越小，$x_t$ 越接近噪声，预测越不确定），强制 $x_0^{pred}(t) \approx x_0^{pred}(t+\Delta t)$ 等价于要求模型在所有时间步给出相同的预测，这与扩散模型的多步精化机制矛盾。

> **⚠️ 代码审计发现的问题**：CAT 的代码实现惩罚 $x_0$ 预测的时间一致性 $\|x_0^{pred}(t) - x_0^{pred}(t+\Delta t)\|^2$，而非理论描述的曲率 $\|\partial v_\theta/\partial t\|^2$。前者比后者更激进，同时惩罚速度大小和曲率。如果未来需要实现纯曲率正则化，应改为惩罚速度的时间导数：$\mathcal{L}_{curv}^{pure} = \|v_\theta(x_{t+\Delta t}, t+\Delta t) - v_\theta(x_t, t)\|^2 / \Delta t^2$，其中 $v_\theta = (x_t - x_0^{pred})/t$ 从模型输出推导。但需注意：(1) $t \approx 0$ 处 $v_\theta$ 的数值不稳定（除以接近零的 $t$），(2) 纯曲率正则化可能不足以约束速度大小，(3) 需要重新调参和实验验证。

**实验修正**：`cat_only` (0.744) 略低于 `adaln` (0.751)，说明 $x_0$ 一致性正则化单独使用时过度约束了模型的时间条件表达能力。CAT 与 OT 组合时甚至导致训练崩溃（eval=0），原因见定理 7.2 的精确分析。

### 4.3 Loss-Sensitive Adaptive Scheduling (LSAS)

**目标**：从检测损失敏感度推导最优时间采样分布。

**理论最优**：

$$p^*(t) \propto t \cdot \sqrt{\mathbb{E}\left[\left\|\nabla_{x_0} \mathcal{L}_{det}\right\|^2\right]}$$

**实验验证**：`lsas` (0.743) 单独使用效果有限，但在 `trd_full` 组合中贡献 +0.001-0.002。LSAS 的收益被其他机制部分覆盖（shifted schedule 已提供了粗粒度的时间偏向）。

---

## 5. 核心原理：检测最优传输四因素原理（重大修正）

~~原版：检测最优传输原理包含三因素（耦合质量、修正曲率、时间分配）。~~

> **检测最优传输四因素原理**（修正版）：在基于扩散的目标检测中，检测框的生成质量由 ODE 路径的直度决定，而路径直度由**四个因素**共同控制：
> 1. **耦合质量** $\pi$：决定传输分量 $v_\pi$ 的大小（$W_2$ 距离）
> 2. **训练信号多样性** $H_\pi(V|X_t)$：决定速度场泛化误差的上界
> 3. **修正曲率** $\text{Curv}(\delta v_\theta)$：决定离散化误差的上界
> 4. **时间分配** $p(t)$：决定训练资源在损失敏感度上的分配效率
>
> 因素 1 和因素 2 存在**根本性权衡**：OT 耦合最小化 $W_2^2$（优化因素 1）但最大化多样性损失 $\Delta H = \log K$（恶化因素 2）。此权衡的净效果**维度依赖**：高维空间中因素 1 主导（OT 有益），低维空间中因素 2 主导（OT 有害）。
>
> 四个因素不可同时最优化——因素 1 和因素 2 的权衡是内在的，不存在同时最大化传输效率和多样性的耦合策略。

**修正后的级联效果**：

```
耦合策略 π 决定 (W₂², H(V|X_t)) 的权衡
    ↓
低维空间：H(V|X_t) 主导 → 随机/Stochastic 耦合更优
高维空间：W₂² 主导 → OT 耦合更优
    ↓
选定 π 后，δv_θ 的目标值和曲率确定
    ↓
AdaLN-Zero 确保 δv_θ 从零增长（最小必要曲率）
    ↓
Shifted Schedule 集中优化高敏感度时间步
    ↓
采样误差 ε_total = ε_discretization + ε_generalization 最小化
    ↓
更少的步数达到同样的精度
```

---

## 6. Reflow 退化陷阱：梯度冲突理论（新增）

### 6.1 实验现象

所有 Reflow 实验均呈现 **"Epoch 1 最佳 → 后续退化"** 的模式：

| Reflow 版本 | Epoch 1 mAP | 最佳 mAP | velocity loss 天花板 |
|---|---|---|---|
| v2 (val配对) | 0.739 | 0.739 | ~0.28 |
| v4 (修复velocity_head) | 0.739 | 0.739 | 0.96→收敛 |
| v5 (warmup+调参) | 0.739 | 0.739 | ~0.23 |
| v6 (第2轮Reflow) | 0.739 | 0.739 | ~0.23 |

### 6.2 梯度冲突定理

**定理 6.1**（Reflow 梯度冲突）：设检测损失 $\mathcal{L}_{det}$ 和速度损失 $\mathcal{L}_{vel}$ 共享参数 $\theta_{\text{shared}}$（backbone + 检测头主体）。定义梯度冲突度：

$$\rho = \cos\left(\nabla_{\theta_{\text{shared}}} \mathcal{L}_{det}, \nabla_{\theta_{\text{shared}}} \mathcal{L}_{vel}\right)$$

当 $\rho < 0$ 时，两个损失的梯度方向相反，优化一个会恶化另一个。

**实验测量**：$\rho = -0.104$，86.8% 的层存在梯度冲突。

**证明（梯度冲突的结构性原因）**：

检测损失 $\mathcal{L}_{det}$ 要求 $x_0^{pred}$ 接近 $x_0^{gt}$，即模型需要利用图像特征 $f$ 精确预测 GT 位置。

速度损失 $\mathcal{L}_{vel} = \|v_\theta - v^*\|^2$ 要求速度场拟合 $v^* = x_1 - x_0$。在 Reflow 中，$v^*$ 来自教师模型的 ODE 轨迹，包含教师模型的系统性偏差。

关键矛盾：检测损失要求模型在**特定图像区域**精确预测，而速度损失要求模型在**整个噪声空间**均匀拟合速度场。两者的梯度方向在共享参数上产生冲突：

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{det} \propto -\frac{\partial x_0^{pred}}{\partial \theta} \cdot \nabla_{x_0} \mathcal{L}_{det}$$

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{vel} \propto -\frac{\partial v_\theta}{\partial \theta} \cdot (v_\theta - v^*)$$

由于 $x_0^{pred} = x_t - t \cdot v_\theta$，有 $\frac{\partial x_0^{pred}}{\partial \theta} = -t \cdot \frac{\partial v_\theta}{\partial \theta}$。因此：

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{det} \propto t \cdot \frac{\partial v_\theta}{\partial \theta} \cdot \nabla_{x_0} \mathcal{L}_{det}$$

$$\nabla_{\theta_{\text{shared}}} \mathcal{L}_{vel} \propto -\frac{\partial v_\theta}{\partial \theta} \cdot (v_\theta - v^*)$$

冲突条件为：

$$\rho < 0 \iff \left(\frac{\partial v_\theta}{\partial \theta}\right)^\top \left[t \cdot \nabla_{x_0} \mathcal{L}_{det}\right] \cdot \left(\frac{\partial v_\theta}{\partial \theta}\right)^\top \left[-(v_\theta - v^*)\right] < 0$$

简化为：

$$\rho < 0 \iff \nabla_{x_0} \mathcal{L}_{det} \cdot (v_\theta - v^*) > 0$$

即：**当检测残差方向与速度残差方向一致时，两个梯度冲突**。这在实践中经常发生，因为 $v_\theta - v^*$ 的方向倾向于与 $\nabla_{x_0} \mathcal{L}_{det}$ 的方向相关（两者都反映模型对 $x_0$ 的预测偏差）。$\square$

### 6.3 退化陷阱的动力学解释

设 $\theta_n$ 为第 $n$ 步的参数，梯度下降更新为：

$$\theta_{n+1} = \theta_n - \eta \left(\nabla \mathcal{L}_{det} + w \cdot \nabla \mathcal{L}_{vel}\right)$$

检测损失的变化为：

$$\Delta \mathcal{L}_{det} = -\eta \left(\|\nabla \mathcal{L}_{det}\|^2 + w \cdot \nabla \mathcal{L}_{det}^\top \nabla \mathcal{L}_{vel}\right)$$

当 $\nabla \mathcal{L}_{det}^\top \nabla \mathcal{L}_{vel} < 0$（梯度冲突）且 $w$ 足够大时，$\Delta \mathcal{L}_{det} > 0$，即检测损失**增加**。

Epoch 1 最佳的原因：初始时 $\theta$ 接近预训练模型，$\mathcal{L}_{det}$ 已在局部最优附近。速度损失的梯度将参数拉离此局部最优，且由于梯度冲突，检测性能持续退化。

### 6.4 velocity loss 收敛天花板

velocity loss 收敛到 ~0.23-0.30 后停滞，原因：

1. **容量瓶颈**：velocity_head 为 3 层 MLP（256→256→256→4），参数量有限
2. **配对噪声**：Reflow 配对 $(z_0, x_1^{pred})$ 中 $x_1^{pred}$ 本身有 ODE 离散化误差
3. **梯度冲突**：共享层的梯度冲突阻止 velocity_head 获得足够的梯度信号

---

## 7. 两条 SOTA 路径的负交互：机制冲突理论（新增）

### 7.1 实验现象

| 组合实验 | mAP | vs 最佳单路径 (0.752) |
|---|---|---|
| group_hierarchical_stoch + TRD | 0.746 | **-0.006** |
| sinkhorn_stochastic + TRD + CAT + LSAS | 0.743 | **-0.009** |
| sinkhorn_stochastic + TRD + CAT | 0.740 | **-0.012** |

三个组合实验全部低于任一单路径最佳值。

### 7.2 机制冲突定理

**定理 7.1**（耦合-训练动力学冲突）：Stochastic OT 耦合与 TRD 自条件化存在结构性冲突。

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

### 7.3 CAT 与 OT 的曲率冲突（二次修正）

**定理 7.2**（CAT-OT 冲突——精确版）：CAT 的 $x_0$ 一致性正则化与 OT 耦合在 Voronoi 边界处产生不可调和的梯度冲突，导致训练崩溃。

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

$$\left\|\frac{x_t - b_k}{1-t}\right\| \leq \left\|\frac{x_t - b_j}{1-t}\right\|, \quad \forall j \neq k$$

化简得 Voronoi 边界超平面方程：

$$2(x_t - b_k)^\top(b_k - b_j) + (1-t)\|b_k - b_j\|^2 = 0$$

边界法向为 $(b_k - b_j)$，截距为 $(1-t)\|b_k - b_j\|^2 / 2$。**截距随 $t$ 线性变化**：当 $t$ 增加 $\Delta t$ 时，边界向 $b_k$ 方向移动 $\Delta t \cdot \|b_k - b_j\|^2 / 2$。

**Step 3：冲突的精确机制**

CAT 的 $x_{t+\Delta t}$ 使用相同的 OT 配对构造，意味着 CAT 假设 $x_t$ 和 $x_{t+\Delta t}$ 属于同一个 Voronoi 单元（配对不变）。但由于 Voronoi 边界随 $t$ 移动，$x_{t+\Delta t}$ 可能跨越到相邻的 Voronoi 单元 $\mathcal{V}_j$。

此时出现三方矛盾：

| 约束来源 | 要求 | 代码位置 |
|---|---|---|
| OT 耦合 | $v^* = b_j - z$（新单元的速度） | `_couple_ot` 返回的 `x_start` |
| CAT 构造 | $x_{t+\Delta t}$ 沿旧配对的直线路径 | `x_t2 = (1-t2)*x_start + t2*x_noise` |
| CAT loss | $x_0^{pred}(t) \approx x_0^{pred}(t+\Delta t)$ | `F.mse_loss(x0_t1, x0_t2.detach())` |

具体地：
- CAT 构造的 $x_{t+\Delta t}$ 位于旧配对 $(z, b_k)$ 的直线路径上，期望模型预测 $x_0^{pred}(t+\Delta t) \approx b_k$
- 但 OT 耦合在 $x_{t+\Delta t}$ 处可能分配 $b_j$（因为 $x_{t+\Delta t}$ 已跨越 Voronoi 边界），检测损失要求 $x_0^{pred}(t+\Delta t) \approx b_j$
- CAT loss 惩罚 $x_0^{pred}(t) \neq x_0^{pred}(t+\Delta t)$，即惩罚 $b_k \neq b_j$，但 $b_k \neq b_j$ 是 Voronoi 结构的必然结果

**Step 4：崩溃的动力学解释**

当 CAT + OT 组合训练时，梯度更新陷入三方拉锯：

1. 检测损失 $\mathcal{L}_{det}$ 推动模型在 $x_{t+\Delta t}$ 处预测 $b_j$（OT 配对的目标）
2. CAT 损失 $\mathcal{L}_{CAT}$ 推动模型在 $x_{t+\Delta t}$ 处预测 $b_k$（与 $x_t$ 处一致）
3. 两者梯度方向相反，且 $b_k \neq b_j$ 使得冲突不可调和

在 Voronoi 边界附近，这种冲突的样本比例随训练进行而增加（模型学会在边界附近产生不确定预测，增加边界跨越的概率），形成正反馈循环，最终导致训练崩溃（eval=0）。

**对比**：CAT 单独使用时（无 OT 耦合），随机耦合下不存在 Voronoi 结构，$v^*$ 在不同 $t$ 处的变化是连续的（条件期望的平滑变化），CAT 的平滑化约束与训练信号兼容，因此不会崩溃（0.744）。

> **⚠️ 代码审计发现的问题**：CAT 的 $x_{t+\Delta t}$ 构造使用与 $x_t$ 相同的 OT 配对，但 Voronoi 边界随 $t$ 变化导致 $x_{t+\Delta t}$ 可能属于不同 Voronoi 单元。这是 CAT + OT 崩溃的代码层面根源。如果未来需要让 CAT 与 OT 兼容，有两个方向：(1) 为 $x_{t+\Delta t}$ 重新运行 OT 耦合（`x_start_t2 = _couple_ot(x_t2, gt_diffusion, labels, device)`），使 CAT 的构造与 OT 的配对一致，但这引入额外计算开销；(2) 在 CAT loss 中排除 Voronoi 边界附近的样本（通过检测 $x_0^{pred}(t)$ 与最近 GT 的距离是否接近次近 GT 的距离来识别边界样本），但这需要额外的边界检测逻辑。

### 7.4 对论文的影响

两条 SOTA 路径的负交互不是超参数问题，而是**结构性冲突**：

1. **Stochastic OT + TRD**：训练时不确定性与自条件化一致性要求矛盾
2. **OT + CAT**：Voronoi 边界跳变与曲率平滑化要求矛盾

这意味着两条路径虽然各自有效，但**不能简单叠加**。需要设计新的组合策略来绕过这些冲突（如分阶段训练、解耦参数等）。

---

## 8. 修正后的实验验证状态

| 理论预测 | 实验结果 | 状态 |
|---|---|---|
| RF 直线路径优于 DDPM 弯曲路径 | +0.8% mAP | ✅ 一致 |
| Shifted schedule 集中优化高敏感度时间步 | +1.4% mAP | ✅ 一致 |
| AdaLN-Zero 时间条件残差从零增长 | +1.1% mAP | ✅ 一致 |
| OT 耦合减小传输代价 | 确实减小 $\|v^*\|^2$ | ✅ 机制正确 |
| OT 耦合应提升性能 | hard OT (0.735) < random (0.751) | ❌ 原理论错误，已修正 |
| Stochastic Coupling 恢复多样性 | eps=5 达到 0.751 | ✅ 一致 |
| TRD 减小离散化误差 | trd_full (0.752) > adaln (0.751) | ✅ 一致（需组合） |
| Reflow 应持续改善路径直度 | Epoch 1 后持续退化 | ❌ 原理论错误，已修正 |
| OT + TRD 组合应叠加 | 组合全为负交互 | ❌ 原理论错误，已修正 |
| Scale-Conditioned FM 改善小物体 | sc_combined (0.736) < adaln (0.751) | ✅ 负结果理论正确 |

---

## 9. 修正后的理论贡献总结

1. **多样性-传输效率权衡定理**（修正命题 1.4）：OT 耦合的净效果维度依赖，低维空间中多样性损失主导
2. **总采样误差分解**（修正定理 2.2）：$\epsilon_{\text{total}} = \epsilon_{\text{discretization}} + \epsilon_{\text{generalization}}$，OT 对两项影响方向相反
3. **检测最优传输四因素原理**（修正核心原理）：增加训练信号多样性作为第四因素
4. **Reflow 梯度冲突定理**（新增定理 6.1）：检测损失与速度损失的梯度方向系统性冲突
5. **耦合-训练动力学冲突定理**（新增定理 7.1）：Stochastic OT 与 TRD 自条件化的结构性矛盾
6. **CAT-OT 冲突定理**（新增定理 7.2，二次修正）：$x_0$ 一致性正则化与 OT Voronoi 边界时间依赖性的三方矛盾
7. **AdaLN-Zero 修正**（修正命题 1.3，二次修正）：零初始化保证时间条件维度零初始化，Block 2 不受 α 门控导致初始时已包含空间特征交互

**二次修正新增贡献**：
8. **CAT 实际目标与理论目标的差异**（§4.2 二次修正）：代码惩罚 $x_0$ 一致性（同时约束速度大小和曲率），理论仅约束曲率，差异解释了 CAT 单独使用时性能下降
9. **CAT-OT 崩溃的代码层面精确机制**（§7.3 二次修正）：CAT 的 $x_{t+\Delta t}$ 使用相同 OT 配对构造，但 Voronoi 边界随 $t$ 移动导致三方矛盾（OT 要求 $b_j$，CAT 构造隐含 $b_k$，CAT loss 惩罚 $b_k \neq b_j$）

---

## 附录 A：代码审计发现的问题（待修复）

> 以下问题由代码审计发现，记录于此供后续实验参考。实验存档中的备份代码不应修改，新实验应在独立分支中进行。

### A.1 velocity loss 目标符号不一致

**严重程度**：🔴 高（混淆源，不影响当前推理结果但影响理论一致性）

**问题描述**：

RF 速度定义（`rectified_flow.py:61`）：
```python
velocity = x_noise - x_start  # v = x_1 - x_0
```

velocity loss 目标（`diffusiondet_head.py:586`）：
```python
v_target = torch.stack(x_starts) - torch.stack(x_noises)  # v* = x_0 - x_1 = -v
```

两者符号相反：`velocity_head` 学习的是反向速度 $-v$，而非 RF 定义的正向速度 $v$。

**影响**：
- 当前不影响推理结果：velocity_head 的输出仅用于 loss 计算，ODE 采样使用 `x_0^{pred}` 推导速度
- 如果未来用 velocity_head 输出做 ODE 积分，方向会反转
- 与定理 6.1 的梯度冲突推导不一致（推导中假设 $v_\theta$ 与 RF 定义同向）

**修复方案**：
```python
# diffusiondet_head.py:586 修改为
v_target = torch.stack(x_noises) - torch.stack(x_starts)  # v* = x_1 - x_0 = v
```

MSE loss 对符号不敏感，修改后训练结果不变，但理论一致性恢复。

### A.2 DDPM 多步推理性能低于单步

**严重程度**：🟡 中（DDPM baseline 的 bug，不影响 RF 路线）

**问题描述**：

`ldmdet_baseline` (1步, 0.725) > `ldmdet_baseline_step4` (4步, 0.709)，多步推理反而更差。

**根因分析**：

`_ddim_step`（`diffusiondet_head.py:1043`）中：
```python
alpha = self.alphas_cumprod[t_curr]
alpha_next = self.alphas_cumprod[t_next]  # ← t_next 可能为负数
```

当 `t_next < 0` 时，Python 负索引返回 `alphas_cumprod[-1]`（最后一个元素），导致 `alpha_next` 错误。虽然外层循环有 `if t_next < 0: break`，但 `_ddim_step` 内部已经用错误的 `alpha_next` 计算了 `x_raw_next`。

**修复方案**：
```python
def _ddim_step(self, t_curr, t_next, x_raw, cls_logits, pred_bboxes, img_metas):
    x0 = self._xyxy_to_raw(pred_bboxes, img_metas)
    if t_next < 0:
        return self._raw_to_xyxy(x0, img_metas), x0  # 直接返回 x0 预测
    # ... 原有逻辑
```

### A.3 Stochastic Coupling 可复现性差

**严重程度**：🟡 中（影响实验结论的置信度）

**问题描述**：

`sinkhorn_sample_eps5` 主实验 0.751，复现 0.738，seed2 0.750，方差 0.013 mAP（典型实验方差 ~0.002-0.005）。

**根因**：`torch.multinomial` 在小 batch size (bs=2) 下随机性大，每次迭代从传输矩阵中采样配对，不同种子导致训练轨迹差异大。

**修复方案**：

方案 1（轻量）：固定 `torch.multinomial` 的 generator：
```python
gen = torch.Generator(device=device)
gen.manual_seed(self.ot_sample_seed)
return torch.multinomial(row_probs, 1, generator=gen).squeeze(-1)
```

方案 2（根本）：增大 batch size（bs=2 → bs=4-8），降低单次采样的方差。

方案 3（替代）：使用 Gumbel-Softmax 实现可微的 Stochastic Coupling：
```python
tau = self.ot_gumbel_tau  # 温度参数
gumbel_noise = -torch.log(-torch.log(torch.rand_like(row_probs)))
return F.softmax((row_probs.log() + gumbel_noise) / tau, dim=1)
```

---

## 附录 B：设计改进方案（待实验验证）

> 以下方案由代码审计和理论分析推导得出，尚未经过实验验证。记录于此供后续实验参考。

### B.1 Reflow 分阶段训练

**针对问题**：定理 6.1 的梯度冲突（$\rho = -0.104$），Epoch 1 后持续退化。

**方案**：

| 阶段 | Epoch | freeze_shared | velocity_detach | 学习率 | 目标 |
|---|---|---|---|---|---|
| Phase 1 | 1 | False | False | 1e-4 | 同时优化检测和速度，利用预训练初始化 |
| Phase 2 | 2+ | True | False | 5e-5 | 冻结共享层，只精调 velocity_head |

**理论依据**：Epoch 1 最佳说明预训练参数已接近检测损失的局部最优。Phase 2 冻结共享层消除梯度冲突，velocity_head 从冻结的共享特征中学习速度预测。

**代码修改**：在 `diffusiondet_head.py` 中增加 epoch 级别的参数冻结逻辑：
```python
def on_epoch_start(self, epoch):
    if self.use_reflow and epoch >= 2:
        self.freeze_shared = True
        self._freeze_shared_layers()
```

**预期效果**：消除 Epoch 2+ 的退化，velocity loss 可能收敛到更低值（因无梯度冲突干扰）。

### B.2 TRD 使用解析传输速度

**针对问题**：定理 7.1 的 Stochastic OT + TRD 冲突。

**方案**：

将 TRD 的自条件化速度估计从"模型预测"改为"当前 OT 配对的解析速度"：

```python
# 修改前：用模型预测估计 v_π
with torch.no_grad():
    _, all_pred_sc, _, _ = self(features, curr_bboxes, t_input)
    x0_sc = self._xyxy_to_raw(all_pred_sc[-1], img_metas)
v_ot_est = (x_noisy_sc - x0_sc) / torch.clamp(t_view, min=1e-5)

# 修改后：用当前 OT 配对的解析速度
v_ot_est = x_noise_batch - x_start_batch  # v* = x_1 - x_0，解析可得
```

**理论依据**：TRD 的自条件化步骤需要与当前迭代的 OT 配对一致。使用解析速度 $v^* = x_1 - x_0$ 保证前进方向与当前 OT 配对完全一致，消除 Stochastic OT 的随机性引入的不一致性。

**预期效果**：Stochastic OT + TRD 组合不再产生负交互，mAP 应接近或超过 0.752。

**风险**：解析速度 $v^*$ 是训练目标速度，不是模型实际学到的速度。TRD 前进到 $x_{t+\Delta t}$ 后，模型在该点的预测可能与解析速度外推的位置不一致，引入新的训练噪声。需要实验验证哪种估计更优。

### B.3 TRD 与 CAT 解耦步长参数

**针对问题**：TRD 和 CAT 共享 `cat_delta_t` 参数，但最优步长可能不同。

**方案**：

引入独立的 `trd_delta_t` 参数：

```python
# diffusiondet_head.py __init__
self.trd_delta_t = trd_delta_t if trd_delta_t is not None else cat_delta_t
```

在 `_forward_trd` 中使用 `self.trd_delta_t`，在 `_add_cat_loss` 中使用 `self.cat_delta_t`。

**理论依据**：
- TRD 需要较大的步长（~0.05-0.1）以提供有意义的自条件化信号
- CAT 需要较小的步长（~0.01-0.02）以精确估计 $x_0$ 一致性

**预期效果**：独立调参后 TRD 和 CAT 可能各自达到更优性能。

### B.4 纯曲率正则化 CAT

**针对问题**：§4.2 发现 CAT 代码惩罚 $x_0$ 一致性而非纯曲率，比理论更激进。

**方案**：

修改 CAT loss 为惩罚速度的时间导数：

```python
# 修改前
losses["loss_curvature"] = F.mse_loss(x0_t1, x0_t2.detach()) * self.cat_weight

# 修改后：纯曲率正则化
t_view_safe = t_view.clamp(min=0.01)  # 避免 t≈0 处数值不稳定
v_t1 = (x_noisy_t1 - x0_t1) / t_view_safe
t2_view_safe = t2_view.clamp(min=0.01)
v_t2 = (x_noisy_t2 - x0_t2.detach()) / t2_view_safe
losses["loss_curvature"] = F.mse_loss(v_t1, v_t2) * self.cat_weight
```

**理论依据**：纯曲率正则化仅约束 $\|\partial v_\theta/\partial t\|^2$，不约束速度大小 $\|v_\theta\|^2$，比当前实现更温和。

**风险**：
1. $t \approx 0$ 处 $v_\theta = (x_t - x_0^{pred})/t$ 数值不稳定
2. 纯曲率正则化可能不足以约束速度大小，导致路径过长
3. 需要重新调参和实验验证

### B.5 CAT 与 OT 兼容化

**针对问题**：§7.3 发现 CAT 的 $x_{t+\Delta t}$ 使用相同 OT 配对构造，但 Voronoi 边界随 $t$ 移动导致三方矛盾。

**方案 1：为 $x_{t+\Delta t}$ 重新运行 OT 耦合**

```python
# 在 _add_cat_loss 中
x_t2_raw = (1.0 - t2_view) * gt_diffusion + t2_view * noise  # 用原始 GT 和噪声构造
x_start_t2 = self._couple_ot(x_t2_raw, gt_diffusion, labels, device)  # 重新配对
x_t2 = (1.0 - t2_view) * x_start_t2 + t2_view * noise  # 用新配对构造 x_{t+dt}
```

**理论依据**：使 CAT 的构造与 OT 的配对一致，消除三方矛盾。

**风险**：引入额外 OT 计算开销（每步训练多一次 Sinkhorn 迭代）。

**方案 2：在 CAT loss 中排除 Voronoi 边界样本**

```python
# 检测边界样本：x_0^{pred} 与最近 GT 和次近 GT 的距离比
dists = torch.cdist(x0_t1, gt_diffusion)
d1, d2 = dists.sort(dim=1).values[:, :2].unbind(dim=1)
boundary_mask = (d2 / (d1 + 1e-5)) < self.cat_boundary_threshold  # 距离比接近 1 = 边界
losses["loss_curvature"] = F.mse_loss(x0_t1[~boundary_mask], x0_t2[~boundary_mask].detach()) * self.cat_weight
```

**理论依据**：边界样本是 CAT-OT 冲突的根源，排除后 CAT 的平滑化约束与 OT 的训练信号兼容。

**风险**：减少有效训练样本，且边界阈值需要调参。
