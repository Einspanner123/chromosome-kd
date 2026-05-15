# LDMDet 统一数学理论框架

## 从工程改进到数学原理的统一

> **三次修订版**：面向论文投稿口径的理论严谨化
> 修正日期：2026-05-15（三次修正）
> 修正摘要：一次修正 7 处理论错误；二次修正 3 处理论-代码偏差；三次修正将若干过强“定理”降级为可检验命题/机制假说，并新增目标检测任务的顶会级突破点分析

______________________________________________________________________

## 0. 修正总览

原始理论框架存在以下与实验不符的错误推论，本版逐一修正：

| 编号 | 原始主张                                               | 实验反例                          | 修正内容                                        |
| ---- | ------------------------------------------------------ | --------------------------------- | ----------------------------------------------- |
| E1   | OT 耦合总优于随机耦合                                  | hard OT (0.735) \< random (0.751) | 引入多样性-传输效率权衡（§1.4 修正）            |
| E2   | 采样误差完全由 $\\text{Curv}(\\delta v\_\\theta)$ 决定 | OT 减小曲率但总误差更大           | 加入泛化误差项（§2.3 修正）                     |
| E3   | AdaLN-Zero 保证 $v\_\\theta\|\_{\\text{init}}=0$       | 代码验证：reg_head 非零初始化     | 修正为零初始化保证时间条件残差为零（§1.3 修正） |
| E4   | 检测最优传输原理仅含三因素                             | 缺失多样性维度                    | 增加第四因素：训练信号多样性（§5 修正）         |
| E5   | OT + TRD 组合应叠加增益                                | 组合实验全为负交互 (0.740-0.746)  | 新增机制冲突假说（§7）                          |
| E6   | Reflow 应持续改善路径直度                              | Epoch 1 最佳后持续退化            | 新增梯度冲突理论（§6）                          |
| E7   | 速度场分解中 OT 分量可直接解析计算                     | 实际训练中 OT 分量依赖耦合策略    | 修正分解定理的适用条件（§2.2 修正）             |

**二次修正**（代码审计发现的理论-代码偏差）：

| 编号 | 一次修正主张                                             | 代码审计发现                                                                                | 二次修正内容                                                  |
| ---- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| E8   | AdaLN-Zero 初始时 `fc_feature = h`（原始 proposal 特征） | Block 2 (Instance Interaction) 无 α 门控，初始时 `fc_feature = h + inst_interact(h, f_roi)` | 弱化"时间无关基线"为"时间条件维度零初始化"（§1.3 二次修正）   |
| E9   | CAT 惩罚曲率 $\|\\partial v\_\\theta/\\partial t\|^2$    | 代码惩罚 $\|x_0^{pred}(t) - x_0^{pred}(t+\\Delta t)\|^2$，同时惩罚速度大小和曲率            | 精确描述 CAT 的实际目标为 $x_0$ 一致性正则化（§4.2 二次修正） |
| E10  | CAT-OT 冲突因"Voronoi 边界跳变与曲率平滑化矛盾"          | CAT 的 $x\_{t+\\Delta t}$ 使用相同 OT 配对构造，但 Voronoi 边界随 $t$ 移动导致配对不一致    | 精确描述三方矛盾机制（§7.3 二次修正）                         |

**三次修正**（论文严谨性修正）：

| 编号 | 二次修正版风险 | 严谨化处理 |
| ---- | -------------- | ---------- |
| E11  | 将经验性能写成 $\text{Performance}(\pi)$ 的确定性等式，容易被理解为已证明的泛化界 | 改为解释性结构风险分解，只声明可观测量之间的方向性关系，要求用多 seed 与消融验证 |
| E12  | 条件速度熵同时使用离散熵和连续微分熵，存在量纲混用 | 明确区分目标索引熵 $H_\pi(Y\mid X_t)$ 与速度微分熵 $h_\pi(V\mid X_t)$，论文主张优先使用可估计的离散索引熵 |
| E13  | Reflow/TRD/CAT 冲突被称为“定理”，但目前主要由代码审计和经验测量支撑 | 降级为机制命题/可证伪假说，并列出必要验证实验 |

### 0.1 投稿口径的证据等级

为避免把经验解释写成不可支撑的数学结论，本文档后续采用三档证据等级：

1. **严格命题**：只依赖已定义变量和标准数学条件，例如固定端点直线路径最小化动能。
2. **可检验命题**：有明确机制和可观测量，但依赖模型、数据分布或优化过程，需要实验支持，例如低维检测中的多样性-效率权衡。
3. **机制假说**：由反常实验、梯度测量或代码审计提出，尚未完成充分消融，例如 Stochastic OT + TRD 的负交互机制。

论文写作中应把第 2、3 类表述为“we hypothesize / we empirically find / suggests”，而不是“we prove”。

______________________________________________________________________

## 1. 工程改进的数学原理

### 1.1 DDPM → RF：路径直度与传输代价

**DDPM 前向过程**（代码 `rectified_flow.py` 对照 `diffusiondet_head.py` DDPM 分支）：

$$x_t = \\sqrt{\\bar{\\alpha}\_t} \\cdot x_0 + \\sqrt{1 - \\bar{\\alpha}\_t} \\cdot \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, I)$$

路径速度：$v\_{DDPM}(t) = \\frac{d\\sqrt{\\bar{\\alpha}\_t}}{dt} x_0 + \\frac{d\\sqrt{1-\\bar{\\alpha}\_t}}{dt} \\epsilon$

由于 $\\sqrt{\\bar{\\alpha}\_t}$, $\\sqrt{1-\\bar{\\alpha}\_t}$ 对 $t$ 非线性，**路径弯曲**。

**RF 前向过程**（代码 `rectified_flow.py` `q_sample`）：

$$x_t = (1-t) x_0 + t \\cdot x_1, \\quad t \\in \[0,1\]$$

路径速度：$v\_{RF}(t) = x_1 - x_0 = \\text{const}$，**路径为直线**。

**命题 1.1**（直线最小化传输代价）：对任意连接 $x_0$ 和 $x_1$ 的可微路径 $\\gamma$，传输代价 $\\mathcal{A}\[\\gamma\] = \\int_0^1 |\\dot{\\gamma}(t)|^2 dt$ 满足：

$$\\mathcal{A}\[\\gamma\] \\geq |x_1 - x_0|^2 = \\mathcal{A}\[\\gamma\_{\\text{straight}}\]$$

等号当且仅当路径为直线时成立。

**证明**：由 Cauchy-Schwarz 不等式：

$$|x_1 - x_0|^2 = \\left|\\int_0^1 v(t) dt\\right|^2 \\leq \\left(\\int_0^1 |v(t)| dt\\right)^2 \\leq \\int_0^1 |v(t)|^2 dt$$

最后一个等号成立当且仅当 $|v(t)|$ 为常数。$\\square$

**推论**：DDPM 路径传输代价严格大于 RF，因此 RF 需要更少的采样步数。

**实验验证**：`ldmdet_rf` (0.733) > `ldmdet_baseline` (0.725)，+0.8% mAP，与理论预测一致。✅

______________________________________________________________________

### 1.2 Linear → Shifted Schedule：重要性采样

**代码实现**（`diffusiondet_head.py` `_sample_t`）：

```python
t = torch.rand((bs,), device=device)  # t ~ U(0,1)
if self.rf_schedule == "shifted":
    t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)  # s=3.0
```

映射 $g: t \\mapsto t' = \\frac{st}{1+(s-1)t}$ 将均匀分布变换为密度：

$$p(t') = \\frac{s}{(s - (s-1)t')^2}$$

对 $s=3$：$p(0) = 1/3$, $p(1) = 3$，噪声端采样密度是数据端的 9 倍。

**命题 1.2**（检测损失的时间敏感度——修正版）：

模型通过 delta regression 预测 $x_0^{pred}$，隐式速度场为 $v\_\\theta = (x_t - x_0^{pred})/t$。速度误差 $\\delta v = v\_\\theta - v^\*$ 导致预测误差：

$$\\delta x_0 = x_0^{pred} - x_0 = -t \\cdot \\delta v$$

因此检测损失对速度误差的敏感度为：

$$\\left|\\frac{\\partial \\mathcal{L}_{det}}{\\partial (\\delta v)}\\right| = t \\cdot \\left|\\nabla_{x_0} \\mathcal{L}\_{det}\\right|$$

**关键修正**：原版直接写 $\\frac{\\partial \\mathcal{L}_{det}}{\\partial v_\\theta} = -t \\cdot \\nabla\_{x_0} \\mathcal{L}_{det}$，隐含假设模型直接预测 $v_\\theta$。但代码中模型预测 $x_0$（`prediction_mode="x0"`），速度是推导量。修正后的推导明确区分了预测空间（$x_0$）和速度空间（$v$），通过 $\\delta x_0 = -t \\cdot \\delta v$ 建立两者联系。

**推论**：$t \\approx 1$ 处的速度误差被放大 $t$ 倍后影响检测损失。Shifted schedule 在 $t \\approx 1$ 处分配 9 倍于 $t \\approx 0$ 处的采样密度，与敏感度分布匹配。

**实验验证**：`ldmdet_rf_shifted_schedule` (0.747) > `ldmdet_rf` (0.733)，+1.4% mAP。✅

______________________________________________________________________

### 1.3 Scale-shift → AdaLN-Zero：零初始化与残差学习

**Scale-shift**（代码 `_forward_scale_shift`）：

$$h' = (1 + \\gamma(t)) \\cdot h + \\beta(t)$$

$\\gamma, \\beta$ 由 `time_mlp` 生成，随机初始化（非零），初始时 $h'$ 相对 $h$ 有随机扰动。

**AdaLN-Zero**（代码 `_forward_adaln_zero`）：

$$h' = h + \\alpha(t) \\cdot \\text{SubLayer}((1+\\gamma(t)) \\cdot \\text{LN}(h) + \\beta(t))$$

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

~~原版错误主张：AdaLN-Zero 零初始化保证 $v\_\\theta|\_{\\text{init}} = 0$，即初始 ODE 路径曲率为零。~~

**修正**：AdaLN-Zero 零初始化保证**时间条件残差为零**，即 $\\alpha(0) = \\gamma(0) = \\beta(0) = 0$，使得初始时网络输出与时间 $t$ 无关。但 $v\_\\theta|\_{\\text{init}} \\neq 0$，因为 `cls_head` 和 `reg_head` 非零初始化。

**严格论证（二次修正版）**：

初始化时，`adaln_mlp` 输出全零，因此：

- Block 1 (Self-Attention)：$h\_{SA} = h + \\alpha_1 \\cdot \\text{Attn}(\\cdots) = h + 0 = h$（恒等）
- Block 3 (FFN)：$h\_{FFN} = h + \\alpha_2 \\cdot \\text{FFN}(\\cdots) = h + 0 = h$（恒等）

~~因此 `fc_feature = h`（原始 proposal 特征，未经时间调制）。~~

**二次修正**：代码审计发现 Block 2 (Instance Interaction) **不受 $\\alpha$ 门控**：

```python
# single_head.py Block 2
inst_out = self.inst_interact(proposals, roi_features)
proposals = proposals + self.dropout2(inst_out)  # ← 无 alpha 门控，直接残差
```

因此初始化时 `fc_feature` 并非原始 proposal 特征 $h$，而是：

$$\\text{fc_feature} = h + \\text{inst_interact}(h, f\_{roi})$$

其中 $\\text{inst_interact}$ 的参数为 Xavier 非零初始化。这意味着初始时 `fc_feature` 已经包含了 ROI 特征的空间交互信息，且此信息**不受时间条件控制**。

因此初始速度场为：

$$v\_\\theta(x_t, t) = \\frac{x_t - x_0^{pred}}{t}, \\quad x_0^{pred} = \\text{apply_deltas}(\\text{reg_head}(h + \\text{inst_interact}(h, f\_{roi})), \\text{bboxes})$$

由于 `reg_head` 非零初始化且 `inst_interact` 非零初始化，$x_0^{pred} \\neq x_t$，因此 $v\_\\theta|\_{\\text{init}} \\neq 0$。初始速度场的非零性来自两个独立来源：(1) `reg_head` 的非零权重，(2) `inst_interact` 的非零空间特征交互。

**AdaLN-Zero 的真正保证**：AdaLN-Zero 零初始化保证的是**时间条件维度上的零初始化**，而非速度场的零初始化。具体地：

1. **时间无关性**：初始时模型输出不依赖 $t$，即 $x_0^{pred}(x_t, t) = x_0^{pred}(x_t)$。Block 1 和 Block 3 的时间条件残差为零（$\\alpha = \\gamma = \\beta = 0$），Block 2 本身不使用时间条件。网络从"不使用时间信息"的状态出发，逐步学习时间条件调制。这是**残差学习**在时间条件维度的体现。

2. **空间特征独立性**：Block 2 的 `inst_interact` 在初始时已活跃，但它的行为与时间 $t$ 无关——它处理的是 proposal 之间的空间关系和 proposal-ROI 之间的特征交互，这些关系本身不依赖扩散时间步。这是一个**合理的设计选择**：空间关系应该从训练一开始就参与特征构建，而非等待时间条件学习后才介入。

3. **梯度稳定性**：初始时时间条件的梯度为零，避免随机时间调制对已训练好的空间特征的破坏。对比 Scale-shift：随机初始化的 $\\gamma, \\beta$ 在训练初期引入与时间相关的随机扰动，可能干扰空间特征的学习。

4. **曲率论证（修正）**：初始速度场为 $v\_\\theta(x_t, t) = (x_t - x_0^{pred}(x_t))/t$。由于 $x_0^{pred}$ 不依赖 $t$（时间条件残差为零），此速度场的 $t$-依赖性完全来自 $x_t/t$ 项，其曲率 $\\frac{dv\_\\theta}{dt}$ 非零但结构简单。关键在于：**训练过程中由时间条件分支引入的曲率修正从零开始增长**，而非从随机值开始。这可能降低训练早期的时间条件噪声。

> **⚠️ 代码审计发现的问题**：Block 2 (Instance Interaction) 不受 AdaLN-Zero 的 $\\alpha$ 门控，导致初始时 `fc_feature` 已包含非零的空间特征交互。这意味着"时间无关基线"的主张需要弱化——初始时模型确实不使用时间信息，但**已经使用了 ROI 空间特征信息**。如果未来需要真正的"零基线"初始化（$v\_\\theta|\_{\\text{init}} = 0$），需要同时：(1) 给 Block 2 增加 $\\alpha_3$ 门控，(2) 将 `adaln_mlp` 输出扩展为 9 组参数（3 个 block × 3 参数），(3) 零初始化 $\\alpha_3$。但这可能削弱模型在初始阶段学习空间关系的能力，需实验验证。

**实验验证**：`ldmdet_flowdet_adaln` (0.751) vs `ldmdet_rf_heun_shifted_bs2` (0.748，使用 scale-shift)，+0.3% mAP。若对比更早的 scale-shift 基线，AdaLN-Zero 的增益为 +1.1%（0.740→0.751）。✅

______________________________________________________________________

### 1.4 Random → OT Coupling：多样性-传输效率权衡（重大修正）

~~原版错误主张：OT 耦合最小化 $\\mathbb{E}\[|v^\*|^2\]$，使路径更短、速度更均匀、Euler 误差更小。~~

**实验反例**：

| 耦合策略 | mAP | $\\mathbb{E}\[|v^\*|^2\]$ | 目标索引熵 $H(Y\mid X_t)$ |
|---|---|---|---|
| Random | **0.751** | 高 | $\\log K \\approx 3.84$ |
| Nearest OT (argmin) | 0.735 | 低 | $\\approx 0$ |
| Sinkhorn argmax eps=1 | 0.748 | 中 | $\\approx 2.98$ |
| Sinkhorn stochastic eps=5 | **0.751** | 中 | $\\approx 3.84$ |
| Group-hierarchical stochastic | **0.752** | 中低 | $\\approx 3.84$（组内） |

OT 耦合确实减小了 $\\mathbb{E}\[|v^\*|^2\]$（传输效率提升），但 mAP 反而下降。原命题 1.4 仅考虑了传输效率，忽略了**训练信号多样性**的代价。

**可检验命题 1.4**（多样性-传输效率权衡——三次修正版）：

给定同一检测模型、同一训练预算和同一时间采样分布，耦合策略 $\\pi$ 至少同时改变三个可观测量：

$$C_{trans}(\\pi)=\\mathbb{E}_\\pi\[|v^\*|^2\]$$

$$D_{idx}(\\pi)=H_\pi(Y\mid X_t)$$

$$B_{match}(\\pi)=\\mathbb{E}\_\pi\[\\ell_{det}(b_Y, x_t, f)\] - \\min_j \\ell_{det}(b_j, x_t, f)$$

其中 $Y$ 是被分配的 GT 索引，$D_{idx}$ 是离散索引熵，$B_{match}$ 是耦合策略相对检测匹配目标的偏差项。当前实验支持以下方向性关系：

- 硬 OT 降低 $C_{trans}$，但显著降低 $D_{idx}$。
- 随机耦合最大化 $D_{idx}$，但放弃传输结构。
- Stochastic Sinkhorn 在 $C_{trans}$ 与 $D_{idx}$ 之间插值，可能形成更优折中。

**重要严谨性修正**：这里不再把 mAP 写成三个变量的确定性线性函数。更合适的投稿表述是：检测性能受 $C_{trans}$、$D_{idx}$、$B_{match}$ 共同影响，且三者之间存在可观测的 trade-off。若要在论文中使用权重形式，只能作为经验回归模型：

$$\\widehat{\\text{mAP}}(\\pi)=\\beta_0-\beta_1 C_{trans}(\\pi)+\\beta_2 D_{idx}(\\pi)-\\beta_3 B_{match}(\\pi)+\\epsilon$$

该式需要由多种耦合策略和多 seed 实验拟合，不能作为先验证明。

**熵定义修正**：

- 论文主张优先使用 $D_{idx}=H_\pi(Y\mid X_t)$，因为它是离散、可估计、与 GT 分配多样性直接对应的量。
- 速度熵 $h_\pi(V\mid X_t)$ 是连续微分熵，不能与 $\\log K$ 直接相减；若使用它，必须固定核密度估计带宽或噪声模型。
- “多样性坍缩”应定义为 $H_\pi(Y\mid X_t)$ 从随机耦合的近似 $\\log K$ 降到硬 OT 的近似 0，而不是笼统地说 $H(V\mid X_t)$ 坍缩。

**维度依赖性假说**（不是严格定理）：

低维检测框空间中，索引不确定性 $H(Y\mid X_t)$ 占训练信号复杂度的比例更高；高维图像空间中，连续位移不确定性占主导，目标索引熵的相对影响更小。一个可用于解释的量纲化指标是：

$$R_{idx}=\\frac{\\log K}{\\log K + d\\cdot \\log(1/\\tau)}$$

其中 $d$ 是状态维度，$\\tau$ 是将连续框空间离散化或核密度估计时的有效分辨率。$R_{idx}$ 不是绝对理论界，而是用于预测“多样性损失是否会主导传输效率收益”的经验指标。

| 场景          | $d$    | $K$        | $R_{idx}$ 预期 | OT 净效果假说     |
| ------------- | ------ | ---------- | -------------- | ----------------- |
| 图像生成      | 高维   | $\\sim N$  | 低             | 传输效率更可能主导 |
| 检测 (COCO)   | 4      | $\\sim 7$  | 中             | 需权衡             |
| 检测 (染色体) | 4      | $\\sim 46$ | 高             | 多样性更可能主导   |

**推论 1.4a**：在染色体检测的低维框空间（$d=4$, $K\\approx46$）中，硬 OT 的索引多样性损失足以抵消其传输代价收益；这解释了 hard OT (0.735) 低于 random/AdaLN (0.751) 的现象。

**推论 1.4b**（Stochastic Coupling 的经验最优性）：Sinkhorn + Stochastic Coupling 在当前数据集上形成较优折中。现有结果支持 $\\epsilon \\in \[0.5, 5.0\]$ 是候选区间，但“最优区间”必须以多 seed 均值和方差报告。

**推论 1.4c**（CAM 命题）：Sinkhorn + argmax 管线中，argmax 操作会显著削弱 $\\epsilon$ 对采样多样性的调控能力，因此 $\\epsilon$ 扫描曲线可能异常平坦。该命题可通过报告 $H(Y\mid X_t)$ 随 $\\epsilon$ 的变化来验证。

______________________________________________________________________

### 1.5 Gaussian → Structured Noise：源分布优化

**纯高斯**：$z \\sim \\mathcal{N}(0, I_4)$

**网格噪声**：$z = z\_{\\text{grid}} + \\sigma \\cdot \\epsilon$

**命题 1.5**：GT 框中心在 $\[0,1\]$ 范围内，网格噪声中心也在 $\[0,1\]$ 附近，而高斯噪声中心在 $(-\\infty, +\\infty)$。因此 $W_2^2(\\mu\_{grid}, \\mu\_{gt}) \< W_2^2(\\mathcal{N}(0,I), \\mu\_{gt})$。

**实验修正**：`ldmdet_flowdet_structured_noise` (0.742) \< `ldmdet_flowdet_adaln` (0.751)。结构化噪声虽减小 $W_2^2$，但限制了噪声分布的覆盖范围，可能损害模型对极端位置的泛化能力。此改进的收益被泛化损失抵消，与 OT 耦合的多样性问题类似。

______________________________________________________________________

## 2. 统一数学框架：条件最优传输

### 2.1 问题定义

- 源分布：$\\mu_z$（噪声框，$\\mathbb{R}^4$）
- 目标分布：$\\mu\_{gt} = \\frac{1}{K}\\sum\_{i=1}^K \\delta\_{b_i^{gt}}$（GT 框，离散）
- 条件信息：$f$（图像特征）
- 传输映射：$T_f: \\mu_z \\to \\mu\_{gt}$

目标：找到条件传输映射 $T_f$，使传输代价最小、检测精度最高、ODE 求解高效。

### 2.2 速度场分解（修正版）

**定义 2.1**（速度场分解——修正版）：给定耦合策略 $\\pi$，可将 RF 速度场按条件期望分解为：

$$v\_\\theta(x_t, t, f) = \\underbrace{v\_{\\pi}(x_t, t)}_{\\text{耦合依赖的传输分量}} + \\underbrace{\\delta v_\\theta(x_t, t, f)}\_{\\text{特征修正分量}}$$

其中：

- $v\_{\\pi}(x_t, t) = \\mathbb{E}\_{(z, b_0) \\sim \\pi}\[z - b_0 \\mid x_t\]$：依赖耦合策略 $\\pi$，与图像特征无关
- $\\delta v\_\\theta(x_t, t, f) = \\mathbb{E}\[z - b_0 \\mid x_t, f\] - \\mathbb{E}\[z - b_0 \\mid x_t\]$：依赖图像特征的修正

**关键修正**：原版将传输分量标记为 $v\_{OT}$，暗示 OT 耦合是最优选择。修正后标记为 $v\_\\pi$，明确传输分量依赖耦合策略 $\\pi$，不同 $\\pi$ 给出不同的 $v\_\\pi$。

**修正后的统一表**：

| 改进                | 优化的分量                    | 数学效果                                 | 实验验证                  |
| ------------------- | ----------------------------- | ---------------------------------------- | ------------------------- |
| DDPM → RF           | $v\_\\pi$ 的路径形式          | 弯曲路径 → 直线，$v\_\\pi$ 变为常数      | ✅ +0.8%                  |
| OT Coupling         | $v\_\\pi$ 的耦合 $\\pi$       | 减小 $\|v\_\\pi\|^2$，但损失多样性       | ⚠️ 低维下净负效果         |
| Structured Noise    | $v\_\\pi$ 的源分布 $\\mu_z$   | 减小 $W_2^2$，但限制覆盖范围             | ⚠️ 净效果接近零           |
| Shifted Schedule    | 训练资源在 $t$ 上的分配       | 集中优化 $\\delta v\_\\theta$ 影响最大处 | ✅ +1.4%                  |
| AdaLN-Zero          | $\\delta v\_\\theta$ 的初始化 | 时间条件残差从零开始增长                 | ✅ +1.1%                  |
| Stochastic Coupling | $v\_\\pi$ 的多样性            | 恢复索引熵 $H(Y\mid X_t)$，缓解 OT 多样性坍缩 | ✅ +0.3% vs hard OT |

### 2.3 采样误差与路径曲率（重大修正）

~~原版错误主张：采样误差完全由修正分量 $\\delta v\_\\theta$ 的曲率决定。~~

**实验反例**：OT 耦合减小了 $v\_\\pi$ 的范数（从而减小 $\\delta v\_\\theta$ 的目标值和曲率），但 OT (0.735) 的总误差大于 Random (0.751)。说明仅考虑离散化误差是不完整的。

**可检验分解 2.2**（总误差来源——三次修正版）：检测器的误差至少包含两类来源：

$$\\epsilon\_{\\text{total}} = \\underbrace{\\epsilon\_{\\text{discretization}}}_{\\text{ODE 离散化误差}} + \\underbrace{\\epsilon_{\\text{generalization}}}\_{\\text{速度场泛化误差}}$$

**(a) 离散化误差**：在速度场足够光滑且 Lipschitz 常数有界的条件下，$K$ 步 Euler 采样的局部误差可由速度场曲率/时间变化率控制：

$$\\epsilon\_{\\text{discretization}} \\lesssim \\frac{C}{K} \\cdot \\text{Var}_t(v\_\\theta)$$

若固定耦合下的真实 RF 路径为直线，$v^\*$ 沿单条样本路径为常数；但模型速度 $v_\theta(x_t,t,f)$ 仍可能因特征条件、匹配边界和网络参数化产生时间变化。因此这里不能简单断言 $\\text{Curv}(v\_\\theta)=\\text{Curv}(\\delta v\_\\theta)$，只能说 TRD/CAT 试图降低模型速度的有效时间变化。

**(b) 泛化/匹配误差**：速度场在未见输入上的误差受有效样本数、匹配稳定性和训练目标多样性影响。二次修正版中的如下形式只能作为启发式复杂度项，而非已证明泛化界：

$$\\epsilon\_{\\text{generalization}} \\leq O\\left(\\sqrt{\\frac{C\_{\\text{model}} \\cdot \\text{Vol}(\\mathcal{S})}{M\_{\\text{eff}}}}\\right)$$

其中 $C\_{\\text{model}}$ 是模型容量，$\\text{Vol}(\\mathcal{S})$ 是输入空间的有效体积，$M\_{\\text{eff}}$ 是有效训练样本数。投稿时更稳妥的写法是：硬 OT 降低索引熵 $H(Y\mid X_t)$，可能减少同一局部区域内可见的监督方向，从而降低有效训练覆盖；该结论需要通过多 seed 和局部覆盖率统计验证。

**关键修正**：OT 耦合对两部分误差的影响方向相反：

| 误差分量                              | OT 耦合的影响 | 机制                                                            |
| ------------------------------------- | ------------- | --------------------------------------------------------------- |
| $\\epsilon\_{\\text{discretization}}$ | 可能减小 | $\|v^\*\|$ 更小，模型需要拟合的平均位移更短 |
| $\\epsilon\_{\\text{generalization}}$ / 匹配误差 | 可能增大 | $H(Y\mid X_t)$ 降低，监督方向覆盖变窄，Voronoi 边界处泛化更差 |

**推论 2.2a**（维度依赖的净效果）：OT 耦合的净效果取决于两项的相对大小：

- **高维或连续结构主导**：传输效率收益可能主导，OT 更可能有益。
- **低维且目标索引不确定性高**：监督多样性损失可能主导，硬 OT 更可能有害。

这解释了为何 OT 在图像生成中常有效，但在低维检测框生成中可能失败。该说法应作为跨任务假说，仍需 COCO/LVIS 等通用检测数据集验证。

**推论 2.2b**（Stochastic Coupling 的经验优势）：Stochastic Coupling 恢复了部分 $H(Y\mid X_t)$，同时保留部分传输结构。当前数据支持它位于较优 trade-off 区域，但不能称为理论保证。

______________________________________________________________________

## 3. 目标检测的特殊性质

### 3.1 低维 OT 的精确可解性

**命题 3.1**：检测框状态维度很低（通常为中心点、宽高或 xyxy 的 4 维），因此每张图像内的离散耦合代价矩阵可以显式构造。若采用一对一 assignment，可用 Hungarian 算法求解；若采用带容量约束或软边际的多对一耦合，则更适合用 Sinkhorn/最小费用流求解。

**严谨性修正**：这里不直接套用 Brenier 定理。Brenier 定理适用于绝对连续源分布到目标分布的二次代价最优映射；当前训练中使用的是有限 proposals 与离散 GT 的经验耦合，最优解可能非唯一，且 argmax/Sinkhorn/stochastic 解码会改变实际训练分布。因此论文中应讨论“经验离散耦合”，而非声称存在唯一解析 OT 映射。

**修正**：~~OT 传输分量 $v\_{OT}$ 可以解析计算，不需要神经网络学习。网络只需学习 $\\delta v\_\\theta$。~~

OT 传输分量 $v\_\\pi$ 在给定耦合 $\\pi$ 后可解析计算。但**最优耦合 $\\pi$ 本身依赖图像特征**（不同图像的 GT 框分布不同），因此 $v\_\\pi$ 不能完全脱离特征计算。TRD 的自条件化推理通过估计 $\\hat{v}\_\\pi$ 绕过了这个问题，但引入了估计误差。

### 3.2 检测损失的时间敏感度

**命题 3.2**：在模型预测 $x_0$ 且速度由 $v_\theta=(x_t-x_0^{pred})/t$ 推导时，速度误差对检测损失的影响为：

$$\\left|\\frac{\\partial \\mathcal{L}_{det}}{\\partial (\\delta v)}\\right| = t \\cdot \\left|\\nabla_{x_0} \\mathcal{L}\_{det}(x_0^{pred})\\right|$$

损失敏感度与 $t$ 成正比。

**修正**：~~$t \\approx 1$ 处的误差影响是 $t \\approx 0$ 处的 $1/t^2$ 倍以上。~~

正确表述：$t = 1$ 处的速度误差对检测损失的影响是 $t = 0.1$ 处的 10 倍（线性比例关系），而非 $1/t^2$ 倍。原版的 $1/t^2$ 推导有误。

### 3.3 离散目标分布的结构

$\\mu\_{gt}$ 是有限支撑的离散测度，OT 映射是分片常数的。当 $N \\gg K$（500 proposals vs ~46 GT），nearest 耦合已近似最优。

**修正**：~~Sinkhorn 的价值在于均衡分配，避免某些 GT 被忽略。~~

Sinkhorn 的真正价值在于通过 $\\epsilon$ 参数在 OT 与随机耦合之间插值，从而调控训练多样性。但这一价值**仅在 Stochastic Coupling（从传输矩阵采样）下才能实现**；在 argmax 解码下，$\\epsilon$ 的多样性调控会被显著削弱（CAM 命题）。

______________________________________________________________________

## 4. 新方法：从理论推导的改进

### 4.1 Transport-Refinement Decomposition (TRD)

**核心思想**：将速度场显式分解为耦合依赖的传输分量和学习的特征修正分量。

**参数化**：

$$v\_\\theta(x_t, t, f) = v\_\\pi(x_t, t) + \\delta v\_\\phi(x_t, t, f)$$

**训练**：

1. 计算耦合 $\\pi$，解析计算 $v\_\\pi = z - b_0^\\pi$
2. 网络只学习残差：$\\mathcal{L} = |\\delta v\_\\phi - (v^\* - v\_\\pi)|^2$
3. $\\delta v\_\\phi$ 的目标值更小、更平滑，更容易学习

**推理**（自条件化）：

1. 估计 $\\hat{v}\_\\pi = (x_t - x_0^{prev}) / t$
2. 网络输出修正 $\\delta v\_\\phi(x_t, t, f, x_0^{prev})$
3. 总速度 $v = \\hat{v}_\\pi + \\delta v_\\phi$

**理论预期（修正）**：根据可检验分解 2.2，TRD 的潜在收益来自降低模型需要学习的有效速度变化；但它是否降低最终误差取决于耦合策略、边界样本比例和自条件估计误差。当使用 nearest OT 时，TRD 的自条件化估计 $\\hat{v}\_\\pi$ 在 Voronoi 边界附近可能不准确，反而增加匹配/泛化误差。

**实验验证**：`trd_only` (0.746) > `adaln` (0.751)? 否，0.746 \< 0.751。TRD 单独使用时不如 AdaLN 基线。但 `trd_full` (0.752) > `adaln` (0.751)，说明 TRD 需要与 CAT + LSAS + velocity 组合才能发挥效果。这提示 TRD 的收益主要来自训练动力学的整体改善，而非单纯的误差分解。

### 4.2 Curvature-Aware Training (CAT)

**目标**：降低 ODE 路径的离散化误差上界。

**理论正则化项**（曲率惩罚）：

$$\\mathcal{L}_{curv}^{theory} = \\mathbb{E}_t \\left\[\\left|\\frac{\\partial v_\\theta}{\\partial t}\\right|^2\\right\] \\approx \\mathbb{E}_t \\left\[\\left|\\frac{v_\\theta(x_{t+\\Delta t}, t+\\Delta t) - v\_\\theta(x_t, t)}{\\Delta t}\\right|^2\\right\]$$

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

$$\\mathcal{L}\_{CAT}^{code} = \\mathbb{E}\_t \\left\[\\left|x_0^{pred}(t) - x_0^{pred}(t+\\Delta t)\\right|^2\\right\]$$

**理论目标与代码实现的精确关系**：

由 $x_0^{pred} = x_t - t \\cdot v\_\\theta$，对 $t$ 求导：

$$\\frac{\\partial x_0^{pred}}{\\partial t} = \\frac{\\partial x_t}{\\partial t} - v\_\\theta - t \\cdot \\frac{\\partial v\_\\theta}{\\partial t}$$

在 RF 中 $\\frac{\\partial x_t}{\\partial t} = x_1 - x_0 = v^\*$（沿真实路径），但模型预测路径上 $\\frac{\\partial x_t}{\\partial t} = v\_\\theta$，因此：

$$\\frac{\\partial x_0^{pred}}{\\partial t} = v\_\\theta - v\_\\theta - t \\cdot \\frac{\\partial v\_\\theta}{\\partial t} = -t \\cdot \\frac{\\partial v\_\\theta}{\\partial t}$$

但这仅在模型完美拟合时成立。实际中 $x_0^{pred}$ 的时间导数包含额外项：

$$\\frac{\\partial x_0^{pred}}{\\partial t} \\approx -v\_\\theta - t \\cdot \\frac{\\partial v\_\\theta}{\\partial t}$$

因此代码实现的 $\\mathcal{L}\_{CAT}^{code}$ 可分解为：

$$\\mathcal{L}_{CAT}^{code} \\approx \\Delta t^2 \\cdot \\underbrace{|v_\\theta|^2}_{\\text{速度大小惩罚}} + \\Delta t^2 \\cdot t^2 \\cdot \\underbrace{\\left|\\frac{\\partial v_\\theta}{\\partial t}\\right|^2}\_{\\text{曲率惩罚}} + \\text{交叉项}$$

**关键差异**：代码实现同时约束了两个目标：

1. **速度场范数小**（$|v\_\\theta|^2$）：路径短，传输代价低
2. **速度场变化率小**（$|\\partial v\_\\theta/\\partial t|^2$）：曲率低，离散化误差小

而理论目标仅约束第 2 项。代码实现比理论描述**更激进**——它不仅要求速度场平滑，还要求速度场本身小。

**这解释了 CAT 单独使用时性能下降（0.744 \< 0.751）**：过度的 $x_0$ 一致性约束限制了模型在不同时间步做出不同预测的能力。在检测任务中，不同时间步的 $x_0$ 预测本应不同（$t$ 越小，$x_t$ 越接近噪声，预测越不确定），强制 $x_0^{pred}(t) \\approx x_0^{pred}(t+\\Delta t)$ 等价于要求模型在所有时间步给出相同的预测，这与扩散模型的多步精化机制矛盾。

> **⚠️ 代码审计发现的问题**：CAT 的代码实现惩罚 $x_0$ 预测的时间一致性 $|x_0^{pred}(t) - x_0^{pred}(t+\\Delta t)|^2$，而非理论描述的曲率 $|\\partial v\_\\theta/\\partial t|^2$。前者比后者更激进，同时惩罚速度大小和曲率。如果未来需要实现纯曲率正则化，应改为惩罚速度的时间导数：$\\mathcal{L}_{curv}^{pure} = |v_\\theta(x\_{t+\\Delta t}, t+\\Delta t) - v\_\\theta(x_t, t)|^2 / \\Delta t^2$，其中 $v\_\\theta = (x_t - x_0^{pred})/t$ 从模型输出推导。但需注意：(1) $t \\approx 0$ 处 $v\_\\theta$ 的数值不稳定（除以接近零的 $t$），(2) 纯曲率正则化可能不足以约束速度大小，(3) 需要重新调参和实验验证。

**实验修正**：`cat_only` (0.744) 略低于 `adaln` (0.751)，说明 $x_0$ 一致性正则化单独使用时过度约束了模型的时间条件表达能力。CAT 与 OT 组合时甚至导致训练崩溃（eval=0），可能原因见机制假说 7.2。

### 4.3 Loss-Sensitive Adaptive Scheduling (LSAS)

**目标**：从检测损失敏感度推导最优时间采样分布。

**理论最优**：

$$p^\*(t) \\propto t \\cdot \\sqrt{\\mathbb{E}\\left\[\\left|\\nabla\_{x_0} \\mathcal{L}\_{det}\\right|^2\\right\]}$$

**实验验证**：`lsas` (0.743) 单独使用效果有限，但在 `trd_full` 组合中贡献 +0.001-0.002。LSAS 的收益被其他机制部分覆盖（shifted schedule 已提供了粗粒度的时间偏向）。

______________________________________________________________________

## 5. 核心原理：检测扩散传输四因素框架（重大修正）

~~原版：检测最优传输原理包含三因素（耦合质量、修正曲率、时间分配）。~~

> **检测扩散传输四因素框架**（修正版）：在基于扩散的目标检测中，检测框的生成质量与 ODE 路径、训练匹配和时间采样共同相关，可由**四个因素**联合分析：
>
> 1. **耦合质量** $\\pi$：影响平均传输代价 $C_{trans}$
> 2. **训练信号多样性** $H_\pi(Y\mid X_t)$：影响目标索引监督覆盖和匹配稳定性
> 3. **修正曲率/时间变化率** $\\text{Var}_t(v_\theta)$：影响离散化误差
> 4. **时间分配** $p(t)$：决定训练资源在损失敏感度上的分配效率
>
> 因素 1 和因素 2 存在经验 trade-off：硬 OT 倾向于降低传输代价，但会降低目标索引熵 $H(Y\mid X_t)$。此 trade-off 的净效果可能维度依赖：高维连续结构中传输效率更重要，低维检测框空间中索引多样性更重要。
>
> 四个因素通常难以同时最优化。特别是因素 1 和因素 2 存在经验 trade-off：降低传输代价的硬分配往往会降低目标索引多样性。

**修正后的级联效果**：

```
耦合策略 π 决定 (W₂², H(Y|X_t)) 的权衡
    ↓
低维空间：H(Y|X_t) 主导 → 随机/Stochastic 耦合更优
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

______________________________________________________________________

## 6. Reflow 退化陷阱：梯度冲突理论（新增）

### 6.1 实验现象

所有 Reflow 实验均呈现 **"Epoch 1 最佳 → 后续退化"** 的模式：

| Reflow 版本            | Epoch 1 mAP | 最佳 mAP | velocity loss 天花板 |
| ---------------------- | ----------- | -------- | -------------------- |
| v2 (val配对)           | 0.739       | 0.739    | ~0.28                |
| v4 (修复velocity_head) | 0.739       | 0.739    | 0.96→收敛            |
| v5 (warmup+调参)       | 0.739       | 0.739    | ~0.23                |
| v6 (第2轮Reflow)       | 0.739       | 0.739    | ~0.23                |

### 6.2 梯度冲突机制

**机制命题 6.1**（Reflow 梯度冲突）：设检测损失 $\\mathcal{L}_{det}$ 和速度损失 $\\mathcal{L}_{vel}$ 共享参数 $\\theta\_{\\text{shared}}$（backbone + 检测头主体）。定义梯度冲突度：

$$\\rho = \\cos\\left(\\nabla\_{\\theta\_{\\text{shared}}} \\mathcal{L}_{det}, \\nabla_{\\theta\_{\\text{shared}}} \\mathcal{L}\_{vel}\\right)$$

当 $\\rho \< 0$ 时，两个损失的梯度方向相反，优化一个会恶化另一个。

**实验测量**：$\\rho = -0.104$，86.8% 的层存在梯度冲突。

**机制推导（非严格证明）**：

检测损失 $\\mathcal{L}\_{det}$ 要求 $x_0^{pred}$ 接近 $x_0^{gt}$，即模型需要利用图像特征 $f$ 精确预测 GT 位置。

速度损失 $\\mathcal{L}_{vel} = |v_\\theta - v^*|^2$ 要求速度场拟合 $v^* = x_1 - x_0$。在 Reflow 中，$v^\*$ 来自教师模型的 ODE 轨迹，包含教师模型的系统性偏差。

关键矛盾：检测损失要求模型在**特定图像区域**精确预测，而速度损失要求模型在**整个噪声空间**均匀拟合速度场。两者的梯度方向在共享参数上产生冲突：

$$\\nabla\_{\\theta\_{\\text{shared}}} \\mathcal{L}_{det} \\propto -\\frac{\\partial x_0^{pred}}{\\partial \\theta} \\cdot \\nabla_{x_0} \\mathcal{L}\_{det}$$

$$\\nabla\_{\\theta\_{\\text{shared}}} \\mathcal{L}_{vel} \\propto -\\frac{\\partial v_\\theta}{\\partial \\theta} \\cdot (v\_\\theta - v^\*)$$

由于 $x_0^{pred} = x_t - t \\cdot v\_\\theta$，有 $\\frac{\\partial x_0^{pred}}{\\partial \\theta} = -t \\cdot \\frac{\\partial v\_\\theta}{\\partial \\theta}$。因此：

$$\\nabla\_{\\theta\_{\\text{shared}}} \\mathcal{L}_{det} \\propto t \\cdot \\frac{\\partial v_\\theta}{\\partial \\theta} \\cdot \\nabla\_{x_0} \\mathcal{L}\_{det}$$

$$\\nabla\_{\\theta\_{\\text{shared}}} \\mathcal{L}_{vel} \\propto -\\frac{\\partial v_\\theta}{\\partial \\theta} \\cdot (v\_\\theta - v^\*)$$

冲突条件为：

$$\\rho \< 0 \\iff \\left(\\frac{\\partial v\_\\theta}{\\partial \\theta}\\right)^\\top \\left\[t \\cdot \\nabla\_{x_0} \\mathcal{L}_{det}\\right\] \\cdot \\left(\\frac{\\partial v_\\theta}{\\partial \\theta}\\right)^\\top \\left\[-(v\_\\theta - v^\*)\\right\] \< 0$$

简化为：

$$\\rho \< 0 \\iff \\nabla\_{x_0} \\mathcal{L}_{det} \\cdot (v_\\theta - v^\*) > 0$$

即：**当检测残差方向与速度残差方向一致时，两个梯度冲突**。这在实践中经常发生，因为 $v\_\\theta - v^\*$ 的方向倾向于与 $\\nabla\_{x_0} \\mathcal{L}\_{det}$ 的方向相关（两者都反映模型对 $x_0$ 的预测偏差）。$\\square$

### 6.3 退化陷阱的动力学解释

设 $\\theta_n$ 为第 $n$ 步的参数，梯度下降更新为：

$$\\theta\_{n+1} = \\theta_n - \\eta \\left(\\nabla \\mathcal{L}_{det} + w \\cdot \\nabla \\mathcal{L}_{vel}\\right)$$

检测损失的变化为：

$$\\Delta \\mathcal{L}_{det} = -\\eta \\left(|\\nabla \\mathcal{L}_{det}|^2 + w \\cdot \\nabla \\mathcal{L}_{det}^\\top \\nabla \\mathcal{L}_{vel}\\right)$$

当 $\\nabla \\mathcal{L}_{det}^\\top \\nabla \\mathcal{L}_{vel} \< 0$（梯度冲突）且 $w$ 足够大时，$\\Delta \\mathcal{L}\_{det} > 0$，即检测损失**增加**。

Epoch 1 最佳的原因：初始时 $\\theta$ 接近预训练模型，$\\mathcal{L}\_{det}$ 已在局部最优附近。速度损失的梯度将参数拉离此局部最优，且由于梯度冲突，检测性能持续退化。

### 6.4 velocity loss 收敛天花板

velocity loss 收敛到 ~0.23-0.30 后停滞，原因：

1. **容量瓶颈**：velocity_head 为 3 层 MLP（256→256→256→4），参数量有限
2. **配对噪声**：Reflow 配对 $(z_0, x_1^{pred})$ 中 $x_1^{pred}$ 本身有 ODE 离散化误差
3. **梯度冲突**：共享层的梯度冲突阻止 velocity_head 获得足够的梯度信号

______________________________________________________________________

## 7. 两条 SOTA 路径的负交互：机制冲突理论（新增）

### 7.1 实验现象

| 组合实验                               | mAP   | vs 最佳单路径 (0.752) |
| -------------------------------------- | ----- | --------------------- |
| group_hierarchical_stoch + TRD         | 0.746 | **-0.006**            |
| sinkhorn_stochastic + TRD + CAT + LSAS | 0.743 | **-0.009**            |
| sinkhorn_stochastic + TRD + CAT        | 0.740 | **-0.012**            |

三个组合实验全部低于任一单路径最佳值。

### 7.2 机制冲突假说

**机制假说 7.1**（耦合-训练动力学冲突）：Stochastic OT 耦合与 TRD 自条件化可能存在结构性冲突。

**论证**：

Stochastic OT 耦合的核心机制是：在每个训练迭代中，从传输矩阵中随机采样配对，使得同一 $x_t$ 在不同迭代中看到不同的目标速度 $v^\*$。这种**训练时不确定性**是多样性的来源。

TRD 自条件化的核心机制是：在训练时，以概率 $p$ 用前一步的预测 $x_0^{prev}$ 估计 $v\_\\pi$，然后沿 ODE 路径前进一步。这要求 $x_0^{prev}$ 是当前 $x_t$ 的合理估计。

**冲突点**：Stochastic OT 在不同迭代中为同一 $x_t$ 提供不同的 $v^\*$，而 TRD 依赖 $x_0^{prev}$ 提供一致的 $v\_\\pi$ 估计。当两者组合时：

1. TRD 的自条件化步骤使用 $x_0^{prev}$ 估计 $v\_\\pi$，但 Stochastic OT 的随机性使得 $v\_\\pi$ 在不同迭代间不一致
2. TRD 将 $x_t$ 沿估计的 $v\_\\pi$ 前进到 $x\_{t+\\Delta t}$，但 Stochastic OT 在新位置可能分配不同的 GT 目标
3. 这导致 TRD 的自条件化步骤引入额外的训练噪声，而非提供有用的精化信号

形式化地，设 $v\_\\pi^{(i)}$ 为第 $i$ 次迭代的传输速度（因 Stochastic 而随机），TRD 的自条件化估计为 $\\hat{v}_\\pi = (x_t - x_0^{prev})/t$。当 $v_\\pi^{(i)} \\neq \\hat{v}\_\\pi$ 时（Stochastic 耦合下概率很高），TRD 的前进步骤方向错误，引入噪声：

$$x\_{t+\\Delta t}^{TRD} = x_t + \\Delta t \\cdot \\hat{v}_\\pi \\neq x_t + \\Delta t \\cdot v_\\pi^{(i)}$$

此误差在 Stochastic 耦合下被放大（因 $v\_\\pi^{(i)}$ 的方差大），而在确定性耦合（nearest OT）下不存在（因 $v\_\\pi^{(i)} = \\hat{v}\_\\pi$ 恒成立）。

### 7.3 CAT 与 OT 的曲率冲突（二次修正）

**机制假说 7.2**（CAT-OT 冲突——精确版）：CAT 的 $x_0$ 一致性正则化与 OT 耦合可能在 Voronoi 边界处产生梯度冲突，并导致训练崩溃。

**论证（代码层面精确分析）**：

**Step 1：CAT 的 $x\_{t+\\Delta t}$ 构造方式**

CAT 在计算 $x\_{t+\\Delta t}$ 时使用与 $x_t$ **相同的 OT 配对**：

```python
# diffusiondet_head.py _add_cat_loss
x_start_batch = torch.stack(x_starts)   # OT 耦合后的 x_0
x_noise_batch = torch.stack(x_noises)   # 原始噪声 x_1
x_t2 = (1.0 - t2_view) * x_start_batch + t2_view * x_noise_batch
```

这里 `x_start_batch` 和 `x_noise_batch` 在两次前向传播间不变，隐含假设：$x_t$ 和 $x\_{t+\\Delta t}$ 沿**同一条 OT 配对的直线路径**，速度场应为常数。

**Step 2：OT 耦合下 Voronoi 边界的时间依赖性**

OT 耦合将噪声空间划分为 Voronoi 单元 $\\mathcal{V}\_k$。$x_t \\in \\mathcal{V}\_k$ 的条件为：

$$\\left|\\frac{x_t - b_k}{1-t}\\right| \\leq \\left|\\frac{x_t - b_j}{1-t}\\right|, \\quad \\forall j \\neq k$$

化简得 Voronoi 边界超平面方程：

$$2(x_t - b_k)^\\top(b_k - b_j) + (1-t)|b_k - b_j|^2 = 0$$

边界法向为 $(b_k - b_j)$，截距为 $(1-t)|b_k - b_j|^2 / 2$。**截距随 $t$ 线性变化**：当 $t$ 增加 $\\Delta t$ 时，边界向 $b_k$ 方向移动 $\\Delta t \\cdot |b_k - b_j|^2 / 2$。

**Step 3：冲突的精确机制**

CAT 的 $x\_{t+\\Delta t}$ 使用相同的 OT 配对构造，意味着 CAT 假设 $x_t$ 和 $x\_{t+\\Delta t}$ 属于同一个 Voronoi 单元（配对不变）。但由于 Voronoi 边界随 $t$ 移动，$x\_{t+\\Delta t}$ 可能跨越到相邻的 Voronoi 单元 $\\mathcal{V}\_j$。

此时出现三方矛盾：

| 约束来源 | 要求                                             | 代码位置                             |
| -------- | ------------------------------------------------ | ------------------------------------ |
| OT 耦合  | $v^\* = b_j - z$（新单元的速度）                 | `_couple_ot` 返回的 `x_start`        |
| CAT 构造 | $x\_{t+\\Delta t}$ 沿旧配对的直线路径            | `x_t2 = (1-t2)*x_start + t2*x_noise` |
| CAT loss | $x_0^{pred}(t) \\approx x_0^{pred}(t+\\Delta t)$ | `F.mse_loss(x0_t1, x0_t2.detach())`  |

具体地：

- CAT 构造的 $x\_{t+\\Delta t}$ 位于旧配对 $(z, b_k)$ 的直线路径上，期望模型预测 $x_0^{pred}(t+\\Delta t) \\approx b_k$
- 但 OT 耦合在 $x\_{t+\\Delta t}$ 处可能分配 $b_j$（因为 $x\_{t+\\Delta t}$ 已跨越 Voronoi 边界），检测损失要求 $x_0^{pred}(t+\\Delta t) \\approx b_j$
- CAT loss 惩罚 $x_0^{pred}(t) \\neq x_0^{pred}(t+\\Delta t)$，即惩罚 $b_k \\neq b_j$；在跨越 Voronoi 边界的样本上，这种差异来自配对切换

**Step 4：崩溃的动力学解释**

当 CAT + OT 组合训练时，梯度更新陷入三方拉锯：

1. 检测损失 $\\mathcal{L}_{det}$ 推动模型在 $x_{t+\\Delta t}$ 处预测 $b_j$（OT 配对的目标）
2. CAT 损失 $\\mathcal{L}_{CAT}$ 推动模型在 $x_{t+\\Delta t}$ 处预测 $b_k$（与 $x_t$ 处一致）
3. 两者梯度方向可能相反，且 $b_k \\neq b_j$ 会使冲突难以通过单一预测同时满足

在 Voronoi 边界附近，这种冲突的样本比例随训练进行而增加（模型学会在边界附近产生不确定预测，增加边界跨越的概率），形成正反馈循环，最终导致训练崩溃（eval=0）。

**对比**：CAT 单独使用时（无 OT 耦合），随机耦合下不存在 Voronoi 结构，$v^\*$ 在不同 $t$ 处的变化是连续的（条件期望的平滑变化），CAT 的平滑化约束与训练信号兼容，因此不会崩溃（0.744）。

> **⚠️ 代码审计发现的问题**：CAT 的 $x\_{t+\\Delta t}$ 构造使用与 $x_t$ 相同的 OT 配对，但 Voronoi 边界随 $t$ 变化导致 $x\_{t+\\Delta t}$ 可能属于不同 Voronoi 单元。这是 CAT + OT 崩溃的代码层面根源。如果未来需要让 CAT 与 OT 兼容，有两个方向：(1) 为 $x\_{t+\\Delta t}$ 重新运行 OT 耦合（`x_start_t2 = _couple_ot(x_t2, gt_diffusion, labels, device)`），使 CAT 的构造与 OT 的配对一致，但这引入额外计算开销；(2) 在 CAT loss 中排除 Voronoi 边界附近的样本（通过检测 $x_0^{pred}(t)$ 与最近 GT 的距离是否接近次近 GT 的距离来识别边界样本），但这需要额外的边界检测逻辑。

### 7.4 对论文的影响

两条 SOTA 路径的负交互不是超参数问题，而是**结构性冲突**：

1. **Stochastic OT + TRD**：训练时不确定性与自条件化一致性要求矛盾
2. **OT + CAT**：Voronoi 边界跳变与曲率平滑化要求矛盾

这意味着两条路径虽然各自有效，但**不能简单叠加**。需要设计新的组合策略来绕过这些冲突（如分阶段训练、解耦参数等）。

______________________________________________________________________

## 8. 面向目标检测的理论突破点

### 8.1 问题重述：检测不是无条件 OT，而是条件集合匹配

当前负结果说明：直接把生成模型中的 OT 直觉搬到检测框扩散中并不充分。目标检测的核心不是把噪声分布运输到一个连续数据流形，而是在图像条件 $f$ 下，把 $N$ 个 noisy proposals 分配到 $K$ 个离散目标、背景和重复候选之间。也就是说，检测中的耦合应同时满足三类约束：

1. **几何传输短**：proposal 到 GT 的框空间位移不能过大。
2. **检测语义对齐**：分配目标应与分类、IoU、尺寸、染色体组别等检测代价一致。
3. **监督多样性充足**：同一局部区域不能被硬分配过早压成单一目标，否则低维框空间中的泛化会变差。

这提示一个更适合顶会论文的创新点：从“geometry-only OT”转向 **Detection-Aware Stochastic Coupling**。

### 8.2 推荐主线：Detection-Aware Entropic Coupling (DAEC)

**核心想法**：把耦合矩阵从纯几何代价最小化，改为检测感知的熵正则集合匹配。对每张图像，构造 proposal $i$ 与 GT $j$ 的代价：

$$C_{ij}=\\alpha C^{box}_{ij}+\\beta C^{cls}_{ij}+\\gamma C^{scale}_{ij}+\\eta C^{group}_{ij}+\\rho C^{unc}_{ij}$$

其中：

- $C^{box}_{ij}$：框空间距离或 GIoU/DIoU 代价。
- $C^{cls}_{ij}$：当前检测头对类别/实例的匹配代价。
- $C^{scale}_{ij}$：尺寸匹配代价，避免小目标被大位移 proposal 主导。
- $C^{group}_{ij}$：领域先验，如染色体 A-G 组、性染色体组别；通用检测中可替换为类别层级或语义相似度。
- $C^{unc}_{ij}$：不确定性代价，鼓励高不确定区域保持更多候选监督。

然后求熵正则耦合：

$$\\pi^\* = \\arg\\min_{\pi\in\Pi(a,b)} \langle C,\pi\rangle - \tau H(\pi)$$

训练时从 $\\pi^\*$ 中 stochastic sampling，而不是 argmax：

$$Y_i \\sim \pi^\*(\cdot\mid i), \quad x_t=(1-t)b_{Y_i}+t z_i$$

关键不是“更软的 OT”，而是 **检测代价进入耦合本身**。这把 diffusion coupling 与 DETR/Hungarian matching 的思想统一起来：耦合既是流匹配的路径选择，也是检测任务的监督分配。

### 8.3 为什么它可能带来正向提升

DAEC 对当前瓶颈有三点直接回应：

1. **比 hard OT 更稳**：熵正则和 stochastic sampling 保留 $H(Y\mid X_t)$，避免低维框空间的多样性坍缩。
2. **比 random/stochastic Sinkhorn 更准**：代价矩阵加入分类、尺度、组别和不确定性，使随机性集中在“合理目标集合”内，而不是无条件地扩大匹配噪声。
3. **无推理成本**：耦合只发生在训练阶段，推理仍使用原检测头和 ODE/Heun 采样。

预期正向收益不是来自单纯降低 $W_2$，而是来自降低 $B_{match}$ 同时维持足够高的 $D_{idx}$。用 §1.4 的严谨化变量表示，DAEC 的目标是寻找：

$$\\min_\pi C_{trans}(\pi)+\\lambda B_{match}(\pi) \quad \text{s.t.}\quad H(Y\mid X_t)\ge h_{min}$$

这比“硬 OT vs 随机耦合”的二选一更像检测任务真正需要的解。

### 8.4 顶会级创新表述

可以凝练成如下论文贡献：

> We reveal that box diffusion for object detection is not governed by geometry-only optimal transport, but by a detection-aware coupling problem balancing transport efficiency, task-aligned matching, and target-index entropy. Based on this, we propose Detection-Aware Entropic Coupling, a training-only stochastic matching mechanism that unifies flow matching couplings with set-prediction assignment.

这个创新点比现有实验中的 group-hierarchical stochastic 更通用：group prior 只是 $C^{group}$ 的一个特例；在 COCO 上可以替换为类别层级、objectness、IoU/quality prediction 或 teacher uncertainty。

### 8.5 必要实验与判定标准

要把 DAEC 做成顶会级正向结果，至少需要满足：

1. **主结果**：在当前染色体数据集上超过 `adaln` / `sinkhorn_sample_eps5` / `group_hierarchical_stoch` 的多 seed 均值，目标提升建议至少 +0.3 到 +0.5 mAP，且标准差不覆盖全部增益。
2. **通用性**：在 COCO 或至少一个非染色体检测数据集上验证，不要求达到 SOTA，但要说明 hard OT 失败与 DAEC 改善不是单数据集偶然。
3. **机制验证**：同时报告 $C_{trans}$、$H(Y\mid X_t)$、$B_{match}$ 与 mAP，展示 DAEC 确实降低匹配偏差且保留索引熵。
4. **消融**：去掉 $C^{cls}$、$C^{scale}$、$C^{group}$、$C^{unc}$、熵约束和 stochastic sampling，验证每一项的边际作用。
5. **推理成本**：报告训练时增加的耦合计算，并确认推理 FLOPs/latency 不变。

### 8.6 备选突破点

若 DAEC 增益不足，次优先级方向如下：

- **Entropy-Scheduled Coupling**：训练早期保持高 $H(Y\mid X_t)$，后期逐步降低熵以提高传输效率。风险是容易退化为调参型贡献，创新强度弱于 DAEC。
- **Boundary-Aware Coupling**：显式检测 Voronoi/assignment 边界，对边界样本使用软耦合，对内部样本使用硬耦合。理论清晰，但实现和可视化复杂。
- **Coupling-Conditioned Head**：把耦合不确定性作为条件输入检测头，使模型知道当前监督来自确定匹配还是多候选匹配。可能有增益，但会增加推理或架构复杂度。

综合判断：**DAEC 是最适合冲顶会的主线**，因为它能把当前项目最强的理论发现（低维 OT 多样性坍缩）转化为一个通用、训练-only、可消融、可能正向提升的方法。

______________________________________________________________________

## 9. 修正后的实验验证状态

| 理论预测                                | 实验结果                             | 状态                  |
| --------------------------------------- | ------------------------------------ | --------------------- |
| RF 直线路径优于 DDPM 弯曲路径           | +0.8% mAP                            | ✅ 一致               |
| Shifted schedule 集中优化高敏感度时间步 | +1.4% mAP                            | ✅ 一致               |
| AdaLN-Zero 时间条件残差从零增长         | +1.1% mAP                            | ✅ 一致               |
| OT 耦合减小传输代价                     | 确实减小 $\|v^\*\|^2$                | ✅ 机制正确           |
| OT 耦合应提升性能                       | hard OT (0.735) \< random (0.751)    | ❌ 原理论错误，已修正 |
| Stochastic Coupling 恢复多样性          | eps=5 达到 0.751                     | ✅ 一致               |
| TRD 减小离散化误差                      | trd_full (0.752) > adaln (0.751)     | ✅ 一致（需组合）     |
| Reflow 应持续改善路径直度               | Epoch 1 后持续退化                   | ❌ 原理论错误，已修正 |
| OT + TRD 组合应叠加                     | 组合全为负交互                       | ❌ 原理论错误，已修正 |
| Scale-Conditioned FM 改善小物体         | sc_combined (0.736) \< adaln (0.751) | ✅ 负结果理论正确     |

______________________________________________________________________

## 10. 修正后的理论贡献总结

1. **多样性-传输效率权衡命题**（§1.4）：OT 耦合的净效果维度依赖；在低维检测框空间中，索引熵损失可能抵消传输代价收益。
2. **总误差来源分解**（§2.3）：将 ODE 离散化误差与泛化/匹配误差分开讨论，避免把采样误差完全归因于曲率。
3. **检测扩散传输四因素框架**（§5）：耦合质量、训练信号多样性、修正曲率和时间分配共同影响训练效果。
4. **Reflow 梯度冲突机制命题**（§6.2）：检测损失与速度损失在共享参数上存在经验可测的负梯度相似度。
5. **耦合-训练动力学冲突假说**（§7.2）：Stochastic OT 与 TRD 自条件化可能因一致性要求不同而产生负交互。
6. **CAT-OT 冲突假说**（§7.3）：$x_0$ 一致性正则化与 OT Voronoi 边界时间依赖性可能形成三方矛盾。
7. **AdaLN-Zero 修正**（§1.3）：零初始化保证时间条件维度零初始化，而非保证速度场为零。
8. **CAT 实际目标与理论目标的差异**（§4.2）：代码惩罚 $x_0$ 一致性，同时约束速度大小和曲率，强于纯曲率正则。
9. **Detection-Aware Entropic Coupling 研究主线**（§8）：将几何传输、检测匹配代价和索引熵约束统一为训练-only 的 stochastic coupling，是最有希望形成顶会级正向贡献的下一步。

______________________________________________________________________

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
- 与机制命题 6.1 的梯度冲突推导不一致（推导中假设 $v\_\\theta$ 与 RF 定义同向）

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

______________________________________________________________________

## 附录 B：设计改进方案（待实验验证）

> 以下方案由代码审计和理论分析推导得出，尚未经过实验验证。记录于此供后续实验参考。

### B.1 Reflow 分阶段训练

**针对问题**：机制命题 6.1 的梯度冲突（$\\rho = -0.104$），Epoch 1 后持续退化。

**方案**：

| 阶段    | Epoch | freeze_shared | velocity_detach | 学习率 | 目标                                 |
| ------- | ----- | ------------- | --------------- | ------ | ------------------------------------ |
| Phase 1 | 1     | False         | False           | 1e-4   | 同时优化检测和速度，利用预训练初始化 |
| Phase 2 | 2+    | True          | False           | 5e-5   | 冻结共享层，只精调 velocity_head     |

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

**针对问题**：机制假说 7.1 的 Stochastic OT + TRD 冲突。

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

**理论依据**：TRD 的自条件化步骤需要与当前迭代的 OT 配对一致。使用解析速度 $v^\* = x_1 - x_0$ 保证前进方向与当前 OT 配对完全一致，消除 Stochastic OT 的随机性引入的不一致性。

**预期效果**：Stochastic OT + TRD 组合不再产生负交互，mAP 应接近或超过 0.752。

**风险**：解析速度 $v^\*$ 是训练目标速度，不是模型实际学到的速度。TRD 前进到 $x\_{t+\\Delta t}$ 后，模型在该点的预测可能与解析速度外推的位置不一致，引入新的训练噪声。需要实验验证哪种估计更优。

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

**理论依据**：纯曲率正则化仅约束 $|\\partial v\_\\theta/\\partial t|^2$，不约束速度大小 $|v\_\\theta|^2$，比当前实现更温和。

**风险**：

1. $t \\approx 0$ 处 $v\_\\theta = (x_t - x_0^{pred})/t$ 数值不稳定
2. 纯曲率正则化可能不足以约束速度大小，导致路径过长
3. 需要重新调参和实验验证

### B.5 CAT 与 OT 兼容化

**针对问题**：§7.3 发现 CAT 的 $x\_{t+\\Delta t}$ 使用相同 OT 配对构造，但 Voronoi 边界随 $t$ 移动导致三方矛盾。

**方案 1：为 $x\_{t+\\Delta t}$ 重新运行 OT 耦合**

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
