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
| E5   | OT + TRD 组合应叠加增益                                | 组合实验全为负交互 (0.740-0.746)  | 新增机制冲突假说（→ THEORY_WHY_FAILED 方向 E）              |
| E6   | Reflow 应持续改善路径直度                              | Epoch 1 最佳后持续退化            | 新增梯度冲突理论（→ THEORY_WHY_FAILED 方向 D）              |
| E7   | 速度场分解中 OT 分量可直接解析计算                     | 实际训练中 OT 分量依赖耦合策略    | 修正分解定理的适用条件（§2.2 修正）             |

**二次修正**（代码审计发现的理论-代码偏差）：

| 编号 | 一次修正主张                                             | 代码审计发现                                                                                | 二次修正内容                                                  |
| ---- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| E8   | AdaLN-Zero 初始时 `fc_feature = h`（原始 proposal 特征） | Block 2 (Instance Interaction) 无 α 门控，初始时 `fc_feature = h + inst_interact(h, f_roi)` | 弱化"时间无关基线"为"时间条件维度零初始化"（§1.3 二次修正）   |
| E9   | CAT 惩罚曲率 $\|\\partial v\_\\theta/\\partial t\|^2$    | 代码默认惩罚 $\|x_0^{pred}(t) - x_0^{pred}(t+\\Delta t)\|^2$（可切换为 `velocity_curvature` 模式，2026-05-28 验证）            | 精确描述 CAT 的双模式目标（§4.2 二次修正） |
| E10  | CAT-OT 冲突因"Voronoi 边界跳变与曲率平滑化矛盾"          | CAT 的 $x\_{t+\\Delta t}$ 使用相同 OT 配对构造，但 Voronoi 边界随 $t$ 移动导致配对不一致    | 精确描述三方矛盾机制（→ THEORY_WHY_FAILED 方向 E）                         |

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

> **代码审计 / 重跑标注（2026-05-15）**：该结论依赖 DDPM baseline。当前 `projects/LDMDet/mods/diffusiondet_head.py::_ddim_step` 在 `t_next < 0` 时仍先访问 `self.alphas_cumprod[t_next]`，会触发 Python 负索引，影响 `ldmdet_baseline` 与 `ldmdet_baseline_step4` 的公平性。应先修复 `_ddim_step` 的终止步逻辑，再重跑 `ldmdet_baseline`、`ldmdet_baseline_step4` 以及 RF-vs-DDPM 公平对比。RF 本身的直线路径命题不受影响，但 `+0.8 mAP` 数值需要重验。

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

> **代码审计 / 重跑标注（2026-05-15）**：未发现 shifted schedule 实现层面的必须修正项。若作为论文主结论，建议在同一 solver、batch size、训练轮数下补 3 seed 均值；但不属于“修代码后必须重跑”的问题。

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

AdaLN-Zero 零初始化保证**时间条件残差为零**，即 $\\alpha(0) = \\gamma(0) = \\beta(0) = 0$，使得初始时网络输出与时间 $t$ 无关。但 $v\_\\theta|\_{\\text{init}} \\neq 0$，因为 `cls_head` 和 `reg_head` 非零初始化。

**严格论证（二次修正版）**：

初始化时，`adaln_mlp` 输出全零，因此：

- Block 1 (Self-Attention)：$h\_{SA} = h + \\alpha_1 \\cdot \\text{Attn}(\\cdots) = h + 0 = h$（恒等）
- Block 3 (FFN)：$h_{FFN} = h + \alpha_2 \cdot \text{FFN}(\cdots) = h + 0 = h$（恒等）

**二次修正**：代码审计发现 Block 2 (Instance Interaction) **不受 $\alpha$ 门控**：

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

> **代码审计 / 重跑标注（2026-05-15）**：AdaLN-Zero 当前实现与“时间条件残差为零”的修正版理论一致，但 `single_head.py` 中 Instance Interaction 不受 `alpha` 门控，因此不能用现有实验支持“全 block 零基线”或“初始速度场为零”。无需为修正版 AdaLN 结论重跑；若要验证“给 Instance Interaction 增加 alpha3 门控是否更优”，应作为新 ablation 重跑。

______________________________________________________________________

### 1.4 Random → OT Coupling：多样性-传输效率权衡（重大修正）

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

> **代码审计 / 重跑标注（2026-05-15）**：`_sinkhorn_match` 和 `_run_group_hierarchical_ot` 中 `torch.multinomial` 未显式传入固定 `torch.Generator`，且文档已有 `sinkhorn_sample_eps5` 主实验 0.751、repro 0.738、seed2 0.750 的大幅波动。所有依赖 stochastic coupling 的数值结论（`sinkhorn_sample_eps*`、`group_hierarchical_stoch`、组合实验中的 stochastic OT）都应以至少 5 seed 的 mean ± std 重跑；若要提升工程可复现性，可增加 `ot_sample_seed` 或记录每次采样 RNG 状态。硬 OT / argmax 的趋势结论不受 multinomial 随机性影响。

______________________________________________________________________

### 1.5 Gaussian → Structured Noise：源分布优化

**纯高斯**：$z \\sim \\mathcal{N}(0, I_4)$

**网格噪声**：$z = z\_{\\text{grid}} + \\sigma \\cdot \\epsilon$

**命题 1.5**：GT 框中心在 $\[0,1\]$ 范围内，网格噪声中心也在 $\[0,1\]$ 附近，而高斯噪声中心在 $(-\\infty, +\\infty)$。因此 $W_2^2(\\mu\_{grid}, \\mu\_{gt}) \< W_2^2(\\mathcal{N}(0,I), \\mu\_{gt})$。

**实验修正**：`ldmdet_flowdet_structured_noise` (0.742) \< `ldmdet_flowdet_adaln` (0.751)。结构化噪声虽减小 $W_2^2$，但限制了噪声分布的覆盖范围，可能损害模型对极端位置的泛化能力。此改进的收益被泛化损失抵消，与 OT 耦合的多样性问题类似。

> **代码审计 / 重跑标注（2026-05-15）**：未在当前主线代码中发现结构化噪声结论对应的明确实现 bug。但该结论目前只说明“当前结构化噪声设计无收益”，不能排除更好的染色体先验噪声。论文中应保留为负结果；若要强 claim，需补多 seed 与覆盖率统计。

______________________________________________________________________

## 2. 统一数学框架：条件最优传输

### 2.1 问题定义

- 源分布：$\\mu_z$（噪声框，$\\mathbb{R}^4$）
- 目标分布：$\\mu\_{gt} = \\frac{1}{K}\\sum\_{i=1}^K \\delta\_{b_i^{gt}}$（GT 框，离散）
- 条件信息：$f$（图像特征）
- 传输映射：$T_f: \\mu_z \\to \\mu\_{gt}$

目标：找到条件传输映射 $T_f$，使传输代价最小、检测精度最高、ODE 求解高效。

### 2.2 速度场分解（修正版）

**定义 2.1**（速度场分解——修正版）：给定耦合策略 $\pi$，可将 RF 速度场按条件期望分解为：

$$v_\theta(x_t, t, f) = \underbrace{v_{\pi}(x_t, t)}_{\text{耦合依赖的传输分量}} + \underbrace{\delta v_\theta(x_t, t, f)}_{\text{特征修正分量}}$$

其中：

- $v_{\pi}(x_t, t) = \mathbb{E}_{(z, b_0) \sim \pi}[z - b_0 \mid x_t]$：依赖耦合策略 $\pi$，与图像特征无关
- $\delta v_\theta(x_t, t, f) = \mathbb{E}[z - b_0 \mid x_t, f] - \mathbb{E}[z - b_0 \mid x_t]$：依赖图像特征的修正

**⚠️ 重要说明**：此分解是**总体层面**（条件期望层面）的分解，而非逐样本的分解。对于特定的训练样本 $(x_t, f, z, b_0)$，$v_\theta(x_t, t, f) \neq v_\pi(x_t, t) + \delta v_\theta(x_t, t, f)$，因为 $v_\pi$ 是对所有耦合配对的条件期望，而单次训练中模型只看到一个特定的配对。此分解的意义在于：它将速度场的统计特性（期望、方差）分解为耦合策略贡献和特征修正贡献，从而解释不同改进措施对总体性能的影响。

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

OT 传输分量 $v_\pi$ 在给定耦合 $\pi$ 后可解析计算。但**最优耦合 $\\pi$ 本身依赖图像特征**（不同图像的 GT 框分布不同），因此 $v\_\\pi$ 不能完全脱离特征计算。TRD 的自条件化推理通过估计 $\\hat{v}\_\\pi$ 绕过了这个问题，但引入了估计误差。

### 3.2 检测损失的时间敏感度

**命题 3.2**：在模型预测 $x_0$ 且速度由 $v_\theta=(x_t-x_0^{pred})/t$ 推导时，速度误差对检测损失的影响为：

$$\\left|\\frac{\\partial \\mathcal{L}_{det}}{\\partial (\\delta v)}\\right| = t \\cdot \\left|\\nabla_{x_0} \\mathcal{L}\_{det}(x_0^{pred})\\right|$$

损失敏感度与 $t$ 成正比。

正确表述：$t = 1$ 处的速度误差对检测损失的影响是 $t = 0.1$ 处的 10 倍（线性比例关系），而非 $1/t^2$ 倍。

### 3.3 离散目标分布的结构

$\\mu\_{gt}$ 是有限支撑的离散测度，OT 映射是分片常数的。当 $N \\gg K$（500 proposals vs ~46 GT），nearest 耦合已近似最优。

Sinkhorn 的真正价值在于通过 $\epsilon$ 参数在 OT 与随机耦合之间插值，从而调控训练多样性。但这一价值**仅在 Stochastic Coupling（从传输矩阵采样）下才能实现**；在 argmax 解码下，$\\epsilon$ 的多样性调控会被显著削弱（CAM 命题）。

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

> **代码审计 / 重跑标注（2026-05-15，2026-05-28 修正）**：`prediction_mode='velocity'` 的实验曾被标记为受 velocity target 符号问题影响。2026-05-28 逐行验证确认 `_add_velocity_loss`（行881）与 `rectified_flow.py`（行50）的 velocity 定义均为 `noise - start`，符号一致，无需修复。TRD 训练/推理复用 `cat_delta_t` 的问题仍然存在，建议新增 `trd_delta_t` 后重跑 TRD 与 TRD+CAT 消融。

### 4.2 Curvature-Aware Training (CAT)

**目标**：降低 ODE 路径的离散化误差上界。

**理论正则化项**（曲率惩罚）：

$$\\mathcal{L}_{curv}^{theory} = \\mathbb{E}_t \\left\[\\left|\\frac{\\partial v_\\theta}{\\partial t}\\right|^2\\right\] \\approx \\mathbb{E}_t \\left\[\\left|\\frac{v_\\theta(x_{t+\\Delta t}, t+\\Delta t) - v\_\\theta(x_t, t)}{\\Delta t}\\right|^2\\right\]$$

**代码实际实现**（二次修正 + 2026-05-28 补充验证）：

CAT 的代码（`diffusiondet_head.py:930-937`）支持**两种模式**，通过 `cat_loss_type` 切换：

**模式 1**（`cat_loss_type='x0_consistency'`，默认）：惩罚 $x_0$ 预测的时间一致性，即 $\mathcal{L}_{CAT}^{x0} = \mathbb{E}_t [|x_0^{pred}(t) - x_0^{pred}(t+\Delta t)|^2]$。

**模式 2**（`cat_loss_type='velocity_curvature'`）：惩罚速度场的时间导数（纯曲率正则），即 $\mathcal{L}_{CAT}^{vel} = \mathbb{E}_t [|v_\theta(t+\Delta t) - v_\theta(t)|^2]$，近似 $|\partial v_\theta/\partial t|^2$。

2026-05-15 审计标注为"代码实现 $x_0$ 一致性而非曲率"——该描述仅对默认模式成立。`velocity_curvature` 模式已在代码中实现，可从 $x_0$ 推导 $v = (x_{noise} - x_0)/t$ 后做时间差分，通过配置开关启用。

**理论目标与代码实现的精确关系**：

由 $x_0^{pred} = x_t - t \\cdot v\_\\theta$，对 $t$ 求导：

$$\\frac{\\partial x_0^{pred}}{\\partial t} = \\frac{\\partial x_t}{\\partial t} - v\_\\theta - t \\cdot \\frac{\\partial v\_\\theta}{\\partial t}$$

此处 $\\partial x_t/\\partial t$ 的含义取决于场景：

- **训练时**：$x_t = (1-t)x_0 + tx_1$ 是由 OT 配对解析构造的，$\\partial x_t/\\partial t = x_1 - x_0 = v^\\*$ 是已知常数。因此：

$$\\frac{\\partial x_0^{pred}}{\\partial t}\\bigg|_{\\text{train}} = v^\\* - v\_\\theta - t \\cdot \\frac{\\partial v\_\\theta}{\\partial t}$$

当模型完美拟合（$v\_\\theta = v^\\*$）时，前两项相消，$\\partial x_0^{pred}/\\partial t = -t \\cdot \\partial v\_\\theta/\\partial t$，即 $x_0$ 一致性等价于曲率正则化。

- **推理时**：$x_t$ 由 ODE 积分生成，$\\partial x_t/\\partial t = v\_\\theta(x_t, t)$ 是模型自身的预测。因此：

$$\\frac{\\partial x_0^{pred}}{\\partial t}\\bigg|_{\\text{infer}} = v\_\\theta - v\_\\theta - t \\cdot \\frac{\\partial v\_\\theta}{\\partial t} = -t \\cdot \\frac{\\partial v\_\\theta}{\\partial t}$$

推理时 $x_0$ 一致性天然等价于曲率正则化，无需模型完美拟合。

**关键区别**：训练时 CAT 的 $x_0$ 一致性目标隐含了 $v^\\* - v\_\\theta$ 项（速度残差），而推理时没有。这意味着训练时的 CAT loss 比推理时多惩罚了速度残差，等价于对 $|v\_\\theta - v^\\*|^2$ 的额外正则化。当 $v\_\\theta \\neq v^\\*$ 时（训练早期），CAT 的训练信号与速度场拟合目标部分重叠，可能加剧过约束。

实际中 $x_0^{pred}$ 的时间导数包含额外项：

$$\\frac{\\partial x_0^{pred}}{\\partial t} \\approx -v\_\\theta - t \\cdot \\frac{\\partial v\_\\theta}{\\partial t}$$

因此代码实现的 $\\mathcal{L}\_{CAT}^{code}$ 可分解为：

$$\\mathcal{L}_{CAT}^{code} \\approx \\Delta t^2 \\cdot \\underbrace{|v_\\theta|^2}_{\\text{速度大小惩罚}} + \\Delta t^2 \\cdot t^2 \\cdot \\underbrace{\\left|\\frac{\\partial v_\\theta}{\\partial t}\\right|^2}\_{\\text{曲率惩罚}} + \\text{交叉项}$$

**关键差异**：代码实现同时约束了两个目标：

1. **速度场范数小**（$|v\_\\theta|^2$）：路径短，传输代价低
2. **速度场变化率小**（$|\\partial v\_\\theta/\\partial t|^2$）：曲率低，离散化误差小

而理论目标仅约束第 2 项。代码实现比理论描述**更激进**——它不仅要求速度场平滑，还要求速度场本身小。

**这解释了 CAT 单独使用时性能下降（0.744 \< 0.751）**：过度的 $x_0$ 一致性约束限制了模型在不同时间步做出不同预测的能力。在检测任务中，不同时间步的 $x_0$ 预测本应不同（$t$ 越小，$x_t$ 越接近噪声，预测越不确定），强制 $x_0^{pred}(t) \\approx x_0^{pred}(t+\\Delta t)$ 等价于要求模型在所有时间步给出相同的预测，这与扩散模型的多步精化机制矛盾。

> **⚠️ 代码审计发现的问题**：CAT 的代码实现惩罚 $x_0$ 预测的时间一致性 $|x_0^{pred}(t) - x_0^{pred}(t+\\Delta t)|^2$，而非理论描述的曲率 $|\\partial v\_\\theta/\\partial t|^2$。前者比后者更激进，同时惩罚速度大小和曲率。如果未来需要实现纯曲率正则化，应改为惩罚速度的时间导数：$\\mathcal{L}_{curv}^{pure} = |v_\\theta(x\_{t+\\Delta t}, t+\\Delta t) - v\_\\theta(x_t, t)|^2 / \\Delta t^2$，其中 $v\_\\theta = (x_t - x_0^{pred})/t$ 从模型输出推导。但需注意：(1) $t \\approx 0$ 处 $v\_\\theta$ 的数值不稳定（除以接近零的 $t$），(2) 纯曲率正则化可能不足以约束速度大小，(3) 需要重新调参和实验验证。

**实验修正**：`cat_only` (0.744) 略低于 `adaln` (0.751)，说明 $x_0$ 一致性正则化单独使用时过度约束了模型的时间条件表达能力。CAT 与 OT 组合时甚至导致训练崩溃（eval=0），可能原因见 THEORY_WHY_FAILED 方向 E。

> **代码审计 / 重跑标注（2026-05-15，2026-05-28 补充）**：CAT 代码已支持两种模式——默认 `x0_consistency` 和可选 `velocity_curvature`（通过 `cat_loss_type` 切换）。2026-05-15 的标注仅针对默认模式。`velocity_curvature` 模式可直接通过配置启用，无需修改代码。论文中若使用"Curvature-Aware Training"机制名，建议启用 `cat_loss_type='velocity_curvature'` 并重跑 `cat_only` 及组合消融。

### 4.3 Loss-Sensitive Adaptive Scheduling (LSAS)

**目标**：从检测损失敏感度推导最优时间采样分布。

**理论最优**：

$$p^\*(t) \\propto t \\cdot \\sqrt{\\mathbb{E}\\left\[\\left|\\nabla\_{x_0} \\mathcal{L}\_{det}\\right|^2\\right\]}$$

**实验验证**：`lsas` (0.743) 单独使用效果有限，但在 `trd_full` 组合中贡献 +0.001-0.002。LSAS 的收益被其他机制部分覆盖（shifted schedule 已提供了粗粒度的时间偏向）。

> **代码审计 / 重跑标注（2026-05-15）**：LSAS 单独实验未发现必须修正项。但 `trd_full` 中的 LSAS 贡献与 velocity/CAT/TRD 混合，且增益小于典型随机波动；修正 velocity target、CAT 目标和 TRD 步长后，需要重新做组合消融才能确认 LSAS 的边际收益。

______________________________________________________________________

## 5. 核心原理：检测扩散传输四因素框架（重大修正）

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

## 6. 失败方向分析与研究路线图（已移出）

Reflow 退化陷阱（梯度冲突）和两条 SOTA 路径的负交互（机制冲突）属于失败方向分析，已移入 [THEORY_WHY_FAILED.md](THEORY_WHY_FAILED.md) 方向 D 和方向 E。

面向目标检测的理论突破点（DAEC/KCEC）属于未来研究规划，已移入 [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)。

______________________________________________________________________

## 7. 修正后的实验验证状态

| 理论预测                                | 实验结果                             | 状态                  | 代码/重跑要求 |
| --------------------------------------- | ------------------------------------ | --------------------- | ------------- |
| RF 直线路径优于 DDPM 弯曲路径           | +0.8% mAP                            | ✅ 方向一致           | 修复 DDPM `_ddim_step` 后重跑数值 |
| Shifted schedule 集中优化高敏感度时间步 | +1.4% mAP                            | ✅ 一致               | 无已知代码修正项，建议多 seed |
| AdaLN-Zero 时间条件残差从零增长         | +1.1% mAP                            | ✅ 一致               | 无需重跑修正版结论；全 block 门控是新消融 |
| OT 耦合减小传输代价                     | 确实减小 $\|v^\*\|^2$                | ✅ 机制正确           | 建议补 $C_{trans}$ 统计 |
| OT 耦合应提升性能                       | hard OT (0.735) \< random (0.751)    | ❌ 原理论错误，已修正 | hard OT 结论可保留；random/stoch 对照需多 seed |
| Stochastic Coupling 恢复多样性          | eps=5 达到 0.751                     | ⚠️ 趋势支持           | 必须多 seed；可加固定 generator |
| TRD 减小离散化误差                      | trd_full (0.752) > adaln (0.751)     | ⚠️ 当前代码下成立     | 修 velocity target、TRD/CAT 步长后重跑 |
| Reflow 应持续改善路径直度               | Epoch 1 后持续退化                   | ❌ 原理论错误，已修正 | 修 velocity target 后重测梯度冲突与 Reflow |
| OT + TRD 组合应叠加                     | 组合全为负交互                       | ⚠️ 当前代码下负交互   | 修 stochastic/velocity/CAT/TRD 后重跑组合 |
| Scale-Conditioned FM 改善小物体         | sc_combined (0.736) \< adaln (0.751) | ⚠️ 负结果趋势         | 无已知代码修正项，但不应写“必然无效” |

______________________________________________________________________

## 8. 修正后的理论贡献总结

1. **多样性-传输效率权衡命题**（§1.4）：OT 耦合的净效果维度依赖；在低维检测框空间中，索引熵损失可能抵消传输代价收益。
2. **总误差来源分解**（§2.3）：将 ODE 离散化误差与泛化/匹配误差分开讨论，避免把采样误差完全归因于曲率。
3. **检测扩散传输四因素框架**（§5）：耦合质量、训练信号多样性、修正曲率和时间分配共同影响训练效果。
4. **Reflow 梯度冲突机制命题**（→ [THEORY_WHY_FAILED.md](THEORY_WHY_FAILED.md) 方向 D）：检测损失与速度损失在共享参数上存在经验可测的负梯度相似度。
5. **耦合-训练动力学冲突假说**（→ [THEORY_WHY_FAILED.md](THEORY_WHY_FAILED.md) 方向 E）：Stochastic OT 与 TRD 自条件化可能因一致性要求不同而产生负交互。
6. **CAT-OT 冲突假说**（→ [THEORY_WHY_FAILED.md](THEORY_WHY_FAILED.md) 方向 E）：$x_0$ 一致性正则化与 OT Voronoi 边界时间依赖性可能形成三方矛盾。
7. **AdaLN-Zero 修正**（§1.3）：零初始化保证时间条件维度零初始化，而非保证速度场为零。
8. **CAT 实际目标与理论目标的差异**（§4.2）：代码惩罚 $x_0$ 一致性，同时约束速度大小和曲率，强于纯曲率正则。
9. **Detection-Aware Entropic Coupling 研究主线**（→ [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) §2）：将几何传输、检测匹配代价和索引熵约束统一为训练-only 的 stochastic coupling，是通用目标检测方向的顶会级候选贡献。
10. **Karyotype-Constrained Entropic Coupling 染色体主线**（→ [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) §3）：将同源交换对称性、软倍性配额、形态先验和异常核型 slack 写入耦合目标，是更贴合染色体任务的顶会级候选贡献。

______________________________________________________________________

## 附录 A：代码审计发现的问题

> **最后一次验证**：2026-05-28。A.1/A.2/A.3 经逐行代码对比，确认当前代码中不存在这些文档记录中描述的 bug。保留历史记录供参考。

> 以下问题由代码审计发现，记录于此供后续实验参考。实验存档中的备份代码不应修改，新实验应在独立分支中进行。

### A.1 velocity loss 目标符号一致性

**严重程度**：✅ 已确认正确（2026-05-28 逐行验证）

**原始记录**（2026-05-15 代码审计标记为不一致）：

RF 速度定义（`rectified_flow.py:50`）：

```python
velocity = x_noise - x_start  # v = x_1 - x_0
```

velocity loss 目标（`diffusiondet_head.py:881`）：

```python
v_target = torch.stack(x_noises) - torch.stack(x_starts)  # v* = x_1 - x_0 = v
```

**逐行验证结论**：两者符号**完全一致**，都是 `noise - start`。2026-05-15 的审计记录中声称 `v_target = x_starts - x_noises` 来源于对旧版本代码的误读或历史版本的真 bug。当前代码无需修复。

**备注**：SOTA 使用 `prediction_mode='x0'`，velocity_head 不参与训练和推理。`prediction_mode='velocity'` 的相关实验（velocity/ITD/Reflow）中的符号问题——如存在——应针对实际代码重新验证，而非依赖此条记录。

### A.2 DDPM 多步推理性能低于单步

**严重程度**：✅ 已修复（2026-05-28 逐行验证）

**原始记录**（2026-05-15 代码审计标记为存在负索引 bug）：

`ldmdet_baseline` (1步, 0.725) > `ldmdet_baseline_step4` (4步, 0.709)，多步推理反而更差。

**实际代码验证**（`diffusiondet_head.py:1328-1338`）：

```python
def _ddim_step(self, t_curr, t_next, ...):
    x0 = self._xyxy_to_raw(pred_bboxes, img_metas)
    if t_next < 0:                         # ← 已存在! 提前返回
        return self._raw_to_xyxy(x0, img_metas), x0
    # ...
    alpha_next = self.alphas_cumprod[t_next]  # 只有 t_next >= 0 才执行
```

负索引访问 `alphas_cumprod[t_next]` 的防护已在当前代码中实现。DDPM 多步退化的根因可能不是负索引 bug，而是训练-推理不一致（训练用单步 DDIM 等价于 Euler-M 1 步暴力跳，推理多步时需要更精确的噪声调度对齐），或 `pred_noise = self.predict_noise_from_start(...)` 在多步场景下的累积误差。

### A.3 Stochastic Coupling 可复现性

**严重程度**：✅ 已实现接口（2026-05-28 逐行验证）

**原始记录**（2026-05-15 标记为需要固定随机种子）：

`sinkhorn_sample_eps5` 主实验 0.751，复现 0.738，seed2 0.750，方差 0.013 mAP。

**实际代码验证**（`diffusiondet_head.py:262-275`）：

```python
def _ot_multinomial(self, row_probs):
    if self.ot_sample_seed is None:              # ← 默认: 全局 RNG
        return torch.multinomial(row_probs, 1).squeeze(-1)
    # ...                                        # ← ot_sample_seed 已实现!
    gen = torch.Generator(device=device)
    gen.manual_seed(int(self.ot_sample_seed))
    return torch.multinomial(row_probs, 1, generator=gen).squeeze(-1)
```

固定随机种子的接口**已在代码中实现**。当前 SOTA 配置未设置 `ot_sample_seed`（使用全局 RNG），这是刻意设计——训练中的随机性是有益的。多 seed 训练（5 seed mean ± std）是处理方差的学术标准做法，优于固定单个 seed。

**论文建议**：推理时设置 `ot_sample_seed=42` 保证结果完全可复现；训练时用 3-5 seed 报告 mean ± std。

______________________________________________________________________

## 附录 B：设计改进方案（待实验验证）

> 以下方案由代码审计和理论分析推导得出，尚未经过实验验证。记录于此供后续实验参考。

> **⚠️ 所有代码修改方案仅为设计建议，需在独立实验分支中实现，严禁修改存档代码。**

### B.1 Reflow 分阶段训练

**针对问题**：THEORY_WHY_FAILED 方向 D 的梯度冲突（$\\rho = -0.104$），Epoch 1 后持续退化。

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

**针对问题**：THEORY_WHY_FAILED 方向 E 的 Stochastic OT + TRD 冲突。

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

**针对问题**：THEORY_WHY_FAILED 方向 E 发现 CAT 的 $x\_{t+\\Delta t}$ 使用相同 OT 配对构造，但 Voronoi 边界随 $t$ 移动导致三方矛盾。

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

______________________________________________________________________

## C. KCEC 最终审计：三项测量与结构性失败分析

> **审计日期**：2026-05-27
> **状态**：KCEC 正式封存。以下为最终裁决依据。

### C.1 实验全景

KCEC (Karyotype-Constrained Entropic Coupling) 经历了 V1→V2→V3→V4 四轮迭代，共 7 个实验：

| 版本 | 配置 | 最高 mAP | vs SOTA (0.753) |
|------|------|----------|-----------------|
| V1 Slot-based | full priors, 46 slots | 0.744 | -0.009 |
| V1 no_prior | morph=0, group=0 | 0.744 | -0.009 |
| V1 redundant_slots | multiplier=2, 92 slots | 0.744 | -0.009 |
| V2 Direct GT Quota | 无 slot, 直接 GT 权重 | 0.732 | -0.021 |
| V3 Scale-Aware & Gated | 尺度感知 + 代价门控 | 进行中 (~0.71) | TBD |
| V4 (未执行) | 梯度投影 + 特征解耦 | — | — |

**关键发现**：没有任何一个 KCEC 变体超过 SOTA。最优变体 (V1, 0.744) 比随机耦合 (0.753) 低 0.9%。

### C.2 三项深度诊断测量

为确定 KCEC 失败的根因，设计了三个正交诊断：

**诊断脚本**：`projects/LDMDet/tools/diagnose_kcec.py`

**对比对象**：
- SOTA: `reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth` (mAP=0.753)
- KCEC V2: `ldmdet_kcec_v2_direct_quota/best_coco_bbox_mAP_epoch_67.pth` (mAP=0.732)

#### 测量 1：Sinkhorn 收敛性审计

**方法**：对比均匀边际 (SOTA) 与 KCEC 非均匀边际 (`col_mass` 含染色体配额) 在 `epsilon=5.0` 下的收敛速度。

**结果**：

| 场景 | iter=20 row_err | iter=20 col_err |
|------|----------------|-----------------|
| 均匀 (SOTA) | 1.7e-09 | 2.6e-09 |
| 非均匀 (KCEC) | 1.9e-09 | 2.8e-09 |
| 极端非均匀 | 1.9e-09 | 3.2e-09 |

**结论**：✅ Sinkhorn 在 `ot_num_iters=20` 下对所有边际分布均充分收敛 (~1e-9)。**KCEC 的非均匀 `col_mass` 不是瓶颈。** `ot_num_iters` 不需要增加。

#### 测量 2：检测头参数一致性

**方法**：逐模块比较 SOTA 与 KCEC V2 的 `head_series` (6 个检测头) 的权重余弦相似度。

**结果**：

| 模块 | 余弦相似度 |
|------|-----------|
| head_series 平均 | 0.350 |
| self_attn (head_0) | **-0.022** |
| adaln_mlp | **0.045** |
| inst_interact | 0.443 |
| cls_head | 0.433 |
| reg_head | 0.415 |
| time_mlp | 0.278 |

**结论**：⚠️ 参数余弦 ~0.35。`self_attn` 几乎正交 (cos=-0.02)，`adaln_mlp` (时间条件化门控) 接近随机 (cos=0.045)。**但缺少对照实验**（同配置不同 seed 的两个 SOTA 模型的参数余弦），无法区分这是 KCEC 导致的还是不同训练轨迹的正常发散。此测量结果标记为 "inconclusive but suspicious"。

#### 测量 3：同源染色体特征对齐审计

**方法**：构造含同源染色体对 (2 个 A1, 2 个 G22) 的 GT，用 KCEC V2 进行 `_run_kcec_ot` 匹配后，跑 forward 获取 `obj_features`，测量同源对内部的类内余弦相似度 vs 异类染色体间的类间余弦相似度。

**结果**：

| 同源对 | 类内 cos | 类间 cos | 对齐比 |
|--------|---------|---------|--------|
| A1 pair (0,1) | 0.8495 | 0.8408 | **1.01** |
| G22 pair (2,3) | 0.8354 | 0.8343 | **1.00** |

**结论**：❌ 同源染色体的 Proposal 特征与异类染色体的 Proposal 特征**毫无区别**（对齐比 ≈ 1.00）。模型内部完全没有"同源染色体等价"的表征。

### C.3 根本原因：耦合层与表征层的"信息截断"

三项测量综合指向一个结构性失败：

```
耦合层 (OT / KCEC)  →  强制同源配额, 编排 Proposal-GT 配对
          ↓
     (x_t, target) 对   ← 信息在此处被"截断"
          ↓
    Transformer  →  只接收配对结果, 不"知道"配额存在
          ↓
    特征空间    →  同源 ≠ 对齐  ← 测量 3 证实
```

在 Rectified Flow 中，Transformer 的学习目标是 $v = \text{noise} - \text{gt}$。无论 noise_i 是被 Random OT 还是 KCEC 分配给哪个 GT_j，Transformer 看到的输入输出对 `(x_t, target)` 在代数上完全相同。**耦合策略的信息从未进入梯度流**——Transformer 不知道也不会去学习"这个配对是因为同源配额才产生的"。

KCEC 在耦合层做的一切精细化编排（Slot、Direct Quota、Scale-Aware Gating），在信息传播到 Transformer 时都已经被"压缩"为一个平凡的 $(x_t, \text{gt}_j)$ 对。模型只是在被动接受 KCEC 的配对结果，然后用一套从未学习过"同源等价"的特征去拟合。

这就是为什么：
- V1 Slot 模式不行（均值化丢失几何分辨率）
- V2 Direct Quota 不行（0.732，比 V1 更差——因为配额干扰了原本稳定的随机配对多样性）
- V3 Scale-Aware 预期也不行（尺度感知只是软化约束强度，不改变"信息截断"的结构性问题）
- 生物先验的有无对结果无影响（no_prior 消融：0.744 = 0.744）

### C.4 KCEC 的可救赎路径 (如果未来重访)

唯一可能让 KCEC 生效的路径是在 Transformer 内部注入信息，而非仅在耦合层：

1. **特征空间对比约束**：在 `single_head.py` 中，对同源类别的 Proposal 特征施加 Contrastive Loss，强制类内聚合、类间分离。
2. **分类头配额正则**：在分类头上加全局核型一致性约束。

但这些方案面临同一个根基问题：**训练时 $x_t$ 充满噪声，所有基于"当前预测"的全局数量约束都不可靠。** 只有当 $t \to 0$（接近推理终点）时，类别的概率分布才有意义，而此时梯度已极弱。

### C.5 正式结论

- **KCEC 封存**。4 轮迭代、7 个实验、3 项诊断测量，一致表明：耦合层约束无法传播到 Transformer 表征层。
- **SOTA 保留为 0.753**（Random OT + Stochastic Sampling + Heun + Shifted Schedule + AdaLN-Zero）。
- **后续方向**：转向工程融合路线（DPM-Solver++、UniPC、ConvNeXt-V2+MAE），做纯性能导向的最强扩散检测器。

______________________________________________________________________

## D. 指数积分器向 Rectified Flow 的适配与验证

> **日期**：2026-05-27 ~ 2026-05-28
> **状态**：推导完成，离线验证通过，8步训练进行中
> **概念修正**（2026-05-28）：原标注为"DPM-Solver++ 移植"，但该方法在数学本质上是**指数积分器（Exponential Integrator）+ 多项式外推**，是半线性 ODE 求解的标准数值方法，并非 DPM-Solver++ 专属创新。DPM-Solver++ 的核心优势在于 $\lambda=\ln(\alpha/\sigma)$ 空间利用信噪比指数变化规律，而 RF 的 $t$ 空间已是数据-噪声线性插值，$\lambda=\ln t$ 并无同等物理意义。保留"DPM"命名仅为工程便利，论文中应正名为"指数积分器"。

### D.1 动机

当前 SOTA 使用 Heun 求解器（2阶 ODE 积分器，2 NFE/步，4步=8 NFE）。指数积分器通过半线性 ODE 分解 + 多项式插值精确积分，在图像生成中实现了 3-4 阶精度。本文将其适配到 Rectified Flow 的目标检测场景。

### D.2 核心推导

RF ODE 的半线性形式：

$$\frac{dx_t}{dt} - \frac{1}{t}x_t = -\frac{1}{t}x_0^{pred}$$

积分因子 $1/t$ 精确求解线性项，仅 $x_0^{pred}$ 的积分需多项式近似：

$$x_{t_{n+1}} = \frac{t_{n+1}}{t_n} x_{t_n} + t_{n+1} \int_{t_{n+1}}^{t_n} \frac{x_0^{pred}(\tau)}{\tau^2} d\tau$$

在 $t$ 空间直接对 $x_0^{pred}$ 做拉格朗日插值：

- **1阶**（等价 Euler）：$x_{n+1} = \frac{t_{n+1}}{t_n}x_n + (1-\frac{t_{n+1}}{t_n})\hat{x}_n$
- **2阶**（RF-EI2）：$x_{n+1} = \text{linear} + \phi_1 D_1$，$\phi_1 = t_{n+1}\ln(t_n/t_{n+1})-t_n+t_{n+1}$
- **3阶**（RF-EI3）：$x_{n+1} = \text{linear} + \phi_1 D_1 + \phi_2 D_2$，$\phi_2 = (t_{n+1}+t_{n-1})(t_n-t_{n+1}) - t_{n+1}(t_n+t_{n-1})\ln(t_n/t_{n+1})$
- **$t_{n+1}=0$ 退化**：$\phi_1 = -t_n$，$\phi_2 = t_{n-1}\cdot t_n$，均有限值，无奇点

> **⚠️ 修正记录**：原3阶公式 $\phi_2$ 缺失 $t_{n-1}$ 依赖（误将二次插值项 $(\tau-t_n)(\tau-t_{n-1})$ 截断为 $(\tau-t_n)^2$），导致 $t_{n+1}=0$ 极限值错误为 $t_n^2/2$。修正后极限值为 $t_{n-1}\cdot t_n$。

完整推导见 `docs/dpm_solver_plus_plus_rf_derivation.md`。

### D.3 实现

| 组件 | 位置 |
|:---|:---|
| 多步法调度器 | `projects/LDMDet/mods/rectified_flow.py::RFDPMSolverMultistep` |
| 推理接入 | `projects/LDMDet/mods/diffusiondet_head.py::predict()` `solver_type='dpm_solver_pp'` |
| 训练配置 | `configs/recipes/ldmdet_dpm_solver_pp.py` (6步), `ldmdet_dpm_solver_pp_s8.py` (8步) |
| 离线对比 | `projects/LDMDet/tools/compare_solver.py` |

### D.4 离线验证（关键结果）

不重新训练，仅替换 SOTA checkpoint 的推理求解器：

| 求解器 | 步数 | NFE | mAP | 结论 |
|:---|:---|:---|:---|:---|
| Heun | 4 | 8 | 0.753 | 基准 |
| DPM-2 | 6 | **7** | **0.753** | **同等精度，−12.5% NFE** |
| DPM-2 | 8 | 9 | **0.755** | **+0.002 mAP, APs +0.004** |

**完整指标**：

| 求解器 | NFE | AP50 | AP75 | APs | APm | APl | AR |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Heun | 8 | 0.943 | 0.844 | 0.521 | 0.745 | 0.642 | 0.810 |
| DPM-2 s6 | 7 | 0.944 | 0.844 | 0.520 | 0.745 | 0.645 | 0.812 |
| DPM-2 s8 | 9 | 0.947 | 0.845 | 0.525 | 0.747 | 0.633 | 0.814 |

### D.5 训练实验

| 配置 | NFE | 最高训练 mAP | 离线推理 mAP（同权重） | 差距 |
|:---|:---|:---|:---|:---|
| DPM-2 6步 | 7 | 0.740 | 0.753 | −0.013 |
| DPM-2 8步 | 9 | 进行中（Epoch 16/150） | 0.755 | — |

**6步训练低于离线推理**的原因：多步法起步阶段（1阶）导致早期验证 mAP 偏低，影响早停和模型选择。8步训练（1/8=12.5% 1阶占比 vs 1/6=16.7%）预期缓解此问题。

### D.6 理论解释：指数积分器为何优于 Heun

Heun 在 $x$ 空间做梯形近似：$x_{t_{n+1}} \approx x_t + \frac{\Delta t}{2}(v_t + v_{t+1})$。但 $v(x,t) = (x - x_0^{pred})/t$ 含 $1/t$ 因子，当 $t \to 0$ 时 $v$ 的局部变化被 $1/t$ 放大，梯形近似的常数因子劣化。

指数积分器将半线性 ODE 的 $1/t$ 线性项精确积分（解析解），仅 $x_0^{pred}$ 的多项式拟合残留误差。这解释了 APs（小物体）+0.004 的增益——小物体在 $t \to 0$ 附近的边界框预测对速度场误差最敏感。

**NFE 现实约束**：在极低 NFE（1-4步）的检测场景中，多步法存在"预热惩罚"——前几步只能用低阶，3阶法至少需要3步预热。因此4步推理下实际只有最后1-2步能用3阶，收益有限。当前 Heun（单步2阶）在4步/8NFE 下已是最优性价比。指数积分器的甜点区间在 6-10 步的中等 NFE 场景。

### D.7 论文中的定位

三层叙事：
1. **效率**：EI-2 6步 = Heun 4步，NFE 减少 12.5%
2. **精度**：EI-2 8步 > Heun 4步，mAP +0.002，APs +0.004
3. **方法论**：将半线性 ODE 的指数积分器适配到 Rectified Flow 检测场景，验证了精确积分线性项 + 多项式近似非线性项的框架在结构化预测中的有效性
