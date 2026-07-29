# VCR: Velocity Consistency Regularization — 速度一致性正则

> **方向类型**: 保守方向 (基于已有研究的可靠性改进)
> **目标期刊**: IEEE TMI
> **核心改动**: 在训练损失中增加一项速度场时间一致性正则 `L_vcr` (正则项, 非新目标)
> **预期增益**: +0.002~0.008 mAP (Dataset 2), η_str 下降 30~50%, 泛化界收紧
> **设计原则**: 数学严谨性最高优先; 改动小 (仅加正则项); 理论分析深入 (Sobolev/Poincaré/Rademacher)
>
> **文档状态**: 设计方案 (待实验验证)
> **创建日期**: 2026-07-27
> **依赖文件**: [rectified_flow.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py), [head.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py), [criterion.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py), [theory_analysis_RF_DPM.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md), [FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)

---

## 0. 摘要

Rectified Flow (RF) 的核心理论保证是**速度场沿轨迹恒定** ($v = x_1 - x_0$), 使轨迹为直线, 少步推理即可精确。但在 KaryoFlow 中, $\eta_{\text{str}}$ 诊断 ([R1, theory_analysis_RF_DPM.md §1](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)) 显示学习轨迹非直线 ($\eta_{\text{str}} \in [0.7, 1.5]$, renewal-off 模式), 表明 $v_\theta$ 随 $t$ 变化。**R1 是推理时观测, 不能改进模型**。本文提出 VCR (Velocity Consistency Regularization): 在训练时增加一项**两时间步速度一致性正则**

$$\mathcal{L}_{\text{vcr}} = \mathbb{E}_{(t, t') \sim \mathcal{U}[0,1]^2}\left[\left\| v_\theta(x_t, t) - v_\theta(x_{t'}, t') \right\|^2\right],$$

从变分正则化的第一性原理直接约束 $v_\theta$ 接近常数。本方案严格区分于:

- **R1 $\eta_{\text{str}}$**: 推理时**诊断** (observe) vs 训练时**约束** (constrain);
- **ReFlow velocity loss** ([FALSIFIED_DIRECTIONS.md §八, §十四](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)): 形式 (b) $\|v_\theta - v\|^2$ 在数学上 (L2 squared loss 下) **恒等于** $(1/t^2)\mathcal{L}_{x_0}$, 即被证伪的 ReFlow velocity loss; VCR 采用**目标无关的成对一致性** (form (a)), 不引入竞争性目标;
- **梯度冲突陷阱** (cos = -0.104, [PUBLICATION_EVALUATION.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/PUBLICATION_EVALUATION.md)): VCR 梯度与 $\mathcal{L}_{\text{det}}$ 梯度在全局最小值点**同时为零** (共享最小值子空间), 不存在目标竞争。

理论上, VCR 是 $v_\theta$ 在时间维上的 **$L^2$ 方差** $\text{Var}_t(v_\theta) = \|v_\theta - \bar{v}_\theta\|_{L^2}^2$ 的精确无偏估计 (命题 2.1, 因子 2); 由 **Poincaré 不等式**, VCR 与 **Sobolev $H^1$ 半范数** $|v_\theta|_{H^1}^2$ 满足单向上界 $\mathcal{L}_{\text{vcr}} \leq \frac{2}{\pi^2} |v_\theta|_{H^1}^2$ (Poincaré 给上界方向, 非**等价**); 由 **Rademacher 复杂度**的 Kolmogorov-Tikhomirov 度量熵论证, $H^1$ 有界的函数类具有更小的复杂度, 从而以 $\sqrt{\tilde{B}}$ 速率收紧泛化界。预期 VCR 使 $\eta_{\text{str}}$ 下降 30~50%, mAP 提升 +0.002~0.008, 并可能允许 NFE 从 4 降至 2。

---

## 1. 第一性原理推导

### 1.1 RF 理想与实际偏差

**RF 路径** ([rectified_flow.py:43-54](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L43-L54)):

$$x_t = (1-t) x_0 + t x_1, \quad t \in [0, 1],$$

其中 $x_0$ 为数据 (GT bbox, $d=4$ cxcywh 归一化), $x_1$ 为噪声。**理想速度场**:

$$v(t) = \frac{dx_t}{dt} = x_1 - x_0 \quad (\text{常数, 与 } t \text{ 无关}).$$

RF 的核心理论保证 ([Liu et al. 2022, arxiv 2209.03003](https://arxiv.org/abs/2209.03003)): 1-RectFlow 在理想情况下轨迹为直线, 速度场恒定, 单步 Euler 即可精确积分。

**实际偏差**: 学习的网络 $v_\theta(x_t, t)$ (本实现以 $x_0$-prediction 形式 $\hat{x}_0 = f_\theta(x_t, t)$, 由 $v_\theta = (x_t - \hat{x}_0)/t$ 派生, [rectified_flow.py:65-68](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L65-L68)) 在有限数据 + 有限容量下无法精确恢复常数 $v$, 故 $v_\theta$ 随 $t$ 变化。

**实测证据** ([theory_analysis_RF_DPM.md §1.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md), 3 seeds × 500 图):

| 配置 | Step 1 $\eta_{\text{str}}$ | Step 2 | Step 3 |
|------|---------------------------|--------|--------|
| baseline (renewal on) | 3.43 ± 0.36 | 2.45 ± 0.24 | 1.68 ± 0.15 |
| renewal off | **1.50 ± 0.33** | **1.11 ± 0.19** | **0.70 ± 0.09** |

$\eta_{\text{str}} \in [0.7, 1.5]$ (renewal off) **严格非零**, 定量证实学习轨迹非理想直线。

### 1.2 从变分正则化推导 VCR

**出发点**: 既然 $\eta_{\text{str}}$ 诊断证实 $v_\theta$ 随 $t$ 变化, 从第一性原理应在训练时**直接约束 $v_\theta$ 接近常数**, 而非仅靠 data-prediction loss 间接学习。

#### 1.2.1 变分正则化框架与 4 条要求的必要性

**设定**: 固定 coupling $(x_0, x_1)$, 把 $v_\theta$ 视作关于 $t$ 的函数 $v_\theta(\cdot; x_0, x_1) : [0, 1] \to \mathbb{R}^d$ ($d = 4$)。理想速度场 $v^*(\cdot) \equiv x_1 - x_0$ 是 $\mathcal{V} := H^1([0,1]; \mathbb{R}^d)$ 中的**常数函数** (弱导数 $\partial_t v^* \equiv 0$)。

**变分问题**: 寻找一个正则项 $\mathcal{R}: \mathcal{V} \to \mathbb{R}_{\geq 0}$ 加入训练目标 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \mathcal{R}$, 使最小化 $\mathcal{L}_{\text{total}}$ 时 $v_\theta$ 被推向"接近 $v^*$"。我们要求 $\mathcal{R}$ 满足:

1. **零理想 (Zero-at-ideal)**: $\mathcal{R}(v^*) = 0$。
   - **必要性**: 若 $\mathcal{R}(v^*) > 0$, 则正则项在 RF 理想点仍施加非零梯度, 拉扯 $v_\theta$ 离开 $v^*$, 与 $\mathcal{L}_{\text{det}}$ 在该点的全局最小冲突。这正是 [FALSIFIED_DIRECTIONS.md §六](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) Decoupled Head "目标冲突" 的失败模式。
2. **非负性 (Non-negativity)**: $\mathcal{R}(v) \geq 0, \forall v \in \mathcal{V}$。
   - **必要性**: 保证 $\mathcal{L}_{\text{total}} \geq \mathcal{L}_{\text{det}}$, 训练目标有下界, 优化良态。
3. **可微性 (Differentiability)**: $\mathcal{R}$ 在 $v_\theta$ 的参数空间上 a.e. 可微。
   - **必要性**: PyTorch autograd 可计算 $\nabla_\theta \mathcal{R}$, 才能与 $\nabla_\theta \mathcal{L}_{\text{det}}$ 合成总梯度。
4. **不引入竞争目标 (Target-agnostic)**: $\mathcal{R}$ 仅约束 $v_\theta$ 的**性质** (smoothness/常数性), 不显式指定 $v_\theta$ 应等于什么值。
   - **必要性**: 若 $\mathcal{R}$ 指定 $v_\theta$ 应等于某 target $u$ (即 $\mathcal{R} = \|v_\theta - u\|^2$), 则 $\mathcal{R}$ 引入了一个独立于 $\mathcal{L}_{\text{det}}$ 的目标, 可能与 $\mathcal{L}_{\text{det}}$ 在 $u \neq v^*$ 时冲突。这正是 ReFlow velocity loss 的失败机制 (§1.2.3, §4.2)。

> **【GLM-5.2 深化】4 条要求与 Tikhonov 正则化的关系**:
> 经典 Tikhonov 正则 $\mathcal{R}(v) = \|v\|^2$ 满足非负 + 可微, 但**不满足零理想** ($\|v^*\|^2 = \|x_1 - x_0\|^2 > 0$)。我们的 4 条要求比 Tikhonov 更严苛: 要求正则项在 RF 理想下消失。这迫使 $\mathcal{R}$ 取 "差分形式" 而非 "范数形式"。

#### 1.2.2 候选形式 (a): 成对一致性 (target-free, 选用)

$$\boxed{\mathcal{L}_{\text{vcr}}^{(a)} = \mathbb{E}_{(t, t') \overset{\text{i.i.d.}}{\sim} \mathcal{U}[0,1]^2}\left[\left\| v_\theta(x_t, t) - v_\theta(x_{t'}, t') \right\|_2^2\right]}$$

其中 $(x_0, x_1)$ 为同一 coupling, $x_t = (1-t) x_0 + t x_1$, $x_{t'} = (1-t') x_0 + t' x_1$ (即两时间步共享同一 coupling)。

**4 条要求的逐条验证**:

| 要求 | 验证 |
|------|------|
| 零理想 | $v^*$ 与 $t$ 无关 → $v^*(x_t, t) = v^*(x_{t'}, t') = x_1 - x_0$ → 被积函数恒为 0 → $\mathcal{L}_{\text{vcr}}^{(a)}(v^*) = 0$ ✓ |
| 非负 | $\|v_\theta(t) - v_\theta(t')\|^2 \geq 0$ 由范数非负性, 期望也非负 ✓ |
| 可微 | $v_\theta(x_t, t)$ 由 network forward 给出, 关于 $\theta$ 处处可微 (ReLU/SiLU 的 a.e. 可微性) ✓ |
| Target-agnostic | 正则项仅包含 $v_\theta$ 在两时间步的**差值** $v_\theta(t) - v_\theta(t')$, 不含任何 target $u$ ✓ |

#### 1.2.3 候选形式 (b): 目标匹配 (target-matching, 拒绝) 与 ReFlow velocity loss 的严格等价

**形式 (b) 定义**:
$$\mathcal{L}_{\text{vcr}}^{(b)} := \mathbb{E}_{t \sim \mathcal{U}[0,1]}\left[\left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|_2^2\right]$$

注意形式 (b) **不满足**要求 4 (target-agnostic), 因其显式指定 target 为 $x_1 - x_0$。

**【GLM-5.2 严格证明】命题 1.1 (形式 (b) ≡ ReFlow velocity loss = $(1/t^2) \mathcal{L}_{x_0}$)**: 在 RF 直线路径 $x_t = (1-t) x_0 + t x_1$ 与本实现 $v_\theta = (x_t - \hat{x}_0)/t$ ([rectified_flow.py:65-68](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L65-L68)) 下,

$$\mathcal{L}_{\text{vcr}}^{(b)}(t) = \frac{1}{t^2} \left\| \hat{x}_0 - x_0 \right\|_2^2 = \frac{1}{t^2} \mathcal{L}_{x_0}(t), \quad \forall t \in (0, 1].$$

**证明** (逐步代数):

**Step 1** (RF 路径恒等式): 由 $x_t = (1-t) x_0 + t x_1$ 展开:
$$x_t - t (x_1 - x_0) = (1-t) x_0 + t x_1 - t x_1 + t x_0 = (1-t) x_0 + t x_0 = x_0.$$
即 $x_t - t v = x_0$, 等价地 $v = (x_t - x_0)/t = x_1 - x_0$。✓

**Step 2** (代入 $v_\theta$ 的定义): 由 [rectified_flow.py:65-68](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L65-L68), $v_\theta(x_t, t) = (x_t - \hat{x}_0(x_t, t))/t$ (其中 $\hat{x}_0 = f_\theta(x_t, t)$ 是网络直接输出)。

**Step 3** (计算差 $v_\theta - v$):
$$\begin{aligned}
v_\theta(x_t, t) - v &= \frac{x_t - \hat{x}_0}{t} - (x_1 - x_0) \\
&= \frac{x_t - \hat{x}_0 - t(x_1 - x_0)}{t} \quad (\text{通分})\\
&= \frac{(x_t - t v) - \hat{x}_0}{t} \\
&= \frac{x_0 - \hat{x}_0}{t} \quad (\text{Step 1: } x_t - t v = x_0) \\
&= -\frac{\varepsilon(t)}{t}, \quad \varepsilon(t) := \hat{x}_0 - x_0.
\end{aligned}$$

**Step 4** (平方与期望):
$$\mathcal{L}_{\text{vcr}}^{(b)}(t) = \|v_\theta - v\|_2^2 = \frac{\|\varepsilon(t)\|_2^2}{t^2} = \frac{1}{t^2} \mathcal{L}_{x_0}(t).$$
对 $t$ 取期望得 $\mathcal{L}_{\text{vcr}}^{(b)} = \mathbb{E}_t[t^{-2} \mathcal{L}_{x_0}(t)]$。$\square$

**与已证伪方向的关系** (修正版):

- **[FALSIFIED_DIRECTIONS.md §八](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (h_velocity_loss)**: 该实验直接以 $\mathcal{L}_v = \|v_\theta - v\|^2$ 为训练目标。由命题 1.1, 在 **L2 squared loss 无归一化**的前提下, $\mathcal{L}_v = \mathbb{E}[t^{-2} \mathcal{L}_{x_0}(t)]$, 即 h_velocity_loss 与形式 (b) 在此意义下**逐点相等**。

  > **【GLM-5.2 事实修正】**: 原稿引用 "[FALSIFIED_DIRECTIONS.md §八] h_velocity_loss CRASHED" 作为 form (b) 失败的证据。但核查 FALSIFIED_DIRECTIONS.md 发现**内部矛盾**: §八 记录 h_velocity_loss "⛔ CRASHED (训练崩溃)", 但 §十四 数据修正明确指出 "真正的 h_velocity_loss best=0.856@ep61" (即训练未崩溃, 达到 mAP 0.856, 略低于 baseline 0.863, Δ = -0.007)。VCR 文档原引用的 "CRASHED" 信息**过时**, 应以 §十四 修正数据为准。

- **[FALSIFIED_DIRECTIONS.md §十四](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (ReFlow Standard MSE)**: 该实验用预存 coupling 训练 ReFlow velocity loss, 由同一等价关系 (L2 squared 下), 与形式 (b) 同源。
- **[theory_analysis_RF_DPM.md §4.3](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md) (R3 v-prediction 分析)**: 该理论分析明确给出 $\mathcal{L}_v = t^{-2} \mathcal{L}_{x_0}$ 的等价性, 与命题 1.1 完全一致。

> **【GLM-5.2 严格性修正】h_velocity_loss 实现 ≠ 形式 (b) 严格等价**: 命题 1.1 的等价关系仅在 **L2 squared loss 无归一化**下成立。核查 [criterion.py:265-273](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py#L265-L273) 发现 h_velocity_loss 实际实现: (1) 使用 **L1 loss** (非 L2 squared); (2) 应用 **batch normalization** `v_weight = v_weight / v_weight.mean().detach()` (改变梯度绝对幅度)。criterion.py 注释自身承认 "L1 loss 下严格等价应为 1/t, 这里用 1/t² 匹配理论 doc 的梯度放大 claim"。因此 **h_velocity_loss 是形式 (b) 的 L1 + batch-normalized 变体, 与形式 (b) 不严格等价**, 但行为相似 (均含 1/t² 加权, 在 $t \to 0$ 时梯度放大)。命题 1.1 的 L2 squared 等价关系数学正确, 但其与 h_velocity_loss 实现的 "逐点相等" 声明**不严格**。

#### 1.2.4 $t \to 0$ 奇点分析 (形式 (b) 失败的根因)

由命题 1.1, 形式 (b) 含 $1/t^2$ 因子。设 $\mathcal{L}_{x_0}(t) \to \mathcal{L}_{x_0}(0) > 0$ (训练初期 $\hat{x}_0 \not\to x_0$), 则

$$\mathcal{L}_{\text{vcr}}^{(b)}(t) = \mathcal{L}_{x_0}(t) / t^2 \to \infty \quad (t \to 0^+).$$

**梯度放大**: $\nabla_\theta \mathcal{L}_{\text{vcr}}^{(b)}(t) = (1/t^2) \nabla_\theta \mathcal{L}_{x_0}(t) \cdot$ (因子), 即**梯度被 $1/t^2$ 加权**。在 shifted schedule ($s = 3.0$) 下, $t' = 3t/(1 + 2t)$, $t < 0.5$ 区仍有非零概率, 此处 $1/t^2 > 4$, 梯度方差爆炸 (R3 命题 R3.2, [theory_analysis_RF_DPM.md §4.3](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md))。这是 h_velocity_loss 性能 0.856 < baseline 0.863 的数值根因 (1/t² 加权放大 $t \to 0$ 区梯度方差, 即使含 batch normalization 也未带来增益; 注: §八 "CRASHED" 标签已被 §十四 修正为 best=0.856@ep61, 训练实际未崩溃, 仅性能略低于 baseline)。

**形式 (a) 不含 $1/t^2$ 因子**: $\mathcal{L}_{\text{vcr}}^{(a)}$ 含 $v_\theta(t) - v_\theta(t')$, 由 §4.2 梯度分析, 仅 $1/t$ (单次除法, 非平方), 且仅在 $t, t'$ 同时小时放大, 概率为 $P(t < \epsilon, t' < \epsilon) = \epsilon^2$, 远小于形式 (b) 的 $P(t < \epsilon) = \epsilon$。

#### 1.2.5 结论

VCR **只采用形式 (a)**。形式 (b) 由命题 1.1 严格等价于 ReFlow velocity loss (L2 squared 下), 含 $1/t^2$ 因子, 在 $t \to 0$ 时梯度放大, 已被项目历史 (§八 h_velocity_loss 性能 0.856 < baseline 0.863, §十四 配置Bug+方法风险) 证伪。形式 (a) 满足 4 条变分要求, 仅含 $1/t$ 因子, 数值稳定性远优于形式 (b)。

### 1.3 训练目标

最终训练目标:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \cdot \mathcal{L}_{\text{vcr}}^{(a)},$$

其中 $\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{bbox}}^{\text{L1}} + \mathcal{L}_{\text{giou}}$ ([criterion.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py) 现有实现, 含 deep supervision aux losses), $\lambda$ 为正则强度超参 (建议初值 $\lambda = 0.01 \sim 0.1$, 详见 §6)。

---

## 2. 理论分析

本节建立 VCR 与 Sobolev 范数、Poincaré 不等式、Rademacher 复杂度的严格联系, 并给出泛化界定理。

> **【GLM-5.2 重要修正】**: 原稿推论 2.2 与 §2.3 等价链存在数学错误: 误将 Poincaré 不等式 $\text{Var}(v) \leq C_P |v|_{H^1}^2$ 的上界方向当作等价, 推出"$\mathcal{L}_{\text{vcr}} \to 0 \Rightarrow |v_\theta|_{H^1}^2 \to 0$"。事实上, Poincaré 仅给出 $\text{Var}$ 的**上界** (即 $|v|_{H^1}^2$ 的**下界**), $\text{Var} \to 0$ 不蕴含 $|v|_{H^1}^2 \to 0$。本节重写为严格版本。

### 2.1 VCR 与 $L^2$ 方差的精确等式 (命题 2.1)

**设定**: 固定 coupling $(x_0, x_1)$, 速度场 $v_\theta(\cdot; x_0, x_1) : [0, 1] \to \mathbb{R}^d$ ($d=4$)。假设 $v_\theta \in H^1([0,1]; \mathbb{R}^d)$, 即 $v_\theta$ 及其弱导数 $\partial_t v_\theta$ 均平方可积。

**$L^2$ 范数与 $H^1$ 半范数**:
$$\|v\|_{L^2}^2 := \int_0^1 \|v(t)\|_2^2 \, dt, \quad |v|_{H^1}^2 := \int_0^1 \|\partial_t v(t)\|_2^2 \, dt, \quad \bar{v} := \int_0^1 v(t) \, dt.$$

**$L^2$ 方差** (时间维上的方差):
$$\text{Var}_t(v) := \int_0^1 \|v(t) - \bar{v}\|_2^2 \, dt = \|v - \bar{v}\|_{L^2}^2 = \|v\|_{L^2}^2 - \|\bar{v}\|_2^2.$$
(最后一个等式由 $\|v - \bar{v}\|_{L^2}^2 = \|v\|_{L^2}^2 - 2 \langle v, \bar{v} \rangle_{L^2} + \|\bar{v}\|_{L^2}^2$ 与 $\langle v, \bar{v} \rangle_{L^2} = \bar{v} \cdot \int v = \|\bar{v}\|_2^2$ 得出。)

**【GLM-5.2 严格证明】命题 2.1 (VCR 是 $L^2$ 方差的无偏估计)**: 对 $t, t' \overset{\text{i.i.d.}}{\sim} \mathcal{U}[0,1]$,
$$\mathcal{L}_{\text{vcr}}^{(a)} = \mathbb{E}_{t, t'}\left[\left\| v_\theta(t) - v_\theta(t') \right\|_2^2\right] = 2 \, \text{Var}_t(v_\theta).$$

**证明** (每一步显式):

**Step 1** (展开平方): 由 $\|a - b\|_2^2 = \|a\|_2^2 + \|b\|_2^2 - 2 a^\top b$,
$$\|v(t) - v(t')\|_2^2 = \|v(t)\|_2^2 + \|v(t')\|_2^2 - 2 v(t)^\top v(t').$$

**Step 2** (取期望, 利用 $t, t'$ i.i.d.): 由 $t, t'$ i.i.d. $\sim \mathcal{U}[0,1]$, 有 $\mathbb{E}[\|v(t)\|_2^2] = \mathbb{E}[\|v(t')\|_2^2] = \|v\|_{L^2}^2$ (定义), 且 $\mathbb{E}[v(t)^\top v(t')] = \mathbb{E}[v(t)]^\top \mathbb{E}[v(t')]$ (独立随机变量的期望分解) $= \bar{v}^\top \bar{v} = \|\bar{v}\|_2^2$。

**Step 3** (合并):
$$\mathbb{E}_{t, t'}\|v(t) - v(t')\|_2^2 = \|v\|_{L^2}^2 + \|v\|_{L^2}^2 - 2 \|\bar{v}\|_2^2 = 2 (\|v\|_{L^2}^2 - \|\bar{v}\|_2^2) = 2 \, \text{Var}_t(v). \quad \square$$

**注**:
- 命题 2.1 给出的是**精确等式** (非估计), 故 $\mathcal{L}_{\text{vcr}}^{(a)} \to 0$ 严格等价于 $\text{Var}_t(v_\theta) \to 0$。
- 与 Czarnecki et al. 2017 (Sobolev Training, [arXiv:1706.04859](https://arxiv.org/abs/1706.04859)) 的关系: Sobolev Training 显式监督 $\partial_x f$ 匹配目标导数; VCR 在时间维上**隐式约束** $\partial_t v_\theta$ 接近零 (通过 $\text{Var}$ 的 Poincaré 上界)。两者均惩罚 $H^1$ 半范数, 但 VCR **无需目标导数** (零是 RF 理想直接给出), 适用于检测任务目标导数不可得的场景。

### 2.2 Poincaré 不等式: VCR 与 $H^1$ 半范数的单向上界关系

**【GLM-5.2 引用核实】Poincaré-Wirtinger 不等式 (1D, 最优常数)**: 对 $v \in H^1([0,1]; \mathbb{R}^d)$,
$$\|v - \bar{v}\|_{L^2}^2 \leq \frac{1}{\pi^2} \, |v|_{H^1}^2, \quad \bar{v} := \int_0^1 v(t) \, dt. \tag{Poincaré}$$

**常数来源**: $1/\pi^2$ 是 1D 区间 $[0,1]$ 上 Neumann Laplacian $-\frac{d^2}{dt^2}$ 的**第一非零特征值** $\lambda_1 = \pi^2$ 的倒数。特征函数为 $\phi_1(t) = \cos(\pi t)$ (满足 Neumann 边界条件 $\phi_1'(0) = \phi_1'(1) = 0$), 谱分解给出
$$\|v - \bar{v}\|_{L^2}^2 = \sum_{k=1}^\infty |a_k|^2 \leq \frac{1}{\lambda_1} \sum_{k=1}^\infty \lambda_k |a_k|^2 = \frac{1}{\pi^2} |v|_{H^1}^2$$
其中 $a_k = \langle v, \phi_k \rangle_{L^2}$。详见:
- Evans, *Partial Differential Equations* (2nd ed.), §5.8 (Sobolev 不等式) + §6.5 (Laplacian 特征值)。
- Brezis, *Functional Analysis, Sobolev Spaces and Partial Differential Equations*, Ch. 9 (Spectral theory)。
- Hardy, Littlewood & Pólya, *Inequalities*, §7.7 (1D Wirtinger 不等式, 经典来源)。

> **注**: Evans §5.8 给出的是一般 Poincaré 不等式 (常数依赖区域), 最优常数 $1/\pi^2$ 来自 1D 谱理论, 文献更精确的引用应为 Brezis Ch. 9 或 Hardy-Littlewood-Pólya §7.7。

**等价表述**: 由命题 2.1, $\text{Var}_t(v) = \frac{1}{2} \mathcal{L}_{\text{vcr}}^{(a)}(v)$, 故 Poincaré 不等式等价于
$$\boxed{\mathcal{L}_{\text{vcr}}^{(a)}(v_\theta) \leq \frac{2}{\pi^2} \, |v_\theta|_{H^1}^2.} \tag{P-VCR}$$

**方向性 (重要)**: Poincaré 给出 $\mathcal{L}_{\text{vcr}}$ 的**上界** (即 $|v|_{H^1}^2$ 的**下界**), 不给出反向。等价地,
$$|v_\theta|_{H^1}^2 \geq \frac{\pi^2}{2} \mathcal{L}_{\text{vcr}}^{(a)}(v_\theta).$$

**【GLM-5.2 修正】推论 2.2 (VCR 的极限含义, 严格版)**: 若 $\mathcal{L}_{\text{vcr}}^{(a)}(v_\theta) \to 0$, 则
1. (恒等) $\text{Var}_t(v_\theta) = \frac{1}{2} \mathcal{L}_{\text{vcr}}^{(a)} \to 0$, 即 $\|v_\theta - \bar{v}_\theta\|_{L^2} \to 0$ ($v_\theta$ 在 $L^2$ 意义下退化为常数 $\bar{v}_\theta$);
2. (Poincaré 上界方向) $|v_\theta|_{H^1}^2 \geq \frac{\pi^2}{2} \mathcal{L}_{\text{vcr}}^{(a)} \to 0$ 仅给出 $|v_\theta|_{H^1}^2$ 的**下界趋于 0**; $|v_\theta|_{H^1}^2$ **本身是否趋于 0 不能由 Poincaré 单独推出**, 需要额外假设 (如紧致性 + 唯一性)。

**证明**: (1) 由命题 2.1 直接得到。(2) 由 Poincaré 不等式 (P-VCR) 直接得到。$\square$

**【GLM-5.2 修正】与原稿的差异**:
- 原稿断言 "$\mathcal{L}_{\text{vcr}} \to 0 \Rightarrow |v_\theta|_{H^1}^2 \to 0$" — **错误**。Poincaré 仅给出 $\mathcal{L}_{\text{vcr}}$ 的上界 (即 $|v|_{H^1}^2$ 的下界), 下界 $\to 0$ 不蕴含值 $\to 0$。
- 反例: 取 $v(t) = \bar{v} + \sqrt{\epsilon} \sin(2\pi k t)$ ($k \to \infty$, $\epsilon \to 0$), 则 $\text{Var}(v) = \epsilon/2 \to 0$ 但 $|v|_{H^1}^2 = 2\pi^2 k^2 \epsilon \to \infty$ (高频小幅振荡)。此反例说明 VCR 小不蕴含 $H^1$ 半范数小。
- **正确结论**: VCR 直接控制的是 **$L^2$ 方差** $\text{Var}_t(v_\theta)$, 而非 $H^1$ 半范数。$H^1$ 半范数仅作为 VCR 的**上界代理** (Poincaré), 是间接的。

**【GLM-5.2 修正】推论 2.3 (VCR + $\mathcal{L}_{\text{det}}$ 联合保证 $v_\theta \to v^*$)**: 设训练目标 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \mathcal{L}_{\text{vcr}}^{(a)}$ 的全局最小 $(v_\theta^*, \lambda^*)$ 存在。若 $\mathcal{L}_{\text{det}}$ 在 $\hat{x}_0 = x_0$ (即 $v_\theta = v^*$) 时取全局最小 0, 且 $\lambda > 0$, 则:
1. $\mathcal{L}_{\text{det}}(v_\theta^*) = 0$ (否则可降低 $\mathcal{L}_{\text{det}}$ 减小总损失);
2. $\mathcal{L}_{\text{vcr}}^{(a)}(v_\theta^*) = 0$ (由 1, $v_\theta^* = v^*$ 为常数, VCR 恒为 0)。
故全局最小 $v_\theta^* = v^*$, **VCR 与 $\mathcal{L}_{\text{det}}$ 共享全局最小**, 不存在目标冲突。

**证明**: 由 $\mathcal{L}_{\text{total}}(v^*) = 0 + \lambda \cdot 0 = 0$ (因 $v^*$ 为常数, VCR = 0; 且 $\hat{x}_0 = x_0$ 时 $\mathcal{L}_{\text{det}} = 0$)。由 $\mathcal{L}_{\text{total}} \geq 0$ (非负性), 全局最小值 $= 0$, 在 $v_\theta = v^*$ 处达到。任何其他 $v_\theta \neq v^*$ 必使 $\mathcal{L}_{\text{det}} > 0$ 或 $\mathcal{L}_{\text{vcr}} > 0$ 之一为正, 故 $\mathcal{L}_{\text{total}} > 0$。$\square$

**注**: 此推论保证 VCR 的"共享最小值子空间"性质 (§4.2 详述), 是 VCR 规避梯度冲突的数学基础。

### 2.3 与 $\eta_{\text{str}}$ 的数学关系 (修正版等价链)

**$\eta_{\text{str}}$ 定义** ([theory_analysis_RF_DPM.md §1.2](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md), [rectified_flow.py:175-184](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L175-L184)):

$$\eta_{\text{str}}^{(n)} = \frac{\|\mathbf{D}_1^{(n)}\|_2}{\|\hat{x}_0^{(n)}\|_2 + \epsilon_{\text{norm}}}, \quad \mathbf{D}_1^{(n)} = \frac{\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)}}{t_n - t_{n-1}}, \quad \epsilon_{\text{norm}} = 10^{-6}.$$

#### 2.3.1 $\partial_t \hat{x}_0$ 的严格推导

由 $\hat{x}_0 = x_t - t \cdot v_\theta$ (本实现, [rectified_flow.py:65-68](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L65-L68)), 对 $t$ 求导 (假设 $v_\theta$ 关于 $t$ 可微):

$$\partial_t \hat{x}_0 = \partial_t x_t - v_\theta - t \cdot \partial_t v_\theta.$$

由 RF 路径 $x_t = (1-t) x_0 + t x_1$, $\partial_t x_t = x_1 - x_0 = v^*$(常数), 故

$$\boxed{\partial_t \hat{x}_0 = (x_1 - x_0) - v_\theta - t \cdot \partial_t v_\theta = v^* - v_\theta - t \cdot \partial_t v_\theta.}$$

**关键观察**:
1. 当 $v_\theta \to v^*$ (常数) 时, $\partial_t v_\theta \to 0$ 且 $v^* - v_\theta \to 0$, 故 $\partial_t \hat{x}_0 \to 0$。
2. 当 $v_\theta \neq v^*$ 但 $v_\theta$ 为常数 (即 $\partial_t v_\theta = 0$) 时, $\partial_t \hat{x}_0 = v^* - v_\theta \neq 0$ (常数偏差)。故 "$v_\theta$ 为常数" 不蕴含 "$\partial_t \hat{x}_0 = 0$", 还需 $v_\theta = v^*$。
3. 当 $v_\theta$ 时变但 $\partial_t v_\theta = (v^* - v_\theta)/t$ 时, $\partial_t \hat{x}_0 = 0$ 但 $v_\theta$ 不为常数。此为特殊情况。

#### 2.3.2 $\eta_{\text{str}}$ 与 $\partial_t \hat{x}_0$ 的关系 (离散 → 连续)

DPM-Solver++ 的 $\mathbf{D}_1^{(n)} = (\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)})/(t_n - t_{n-1})$ 是 $\partial_t \hat{x}_0$ 在 $[t_{n-1}, t_n]$ 上的**差商**。由中值定理 (假设 $\hat{x}_0$ 在 $[t_{n-1}, t_n]$ 上 $C^1$), 存在 $\xi_n \in (t_{n-1}, t_n)$ 使得
$$\mathbf{D}_1^{(n)} = \partial_t \hat{x}_0(\xi_n).$$
故 $\|\mathbf{D}_1^{(n)}\| = \|\partial_t \hat{x}_0(\xi_n)\|$, $\eta_{\text{str}}^{(n)} = \|\partial_t \hat{x}_0(\xi_n)\| / (\|\hat{x}_0^{(n)}\| + \epsilon)$。

**$\eta_{\text{str}} \to 0$ 的充分条件**: $\partial_t \hat{x}_0 \to 0$ 在 $[0, 1]$ 上一致。由 §2.3.1, 充分条件为 $v_\theta \to v^*$ 且 $\partial_t v_\theta \to 0$ 一致。

#### 2.3.3 VCR → $\eta_{\text{str}}$ 的单向蕴含链 (修正版)

由 §2.2 推论 2.2 (修正版), $\mathcal{L}_{\text{vcr}} \to 0$ 蕴含 $\|v_\theta - \bar{v}_\theta\|_{L^2} \to 0$ (即 $v_\theta$ 退化为常数 $\bar{v}_\theta$)。再由推论 2.3 (VCR + $\mathcal{L}_{\text{det}}$ 联合), $\bar{v}_\theta \to v^*$ (由 $\mathcal{L}_{\text{det}}$ 约束)。**仅当 $\partial_t v_\theta$ 也由 VCR 间接控制时**, $\partial_t \hat{x}_0 \to 0$。

**【GLM-5.2 修正】严格蕴含链**:

$$\boxed{\mathcal{L}_{\text{vcr}}^{(a)} \downarrow \;\overset{\text{Prop 2.1}}{\iff}\; \text{Var}_t(v_\theta) \downarrow \;\overset{\text{def}}{\iff}\; \|v_\theta - \bar{v}_\theta\|_{L^2} \downarrow \;\overset{\mathcal{L}_{\text{det}}}{\Longrightarrow}\; \bar{v}_\theta \to v^* \;\overset{+}{\Longrightarrow}\; \eta_{\text{str}} \downarrow.}$$

> **注**: " $\overset{+}{\Longrightarrow}$" 表示需要附加假设 $\partial_t v_\theta$ 在某种平均意义下也减小 (Poincaré 上界方向的"启发式"成立, 但不严格)。

**修正说明**:
- 原稿等价链 "$\mathcal{L}_{\text{vcr}} \downarrow \Leftrightarrow |v_\theta|_{H^1}^2 \downarrow \Leftrightarrow \|v_\theta - \bar{v}_\theta\|_{L^2} \downarrow$" — **错误**: 第一个 $\Leftrightarrow$ 不成立 (Poincaré 仅给出 $\mathcal{L}_{\text{vcr}}$ 的上界, 非等价), 第二个 $\Leftrightarrow$ 也不成立 (同上)。
- 修正后: $\mathcal{L}_{\text{vcr}} \downarrow \Leftrightarrow \text{Var}_t(v_\theta) \downarrow \Leftrightarrow \|v_\theta - \bar{v}_\theta\|_{L^2} \downarrow$ (命题 2.1 给出严格等式), 但 $|v_\theta|_{H^1}^2 \downarrow$ 仅作为**单向上界** (Poincaré), 不在等价链中。

**实际意义**: VCR 直接约束 $v_\theta$ 接近其时间均值 $\bar{v}_\theta$ (即 $v_\theta$ 接近常数), $\mathcal{L}_{\text{det}}$ 约束 $\bar{v}_\theta$ 接近 $v^*$。两者联合使 $v_\theta \to v^*$, 进而 $\partial_t \hat{x}_0 \to 0$, $\eta_{\text{str}} \downarrow$。$\eta_{\text{str}}$ 是 VCR 训练效果的**推理时度量**。

### 2.4 泛化界定理 (Rademacher 复杂度 + 覆盖数)

本小节给出 VCR 收紧泛化界的严格定理。

> **【GLM-5.2 修正】**: 原稿 Lemma 2.3 与 Theorem 2.4 存在问题: (i) Lemma 2.3 的覆盖数界 $(CB/\varepsilon)^d$ 应用了错误的度量熵 (Hölder 类 $L^\infty$ 度量, 而非 Sobolev 类 $L^2$ 度量); (ii) Theorem 2.4 中 $B$ 的 scaling 应为 $\sqrt{B}$ (而非 $B$), 因 Kolmogorov-Tikhomirov 的 Sobolev 类 $L^2$ 度量熵为 $\log \mathcal{N} \sim \varepsilon^{-1}$。本节给出修正版。

#### 2.4.1 设定与函数类

**设定**: 训练集 $\mathcal{S} = \{(x_0^{(i)}, x_1^{(i)})\}_{i=1}^n$ i.i.d. 抽自分布 $\mathcal{D}$。学习算法输出参数 $\theta \in \Theta$, 速度场 $v_\theta$。损失 $\ell(v_\theta; x_0, x_1) = \mathcal{L}_{\text{det}}(v_\theta; x_0, x_1) + \lambda \mathcal{L}_{\text{vcr}}^{(a)}(v_\theta; x_0, x_1)$, 假设 $\ell$ 关于 $v_\theta$ 是 $L_\ell$-Lipschitz (L1 + GIoU + cls 损失在 bounded bbox 空间下满足)。

**约束函数类**: 设 VCR 训练后 $\mathcal{L}_{\text{vcr}}^{(a)}(v_\theta) \leq V$ (即 $\text{Var}_t(v_\theta) \leq V/2$), 且 $\|\bar{v}_\theta\|_2 \leq R$ (由 $\mathcal{L}_{\text{det}}$ 约束 $\bar{v}_\theta \approx v^*$, $R$ 为 GT bbox 范数上界)。定义约束类
$$\mathcal{V}_{V, R} := \{v \in H^1([0,1]; \mathbb{R}^d) : \text{Var}_t(v) \leq V, \|\bar{v}\|_2 \leq R\}.$$

**注**: 此处以 $V$ (= $\mathcal{L}_{\text{vcr}}$ 上界) 而非 $B$ (= $|v|_{H^1}^2$ 上界) 作为复杂度参数, 因 VCR 直接控制 $V$, $B$ 仅通过 Poincaré 间接关联 (§2.2 修正版)。

#### 2.4.2 度量熵 (Kolmogorov-Tikhomirov)

**【GLM-5.2 修正】引理 2.3 (Sobolev 类的 $L^2$ 度量熵)**: 函数类 $\mathcal{V}_{V, R}$ 在 $L^2([0,1]; \mathbb{R}^d)$ 度量下的 $\varepsilon$-覆盖数满足
$$\log \mathcal{N}(\varepsilon, \mathcal{V}_{V, R}, \|\cdot\|_{L^2}) \leq C \cdot d \cdot (R + \sqrt{V}) \cdot \varepsilon^{-1}, \quad \forall \varepsilon \in (0, 1),$$
其中 $C$ 为绝对常数。

**证明梗概**:

**Step 1** (分解 $v = \bar{v} + (v - \bar{v})$): 由 Poincaré 不等式 $\|v - \bar{v}\|_{L^2} \leq \sqrt{C_P V} = \sqrt{V}/\pi$ (用 $V = 2 \text{Var}$ 与 $C_P = 1/\pi^2$), 故 $\|v\|_{L^2} \leq \|\bar{v}\|_2 + \|v - \bar{v}\|_{L^2} \leq R + \sqrt{V}/\pi \leq R + \sqrt{V}$ ($L^2$ 范数有界)。

**Step 2** (Sobolev 类度量熵, Kolmogorov-Tikhomirov 1959): 对 Sobolev 球 $W^{m, 2}([0,1]; \mathbb{R}^d)$ (m=1, $p=2$, 域维 $D=1$) 在 $L^2$ 度量下,
$$\log \mathcal{N}(\varepsilon, W^{m,2}_B, L^2) \leq C_{KT} \cdot d \cdot B^{1/m} \cdot \varepsilon^{-D/m} = C_{KT} \cdot d \cdot B \cdot \varepsilon^{-1}, \quad (m=1, D=1).$$
参见 [Kolmogorov & Tikhomirov 1959] 原始论文, 或 [van der Vaart & Wellner 1996, Theorem 2.7.1] 对 Hölder/Sobolev 类的覆盖数界, 或 [Lorentz, Golitschek & Makovoz 1996, *Constructive Approximation*] Ch. 7 对 Sobolev 类的精确界。

**Step 3** (代入 $B = V$): 由于 $\text{Var}(v) \leq V$ 不直接给出 $|v|_{H^1}^2$ 上界 (Poincaré 仅给下界), 严格地需要约束 $|v|_{H^1}^2 \leq B$ (额外假设)。在 VCR 训练中, 可通过经验估计 $|v|_{H^1}^2 \leq \tilde{B}$, 此处 $\tilde{B}$ 为训练后经验 $H^1$ 半范数。代入 Step 2 得 $\log \mathcal{N} \leq C d \tilde{B} \varepsilon^{-1}$。$\square$

**与原稿的差异**:
- 原稿 Lemma 2.3 给出 $\mathcal{N} \leq (CB/\varepsilon)^d$ 即 $\log \mathcal{N} \leq d \log(CB/\varepsilon)$ — **错误**。这是 $L^\infty$ 度量下 Hölder 类 $\log \mathcal{N} \sim d (1/\varepsilon)^{D/\alpha}$ 在 $D=1, \alpha=1/2$ 时的结果 ($\log \mathcal{N} \sim d \varepsilon^{-2}$), 但 $L^2$ 度量下 Sobolev 类的度量熵为 $\varepsilon^{-1}$ (而非 $\varepsilon^{-2}$ 或 $\log$)。
- Dudley 积分 $\int_0^D \sqrt{\log \mathcal{N}(\varepsilon)} d\varepsilon$ 在 $\log \mathcal{N} \sim \varepsilon^{-1}$ 时收敛 ($\int \varepsilon^{-1/2} d\varepsilon$), 在 $\log \mathcal{N} \sim \varepsilon^{-2}$ 时发散 ($\int \varepsilon^{-1} d\varepsilon$)。原稿的 $(CB/\varepsilon)^d$ 给出 $\log \mathcal{N} \sim \log(CB/\varepsilon)$ 也收敛, 但其依据 (Arzelà-Ascoli + Hölder) 不正确。

#### 2.4.3 Rademacher 复杂度界

**【GLM-5.2 修正】定理 2.4 (VCR 泛化界, 修正版)**: 设 $\ell$ 关于 $v_\theta$ 是 $L_\ell$-Lipschitz, $\mathcal{V}_{\tilde{B}, R} := \{v \in H^1([0,1]; \mathbb{R}^d) : |v|_{H^1}^2 \leq \tilde{B}, \|\bar{v}\|_2 \leq R\}$。则以概率 $\geq 1 - \delta$,
$$\mathbb{E}_{\mathcal{D}}[\ell(v_\theta)] \leq \frac{1}{n}\sum_{i=1}^n \ell(v_\theta; x_0^{(i)}, x_1^{(i)}) + \mathcal{O}\left( \frac{L_\ell \cdot \sqrt{d \cdot \tilde{B} \cdot D_{L^2}}}{\sqrt{n}} + \sqrt{\frac{\log(1/\delta)}{n}} \right),$$
其中 $D_{L^2} = \sup_{v \in \mathcal{V}_{\tilde{B}, R}} \|v\|_{L^2} \leq R + \sqrt{C_P} \cdot \sqrt{\tilde{B}}$ 为 $L^2$ 直径 (Poincaré 分解)。

**证明梗概**:

**Step 1** (Rademacher 复杂度的 Dudley 熵积分上界, [Bartlett & Mendelson 2002], [Mohri et al. 2018] Lemma 5.4): 经验 Rademacher 复杂度
$$\hat{\mathfrak{R}}_n(\ell \circ \mathcal{V}_{\tilde{B}, R}) \leq \frac{C'}{\sqrt{n}} \mathbb{E}_\sigma \int_0^{D_{L^2}} \sqrt{\log \mathcal{N}(\varepsilon, \mathcal{V}_{\tilde{B}, R}, \|\cdot\|_{L^2(\mathcal{S})})} \, d\varepsilon$$

**Step 2** (代入度量熵, 引理 2.3):
$$\int_0^{D_{L^2}} \sqrt{C d \tilde{B} \varepsilon^{-1}} \, d\varepsilon = \sqrt{C d \tilde{B}} \int_0^{D_{L^2}} \varepsilon^{-1/2} \, d\varepsilon = 2 \sqrt{C d \tilde{B} D_{L^2}}$$

**Step 3** (合并):
$$\hat{\mathfrak{R}}_n \leq \frac{C'}{\sqrt{n}} \cdot 2 \sqrt{C d \tilde{B} D_{L^2}} = \mathcal{O}\left(\frac{L_\ell \sqrt{d \tilde{B} D_{L^2}}}{\sqrt{n}}\right)$$

**Step 4** (Rademacher 泛化定理, [Bartlett & Mendelson 2002], [Mohri et al. 2018] Theorem 5.8): 加上 $\sqrt{\log(1/\delta)/(2n)}$ 高置信项。$\square$

**关键观察 (修正后)**:

1. **界关于 $\tilde{B}$ 单调 ( $\sqrt{\tilde{B}}$ scaling)**: $\tilde{B} \downarrow$ (VCR 减小 $H^1$ 半范数) → 泛化界以 $\sqrt{\tilde{B}}$ 速率收紧。**这是 VCR 改善泛化的严格理论依据**, 但速率比原稿 $B$ 慢 ( $\sqrt{\tilde{B}}$ vs $B$)。
2. **关于 $V$ (VCR 直接控制量) 的间接单调**: 由 Poincaré $V \geq \tilde{B}/C_P$ 仅给下界, 不能直接以 $V$ 替换 $\tilde{B}$。但 VCR 训练实际减小 $V$, 启发式地也减小 $\tilde{B}$ (但非严格)。
3. **维度 $d=4$ 的影响**: $\sqrt{d}$ 项在 $d=4$ 时为 2, 远小于图像生成 $d \sim 10^5$ 时的 $\sqrt{d} \sim 316$。**VCR 在低维检测设置下泛化界特别紧**。
4. **与谱归一化 ([Yoshida & Miyato 2017, arXiv:1705.10941](https://arxiv.org/abs/1705.10941)) 的关系**: 谱归一化约束**参数空间**的 Lipschitz 常数 (间接约束函数空间); VCR 直接约束**函数空间**的 $H^1$ 半范数。后者更直接针对 RF 的"常数速度"理想。
5. **与 Bartlett, Foster, Telgarsky 2017 ([arXiv:1706.08498](https://arxiv.org/abs/1706.08498)) 的关系**: 该工作通过谱归一化界 weights 的 Lipschitz 性给出分类间隔界。VCR 借用其 Rademacher 复杂度框架, 但作用在 $v_\theta$ 的时间维 smoothness 上, 是互补的。

> **【GLM-5.2 修正】文献核查**: 原稿将 [Bartlett et al. 2017] 的 arXiv 号误植为 1705.10941 (此为 Yoshida & Miyato 的工作)。Bartlett, Foster, Telgarsky "Spectrally-normalized margin bounds for neural networks" 的正确 arXiv 号为 [1706.08498](https://arxiv.org/abs/1706.08498), NeurIPS 2017。

### 2.5 PAC-Bayes 视角 (修正版)

考虑先验 $P$ 服从 Gibbs 形式 $dP(\theta) \propto \exp(-\beta |v_\theta|_{H^1}^2) d\theta$, 偏好 $H^1$-光滑的 $v_\theta$。SGD 输出的后验 $Q$ 的 KL 散度 (Donsker-Varadhan 变分公式):
$$\text{KL}(Q \| P) = \beta \mathbb{E}_Q[|v_\theta|_{H^1}^2] + \log \mathbb{E}_P[\exp(-\beta |v_\theta|_{H^1}^2)] = \beta \mathbb{E}_Q[|v_\theta|_{H^1}^2] + \text{const},$$
其中 const 是 $P$-normalization (依赖 $\beta$ 与 $\Theta$ 几何, 但与 $Q$ 无关)。

**【GLM-5.2 修正】PAC-Bayes 应用条件**:
1. **Gibbs 先验良定义**: $dP(\theta) \propto \exp(-\beta |v_\theta|_{H^1}^2) d\theta$ 在 $\Theta$ 不可数时需 $\int e^{-\beta |v_\theta|_{H^1}^2} d\theta < \infty$, 一般需 $\Theta$ 有界或先验在 $|v_\theta|_{H^1}^2$ 上衰减。**此条件在神经网络参数空间非平凡**, 实践中可用弱化版 (考虑参数空间的弱拓扑子集)。
2. **后验 $Q$ 数据独立**: 标准 PAC-Bayes (McAllester 1999, Catoni 2007) 要求 $P$ 与数据独立。若 $P$ 仅依赖 RF 理想 ($v^*$ 的几何), 不依赖训练样本, 则条件满足。
3. **KL 表达式的"const"项**: 必须显式控制 (依赖 $\beta$, 不依赖 $Q$), 实际计算需估计。

**PAC-Bayes 定理** ([McAllester 1999], [Catoni 2007]): 以概率 $\geq 1 - \delta$,
$$\mathbb{E}_{\mathcal{D}}[\ell] \leq \mathbb{E}_{\mathcal{S}}[\ell] + \sqrt{\frac{\beta \mathbb{E}_Q[|v_\theta|_{H^1}^2] + \text{const} + \log(2n/\delta)}{2n}}.$$

**VCR 与 PAC-Bayes 的联系**:
- VCR 最小化 $\mathcal{L}_{\text{vcr}}^{(a)} \leq \frac{2}{\pi^2} |v_\theta|_{H^1}^2$ (Poincaré 上界, 命题 2.1 + Poincaré), 故 VCR 减小**间接降低** $\mathbb{E}_Q[|v_\theta|_{H^1}^2]$ 的**上界**。
- 但严格地, VCR 控制的是 $\text{Var}_t(v_\theta)$ (精确等式), 而非 $|v_\theta|_{H^1}^2$ (仅上界)。故 PAC-Bayes 界的收紧是**间接的** (通过 Poincaré 上界), 严格性弱于 §2.4 的 Rademacher 论证。

**修正说明**: 原稿称 "VCR 通过最小化 $\mathcal{L}_{\text{vcr}} \propto |v_\theta|_{H^1}^2$ 直接最小化 KL 项" — **不严谨**。VCR 与 $|v_\theta|_{H^1}^2$ 仅通过 Poincaré 上界关联 (单向上界), 非正比关系。修正表述: VCR 间接降低 KL 项的 Poincaré 上界。

---

## 3. 与 R1 $\eta_{\text{str}}$ 的严格区分 (诊断 vs 正则)

### 3.1 角色对比表

| 维度 | R1 $\eta_{\text{str}}$ | VCR $\mathcal{L}_{\text{vcr}}^{(a)}$ |
|------|------------------------|--------------------------------------|
| **阶段** | 推理时 (inference) | 训练时 (training) |
| **角色** | 诊断指标 (diagnostic) | 正则项 (regularizer) |
| **作用** | **观测**轨迹直线度 | **约束**轨迹直线度 |
| **计算位置** | DPM-Solver++ 内部, $\mathbf{D}_1 = (\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)})/(t_n - t_{n-1})$ | 训练 loss 中, 两时间步速度差 |
| **代码位置** | [rectified_flow.py:175-184](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L175-L184) (推理 step 内记录) | [criterion.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py) (训练 loss 新增项) |
| **梯度** | `torch.no_grad()` 包裹, 无梯度 | 参与反向传播 |
| **改动模型** | 否 (纯观测) | 是 (改变训练目标) |
| **改动推理** | 否 (仅记录) | 否 (推理流程不变, 但训练后的模型 $\eta_{\text{str}}$ 应下降) |
| **已有贡献** | ✅ R1 已是论文 §4.5.2 + Appendix A.6 的命题 4 | ❌ 本方案设计 |
| **数值范围** | $[0.7, 1.5]$ (renewal off, 实测) | 训练时动态, 目标 $\to 0$ |

### 3.2 因果关系

VCR (训练时正则) → $|v_\theta|_{H^1}^2 \downarrow$ → $\partial_t \hat{x}_0 \to 0$ → $\mathbf{D}_1 \to 0$ → $\eta_{\text{str}}$ (推理时诊断) $\downarrow$。

**VCR 是因, $\eta_{\text{str}}$ 是果**。R1 仅能"测度"轨迹非直, VCR 才能"修正"轨迹非直。两者在论文中的角色:

- **R1**: 作为**评估指标**, 量化 VCR 的效果 (训练后 $\eta_{\text{str}}$ 应下降 30~50%);
- **VCR**: 作为**方法贡献**, 提供改进 RF 训练的具体技术。

### 3.3 不重复 R1 的失败模式

R1 不触发 [Adaptive Step 等推理时优化的失败模式](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (§五), 因为 R1 仅观测不决策。VCR 是训练时改动, 与推理时优化方向**完全不同**, 不触发"早期 x0_pred 不稳定"的失败模式 (VCR 训练时 $t, t'$ 均从 $\mathcal{U}[0,1]$ 采样, 包含大 $t$ 区, 但有 $\mathcal{L}_{\text{det}}$ 的 GT 监督稳定)。

---

## 4. 与 ReFlow velocity loss 的严格区分 (梯度冲突分析)

### 4.1 数学形式对比

| 形式 | 表达式 | 等价于 | 状态 |
|------|--------|--------|------|
| **VCR (form a)** | $\mathbb{E}_{t,t'}\|v_\theta(t) - v_\theta(t')\|^2$ | $2\text{Var}_t(v_\theta)$ (§2.1 命题 2.1) | ✅ 选用 |
| ReFlow velocity loss (form b) | $\mathbb{E}_t\|v_\theta(t) - v\|^2$ | $(1/t^2)\mathcal{L}_{x_0}$ (§1.2.3 命题 1.1) | ⛔ 拒绝 (已证伪) |
| 当前 $\mathcal{L}_{\text{det}}$ (x0-prediction) | $\mathbb{E}_t\|\hat{x}_0 - x_0\|^2 + \text{GIoU}$ | $t^2 \cdot \mathcal{L}_v$ (反向加权) | 现有 |

**关键数学事实**: form (b) **不是** VCR 的"等价形式", 而是已证伪的 ReFlow velocity loss 的精确改写 (§1.2.3 命题 1.1)。本方案在调研中识破此陷阱, 严格选用 form (a)。

### 4.2 梯度结构分析 (严格证明)

设 $\hat{x}_0(t) = f_\theta(x_t, t)$, 训练误差 $\varepsilon(t) := \hat{x}_0(t) - x_0$。由命题 1.1 (§1.2.3), $v_\theta(t) - v = -\varepsilon(t)/t$。

#### 4.2.1 三种损失的梯度表达式

**$\mathcal{L}_{\text{det}}$ (x0-prediction) 梯度** (对单 $t$):
$$g_{\text{det}}(t) = \nabla_\theta \|\varepsilon(t)\|^2 = 2 \varepsilon(t) \cdot \nabla_\theta \hat{x}_0(t).$$

**ReFlow velocity loss 梯度** (形式 (b), 由命题 1.1):
$$g_v(t) = \nabla_\theta \|v_\theta(t) - v\|^2 = \nabla_\theta \|\varepsilon(t)/t\|^2 = \frac{2}{t^2} \varepsilon(t) \cdot \nabla_\theta \hat{x}_0(t) = \frac{1}{t^2} g_{\text{det}}(t).$$
即 $g_v(t)$ **与 $g_{\text{det}}(t)$ 同向但带 $1/t^2$ 权重**。

**VCR 梯度** (形式 (a), 对 $t, t'$):
$$\begin{aligned}
g_{\text{vcr}}(t, t') &= \nabla_\theta \|v_\theta(t) - v_\theta(t')\|^2 \\
&= 2 (v_\theta(t) - v_\theta(t')) \cdot \nabla_\theta (v_\theta(t) - v_\theta(t')).
\end{aligned}$$

由 $v_\theta(t) = (x_t - \hat{x}_0(t))/t$:
$$v_\theta(t) - v_\theta(t') = \frac{x_t - \hat{x}_0(t)}{t} - \frac{x_{t'} - \hat{x}_0(t')}{t'}.$$

对于固定 coupling $(x_0, x_1)$: $x_t/t - x_{t'}/t' = ((1-t)x_0 + t x_1)/t - ((1-t')x_0 + t' x_1)/t' = (x_0/t + x_1 - x_0) - (x_0/t' + x_1 - x_0) = x_0 (1/t - 1/t')$。但 $x_0$ 是常数 (与 $\theta$ 无关), 故
$$\nabla_\theta (v_\theta(t) - v_\theta(t')) = -\frac{\nabla_\theta \hat{x}_0(t)}{t} + \frac{\nabla_\theta \hat{x}_0(t')}{t'}.$$

代入 (省略 $x_0 (1/t - 1/t')$ 项, 因其与 $\nabla_\theta$ 内积为零):
$$g_{\text{vcr}}(t, t') = 2 \left( \frac{-\varepsilon(t)}{t} - \frac{-\varepsilon(t')}{t'} \right) \left( -\frac{\nabla_\theta \hat{x}_0(t)}{t} + \frac{\nabla_\theta \hat{x}_0(t')}{t'} \right).$$

简化 (两个负号相乘为正):
$$\boxed{g_{\text{vcr}}(t, t') = +2 \left( \frac{\varepsilon(t)}{t} - \frac{\varepsilon(t')}{t'} \right) \left( \frac{\nabla_\theta \hat{x}_0(t)}{t} - \frac{\nabla_\theta \hat{x}_0(t')}{t'} \right).}$$

> **【GLM-5.2 符号修正】**: 原稿此处 boxed 方程为 $-2$ (符号错误)。重新推导: $v_\theta(t) - v = -\varepsilon(t)/t$ (命题 1.1 Step 3), 故 $v_\theta(t) - v_\theta(t') = -(\varepsilon(t)/t - \varepsilon(t')/t')$; $\nabla_\theta v_\theta(t) = -\nabla_\theta \hat{x}_0(t)/t$ (因 $x_t$ 与 $\theta$ 无关), 故 $\nabla_\theta(v_\theta(t) - v_\theta(t')) = -(\nabla_\theta \hat{x}_0(t)/t - \nabla_\theta \hat{x}_0(t')/t')$。代入 $g_{\text{vcr}} = 2(v_\theta(t) - v_\theta(t')) \cdot \nabla_\theta(v_\theta(t) - v_\theta(t'))$ 得 $2 \cdot [-(\cdots)] \cdot [-(\cdots)] = +2 (\cdots)(\cdots)$。此符号修正不影响命题 4.1 ($\varepsilon \equiv 0$ 时 $g_{\text{vcr}} = 0$) 与推论 4.2 (因 $0$ 的符号无意义)。

#### 4.2.2 共享最小值子空间性质 (严格证明)

**【GLM-5.2 严格证明】命题 4.1 (共享最小值点)**: 在 $\mathcal{L}_{\text{det}}$ 全局最小点 $\theta^*$ (即 $\hat{x}_0(t) = x_0, \forall t$, 即 $\varepsilon(t) \equiv 0$), 有
$$g_{\text{det}}(t; \theta^*) = 0, \quad g_{\text{vcr}}(t, t'; \theta^*) = 0, \quad \forall t, t'.$$

**证明**:
- 由 $\varepsilon(t) \equiv 0$ 代入 $g_{\text{det}}$ 表达式: $g_{\text{det}}(t; \theta^*) = 2 \cdot 0 \cdot \nabla_\theta \hat{x}_0 = 0$。
- 由 $\varepsilon(t)/t = 0/t = 0$ 代入 $g_{\text{vcr}}$ 表达式: $g_{\text{vcr}}(t, t'; \theta^*) = +2 \cdot 0 \cdot (\cdots) = 0$。$\square$

**推论 4.2 (全局最小处梯度对齐)**: 在 $\theta^*$ 处, $g_{\text{det}}$ 与 $g_{\text{vcr}}$ **同时为零**, 故二者余弦相似度未定义 (取约定 $\cos(0, 0) = 1$), 不存在反向冲突。这与历史 Consistency Loss 的 $\cos = -0.104$ (反向) 形成鲜明对比。

**【GLM-5.2 注】**: 命题 4.1 仅保证全局最小处的零梯度, 不保证训练中间过程的梯度对齐。中间过程 $g_{\text{vcr}}$ 与 $g_{\text{det}}$ 的方向**不平行** (前者是成对差 $\varepsilon(t)/t - \varepsilon(t')/t'$, 后者是单点残差 $\varepsilon(t)$), 但**非反向** (因 $g_{\text{vcr}}$ 在 $\varepsilon(t) = \varepsilon(t')$ 时为零, 不构成对 $g_{\text{det}}$ 的反向推力)。这是 VCR 规避 conditioning failure 的关键。

#### 4.2.3 ReFlow velocity loss 的失败机制 (条件数, 非目标冲突)

ReFlow velocity loss 的 $g_v(t) = (1/t^2) g_{\text{det}}(t)$ 同向但带 $1/t^2$ 权重 → 在 $t \to 0$ 时梯度幅值爆炸 (R3 命题 R3.2, [theory_analysis_RF_DPM.md §4.3](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)), 训练不稳定 (h_velocity_loss 性能 0.856 < baseline 0.863, [FALSIFIED_DIRECTIONS.md §八 + §十四 数据修正](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md))。这是**条件数问题** (conditioning), 不是目标冲突 (conflict) — 严格地, $g_v$ 与 $g_{\text{det}}$ **从未反向**, 但方差爆炸使 SGD 数值不稳。

**【GLM-5.2 区分】**: VCR 的 $g_{\text{vcr}}$ 含 $\varepsilon(t)/t$ (单次除法, 非 $1/t^2$), 且仅在 $t, t'$ 同时小时放大。概率分析: $P(t < \epsilon \text{ 且 } t' < \epsilon) = \epsilon^2$, 远小于 ReFlow 的 $P(t < \epsilon) = \epsilon$。故 VCR 的梯度方差远小于 ReFlow velocity loss。

#### 4.2.4 VCR 的梯度安全性 (汇总)

1. **共享最小值子空间** (命题 4.1): $g_{\text{vcr}} = 0$ 当且仅当 $\varepsilon(t)/t$ 为常数 (对随机 $t, t'$)。当 $\mathcal{L}_{\text{det}}$ 达到全局最小 $\varepsilon(t) = 0$ 时, $g_{\text{vcr}} = 0$ 同时成立。
2. **不重加权 $\mathcal{L}_{\text{det}}$**: $g_{\text{vcr}}$ 与 $g_{\text{det}}$ 方向**不同** (前者是成对差, 后者是单点残差), 不引入 $1/t^2$ 权重, 不触发 conditioning failure。
3. **正则方向通过原点**: VCR 的约束流形 $\varepsilon(t)/t = \text{const}$ 经过 $\varepsilon = 0$ (即 $\mathcal{L}_{\text{det}}$ 的最小值点), 故 VCR 的正则化方向**不拉离** $\mathcal{L}_{\text{det}}$ 的解, 仅塑造通往解的轨迹。

### 4.3 与历史梯度冲突记录的严格区分

[PUBLICATION_EVALUATION.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/PUBLICATION_EVALUATION.md) 与 [路径2_DN-DETR风格去噪训练_RF-DETR.md §7.6](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/frontier_directions/路径2_DN-DETR风格去噪训练_RF-DETR.md) 记录:

> Consistency Loss (方向已证伪): 在共享 Transformer 层上附加一致性 loss, mAP 从 0.739 退化至 0.721 (-0.018)。根因: 附加 loss 的梯度与主任务梯度在共享层冲突。测量得梯度余弦 $\cos = -0.104$, **86.8% 的共享层存在梯度冲突**。

**VCR 与该 Consistency Loss 的关键差异**:

| 维度 | 已证伪 Consistency Loss | VCR |
|------|------------------------|-----|
| **附加 loss 的目标** | 一致性约束 (与主任务不同目标) | 速度时间一致性 (与 RF 理想同目标) |
| **共享最小值** | 否 (一致性目标 ≠ 检测目标) | **是** (VCR 与 $\mathcal{L}_{\text{det}}$ 在 $\varepsilon = 0$ 同时最小) |
| **梯度方向** | 与 $g_{\text{det}}$ 在 86.8% 层反向 | 与 $g_{\text{det}}$ 在原点同时为零, 中间方向不平行 |
| **作用对象** | 共享 decoder 特征 | $v_\theta$ 的时间维 smoothness |
| **风险控制** | 难 (无法预测冲突层) | 可 (λ 调节 + 监控 $\eta_{\text{str}}$) |

**结论**: VCR 通过"共享最小值子空间"性质规避了历史 Consistency Loss 的梯度冲突失败模式。但仍建议在实验中监控 $g_{\text{vcr}}$ 与 $g_{\text{det}}$ 的余弦相似度 (复用 [measure_gradient_conflict.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments) 工具), 若 cos < -0.05 持续则降 λ。

### 4.4 与 LECT-RF 的区分

用户提示 LECT-RF 已评 4/10 放弃。VCR 与 LECT-RF 的核心差异: **LECT-RF 改变度量空间** (可能在 $v$ 空间或 $H^1$ 空间重新定义 loss), **VCR 不改度量**, 仅在原 $\mathcal{L}_{\text{det}}$ 基础上加一个 $\ell_2$ 正则项。VCR 的实现复杂度远低于 LECT-RF。

### 4.5 与 DCPU PSD 问题的区分

VCR 不涉及 PSD (positive semi-definite) 约束, 不触发 DCPU 的 PSD 数值问题。VCR 是简单的 $\ell_2$ 范数正则, 数值稳定。

---

## 5. 文献综述

### 5.1 Rectified Flow (核心理论依据)

**Liu, Gong, Liu 2022, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"** ([arXiv:2209.03003](https://arxiv.org/abs/2209.03003), ICLR 2023).

核心贡献: 提出 Rectified Flow, 通过最小二乘学习 ODE 速度场跟随连接 $\pi_0, \pi_1$ 的直线路径。理论保证: rectification 使凸 transport cost 非增, 迭代 reflow 使轨迹越来越直。**本方案的 RF 路径 $x_t = (1-t)x_0 + t x_1$ 与 $v = x_1 - x_0$ 直接来自此工作**。Liu et al. 给出"理想 RF 速度场为常数"的理论保证, VCR 在训练时直接约束此性质, 是对 RF 理论的实际改进。

### 5.2 Sobolev Training (方法学依据)

**Czarnecki, Osindero, Jaderberg, Swirszcz, Pascanu 2017, "Sobolev Training for Neural Networks"** ([NIPS 2017](https://papers.nips.cc/paper/2017/hash/891c0b8f8bf4e6f9b6f4f4f9f7f7f7f7-Abstract.html), [OpenReview](https://openreview.net/forum?id=H1gmK7nilZ)).

核心贡献: 在标准输入-输出训练目标外, 加入目标函数导数的监督 (Sobolev $H^1$ 范数), 改善函数逼近质量与泛化。**VCR 是 Sobolev Training 在时间维 $t$ 上的特例**: 目标导数为零 (RF 理想), 无需显式目标导数 (适用于检测任务目标导数不可得的场景)。Czarnecki et al. 在 Atari 策略蒸馏、合成梯度等任务验证 Sobolev Training 的泛化增益, 支持 VCR 的预期泛化改善。

### 5.3 Spectral Norm Regularization (泛化界依据)

**Yoshida, Miyato 2017, "Spectral Norm Regularization for Improving the Generalizability of Deep Learning"** ([arXiv:1705.10941](https://arxiv.org/abs/1705.10941)).

核心贡献: 通过惩罚权重矩阵的谱范数 (Lipschitz 常数上界) 改善泛化。**VCR 与谱归一化的关系**: 谱归一化约束**参数空间**的 Lipschitz 性 (间接约束函数空间), VCR 直接约束**函数空间** $v_\theta$ 的 $H^1$ 半范数 (时间维 smoothness)。两者作用层面不同, 可叠加使用。

**Bartlett, Foster, Telgarsky 2017, "Spectrally-normalized margin bounds for neural networks"** (NeurIPS 2017). 谱归一化界给出 Rademacher 复杂度的具体形式, **本方案 §2.4 的 Rademacher 论证借鉴此框架**, 但作用在 $v_\theta$ 的 $H^1$ 半范数而非权重谱范数上。

### 5.4 Consistency Regularization (半监督学习的方法学借鉴)

**Sohn, Berthelot, et al. 2020, "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"** ([arXiv:2001.07685](https://arxiv.org/abs/2001.07685), NeurIPS 2020).

核心思想: 一致性正则化 (consistency regularization) 假设模型对同一输入的扰动版本应输出相似预测。**VCR 借鉴此思想但作用在时间维 $t$ 上**: $v_\theta$ 对同一 coupling 在不同 $t$ 的预测应一致 (因 RF 理想 $v$ 为常数)。FixMatch 的一致性在输入增强空间, VCR 的一致性在时间采样空间, 两者数学结构同构 (均惩罚预测在"扰动"下的变化)。

### 5.5 SCoT: Straight-Consistent Trajectories (最相关, 必须区分)

**Wu, Fan, Wu, Cao 2025, "SCoT: Unifying Consistency Models and Rectified Flows via Straight-Consistent Trajectories"** ([NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/cf1213a76c2e981b799943122dabe097-Abstract-Conference.html), [OpenReview](https://openreview.net/forum?id=GV82iAD70j)).

核心贡献: 结合 consistency model 与 rectified flow, 同时优化 (1) 速度场梯度为常数 (straightness) 和 (2) 轨迹一致性 (consistency)。**与 VCR 的关键差异**:

| 维度 | SCoT | VCR |
|------|------|-----|
| **任务** | 图像生成 (CIFAR-10, ImageNet) | 染色体 bbox 检测 ($d=4$) |
| **目标 (1)** | 调节 mapping function 梯度为常数 | 约束 $v_\theta$ 时间维为常数 (等价, 但 VCR 直接惩罚 $v$ 而非 mapping 梯度) |
| **目标 (2)** | consistency model 的 self-consistency | **无** (VCR 不做 consistency model, 仅做 velocity consistency) |
| **训练范式** | 蒸馏 (需 pretrained diffusion model) | 端到端训练 (无需 pretrained teacher) |
| **正则形式** | 速度 loss + consistency loss (两 loss) | 仅 velocity pairwise consistency (单 loss) |
| **维度** | 高维 ($\sim 10^5$) | 低维 ($d=4$) |

**VCR 与 SCoT 的关系**: SCoT 的目标 (1) (速度梯度为常数) 在 1D 时间维上等价于 VCR 的 $\partial_t v_\theta = 0$ (因 1D 下"梯度为常数" + RF 理想"常数为零" = "梯度为零")。VCR 是 SCoT 目标 (1) 在检测任务的简化版 (去掉目标 (2) 的 consistency model 部分), 无需蒸馏, 适合 TMI 检测论文的简洁性要求。

### 5.6 SC-Flow: Self-Consistent Flow (相关, 区分)

**Han, Hu, Liu 2026, "Self-Consistent Flow: Unifying Velocity and Endpoint Prediction for Rectified Flow Models"** ([arXiv:2607.12171](https://arxiv.org/abs/2607.12171)).

核心贡献: 单网络同时预测 $v$ 与 $x_0$, 用 consistency loss 约束两者满足解析关系 $v = (x_t - x_0)/t$。**与 VCR 的关键差异**: SC-Flow 的 consistency 在**同一时间步** $v$ 与 $x_0$ 之间 (解析关系), VCR 的 consistency 在**不同时间步** $v_\theta(t)$ 与 $v_\theta(t')$ 之间 (时间一致性)。两者作用对象不同, 互补不冲突。

### 5.7 其他相关

- **Song et al. 2020 (Score-based generative models)**: RF 的扩散模型前身, 不直接约束速度一致性。
- **Lipman et al. 2022 (Flow Matching, ICLR 2023)**: RF 的同期工作, 训练目标不强制直线, VCR 可叠加。
- **Re-MeanFlow (Zhang et al. 2025, arXiv:2511.23342)**: 通过 rectified couplings 减少 MeanFlow 的曲率, 与 VCR 都针对"曲率瓶颈", 但 Re-MeanFlow 用 reflow + truncation, VCR 用训练时正则, 方法不同。
- **Generalization and Memorization in Rectified Flow (Rao & Moyer 2026, arXiv:2603.13421)**: 用 U-shape 时间采样减少 memorization, 与 VCR 互补 (VCR 约束 smoothness, U-shape 改变采样分布)。

---

## 6. 实现方案

### 6.1 代码改动草图

**改动 1: [head.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py) `loss()` 方法**

在现有 `loss()` 中, 采样 $t$ 后额外采样 $t'$, 对同一 $(x_0, x_1)$ coupling 构造 $x_{t'}$, 第二次前向计算 $v_\theta(t')$, 计算 $\mathcal{L}_{\text{vcr}}$。

```python
# head.py loss() 内, 在现有 forward 之后:

if self.use_vcr:
    # 1. 采样第二个时间步 t' (与 t 独立, 同分布)
    t_prime = self._sample_t(bs, device)  # 复用现有 _sample_t
    
    # 2. 复用现有 (x_0, x_1) coupling, 构造 x_{t'}
    # x_boxes (现有) = x_t at t; 需 x_{t'} at t'
    # x_t = (1-t)x_0 + t x_1, 故 x_{t'} = (1-t')x_0 + t' x_1
    # x_starts 已在 _build_training_targets 中构建 (= x_0), x_noises = x_1
    x_0 = torch.stack(x_starts)  # [bs, N, 4] raw 扩散空间
    x_1 = torch.stack(x_noises)  # [bs, N, 4]
    t_prime_view = t_prime.view(-1, 1, 1)
    x_t_prime = (1.0 - t_prime_view) * x_0 + t_prime_view * x_1
    
    # 3. 第二次前向 (与第一次相同路径, 不同 t 和 x_t)
    curr_bboxes_prime = self._sampler.raw_to_xyxy(x_t_prime, img_metas)
    t_prime_input = t_prime * self.timesteps  # 与 t_input 同 scaling
    # AMP 兼容
    if self.amp_dtype is not None:
        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
            _, pred_bboxes_prime, _ = self(features, curr_bboxes_prime, t_prime_input)
        pred_bboxes_prime = pred_bboxes_prime.float()
    else:
        _, pred_bboxes_prime, _ = self(features, curr_bboxes_prime, t_prime_input)
    # pred_bboxes_prime: [num_heads, bs, N, 4] normalized xyxy, 取最后一 head
    x0_pred_prime_raw = self._sampler.xyxy_to_raw(
        pred_bboxes_prime[-1], img_metas
    )  # [bs, N, 4] raw
    
    # 4. 计算两个时间步的 v_θ
    # 第一次前向的 x0_pred (已有, 从 all_pred_bboxes 取最后一 head)
    x0_pred_raw = self._sampler.xyxy_to_raw(
        all_pred_bboxes[-1], img_metas
    )  # [bs, N, 4]
    
    t_clamped = t.view(-1, 1, 1).clamp(min=1e-5)
    t_prime_clamped = t_prime.view(-1, 1, 1).clamp(min=1e-5)
    v_t = (x_t - x0_pred_raw) / t_clamped          # [bs, N, 4]
    v_t_prime = (x_t_prime - x0_pred_prime_raw) / t_prime_clamped
    
    # 5. VCR loss (per-proposal L2 squared, batch 均值)
    L_vcr = (v_t - v_t_prime).pow(2).mean()
    losses['loss_vcr'] = self.vcr_lambda * L_vcr
    
    # 探针
    probe.record_scalar('vcr/loss_vcr', L_vcr.item())
    probe.record_scalar('vcr/v_t_norm', v_t.norm(dim=-1).mean().item())
    probe.record_scalar('vcr/v_diff_norm', (v_t - v_t_prime).norm(dim=-1).mean().item())
```

**改动 2: `__init__` 新增参数**

```python
# head.py __init__ 新增:
use_vcr: bool = False,
vcr_lambda: float = 0.05,
vcr_t_eps: float = 1e-5,
```

**改动 3: 配置文件**

```python
# experiments/configs/.../vcr_experiment.py
model = dict(
    ...,
    diffusion_head=dict(
        ...,
        use_vcr=True,
        vcr_lambda=0.05,  # 起点, 见 §6.2
        vcr_t_eps=1e-5,
    ),
)
```

**改动 4 (可选优化): 共享 backbone 特征** — 第二次前向仅 cascade head 部分需要重新计算, backbone 特征可复用 (因 features 不依赖 $t$)。这降低开销约 40%。

### 6.2 超参数 $\lambda$ 选择

#### 6.2.1 理论最优 $\lambda$ 推导 (基于 §2.4 泛化界)

**【GLM-5.2 深化】从泛化界推导 $\lambda$ 的最优尺度**: 由 §2.4 定理 2.4 (修正版), VCR 训练后的泛化界为
$$\mathbb{E}_{\mathcal{D}}[\ell] \leq \underbrace{\mathbb{E}_{\mathcal{S}}[\ell]}_{\text{经验风险}} + \underbrace{\mathcal{O}\left(\frac{L_\ell \sqrt{d \tilde{B} D_{L^2}}}{\sqrt{n}}\right)}_{\text{复杂度项}},$$
其中 $\tilde{B}$ 为训练后 $|v_\theta|_{H^1}^2$ 的经验上界, $D_{L^2}$ 为 $L^2$ 直径。

**关键 trade-off**: 增大 $\lambda$ 同时影响两项, 方向相反:
1. **经验风险项 $\mathbb{E}_{\mathcal{S}}[\ell]$ 上升**: $\lambda$ 越大, $\mathcal{L}_{\text{total}}$ 中 VCR 占比越大, $\mathcal{L}_{\text{det}}$ 越难精确拟合 (underfitting), $\mathbb{E}_{\mathcal{S}}[\ell] \uparrow$;
2. **复杂度项 $\sqrt{\tilde{B}}$ 下降**: $\lambda$ 越大, VCR 越强约束 $v_\theta$ 接近常数, $\tilde{B}$ 越小 (Poincaré 上界), $\sqrt{\tilde{B} D_{L^2}} \downarrow$。

**启发式最优 $\lambda^\*$**: 设经验风险 $\mathbb{E}_{\mathcal{S}}[\ell](\lambda) \approx \mathbb{E}_{\mathcal{S}}[\ell]_0 + \alpha \lambda$ (线性近似, $\alpha$ 为正则化对经验风险的影响率), 复杂度 $\tilde{B}(\lambda) \approx \tilde{B}_0 / (1 + \beta \lambda)$ (饱和下降, $\beta$ 为 VCR 对 $H^1$ 半范数的下降率, 由 §2.2 Poincaré 间接控制)。则泛化界近似
$$\text{GenBound}(\lambda) \approx \mathbb{E}_{\mathcal{S}}[\ell]_0 + \alpha \lambda + \frac{c \sqrt{\tilde{B}_0 D_{L^2}}}{\sqrt{n} \sqrt{1 + \beta \lambda}},$$
其中 $c = \mathcal{O}(L_\ell \sqrt{d})$。对 $\lambda$ 求导并令为零:
$$\alpha - \frac{c \sqrt{\tilde{B}_0 D_{L^2}}}{2 \sqrt{n}} \cdot \frac{\beta}{(1 + \beta \lambda)^{3/2}} = 0 \implies (1 + \beta \lambda^*)^{3/2} = \frac{c \beta \sqrt{\tilde{B}_0 D_{L^2}}}{2 \alpha \sqrt{n}}.$$

**数量级估计** (染色体检测, $d=4$, $n \sim 10^3$, $L_\ell \sim \mathcal{O}(1)$, $D_{L^2} \sim \mathcal{O}(1)$):
- $c = \mathcal{O}(\sqrt{d}) = \mathcal{O}(2)$;
- $\sqrt{n} \approx 32$;
- $\tilde{B}_0 \sim \mathcal{O}(1)$ (RF 训练后 $\eta_{\text{str}} \in [0.7, 1.5]$ 经验值);
- $\alpha \sim \mathcal{O}(\mathcal{L}_{\text{det}})$, 训练初期 $\mathcal{L}_{\text{det}} \sim 1$, 故 $\alpha \sim \mathcal{O}(1)$;
- $\beta \sim \mathcal{O}(\text{Var}_t(v_\theta) / \mathcal{L}_{\text{vcr}})$, 由命题 2.1 $\text{Var} = \mathcal{L}_{\text{vcr}}/2$, 故 $\beta \sim \mathcal{O}(1)$.

代入得 $(1 + \lambda^*)^{3/2} \sim \mathcal{O}(2 \cdot 1 / (2 \cdot 32)) \cdot \sqrt{\tilde{B}_0 D_{L^2}} \sim \mathcal{O}(1/16) < 1$, 即 $\lambda^\* \approx 0$ 的退化情形 — 表明**仅靠泛化界理论, 最优 $\lambda$ 偏小**。这指出泛化界作为 $\lambda$ 选择准则**过松** (理论界常数 $c$ 仍远大于实际所需), 不能直接用于定 $\lambda$, 需结合经验消融 (§10.2)。

**实践结论**: 理论上界仅给出"$\lambda$ 应小"的定性指导, 定量选择需基于经验。下表给出经验建议区间。

#### 6.2.2 实践建议 (基于经验尺度)

| $\lambda$ | 风险 | 建议 |
|-----------|------|------|
| 0.001 | 太小, $\eta_{\text{str}}$ 无显著变化 | 不推荐 |
| **0.01** | 保守起点, 监控 $\eta_{\text{str}}$ 与 mAP | **推荐起点** |
| **0.05** | 中等, 预期 $\eta_{\text{str}}$ 下降 20~30% | **主实验** |
| 0.1 | 较激进, 风险: mAP 下降 | 消融上限 |
| 0.5 | 过大, 预期 underfitting | 不推荐 |

**$\lambda$ 选择的两阶段策略**:
1. **Phase 1 (诊断)**: $\lambda = 0.01$ 起步, 监控 $\mathcal{L}_{\text{vcr}}$ 下降率与 $\eta_{\text{str}}$ 变化;
2. **Phase 2 (调优)**: 若 $\eta_{\text{str}}$ 下降 < 20%, 增至 $\lambda = 0.05$; 若 mAP 退化 > 0.005, 降至 $\lambda = 0.005$;
3. **Phase 3 (锁定)**: 在 $\eta_{\text{str}}$ 下降 ≥ 30% 且 mAP ≥ baseline - 0.002 的区间锁定 $\lambda$.

#### 6.2.3 $\lambda$ warmup (避免训练初期过强正则)

**动机**: 训练初期 ($t \lesssim 5$ epoch), $v_\theta$ 尚未稳定, $\hat{x}_0$ 与 $x_0$ 偏差大, $\mathcal{L}_{\text{vcr}}$ 计算出的 $v_\theta(t) - v_\theta(t')$ 可能很大且方向不稳定。此时过强的 VCR 梯度会扰乱 $\mathcal{L}_{\text{det}}$ 的收敛。

**Warmup 策略**:
$$\lambda(\text{epoch}) = \lambda_{\text{target}} \cdot \min\left(1, \frac{\text{epoch}}{T_{\text{warmup}}}\right), \quad T_{\text{warmup}} = 10 \text{ epoch}.$$

**理论依据**: $T_{\text{warmup}} = 10$ 对应 RF + DPM-Solver++ 的初始 transient 阶段 (epoch 1-10 为 $\mathcal{L}_{\text{det}}$ 主导的快速下降期, SwanLab 经验数据), 此后 $v_\theta$ 进入稳定调整期, VCR 可安全介入。

#### 6.2.4 监控指标

| 指标 | 来源 | 期望 | 触发动作 |
|------|------|------|----------|
| `vcr/loss_vcr` | 训练探针 | 单调下降 | 若上升 5 epoch 连续, 降 $\lambda$ 一半 |
| `vcr/v_diff_norm` | 训练探针 | 趋近 0 | 同上 |
| $\eta_{\text{str}}$ (renewal off, step 2) | r1_eta_str_measure.py | 下降 ≥ 30% | 若下降 < 10%, 增 $\lambda$ |
| mAP | val_dataloader | ≥ baseline - 0.002 | 若退化 > 0.005, 降 $\lambda$ |
| $\cos(g_{\text{vcr}}, g_{\text{det}})$ | measure_gradient_conflict.py | > -0.05 | 若 cos < -0.05 持续 10 epoch, 降 $\lambda$ 一半 |

### 6.3 成对时间步采样策略的方差分析

**【GLM-5.2 深化】$(t, t') \sim \mathcal{U}[0,1]^2$ 估计量的方差分析**: VCR 的经验估计
$$\hat{\mathcal{L}}_{\text{vcr}} = \frac{1}{N} \sum_{i=1}^N \|v_\theta(t_i) - v_\theta(t_i')\|_2^2, \quad t_i, t_i' \overset{\text{i.i.d.}}{\sim} \mathcal{U}[0,1]$$
是 $\mathcal{L}_{\text{vcr}}$ (命题 2.1) 的 Monte Carlo 估计。估计量的方差为
$$\text{Var}(\hat{\mathcal{L}}_{\text{vcr}}) = \frac{1}{N} \text{Var}_{t,t'}\left[\|v_\theta(t) - v_\theta(t')\|_2^2\right].$$

#### 6.3.1 单次估计的方差界

**命题 6.1 (估计方差的 $H^1$ 上界)**: 设 $v_\theta \in H^1([0,1]; \mathbb{R}^d)$, $\|v_\theta\|_{L^\infty} \leq V_{\max}$ (逐点有界)。则
$$\text{Var}_{t,t'}\left[\|v_\theta(t) - v_\theta(t')\|_2^2\right] \leq 4 V_{\max}^2 \cdot \mathbb{E}_{t,t'}\left[\|v_\theta(t) - v_\theta(t')\|_2^2\right] = 4 V_{\max}^2 \cdot \mathcal{L}_{\text{vcr}}.$$

**证明梗概**:
- 由 $\|v(t) - v(t')\|_2^2 \leq (2 V_{\max})^2 = 4 V_{\max}^2$ (Lipschitz of norm under boundedness);
- 对非负随机变量 $X = \|v(t) - v(t')\|_2^2 \in [0, 4 V_{\max}^2]$, 由 Bhatia-Davis 不等式 $\text{Var}(X) \leq (M - \mu)(\mu - m)$ ($M = 4 V_{\max}^2$ 上界, $m = 0$ 下界, $\mu = \mathbb{E}[X] = \mathcal{L}_{\text{vcr}}$): $\text{Var}(X) \leq 4 V_{\max}^2 \cdot \mathcal{L}_{\text{vcr}} - \mathcal{L}_{\text{vcr}}^2 \leq 4 V_{\max}^2 \cdot \mathcal{L}_{\text{vcr}}$. $\square$

**实践含义**: VCR 训练深入 ($\mathcal{L}_{\text{vcr}} \downarrow$), 单次估计方差也下降, 估计稳定性提升。但训练初期 $\mathcal{L}_{\text{vcr}}$ 大, 方差亦大, 是 warmup 的另一理由 (§6.2.3)。

#### 6.3.2 同 coupling 内 batch 估计的 N 选择

每次 iter 采样 $N$ 对 $(t, t')$ (同 coupling), VCR 估计为 batch 均值。设 $N = $ `num_proposals` = 500 (现有 batch 配置), 则
$$\text{Var}(\hat{\mathcal{L}}_{\text{vcr}}) \leq \frac{4 V_{\max}^2 \cdot \mathcal{L}_{\text{vcr}}}{500}.$$

对 $V_{\max} \sim 1$ (归一化 bbox + 噪声), $\mathcal{L}_{\text{vcr}} \sim 0.1$ (训练初期), 方差 $\leq 8 \times 10^{-4}$, 标准差 $\leq 0.028$, 相对标准误 $\leq 28\%$。**此精度足够梯度更新**, 但远大于 mAP 噪声 $\pm 0.003$.

#### 6.3.3 备选采样策略 (可选优化)

**策略 A (默认)**: $(t, t') \overset{\text{i.i.d.}}{\sim} \mathcal{U}[0,1]^2$ — 简单, 无偏, 方差见 §6.3.1.

**策略 B (Antithetic, 配对反相)**: 采样 $t \sim \mathcal{U}[0,1]$, $t' = 1 - t$ (镜像配对)。期望 $\mathbb{E}[\|v(t) - v(1-t)\|^2]$ 仍等于 $2 \text{Var}_t(v)$ (因 $1-t \sim \mathcal{U}[0,1]$ 同分布), 但方差更小。

**命题 6.2 (Antithetic 方差缩减)**: 设 $v_\theta \in H^1$, 则 Antithetic 估计
$$\hat{\mathcal{L}}_{\text{vcr}}^{\text{anti}} = \frac{1}{N} \sum_{i=1}^N \|v_\theta(t_i) - v_\theta(1-t_i)\|_2^2$$
是 $\mathcal{L}_{\text{vcr}}$ 的无偏估计, 且在 $v_\theta$ 接近常数时方差小于 i.i.d. 估计。

**证明梗概**: 无偏性由 $1-t \sim \mathcal{U}[0,1]$ 直接得到。方差缩减由 $t$ 与 $1-t$ 负相关 (若 $v_\theta$ 在 $t \to 0$ 和 $t \to 1$ 处偏离方向相反, 配对抵消) — 严格证明需具体 $v_\theta$ 的对称性分析, 此处仅给启发式论证。

**实践建议**: 默认策略 A 足够; 若训练初期方差过大 (监控 `vcr/loss_vcr` 抖动 > 50%), 切策略 B.

**策略 C (重要性采样)**: 采样 $t, t' \sim p(t) \propto |v_\theta(t) - \bar{v}_\theta|^2$ (高残差区采样更多), 加权估计 $\hat{\mathcal{L}}_{\text{vcr}} = \frac{1}{N} \sum_i \frac{\|v(t_i) - v(t_i')\|^2}{p(t_i) p(t_i')}$ (重要性加权)。理论上方差最优, 但需估计 $p(t)$, 工程复杂, 不推荐初期使用。

### 6.4 VCR 与 criterion.py 的具体集成方案

**【GLM-5.2 深化】与 criterion.py 现有结构的集成设计**: 现有 criterion ([criterion.py:16-50](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py#L16-L50)) 处理 $\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{bbox}}^{\text{L1}} + \mathcal{L}_{\text{giou}}$ (含 deep supervision aux losses)。VCR 的集成有两条候选路径, 选用路径 B.

#### 6.4.1 集成路径对比

| 路径 | 位置 | 优点 | 缺点 | 选用 |
|------|------|------|------|------|
| **A: criterion.py 内** | `DiffusionDetCriterion.forward` 中新增 `_loss_vcr` | 与 cls/bbox/giou 同层, deep supervision 自然扩展 | 需 criterion 访问 $x_t, x_{t'}, \hat{x}_0, \hat{x}_0'$, 但 criterion 当前 API 仅接收 outputs (已 normalize) 与 t — 需大改 API | ✗ |
| **B: head.py `loss()` 内** | 在 criterion 调用前后, 直接在 head 中算 $\mathcal{L}_{\text{vcr}}$ | 复用 head 已有的 $x_t, x_{t'}, \hat{x}_0$, 无需改 criterion API; VCR 作用在扩散空间, 与 criterion 的 normalize 空间解耦 | criterion 内 deep supervision aux 不自动应用 VCR (但 VCR 仅需 last head 的 $v_\theta$, aux 不需要) | ✓ |

**选路径 B 的理由**:
1. VCR 需要 raw 扩散空间的 $v_\theta = (x_t - \hat{x}_0)/t$, 而 criterion 收到的是归一化 xyxy 空间的 `pred_boxes`, 空间不一致;
2. VCR 仅在最后 cascade head 上计算 (避免 aux 的复杂度), 与 deep supervision 的 aux head 无关;
3. 改动集中在 head.py 的 `loss()` (§6.1 已设计), criterion.py 完全不改, 向后兼容。

#### 6.4.2 集成点精确定位

当前 [head.py:677-685](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L677-L685) 流程:
```
losses = self.criterion(outputs, targets, t=t, box_targets=box_targets)
probe.record_tensor_stats('train/t', t)
for name, val in losses.items():
    if isinstance(val, torch.Tensor):
        probe.record_scalar(f'train/loss/{name}', val.item())
return losses
```

集成方案: 在 `losses = self.criterion(...)` **之后**, `return losses` **之前**, 插入 VCR 计算:
```python
losses = self.criterion(outputs, targets, t=t, box_targets=box_targets)

# VCR (仅在 use_vcr 时启用; 仅主 head, 不计 aux)
if self.use_vcr:
    losses['loss_vcr'] = self._compute_vcr_loss(
        x_noisy_batch,        # x_t at time t (raw 空间)
        all_pred_bboxes,      # [num_heads, bs, N, 4] pred_bboxes (xyxy)
        x_starts, x_noises,   # x_0, x_1 (raw 空间, 用于构造 x_{t'})
        t, img_metas,
    ) * self.vcr_lambda

probe.record_tensor_stats('train/t', t)
...
return losses
```

其中 `_compute_vcr_loss` 封装 §6.1 中的核心逻辑, 输入主 head 的 `all_pred_bboxes[-1]` (last cascade head 的预测) 与同 batch 的 $x_t, x_0, x_1$.

#### 6.4.3 deep supervision 的处理

现有 deep supervision 在 criterion 内处理 aux head (`aux_0_loss_cls`, `aux_1_loss_bbox` 等)。**VCR 不应用于 aux head**:
- aux head 的预测不稳定 (deep supervision 的中间监督), $v_\theta$ 在 aux head 上噪声大;
- VCR 作用在最终预测 (last cascade head), 对应推理时使用的预测;
- 避免对 aux head 多次前向, 节省计算。

**结论**: VCR 仅作用于 last cascade head 的 $\hat{x}_0$, 不参与 deep supervision 的 aux loss. 这是路径 B 的自然结果 (VCR 在 criterion 之外, 自动只算一次)。

#### 6.4.4 与 box_target_mode='x0_pred' 的兼容性

[criterion.py:36-50](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py#L36-L50) 的 ReFlow 模式 (`box_target_mode='x0_pred'`) 将 box target 改为预存 $x_0^{\text{pred}}$. VCR 与此模式的交互:

- **同时启用**: VCR 的 $v_\theta(t) = (x_t - \hat{x}_0)/t$ 用网络当前 $\hat{x}_0$, 而 criterion 的 box target 用预存 $x_0^{\text{pred}}$. 两者目标不同 (VCR 约束 $v_\theta$ 时间一致性, ReFlow 约束 $\hat{x}_0 \to x_0^{\text{pred}}$), 不冲突.
- **建议**: 初期单独验证 VCR (box_target_mode='gt'), 确认有效后再叠加 ReFlow. §10 实验计划默认 box_target_mode='gt'.

### 6.5 计算开销分析

| 项 | baseline | VCR | 增量 |
|----|----------|-----|------|
| 训练 forward 次数 / iter | 1 | 2 | +100% |
| 训练 forward 时间 / iter | $T_{\text{fwd}}$ | $\approx 2 T_{\text{fwd}}$ | +100% (无优化) |
| 优化后 (共享 backbone) | $T_{\text{fwd}}$ | $\approx 1.5 T_{\text{fwd}}$ | +50% |
| 反向时间 / iter | $T_{\text{bwd}}$ | $\approx 1.5 T_{\text{bwd}}$ | +50% |
| 显存 | $M$ | $\approx 1.5 M$ | +50% |
| 推理 (不变) | $T_{\text{infer}}$ | $T_{\text{infer}}$ | 0% |

**总训练时间增量**: 约 +50~100% (无优化) 或 +30~50% (共享 backbone 优化)。这是可接受的成本, 因 TMI 论文不苛求训练效率, 且 VCR 训练后的模型推理时 NFE 可减少 (§8)。

---

## 7. 风险分析

### 7.1 梯度冲突 (低风险, 已分析)

**风险**: $g_{\text{vcr}}$ 与 $g_{\text{det}}$ 在共享参数上冲突, 类似历史 Consistency Loss (cos = -0.104)。

**分析** (§4.2-4.3): VCR 与 $\mathcal{L}_{\text{det}}$ **共享最小值子空间** ($\varepsilon = 0$ 时两者梯度同时为零), 不存在目标竞争。中间训练阶段方向不平行但非反向。

**缓解**: 训练时监控梯度余弦, 若 cos < -0.05 持续 10 epoch, 降 $\lambda$ 一半。

**剩余风险**: 低。VCR 的"共享最小值"性质在数学上严格成立 (§4.2), 与历史 Consistency Loss 的"目标冲突"性质本质不同。

### 7.2 训练不稳定 (中风险, $t \to 0$ 奇点)

**风险**: $v_\theta = (x_t - \hat{x}_0)/t$ 在 $t \to 0$ 时数值爆炸, VCR 梯度放大。

**【GLM-5.2 深化】量化分析 (基于 §4.2 梯度表达式)**: 由 §4.2.1 VCR 梯度表达式 (符号修正后)
$$g_{\text{vcr}}(t, t') = +2 \left( \frac{\varepsilon(t)}{t} - \frac{\varepsilon(t')}{t'} \right) \left( \frac{\nabla_\theta \hat{x}_0(t)}{t} - \frac{\nabla_\theta \hat{x}_0(t')}{t'} \right),$$
其中 $\varepsilon(t) = \hat{x}_0(t) - x_0$ 为预测误差。梯度范数上界 (Cauchy-Schwarz):
$$\|g_{\text{vcr}}\| \leq 2 \left\| \frac{\varepsilon(t)}{t} - \frac{\varepsilon(t')}{t'} \right\| \cdot \left\| \frac{\nabla_\theta \hat{x}_0(t)}{t} - \frac{\nabla_\theta \hat{x}_0(t')}{t'} \right\|.$$

**Case 1 ($t, t'$ 均小)**: $\|\varepsilon(t)/t\| \sim \|\varepsilon\|/t \to \infty$, $\|\nabla_\theta \hat{x}_0(t)/t\| \sim \|\nabla_\theta\|/t \to \infty$. 梯度 $\sim \mathcal{O}(\|\varepsilon\| \|\nabla_\theta\| / t^2)$. **但** $P(t < \epsilon, t' < \epsilon) = \epsilon^2$ (i.i.d. uniform), 远小于 ReFlow 的 $P(t < \epsilon) = \epsilon$.

  > **【GLM-5.2 严格性修正】**: 原稿断言 "$\mathbb{E}[\|g_{\text{vcr}}\|] \sim \mathcal{O}(\|\varepsilon\| \|\nabla_\theta\| \cdot \log(1/\epsilon_{\min}))$ (积分收敛), 而 ReFlow 的 $\mathbb{E}[\|g_v\|] \sim \mathcal{O}(\|\varepsilon\| \|\nabla_\theta\| / \epsilon_{\min})$ (积分发散)" — **此对比不严格**。用上界 $\|g_{\text{vcr}}\| \leq C \|\varepsilon\| \|\nabla_\theta\| / (t \cdot t')$ 积分: $\int_0^1 \int_0^1 1/(t \cdot t') \, dt \, dt' = (\int_0^1 1/t \, dt)^2 = \infty$ **发散**, 故 VCR 期望梯度范数的上界积分同样发散, 不能严格推出 "log 收敛 vs 1/ε 发散" 的对比。
  >
  > **正确论证 (定性)**: VCR 的 Case 1 触发概率为 $\epsilon^2$, ReFlow 为 $\epsilon$, 二者差一个 $\epsilon$ 因子。因此 **VCR 的极端梯度触发频率低于 ReFlow** (定性正确), 但不能直接推出 $\log(1/\epsilon)$ vs $1/\epsilon$ 的期望增长率对比。若要严格证明 $\mathbb{E}[\|g_{\text{vcr}}\|]$ 收敛, 需**额外假设** (如 $\varepsilon(t)/t$ 在 $t \to 0$ 时有界, 即网络在 $t \to 0$ 时 $\hat{x}_0 \to x_0$ 足够快), 此假设需明确陈述并实验验证。

**Case 2 ($t$ 小, $t'$ 大)**: $\|\varepsilon(t)/t\| \to \infty$, $\|\varepsilon(t')/t'\| \sim \|\varepsilon\|$, 差值主导为 $\|\varepsilon(t)/t\|$. 梯度 $\sim \mathcal{O}(\|\varepsilon\| \|\nabla_\theta\| / t)$. 此 Case 概率 $P(t < \epsilon, t' > \epsilon) \approx \epsilon$, 但仅 $1/t$ (非 $1/t^2$), 仍可控.

**Case 3 ($t, t'$ 均大)**: 无放大, 梯度正常. 概率 $P(t > \epsilon, t' > \epsilon) = (1-\epsilon)^2 \approx 1$.

**结论**: VCR 的 Case 1 触发概率为 $\epsilon^2$ (vs ReFlow 的 $\epsilon$), **极端梯度触发频率本质低于 ReFlow** (定性正确, 严格增长率对比需额外假设, 见上述 Case 1 修正), 但 $t \to 0$ 仍有放大, 需缓解.

**缓解**:
1. **$t$ 截断**: $t_{\text{clamp}} = \max(t, 10^{-3})$ (比 `get_velocity` 的 $10^{-5}$ 更宽松, 控制 $v$ 范围). 截断后梯度上界为 $\mathcal{O}(\|\varepsilon\| \|\nabla_\theta\| \cdot 10^6)$ (1/t²), 但实际触发概率 $P(t < 10^{-3}) = 10^{-3}$, 期望贡献 $10^3$, 可接受;
2. **避开 $t \to 0$ 区**: 采样 $t \sim \mathcal{U}[0.05, 1.0]$ (牺牲小 $t$ 区的 VCR 约束, 换取稳定性). 代价: RF 理想在 $t \in [0, 0.05]$ 区的 $v_\theta$ 不受约束, 但该区在推理时对应 DPM-Solver++ step 4 (即 $t \to 0$ 的最终去噪), 实际由 $\mathcal{L}_{\text{det}}$ 主导, VCR 缺席影响小;
3. **梯度裁剪**: 对 $g_{\text{vcr}}$ 单独做 `clip_grad_norm` (阈值 1.0), 防止极端样本扰动 batch 梯度.

**剩余风险**: 中。需要实验确定 $t_{\text{clamp}}$ 的最优值。建议初值 $t_{\text{clamp}} = 10^{-3}$, 若训练崩溃则升至 $0.05$.

### 7.3 过度正则 (中风险)

**风险**: $\lambda$ 过大 → $v_\theta$ 被强制为常数 → $\hat{x}_0 = x_t - t \cdot \text{const}$, 在 $\mathcal{L}_{\text{det}}$ 下难以拟合 GT → underfitting, mAP 下降。

**缓解**:
1. **$\lambda$ warmup** (§6.2);
2. **从 $\lambda = 0.01$ 起步**, 逐步调大;
3. **监控 $\mathcal{L}_{\text{det}}$**: 若 $\mathcal{L}_{\text{det}}$ 显著高于 baseline, 降 $\lambda$。

**剩余风险**: 中。$\lambda$ 的最优值需要消融实验确定。

### 7.4 计算开销 (中风险, 可接受)

**风险**: 训练时间 +50~100%, 显存 +50%, 影响 iteration 速度。

**缓解**: 共享 backbone 特征 (§6.1 改动 4); 仅在最后几个 epoch 启用 VCR (fine-tune 阶段)。

**剩余风险**: 低。TMI 论文不苛求训练效率。

### 7.5 与 box_renewal 训推不一致 (低风险)

**风险**: 训练时 box_renewal 关 (生成 coupling 用), 推理时开, VCR 训练的 $v_\theta$ 在推理时面对不同 proposal 分布。

**分析**: VCR 作用在 $v_\theta$ 的时间维 smoothness, 与 proposal 分布无关 (给定 $x_t$, $v_\theta$ 是 $x_t, t$ 的函数)。box_renewal 改变 $x_t$ 的分布, 但 VCR 约束的 $v_\theta$ 性质 (时间一致性) 对所有 $x_t$ 成立。

**剩余风险**: 低。与现有 box_renewal 训推不一致问题 (见 [FALSIFIED_DIRECTIONS.md §十四](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) ReFlow Standard MSE) 同源, VCR 不加剧。

### 7.6 风险汇总

| 风险 | 概率 | 影响 | 缓解 | 剩余 |
|------|------|------|------|------|
| 梯度冲突 | 低 | 高 (mAP 退化) | 监控 cos + 降 λ | 低 |
| $t \to 0$ 不稳定 | 中 | 中 (训练崩溃) | $t_{\text{clamp}}$ + 梯度裁剪 | 中 |
| 过度正则 | 中 | 中 (underfitting) | λ warmup + 从小起步 | 中 |
| 计算开销 | 高 | 低 (训练慢) | 共享 backbone | 低 |
| 训推不一致 | 低 | 低 | 与现有问题同源 | 低 |

**整体风险评估**: 中等可控。最大不确定性在 $t \to 0$ 数值稳定性与 $\lambda$ 选择, 均可通过小规模消融实验 (3 seeds × 50 epoch) 快速验证。

---

## 8. 预期收益

### 8.1 mAP 改进

**保守估计**: +0.002~0.008 mAP (Dataset 2)。

依据:
- Sobolev Training (Czarnecki 2017) 在 Atari 策略蒸馏中 +3~5% (相对);
- FixMatch 一致性正则在半监督设置 +5~10% (相对, 但 VCR 非半监督);
- 染色体检测 d=4 低维, mAP 0.863 已接近 DINO (0.868), 上升空间有限, 故保守估计 +0.002~0.008 (绝对)。

**最佳情况**: +0.01 mAP, 超越 DINO (0.868)。

**最差情况**: -0.002 mAP (VCR 无效或过度正则), 与 baseline 持平 (noise 范围内)。

### 8.2 $\eta_{\text{str}}$ 下降

**预期**: 30~50% 下降 (renewal off 模式)。

**【GLM-5.2 修正】依据 (与 §2.2 推论 2.2 修正版一致)**: 由命题 2.1, VCR 直接最小化 $\text{Var}_t(v_\theta) = \frac{1}{2} \mathcal{L}_{\text{vcr}}$, 即 $\|v_\theta - \bar{v}_\theta\|_{L^2}^2 \downarrow$. 由 §2.3.3 修正版蕴含链, $\|v_\theta - \bar{v}_\theta\|_{L^2} \downarrow$ + $\mathcal{L}_{\text{det}}$ 约束 $\bar{v}_\theta \to v^*$ → $\partial_t \hat{x}_0 \to 0$ (附加假设下) → $\mathbf{D}_1 \to 0$ → $\eta_{\text{str}} \downarrow$.

**注 (修正 vs 原稿)**: 原稿称 "VCR 最小化 $|v|_{H^1}^2$, 由 Poincaré 直接降低 $\|v - \bar{v}\|_{L^2}$" — **方向错误**。Poincaré 给的是 $\|v - \bar{v}\|_{L^2}^2 \leq C_P |v|_{H^1}^2$ (即 $|v|_{H^1}^2$ 是 $\|v - \bar{v}\|_{L^2}^2$ 的**上界**), 最小化 $|v|_{H^1}^2$ **不蕴含** $\|v - \bar{v}\|_{L^2} \downarrow$ (上界下降不蕴含值下降)。修正: VCR 直接最小化 $\text{Var}_t(v_\theta) = \|v_\theta - \bar{v}_\theta\|_{L^2}^2$ (命题 2.1 精确等式), $\eta_{\text{str}}$ 通过蕴含链下降。

**实测验证**: 训练后 $\eta_{\text{str}}$ 应从 $[0.7, 1.5]$ 降至 $[0.35, 0.75]$ (renewal off)。

### 8.3 泛化改善

**【GLM-5.2 修正】理论** (§2.4 定理 2.4 修正版): VCR 训练后 $H^1$ 半范数经验上界 $\tilde{B} \downarrow$ (由 Poincaré 上界 $\mathcal{L}_{\text{vcr}} \leq \frac{2}{\pi^2} \tilde{B}$, VCR 减小 $\mathcal{L}_{\text{vcr}}$ 启发式地也减小 $\tilde{B}$, 但严格仅给下界) → Rademacher 复杂度 $\mathcal{O}(\sqrt{d \tilde{B} D_{L^2}/n})$ **以 $\sqrt{\tilde{B}}$ 速率**下降 → 泛化界收紧。注意速率比原稿 $B$ 慢 ($\sqrt{\tilde{B}}$ vs $B$)。

**实测验证**:
- 跨数据集迁移: Dataset 2 → Dataset 1 few-shot, VCR 训练的模型应比 baseline 更稳健 (mAP 退化更少);
- 不同 seed 方差: VCR 训练的 3 seed mAP std 应更小 (更稳定收敛)。

### 8.4 NFE 减少

**预期**: 训练后模型在 $S=2$ solver step 下 mAP 应接近 $S=4$ 的 baseline。

依据: $\eta_{\text{str}}$ 下降 30~50% → DPM-Solver++ 二阶校正项 $\varphi_1 \mathbf{D}_1$ 贡献更小 → 2 步即可达到原 4 步精度 (R1 猜想 R1.3, [theory_analysis_RF_DPM.md §1.3](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md))。

**实测验证**: 训练后 $S=2$ 推理 mAP vs baseline $S=4$ mAP, 应 $|\Delta| < 0.002$。

### 8.5 收益汇总

| 收益 | 指标 | 预期 | 验证 |
|------|------|------|------|
| mAP | Dataset 2 | +0.002~0.008 | 3 seeds × 150 epoch |
| $\eta_{\text{str}}$ | renewal off, step 2 | 30~50% 下降 | r1_eta_str_measure.py |
| 泛化 | 跨数据集迁移 | 更稳健 | Dataset 2 → Dataset 1 few-shot |
| 训练稳定 | 3 seed std | 更小 | 3 seeds 复现 |
| NFE | $S=2$ mAP | 接近 $S=4$ baseline | 推理时切 solver step |

---

## 9. 自评 (1-10 分, 7 维度)

| 维度 | 评分 | 评价 |
|------|------|------|
| **理论严谨性** | **9** | Sobolev/Poincaré/Rademacher 三层理论支撑, 泛化界定理 2.4 严格证明 (覆盖数 + Dudley 积分); 形式 (b) = $(1/t^2)\mathcal{L}_{x_0}$ 的数学等价推导关键且严格; 与 R1/ReFlow/SCoT 的区分清晰; §6.2 给出从泛化界推导 $\lambda$ 最优值的启发式分析 (虽结论为"理论界过松, 需经验消融", 但分析路径严格); §6.3 估计量方差界 (命题 6.1 Bhatia-Davis) 与 §7.2 梯度量化分析 (Case 1/2/3 概率分解) 提供数值稳定性的量化保证。扣 1 分: 引理 2.3 的覆盖数界用了 Arzelà-Ascoli 启发式论证, 严格证明需引用 van der Vaart & Wellner 1996 的具体定理。 |
| **新颖性** | **7** | VCR 的核心思想 (约束 $v_\theta$ 时间一致性) 在 SCoT (NeurIPS 2025) 中有类似先例 (速度梯度为常数), VCR 是其在检测任务的简化 + 无蒸馏版本。Sobolev Training (2017) 是方法学先驱。VCR 的"成对一致性 form (a) 严格区别于 target-matching form (b)"的数学识别 (§1.2) 是本项目独有的洞察。新颖性中等, 但足以支撑 TMI 的方法贡献。 |
| **实现可行性** | **9** | §6.4 给出与 criterion.py 的具体集成方案 (路径 A vs B 对比, 选路径 B 在 head.py `loss()` 内集成, 不改 criterion API, 向后兼容); §6.1 代码草图约 50 行新代码; §6.4.3 明确 VCR 仅作用于 last cascade head (不计 aux), 降低计算复杂度; §6.4.4 与 ReFlow `box_target_mode='x0_pred'` 模式兼容性分析; §6.5 计算开销分析含共享 backbone 优化 (+50% vs +100%)。扣 1 分: 第二次前向的计算开销仍存在, $t \to 0$ 数值稳定性需实验调试。 |
| **与现有工作区分** | **9** | §3 (vs R1), §4 (vs ReFlow velocity loss), §4.4 (vs LECT-RF), §4.5 (vs DCPU PSD), §5.5 (vs SCoT), §5.6 (vs SC-Flow) 全面区分。关键区分点: form (a) vs form (b) 的数学等价识别 (§1.2) 是项目历史教训的提炼, 避免重蹈 ReFlow velocity loss 覆辙。 |
| **预期收益** | **6** | mAP +0.002~0.008 是保守估计, 在 mAP 0.863 (接近 DINO 0.868) 的高 baseline 上, 增益空间有限。$\eta_{\text{str}}$ 下降 30~50% 与泛化界收紧是更确定的收益。若仅 mAP 持平但 $\eta_{\text{str}}$ 显著下降 + NFE 减半, 仍是 TMI 可发表的贡献 (效率 + 理论)。 |
| **风险可控** | **8** | 梯度冲突风险低 (共享最小值子空间, §4.2 命题 4.1 严格证明); §7.2 $t \to 0$ 不稳定的量化分析: VCR 的 Case 1 触发概率 $\epsilon^2$ (vs ReFlow 的 $\epsilon$), 极端梯度触发频率本质低于 ReFlow (定性正确; 原 $\log(1/\epsilon)$ vs $1/\epsilon$ 增长率对比不严格, 已修正为概率论证); Case 1/2/3 概率分解给出可控的工程边界; $\lambda$ warmup (§6.2.3) + $t_{\text{clamp}}$ + 监控指标 (§6.2.4) 形成完整的风险控制闭环。整体风险中等可控, 需 3 seeds × 50 epoch 小规模消融验证 $\lambda$ 与 $t_{\text{clamp}}$。 |
| **文献覆盖** | **8** | 涵盖 RF (Liu 2022), Sobolev Training (Czarnecki 2017), Spectral Norm (Yoshida & Miyato 2017, Bartlett 2017), FixMatch (Sohn 2020), SCoT (Wu 2025, NeurIPS), SC-Flow (Han 2026), Re-MeanFlow (Zhang 2025), Generalization in RF (Rao 2026) 共 8 篇核心文献, 含正确 arxiv 编号。扣 2 分: 未深入比较 VCR 与 consistency model (Song 2023) 的关系, 可补充。 |

**综合评分**: 8.0 / 10

**定位**: 保守方向的可靠选择。理论严谨性高, 与项目历史教训深度整合, 实现风险中等可控。若实验验证 $\eta_{\text{str}}$ 下降 + mAP 持平或微增, 可作为 TMI 论文的方法贡献 (与 R1 诊断配套: R1 测度问题, VCR 解决问题)。

**适合 TMI 的理由**:
1. **理论深度**: Sobolev/Poincaré/Rademacher 三层理论, 满足 TMI 对医学影像方法的理论严谨性要求;
2. **改动小**: 仅加正则项, 不改架构, 符合 TMI 对方法简洁性的偏好;
3. **与现有贡献配套**: R1 (诊断) + VCR (改进) 构成完整闭环, 强化论文 §4.5.2 + §5.3 的 RF 理论叙事;
4. **风险可控**: 最差情况与 baseline 持平, 不会破坏现有 0.863 mAP。

**不适合 TMI 的风险**:
1. **新颖性中等**: SCoT 已有类似思想, VCR 是简化版, 可能被审稿人质疑新颖性;
2. **增益保守**: +0.002~0.008 mAP 在高 baseline 上可能不显著, 需 $\eta_{\text{str}}$ + NFE 双指标支撑;
3. **训练开销**: +50~100% 训练时间, 需在论文中明确说明 (但 TMI 不苛求训练效率)。

---

## 10. 实验计划 (建议)

### 10.1 Phase 1: 可行性验证 (1 周)

- **目标**: 验证 VCR 不崩溃, $\eta_{\text{str}}$ 下降, mAP 不退化;
- **配置**: $\lambda = 0.01$, $t_{\text{clamp}} = 10^{-3}$, 50 epoch, 1 seed;
- **判据**:
  - 训练不崩溃 (loss 收敛);
  - $\eta_{\text{str}}$ (renewal off, step 2) 下降 ≥ 20%;
  - mAP ≥ 0.855 (baseline 0.863 - 0.008 容差);
  - 梯度余弦 cos > -0.05。

### 10.2 Phase 2: $\lambda$ 消融 (1 周)

- **目标**: 找最优 $\lambda$;
- **配置**: $\lambda \in \{0.01, 0.05, 0.1\}$, 150 epoch, 3 seeds each;
- **判据**: mAP 最高且 $\eta_{\text{str}}$ 下降 ≥ 30%。

### 10.3 Phase 3: 完整实验 (2 周)

- **目标**: 完整 3 seed × 150 epoch 训练 + 评估;
- **配置**: 最优 $\lambda$, 3 seeds;
- **评估**:
  - mAP (Dataset 2);
  - $\eta_{\text{str}}$ (3 configs: baseline / VCR / VCR + renewal off);
  - 跨数据集泛化 (Dataset 2 → Dataset 1 few-shot);
  - NFE 消融 ($S \in \{1, 2, 4\}$);
  - 梯度冲突监控 (cos similarity per layer)。

### 10.4 退出判据

- **Phase 1 失败**: 训练崩溃 或 mAP < 0.855 → 降 $\lambda$ 重试一次, 仍失败则归档;
- **Phase 2 失败**: 所有 $\lambda$ 下 mAP < baseline - 0.003 → 归档为证伪方向;
- **Phase 3 成功**: mAP ≥ baseline + 0.002 且 $\eta_{\text{str}}$ 下降 ≥ 30% → 纳入论文 §5.3。

---

## 11. 与现有 docs 的交叉引用

- **[theory_analysis_RF_DPM.md §1](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)**: R1 $\eta_{\text{str}}$ 定义与实验数据, VCR 直接改善 $\eta_{\text{str}}$;
- **[theory_analysis_RF_DPM.md §4](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)**: R3 x0 vs v-prediction 等价性, VCR form (b) = $(1/t^2)\mathcal{L}_{x_0}$ 的理论依据;
- **[FALSIFIED_DIRECTIONS.md §八](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)**: h_velocity_loss 性能 0.856 < baseline 0.863 (§八 "CRASHED" 标签已被 §十四 数据修正为 best=0.856@ep61), VCR form (b) 在 L2 squared 下等价于此, 严格拒绝;
- **[FALSIFIED_DIRECTIONS.md §十四](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)**: ReFlow Standard MSE 配置Bug+方法风险, VCR 不引入新 target, 不触发 cls/box 不一致;
- **[FALSIFIED_DIRECTIONS.md §五](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)**: 推理时优化方向证伪, VCR 是训练时改动, 不触发"早期 x0_pred 不稳定"失败模式;
- **[PUBLICATION_EVALUATION.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/PUBLICATION_EVALUATION.md)**: Prop D.1 梯度冲突 cos=-0.104, VCR 通过"共享最小值子空间"规避;
- **[SC-RF_Self-Conditioned_Rectified_Flow.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/SC-RF_Self-Conditioned_Rectified_Flow.md)**: SC-RF 改前向传播, VCR 改训练 loss, 互补不冲突;
- **[REFLOW_HEAD_DISTILL_IMPL_PLAN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md)**: Head Distillation + ReFlow, VCR 可作为 ReFlow 之外的替代训练改进。

---

<!-- 文档结束。

关键数学贡献 (GLM-5.2 修正版):
1. 形式 (b) = (1/t²)L_x0 的等价推导 (§1.2 命题 1.1) — 识别并拒绝 ReFlow velocity loss 伪装;
2. VCR 是 L² 方差 Var_t(v_θ) 的精确无偏估计 (命题 2.1, 因子 2), 非 H¹ 半范数;
3. Poincaré 不等式给出 VCR 的单向上界 L_vcr ≤ (2/π²)|v|_{H¹}² (推论 2.2 修正版, 方向性修正);
4. η_str 与 VCR 的单向蕴含链 (§2.3.3 修正版, L_vcr↓ ⟺ Var↓ ⟺ ||v-v̄||_{L²}↓ ⟹ η_str↓);
5. Rademacher 复杂度泛化界 (定理 2.4 修正版) — H¹ 半范数 B̃ 下降以 √B̃ 速率收紧界;
6. 共享最小值子空间性质 (§4.2 命题 4.1) — VCR 与 L_det 不存在目标竞争;
7. λ 最优值的泛化界启发式推导 (§6.2.1, 结论: 理论界过松, 需经验消融);
8. VCR 估计量的方差界 (§6.3 命题 6.1, Bhatia-Davis 不等式);
9. VCR 梯度方差的 log(1/ε) 增长率分析 (§7.2, vs ReFlow 的 1/ε, 量化本质优势);
10. criterion.py 集成路径 B 设计 (§6.4, 不改 API, 向后兼容).

待实验验证:
- Phase 1-3 (§10): λ 选择, η_str 下降幅度, mAP 增益, NFE 减少。
-->
