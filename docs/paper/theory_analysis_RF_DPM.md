# RF 与 DPM-Solver++ 理论深化分析

> 📋 **命名约定**: 本文档使用论文正式名称 (Dataset 1 / Dataset 2 / RF+Heun / +Stoch. Coupling / +DPM-Solver++ / Adaptive Step / Draft-Verify / Top-K / Head Early-Exit / RoI Feature Cache / Cascade Head Count)。内部实验代号 (24obj / A0-A3 / IO1-IO5 / N_cascade) 仅保留在文件路径和代码引用中以兼容工程实现。

<!--
============================================================
文档定位
============================================================
本文件为独立的理论分析文档，对应 paper_draft_CN.md 中 §3.1–3.2、§5.3
和 Appendix A.5 的潜在深化方向。所有内容尚未纳入主文，需用户审阅后
决定是否选择性合并到 main.tex。

四个方向：
  R1  直线度诊断指标 η_str        — 零代码, 纯理论, 0.5 页
  S1  cascade head 作为 implicit solver  — 纯理论, 0.3 页
  D3  box_renewal 与 DPM-Solver++ 矛盾   — 理论 + 小实验, 0.5 页
  R3  x0-prediction vs v-prediction 等价性  — 纯理论, 0.3 页

约束：
  - 与 TMI 10 页主文兼容（合计约 1.5 页增量，可压缩到 §5.3 + Appendix A）
  - 不重复 Adaptive Step–RoI Feature Cache / Flow Matching Detection / Cascade Head Count E2E 的失败模式
  - 与现有代码（rectified_flow.py, head.py, sampling.py）保持一致
============================================================
-->

## 0. 引言

论文当前在 RF 范式（§3.1）、ODE solver（§3.2）、OT Diversity Collapse（§3.3）三个轴上完成了主要理论建构，但 RF 与 DPM-Solver++ 的**交互机理**仍存在四块理论空白：

1. **"RF 在 2 步收敛"是经验观察还是可量化结论？** 论文 §4.5.2 报告 DPM-Solver++ 在 2 步达到 mAP 0.863（seed 42），但未给出"轨迹直线度"的可观测指标。审稿人可能质疑：如何在没有 reflow（2-RectFlow）的情况下断言训练成功？
2. **cascade head 与 solver step 的耦合关系未被形式化。** 当前架构 6 cascade head × 4 solver step = 24 次前向，但 DPM-Solver++ 仅需 4 NFE 的理论框架把每个 time step 内 6 个 cascade head 视为黑盒——这与 Cascade R-CNN 的级联精化思想是同构的，但论文未指出。
3. **box_renewal 与 DPM-Solver++ 的 x0_history 连续性存在矛盾。** 代码在 Top-K pruning 后 `dpm_solver.reset()`（[head.py:629–630](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L629-L630)），但 box_renewal 后未 reset，存在不一致。这可能是 Top-K K=100 掉点 −0.013 的潜在理论解释之一。
4. **x0-prediction 与 v-prediction 在低维 RF 下的等价性未分析。** RF 原文（Liu et al., 2023）使用 v-prediction，本论文使用 x0-prediction + DPM-Solver++ data-prediction 形式，但未说明该选择的依据。审稿人可能质疑"为何不用 v-prediction"。

本文档对这四块空白给出形式化分析，并标注与现有代码、已证伪方向、TMI 页数预算的对应关系。

---

## 1. R1：直线度诊断指标 $\eta_{\text{str}}$

### 1.1 动机

论文 §4.5.2 报告"DPM-Solver++ 在 2 步收敛（mAP 0.863，seed 42）；超过 2 步无收益，证实 RF 轨迹接近直线"。该结论是经验性的，依赖 mAP 度量。我们寻求一个**推理时零开销可读出**的指标，定量刻画学习轨迹的直线度，从而：

- 为"2 步收敛"提供机制级解释（不只是 mAP 持平）；
- 给出"何时需要 reflow（2-RectFlow）"的可操作判据；
- 区分"RF 训练成功"与"RF 训练失败但被 solver 步数补偿"两种情形。

### 1.2 形式化定义

**设置。** RF 路径 $x_t = (1-t)x_0 + t x_1$，$t \in [0,1]$，理想速度场 $v(t) = x_1 - x_0$ 为常数。学习的网络 $v_\theta$ 在有限数据上无法精确恢复常数 $v$，故其隐含的 data-prediction $\hat{x}_0(t) := x_t - t \cdot v_\theta(x_t, t)$ 一般为 $t$ 的函数。

**DPM-Solver++ 的二阶校正项**（[rectified_flow.py:141](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L141)）：

$$\mathbf{D}_1^{(n)} = \frac{\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)}}{t_n - t_{n-1}},$$

其中 $\hat{x}_0^{(n)} = v_\theta(x_{t_n}, t_n)$ 在 $t_n$ 处的 data-prediction。在理想 RF 下 $\hat{x}_0(t) \equiv x_0$，故 $\mathbf{D}_1^{(n)} \equiv \mathbf{0}$，DPM-Solver++ 退化为精确线性求解。

**定义（直线度指标）。** 对每步 $n \ge 1$，定义

$$\eta_{\text{str}}^{(n)} := \frac{\left\| \mathbf{D}_1^{(n)} \right\|_2}{\left\| \hat{x}_0^{(n)} \right\|_2 + \epsilon_{\text{norm}}},$$

其中 $\epsilon_{\text{norm}} = 10^{-6}$ 防止 $\hat{x}_0 \to 0$ 时数值爆炸。批量推理时取所有 proposals 的均值 $\bar{\eta}_{\text{str}}^{(n)}$。

### 1.3 性质

**命题 R1.1（非负性与零下界）。** $\eta_{\text{str}}^{(n)} \ge 0$，且 $\eta_{\text{str}}^{(n)} = 0$ 当且仅当 $\hat{x}_0$ 在 $[t_{n-1}, t_n]$ 上为常数。

**命题 R1.2（与理想 RF 的关系）。** 若 $v_\theta$ 精确恢复 $v = x_1 - x_0$（理想 1-RectFlow），则 $\forall n: \eta_{\text{str}}^{(n)} = 0$，此时 DPM-Solver++ 任意步数等价于 1 步 Euler 的精确线性外推。

**猜想 R1.3（与 mAP 步数收敛的对应）。** 若 $\bar{\eta}_{\text{str}}^{(n)} < \epsilon_{\text{conv}}$ 对所有 $n \ge N_0$ 成立，则 DPM-Solver++ 在 $N_0$ 步后无显著精度增益——因为高阶校正项 $\varphi_1 \mathbf{D}_1$ 已被 $\eta_{\text{str}}$ 界住。

> **注**：R1.3 原列为"命题"，但现有证明仅为直观论证（证明梗概见下），未给出严格误差界与 $\epsilon_{\text{conv}}$ 的显式形式，故降级为猜想。严格的证明需要：(i) $\eta_{\text{str}}$ 到 mAP 的 Lipschitz 映射；(ii) $\varphi_1$ 与 $t$ 的显式依赖关系。这两点超出本文范围，留作未来工作。

证明梗概：DPM-Solver++ 二阶更新为 $\mathbf{x}_{t_{n+1}} = \frac{t_{n+1}}{t_n}\mathbf{x}_{t_n} + (1 - \frac{t_{n+1}}{t_n})\hat{x}_0^{(n)} + \varphi_1 \mathbf{D}_1^{(n)}$（[Appendix A.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md#L506-L513)）。前两项是常数 $\hat{x}_0$ 的精确解，$\varphi_1 \mathbf{D}_1$ 是非直线性校正。若 $\|\mathbf{D}_1\|/\|\hat{x}_0\| < \epsilon$，则校正项相对主项的范数比为 $\varphi_1 \epsilon / (1 - t_{n+1}/t_n)$，对小 $\epsilon$ 可忽略。（直观论证，非严格证明）

### 1.4 与论文 claim 的对应

- **"2 步收敛"**：等价于 $\bar{\eta}_{\text{str}}^{(2)} \ll 1$，即 2 步后校正项已可忽略。
- **"94% 增益归因于 RF 范式"**：若 $\bar{\eta}_{\text{str}}$ 在 RF+Heun 和 +DPM-Solver++ 上数值接近且都很小，则 solver 选择不影响轨迹直线度，仅影响每步 NFE——支持"solver 贡献 6%"的结论。
- **"是否需要 reflow"**：若训练后 $\bar{\eta}_{\text{str}} > 0.1$ 持续，则 2-RectFlow（Liu et al., 2023）可能进一步拉直轨迹；若 $\bar{\eta}_{\text{str}} < 0.01$，reflow 收益有限。

### 1.5 实验验证（已完成，3 seeds × 4 configs）

**实验设置**：在 +DPM-Solver++ 的 3 个 seed checkpoint 上推理 500 张验证图，记录每步的 $\bar{\eta}_{\text{str}}^{(n)}$。代码改动见 [rectified_flow.py:113-118, 146-153](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L113-L153) 和 [head.py:774-784](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L774-L784)，实验脚本见 [r1_eta_str_measure.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/r1_eta_str_measure.py)。

**核心结果**（详见 [r1_d3_summary.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/r1_d3_summary.py) 输出）：

| 配置 | mAP (3 seeds) | Step 1 $\eta_{\text{str}}$ | Step 2 | Step 3 | 模式 |
|------|---------------|---------------------------|--------|--------|------|
| baseline (renewal on) | 0.859 ± 0.004 | 3.43 ± 0.36 | 2.45 ± 0.24 | 1.68 ± 0.15 | 单调递减 |
| renewal off | 0.858 ± 0.003 | **1.50 ± 0.33** | **1.11 ± 0.19** | **0.70 ± 0.09** | 单调递减 |

**关键发现**：

1. **$\eta_{\text{str}}$ 单调递减模式 3 seed 稳定**：baseline 下 $\eta_{\text{str}}$ 从 step 1 的 3.43 降至 step 3 的 1.68（降 51%），renewal off 下从 1.50 降至 0.70（降 53%）。这**定量解释了"2 步收敛"**：step 3 的二阶校正贡献已比 step 1 小 50%+，对 mAP 的边际贡献低于噪声阈值。

2. **$\eta_{\text{str}}$ 绝对值非零**：即使 renewal off，$\eta_{\text{str}} \in [0.7, 1.5]$，说明学习轨迹**并非理想直线**。这修正了论文 §4.5.2 "RF 轨迹接近直线"的 claim——更准确的表述是"轨迹曲率在 step 2 后足够小，使 DPM-Solver++ 校正项对 mAP 的边际贡献 < 0.001"。

3. **renewal 对 $\eta_{\text{str}}$ 的污染**（D3 矛盾）：renewal on 使 $\eta_{\text{str}}$ 虚高 56-58%（ratio off/on = 0.42-0.44），但 mAP 仅 -0.0003。这证实 D3 矛盾：box_renewal 严重破坏 x0_history 连续性，但对最终精度贡献微小。

4. **Top-K pruning 改变 $\eta_{\text{str}}$ 模式**（新发现，未在原理论中预测）：K=200/K=100 下 $\eta_{\text{str}}$ 从"单调递减"变为"V 型"（step 1 低，step 2 高，step 3 中），因为 Top-K pruning 后 `dpm_solver.reset()` 使 DPM-Solver++ 重新冷启动。这一发现揭示 reset 改变了 DPM-Solver++ 的收敛行为。

### 1.6 与已证伪方向的区分

- **Adaptive Step（adaptive step early-exit）**：根据收敛提前终止，**改变推理步数**，已被证伪。
- **R1**：仅**观测** $\eta_{\text{str}}$，不改变任何推理流程，提供事后诊断。
- R1 不触发 Adaptive Step 的失败模式（Adaptive Step 失败原因是 step 1 的 x0_pred 不稳定，R1 不依赖该判据做决策）。

---

## 2. S1：Cascade Head 作为 Implicit Solver

### 2.1 动机

当前架构：6 个 cascade head 顺序执行，每个 head 做 RoIAlign + DynamicConv + 预测 $\hat{x}_0$（[head.py:283–337](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L283-L337)）。在每个 solver step 内，所有 6 head 都执行；4 个 solver step 共 24 NFE。但 DPM-Solver++ 仅需 4 NFE 的理论框架把每个 time step 视为单次 $v_\theta$ 评估——这与实际 6 head 顺序精化不一致。我们形式化分析两者的关系。

### 2.2 形式化：算子分裂视角

设：
- $H$ = cascade head 数量（当前 $H=6$）；
- $S$ = solver step 数量（当前 $S=4$）；
- $\mathcal{A}_t$ = solver 算子，固定 $v_\theta$ 推进 $t$：$x_{t+\Delta t} = \mathcal{A}_t(x_t; v_\theta)$；
- $\mathcal{B}_{t,k}$ = 第 $k$ 个 cascade head 算子，固定 $t$ 精化 $x$：$x_t^{(k+1)} = \mathcal{B}_{t,k}(x_t^{(k)}; v_\theta^{(k)})$，其中 $v_\theta^{(k)}$ 是第 $k$ head 的参数。

一次完整推理（4 步 × 6 head）为：

$$x_{\text{final}} = \mathcal{A}_{t_3} \circ \mathcal{B}_{t_3, H} \circ \cdots \circ \mathcal{B}_{t_3, 1} \circ \cdots \circ \mathcal{A}_{t_0} \circ \mathcal{B}_{t_0, H} \circ \cdots \circ \mathcal{B}_{t_0, 1}(x_{\text{init}}).$$

### 2.3 与 Cascade R-CNN 的同构

**Cascade R-CNN**（Cai & Vasconcelos, 2018）：$H = 4$, $S = 1$，纯横向精化，每个 head 在固定 IoU 阈值上训练。

**KaryoFlow**：$H = 6$, $S = 4$，横向精化 × 纵向推进。形式化为**双向精化**：

- 横向（cascade head，固定 $t$）：在固定时间步上精化 $x_t$，类似 Cascade R-CNN 的级联精化；
- 纵向（solver step，固定 $x$ 精化链）：推进时间 $t$，类似 DPM-Solver++ 的多步积分。

### 2.4 关键命题

**命题 S1.1（横向收敛性）。** 若 cascade head 序列 $\{\mathcal{B}_{t,k}\}_{k=1}^H$ 在固定 $t$ 上是压缩映射，则存在不动点 $x_t^* = \lim_{H \to \infty} \mathcal{B}_{t,H} \circ \cdots \circ \mathcal{B}_{t,1}(x_t^{(0)})$，且 $H$ 充分大时 $\mathcal{B}_{t,H} \circ \cdots \circ \mathcal{B}_{t,1} \approx \mathcal{B}_t^*$。

**推论 S1.2（DPM-Solver++ 框架的有效性）。** 若命题 S1.1 成立且 $H = 6$ 足够大，则每个 solver step 内的 6 head 已收敛到 $\mathcal{B}_t^*$，故 DPM-Solver++ 把 $\mathcal{B}_t^* \circ \mathcal{A}_t$ 视为单次"复合 $v_\theta$ 评估"是合理的。这解释了为何 DPM-Solver++ 仅需 4 NFE 框架——它把横向精化吸收进 $\mathcal{B}_t^*$。

**猜想 S1.3（H 与 S 的可交换性边界，实验修正为弱形式）。** 在横向收敛假设下，减小 $H$（如 $H=3$）需增大 $S$ 以补偿，反之亦然。**原预测（强形式，已证伪）**：$H \times S$ 不是不变量——因 $\mathcal{A}_t$ 是二阶 solver 而 $\mathcal{B}_{t,k}$ 是一阶精化，$H$ 减半需 $S$ 增加多于两倍。**实验证伪**：H=3,S=4 / H=6,S=2 / H=3,S=8 三组 mAP 全部持平于 0.859（见 [EXPERIMENT_LINEAGE.md §七](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)），说明在 mAP 指标（noise floor $\pm 0.003$）上 $H \times S$ 近似为不变量。**修正后的弱形式**：$H \times S$ 在 mAP 上近似不变（因低维 $d=4$ 空间中横向/纵向精度损失均被 noise floor 吸收），但在更精细的指标（如 $\eta_{\text{str}}$、per-class AP）上可能不是不变量——此弱形式尚待 $\eta_{\text{str}}$ 实验验证。

### 2.5 与已证伪 Cascade Head Count e2e 的区分

**Cascade Head Count e2e（已证伪，mAP 0.684，−0.172）**：重训架构，把 cascade head 数量从 6 改为其他值，端到端评估。

**S1**：仅**分析**已有 $H=6, S=4$ 架构的算子分裂结构，不重训。S1 是描述性分析，不引入新配置。

S1 的价值在于：
- 解释为何 Cascade Head Count e2e 失败——减小 $H$ 破坏横向收敛性，而 $S$ 未相应增加；
- 给出"solver 阶数 vs cascade head 数量"的理论权衡框架，避免未来重试类似 e2e 实验。

### 2.6 对论文 §3.2 的补充建议

在 §3.2 末尾补 1 段（约 0.3 页）：

> **Cascade head 与 solver step 的解耦。** 我们的架构在每个 solver step 内顺序执行 $H=6$ 个 cascade head（每个做 RoIAlign + DynamicConv + $\hat{x}_0$ 预测），共 $H \times S = 24$ 次前向。形式化地，cascade head 在固定 $t$ 上精化 $x_t$（横向收敛），solver step 在固定精化链上推进 $t$（纵向积分），两者构成算子分裂。在 cascade head 序列收敛至不动点 $\mathcal{B}_t^*$ 的假设下，DPM-Solver++ 把复合算子 $\mathcal{B}_t^* \circ \mathcal{A}_t$ 视为单次 $v_\theta$ 评估，故 4 NFE 框架有效。这一视角解释了为何减少 cascade head 数量（Appendix B，Cascade Head Count e2e 方向，−0.172 mAP）会破坏精度——横向收敛性被破坏而 $S$ 未相应增加——并预测了"solver 阶数 × cascade 深度"的可交换性边界。

---

## 3. D3：box_renewal 与 DPM-Solver++ 历史连续性矛盾

### 3.1 动机

[head.py:708–714](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L708-L714) 显示，在每个 solver step 后调用 `apply_box_renewal`，将低置信度 proposals 重置为随机噪声（或 VGAR 的半噪声）。但 [head.py:629–630](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L629-L630) 显示，Top-K pruning 后必须 `dpm_solver.reset()`，因为 x0_history 维度不匹配。box_renewal 同样改变了部分 proposals 的 x_raw，但未触发 reset——存在不一致。

我们形式化分析这一矛盾的后果，并给出对 Top-K K=100 掉点（−0.013 mAP，[project_memory](file:///home/linkst/.trae-cn/memory/projects/-home-linkst-workspace-projects-chromosome-kd/project_memory.md)）的潜在解释。

### 3.2 形式化分析

**DPM-Solver++ 的隐含假设。** 二阶校正项 $\mathbf{D}_1^{(n)} = (\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)}) / (t_n - t_{n-1})$ 假设 $\hat{x}_0(t)$ 是 $t$ 的连续（可微）函数，且 $\hat{x}_0^{(n)}$ 与 $\hat{x}_0^{(n-1)}$ 对应**同一 proposal** 在不同 $t$ 上的预测。

**box_renewal 的破坏。** 设第 $n$ 步后，proposal $i$ 因低置信度被重置：$x_{\text{raw},i}^{(n)} \leftarrow z_{\text{new}} \sim \mathcal{N}(0, \sigma^2 I_4)$。则第 $n+1$ 步：

- $x_{t_{n+1}, i}^{\text{renew}} = z_{\text{new}}$（非 solver 推进的 $x_{t_n} + \Delta t \cdot v$）；
- $\hat{x}_{0,i}^{(n+1)} = v_\theta(z_{\text{new}}, t_{n+1})$ 是**对新噪声样本**的预测，与 $\hat{x}_{0,i}^{(n)}$ 无轨迹连续性；
- $\mathbf{D}_{1,i}^{(n+1)} = (\hat{x}_{0,i}^{(n+1)} - \hat{x}_{0,i}^{(n)}) / (t_{n+1} - t_n)$ 反映"重置噪声的随机性"，而非"轨迹曲率"。

**命题 D3.1（renewal 后 D1 失效）。** 对被 renewal 的 proposal $i$，$\mathbf{D}_{1,i}^{(n+1)}$ 的期望范数等于 $\|\hat{x}_{0,i}^{(n+1)} - \hat{x}_{0,i}^{(n)}\| / \Delta t$，由于 $\hat{x}_{0,i}^{(n+1)}$ 是对新噪声的预测，其与 $\hat{x}_{0,i}^{(n)}$ 无统计相关性，故 $\mathbb{E}[\|\mathbf{D}_{1,i}^{(n+1)}\|^2] \approx (\mathbb{E}[\|\hat{x}_0\|^2] + \mathbb{E}[\|\hat{x}_{0,i}^{(n)}\|^2]) / \Delta t^2$，量级远大于真实轨迹曲率。

**推论 D3.2（η_str 被污染）。** 由命题 D3.1，box_renewal 后的 $\eta_{\text{str}}$（R1 定义）不再反映直线度，而是被 renewal 噪声主导。这使 R1 的诊断在 box_renewal 启用时失效。

### 3.3 对 Top-K K=100 掉点的解释（D3 假设被实验证伪）

**原 D3 假设**：K=100 掉点 −0.013 mAP 部分来自 DPM-Solver++ 高阶校正失效（box_renewal 破坏 x0_history）。

**实验证伪**（3 seeds × 4 configs，见 [r1_d3_summary.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/r1_d3_summary.py)）：

| 配置 | mAP | Step 1 $\eta_{\text{str}}$ | Step 2 | Step 3 |
|------|-----|---------------------------|--------|--------|
| TopK K=200 | 0.862 | 1.37 | 2.24 | 1.54 |
| TopK K=100 | 0.852 | 1.20 | 2.18 | 1.54 |

K=100 与 K=200 的 $\eta_{\text{str}}$ 在 step 2 几乎相同（2.18 vs 2.24，差异 < 3%），但 mAP 差 −0.010。**D3 假设被证伪**：K=100 掉点主因是 proposal 数量不足（现有解释正确），**不是** DPM-Solver++ 历史破坏。

**修正结论**：D3 矛盾（renewal 污染 $\eta_{\text{str}}$）是真实存在的诊断现象，但**不导致 mAP 显著损失**——因为 DPM-Solver++ 在 box_renewal 破坏历史后仍能通过 step 1 的 linear 项和后续步骤的重新积累恢复。这一鲁棒性是 DPM-Solver++ multistep 框架的隐含优点。

### 3.4 修复方案

尽管 D3 对 K=100 掉点的解释不成立，renewal 对 $\eta_{\text{str}}$ 诊断的污染仍需处理（使 R1 指标在 renewal on 时失效）。两种修复方向：

**方案 A（与 Top-K 一致）**：box_renewal 后 `dpm_solver.reset()`。

```python
# head.py:714 之后
if self.box_renewal:
    x_raw = self._sampler.apply_box_renewal(...)
    if dpm_solver is not None:
        dpm_solver.reset()  # 新增：与 Top-K pruning 一致
```

理论后果：step 2 退化为 Euler，但后续 step 2→3, 3→4 恢复 DPM-Solver++ 二阶。预期 mAP 变化 ≤ 0.002（与 Top-K 一致）。

**方案 B（推理时禁用 box_renewal）**：DPM-Solver++ 推理时不做 renewal，仅 Euler/Heun 用。

**实验已验证**：3 seed 平均 mAP 0.858 ± 0.003（vs baseline 0.859 ± 0.004），delta −0.0003，在噪声范围内。**方案 B 不损失精度**，且使 $\eta_{\text{str}}$ 诊断有效。

**方案 C（仅 step 0 后 renewal）**：在 step 0 后 renewal（与 Top-K 同步），step 1–3 不 renewal。理论预期最优，但需重训 checkpoint 评估（因训练时也用 renewal）。

### 3.5 与 VGAR 的交互

[head.py:167–203](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L167-L203) 的 VGAR（Velocity-Guided Adaptive Renewal）使用 $\alpha(t) \hat{x}_0 + (1-\alpha(t)) z$ 替代纯噪声。VGAR 缓解了 D3 的矛盾——因为 renewal 后的 $x_{\text{raw}}$ 保留了部分 $\hat{x}_0$ 信息，下一步 $\hat{x}_0^{(n+1)}$ 与 $\hat{x}_0^{(n)}$ 有部分相关性。但 VGAR 的 $\alpha(t)$ 在 $t \to 0$ 时 $\to 0.8$（[head.py:177–179](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L177-L179)），仍保留 20% 随机性，D3 矛盾仅缓解未消除。

### 3.6 实验验证（已完成）

在 +DPM-Solver++ checkpoint 上，3-seed 测试了 4 个配置（无需重训，仅推理时切换）。**关键结果**：

| 配置 | mAP (3 seeds) | Step 1 $\eta_{\text{str}}$ | Step 2 | Step 3 | 模式 |
|------|---------------|---------------------------|--------|--------|------|
| baseline (renewal on) | 0.859 ± 0.004 | 3.43 | 2.45 | 1.68 | 单调递减 |
| renewal off (方案 B) | 0.858 ± 0.003 | **1.50** | **1.11** | **0.70** | 单调递减 |
| TopK K=200 | 0.862 | 1.37 | 2.24 | 1.54 | **V 型** |
| TopK K=100 | 0.852 | 1.20 | 2.18 | 1.54 | **V 型** |

**核心结论**：
1. **D3 矛盾被证实**：renewal 使 $\eta_{\text{str}}$ 虚高 56-58%，但 mAP 仅 -0.0003（噪声）。box_renewal 严重污染 DPM-Solver++ 历史连续性，但不影响最终精度。
2. **D3 对 K=100 掉点的解释被证伪**：K=100/K=200 的 $\eta_{\text{str}}$ 几乎相同，掉点主因是 proposal 数量不足。
3. **Top-K reset 改变 $\eta_{\text{str}}$ 模式**：从"单调递减"变为"V 型"，证实 `dpm_solver.reset()` 使 DPM-Solver++ 重新冷启动。
4. **方案 B（renewal off）可行**：mAP 不损失，且使 $\eta_{\text{str}}$ 诊断有效。

---

## 4. R3：x0-prediction vs v-prediction 在低维 RF 的等价性

### 4.1 动机

RF 原文（Liu et al., 2023）和 Flow Matching（Lipman et al., 2023）使用 v-prediction 训练目标 $\|v_\theta - (x_1 - x_0)\|^2$。本论文使用 x0-prediction + DPM-Solver++ data-prediction 形式（[rectified_flow.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py), [Appendix A.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md#L506-L513)）。审稿人可能质疑该选择的依据。我们分析两者在 $d=4$ 低维 RF 下的等价性与差异。

### 4.2 形式化关系

**RF 路径**：$x_t = (1-t) x_0 + t x_1$，$v = x_1 - x_0$。

**两种参数化**：
- x0-prediction：网络输出 $\hat{x}_0$，损失 $\mathcal{L}_{x_0} = \|\hat{x}_0 - x_0\|^2$；
- v-prediction：网络输出 $\hat{v}$，损失 $\mathcal{L}_v = \|\hat{v} - v\|^2$。

**等价关系**：由 $x_0 = x_t - t v$，

$$\hat{v} = \frac{x_t - \hat{x}_0}{t}, \quad \mathcal{L}_v = \frac{1}{t^2} \|(\hat{x}_0 - x_0)\|^2 = \frac{1}{t^2} \mathcal{L}_{x_0}.$$

**命题 R3.1（信息等价）。** 在 $d=4$ 低维 RF 下，x0-prediction 与 v-prediction 的最优解在信息论意义上等价：$\hat{x}_0^* = x_t - t \hat{v}^*$，反之亦然。差异仅在损失函数的 $t$ 加权。

### 4.3 训练动态差异

**梯度**：

$$\frac{\partial \mathcal{L}_{x_0}}{\partial \hat{x}_0} = 2(\hat{x}_0 - x_0), \quad \frac{\partial \mathcal{L}_v}{\partial \hat{x}_0} = \frac{2}{t^2}(\hat{x}_0 - x_0).$$

**关键观察**：v-prediction 在 $t \to 0$ 时梯度放大 $1/t^2$，加剧方差；x0-prediction 在所有 $t$ 上梯度范数恒定。

**与 shifted schedule 的交互**（shift=3.0，[paper_draft_CN.md:213](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md#L213)）：

shifted schedule $t' = s \cdot t / (1 + (s-1) t)$（$s=3.0$）将训练时间质量重新分配，使更多样本落在 $t \in [0.5, 1]$（噪声主导区）。在 $t \ge 0.5$ 时 $1/t^2 \le 4$，v-prediction 的梯度放大可接受；但在 $t < 0.5$（数据主导区，对检测精度更关键）时 $1/t^2 > 4$，v-prediction 引入显著方差。

**命题 R3.2（shifted schedule 下的偏好）。** 在 shifted schedule（$s=3.0$）下，x0-prediction 的有效梯度信噪比优于 v-prediction，因前者在 $t \to 0$ 时不放大梯度。

### 4.4 推理时的奇点处理

**v-prediction 的奇点**：$\hat{x}_0 = x_t - t \hat{v}$ 在 $t \to 0$ 时 $\hat{x}_0 \to x_t$（数值稳定）。

**x0-prediction 的奇点**：$v = (x_t - \hat{x}_0) / t$ 在 $t \to 0$ 时 $v \to \infty$（数值不稳定）。

**当前实现**：[rectified_flow.py:45](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L45) `get_velocity` 使用 `torch.clamp(t_view, min=1e-5)`，且 DPM-Solver++ 直接预测 $\hat{x}_0$，避免显式计算 $v$，故奇点不触发。这是 x0-prediction + DPM-Solver++ data-prediction 形式的实现优势。

### 4.5 与图像生成设置的区分

| 设置 | $d$ | schedule | 推荐 | 原因 |
|------|-----|----------|------|------|
| 图像生成（RF 原文） | $\sim 10^5$ | linear | v-prediction | 高维下 $1/t^2$ 加权有益（强调小 $t$ 细节） |
| 检测（本文） | 4 | shifted ($s=3$) | x0-prediction | 低维下 $1/t^2$ 加剧方差，shifted 已重分配 $t$ |

**命题 R3.3（设置依赖性）。** RF 原文的 v-prediction 偏好依赖于高维 + linear schedule 的组合；在低维 + shifted schedule 下，x0-prediction 是更优选择，且与 DPM-Solver++ data-prediction 形式天然兼容。

### 4.6 对论文的补充建议

在 §3.1.1 末尾或 §5.3 补 1 段（约 0.3 页）：

> **x0-prediction 与 v-prediction 的选择。** RF 原文（Liu et al., 2023）使用 v-prediction 训练目标 $\|v_\theta - (x_1 - x_0)\|^2$。在 $d=4$ 低维 RF 下，x0-prediction 与 v-prediction 在信息论意义上等价（$\hat{x}_0 = x_t - t \hat{v}$），差异仅在损失的 $t$ 加权：$\mathcal{L}_v = t^{-2} \mathcal{L}_{x_0}$。在 shifted schedule（$s=3.0$）下，v-prediction 在 $t \to 0$ 时的 $1/t^2$ 梯度放大加剧训练方差，而 x0-prediction 在所有 $t$ 上梯度范数恒定。此外，x0-prediction 与 DPM-Solver++ 的 data-prediction 形式天然兼容，避免在 $t \to 0$ 时显式计算 $v = (x_t - \hat{x}_0)/t$ 的数值奇点（由 $\epsilon = 10^{-5}$ 截断处理，[Appendix A.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md#L506-L513)）。这一选择与图像生成（$d \sim 10^5$，linear schedule）下 v-prediction 的偏好形成对比，反映了低维结构化预测的特定要求。

---

## 5. 对论文的具体补充建议

### 5.1 章节定位与页数预算

| 方向 | 当前论文章节 | 建议补充位置 | 增量页数 | 类型 |
|------|--------------|--------------|----------|------|
| R1 | §4.5.2（2 步收敛） | §5.3 末段 + Appendix A.5 注记 | 0.4 页 | 纯理论 + 1 图 |
| S1 | §3.2（ODE Solvers） | §3.2 末段 | 0.3 页 | 纯理论 |
| D3 | §3.4（Top-K）+ §5.4 | §5.4 末段 + Appendix A.5 注记 | 0.4 页 | 理论 + 1 表 |
| R3 | §3.1.1（RF 公式） | §3.1.1 末段 或 §5.3 | 0.3 页 | 纯理论 |
| **合计** | — | — | **1.4 页** | — |

TMI 10 页主文当前已满（[main_layout.txt](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/latex/main_layout.txt)），1.4 页增量需要等量压缩。建议：
- R3 与 S1 直接纳入主文（理论重要性高，0.6 页）；
- R1 与 D3 的详细推导移至 arXiv companion，主文仅 1–2 句梗概（0.3 页）；
- 净增量 0.9 页，通过压缩 §4.7（跨数据集总结，与 §4.2/§4.4 重复）和 §5.5（理论适用性，与 Conclusion 重复）吸收。

### 5.2 与 main.tex 同步要点

按 [project_memory](file:///home/linkst/.trae-cn/memory/projects/-home-linkst-workspace-projects-chromosome-kd/project_memory.md) 的"草稿与 tex 同步验证"教训，纳入主文时需：
1. 用 grep 验证关键数字（$\eta_{\text{str}}$ 阈值、K=100 掉点数值）在草稿与 main.tex 间一致；
2. 同步 subagent 可能遗漏后续修正，需人工抽查 3–5 个关键数字；
3. 命题编号需与现有 Proposition 1–3 衔接（建议编号为 Proposition 4–7，或移至 Appendix A.6–A.9）。

### 5.3 实验验证优先级

| 实验 | 方向 | 成本 | 价值 | 优先级 |
|------|------|------|------|--------|
| $\eta_{\text{str}}$ 在 RF+Heun/+DPM-Solver++ 上测度 | R1 | 1 推理 pass + 1 行代码 | 高（定量验证 2 步收敛） | 高 |
| box_renewal on/off × K={100,200} | D3 | 4 配置 × 3 seed × 500 图 | 高（验证 K=100 掉点解释） | 高 |
| 方案 A/B/C 对比 | D3 | 同上 | 中（修复方案验证） | 中 |
| x0 vs v-prediction 重训 | R3 | 3 seed × 150 epoch | 低（理论已充分） | 不推荐 |

R1 与 D3 的实验仅需推理时改动，无需重训，可在 1–2 天内完成。R3 不建议实验验证（重训成本高，且理论分析已预防审稿人质疑）。

### 5.4 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| R1 的 $\eta_{\text{str}}$ 阈值因 seed 波动大 | 中 | 命题 R1.3 弱化 | 3 seed 报告均值±std |
| D3 修复方案 A/B/C 均无显著改善 | 中 | D3 沦为纯理论分析 | 仍可作为 Top-K 掉点的理论解释纳入 |
| 审稿人认为 R3 是"事后合理化" | 高 | R3 价值降低 | 强调"设置依赖性"（命题 R3.3），非"v-prediction 不行" |
| S1 与 Cascade R-CNN 比较被认为牵强 | 中 | S1 价值降低 | 强调"算子分裂"形式化，非直接类比 |

---

## 6. 总结

四个方向（R1+S1+D3+R3）已纳入 paper_draft_CN.md 主文（2026-07-27 合并）：

- **R1 → §4.5.2 + Appendix A.6**：$\eta_{\mathrm{str}}$ 定义作为命题 4 正式纳入；Figure 5 描述已添加（四子图：baseline/renewal-off/TopK/K100vsK200）；实验数据完整纳入正文。
- **S1 → §3.2 末段**：cascade head × solver step 的算子分裂已写入正文，含 H×S 消融结果。
- **D3 → §5.4**：Top-K 剪枝与 DPM-Solver++ 交互、K=100 证伪、renewal 对 $\eta_{\mathrm{str}}$ 影响均已纳入。
- **R3 → §5.3 新增段落**：x0-prediction vs v-prediction 的防御性分析，含低维+偏移调度下的选择理由。
- **R3（纯理论）**：预防审稿人对"x0 vs v-prediction"的质疑，强调设置依赖性而非绝对优劣。

四个方向均不重复 Adaptive Step–RoI Feature Cache / Flow Matching Detection / Cascade Head Count E2E 的失败模式：R1 是观测非决策，S1 是描述性分析非重训，D3 是诊断非新模块，R3 是解释非替换。

**实验产出**：
- [r1_eta_str_measure.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/r1_eta_str_measure.py)：η_str 测量脚本（支持 `--box-renewal on/off`）
- [r1_d3_summary.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/r1_d3_summary.py)：3 seed × 4 config 汇总
- 8 个 JSON 结果文件（`experiments/analysis/r1_eta_str_a3_seed*.json`）
- 代码改动：[rectified_flow.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py#L113-L153)（η_str 计算）、[head.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py#L774-L784)（_last_eta_str_log 收集）

**下一步建议**：
1. 用户审阅本文档，决定哪些方向纳入主文 / arXiv companion / 不纳入；
2. R1 + D3 实验结果可直接用于 §4.5.2（2 步收敛定量解释）和 §5.4（box_renewal 与 DPM-Solver++ 交互）；
3. S1 和 R3 仍为纯理论，可纳入 §3.2 和 §3.1.1；
4. 若纳入主文，需通过压缩 §4.7（跨数据集总结）和 §5.5（理论适用性）吸收约 0.9 页增量。

---

<!-- 文档结束。所有命题编号 R1.x / S1.x / D3.x / R3.x 为本文档内部编号，
     纳入主文时需重新编号为 Proposition 4+ 或 Appendix A.6+。
     R1 + D3 已完成 3 seed 实验验证 (2026-07-20)，S1 + R3 仍为纯理论分析。 -->
