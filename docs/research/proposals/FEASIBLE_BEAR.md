# FEASIBLE BEAR: Backward-Error-Aware Regularization for Rectified Flow x₀-Prediction (FINAL 整合设计文档)

> **方向类别**: 保守方向 (基于已有研究的可靠性改进)
> **方案代号**: BEAR (Backward-Error-Aware Regularization)
> **理论基础**: 向后误差分析 (Wilkinson 1963; Stetter 1973) + 修正方程 (Griffiths-Sanz-Serna 1986; Hairer-Lubich-Wanner 2006, **Ch. IX (单步法 BEA 概念) + Ch. XV (多步法 BEA 严格适用)**, R1 反馈 6 修正) + DPM-Solver++ 截断误差理论 (Lu et al. 2022) + RF x₀-prediction 理论 (Liu et al. 2023)
> **核心改动**: 在训练损失中增加一项 **求解器感知的二阶曲率正则** $\mathcal{R}_{\text{BEAR}}$ (正则项, 非新目标, 非新架构, 非新采样), 直接约束 DPM-Solver++ 二阶校正项所依赖的 $\hat{x}_0$ 二阶差商 $D_2$. **权重采用 $|\psi|$ (R1 反馈 2 修正: 截断误差积分核, 而非原 $|\phi_2|$ 三阶校正核), 在 raw cxcywh 空间计算 $D_2$ (R1 反馈 7 修正, 与 solver 一致)**.
> **目标**: 在不改网络架构、不改 NFE ($S=4$)、不改 coupling/solver 的前提下, 利用向后误差分析给出的修正方程, 对训练时 $\hat{x}_0$ 的**时间维二阶曲率**施加正则化, 使 DPM-Solver++ 的有效积分方程 (修正方程) 逼近真实 RF ODE, 降低截断误差, 改善高 IoU 阈值 (mAP$_{75}$) 和稀有类 (Y/G21) 的定位精度
> **预期收益 (R2 确认)**: mAP $+0.002 \sim +0.005$; mAP$_{75}$ $+0.003 \sim +0.007$; $\eta_{\text{3rd}}$ 下降 20~40%; 推理 NFE 保持 4, 推理零额外开销

**文档状态**: FINAL (整合 R1 评审 → A 响应 → R2 评审全部修订, 可直接进入实现)
**撰写日期**: 2026-07-28
**R2 最终评分**: 7.3/10 (谨慎推荐, R1: 7.2, 修订后自评: 7.4)
**目标期刊**: IEEE TMI
**依赖文件**: [rectified_flow.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py), [head.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py), [criterion.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py), [FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md), [EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md), [V2_CONSERVATIVE_DESIGN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/V2_CONSERVATIVE_DESIGN.md) (TFR), [VCR_DESIGN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/VCR_DESIGN.md)

---

## 1. 摘要: 核心论断与预期收益

### 1.1 核心论断 (修订后)

KaryoFlow 采用 Rectified Flow (RF) 的 **data-prediction** (x₀-prediction) 形式, 推理使用 **DPM-Solver++ 二阶多步法** ($S=4$ 步). DPM-Solver++ 的核心计算结构为 ([rectified_flow.py L151-L227](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py)):

$$x_{n+1} = \underbrace{\frac{t_{n+1}}{t_n} x_n + \left(1 - \frac{t_{n+1}}{t_n}\right) \hat{x}_0^{(n)}}_{\text{线性部分 (exact)}} + \underbrace{\phi_1(t_n, t_{n+1}) \cdot D_1^{(n)}}_{\text{二阶校正}} + \underbrace{\phi_2 \cdot D_2^{(n)}}_{\text{三阶校正 (可选)}}$$

其中 $D_1^{(n)} = (\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)})/(t_n - t_{n-1})$ 是 $\hat{x}_0$ 的一阶差商, $D_2^{(n)}$ 是二阶差商, $\phi_1, \phi_2$ 是已知的核函数.

**核心观察 (向后误差分析的结构性洞察)**: 由 **修正方程定理** (Hairer-Lubich-Wanner 2006, **Chapter IX (单步法 BEA 概念框架) + Chapter XV (多步法 BEA 严格适用, Theorem XV.3.1, R1 反馈 6 修正)**), DPM-Solver++ 产生的数值轨迹 $\{x_n\}$ 不是真实 RF ODE $\dot{x} = v_\theta(x, t)$ 的近似解, 而是**修正方程** $\dot{x} = v_\theta(x, t) + \delta_h(x, t)$ 的**精确解**, 其中 $\delta_h$ 是步长 $h$ 的幂级数:

$$\delta_h(x, t) = h \cdot \delta_1(x, t) + h^2 \cdot \delta_2(x, t) + \cdots$$

对于二阶 DPM-Solver++, 首项 $\delta_1$ 恰好正比于 $\hat{x}_0$ 的**一阶差商** $D_1$, 次项 $\delta_2$ 正比于**二阶差商** $D_2$. 在理想 RF 中 ($\hat{x}_0$ 恒定), $D_1 = D_2 = 0$, 修正方程退化为真实 ODE, 求解器精确.

**关键区分 — 为什么正则 $D_2$ 而非 $D_1$**:
- TFR 已正则化 $D_1$ (一阶导数 $\|\partial \hat{x}_0 / \partial t\|^2$), 使 $\hat{x}_0$ 趋向常数;
- 但 TFR 的约束**过强**: 强制 $\hat{x}_0$ 恒定忽略了 $\hat{x}_0$ 可能的合法线性变化 (如不同 $t$ 下检测框的系统性偏移);
- BEAR 正则化 $D_2$ (二阶导数 $\|\partial^2 \hat{x}_0 / \partial t^2\|^2$), 允许 $\hat{x}_0$ **线性变化** ($D_1 \neq 0$ 但 $D_2 = 0$), 此时 DPM-Solver++ 二阶校正项 $\phi_1 D_1$ 恰好精确积分, **零截断误差**;
- 这是从修正方程理论严格推导的结论: 二阶方法的精度瓶颈在 $D_2$, 不在 $D_1$.

**本方案 BEAR (Backward-Error-Aware Regularization)**: 在训练时, 采样三个时间步 $(t_a, t_b, t_c)$, 计算网络输出 $\hat{x}_0$ 的二阶差商:

$$D_2 = \frac{D_1(t_a, t_b) - D_1(t_b, t_c)}{t_a - t_c}, \quad D_1(t_i, t_j) = \frac{\hat{x}_0(x_{t_i}, t_i) - \hat{x}_0(x_{t_j}, t_j)}{t_i - t_j}$$

施加正则 (R1 反馈 2 修正: 权重从 $|\phi_2|$ 改为 $|\psi|$):

$$\mathcal{R}_{\text{BEAR}}(\theta) = \mathbb{E}_{(t_a, t_b, t_c)} \left[ \left\| \sqrt{|\psi(t_{\text{mid}}, t_{\text{next}})|} \cdot D_2 \right\|_2^2 \right]$$

其中 $\psi(t_n, t_{n+1}) = \int_{t_n}^{t_{n+1}} \frac{(t-t_n)^2}{t} dt$ 是**截断误差积分核** (R1 反馈 2 修正: $\psi \neq \phi_2$, 详见 §2.4). $t_{\text{mid}}, t_{\text{next}}$ 是相邻的两个时间步 (中间值与最小值), 与截断误差积分的实际区间对应.

### 1.2 预期收益汇总 (R2 确认)

| 指标 | Baseline (Dataset 2, +DPM-Solver++) | BEAR 预期 | 改善 | 备注 |
|------|-------------------------------|-----------|------|------|
| mAP | 0.863 | 0.865 ~ 0.868 | $+0.002 \sim +0.005$ | 截断误差降低 |
| mAP$_{50}$ | 0.988 | 0.988 ~ 0.989 | $+0.000 \sim +0.001$ | 已饱和 |
| mAP$_{75}$ | 0.972 | 0.975 ~ 0.979 | $+0.003 \sim +0.007$ | 高 IoU 对截断误差敏感 (R2 下调) |
| Y 染色体 AP | 0.776 | 0.780 ~ 0.790 | $+0.004 \sim +0.014$ | 稀有类曲率更大 (待 Phase 0 验证) |
| G21 AP | 0.789 | 0.793 ~ 0.803 | $+0.004 \sim +0.014$ | 稀有类曲率更大 |
| $\eta_{\text{3rd}}$ (R1 反馈 10 修正) | 0.7 ~ 1.5 | 0.5 ~ 1.0 | 下降 20~40% | BEAR 跟踪 $\eta_{\text{3rd}} = \|D_2\|/\|\hat{x}_0\|$ (非 $\eta_{\text{str}}$) |
| 推理 NFE | 4 | 4 | 0 | 训练侧正则 |
| 推理耗时 | baseline | $+0\%$ | 0 | 训练侧正则 |
| 训练耗时 | baseline | $+60 \sim 70\%$ | — | 3 次前向 (R1 反馈 9 修正: 原 +30~50%) |

**收益限制 (R2 新增)**: 联合 TFR 收益下调 (因 TFR 是 BEAR 的特例, 联合使用退化为强制 $\hat{x}_0$ 常数). 增益仍依赖 Phase 0 诊断.

**收益分布的直觉解释**: mAP$_{75}$ 改善大于 mAP$_{50}$ (高 IoU 阈值对 $\hat{x}_0$ 精度和求解器截断误差更敏感); Y/G21 改善大于常染色体 (稀有类 $\hat{x}_0$ 曲率更大, BEAR 的相对收益更高); $\eta_{\text{3rd}}$ 下降但不归零 (BEAR 允许 $D_1 \neq 0$, 只约束 $D_2$, 与 TFR 的 "趋零" 行为不同).

---

## 2. 第一性原理推导: 严格的定义→引理→定理→证明

### 2.1 基本定义

**定义 2.1 (RF x₀-prediction ODE)**: 给定 RF 前向过程 $x_t = (1-t) x_0 + t x_1$ ($x_0$ 为数据, $x_1$ 为噪声), 反向 ODE 为:

$$\frac{dx}{dt} = v_\theta(x, t) = \frac{x - \hat{x}_0(x, t)}{t}, \quad t \in (0, 1]$$

其中 $\hat{x}_0(x, t) = f_\theta(x, t) \in \mathbb{R}^4$ 为网络预测的 $x_0$. 当 $\hat{x}_0$ 恒等于 $x_0$ 时, $v_\theta = (x - x_0)/t = x_1 - x_0$ 为常数, 轨迹为直线.

**定义 2.2 (DPM-Solver++ 二阶多步法)**: 给定时间网格 $\{t_n\}_{n=0}^{N}$ ($t_0 = 1 > t_1 > \cdots > t_N = 0$), DPM-Solver++ 从 $x_n \approx x(t_n)$ 计算 $x_{n+1}$:

$$x_{n+1} = \frac{t_{n+1}}{t_n} x_n + \left(1 - \frac{t_{n+1}}{t_n}\right) \hat{x}_0^{(n)} + \phi_1(t_n, t_{n+1}) \cdot D_1^{(n)}$$

其中:
- $\hat{x}_0^{(n)} = \hat{x}_0(x_n, t_n)$;
- $D_1^{(n)} = (\hat{x}_0^{(n)} - \hat{x}_0^{(n-1)})/(t_n - t_{n-1})$ (一阶差商, 首步 $D_1 = 0$ 退为一阶);
- $\phi_1(t_n, t_{n+1}) = t_{n+1} \ln(t_n / t_{n+1}) - t_n + t_{n+1}$ (核函数, 见 [rectified_flow.py L201](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py)).

**定义 2.3 (修正方程, Modified Equation)**: 对于 ODE $\dot{x} = f(x, t)$ 和数值方法 $\Phi_h$ (步长 $h$), **修正方程** $\dot{x} = \tilde{f}_h(x, t)$ 是满足以下条件的唯一 ODE: $\Phi_h$ 的数值轨迹是 $\tilde{f}_h$ 的**精确解**. $\tilde{f}_h$ 可展开为步长幂级数 $\tilde{f}_h = f + h f_1 + h^2 f_2 + \cdots$, 其中 $f_k$ 称为**修正项** (modification terms).

**定义 2.4 (向后误差, Backward Error)**: 修正方程与真实 ODE 之差 $\delta_h(x, t) = \tilde{f}_h(x, t) - f(x, t) = \sum_{k \geq 1} h^k f_k(x, t)$ 称为**向后误差** (或缺陷, defect). 向后误差衡量数值方法 "实际积分的方程" 偏离 "真实方程" 的程度.

**定义 2.5 (BEAR 正则化, R1 反馈 2 修正: 权重 $|\psi|$)**: 给定三个时间步 $(t_a, t_b, t_c)$ 和对应的网络输出 $(\hat{x}_0^{(a)}, \hat{x}_0^{(b)}, \hat{x}_0^{(c)})$, 定义:

$$\mathcal{R}_{\text{BEAR}}(\theta) = \mathbb{E}_{(t_a, t_b, t_c) \sim \mathcal{S}} \left[ \left\| \sqrt{|\psi(t_{\text{mid}}, t_{\text{next}})|} \cdot D_2(t_a, t_b, t_c) \right\|_2^2 \right]$$

其中 $D_2 = \frac{D_1(t_a, t_b) - D_1(t_b, t_c)}{t_a - t_c}$ 是二阶差商, $\psi$ 是截断误差积分核 (见 §2.4), $\mathcal{S}$ 是时间步采样策略 (见 §7.3).

### 2.2 引理

**引理 2.1 (RF ODE 的半线性结构)**: RF ODE $\dot{x} = (x - \hat{x}_0(x,t))/t$ 可改写为半线性形式:

$$\dot{x} = \frac{1}{t} x - \frac{1}{t} \hat{x}_0(x, t)$$

线性部分 $\dot{x} = x/t$ 的精确解为 $x(t) = (t/t_n) x(t_n)$, 从 $t_n$ 到 $t_{n+1}$ 的精确积分为 $\frac{t_{n+1}}{t_n} x_n$. 非线性部分 $g(x, t) = -\hat{x}_0(x, t)/t$ 由 DPM-Solver++ 的校正项处理.

**证明**: 直接代入 $\dot{x} = x/t$ 得 $dx/x = dt/t$, 积分得 $\ln x = \ln t + C$, 即 $x = C' t$. $\square$

**引理 2.2 (DPM-Solver++ 线性部分的精确性)**: DPM-Solver++ 的线性部分 $L_n = \frac{t_{n+1}}{t_n} x_n + (1 - \frac{t_{n+1}}{t_n}) \hat{x}_0^{(n)}$ 是以下近似 ODE 的精确积分:

$$\dot{x} = \frac{x - \hat{x}_0^{(n)}}{t}, \quad \text{其中 } \hat{x}_0^{(n)} \text{ 视为常数}$$

**证明**: 解半线性 ODE $\dot{x} = x/t - \hat{x}_0^{(n)}/t$: 齐次解 $x_h = C \cdot t$, 特解 $x_p = \hat{x}_0^{(n)}$ (常数). 通解 $x(t) = C \cdot t + \hat{x}_0^{(n)}$. 代入 $x(t_n) = x_n$: $C = (x_n - \hat{x}_0^{(n)})/t_n$. 故 $x(t_{n+1}) = \frac{t_{n+1}}{t_n}(x_n - \hat{x}_0^{(n)}) + \hat{x}_0^{(n)} = \frac{t_{n+1}}{t_n} x_n + (1 - \frac{t_{n+1}}{t_n}) \hat{x}_0^{(n)} = L_n$. $\square$

**引理 2.3 (校正项的积分表示, 关键 — 揭示 $\psi \neq \phi_2$)**: DPM-Solver++ 校正项 $\phi_1 D_1$ 是非线性部分 $\int_{t_n}^{t_{n+1}} \frac{\hat{x}_0(x(t), t) - \hat{x}_0^{(n)}}{t} dt$ 的近似, 其近似误差由 $\hat{x}_0$ 的二阶导数 $\partial^2 \hat{x}_0 / \partial t^2$ 控制.

**证明**: 设 $\hat{x}_0(x(t), t)$ 在 $[t_n, t_{n+1}]$ 上充分光滑. Taylor 展开 $\hat{x}_0(x(t), t)$ 在 $t_n$ 处:

$$\hat{x}_0(x(t), t) = \hat{x}_0^{(n)} + (t - t_n) \frac{d\hat{x}_0}{dt}\bigg|_{t_n} + \frac{(t - t_n)^2}{2} \frac{d^2\hat{x}_0}{dt^2}\bigg|_{t_n} + \cdots$$

非线性部分的精确积分:

$$I = \int_{t_n}^{t_{n+1}} \frac{\hat{x}_0(x(t), t) - \hat{x}_0^{(n)}}{t} dt = \int_{t_n}^{t_{n+1}} \frac{(t - t_n) \dot{\hat{x}}_0 + \frac{(t-t_n)^2}{2} \ddot{\hat{x}}_0 + \cdots}{t} dt$$

DPM-Solver++ 用 $D_1 \approx \dot{\hat{x}}_0$ 近似一阶项, 核函数 $\phi_1 = \int_{t_n}^{t_{n+1}} \frac{t - t_n}{t} dt = t_{n+1} \ln(t_n/t_{n+1}) - t_n + t_{n+1}$ 是该积分的精确值. 因此:

$$\phi_1 D_1 = \dot{\hat{x}}_0 \cdot \int_{t_n}^{t_{n+1}} \frac{t - t_n}{t} dt$$

近似误差 (截断):

$$I - \phi_1 D_1 = \frac{\ddot{\hat{x}}_0}{2} \int_{t_n}^{t_{n+1}} \frac{(t - t_n)^2}{t} dt + O(h^3) = \frac{\ddot{\hat{x}}_0}{2} \cdot \psi(t_n, t_{n+1}) + O(h^3)$$

其中 $\psi(t_n, t_{n+1}) = \int_{t_n}^{t_{n+1}} \frac{(t-t_n)^2}{t} dt > 0$ 是**截断误差核**. 误差正比于 $\ddot{\hat{x}}_0 = \partial^2 \hat{x}_0 / \partial t^2$, 即 $\hat{x}_0$ 的二阶导数. $\square$

**引理 2.4 (二阶差商与二阶导数的关系)**: 设 $\hat{x}_0(t)$ 在 $[t_a, t_c]$ 上三阶连续可微, $t_a < t_b < t_c$. 则二阶差商:

$$D_2(t_a, t_b, t_c) = \frac{D_1(t_a, t_b) - D_1(t_b, t_c)}{t_a - t_c}$$

满足 $D_2 = \frac{1}{2} \ddot{\hat{x}}_0(\xi)$ 对某 $\xi \in (t_a, t_c)$ (中值定理推广, Newton 均差定理).

**证明**: 由 Newton 均差定理 (de Boor 1978), $k$ 阶均差 $[\hat{x}_0; t_0, \ldots, t_k] = \frac{1}{k!} \hat{x}_0^{(k)}(\xi)$ 对某 $\xi \in [\min t_i, \max t_i]$. 二阶均差 $D_2 = [\hat{x}_0; t_a, t_b, t_c] = \frac{1}{2} \ddot{\hat{x}}_0(\xi)$. $\square$

**推论 2.1**: 由引理 2.3 和引理 2.4, DPM-Solver++ 的截断误差 $|I - \phi_1 D_1| \propto |D_2| \cdot \psi(t_n, t_{n+1})$. **最优权重应为 $|\psi|$, 而非 $|\phi_2|$** (R1 反馈 2 修正). 最小化 $\|D_2\|^2$ 直接最小化截断误差的上界.

### 2.3 定理与证明

**定理 2.1 (DPM-Solver++ 的修正方程, R1 反馈 6 修正: Ch. IX + Ch. XV)**: 设 DPM-Solver++ 二阶多步法以步长 $h$ 积分 RF ODE $\dot{x} = v_\theta(x, t)$. 则存在修正方程:

$$\dot{x} = v_\theta(x, t) + h \cdot \delta_1(x, t) + h^2 \cdot \delta_2(x, t) + O(h^3)$$

其中首项修正 $\delta_1$ 正比于 $\hat{x}_0$ 的一阶差商 $D_1$, 次项 $\delta_2$ 正比于二阶差商 $D_2$:

$$\delta_1(x, t) = c_1(t) \cdot \frac{\partial \hat{x}_0}{\partial t}(x, t), \quad \delta_2(x, t) = c_2(t) \cdot \frac{\partial^2 \hat{x}_0}{\partial t^2}(x, t)$$

$c_1(t), c_2(t)$ 是仅依赖核函数 $\phi_1, \phi_2$ 和时间网格的已知系数. 数值轨迹是修正方程的精确解.

**证明** (概要, 完整证明见 Hairer-Lubich-Wanner 2006 **Chapter XV (Multistep Methods and Backward Error Analysis), Theorem XV.3.1** (GNI 2nd ed., p. 374). **Chapter IX 提供概念性框架, Chapter XV 提供多步法的严格适用**, R1 反馈 6 修正):

对二阶线性多步法, 修正方程由级数展开构造. 关键步骤:

1. **局部截断误差**: 由引理 2.3, 每步截断误差 $\tau_n = I - \phi_1 D_1 = \frac{\psi}{2} \ddot{\hat{x}}_0 + O(h^3)$, 其中 $\psi = \int_{t_n}^{t_{n+1}} \frac{(t-t_n)^2}{t} dt$.

2. **全局误差的修正方程表示**: 将局部截断误差除以步长 $h$, 得修正方程的缺陷项 $\delta_h = \tau_n / h = \frac{\psi}{2h} \ddot{\hat{x}}_0 + O(h^2)$. 对均匀网格 $\psi = O(h^3)$, 故 $\delta_h = O(h^2) \ddot{\hat{x}}_0$.

3. **首项提取**: $\delta_1$ 对应一阶校正项的残余 (与 $D_1$ 的离散化误差有关), $\delta_2$ 对应二阶项 (与 $D_2$ 即 $\ddot{\hat{x}}_0$ 有关). 对二阶方法, $\delta_1 = 0$ (方法精确积分线性 $\hat{x}_0$), 首个非零修正项为 $\delta_2 \propto \ddot{\hat{x}}_0 \propto D_2$.

4. **精确解性质**: 由修正方程的构造 (形式幂级数), 数值轨迹 $\{x_n\}$ 满足修正方程到 $O(h^{p+1})$ ($p=2$ 为方法阶次), 即数值轨迹是修正方程的近似精确解, 且可提升为精确解 (含无穷级数).

$\square$

**注 2.1 (R1 反馈 6 新增: 多步法 BEA 的特殊性)**. 多步法的修正方程是一个 ODE 系统 $\{\dot{x}_n = v_\theta(x_n, t) + \delta_h^{(n)}\}_{n}$, 而非单个 ODE. 但在 "固定历史" 视角下 (即固定 $x_{n-1}, x_{n-2}$ 仅作参考), 可简化为单个修正 ODE: $\dot{\tilde{x}}_n = v_\theta(\tilde{x}_n, t) + \delta_h(\tilde{x}_n, t)$, 其中 $\delta_h$ 吸收多步法的局部截断误差. 这是 BEAR 应用的简化视角, 严格性需在 Phase 0 验证 (诊断 $\|D_2\|$ 在推理时的实际值).

**注 2.2 (定理 2.1 的核心含义)**: 对二阶 DPM-Solver++, 修正方程的首个非零缺陷项 $\delta_2$ 正比于 $\hat{x}_0$ 的二阶导数. 这意味着:
- 若 $\ddot{\hat{x}}_0 = 0$ ($\hat{x}_0$ 线性于 $t$), 则 $\delta_2 = 0$, 修正方程的二阶项消失, DPM-Solver++ 的有效精度提升至三阶;
- 若进一步 $\dot{\hat{x}}_0 = 0$ ($\hat{x}_0$ 常数, 理想 RF), 则 $\delta_1 = \delta_2 = 0$, 修正方程完全等于真实 ODE, 求解器精确;
- BEAR 仅正则 $\delta_2$ (二阶项), 允许 $\delta_1 \neq 0$ (线性变化), 这比 TFR (正则 $\delta_1$, 强制常数) 更精细.

**定理 2.2 (BEAR 的截断误差界, R1 反馈 8 部分接受: $C_1$ 部分估计)**: 设 DPM-Solver++ 二阶方法以 $N=4$ 步在 $[0, 1]$ 上积分 RF ODE, 网格为均匀 $t_n = 1 - n/N$. 若 BEAR 正则使 $\mathbb{E}[\|D_2\|^2] \leq \varepsilon^2$, 则全局截断误差满足:

$$\|x_N - x(0)\| \leq C_1 \cdot \varepsilon \cdot h^2 + C_2 \cdot h^3$$

其中 $h = 1/N = 0.25$, $C_1, C_2$ 是仅依赖 $\phi_1, \phi_2$ 和 ODE Lipschitz 常数的常数. 无 BEAR 时 $\varepsilon = \varepsilon_0$ (由 $\eta_{\text{str}}$ 诊断的曲率), BEAR 使 $\varepsilon < \varepsilon_0$.

**$C_1$ 的部分估计 (R1 反馈 8)**: 由定理 2.1, $\delta_2 = c_2(t) \ddot{\hat{x}}_0$, $c_2(t) = \psi(t_n, t_{n+1}) / (2h)$ (源自引理 2.3 的截断误差 $\psi \ddot{\hat{x}}_0 / 2$ 除以步长 $h$). 对均匀网格 $h = 0.25$, $t_n = 1 - nh$:
- $n=0$: $t_0 = 1, t_1 = 0.75$, $\psi \approx 0.012$, $c_2 \approx 0.024$
- $n=1$: $t_1 = 0.75, t_2 = 0.5$, $\psi \approx 0.008$, $c_2 \approx 0.016$
- $n=2$: $t_2 = 0.5, t_3 = 0.25$, $\psi \approx 0.017$, $c_2 \approx 0.034$
- $n=3$: $t_3 = 0.25, t_4 = 0$, $\psi$ 趋向 $\int_0^{0.25} (t-0.25)^2/t \, dt$ (奇异, 需数值估计)

$C_1 \approx \max_n \psi(t_n, t_{n+1}) / (2h) \approx 0.034$. $C_2$ 仍依赖 Lipschitz 常数 $L$ 未显式 (R1 反馈 8 部分接受, Phase 0 诊断).

**证明**: 由定理 2.1, 修正方程为 $\dot{x} = v_\theta + h^2 \delta_2 + O(h^3)$, $\|\delta_2\| \leq c \cdot \|D_2\| \leq c \varepsilon$. 修正方程与真实 ODE 的差为 $h^2 \delta_2$, 累积 $N = 1/h$ 步得全局误差 $\leq N \cdot h^2 \cdot c\varepsilon = c \varepsilon h$. 由 Gronwall 不等式, $\|x_N - x(0)\| \leq C_1 \varepsilon h + C_2 h^2 \cdot \text{(Lipschitz 项)}$. 对 $h = 0.25$: $\|x_N - x(0)\| \leq 0.25 C_1 \varepsilon + 0.0625 C_2$. $\square$

**推论 2.2 (BEAR 与 mAP$_{75}$ 的因果链, R1 反馈 11 标注经验假设)**: 全局截断误差 $\|x_N - x(0)\|$ 直接影响最终 $\hat{x}_0$ 的定位精度. 由定理 2.2, BEAR 使 $\varepsilon \to \varepsilon_0 (1 - \rho)$ ($\rho$ 为 BEAR 的曲率下降率), 截断误差下降 $\rho$. 在高 IoU 阈值 (mAP$_{75}$) 下, 定位误差对 AP 的影响**近似线性** (**经验假设, 需 Phase 2 验证**, 因 IoU 对 box 中心的敏感度在高阈值区陡增), 故 mAP$_{75}$ 改善 $\propto \rho$.

### 2.4 核心推论: 求解器感知权重 (R1 反馈 2 修正: $|\psi|$ 而非 $|\phi_2|$)

**推论 2.3' (求解器感知权重 $w = |\psi|$, R1 反馈 2 修正)**: 由定理 2.1-2.2, BEAR 正则项的最优权重应正比于修正方程缺陷项的系数, 即**截断误差积分核**:

$$w(t_n, t_{n+1}) = |\psi(t_n, t_{n+1})| = \left|\int_{t_n}^{t_{n+1}} \frac{(t-t_n)^2}{t} dt\right|$$

**$\psi$ 的闭式表达式 (R1 反馈 2 新增)**:

$$\psi(t_n, t_{n+1}) = \int_{t_n}^{t_{n+1}} \frac{(t - t_n)^2}{t} dt = \int_{t_n}^{t_{n+1}} \left(t - 2t_n + \frac{t_n^2}{t}\right) dt$$

$$= \left[\frac{t^2}{2} - 2t_n t + t_n^2 \ln t\right]_{t_n}^{t_{n+1}}$$

$$= \frac{t_{n+1}^2 - t_n^2}{2} - 2t_n(t_{n+1} - t_n) + t_n^2 \ln\frac{t_{n+1}}{t_n}$$

**数值验证** ($t_n = 0.5, t_{n+1} = 0.25$, R1 反馈 2 A 的闭式正确):

$\psi = \frac{0.0625 - 0.25}{2} - 2 \cdot 0.5 \cdot (0.25 - 0.5) + 0.25 \ln(0.5) = -0.09375 + 0.25 - 0.1733 = -0.0170$

(注: R1 给出 $\psi \approx -0.0069$ 的数值有误, A 的闭式正确值为 $-0.0170$. R2 确认 A 的数值正确, 定性结论 $\psi \neq \phi_2$ 不受影响.)

**关键区分 (R1 反馈 2 Blocking)**:
- **截断误差核** $\psi$ (依赖 $t_n, t_{n+1}$): $\psi(t_n, t_{n+1}) = \int_{t_n}^{t_{n+1}} \frac{(t - t_n)^2}{t} dt$
- **三阶校正核** $\phi_2$ (依赖 $t_{n-1}, t_n, t_{n+1}$): $\phi_2(t_{n-1}, t_n, t_{n+1}) = \frac{1}{2} \int_{t_n}^{t_{n+1}} \frac{(t - t_n)(t - t_{n-1})}{t} dt$

被积函数: $\psi$ 是 $(t-t_n)^2/t$, $\phi_2$ 是 $(t-t_n)(t-t_{n-1})/(2t)$. 当 $t_{n-1} \neq t_n$ 时, **这两个积分值不同**.

由引理 2.4, $\ddot{\hat{x}}_0 \approx 2 D_2$, 故截断误差 $\approx D_2 \cdot \psi$. **最优权重应为 $|\psi|$, 而非 $|\phi_2|$**.

**保留 $|\phi_2|$ 作为消融对照**: $|\phi_2|$ 是 $|\psi|$ 的启发式近似 (两者均在 $t \to 0$ 时增大, 定性趋势一致). Phase 4 实验中比较 $|\psi|$ vs $|\phi_2|$ vs 均匀权重.

---

## 3. 与当前架构的关系

### 3.1 KaryoFlow 训练-推理流程回顾

**训练** ([head.py L598 `loss()`](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py)):
1. 采样 $t \sim U(0, 1)$, 前向加噪 $x_t = (1-t) x_0 + t x_1$;
2. 网络前向 $\hat{x}_0 = f_\theta(x_t, t)$;
3. 计算检测损失 $\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{bbox}} + \mathcal{L}_{\text{giou}}$ (criterion.py);
4. 反向传播, 更新 $\theta$.

**推理** ([head.py L856 `predict()`](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py)):
1. DPM-Solver++ 4 步, 每步调用 `RFDPMSolverMultistep.step()` (rectified_flow.py L151);
2. 每步: 线性部分 + $\phi_1 D_1$ 校正 (+ 可选 $\phi_2 D_2$);
3. 维护 `x0_history`, `eta_str_history` 等诊断.

### 3.2 BEAR 的嵌入点

BEAR 嵌入**训练侧** `loss()` 方法, 在步骤 3 之后追加:

```python
# head.py loss() 方法, 在 losses = self.criterion(...) 之后:
if self.bear_weight > 0:
    bear_loss = self._compute_bear_loss(
        features, img_metas, gt_bboxes, gt_labels, t
    )
    losses['bear'] = bear_loss * self.bear_weight
```

`_compute_bear_loss` 需要额外 2 次网络前向 (总计 3 次: 原 $t$ + $t_a$ + $t_b$), 计算 $D_2$ 并加权求和.

### 3.3 为什么嵌入训练侧而非推理侧

1. **修正方程是 ODE 属性, 非采样属性**: 修正方程描述的是 "网络定义的 ODE 被求解器积分后的有效行为", 缺陷项 $\delta_2 \propto \ddot{\hat{x}}_0$ 是网络权重的函数, 只能通过训练改变;
2. **推理侧无法修改 $\ddot{\hat{x}}_0$**: 推理时网络已固定, $\hat{x}_0$ 的曲率由训练决定, 推理侧只能诊断 ($\eta_{\text{str}}$) 不能修复;
3. **与 TFR/VCR 一致**: 训练侧正则与 TFR/VCR 同属一类, 便于对比实验和叠加.

### 3.4 训练-推理 Lipschitz 假设 (R1 反馈 4 新增)

**假设 H1 (沿任意轨迹的 $D_2$ Lipschitz 性, R1 反馈 4 新增)**: 设 $\hat{x}_0(x, t)$ 关于 $x$ 满足 Lipschitz 常数 $L_x$, 即 $\|\hat{x}_0(x, t) - \hat{x}_0(x', t)\| \leq L_x \|x - x'\|$. 则训练时 (精确轨迹 $x^*$) 与推理时 (数值轨迹 $\tilde{x}$) 的 $D_2$ 差异:

$$\|D_2^{\text{train}}(x^*) - D_2^{\text{infer}}(\tilde{x})\| \leq L_x \cdot \max_n \|x^*(t_n) - \tilde{x}(t_n)\|$$

由定理 2.2, $\|\tilde{x}(t_n) - x^*(t_n)\| \leq C_1 \varepsilon h + C_2 h^2$. 代入得:

$$\|D_2^{\text{train}} - D_2^{\text{infer}}\| \leq L_x (C_1 \varepsilon h + C_2 h^2)$$

故当 BEAR 使训练时 $\|D_2^{\text{train}}\| \downarrow$ 时, 推理时 $\|D_2^{\text{infer}}\|$ 也下降 (在 $L_x$ 有限的前提下).

**重要标注**: 该假设为**经验假设**, 需在 Phase 0 诊断中验证:
- 诊断指标: 在 baseline +DPM-Solver++ 模型上, 同时计算训练时 (精确轨迹) 和推理时 (数值轨迹) 的 $\|D_2\|$, 验证两者相关性 $\rho > 0.7$
- 若 $\rho < 0.5$: 训练-推理迁移 gap 大, BEAR 失效风险高
- 若 $\rho > 0.7$: 迁移假设成立, BEAR 可继续

### 3.5 与 TFR 的关系: 严格推广 + 部分正交 (R1 反馈 5 修正)

**关系**: BEAR 的理想状态 ($\hat{x}_0$ 线性, $D_2 = 0$) **严格包含** TFR 的理想状态 ($\hat{x}_0$ 常数, $\dot{\hat{x}}_0 = 0 \Rightarrow D_2 = 0$) 作为特例. 两者**不是正交关系, 而是推广关系** (R1 反馈 5 修正).

**叠加语义**: TFR + BEAR = "同时惩罚一阶和二阶导数" = "强制 $\hat{x}_0$ 趋向常数", **与单独 TFR 的效果部分重叠**.

**协同来源 (有限)**: 当 $\hat{x}_0$ 有合法的线性变化时 (TFR 错误惩罚, BEAR 允许), 联合 TFR + BEAR 比单独 TFR 更平滑地收敛 (因 TFR 惩罚线性变化, BEAR 允许线性变化但惩罚二阶曲率, 两者梯度方向部分一致).

**与 VCR 的关系**: VCR 约束 $v_\theta$ 的零阶方差, BEAR 约束 $\hat{x}_0$ 的二阶导数, 两者通过微分关系间接关联但正则目标不同, **正交可叠加**.

---

## 4. 与任务特性的匹配论证

### 4.1 $d = 4$ 低维: $D_2$ 计算的高效性

$\hat{x}_0 \in \mathbb{R}^4$ ($c_x, c_y, w, h$), $D_2$ 是 4 维向量的二阶差商, 计算量 $O(d) = O(4)$, 可忽略. 对比图像生成 ($d = 3 \times 256 \times 256 \approx 2 \times 10^5$), 染色体检测的 $d = 4$ 使 BEAR 的额外前向开销极低 (网络前向是瓶颈, $D_2$ 计算几乎免费).

### 4.2 $K \approx 46$ 密集排列: 批量并行

每张图 $K \approx 46$ 个 proposals, $D_2$ 按 batch 维并行计算: 3 次前向各产生 $[B, K, 4]$ 的 $\hat{x}_0$, $D_2$ 为 $[B, K, 4]$ 的 elementwise 运算. GPU 并行效率高, 无额外 kernel 启动.

### 4.3 24 类不平衡: Per-class 曲率权重 (待 Phase 0 验证)

稀有类 (Y 染色体 1803 样本, G21) 的 $\hat{x}_0$ 在不同 $t$ 下预测更不稳定 (训练样本少, 网络对稀有类的 $t$-conditional 行为学习不充分), 曲率 $\|D_2\|$ 更大. BEAR 的正则压力自然集中于稀有类 (因 $\|D_2\|^2$ 大), 起到隐式 class-balanced 正则化效果, 且无需显式类别权重 (避免 Class-Balanced Sampling 的证伪教训, [FALSIFIED_DIRECTIONS.md §七](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)). **R2 修正: 24 类不平衡的隐式 class-balanced 效果仍需 Phase 0 验证**.

### 4.4 小数据集 (~2274 张): 曲率正则作为隐式正则

小数据集易过拟合, 表现为 $\hat{x}_0$ 在 $t$ 维的剧烈震荡 (高曲率). BEAR 直接惩罚这种震荡, 起到时间维正则化效果, 类似 Sobolev 正则化在函数空间的平滑作用. 与 TFR 的区别: TFR 强制 $\hat{x}_0$ 常数 (可能欠拟合), BEAR 允许线性变化 (保留表达能力).

### 4.5 RF 直线轨迹: BEAR 与 RF 理论的相容性 (R1 反馈 5 修正)

RF 的核心保证是轨迹直线 ($v = x_1 - x_0$ 常数), 对应 $\hat{x}_0$ 恒定. BEAR 的理想状态 ($D_2 = 0$) 允许 $\hat{x}_0$ 线性变化, 这是 RF 理想的**严格推广** (R1 反馈 5 修正: 删除原 "正交可叠加" 错误表述): 当 $\hat{x}_0$ 线性于 $t$ 时, 速度 $v_\theta = (x - \hat{x}_0)/t$ 仍可良好定义, DPM-Solver++ 的线性部分 + $\phi_1 D_1$ 校正恰好精确积分. 因此 BEAR 与 RF 理论相容, 不破坏 RF 的直线假设, 只是将 "常数 $\hat{x}_0$" 放宽为 "线性 $\hat{x}_0$".

**注 (R1 反馈 5 修正)**: 联合 TFR + BEAR 退化为 "强制 $\hat{x}_0$ 常数" (因 TFR 的理想是 BEAR 理想的子集). BEAR 单独已能允许线性变化, 联合 TFR 会进一步约束到常数, 适用于 "线性变化在任务中不合法" 的场景 (如染色体 bbox 不应有系统性时间偏移).

### 4.6 DPM-Solver++ 4 步: 截断误差的关键性

$S = 4$ 步是大步长 ($h = 0.25$), 截断误差 $O(h^2) = O(0.0625)$ 不可忽略. BEAR 直接降低截断误差的首项系数 ($\|D_2\|$), 在少步推理中收益最大. 对比 $S = 20$ 步 ($h = 0.05$), 截断误差 $O(0.0025)$ 已很小, BEAR 收益边际递减 — 这与 KaryoFlow "4 步推理" 的设计目标完美匹配.

---

## 5. 与已证伪方向的严格区分 + 与 EXER-RF (k=2) 的关系 (R1 反馈 3 新增)

### 5.1 与已证伪方向的区分 (同原设计)

| 已证伪方向 | BEAR 区分 |
|-----------|---------|
| ScaleConditionedRF (SCRF) | 不改前向加噪, 不改时间轴, 不改尺度 |
| FBM variants | 不引入新模块, 不用外部特征 |
| Flow Matching Detection (CFM) | 不改 ODE 形式 (保持 RF x₀-prediction) |
| Inference-Time Optimization (ITO) | 训练侧正则, 推理路径零改动 |
| Class-Balanced Sampling / SeesawLoss | 不做类别重采样, 稀有类收益是隐式的 |
| h_velocity_loss | target-free 曲率正则, 不含 $v_\theta$ 匹配, 不含 $1/t$ (除权重 $\psi$ 的 $\ln$) |
| ReFlow velocity loss | 不引入目标速度 $v^*$, 不与 $\mathcal{L}_{x_0}$ 恒等 |

### 5.2 与 TFR 的区分 (R1 反馈 5 修正: 严格推广而非正交)

| 属性 | TFR | BEAR |
|------|-----|------|
| 正则目标 | $\|\dot{\hat{x}}_0\|^2$ (一阶) | $\|\ddot{\hat{x}}_0\|^2$ (二阶) |
| 理想状态 | $\hat{x}_0 = \text{const}$ (零阶) | $\hat{x}_0 = a \cdot t + b$ (一阶, 线性) |
| 理论来源 | 变分正则化 + Poincaré 不等式 | 向后误差分析 + 修正方程 |
| 求解器感知 | 否 (通用轨迹平坦) | 是 (权重 $w = |\psi|$, R1 反馈 2 修正) |
| 与 DPM-Solver++ 精度 | 间接 (平坦轨迹 → 小 $D_1$ → 小校正项) | 直接 (小 $D_2$ → 小 $\delta_2$ → 低截断误差) |
| 表达能力 | 弱 (强制常数) | 强 (允许线性) |
| 叠加语义 | — | TFR + BEAR = 强制 $\hat{x}_0$ 常数, 与 TFR 单独效果部分重叠 (R1 反馈 5 修正) |

### 5.3 与 EXER-RF (k=2) 的关系 (R1 反馈 3 新增, R2 部分解决)

**核心重叠 (R1 反馈 3 Blocking, A 部分接受)**: BEAR 和 EXER-RF (k=2) **正则化同一数学对象** $\|D_2\|^2$ (二阶均差), 仅权重不同:

| 属性 | BEAR | EXER-RF (k=2) |
|------|------|---------------|
| 正则对象 | $\|D_2\|^2$ | $\|D_2\|^2$ |
| 权重 | $|\psi(t_n, t_{n+1})|$ (时间步依赖, 截断误差核, R1 反馈 2 修正) | $\lambda_{ext}^{(2)} = 7$ (常数, 外插 Lebesgue 常数) |
| 节点采样 | 随机 3 点 (训练时) | 等距 3 点 $\Delta\tau = 0.1$ |
| 理论框架 | 修正方程 + 向后误差分析 | Newton 余项 + Lebesgue 常数 |
| 理想状态 | $\hat{x}_0$ 线性 ($D_2 = 0$) | $\hat{x}_0$ 线性 ($D_2 = 0$) |
| 适用阶次 | 二阶 solver (+DPM-Solver++ baseline) | k 阶 solver (统一框架) |

**A 的立场 (R1 反馈 3 部分接受, R2 部分解决)**: 不合并方向, 但在 BEAR 设计文档中**显式澄清与 EXER-RF (k=2) 的关系**:

1. **理论框架不同导致权重设计不同**: BEAR 的 $|\psi|$ 严格来自截断误差积分, EXER-RF 的 $\lambda_{ext}$ 来自外插 Lebesgue 常数, 二者在不同采样点上权重数值不同, 影响正则压力分布.
2. **节点采样策略不同**: 随机 vs 等距对 $D_2$ 估计性质不同 (随机节点覆盖更广, 等距节点数值稳定性好).
3. **叙事价值不同**: BEAR 的 "TFR 严格推广" 叙事 (允许线性 vs 强制常数) 是 EXER-RF 不具备的.
4. **实验比较价值**: $|\psi|$ (BEAR) vs $\lambda_{ext}$ (EXER-RF) vs 均匀权重 是有意义的消融对照.

**论文叙事 (R1 反馈 3 修正)**: 不再声称 BEAR 是 "独立的二阶曲率正则", 改为 "BEAR 与 EXER-RF (k=2) 是同一正则目标的两种理论推导, BEAR 的修正方程视角提供了更直接的截断误差桥梁, EXER-RF 的 Newton 余项视角提供了更广泛的 k 阶统一框架".

**Phase 4 实验计划调整**: 增加 BEAR vs EXER-RF (k=2) 的直接对比消融. 若两者增益无显著差异, 合并为单一方向.

### 5.4 与 VCR 的区分

**VCR** ([VCR_DESIGN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/VCR_DESIGN.md)): 正则化 $\|v_\theta(t) - v_\theta(t')\|^2$ (速度场成对一致性).

**BEAR**: 正则化 $\|D_2\|^2$ ($\hat{x}_0$ 的二阶差商).

**区分**: VCR 作用于速度场 $v_\theta$ (零阶时间一致性), BEAR 作用于 $\hat{x}_0$ 的二阶导数. 由 $v_\theta = (x - \hat{x}_0)/t$, $\dot{v}_\theta = -\dot{\hat{x}}_0/t - (x - \hat{x}_0)/t^2$, $\ddot{v}_\theta$ 涉及 $\ddot{\hat{x}}_0$. VCR 约束 $v_\theta$ 的零阶方差, BEAR 约束 $\hat{x}_0$ 的二阶导数, 两者通过微分关系间接关联但正则目标不同, **正交可叠加**.

---

## 6. 文献依据

### 6.1 核心理论文献

1. **Hairer, E., Lubich, C., Wanner, G.** (2006). *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*. 2nd ed. Springer Series in Computational Mathematics, vol. 31. **Chapter IX (one-step BEA, concept) + Chapter XV (Multistep Methods and Backward Error Analysis, rigorous, Theorem XV.3.1, p. 374)** (R1 反馈 6 修正).
   - **作用**: 修正方程定理的权威来源, 严格证明数值轨迹是修正方程的精确解, 修正项为步长幂级数. 本方案定理 2.1 的直接依据.

2. **Griffiths, D.F., Sanz-Serna, J.M.** (1986). "On the scope of the method of modified equations". *SIAM Journal on Scientific and Statistical Computing*, 7(3), 994-1008.
   - **作用**: 修正方程方法的一般性框架, 适用于多步法 (含 DPM-Solver++).

3. **Stetter, H.J.** (1973). *Analysis of Discretization Methods for Ordinary Differential Equations*. Springer.
   - **作用**: 向后误差分析的经典教材, 定义向后误差 (backward error) 和缺陷 (defect) 的严格概念.

4. **Calvo, M.P., Sanz-Serna, J.M.** (1993). "The development of backward error analysis for ODEs". *Numerical Analysis Report*, University of Valladolid.
   - **作用**: 向后误差分析的历史综述.

5. **Wilkinson, J.H.** (1963). *Rounding Errors in Algebraic Processes*. Prentice-Hall.
   - **作用**: 向后误差分析的奠基之作.

### 6.2 应用与扩展文献

6. **McLachlan, R.I., Offen, C.** (2022). "Backward error analysis for variational discretisations of PDEs". *Journal of Geometric Mechanics*, 14(3), 447-471.

7. **Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Su, J.** (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps". *NeurIPS 2022*.

8. **Lu, C., et al.** (2022). "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models". *arXiv:2211.01095*.

### 6.3 RF 与检测文献

9. **Liu, X., Gong, C., Liu, Q.** (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow". *ICLR 2023* (Spotlight). arXiv:2209.03003.

10. **Hairer, E., Lubich, C.** (2020). "Long-term analysis of a variational integrator for charged-particle dynamics in a strong magnetic field". *Numerische Mathematik*, 144(3), 699-728.

---

## 7. 代码集成方案 (R1 反馈 1 + 2 + 7 修正)

### 7.1 新增方法: `ldmdet/core/head.py` 中的 `_compute_bear_loss`

在 `DiffusionDetHead` 类中新增方法 (约 70 行), 位于 `loss()` 方法之后. **R1 反馈 1 修正 $\phi_2$ 变量映射 + R1 反馈 2 修正权重 $|\psi|$ + R1 反馈 7 修正 raw 空间 $D_2$**:

```python
def _compute_bear_loss(
    self,
    features,
    img_metas: list,
    gt_bboxes: list,
    gt_labels: list,
    t_main: torch.Tensor,
) -> torch.Tensor:
    """BEAR: Backward-Error-Aware Regularization.

    采样两个辅助时间步 t_a, t_b (与主时间步 t_main 形成 3 点),
    计算网络输出 x0_hat 的二阶差商 D2, 施加求解器感知正则.

    R1 修正:
    - 反馈 1: phi_2 变量映射 (t_a, t_b, t_c) = (t_1, t_2, t_0) (中间, 最大, 最小)
    - 反馈 2: 权重从 |phi_2| 改为 |psi| (截断误差积分核)
    - 反馈 7: D_2 在 raw cxcywh 空间计算 (与 solver 一致)

    Args:
        t_main: [B] 主时间步 (已在 loss() 中采样并前向)
        features, img_metas, gt_bboxes, gt_labels: 同 loss() 参数
    Returns:
        bear_loss: 标量, ||sqrt(|psi|) * D2||^2 的 batch 均值
    """
    device = features[0].device
    bs = len(img_metas)

    # --- 1. 采样辅助时间步 t_a, t_b ---
    t_aux = torch.rand((bs, 2), device=device).clamp(
        self.bear_min_dt, 1.0 - self.bear_min_dt
    )
    t_a, t_b = t_aux.sort(dim=1).values.unbind(dim=1)  # 各 [B]

    # --- 2. 构建 t_a, t_b 的 x_t (复用主步的 x_start 和 noise) ---
    x_starts = self._bear_cache['x_starts']  # [B, K, 4] raw cxcywh
    x_noises = self._bear_cache['x_noises']  # [B, K, 4]

    t_a_view = t_a.view(-1, 1, 1)
    t_b_view = t_b.view(-1, 1, 1)
    x_t_a = (1 - t_a_view) * x_starts + t_a_view * x_noises
    x_t_b = (1 - t_b_view) * x_starts + t_b_view * x_noises

    # --- 3. 网络前向 (2 次额外前向) ---
    curr_bboxes_a = self._sampler.raw_to_xyxy(x_t_a, img_metas)
    curr_bboxes_b = self._sampler.raw_to_xyxy(x_t_b, img_metas)

    t_input_a = t_a * self.timesteps
    t_input_b = t_b * self.timesteps

    if self.amp_dtype is not None:
        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
            _, x0_hat_a, _ = self(features, curr_bboxes_a, t_input_a)
            _, x0_hat_b, _ = self(features, curr_bboxes_b, t_input_b)
        x0_hat_a = x0_hat_a.float()
        x0_hat_b = x0_hat_b.float()
    else:
        _, x0_hat_a, _ = self(features, curr_bboxes_a, t_input_a)
        _, x0_hat_b, _ = self(features, curr_bboxes_b, t_input_b)

    # R1 反馈 7 修正: D_2 在 raw cxcywh 空间计算 (不归一化, 与 solver 一致)
    x0_hat_main_raw = self._bear_cache['x0_hat_main_raw']  # [B, K, 4] raw cxcywh (新增缓存)
    x0_hat_a_raw = x0_hat_a  # 保持 raw cxcywh
    x0_hat_b_raw = x0_hat_b  # 保持 raw cxcywh

    # --- 4. 计算二阶差商 D2 (在 raw cxcywh 空间, 与 solver 一致, R1 反馈 7 修正) ---
    t_all = torch.stack([t_main, t_a, t_b], dim=1)  # [B, 3]
    x0_all = torch.stack([x0_hat_main_raw, x0_hat_a_raw, x0_hat_b_raw], dim=1)  # [B, 3, K, 4]

    sort_idx = t_all.argsort(dim=1)  # [B, 3]
    t_sorted = t_all.gather(1, sort_idx)
    idx_exp = sort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x0_all.size(2), x0_all.size(3))
    x0_sorted = x0_all.gather(1, idx_exp)  # [B, 3, K, 4]

    t_0, t_1, t_2 = t_sorted.unbind(dim=1)  # 各 [B], t_0 < t_1 < t_2 (升序)
    x0_0, x0_1, x0_2 = x0_sorted.unbind(dim=1)  # 各 [B, K, 4]

    t_0v, t_1v, t_2v = t_0.view(-1,1,1), t_1.view(-1,1,1), t_2.view(-1,1,1)
    dt_01 = (t_1v - t_0v).clamp(min=self.bear_min_dt)
    dt_12 = (t_2v - t_1v).clamp(min=self.bear_min_dt)
    dt_02 = (t_2v - t_0v).clamp(min=self.bear_min_dt)

    D1_01 = (x0_1 - x0_0) / dt_01
    D1_12 = (x0_2 - x0_1) / dt_12
    D2 = (D1_01 - D1_12) / dt_02  # [B, K, 4]

    # --- 5. 求解器感知权重 w = |psi| (R1 反馈 2 修正: 原为 |phi_2|, 应为截断误差积分核) ---
    # psi(t_mid, t_next) = (t_next^2 - t_mid^2) / 2 - 2*t_mid*(t_next - t_mid) + t_mid^2 * log(t_next / t_mid)
    # 注意: t_next < t_mid (因为积分方向 t_n -> t_{n+1}, 而 t_{n+1} < t_n)
    # 映射 (R1 反馈 1 修正): t_mid = t_1 (中间, current), t_next = t_0 (最小, next)
    t_mid = t_1v  # 中间值 (current)
    t_next_v = t_0v  # 最小值 (next)
    psi = 0.5 * (t_next_v**2 - t_mid**2) \
          - 2.0 * t_mid * (t_next_v - t_mid) \
          + t_mid**2 * torch.log(t_next_v.clamp(min=1e-6) / t_mid.clamp(min=1e-6))
    w = psi.abs()

    # --- 6. BEAR 损失 ---
    bear_loss = (w * (D2 ** 2)).sum(dim=-1).mean()

    probe.record_scalar('train/bear_loss', bear_loss.item())
    probe.record_scalar('train/bear_D2_norm', D2.norm(dim=-1).mean().item())
    probe.record_scalar('train/bear_psi', psi.abs().mean().item())

    return bear_loss
```

**单元测试 (R1 反馈 1 新增)**:

```python
def test_psi_numerical():
    """验证 BEAR 的 psi 计算与数值积分一致"""
    # t_mid = 0.5, t_next = 0.25
    # psi = 0.5*(0.0625 - 0.25) - 2*0.5*(0.25 - 0.5) + 0.25*log(0.5/0.5)
    #     = -0.09375 + 0.25 + 0.25*log(0.5) = -0.09375 + 0.25 - 0.1733 = -0.0170
    expected = -0.0170
    # ... (数值验证代码)


def test_phi2_matches_solver():
    """验证 BEAR 的 phi2 计算与 rectified_flow.py 一致 (消融对照)"""
    # 取等距 3 点 t_0=0.25, t_1=0.5, t_2=0.75
    # 对应 solver 的 (t_next, t_n, t_p) = (0.25, 0.5, 0.75)
    # 预期 phi2 = 0.0334 (与 rectified_flow.py L215-217 一致)
    expected = 0.0334
    # ... (数值验证代码)
```

### 7.2 修改 `head.py` 的 `loss()` 方法

在 [head.py L598 `loss()`](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py) 中, 在 `losses = self.criterion(...)` 之后追加 (R1 反馈 7 修正: 缓存 `x0_hat_main_raw` 而非 `x0_hat_main`):

```python
        losses = self.criterion(outputs, targets, t=t, box_targets=box_targets)

        # === BEAR: Backward-Error-Aware Regularization ===
        if self.bear_weight > 0:
            self._bear_cache = {
                'x_starts': torch.stack(x_starts) if isinstance(x_starts, list) else x_starts,
                'x_noises': torch.stack(x_noises) if isinstance(x_noises, list) else x_noises,
                # R1 反馈 7 修正: 不归一化, 保持 raw cxcywh (与 solver 的 x0_history 一致)
                'x0_hat_main_raw': all_pred_bboxes,  # raw cxcywh
            }
            bear_loss = self._compute_bear_loss(
                features, img_metas, gt_bboxes, gt_labels, t
            )
            losses['bear'] = bear_loss * self.bear_weight
            self._bear_cache = None
        # === BEAR END ===
```

### 7.3 修改 `__init__` 添加 BEAR 配置

```python
        # BEAR: Backward-Error-Aware Regularization
        self.bear_weight = bear_weight  # 默认 0.0, 典型值 0.01~0.5
        self.bear_min_dt = bear_min_dt  # 最小时间步间隔, 默认 0.05
        self._bear_cache = None
```

### 7.4 配置文件修改

新建 `configs/karyoflow_a4_bear.py`:

```python
_base_ = './karyoflow_a4_dpm_solver.py'
model = dict(
    head=dict(
        bear_weight=0.1,    # BEAR 正则强度, 消融: {0.01, 0.05, 0.1, 0.5}
        bear_min_dt=0.05,   # 最小时间步间隔
    ),
)
```

### 7.5 改动量总结

| 文件 | 改动 | 行数 |
|------|------|------|
| `ldmdet/core/head.py` | 新增 `_compute_bear_loss` 方法 (R1 修正后) | +70 |
| `ldmdet/core/head.py` | 修改 `loss()` 追加 BEAR 分支 | +10 |
| `ldmdet/core/head.py` | 修改 `__init__` 添加配置 | +5 |
| `ldmdet/core/head.py` | 新增 `test_psi_numerical` + `test_phi2_matches_solver` 单元测试 | +30 |
| `configs/karyoflow_a4_bear.py` | 新建配置 | +15 |
| **总计** | | **~130 行** |

**零改动文件**: `rectified_flow.py`, `criterion.py`, `sampling.py`.

---

## 8. 预期收益分析

### 8.1 截断误差降低的直接传导

由定理 2.2, BEAR 使全局截断误差 $\|x_N - x(0)\| \leq C_1 \varepsilon h + C_2 h^2$. 若 BEAR 使曲率 $\varepsilon$ 下降 $\rho$ (如 30%), 截断误差首项下降 $\rho$. 对 $h = 0.25$ ($S = 4$), 截断误差首项的绝对下降量可观.

### 8.2 与 mAP 的关联

$\hat{x}_0$ 的定位误差由网络预测误差 (训练损失控制) 和求解器截断误差 (BEAR 控制) 组成. BEAR 降低后者. 在高 IoU 阈值下, 定位误差对 AP 的影响近似线性 (经验假设, 需 Phase 2 验证, R1 反馈 11), 故 mAP$_{75}$ 改善 $\propto \rho$. 经验估计: $\Delta \text{mAP}_{75} \approx +0.003 \sim +0.007$ (R2 下调).

### 8.3 稀有类的额外收益

稀有类 (Y, G21) 的 $\hat{x}_0$ 曲率 $\|D_2\|$ 更大, BEAR 正则压力自然集中于稀有类, 起到隐式 class-balanced 效果. 预期 Y 染色体 AP 改善 +0.004~+0.014.

### 8.4 与 TFR/VCR 联合的协同效应 (R1 反馈 5 修正)

BEAR (二阶) + TFR (一阶) 联合: TFR 降低 $D_1$, BEAR 降低 $D_2$. **但 TFR 是 BEAR 的特例, 联合使用退化为强制 $\hat{x}_0$ 常数, 与单独 TFR 效果部分重叠** (R1 反馈 5 修正: 删除原 "联合收益 +0.004~+0.012" 过强声称). 联合收益预期与单独 BEAR 相当.

### 8.5 定量对比预期 (R1 反馈 5 + 9 修正)

| 配置 | mAP | mAP$_{75}$ | $\eta_{\text{3rd}}$ | 训练开销 | 推理 NFE |
|------|-----|-----------|---------------------|---------|---------|
| Baseline (+DPM-Solver++) | 0.863 | 0.972 | 0.7~1.5 | 1x | 4 |
| + TFR | 0.865~0.869 | 0.974~0.980 | 0.4~0.8 | 2x | 4 |
| + BEAR | 0.865~0.868 | 0.975~0.979 | 0.5~1.0 | 1.67x (R1 反馈 9 修正) | 4 |
| + TFR + BEAR (R1 反馈 5 修正) | 0.866~0.869 | 0.976~0.980 | 0.4~0.8 | 1.67x | 4 |
| + VCR + BEAR | 0.866~0.869 | 0.976~0.980 | 0.4~0.8 | 2.5x | 4 |

---

## 9. 风险分析 (R1 反馈 4 + 9 修正)

### 9.1 风险 1: 3 次前向的训练开销 (R1 反馈 9 修正: +60~70%)

**风险**: BEAR 需 3 次网络前向, 训练耗时增加 ~1.67x (R1 反馈 9 修正: 原 +30~50%, 实际 +60~70%).
**缓解**: R1-A AMP 半精度; R1-B 辅助步不计算检测损失; R1-C 每 $k$ 步正则 1 次 ($k=4$, 开销 1.17x); R1-D 共享 backbone 特征 (开销 ~1.5x).
**判据**: 若训练耗时 > 1.67x 且 mAP 改善 < +0.002, 启用 R1-C/D.

### 9.2 风险 2: $D_2$ 数值不稳定

**风险**: $|t_i - t_j|$ 小时 $D_2$ 分母小, 数值放大.
**缓解**: R2-A `bear_min_dt=0.05` 钳制; R2-B $\|D_2\|$ 钳制上界; R2-C 梯度裁剪.

### 9.3 风险 3: BEAR 与检测损失梯度冲突

**风险**: BEAR 梯度 (约束 $\hat{x}_0$ 光滑) 可能与 $\mathcal{L}_{\text{det}}$ 梯度 (约束 $\hat{x}_0 \to x_0$) 冲突.
**缓解**: R3-A 小 $\lambda$ (0.01~0.1); R3-B 仅在 $t \in [0.2, 0.8]$ 采样; R3-C 监控梯度余弦相似度.

### 9.4 风险 4: 修正方程假设失效

**风险**: 修正方程定理假设 $\hat{x}_0$ 充分光滑 ($C^3$), 神经网络可能不满足.
**缓解**: R4-A 网络激活使 $\hat{x}_0$ 近似光滑; R4-B 差商用离散点, 对非光滑仍有效; R4-C Phase 0 诊断验证.

### 9.5 风险 5: 改善幅度低于预期

**风险**: $\|D_2\|$ 本身已很小 (如 TFR 已使 $\hat{x}_0$ 接近常数).
**缓解**: R5-A Phase 0 诊断优先; R5-B 退为理论分析; R5-C 联合 TFR.

### 9.6 风险 6: 与 TFR + EXER-RF (k=2) 双重重叠 (R1 反馈 3 + 5 扩展)

**风险**: 审稿人质疑 BEAR 与 TFR 仅是 "一阶 vs 二阶导数" (R1 反馈 5), 且与 EXER-RF (k=2) 正则同一对象 (R1 反馈 3).
**缓解**: R6-A §5.2 严格区分 (BEAR 允许线性, TFR 强制常数); R6-B 求解器感知权重 $|\psi|$ 独有; R6-C 联合实验证明互补; R6-D §5.3 显式澄清与 EXER-RF (k=2) 的关系, Phase 4 增加直接对比消融.

### 9.7 风险 7: 训练-推理迁移 gap (R1 反馈 4 新增)

**风险**: 训练时 $D_2$ 在精确 RF 轨迹上计算, 推理时 $D_2$ 在数值轨迹上计算. 减小训练 $D_2$ 不严格保证减小推理 $D_2$.
**缓解**: R7-A 假设 H1 (沿任意轨迹的 $D_2$ Lipschitz 性, §3.4); R7-B Phase 0 诊断验证训练-推理相关性 $\rho > 0.7$; R7-C 若 $\rho < 0.5$, BEAR 失效风险高, 考虑放弃.

### 9.8 风险总结

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| R1: 训练开销 1.67x (R1 反馈 9 修正) | 高 | 中 | AMP + 间隔正则 + 共享特征 |
| R2: $D_2$ 不稳定 | 中 | 低 | min_dt 钳制 + 梯度裁剪 |
| R3: 梯度冲突 | 中 | 中 | 小 $\lambda$ + $t \in [0.2, 0.8]$ |
| R4: 光滑假设失效 | 低 | 高 | Phase 0 诊断 |
| R5: 改善低于预期 | 中 | 中 | Phase 0 诊断 + 联合 TFR |
| R6: 与 TFR + EXER-RF (k=2) 双重重叠 (R1 反馈 3+5 扩展) | 中 | 中 | §5.2 + §5.3 严格区分 + Phase 4 直接对比消融 |
| R7: 训练-推理迁移 gap (R1 反馈 4 新增) | 中 | 高 | 假设 H1 + Phase 0 诊断 $\rho > 0.7$ |

---

## 10. 实验计划

### Phase 0: 诊断验证 (0.5 天, 零训练, R1 反馈 4 新增训练-推理相关性)

**目标**: 验证 $\|D_2\|$ 分布, 判断 BEAR 是否有修复空间; 验证训练-推理迁移假设 H1.

**方法**: 用 baseline +DPM-Solver++ 模型推理, 记录 $D_1, D_2, \eta_{\text{str}}, \eta_{\text{3rd}}$ (代码已有 `eta_str_history`, `eta_3rd_history` 诊断). 同时计算训练时 (精确轨迹) 和推理时 (数值轨迹) 的 $\|D_2\|$, 验证两者相关性 $\rho$.

**判据**:
- 继续: $\|D_2\| > 0.01$ 且 $\rho > 0.7$;
- 放弃: $\|D_2\| < 0.005$ 或方差极大 或 $\rho < 0.5$ (训练-推理迁移 gap 大).

### Phase 1: $\lambda$ 消融 (3 天, 4 配置)

| 配置 | $\lambda_{\text{BEAR}}$ | 备注 |
|------|------------------------|------|
| BEAR-0.01 | 0.01 | 弱正则 |
| BEAR-0.05 | 0.05 | 中正则 |
| BEAR-0.1 | 0.1 | 强正则 (推荐) |
| BEAR-0.5 | 0.5 | 极强正则 |

**判据**: 最优 $\lambda^* = \arg\max \text{mAP}$; 若全部 mAP < baseline - 0.003, BEAR 有害.

### Phase 2: 3-seed 验证 (4.5 天)

**配置**: BEAR-$\lambda^*$, 3 seeds, 150 epochs; baseline 3 seeds (已有).
**判据**: 成功: mAP > baseline + 0.003; 边际: $|\Delta| < 0.003$; 失败: $\Delta < -0.003$.

### Phase 3: 联合实验 (3 天, R1 反馈 5 修正: 联合收益下调)

**配置**: BEAR + TFR; BEAR + VCR; BEAR + TFR + VCR.
**判据 (R1 反馈 5 修正)**: 联合 > 单独 max + 0.001 → 协同有效. 但 TFR + BEAR 联合预期与单独 BEAR 相当 (因 TFR 是 BEAR 特例).

### Phase 4: 论文实验 + BEAR vs EXER-RF (k=2) 直接对比 (3 天, R1 反馈 3 新增)

1. 主表: KaryoFlow + BEAR vs baseline vs DINO R50 vs RTMDet-L;
2. 消融: $\lambda$ 消融, $w = |\psi|$ vs $|\phi_2|$ vs 均匀权重 (R1 反馈 2 新增);
3. **BEAR vs EXER-RF (k=2) 直接对比消融** (R1 反馈 3 新增): $|\psi|$ (BEAR) vs $\lambda_{ext}=7$ (EXER-RF) vs 均匀权重; 若两者增益无显著差异, 合并为单一方向;
4. per-class AP (重点 Y, G21);
5. $\|D_2\|$ vs $t$ 曲线;
6. 修正方程验证 ($\eta_{\text{str}}$, $\eta_{\text{3rd}}$).

**总工期**: 14 天.

---

## 11. R2 评审剩余问题 (供后续 R3 参考)

1. **与 EXER-RF (k=2) 的最终去留 (R2 剩余 1, 中等)**: A 保留两个独立方向, 但论证依赖软指标 ("叙事价值" + "权重设计不同"). 建议 R3 评估是否强制合并 (保留 BEAR 的修正方程叙事 + $|\psi|$ 权重, 将 EXER-RF 的 Lebesgue 常数作为消融对照). Phase 4 的 BEAR vs EXER-RF (k=2) 直接对比消融是关键决策点.

2. **$\psi$ 的 R1 数值错误 (R2 剩余 2, 轻微)**: B 的 R1 给出 $\psi \approx -0.0069$ 有误, 正确值为 $\psi \approx -0.0170$ (A 的闭式正确). 不影响定性结论 ($\psi \neq \phi_2$), R3 应以 A 的数值为准.

3. **TFR + BEAR 联合使用的实际价值 (R2 剩余 3, 中等)**: 修订后 A 承认联合使用 "退化为强制 $\hat{x}_0$ 常数", 联合收益与单独 BEAR 相当. R3 应评估是否在论文中删除联合实验, 或重新定位联合场景 (如 "线性变化在任务中不合法" 的场景).

4. **Phase 0 诊断的决策门槛 (R2 剩余 4, 轻微)**: A 新增训练-推理相关性 $\rho > 0.7$ 的判据, 但该阈值缺乏先验依据. R3 应评估是否需要更严格的判据 (如 $\rho > 0.8$) 或增加备用判据 (如 $\|D_2^{\text{infer}}\|$ 的绝对值).

5. **定理 2.1 证明的严格性 (R2 剩余 5, 中等)**: 仍为概要形式, Ch. XV 引用修正后未展开多步法 BEA 的具体构造. R3 若要求 TMI 级严格性, 应补充完整证明或引用具体定理编号.

---

## 12. 自评 (修订后)

### 12.1 评分 (10 分制, R2 修正后)

| 维度 | 分数 | 理由 |
|------|------|------|
| 理论严谨性 | 7.5 | $\phi_2$ 映射修正完全正确 (数学验证); $\psi$ 闭式表达式完全正确 (数学验证); Ch. IX → Ch. XV 引用修正合理; 训练-推理 Lipschitz 假设 H1 严格. 扣分: 定理 2.1 证明仍为概要, $C_2$ 常数未显式 |
| 实现可行性 | 7.0 | $\phi_2$ bug 修正 (4.75x 偏差消除); raw 空间统一 (与 solver 一致); 单元测试新增 (验证 $\phi_2$ 与代码一致性). 扣分: 1.67x 开销仍高于 MEC-RF (1.3x) 和 TFR (2x 但仅 1 次额外前向) |
| 风险可控 | 7.5 | 7 个风险每个有 fallback; Phase 0 诊断新增训练-推理相关性 $\rho > 0.7$ 验证; $\lambda=0$ 零风险回退. 扣分: 与 EXER-RF 的关系澄清后, R6 与 TFR 重叠风险扩展为 "与 TFR + EXER-RF (k=2) 双重重叠" |
| 新颖性 | 6.5 | 向后误差分析 + 修正方程是 60+ 年经典理论; "二阶方法精度瓶颈在 $D_2$" 洞察优雅. 扣分: 与 EXER-RF (k=2) 冗余承认后, 独立新颖性未提升; 权重从 $|\phi_2|$ 改为 $|\psi|$ 是修正而非创新 |
| 任务匹配 | 8.0 | d=4 低维 $D_2$ 计算 O(d) 可忽略; S=4 大步长截断误差关键; RF 直线理想严格推广. 扣分: $\eta_{3rd}$ 跟踪修正后, 24 类不平衡的隐式 class-balanced 效果仍需 Phase 0 验证 |
| 预期增益 | 6.5 | mAP +0.002~+0.005, mAP@75 +0.003~+0.007; 截断误差路径方向正确. 扣分: 联合 TFR 收益下调 (因 TFR 是 BEAR 特例); 增益仍依赖 Phase 0 诊断 |
| 可证伪性 | 8.5 | Phase 0 诊断新增训练-推理相关性 $\rho > 0.7$ 判据; $\lambda$ 消融 (0.01/0.05/0.1/0.5) + 3-seed 验证; $|\psi|$ vs $|\phi_2|$ vs 均匀权重消融 (新增); BEAR vs EXER-RF (k=2) 直接对比消融 (新增) |
| 论文价值 | 6.5 | T1 (BEA 引入 RF 检测) + T3 (二阶曲率正则) 是独立贡献. 扣分: T2 (修正方程与 DPM-Solver++ 截断误差关系) 需 Ch. XV 补证; 与 EXER-RF 关系澄清后独立贡献减弱 (承认同一正则目标的两种推导); 联合 TFR 收益下调削弱 "全阶导数约束框架" 叙事 |

**综合评分**: 7.3/10

### 12.2 核心优势保留

- "二阶方法精度瓶颈在 $D_2$" 的洞察 (定理 2.1 注 2.2)
- "BEAR 是 TFR 的严格推广" (允许线性 vs 强制常数, R1 反馈 5 修正后叙事价值减弱但仍成立)
- 求解器感知权重 $|\psi|$ (R1 反馈 2 修正后严格, 而非原 $|\phi_2|$ 启发式)
- 修正方程理论框架 (R1 反馈 6 修正后 Ch. IX + Ch. XV 引用严格)
- 训练-推理 Lipschitz 假设 H1 (R1 反馈 4 新增, Phase 0 验证)
- raw 空间 $D_2$ 计算 (R1 反馈 7 修正, 与 solver 一致)
- 推理零开销 (NFE=4)

### 12.3 核心弱点承认

- 定理 2.1 证明仍为概要形式, Ch. XV 引用修正后未展开多步法 BEA 的具体构造
- $C_2$ 常数未显式 (R1 反馈 8 部分接受)
- 与 EXER-RF (k=2) 冗余 (R1 反馈 3 部分接受, 保留独立方向但论证依赖软指标)
- 联合 TFR 收益下调 (R1 反馈 5 修正, 因 TFR 是 BEAR 特例)
- 训练-推理迁移假设 H1 为经验假设, 需 Phase 0 验证

### 12.4 投稿建议

1. **TMI 正文**: "基于向后误差分析的二阶曲率正则化" 作为训练正则化贡献. 强调:
   - 首次将 BEA 引入 RF 检测 (T1);
   - 二阶方法精度瓶颈在 $D_2$ 的洞察 (定理 2.1 注 2.2);
   - 求解器感知权重 $|\psi|$ 严格来自截断误差积分 (R1 反馈 2 修正后);
   - BEAR 是 TFR 的严格推广 (允许线性 vs 强制常数, R1 反馈 5 修正后).

2. **arXiv companion**: 定理详细证明 (含 Ch. XV 多步法 BEA 构造), Phase 0 诊断完整数据 (含训练-推理相关性 $\rho$), Phase 4 完整消融表 (含 BEAR vs EXER-RF (k=2) 直接对比), per-class AP 详细图.

3. **创新点叙事**: "首次将向后误差分析 (Wilkinson 1963) 和修正方程 (Hairer-Lubich-Wanner 2006, Ch. XV) 引入 RF 检测, 从 DPM-Solver++ 修正方程缺陷项推导 $\hat{x}_0$ 二阶曲率正则, 以截断误差积分核 $|\psi|$ 作为求解器感知权重, 是 TFR (一阶) 的严格推广, 允许 $\hat{x}_0$ 线性变化的同时保持二阶求解器精度."

4. **与 EXER-RF (k=2) 关系处理**: 论文中诚实标注 "BEAR 与 EXER-RF (k=2) 是同一正则目标的两种理论推导, BEAR 的修正方程视角提供了更直接的截断误差桥梁, EXER-RF 的 Newton 余项视角提供了更广泛的 k 阶统一框架", 并在 Phase 4 实验中提供直接对比消融.

---

## 附录 A: 符号表

| 符号 | 含义 |
|------|------|
| $x_0$ | GT bbox (normalized cxcywh, $d=4$) |
| $x_1$ | 高斯噪声 |
| $x_t$ | 前向插值 $(1-t)x_0 + t x_1$ |
| $\hat{x}_0(t)$ | 网络预测 $f_\theta(x_t, t) \in \mathbb{R}^4$ |
| $v_\theta$ | 学习速度场 $(x_t - \hat{x}_0)/t$ |
| $D_1$ | $\hat{x}_0$ 的一阶差商 (均差) |
| $D_2$ | $\hat{x}_0$ 的二阶差商 (均差) |
| $\phi_1, \phi_2$ | DPM-Solver++ 核函数 ($\phi_1$ 二阶校正, $\phi_2$ 三阶校正) |
| $\psi$ | **截断误差积分核** $\int_{t_n}^{t_{n+1}} \frac{(t-t_n)^2}{t} dt$ (R1 反馈 2 修正, $\psi \neq \phi_2$) |
| $\delta_h$ | 修正方程的向后误差 (缺陷) |
| $\delta_1, \delta_2$ | 修正方程的首项, 次项 |
| $\lambda_{\text{BEAR}}$ | BEAR 正则化强度 |
| $w = |\psi|$ | 求解器感知权重 (R1 反馈 2 修正, 原为 $|\phi_2|$) |
| $\eta_{\text{str}}$ | RF 轨迹直度诊断 $\|D_1\|/\|\hat{x}_0\|$ |
| $\eta_{\text{3rd}}$ | 三阶校正诊断 $\|D_2\|/\|\hat{x}_0\|$ (R1 反馈 10 修正, BEAR 主要跟踪 $\eta_{3rd}$) |
| $L_x$ | $\hat{x}_0$ 关于 $x$ 的 Lipschitz 常数 (假设 H1) |
| $\rho$ | 训练-推理 $D_2$ 相关性 (Phase 0 诊断) |
| $N$ | DPM-Solver++ 步数 ($N = 4$) |
| $K$ | proposals 数量 ($K \approx 46$) |
| $d$ | bbox 维度 ($d = 4$) |
| $h$ | 求解器步长 ($h = 1/N = 0.25$) |

## 附录 B: 与 TFR 的叠加方案 (R1 反馈 5 修正)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda_1 \underbrace{\|\dot{\hat{x}}_0\|^2}_{\text{TFR}} + \lambda_2 \underbrace{\|\ddot{\hat{x}}_0\|^2}_{\text{BEAR}}$$

**协同 (R1 反馈 5 修正)**: TFR 使 $\hat{x}_0 \to$ 常数 ($\dot{\hat{x}}_0 \to 0$); 若 TFR 完全成功则 $\ddot{\hat{x}}_0 = 0$ (BEAR 自动满足). **但联合 TFR + BEAR 退化为 '强制 $\hat{x}_0$ 常数' (因 TFR 的理想是 BEAR 理想的子集)**. BEAR 单独已能允许线性变化, 联合 TFR 会进一步约束到常数, 适用于 '线性变化在任务中不合法' 的场景 (如染色体 bbox 不应有系统性时间偏移).

**超参数 (R1 反馈 5 修正)**: $\lambda_1 = 0.1$ (TFR 推荐), $\lambda_2 = 0.1$ (BEAR 推荐). **若联合使用, 建议 $\lambda_1 \leq \lambda_2 / 2$ (TFR 是 BEAR 的特例, 过强 $\lambda_1$ 会过度约束线性变化, 削弱 BEAR 的优势)** (R1 反馈 5 修正: 删除原 "若冲突, 优先 BEAR").

## 附录 C: DPM-Solver++ 核函数推导 + $\psi$ 闭式 (R1 反馈 2 修正)

### $\phi_1$ (二阶校正)

$$\phi_1 = \int_{t_n}^{t_{n+1}} \frac{t - t_n}{t} dt = t_{n+1} \ln\frac{t_n}{t_{n+1}} - t_n + t_{n+1}$$

与 [rectified_flow.py L201](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py) 一致.

### $\phi_2$ (三阶校正, 消融对照)

$$\phi_2 = \frac{1}{2} \int_{t_n}^{t_{n+1}} \frac{(t - t_n)(t - t_{n-1})}{t} dt$$

展开积分后与代码 [rectified_flow.py L218](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py) 一致: `phi2 = (t_next + t_p) * (t_n - t_next) - t_next * (t_n + t_p) * log(t_n / t_next)`.

**变量映射 (R1 反馈 1 修正)**: 排序后 $t_0 < t_1 < t_2$, 正确映射 $(t_a, t_b, t_c) = (t_1, t_2, t_0)$ (中间值作 current, 最大值作 previous, 最小值作 next):

$$\phi_2 = (t_0 + t_2)(t_1 - t_0) - t_0(t_1 + t_2)\ln(t_1 / t_0)$$

**数值验证** ($t_0 = 0.25, t_1 = 0.5, t_2 = 0.75$): $\phi_2^{correct} = (0.25 + 0.75)(0.5 - 0.25) - 0.25(0.5 + 0.75)\ln(0.5/0.25) = 0.25 - 0.2166 = 0.0334$ (与代码 L215-217 一致, 原 buggy 版 $\phi_2 = -0.0070$, 偏差 4.75x).

### $\psi$ (截断误差积分核, R1 反馈 2 修正, 主方案权重)

$$\psi(t_n, t_{n+1}) = \int_{t_n}^{t_{n+1}} \frac{(t - t_n)^2}{t} dt = \frac{t_{n+1}^2 - t_n^2}{2} - 2t_n(t_{n+1} - t_n) + t_n^2 \ln\frac{t_{n+1}}{t_n}$$

**数值验证** ($t_n = 0.5, t_{n+1} = 0.25$): $\psi = \frac{0.0625 - 0.25}{2} - 2 \cdot 0.5 \cdot (0.25 - 0.5) + 0.25 \ln(0.5) = -0.09375 + 0.25 - 0.1733 = -0.0170$ (A 的闭式正确, R1 的 $\psi \approx -0.0069$ 有误, R2 确认 A 的数值).

**$|\psi|$ vs $|\phi_2|$ 对比**: $|\psi| \approx 0.0170$ vs $|\phi_2| \approx 0.0334$, 差异约 2x. $\psi$ 仅依赖 $(t_n, t_{n+1})$, $\phi_2$ 依赖 $(t_{n-1}, t_n, t_{n+1})$. 两者在 $t \to 0$ 时均增大, 定性趋势一致, 但数值不同. **BEAR 主方案使用 $|\psi|$, $|\phi_2|$ 作为消融对照** (R1 反馈 2 修正).

---

<!-- 文档结束. BEAR: Backward-Error-Aware Regularization. R1 修正: $\phi_2$ 变量映射 (反馈 1) + 权重 $|\psi|$ (反馈 2) + EXER-RF 关系澄清 (反馈 3) + 训练-推理 Lipschitz 假设 H1 (反馈 4) + TFR 严格推广 (反馈 5) + Ch. XV 引用 (反馈 6) + raw 空间 $D_2$ (反馈 7) + $C_1$ 部分估计 (反馈 8) + 开销 1.67x (反馈 9) + $\eta_{3rd}$ 跟踪 (反馈 10) + mAP@75 经验假设标注 (反馈 11). 与 TFR (一阶) 严格推广, 与 VCR 正交可叠加, 与 EXER-RF (k=2) 同一正则目标两种推导. 代码集成: ldmdet/core/head.py (loss, _compute_bear_loss). 核心创新: 将向后误差分析 (Wilkinson 1963) 和修正方程 (Hairer-Lubich-Wanner 2006, Ch. IX + XV) 引入 RF 检测, 从 DPM-Solver++ 修正方程缺陷项推导 $\hat{x}_0$ 二阶曲率正则, 以截断误差积分核 $|\psi|$ 作为求解器感知权重. -->
