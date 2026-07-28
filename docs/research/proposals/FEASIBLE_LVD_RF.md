# FEASIBLE LVD-RF: Lyapunov 速度方向正则化 (FINAL 整合设计文档)

> **方向类别**: 保守方向 (基于已有成熟动力学系统理论的可靠性改进)
> **方案代号**: LVD-RF (Lyapunov Velocity Direction Regularization)
> **理论基础**: Lyapunov 稳定性理论 (Lyapunov 1892) + 收缩理论 (Lohmiller-Slotine 1998) + Rectified Flow x0-prediction (Liu-Gong-Liu 2023) + DPM-Solver++ (Lu et al. 2022) + ProReflow 方向分解 (Ke et al., CVPR 2025)
> **核心改动**: 在训练损失中增加 **Lyapunov 方向余弦正则项** $\mathcal{L}_{\text{LVD}}$ (默认 sin² 形式), 利用 $d=4$ 低维优势以**零额外前向传播**计算方向余弦, 通过 Lyapunov 稳定性条件约束速度场方向, 间接降低 $\eta_{\text{str}}$ 并减少 DPM-Solver++ 截断误差
> **目标**: 在不改网络架构、不改 NFE ($S=4$)、不改 coupling/solver 的前提下, 把 Dataset 2 baseline mAP=0.863 提升至 0.865~0.870
> **预期收益 (R2 确认)**: mAP $+0.002 \sim +0.007$; mAP$_{75}$ $+0.001 \sim +0.004$; Y 染色体 AP $+0.002 \sim +0.010$; per-dim $\eta_{\text{str}}$ 下降 30~43%; 推理 NFE 保持 24, 推理零额外开销, 训练开销 $<2\%$

**文档状态**: FINAL (整合 R1 评审 → A 响应 → R2 评审全部修订, 可直接进入实现)
**撰写日期**: 2026-07-28
**R2 最终评分**: 7.5/10 (推荐, R1: 7.1, 修订后自评: 7.5)
**目标期刊**: IEEE TMI
**依赖文件**: [rectified_flow.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py), [head.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py), [criterion.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py), [FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md), [EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

---

## 1. 摘要: 核心论断与预期收益

### 1.1 核心论断

KaryoFlow 采用 Rectified Flow (RF) 的 **data-prediction** (x0-prediction) 形式, 推理使用 **DPM-Solver++ 二阶多步法** ($S=4$ 步). 理想 RF 中, 速度场 $v^* = x_1 - x_0$ 恒定, 轨迹为直线. 实际训练中, 网络预测 $\hat{x}_0(x_t, t; \theta)$ 偏离 $x_0$, 导致速度场 $v_\theta = (x_t - \hat{x}_0)/t$ 的方向偏离理想方向 $(x_t - x_0)/t$, 采样轨迹发生弯曲, DPM-Solver++ 截断误差随之增大. 本仓库已有的 $\eta_{\text{str}}$ 诊断 ([rectified_flow.py L141-L192](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py)) 实测 $\eta_{\text{str}} \in [4, 50]$, per-dim 诊断显示 cx/cy 维 $\eta_{\text{str}} = 17\sim50$ 而 w/h 维仅 $4\sim11$, 表明速度场方向偏差在中心维度上尤为严重.

**本方案 LVD-RF (Lyapunov Velocity Direction Regularization)**: 以 **Lyapunov 稳定性理论** 为基础, 构造以真值 $x_0$ 为平衡点的 Lyapunov 函数 $V(x) = \frac{1}{2}\|x - x_0\|^2$, 推导出保证采样过程收敛的 **方向余弦正则化** 损失 (默认 sin² 形式以缓解梯度消失):

$$
\mathcal{L}_{\text{LVD}}(\theta) = \mathbb{E}_{(x_0, x_1) \sim \mathcal{D}, \, t \sim \mathcal{U}(0,1)} \left[ \sin^2(\alpha) \right] = \mathbb{E}\left[1 - \left(\frac{(x_t - \hat{x}_0)^\top (x_t - x_0)}{\big(\|x_t - \hat{x}_0\| + \epsilon\big)\big(\|x_t - x_0\| + \epsilon\big)}\right)^2\right]
$$

其中 $\alpha$ 为 $v_\theta$ 与 $(x_t - x_0)$ 的夹角, $\epsilon = 10^{-6}$ 为数值稳定常数. 由定理 2.4, $1/t$ 在分子分母抵消, 可直接用 $(x_t, \hat{x}_0, x_0)$ 计算, 避免显式除以 $t$.

### 1.2 核心论断 (修订后)

1. **(per-proposal 条件 Lyapunov 稳定性保证, 定理 2.1 + §2.7)** 构造 GT 锚定 Lyapunov 函数 $V(x) = \frac{1}{2}\|x - x_0\|^2$. 在采样 (逆向) 过程中, $V$ 沿轨迹的变化率 $\dot{V} = -(x - x_0)^\top v_\theta$. 当 $\cos(v_\theta, x - x_0) \geq \delta > 0$ 时, $\dot{V} \leq -\delta \|x - x_0\| \|v_\theta\| \leq 0$, 由 Lyapunov 直接法保证采样轨迹收敛至 $x_0$. LVD-RF 直接优化该余弦项. **修订说明**: 经 R1 反馈 K2, 此为 $K \approx 46$ 个独立 per-proposal Lyapunov 函数 (非传统单一平衡点), 由 §2.7 多平衡点 Lyapunov 合法性论证支撑 (独立性 + 隔离性 + per-proposal 收敛, 对应 Khalil §4.6).

2. **(方向约束的 Lyapunov 必要性, 独立推导 + ProReflow CVPR 2025 经验辅助)** (a) **Lyapunov 必要性 (独立)**: 由定理 2.1, Lyapunov 下降条件 $\dot{V} \leq 0$ 等价于 $\rho = \cos(v_\theta, x - x_0) \geq 0$. 方向约束 ($\rho \geq \delta$) 是 Lyapunov 稳定性的**直接要求**, 而幅度 $\|v_\theta\|$ 不出现在下降条件中. (b) **ProReflow 经验支持 (辅助)**: ProReflow (Ke et al., **CVPR 2025**, pp. 28029-28038, arXiv:2503.04824) 的实验独立验证了 "方向优于幅度" 的经验结论 (Figure 1(b): 方向噪声比幅度噪声导致更大 FID 退化). 此经验结论与 Lyapunov 理论推导**一致但非依赖**: Lyapunov 证明必要性, ProReflow 证明充分性.

3. **(与 TFR 的核心区分)** TFR 最小化 $\|\hat{x}_0(t) - \mathbb{E}[\hat{x}_0]\|^2$ (x0 的时间方差, 幅度正则); LVD-RF 最小化方向余弦偏差 (方向正则, 尺度不变). TFR 禁止 x0 漂移 (锁定均值), LVD-RF 允许 x0 漂移只要方向对齐 (减少过拟合风险).

4. **(与 VCR 的核心区分)** VCR 在速度空间最小化 $\|v_\theta(t) - v_\theta(t')\|^2$ (跨时间速度一致性, 幅度正则, 需双前向); LVD-RF 在方向空间最小化角度偏差 (单前向, $1/t$ 在余弦中抵消, $t \to 0$ 稳定).

5. **(与已诊断 $\eta_{\text{str}}$ 的闭环)** $\eta_{\text{str}} = \|D_1\| / \|\hat{x}_0\|$ (R1 诊断) 测量 x0 预测的时间差商. 由定理 2.3' (条件性界), LVD-RF 通过方向约束间接降低 $\eta_{\text{str}}$, 形成 "诊断 → 正则 → 验证" 闭环.

6. **(d=4 低维红利 + 零额外前向)** 与 TFR/VCR 需要双前向不同, LVD-RF 的方向余弦仅需训练步已有的 $(x_t, \hat{x}_0, x_0)$, **零额外前向传播**. 在 $d=4$ 下方向计算 $O(N \cdot d) = O(N \cdot 4)$ 可忽略, 训练开销 $< 2\%$.

7. **(推理零开销)** LVD-RF 仅训练时生效, 不改变模型架构、不改变 RF 路径、不改变 DPM-Solver++ 推理流程、不增加 NFE (仍为 24).

### 1.3 预期收益汇总 (R2 确认)

| 指标 | Baseline (Dataset 2, A4 DPM++) | LVD-RF 预期 | 改善 | 备注 |
|------|-------------------------------|-----------|------|------|
| mAP | 0.863 | 0.865 ~ 0.870 | $+0.002 \sim +0.007$ | $\eta_{\text{str}}$ 降低 → 截断误差降 → 4 步精度升 |
| mAP$_{50}$ | 0.990 | 0.990 ~ 0.991 | $+0.000 \sim +0.001$ | 已饱和, 增量小 |
| mAP$_{75}$ | 0.974 | 0.975 ~ 0.978 | $+0.001 \sim +0.004$ | 方向对齐 → 轨迹更直 → 定位精度升 |
| Y 染色体 AP | 0.771 (3-seed) | 0.772 ~ 0.781 | $+0.002 \sim +0.010$ | 类无关正则 + 小数据泛化收紧 |
| per-dim $\eta_{\text{str}}$ 均值 | cx/cy ~30, h ~7 | cx/cy ~17, h ~5 | -30~43% | 方向对齐 → $\hat{x}_0$ 稳定 |
| 训练 cos_sim | (未测) | > 0.93 | — | LVD-RF 直接优化目标 (条件性 $O(\sqrt{\epsilon})$ 界) |
| 推理 NFE | 24 | 24 | 0 | 仅改训练 |
| 训练开销 | baseline | $+<2\%$ | — | 零额外前向 |
| 推理耗时 | baseline | $+0\%$ | 0 | 仅训练时正则 |

**收益分布的直觉解释**: mAP$_{75}$ 改善大于 mAP$_{50}$ (高 IoU 阈值对轨迹直线度更敏感); Y/G21 改善来自类无关正则 (稀有类不受额外惩罚); $\eta_{\text{str}}$ 下降但不归零 (定理 2.3' 为条件性界, 实际下降幅度依赖 $\|v_\theta\|$ 的稳定性).

---

## 2. 第一性原理推导

### 2.1 基本设定与记号

| 符号 | 含义 |
|------|------|
| $x_0 \in \mathbb{R}^{K \times 4}$ | 真实 bbox (cxcywh, 归一化), $K \approx 46$ |
| $x_1 \in \mathbb{R}^{K \times 4}$ | 高斯噪声 ($\sim \mathcal{N}(0, I)$) |
| $x_t = (1-t) x_0 + t x_1$ | RF 插值样本, $t \in [0, 1]$ |
| $\hat{x}_0(x_t, t; \theta)$ | 网络预测的 $x_0$ (x0-prediction 参数化) |
| $v_\theta(x_t, t) = (x_t - \hat{x}_0) / t$ | 由 x0 预测导出的速度场 (代码 `rectified_flow.py:65-68`) |
| $v^* = x_1 - x_0$ | 理想恒定速度 (直线轨迹) |
| $V(x) = \frac{1}{2}\|x - x_0\|^2$ | GT 锚定 Lyapunov 函数 |
| $\rho = \cos(v_\theta, x_t - x_0)$ | 方向余弦 |
| $\eta_{\text{str}} = \|D_1\| / \|\hat{x}_0\|$ | 直线度诊断指标 (R1) |
| $D_1 = [\hat{x}_0(t_n) - \hat{x}_0(t_{n-1})] / (t_n - t_{n-1})$ | x0 预测一阶差商 |

### 2.2 RF x0-prediction ODE

**定义 2.1 (RF x0-prediction ODE)**. 给定训练好的网络 $\hat{x}_0(\cdot, \cdot; \theta)$, RF 的前向 ODE 为:

$$
\frac{dx}{dt} = v_\theta(x, t) = \frac{x - \hat{x}_0(x, t; \theta)}{t}, \quad t \in (0, 1], \quad x(1) = x_1
$$

**性质 2.1 (RF 直线轨迹理想)**. 若 $\hat{x}_0 \equiv x_0$ (常数), 则 $v_\theta = (x - x_0)/t$, 解为 $x(t) = (1-t) x_0 + t x_1$ (直线), $v_\theta = x_1 - x_0 = v^*$ (恒定). 此时 $D_1 \equiv 0$, $\eta_{\text{str}} \equiv 0$.

**注 2.1**. 实际训练中 $\hat{x}_0$ 是神经网络输出, 不可能精确为常数. R1 诊断实测 $\eta_{\text{str}} \in [4, 50]$, 表明速度场方向显著偏离理想方向.

### 2.3 Lyapunov 稳定性理论基本定义

**定义 2.2 (Lyapunov 函数, Lyapunov 1892)**. 考虑动态系统 $\dot{x} = f(x, t)$, 平衡点 $x^*$. 标量函数 $V(x, t)$ 称为 Lyapunov 函数, 若满足: (1) 正定性: $V(x^*, t) = 0$, $V(x, t) > 0$ for $x \neq x^*$; (2) 沿轨迹非增: $\dot{V}(x, t) = \frac{\partial V}{\partial t} + \nabla_x V^\top f(x, t) \leq 0$. 若 $\dot{V} \leq 0$, 平衡点 $x^*$ 稳定; 若 $\dot{V} < 0$ (严格负), $x^*$ 渐近稳定.

**定义 2.3 (收缩性, Lohmiller-Slotine 1998)**. 系统 $\dot{x} = f(x, t)$ 在区域 $\mathcal{C}$ 中收缩, 若存在度量 $M(x, t)$ 和收缩率 $\lambda > 0$, 使得广义 Jacobian $F = M (\partial f / \partial x) M^{-1} + \dot{M} M^{-1}$ 的对称部分满足 $\frac{1}{2}(F + F^\top) \leq -\lambda I$.

### 2.4 GT 锚定 Lyapunov 函数

**定义 2.4 (GT 锚定 Lyapunov 函数)**. 对 RF 采样过程, 定义:

$$
V(x) = \frac{1}{2} \|x - x_0\|^2
$$

**性质 2.2**. $V(x)$ 满足: (a) $V(x_0) = 0$, $V(x) > 0$ for $x \neq x_0$ (正定); (b) 沿 RF 前向轨迹 $x^*(t)$, $V$ 从 $0$ 单调增至 $\frac{1}{2}\|x_1 - x_0\|^2$; (c) 沿 RF 逆向 (采样) 轨迹, $V$ 从 $\frac{1}{2}\|x_1 - x_0\|^2$ 降至 $0$ (理想情况).

### 2.5 Lyapunov 下降条件与方向余弦

**引理 2.1 (Lyapunov 下降条件)**. 以 $s = 1 - t$ 参数化采样过程 ($s: 0 \to 1$, $t: 1 \to 0$), $\frac{dx}{ds} = -v_\theta(x, 1-s)$. $V$ 沿采样轨迹的变化率:

$$
\frac{dV}{ds} = \nabla_x V^\top \frac{dx}{ds} = (x - x_0)^\top \cdot (-v_\theta) = -(x - x_0)^\top v_\theta
$$

$V$ 下降 (收敛至 $x_0$) 的充要条件为 $\frac{dV}{ds} \leq 0$, 即 $(x - x_0)^\top v_\theta \geq 0$.

**定理 2.1 (方向余弦与 Lyapunov 稳定性)**. 定义方向余弦 $\rho(x, t) = \cos(v_\theta, x - x_0) = \frac{(x - x_0)^\top v_\theta}{\|x - x_0\| \cdot \|v_\theta\|}$. Lyapunov 下降条件等价于 $\rho \geq 0$. 进一步, 若 $\rho \geq \delta > 0$ 沿整条采样轨迹, 则:

$$
\frac{dV}{ds} \leq -\delta \cdot \|x - x_0\| \cdot \|v_\theta\| = -\delta \sqrt{2V} \cdot \|v_\theta\| \leq 0
$$

$V$ 严格递减, 采样轨迹渐近稳定收敛至 $x_0$.

**注 2.2 (理想 RF 满足 $\rho = 1$)**. 理想 RF ($\hat{x}_0 = x_0$) 下, $v_\theta = (x - x_0)/t$, $\rho = \cos((x-x_0)/t, x-x_0) = 1$ (同向). LVD-RF 推动 $\rho \to 1$.

### 2.6 方向余弦正则化损失 (默认 sin² 形式)

**定义 2.5 (LVD-RF 正则项, 默认 sin²)**. 在训练时, 对每个 batch 中的 proposal, 采样 $t \sim \text{Uniform}(0, 1)$, 计算 $x_t = (1-t) x_0 + t x_1$, 前向得 $\hat{x}_0 = f_\theta(x_t, t)$. LVD-RF 正则项 (默认 sin² 形式以缓解梯度消失):

$$
\mathcal{L}_{\text{LVD}}(\theta) = \mathbb{E}\left[\sin^2(\alpha)\right] = \mathbb{E}\left[1 - \cos^2(\alpha)\right] = \mathbb{E}\left[1 - \left(\frac{(x_t - \hat{x}_0)^\top (x_t - x_0)}{(\|x_t - \hat{x}_0\| + \epsilon)(\|x_t - x_0\| + \epsilon)}\right)^2\right]
$$

其中 $\alpha$ 为 $v_\theta$ 与 $(x_t - x_0)$ 的夹角, $\epsilon = 10^{-6}$.

**梯度对比 (R1 K3 修正)**:
- $1 - \cos(\alpha)$: $\frac{d\mathcal{L}}{d\alpha} = \sin(\alpha) \approx \alpha$ (线性消失, $\cos_{\text{sim}} = 0.99$ 时梯度 $\approx 14\%$)
- $\sin^2(\alpha) = 1 - \cos^2(\alpha)$: $\frac{d\mathcal{L}}{d\alpha} = 2 \sin(\alpha) \cos(\alpha) = \sin(2\alpha) \approx 2\alpha$ (2× 强梯度, $\cos_{\text{sim}} = 0.99$ 时梯度 $\approx 28\%$)

**注 2.3 (sin² 仍线性消失, 自适应切换 sqrt)**. sin² 形式仍是线性消失 (改善 2× 但非根本解决). 当 $\cos_{\text{sim}} > 0.99$ 持续 1000 iter 时, 自动切换至 $\sqrt{1 - \cos + \epsilon}$ 形式 (非零梯度 $0.5/\sqrt{\epsilon}$). 详见 §7.2.

**定义 2.6 (训练目标)**.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{bbox}} + \mathcal{L}_{\text{giou}} + \lambda_{\text{LVD}} \cdot \mathcal{L}_{\text{LVD}}(\theta)
$$

$\lambda_{\text{LVD}} = 0.1$ (默认, 经验初值, 需消融).

### 2.7 多平衡点 Lyapunov 分析的合法性 (R1 K2 修正新增)

**§2.7 多平衡点 Lyapunov 分析**.

经典 Lyapunov 稳定性 (Lyapunov 1892, Khalil §4) 分析单一平衡点. LVD-RF 的设定是**多平衡点系统**: $K \approx 46$ 个 proposals 各自收敛至其匹配 GT $x_0^{(k)}$.

**合法性论证**:
1. **条件独立性**: 每个 proposal $k$ 的速度场 $v_\theta^{(k)}(x, t) = (x - \hat{x}_0^{(k)}) / t$ 在给定 RoI 特征后仅依赖其自身的 $\hat{x}_0^{(k)}$ 预测. (R2 N2 修正: 严格来说是条件独立性 — backbone (ResNet-50) 和 RoI extractor 共享, 不同 proposals 在特征提取层间接耦合, 但在 head 层条件独立. 重叠染色体的 RoI 特征干扰是二阶效应, Phase 1 实测验证.)
2. **隔离性**: 不同 GT $x_0^{(k)} \neq x_0^{(j)}$ ($k \neq j$) 在 cxcywh 空间中相互隔离 (不同染色体的 bbox 不重合).
3. **per-proposal 收敛**: 对每个 $k$, $V^{(k)}(x) = \frac{1}{2}\|x - x_0^{(k)}\|^2$ 是独立 Lyapunov 函数, 沿 proposal $k$ 的轨迹 $V^{(k)}$ 下降 (定理 2.1).

**与 Khalil §4.6 的对应**: Khalil §4.6 讨论多平衡点系统, 要求平衡点相互隔离且每个有独立 Lyapunov 函数. LVD-RF 满足此条件 (上述 1-3).

**局限**: (a) **匹配条件**: 稳定性仅在 GT-proposal 匹配正确时成立; (b) **重叠染色体干扰**: 重叠染色体的 proposals 在特征空间可能相互干扰 (RoI 特征重叠), 影响 $\hat{x}_0^{(k)}$ 预测的独立性. 此为二阶效应, Phase 1 实测验证.

### 2.8 定理与证明

**定理 2.2 (LVD-RF 降低 Lyapunov 函数上界)**. 若 $\mathcal{L}_{\text{LVD}} \leq \epsilon_{\text{LVD}}$, 则训练分布上 $\mathbb{E}[\rho] \geq 1 - \epsilon_{\text{LVD}}$ (因 $\sin^2 \alpha \leq \epsilon$ 蕴含 $|\sin \alpha| \leq \sqrt{\epsilon}$, 即 $\rho = \cos \alpha \geq 1 - \epsilon$ 的一阶近似). 由定理 2.1, 采样轨迹满足 $\frac{dV}{ds} \leq -(1 - \epsilon_{\text{LVD}}) \sqrt{2V} \|v_\theta\|$, Lyapunov 下降率至少为 $(1 - \epsilon_{\text{LVD}}) \|v_\theta\|$.

**定理 2.3' (与 $\eta_{\text{str}}$ 的关系, 条件性界, R1 K1 修正)**. 设 $\rho = \cos(v_\theta, x_t - x_0) \geq 1 - \epsilon$, 且假设:
- (A1) 速度场幅度有界: $\|v_\theta(x_t, t)\| \leq V_{\max}$ for all $x_t, t$
- (A2) 幅度变化 Lipschitz: $|\Delta \|v_\theta\|| \leq L_v^{\text{mag}} \cdot |\Delta t|$
- (A3) $\hat{x}_0$ 非退化: $\|\hat{x}_0\| \geq X_{\min} > 0$

则 $\hat{x}_0$ 的时间变化率有界:

$$
\|\hat{x}_0(x_{t_a}, t_a) - \hat{x}_0(x_{t_b}, t_b)\| \leq t_{\max} \cdot \left(L_v^{\text{mag}} \cdot |t_a - t_b| + V_{\max} \cdot \sqrt{2\epsilon}\right) + O(\epsilon)
$$

其中 $t_{\max} = \max(t_a, t_b)$. 因此

$$
\eta_{\text{str}} = \frac{\|D_1\|}{\|\hat{x}_0\|} \leq \frac{t_{\max}}{X_{\min}} \cdot \left(L_v^{\text{mag}} + V_{\max} \cdot \sqrt{2\epsilon} / |t_a - t_b|\right)
$$

**证明**. 由 $v_\theta = \|v_\theta\| \hat{u}_\theta$ ($\hat{u}_\theta$ 为单位方向), $\hat{x}_0 = x_t - t v_\theta$, 对 $t$ 求差分:

$$
\Delta \hat{x}_0 = \Delta x_t - \Delta(t v_\theta) = \Delta x_t - (\Delta t) v_\theta - t \Delta v_\theta
$$

其中 $\Delta v_\theta = (\Delta \|v_\theta\|) \hat{u}_\theta + \|v_\theta\| \Delta \hat{u}_\theta$. 由 LVD-RF 约束 $\|\Delta \hat{u}_\theta\| \leq \sin \alpha \leq \sqrt{2\epsilon}$:

$$
\|\Delta v_\theta\| \leq |\Delta \|v_\theta\|| + \|v_\theta\| \sqrt{2\epsilon} \leq L_v^{\text{mag}} |\Delta t| + V_{\max} \sqrt{2\epsilon}
$$

故 $\|\Delta \hat{x}_0\| \leq t_{\max} (L_v^{\text{mag}} |\Delta t| + V_{\max} \sqrt{2\epsilon})$. 除以 $\|\hat{x}_0\| \geq X_{\min}$ 得 $\eta_{\text{str}}$ 界. $\square$

**注 2.3' (LVD-RF 的局限, R1 K1 修正)**. LVD-RF 仅约束方向 $\hat{u}_\theta$, 不约束幅度 $\|v_\theta\|$. 定理 2.3' 表明:
- 若训练后 $\|v_\theta\|$ 在 $t$ 上变化平缓 ($L_v^{\text{mag}}$ 小), LVD-RF 有效降低 $\eta_{\text{str}}$
- 若 $\|v_\theta\|$ 在 $t$ 上剧烈变化, LVD-RF 的方向约束收益被幅度变化抵消
- **建议**: Phase 1 实测 $\|v_\theta\|(t)$ 的变化曲线, 验证 $L_v^{\text{mag}}$ 是否足够小

**常数 $C$ 的具体形式**:

$$
C = \frac{t_{\max}}{X_{\min}} \cdot V_{\max}
$$

其中 $t_{\max} \approx 1$, $X_{\min} \approx \|\hat{x}_0\|_{\text{typical}}$, $V_{\max} \approx \|x_1 - x_0\|_{\max} / t_{\min}$ ($t_{\min} = 0.25$ for 4-step DPM-Solver++).

**推论 2.1' (条件性结论)**. 在假设 (A1)-(A3) 下, LVD-RF 将 $\eta_{\text{str}}$ 从 $O(1)$ 降至 $O(\sqrt{\epsilon_{\text{LVD}}})$. **若 (A1) 或 (A2) 不成立, 收益不确定**. DPM-Solver++ 截断误差 $\propto \eta_{\text{str}}$ 同步降低 (条件性).

**定理 2.4 (方向 vs 幅度的根本区别)**. LVD-RF 仅约束方向余弦 $\rho$, 不约束 $\|v_\theta\|$. 设 $v_\theta = \|v_\theta\| \hat{u}_\theta$, 理想方向 $\hat{u}^* = (x_1 - x_0)/\|x_1 - x_0\|$.

(a) **LVD-RF 梯度**: $\nabla_\theta \mathcal{L}_{\text{LVD}} \propto \nabla_\theta (1 - \hat{u}_\theta^\top \hat{u}^*)$, 仅作用于方向 $\hat{u}_\theta$, 对幅度 $\|v_\theta\|$ 无梯度 (尺度不变).

(b) **TFR/VCR 梯度**: $\nabla_\theta \|\hat{x}_0(t) - \bar{x}_0\|^2 \propto \nabla_\theta (\hat{x}_0 - \bar{x}_0)$, 同时作用于方向和幅度.

**区别**: LVD-RF 允许 $\|v_\theta\|$ 自由变化 (只要求方向对), TFR/VCR 同时约束方向和幅度.

---

## 3. 与当前架构的关系

### 3.1 KaryoFlow 训练-推理流程回顾

**训练** ([head.py:598 `loss()`](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py)):
1. 采样 $t \sim \text{Uniform}(0, 1)$, 前向加噪 $x_t = (1-t) x_0 + t x_1$;
2. 网络前向 $\hat{x}_0 = f_\theta(x_t, t)$ (6 级 cascade head);
3. 计算检测损失 $\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{bbox}} + \mathcal{L}_{\text{giou}}$ (criterion.py);
4. 反向传播, 更新 $\theta$.

**推理** ([head.py:856 `predict()`](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py), [rectified_flow.py:151 `step()`](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py)): DPM-Solver++ 4 步 $t_0=1 \to t_4=0$, 每步 $x_{n+1} = \text{linear} + \phi_1 \cdot D_1$, Top-K 剪枝 $K=300$, NFE $= 4 \times 6 = 24$.

### 3.2 LVD-RF 嵌入点 (零额外前向)

LVD-RF 的关键优势: 方向余弦 $\cos(x_t - \hat{x}_0, x_t - x_0)$ 所需的三个量 ($x_t, \hat{x}_0, x_0$) 在训练步中**均已存在**:

| 量 | 代码位置 | 空间 |
|----|----------|------|
| $x_t$ | `x_boxes` (`head.py:626`) | raw cxcywh |
| $x_0$ | `x_starts` (`head.py:626`) | raw cxcywh |
| $\hat{x}_0$ | `all_pred_bboxes` (`head.py:645`) | xyxy 像素 (需转换) |

**空间一致性处理**: `all_pred_bboxes` 在 xyxy 像素空间, 需转换至 raw cxcywh (与 $x_t, x_0$ 同空间). 转换链:

```
all_pred_bboxes (xyxy 像素)
  → norm_pred_bboxes (xyxy [0,1])     [_normalize_pred_bboxes, head.py:662]
  → norm_cxcywh (cxcywh [0,1])        [bbox_xyxy_to_cxcywh]
  → raw_cxcywh (cxcywh [-s,s])        [(norm * 2 - 1) * snr_scale]
```

### 3.3 与 R1 $\eta_{\text{str}}$ 诊断的闭环

| 环节 | 代码位置 | 角色 |
|------|----------|------|
| 诊断 (R1) | `rectified_flow.py:141-192` | 推理时测量 $\eta_{\text{str}} = \|D_1\|/\|\hat{x}_0\|$ |
| 正则 (LVD-RF) | `head.py:loss()` (新增) | 训练时最小化方向余弦偏差 |
| 验证 (R1) | `rectified_flow.py:141-192` | 推理时复测 $\eta_{\text{str}}$, 验证下降 |

闭环: 诊断 (R1, $\eta_{\text{str}}$ 偏高) → 正则 (LVD-RF, 方向对齐) → 验证 (R1, $\eta_{\text{str}}$ 下降).

### 3.4 与 DPM-Solver++ 4-step 协同

由定理 2.3' (条件性), LVD-RF 使 $\eta_{\text{str}} = O(\sqrt{\epsilon_{\text{LVD}}})$ (在 (A1)(A2) 假设下), DPM-Solver++ 校正项 $\phi_1 \cdot D_1$ 的量级同步降低. $D_1 \to 0$ 时, DPM-Solver++ 退化为 1 阶 (更稳定, 减少 $t \to 0$ 振荡), 但 2 阶校正仍保留以处理残余曲率.

### 3.5 与现有特性的兼容性

| 现有特性 | 兼容性 | 说明 |
|----------|--------|------|
| DPM-Solver++ (A4) | 完全兼容 | LVD-RF 仅改训练, 推理不变 |
| Top-K pruning (IO3) | 完全兼容 | 正则化在 pruning 前 |
| Stochastic Coupling | 完全兼容 | 正则化 per-proposal, 与 coupling 独立 |
| v_prediction (R3) | 可共存 | LVD-RF 正则化方向, v_prediction 改损失加权, 正交 |
| ReFlow coupling | 可共存 | LVD-RF 正则化方向, ReFlow 改训练 target, 正交 |
| cascade_detach | 完全兼容 | 末级正则, 不跨级 |

---

## 4. 与染色体检测任务特性的匹配论证

### 4.1 $d=4$ 低维: 计算效率与方向空间几何优势 (R1 K5 修正)

**任务特性**: 染色体 bbox 为 $d=4$ (cxcywh 归一化), 远低于图像生成的 $d \sim 10^5$.

**匹配论证 (修正后)**:

1. **计算效率 (核心优势)**: 方向余弦计算复杂度 $O(N \cdot d) = O(N \cdot 4)$, $N \approx 500$ proposals, 可忽略. 相比之下, 图像生成中方向正则化 $O(N \cdot 10^5)$ 开销显著.

2. **低维方向空间几何**: $d=4$ 时, 单位球面 $S^3$ 是 3 维紧致流形, 方向空间有效自由度为 3 (比幅度自由度 4 少 1), 正则化更聚焦. 方向偏差 $\alpha \in [0, \pi]$ 在 $S^3$ 上有清晰的几何意义 (两向量的夹角).

3. **零额外前向 (与 TFR/VCR 对比)**: LVD-RF 复用训练步已有量 $(x_t, \hat{x}_0, x_0)$, 无需额外前向传播. 这在 $d=4$ 的低维检测场景下使训练开销接近零 ($< 2\%$).

**对比图像生成**: 图像生成中方向正则化 (如 VeCoR, Hong et al. 2025) 需要额外的负样本前向, 开销显著; LVD-RF 在 $d=4$ 下零额外前向, 是检测任务相对生成任务的独特优势.

**(R1 K5 修正说明)**: 删除原 §4.1 中 "高维随机方向正交, 方向正则梯度信号弱" 的错误论断. 余弦约束的梯度在任何维度都有定义, $d=4$ 的真正优势是计算效率 + 低维几何结构, 而非梯度信号强度.

### 4.2 $K \approx 46$ 密集排列: per-proposal 方向独立

**任务特性**: 染色体图像中 $K \approx 46$ 条染色体密集排列, 高度重叠.

**匹配论证**: LVD-RF 对每个 proposal 独立计算方向余弦 $\rho^{(k)} = \cos(v_\theta^{(k)}, x_t^{(k)} - x_0^{(k)})$ ($k = 1, \ldots, K$). 不同 proposal 可有不同方向偏差, 正则化不强制所有 proposal 同方向, 仅要求各自对齐其 GT 方向. 这匹配染色体检测的多目标独立性 (条件独立性, §2.7).

### 4.3 24 类不平衡: 方向正则的类无关性

**任务特性**: 24 类染色体中, Y 染色体仅 1803 样本 (最少), 不平衡比 $\sim 2.5\times$.

**匹配论证**: LVD-RF 不涉及类别信息, 对稀有类 (Y $\sim$ 1803) 和多数类 ($\sim$ 7000) 施加相同方向正则, 不加剧不平衡. 方向余弦是纯几何量 (与类别无关), 避免了已证伪的 Class-Balanced Sampling 和 SeesawLoss 的频率敏感问题.

### 4.4 小数据集 ($n \sim 2274$): Lyapunov 先验减少自由度 (R1 S1 修正)

**任务特性**: Dataset 2 仅 2274 张训练图, 远低于 COCO 的 $\sim 8 \times 10^4$.

**匹配论证 (修正后)**: 由 Rademacher 复杂度理论, 函数类的泛化间隙 $\leq 2 \mathcal{R}_n(\mathcal{F})$. LVD-RF 将函数类从 "任意方向的速度场" 限制为 "方向对齐 GT 的速度场":

$$
\mathcal{F}_{\text{LVD}} = \{v_\theta : \mathbb{E}[\cos(v_\theta, x_t - x_0)] \geq 1 - \epsilon\}
$$

**启发式论断 4.1 (Rademacher 复杂度收紧, 降级, R1 S1 修正)**. 对方向受限的函数类 $\mathcal{F}_{\text{LVD}}$, 其 Rademacher 复杂度满足 $\mathcal{R}_n(\mathcal{F}_{\text{LVD}}) \leq \sqrt{1 - (1-\epsilon)^2} \cdot \mathcal{R}_n(\mathcal{F}_{\text{full}}) \approx \sqrt{2\epsilon} \cdot \mathcal{R}_n(\mathcal{F}_{\text{full}})$ (小 $\epsilon$ 近似). 在 $n = 2274$, $\epsilon = 0.1$ 下, 复杂度收紧约 $\sqrt{0.2} \approx 0.45$ 倍.

**注 (R1 S1 修正)**: 本论断为**启发式估计**, 严格证明需要: (a) 半锥在 $d=4$ 中的覆盖数具体形式; (b) 期望约束到逐点约束的转换 (Dudley 链方法); (c) 函数类 $\mathcal{F}_{\text{LVD}}$ 的 Rademacher 复杂度上界. 留作未来工作. 不再作为核心论据, 仅作为小数据集优势的直觉支持.

### 4.5 RF 直线轨迹: Lyapunov 几何意义

**任务特性**: KaryoFlow 使用 RF (直线轨迹理想), $x_t = (1-t) x_0 + t x_1$.

**匹配论证**: LVD-RF 最小化方向偏差, 等价于推动 $v_\theta$ 平行于 $(x_1 - x_0)$, 即 RF 的理想恒定速度. 由定理 2.3', 这间接使 $\hat{x}_0 \to x_0$ (常数), $D_1 \to 0$, $\eta_{\text{str}} \to 0$, RF 轨迹趋近直线 (在 (A1)(A2) 假设下).

### 4.6 DPM-Solver++ 4-step: 截断误差降低

**任务特性**: 推理使用 DPM-Solver++ 4 步, 截断误差 $\propto \eta_{\text{str}}$.

**匹配论证**: 由定理 2.3' (条件性), LVD-RF 使 $\eta_{\text{str}} = O(\sqrt{\epsilon_{\text{LVD}}})$, DPM-Solver++ 校正项 $\phi_1 \cdot D_1$ 量级降低, 4 步采样的截断误差同步降低. 在 4 步 (少步) 场景, 截断误差对最终精度的影响被放大, LVD-RF 的收益更显著.

---

## 5. 与已证伪方向的严格区分

### 5.1 与 TFR 的区分

| 维度 | TFR (V2) | LVD-RF |
|------|----------|--------|
| 正则目标 | $\|\hat{x}_0(t) - \mathbb{E}[\hat{x}_0]\|^2$ (方差, 幅度) | $1 - \cos(v_\theta, x_t - x_0)$ (方向余弦) |
| 理论根 | 变分正则化 + Poincaré 不等式 | Lyapunov 稳定性理论 + 收缩理论 |
| 作用空间 | 幅度空间 (L2 范数) | 方向空间 (角度/余弦) |
| 尺度依赖 | 是 (L2 范数依赖幅度) | 否 (余弦尺度不变) |
| x0 漂移 | 禁止 (锁定均值) | 允许 (仅约束方向) |
| 额外前向 | 需要 (多时间步采样) | 不需要 (单步已有量) |
| 训练开销 | ~25% (subset 双前向) | <2% (仅余弦计算) |
| 过正则风险 | 中 (禁止合理幅度变化) | 低 (仅约束方向) |

### 5.2 与 VCR 的区分

| 维度 | VCR | LVD-RF |
|------|-----|--------|
| 正则目标 | $\|v_\theta(t) - v_\theta(t')\|^2$ (速度一致性) | $1 - \cos(v_\theta, x_t - x_0)$ (方向余弦) |
| 理论根 | Sobolev 训练 + 谱范数正则 | Lyapunov 稳定性理论 |
| 参考系 | 跨时间自身一致 (无外部锚点) | GT 锚定 ($x_t - x_0$) |
| 幅度约束 | 是 (L2 范数) | 否 (余弦尺度不变) |
| 额外前向 | 需要 (双时间步) | 不需要 |
| $t \to 0$ 稳定性 | 差 (1/t 缩放放大噪声) | 好 (余弦抵消 1/t) |

### 5.3-5.11 其他证伪方向区分

(与 ScaleConditionedRF, Cascade Head Count, CFM, Class-Balanced Sampling, h_velocity_loss, ReFlow, Inference-Time Optimization, FBM variants 的区分均清晰, 详见原 V3 设计文档 §5.3-5.11, 此处从略.)

### 5.12 与 scale_aware_loss 的区分 (R1 S2 新增)

| 维度 | scale_aware_loss (证伪) | LVD-RF |
|------|--------------------------|--------|
| 机制 | 尺度加权检测损失 ($\lambda(s) \cdot \mathcal{L}_{\text{det}}$) | 几何方向约束 ($1 - \cos$) |
| 尺度依赖 | 是 (依赖 $s = \sqrt{w \cdot h}$) | 否 (余弦尺度不变) |
| 训练-推理一致性 | 不一致 (训练用 GT 尺度, 推理用 $\hat{x}_0$ 尺度) | 一致 (训练推理均用 $\hat{x}_0$) |
| 失败原因 | 训练-推理尺度不一致导致性能下降 0.005 | LVD-RF 不引入尺度, 无此风险 |

**核心区分**: scale_aware_loss 引入**尺度先验** ($s = \sqrt{w \cdot h}$), 训练时从 GT 计算, 推理时从 $\hat{x}_0$ 计算, 二者不一致导致失败. LVD-RF 是**纯几何方向约束** (余弦, 尺度不变), 训练推理均在同一空间 (raw cxcywh) 计算, 无训推不一致风险.

---

## 6. 文献依据

### 6.1 核心理论文献

1. **Lyapunov, M. A. (1892).** *The General Problem of the Stability of Motion*. Taylor & Francis, 1992 (英译). — Lyapunov 稳定性理论的奠基性工作, 定义 2.2 的原始来源.

2. **Lohmiller, W., & Slotine, J. J. E. (1998).** "On Contraction Analysis for Non-linear Systems." *Automatica*, 34(6), 683-696. — 收缩理论奠基, 定义 2.3 的来源.

3. **Liu, X., Gong, C., & Liu, Q. (2023).** "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR 2023*. — RF 理论, 定义 2.1 的来源.

4. **Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).** "DPM-Solver++: Fast Solver for Guided Sampling of Probability Diffusion Models." *arXiv:2211.01095*. — DPM-Solver++ 多步法.

### 6.2 方向正则化文献 (核心支撑, R1 K4 修正)

5. **Ke, L., Xu, H., Ning, X., et al. (2025).** "ProReflow: Progressive Reflow with Decomposed Velocity." In *Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)*, pp. 28029-28038. arXiv:2503.04824. — **CVPR 2025 同行评审发表** (非仅 preprint). 核心支撑: 实验证明 "velocity direction is more critical for generation quality than magnitude" (Figure 1(b)). 提出 aligned v-prediction 强调方向匹配. (R1 K4 修正: B 在 R1 中对 ProReflow 发表状态的判断有误, WebSearch 三方证实: OpenReview "Published: 2025, CVPR 2025"; IEEE Xplore "2025 IEEE/CVF CVPR"; 华东师大教师页面. A 的拒绝正确.)

6. **Hong, Z.-W., Li, J.-L., Li, L.-Z., et al. (2026).** "VeCoR: Velocity Contrastive Regularization for Flow Matching." *arXiv:2511.18942* (preprint, 2025-11 提交, 2026-03 修订). — 在 flow matching 中引入速度对比正则, ImageNet-1K 上 FID 降 22%-35%.

7. **Khan, T. (2026).** "Isokinetic Flow Matching for Pathwise Straightening of Generative Flows." *arXiv:2604.04491* (workshop preprint). — Iso-FM 惩罚物质导数 $Dv/Dt$ 实现轨迹直线化.

### 6.3 Lyapunov 稳定性在神经网络中的应用

8. **Yang, L., Dai, H., Shi, Z., et al. (2024).** "Lyapunov-stable Neural Control for State and Output Feedback." *ICML 2024*. arXiv:2404.07956. — 将 Lyapunov 稳定性条件嵌入神经网络控制器训练.

9. **Tang, D., Yang, N., Deng, Y., et al. (2024).** "Stability-Driven CNN Training with Lyapunov-Based Dynamic Learning Rate." *ADC 2024*. — 将 CNN 训练建模为动态控制系统.

10. **Shi, Z., Li, H., Hsieh, C.-J., & Zhang, H. (2024).** "Certified Training with Branch-and-Bound for Lyapunov-stable Neural Control." *arXiv:2411.18235*. — Lyapunov 稳定性的认证训练.

### 6.4 收缩理论在扩散/流模型中的应用 (preprint, 谨慎引用)

11. **Abyaneh, A., Morissette, C., Danesh, M. H., et al. (2026).** "Contractive Diffusion Policies." *arXiv:2601.01003* (preprint, 未同行评审). — 将收缩理论引入扩散策略.

12. **Baheri, A. (2026).** "Wasserstein Stability of Contracting Flows: Effective Rates, Euler Self-Correction, and Noise Tightening." *arXiv:2607.14291* (preprint, 未同行评审). — 收缩流的 Wasserstein 稳定性.

### 6.5 RF 轨迹直线化文献

13. **Roy, S., Bansal, V., Sarkar, P., & Rinaldo, A. (2025).** "2-Rectifications Are Enough for Straight Flows." *ICLR 2025*. arXiv:2410.14949. — RF 的 Wasserstein 收敛性和直线度理论分析.

### 6.6 LVD-RF 独创性

1. **首次将 Lyapunov 稳定性理论用于 RF 检测训练正则化**: 现有 Lyapunov NN 文献 (8-10) 聚焦控制系统, 现有 RF 正则文献 (5-7) 聚焦图像生成. LVD-RF 在 $d=4$ 检测场景下, 以 GT 锚定 Lyapunov 函数推导方向余弦正则, 填补空白.
2. **方向 vs 幅度的严格区分**: 现有 TFR/VCR/Iso-FM 均在幅度空间正则, LVD-RF 在方向空间正则 (尺度不变), 与 ProReflow (5) 的 "方向优于幅度" 实验结论呼应.
3. **零额外前向的方向正则**: 现有方向正则 (VeCoR 6) 需负样本前向, LVD-RF 仅用训练步已有量, 在 $d=4$ 下计算可忽略.

---

## 7. 代码集成方案

### 7.1 修改文件清单

| 文件 | 修改内容 | 改动量 |
|------|----------|--------|
| `ldmdet/core/head.py` | `__init__` 加 LVD 超参; `loss()` 加 LVD 损失; 新增 `_compute_lvd_loss()` | ~110 行 |
| `configs/karyoflow_lvd.py` | 新配置 | ~30 行 |
| `ldmdet/diffusion/rectified_flow.py` | 无改动 (诊断已有) | 0 |

### 7.2 核心代码 (`head.py`)

**`__init__` 追加**:

```python
# LVD-RF: Lyapunov Velocity Direction Regularization
self.use_lvd = cfg.get('use_lvd', False)
self.lvd_lambda = cfg.get('lvd_lambda', 0.1)
self.lvd_eps = cfg.get('lvd_eps', 1e-6)        # 数值稳定常数
self.lvd_t_threshold = cfg.get('lvd_t_threshold', 0.05)  # t 过小时跳过
self.lvd_space = cfg.get('lvd_space', 'raw_cxcywh')  # 计算空间
self.lvd_form = cfg.get('lvd_form', 'sin2')    # R1 K3 修正: 默认 sin² (梯度比 1-cos 强 2×)
self._cos_sim_high_count = 0                    # 自适应切换计数器
```

**`loss()` 追加** (在 `losses = self.criterion(...)` 之后):

```python
losses = self.criterion(outputs, targets, t=t, box_targets=box_targets)

# LVD-RF 正则化 (零额外前向, 复用 x_boxes/x_starts/all_pred_bboxes)
if self.use_lvd and self.training:
    lvd_loss = self._compute_lvd_loss(
        x_boxes=x_boxes,           # x_t, raw cxcywh [bs, N, 4]
        x_starts=x_starts,         # x_0 (GT), raw cxcywh [bs, N, 4]
        all_pred_bboxes=all_pred_bboxes,  # \hat{x}_0, xyxy 像素 [H, bs, N, 4]
        t=t,                        # [bs]
        img_metas=img_metas,
    )
    losses['loss_lvd'] = self.lvd_lambda * lvd_loss
```

**新增 `_compute_lvd_loss()` 方法**:

```python
def _compute_lvd_loss(self, x_boxes, x_starts, all_pred_bboxes, t, img_metas):
    """LVD-RF: Lyapunov Velocity Direction Regularization.

    L_LVD = E[sin²(α)] = E[1 - cos²(α)]   (默认 sin² 形式, R1 K3 修正)

    由定理 2.4, 方向余弦尺度不变, 1/t 在分子分母抵消, 可直接用
    (x_t - x_hat_0) 和 (x_t - x_0) 计算余弦, 无需显式除以 t。

    Args:
        x_boxes: x_t, raw cxcywh [bs, num_proposals, 4]
        x_starts: x_0 (GT), raw cxcywh [bs, num_proposals, 4]
        all_pred_bboxes: \hat{x}_0, xyxy 像素 [num_heads, bs, num_proposals, 4]
        t: [bs] 扩散时间
        img_metas: 图像元数据 (用于空间转换)
    Returns:
        lvd_loss: 标量
    """
    bs = len(img_metas)
    device = x_boxes.device

    # 取末级 cascade head 的预测 (与 TFR/VCR 一致, 仅正则末级)
    pred_xyxy_pixel = all_pred_bboxes[-1]  # [bs, N, 4] xyxy 像素

    # 转换 \hat{x}_0 到 raw cxcywh (与 x_t, x_0 同空间)
    # xyxy 像素 → 归一化 xyxy [0,1] → 归一化 cxcywh [0,1] → raw cxcywh [-s, s]
    scales = x_boxes.new_zeros(bs, 4)
    for i, meta in enumerate(img_metas):
        h, w = meta['img_shape'][:2]
        scales[i] = x_boxes.new_tensor([w, h, w, h])
    norm_xyxy = pred_xyxy_pixel / scales.unsqueeze(1).clamp(min=1.0)
    from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
    norm_cxcywh = bbox_xyxy_to_cxcywh(norm_xyxy)
    raw_cxcywh = (norm_cxcywh * 2 - 1) * self.snr_scale  # [-s, s]

    x_hat_0 = raw_cxcywh  # [bs, N, 4]

    # 方向 1: 预测残差方向 (x_t - x_hat_0)
    d_pred = x_boxes - x_hat_0  # [bs, N, 4]

    # 方向 2: GT 锚定方向 (x_t - x_0) = t * (x_1 - x_0)
    d_gt = x_boxes - x_starts  # [bs, N, 4]

    # 跳过 t 过小的样本 (||x_t - x_0|| = t * ||x_1 - x_0|| → 0, 余弦不稳定)
    t_mask = (t >= self.lvd_t_threshold).float()  # [bs]
    t_mask = t_mask.view(bs, 1, 1)  # [bs, 1, 1] for broadcast

    # 余弦相似度: cos(d_pred, d_gt) = (d_pred . d_gt) / (||d_pred|| ||d_gt||)
    dot = (d_pred * d_gt).sum(dim=-1)  # [bs, N]
    norm_pred = d_pred.norm(dim=-1).clamp(min=self.lvd_eps)  # [bs, N]
    norm_gt = d_gt.norm(dim=-1).clamp(min=self.lvd_eps)  # [bs, N]
    cos_sim = dot / (norm_pred * norm_gt)  # [bs, N]

    # R1 S4 修正: xyxy 有效性检查 (x2 > x1, y2 > y1), 对无效框跳过 LVD 计算
    valid_mask = (
        (pred_xyxy_pixel[..., 2] > pred_xyxy_pixel[..., 0]) &
        (pred_xyxy_pixel[..., 3] > pred_xyxy_pixel[..., 1])
    )  # [bs, N]
    cos_sim = torch.where(valid_mask, cos_sim, torch.ones_like(cos_sim))

    # R1 K3 修正: 默认 sin² 形式 (梯度比 1-cos 强 2×)
    if self.lvd_form == 'sin2':
        lvd_per_elem = (1.0 - cos_sim ** 2) * t_mask.squeeze(-1)  # sin²(α)
    elif self.lvd_form == 'cos':
        lvd_per_elem = (1.0 - cos_sim) * t_mask.squeeze(-1)  # 1 - cos(α)
    elif self.lvd_form == 'sqrt':
        # 自适应: cos_sim 接近 1 时切换为 sqrt 形式 (非零梯度)
        # sqrt(1 - cos_sim + ε) 在 cos_sim→1 时梯度 → 0.5/sqrt(ε) (有界非零)
        lvd_per_elem = torch.sqrt(
            (1.0 - cos_sim).clamp(min=self.lvd_eps) * t_mask.squeeze(-1)
        )
    else:
        raise ValueError(f"Unknown lvd_form: {self.lvd_form}")

    n_valid = t_mask.sum().clamp(min=1.0) * x_boxes.shape[1]
    lvd_loss = lvd_per_elem.sum() / n_valid

    # 诊断: 训练时 cos_sim 分布 + 自适应切换逻辑 (R1 K3 修正)
    with torch.no_grad():
        probe.record_scalar('train/lvd_loss', lvd_loss.item())
        probe.record_scalar('train/cos_sim_mean', cos_sim.mean().item())
        probe.record_scalar('train/cos_sim_min', cos_sim.min().item())
        # 梯度健康度: 若 cos_sim_mean > 0.99 持续 1000 iter, 触发切换
        if cos_sim.mean().item() > 0.99:
            self._cos_sim_high_count += 1
            if self._cos_sim_high_count > 1000 and self.lvd_form == 'sin2':
                self.lvd_form = 'sqrt'  # 自动切换
                logger.info('LVD-RF: 切换至 sqrt 形式 (避免梯度消失)')
        else:
            self._cos_sim_high_count = 0
        # per-dim 方向偏差 (各维度对余弦的贡献)
        d_pred_normed = d_pred / norm_pred.unsqueeze(-1)
        d_gt_normed = d_gt / norm_gt.unsqueeze(-1)
        for i, name in enumerate(['cx', 'cy', 'w', 'h']):
            probe.record_scalar(
                f'train/lvd_dir_{name}',
                (d_pred_normed[..., i] * d_gt_normed[..., i]).mean().item()
            )
    return lvd_loss
```

### 7.3 配置文件 (`configs/karyoflow_lvd.py`)

```python
_base_ = './karyoflow_a4_io3.py'
model = dict(
    bbox_head=dict(
        use_lvd=True,
        lvd_lambda=0.1,
        lvd_eps=1e-6,
        lvd_t_threshold=0.05,
        lvd_space='raw_cxcywh',
        lvd_form='sin2',  # R1 K3 修正: 默认 sin², 可选 'cos' / 'sin2' / 'sqrt'
    )
)
```

### 7.4 复杂度分析

| 维度 | 评估 |
|------|------|
| 代码改动 | ~140 行 (head.py ~110 + config ~30) |
| 额外前向传播 | 0 (复用训练步已有量) |
| 额外反传链 | 1 条 (通过 `_compute_lvd_loss` 反传至 $\hat{x}_0$) |
| 训练时间开销 | < 2% (方向余弦 $O(N \cdot d) = O(500 \cdot 4)$, 可忽略) |
| 推理开销 | 0 (仅训练时) |
| 显存开销 | ~2 MB (中间张量 $[bs, N, 4]$) |
| 向后兼容 | 是 (`use_lvd=False` 完全回退) |

---

## 8. 预期收益分析

### 8.1 理论收益链条

从第一性原理到 mAP 的机制链 (条件性):

1. **Lyapunov 稳定性** (定理 2.1): $\cos(v_\theta, x_t - x_0) \geq \delta \Rightarrow$ 采样轨迹收敛至 $x_0$;
2. **方向对齐** (定理 2.4): 仅约束方向, 不约束幅度, 避免过度正则;
3. **$\eta_{\text{str}}$ 降低** (定理 2.3', 条件性): 在 (A1)(A2) 假设下, 方向对齐 $\Rightarrow \hat{x}_0 \approx x_0$ (常数) $\Rightarrow D_1 \approx 0 \Rightarrow \eta_{\text{str}} = O(\sqrt{\epsilon})$;
4. **DPM-Solver++ 截断误差降低**: 校正项 $\phi_1 \cdot D_1 \propto \eta_{\text{str}}$, 4 步采样精度提升;
5. **泛化界收紧** (启发式论断 4.1, 弱): 方向约束减少函数类复杂度, Rademacher 复杂度 $\times \sqrt{2\epsilon}$ (启发式, 非严格);
6. → **mAP 提升 + $\eta_{\text{str}}$ 下降 + per-dim 均衡**.

### 8.2 收益来源归因 (消融可拆解)

| 来源 | 预期增量 | 消融方式 |
|------|----------|----------|
| 方向对齐 → $\eta_{\text{str}}$ 降低 → 截断误差降 | +0.001~0.004 | 对比 $\eta_{\text{str}}$ 前后 |
| Lyapunov 稳定性 → 采样收敛保证 | +0.001~0.002 | 对比采样轨迹 $V(x)$ 下降率 |
| 泛化界收紧 → 小数据改善 (弱) | +0.000~0.002 | 对比 train/val gap |
| 方向 vs 幅度 (避免过正则) | +0.000~0.001 | 对比 TFR (幅度正则) |

---

## 9. 风险分析 (R1 K3 + R1 S 修正)

| 风险 | 等级 | 缓解 |
|------|------|------|
| **方向约束过强** (抑制合理方向变化) | 中 | $\lambda_{\text{LVD}}$ 从 0.01 起步, 监测 cos_sim 不超过 0.99; 消融 $\lambda \in \{0.01, 0.1, 0.5, 1.0\}$ |
| **$t \to 0$ 数值不稳定** ($\|x_t - x_0\| \to 0$) | 低 | `lvd_t_threshold=0.05` 跳过小 $t$; $\epsilon=10^{-6}$ 防除零 |
| **空间转换误差** (xyxy ↔ cxcywh) | 低 | 复用 `bbox_xyxy_to_cxcywh` 现有工具; xyxy 有效性检查 (R1 S4 新增) |
| **与 ReFlow 冲突** | 低 | 不同时启用; ReFlow 改 target, LVD-RF 加正则, 正交可叠加但需调 $\lambda$ |
| **方向正则梯度消失** (cos_sim → 1 时梯度 → 0) | **高** (R1 K3 上调) | 默认 sin² 形式 (2× 梯度); 监控 cos_sim; 若 > 0.99 持续 1000 iter, 自动切换 sqrt 形式 (非零梯度) |
| **理论界过松** (定理 2.3' 的 $O(\sqrt{\epsilon})$ 条件性) | **高** (R1 K1 上调) | 条件性结论, Phase 1 实测 $\|v_\theta\|(t)$ 验证 (A1)(A2); 若不成立, 承认收益不确定 |
| **per-dim 不均衡加剧** (cx/cy 方向偏差大, 正则梯度被 w/h 主导) | 中 | 监测 per-dim 方向贡献 (`train/lvd_dir_{cx,cy,w,h}`); 必要时加 per-dim 加权 (R1 S3 修正: 默认关闭, 阈值触发) |
| **与 cascade head 训练动态冲突** | 低 | 末级正则, 不跨级; cascade_detach 兼容 |
| **fallback** | — | `use_lvd=False` 完全回退至 baseline, 零回归风险 |

---

## 10. 实验计划 (含 Phase 0/1 诊断)

### Phase 0: 假设验证 (1 天, R2 N1 新增)

**目标**: 验证 (A1)(A2) 假设 (定理 2.3' 的前提条件).

**配置**: A4 baseline checkpoint, 推理时记录 $\|v_\theta\|(t)$ 曲线 (在 DPM-Solver++ 4 步采样过程中).

**监测指标**:
- $\|v_\theta\|(t)$ 在 $t \in \{0.25, 0.5, 0.75, 1.0\}$ 的值;
- $|\Delta \|v_\theta\|| / |\Delta t|$ 的估计 (Lipschitz 常数 $L_v^{\text{mag}}$);
- $\cos_{\text{sim}}$ 在 baseline 训练中的分布 (是否已接近 1).

**通过标准**: (a) $\|v_\theta\|$ 在 $t$ 上变化平缓 ($L_v^{\text{mag}}$ 有限); (b) $\cos_{\text{sim}}$ 未饱和 (mean < 0.95). 若 (A1)(A2) 不成立, 需考虑补充幅度正则 (与 TFR/VCR 叠加).

**SwanLab run**: `lvd_phase0_assumption`

### Phase 1: 诊断验证 (3 天)

**目标**: 验证 LVD-RF 能否降低 $\eta_{\text{str}}$ 且不掉点.

**配置**: A4 checkpoint + LVD-RF ($\lambda=0.1$, `lvd_form='sin2'`, `lvd_t_threshold=0.05`), 50 epochs (短训验证).

**监测指标**:
- `train/lvd_loss`, `train/cos_sim_mean`, `train/cos_sim_min` (LVD-RF 直接目标);
- `train/lvd_dir_{cx,cy,w,h}` (per-dim 方向贡献);
- `train/eta_str` (训练时 $\eta_{\text{str}}$ 估计, 复用 R1 诊断逻辑);
- `val/mAP` (每 10 epoch, 监测不掉点).

**通过标准**: (a) `cos_sim_mean` > 0.90; (b) `eta_str` 较 baseline 下降 > 20%; (c) `val/mAP` ≥ 0.863 (不掉点).

**SwanLab run**: `lvd_phase1_diag`

### Phase 2: 消融实验 (1 周)

**目标**: 确定 $\lambda_{\text{LVD}}$ 最优值, 验证方向 vs 幅度的区分.

**实验矩阵** (9 实验):

| 实验 | $\lambda_{\text{LVD}}$ | 正则类型 | 时间对 | 备注 |
|------|------------------------|----------|--------|------|
| LVD-1 | 0.01 | 方向 (sin²) | 单步 | 弱正则 |
| LVD-2 | 0.1 | 方向 (sin²) | 单步 | 默认 |
| LVD-3 | 0.5 | 方向 (sin²) | 单步 | 强正则 |
| LVD-4 | 1.0 | 方向 (sin²) | 单步 | 极强正则 |
| LVD-5 | 0.1 | cos (1-cos) | 单步 | 对照 (梯度消失) |
| LVD-6 | 0.1 | sqrt 形式 | 单步 | 对照 (非零梯度) |
| TFR-bl | 0.01 | 幅度 (var) | 随机 | TFR 对照 |
| VCR-bl | 0.01 | 幅度 (vel L2) | 双步 | VCR 对照 |
| baseline | 0 | — | — | A4 baseline |

**关键对比**:
- LVD-2 vs TFR-bl: 方向 vs 幅度 (核心对比);
- LVD-2 vs VCR-bl: 方向 vs 速度幅度;
- LVD-2 vs LVD-5: sin² vs cos (梯度消失问题);
- LVD-2 vs LVD-6: sin² vs sqrt (自动切换是否优于固定形式, R2 N3 新增).

**配置**: 100 epochs, 3 seeds (仅最优配置).

**SwanLab runs**: `lvd_phase2_{lvd1..lvd6,tfr_bl,vcr_bl,baseline}`

### Phase 3: 3-seed 验证 (1 周)

**目标**: 最优配置 3-seed 验证, 确认统计显著性.

**配置**: Phase 2 最优 $\lambda^*$, 150 epochs, 3 seeds.

**预期**: mAP $0.865 \sim 0.870$, std $\leq 0.003$.

**统计检验**: Wilcoxon signed-rank test (per-image mAP, LVD-RF vs baseline, $\alpha = 0.05$).

**SwanLab runs**: `lvd_phase3_seed{0,1,2}`

---

## 11. R2 评审剩余问题 (供后续 R3 参考)

1. **(A1)(A2) 假设验证 (R2 N1, 中等)**: Phase 0/1 必须实测 $\|v_\theta\|(t)$ 曲线, 验证幅度有界性和 Lipschitz 性. 若不成立, 需补充幅度正则 (与 TFR/VCR 叠加) 或承认定理 2.3' 收益不确定.
2. **ProReflow 跨域迁移性 (R2 N4, 轻微)**: ProReflow 的 "方向优于幅度" 实验结论基于图像生成 ($d \sim 10^5$), LVD-RF 应用于目标检测 ($d=4$). Phase 2 的 LVD-2 (方向) vs TFR-bl (幅度) 对比是验证跨域迁移性的关键实验, 需严格 3-seed 验证.
3. **独立性假设严格化 (R2 N2, 轻微)**: §2.7 的 "独立性" 应修订为 "条件独立性 (给定共享参数和 RoI 特征)", 并在 Phase 1 实测重叠染色体 proposals 的 $\hat{x}_0$ 干扰.
4. **sin² vs sqrt vs 自动切换 (R2 N3, 轻微)**: Phase 2 需 3-way 对比 (sin² 全程 / sqrt 全程 / 自动切换), 验证自动切换是否优于固定形式.
5. **条件性保证的论文表述**: 论文中应明确 "LVD-RF 在 $\|v_\theta\|$ 幅度变化平缓时有效降低 $\eta_{\text{str}}$", 而非无条件声称 "$\eta_{\text{str}} = O(\sqrt{\epsilon})$".

---

## 12. 自评 (修订后)

### 12.1 评分 (10 分制, R2 修正后)

| 维度 | 分数 | 理由 |
|------|------|------|
| 理论严谨性 | 7.8 | 定理 2.3' 修正后严格 (含幅度变化项 + 条件假设); §2.7 多平衡点 Lyapunov 合法性论证合理; 定理 4.1 降级为启发式论断诚实. 扣分: 条件性保证的传递性, 独立性假设不完全成立 |
| 文献验证 | 8.0 | ProReflow CVPR 2025 发表确认; A 补充独立 Lyapunov 论证不依赖 ProReflow. 扣分: ProReflow 跨域迁移性未验证 |
| 与 FALSIFIED 区分 | 8.5 | §5.12 与 scale_aware_loss 严格区分; per-dim 不均衡缓解方案具体化 |
| 实现可行性 | 7.5 | 零额外前向优势保留; xyxy 有效性检查补充. 扣分: 自动切换 sqrt 引入复杂度, 新超参数 (阈值, patience) |
| 任务匹配 | 7.5 | §4.1 d=4 优势论证修正后更准确 (计算效率 + 低维几何); sin² 默认形式更稳健. 扣分: ProReflow 跨域迁移性不确定 |
| 预期收益 | 6.5 | 收益预期诚实下调; 条件性 $O(\sqrt{\epsilon})$ 界提供理论上限. 扣分: 条件性保证使收益不确定性增加 |
| 风险可控 | 7.0 | 梯度消失风险上调为 "高" 但有 sin² + sqrt fallback; 理论界过松风险上调为 "高" 但有 Phase 0/1 实测验证. 风险更坦诚, 但风险等级未降 |
| 论文叙事 | 7.5 | "per-proposal 条件 Lyapunov 稳定性" 叙事更准确; ProReflow CVPR 2025 引用增强可信度; 诊断-正则-验证闭环保留 |

**综合评分**: 7.5/10

### 12.2 核心优势保留

- **零额外前向**: 仍成立, 工程优势显著 (训练开销 < 2%)
- **per-proposal 条件 Lyapunov 稳定性**: 修正后仍成立 (§2.7 合法性论证)
- **尺度不变 (余弦)**: 仍成立
- **与 R1 $\eta_{\text{str}}$ 诊断闭环**: 仍成立
- **推理零开销**: 仍成立 (NFE=24)
- **可叠加 (与 TFR/VCR)**: 仍成立 (方向 vs 幅度正交)

### 12.3 核心弱点承认

- 定理 2.3' 的 $O(\sqrt{\epsilon})$ 界为**条件性**, 依赖 (A1)(A2) 假设不由 LVD-RF 保证
- per-proposal 独立性假设不完全成立 (共享 backbone), 应为条件独立性
- sin² 仍线性消失, 自动切换增加复杂度
- ProReflow 跨域迁移性 (图像生成 → 目标检测) 未验证
- 启发式论断 4.1 (Rademacher 收紧) 不严格, 已降级

### 12.4 投稿建议

1. **TMI 正文**: "基于 Lyapunov 稳定性的方向正则化" 作为训练正则化贡献, 与 R1 诊断闭环. 强调:
   - per-proposal 条件 Lyapunov 稳定性保证 (定理 2.1, 收敛性);
   - 方向优于幅度 (ProReflow CVPR 2025 实验支撑 + Lyapunov 独立推导, 定理 2.4);
   - 零额外前向 (工程优势, 表 7.4);
   - d=4 低维方向空间优势 (任务特性结合, 计算效率 + 低维几何).

2. **arXiv companion**: 定理详细证明 (定理 2.3' 的条件性 $O(\sqrt{\epsilon})$ 推导), Phase 2 完整消融表, per-dim 诊断详细图, Phase 0 $\|v_\theta\|(t)$ 曲线.

3. **创新点叙事**: "首次将 Lyapunov 稳定性理论用于 RF 检测训练正则化, 以 GT 锚定 Lyapunov 函数推导方向余弦正则, 在 $d=4$ 低维空间实现零额外前向的方向对齐, 通过定理 2.1 保证采样轨迹的 per-proposal 条件 Lyapunov 稳定收敛."

---

## 附录 A: 符号表

| 符号 | 含义 |
|------|------|
| $x_0$ | 真实 bbox (cxcywh, 归一化) |
| $x_1$ | 高斯噪声 |
| $x_t = (1-t) x_0 + t x_1$ | RF 插值样本 |
| $\hat{x}_0$ | 网络预测 $x_0$ |
| $v_\theta = (x_t - \hat{x}_0) / t$ | 速度场 |
| $v^* = x_1 - x_0$ | 理想恒定速度 |
| $V(x) = \frac{1}{2}\|x - x_0\|^2$ | GT 锚定 Lyapunov 函数 |
| $\rho = \cos(v_\theta, x_t - x_0)$ | 方向余弦 |
| $\alpha = \arccos(\rho)$ | 方向偏差角 |
| $\eta_{\text{str}} = \|D_1\| / \|\hat{x}_0\|$ | 直线度诊断 |
| $D_1$ | x0 预测一阶差商 |
| $\lambda_{\text{LVD}}$ | LVD-RF 正则化系数 |
| $K$ | proposals 数 ($\approx 46$) |
| $d$ | bbox 维度 ($=4$) |
| $H$ | cascade head 级数 ($=6$) |
| $N$ | solver 步数 ($=4$) |
| NFE | 前向评估次数 ($=24$) |

## 附录 B: 方向余弦与速度的等价性

由 $v_\theta = (x_t - \hat{x}_0)/t$ 和 $x_t - x_0 = t(x_1 - x_0)$:

$$
\cos(v_\theta, x_t - x_0) = \cos\!\left(\frac{x_t - \hat{x}_0}{t},\; t(x_1 - x_0)\right) = \cos(x_t - \hat{x}_0,\; x_1 - x_0) = \cos(x_t - \hat{x}_0,\; x_t - x_0)
$$

正标量 $1/t > 0$ 和 $t > 0$ 不改变向量方向. 因此 $\mathcal{L}_{\text{LVD}}$ 可直接用 $(x_t, \hat{x}_0, x_0)$ 计算, **无需显式除以 $t$**, 数值稳定 (避免 $t \to 0$ 时 $1/t$ 发散).

---

*FINAL 文档结束. LVD-RF 整合了 R1 评审 (K1-K5 + S1-S4) → A 响应 (接受 K1/K2/K3/K5/S1/S2/S4, 拒绝 K4) → R2 评审 (确认修订, R2 评分 7.5/10) 的全部修订, 可直接进入 Phase 0 假设验证.*
