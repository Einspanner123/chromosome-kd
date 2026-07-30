# FEASIBLE ISLR-RF: Input-Space Lipschitz Regularization for Rectified Flow (FINAL 整合设计文档)

> **方向类别**: 保守方向 (基于已有研究的可靠性改进)
> **方案代号**: ISLR-RF (Input-Space Lipschitz Regularization for Rectified Flow)
> **理论基础**: Lipschitz 连续性 (Lipschitz 1886) + Grönwall 引理 (1919) + Rademacher 复杂度界 (Sokolić et al. 2017) + 频率原理 (Rahaman et al. 2019) + 输入梯度正则化 (Drucker & Le Cun 1992; Sokolić et al. 2017; Finlay & Oberman 2019) + RF x0-prediction (Liu-Gong-Liu 2023) + DPM-Solver++ (Lu et al. 2022)
> **核心改动**: 在训练损失中新增 $\mathcal{L}_{\text{ISLR}} = \lambda_0 \cdot \mathbb{E}[\lambda(t) \cdot \|\nabla_{x_t} f_\theta\|_F^2]$ (R1 反馈 3 修正: 新增 $t$-dependent 加权 $\lambda(t) = \lambda_0 \cdot t^p$), 利用 $d=4$ 低维优势以 4 次 VJP 精确计算 $4 \times 4$ 雅可比, 强制子采样 $K_{\text{islr}}=16$ 控制开销. 仅改 criterion.py 与 head.py, 推理零改动, NFE=24 不变.
> **目标**: 在不改网络架构、不改 NFE ($S=4$)、不改 coupling/solver 的前提下, 把 Dataset 2 baseline mAP=0.863 提升至 0.864~0.868
> **预期收益 (R2 确认)**: mAP $+0.001 \sim +0.005$; mAP$_{75}$ $+0.002 \sim +0.006$; Y 染色体 AP $+0.000 \sim +0.010$ (待 Phase 1 验证稀有类雅可比更大的假设); $\eta_{\text{str}}$ 下降 $7 \sim 13\%$; 推理 seed std 下降 $10 \sim 20\%$

**文档状态**: FINAL (整合 R1 评审 → A 响应 → R2 评审全部修订, 可直接进入实现)
**撰写日期**: 2026-07-28
**R2 最终评分**: 7.4/10 (谨慎推荐, R1: 7.0, 修订后自评: 7.3)
**目标期刊**: IEEE TMI
**依赖文件**: [rectified_flow.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py), [head.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py), [criterion.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/criterion/criterion.py), [FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md), [EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

---

## 1. 摘要: 核心论断与预期收益

### 1.1 核心论断 (修订后)

KaryoFlow baseline mAP = 0.863, 受限于: (1) ODE 适定性风险 (速度场 Lipschitz 常数 $L_v \propto 1/t$ 在小 $t$ 爆炸); (2) Grönwall 误差指数传播 (推理 seed 方差大, 在 $t \in [0.25, 1]$ 范围内 Grönwall 指数 $e^{0.75(1+L_f)}$); (3) 小数据集 ($n=2274$) 泛化瓶颈.

TFR/VCR 约束时间轴 $\partial \hat{x}_0/\partial t$; SDR-RF 约束散度 $\text{tr}(J_f)^2$; CFR 约束旋度 $\|\text{antisym}(J_f)\|_F^2$. 然而**唯一直接界住 Lipschitz 常数** $L_f \leq \sup \|J_f\|_{\text{op}} \leq \sup \|J_f\|_F$ 的全 Frobenius 正则 $\|J_f\|_F^2$ 尚未被完整形式化.

**本方案 ISLR-RF**: 基于 Lipschitz (1886) + Grönwall (1919) + Sokolić et al. (2017) Rademacher 复杂度, 在训练损失中增加 $t$-dependent 加权的 $\mathcal{L}_{\text{ISLR}} = \lambda_0 \cdot \mathbb{E}[\lambda(t) \cdot \|J_f\|_F^2]$, 其中 $\lambda(t) = \lambda_0 \cdot t^p$ ($p=2$ 默认). $d=4$ 时 $J_f$ 通过 4 次 VJP 精确计算, 无需 Hutchinson 估计.

**关键论断 (修订后)**:

- **(C1) 全 Frobenius 范数直接界住 $L_f$ (定理 1.4)**: $\|J_f\|_F^2 \geq \|J_f\|_{\text{op}}^2$, SDR-RF/CFR 不能 (命题 1.5). ISLR-RF 是唯一直接界住 Lipschitz 常数的方案.
- **(C2) $L_f \downarrow \Rightarrow L_v \downarrow \Rightarrow$ ODE 适定性 (定理 2.1)**: $L_v \leq (1+L_f)/t$, ISLR 降低 $L_f$ 间接降低 $L_v$.
- **(C3) Grönwall 误差传播抑制 (定理 2.3', $t$ 范围修正, R1 反馈 5)**: 在 DPM-Solver++ 4 步采样有效 $t$ 范围 $[t_{\min}, 1] = [0.25, 1]$ 内, 累积 Grönwall 指数 $e^{0.75(1+L_f)}$. 若 ISLR 使 $L_f \to 0.5 L_f$ ($L_f \approx 1$), 改善约 31% (指数下降), 对应 seed std 下降约 15%.
- **(C4) Rademacher 复杂度收紧 (R1 反馈 1 修正, Sokolić et al. 2017)**: 对函数类 $\mathcal{F} = \{f_\theta : \mathbb{E}[\|J_f\|_F^2] \leq M\}$, 其经验 Rademacher 复杂度满足 $\hat{\mathcal{R}}_n(\mathcal{F}) \leq O(B_x \sqrt{M}/\sqrt{n})$. ISLR 通过降低 $M$ 直接收紧 Rademacher 复杂度界. **(原 Bartlett 界论证被过度引申, 已切换到 Sokolić Rademacher 界, 理论链条严格; Rademacher 界的收紧为线性 $\sqrt{M}$ 关系, 弱于原 Bartlett 的平方关系.)**
- **(C5) 理想 RF 下 $J_f$ 行为修正 (命题 1.8', R1 反馈 3 修正)**: (a) 若网络 "已知" GT ($\hat{x}_0 \equiv x_0$), $J_f = 0$; (b) 若网络从 $x_t$ 推断 $x_0$, 理想 $J_f$ 依赖 $t$: $t \approx 0$ 时 $J_f \approx I$, $t \approx 1$ 时 $J_f \approx 0$, 中间 $t$ 非零. **ISLR 的目标调整为约束 $J_f$ 在 $t \approx 1$ (噪声端) 接近 0, 在 $t \approx 0$ (数据端) 允许非零**, 通过 $t$-dependent 加权 $\lambda(t) = \lambda_0 t^p$ 实现.

### 1.2 预期收益汇总 (R2 确认)

| 指标 | Baseline (Dataset 2, +DPM-Solver++) | ISLR-RF 预期 | 改善 | 备注 |
|------|-------------------------------|-----------|------|------|
| mAP | 0.863 | 0.864 ~ 0.868 | $+0.001 \sim +0.005$ | Rademacher 收紧 + ODE 适定性 (R1 反馈 1 下调) |
| mAP$_{75}$ | 0.974 | 0.976 ~ 0.980 | $+0.002 \sim +0.006$ | Grönwall 改善有限 (R1 反馈 5 下调) |
| Y 染色体 AP | 0.781 (3-seed) | 0.781 ~ 0.791 | $+0.000 \sim +0.010$ | 待 Phase 1 验证稀有类雅可比更大假设 (R1 反馈 4 下调) |
| $\eta_{\text{str}}$ | ~0.15 | ~0.13 ~ 0.14 | $-7 \sim 13\%$ | $J_f \downarrow \Rightarrow \eta_{\text{str}} \downarrow$ (R1 反馈 5 下调) |
| 推理 seed std | ~0.003 | ~0.0025 ~ 0.003 | $-10 \sim 20\%$ | 基于 $t_{\min}=0.25$ 的 Grönwall 分析 (R1 反馈 5 下调) |
| 推理 NFE | 24 | 24 | 0 | 推理零改动 |
| 训练开销 | baseline | $+80 \sim 200\%$ | — | 4 VJP + 二阶图, 强制 $K_{\text{islr}}=16$ 子采样 (R1 反馈 2 修正) |

**收益限制 (R2 新增)**: Rademacher 界线性收紧弱于 Bartlett 界平方收紧, 小数据集收益论点进一步削弱. 实际收益需 Phase 0/1 实验验证.

---

## 2. 第一性原理推导

### 2.1 基本定义

**定义 1.1 (Lipschitz 连续性)**. $f: \mathbb{R}^d \to \mathbb{R}^d$ 是 $L$-Lipschitz 的, 若 $\|f(x)-f(y)\| \leq L\|x-y\|$, $\forall x,y$. $L_f := \sup_{x \neq y} \frac{\|f(x)-f(y)\|}{\|x-y\|}$.

**定义 1.2 (输入雅可比)**. $J_f(x) := \nabla_x f_\theta(x,t) \in \mathbb{R}^{d \times d}$, $\|J_f\|_F := \sqrt{\sum_{i,j} (\partial f_i/\partial x_j)^2}$, $\|J_f\|_{\text{op}} := \sigma_{\max}(J_f)$.

### 2.2 核心引理与定理

**引理 1.4 (Frobenius 界住 Lipschitz)**. $L_f \leq \sup_x \|J_f(x)\|_{\text{op}} \leq \sup_x \|J_f(x)\|_F$.

*证明*. 由积分中值定理 $f(x)-f(y) = \int_0^1 J_f(y+s(x-y))(x-y)ds$, 取范数并用 $\|Jv\| \leq \|J\|_{\text{op}}\|v\| \leq \|J\|_F\|v\|$. $\square$

**命题 1.5 (SDR-RF/CFR 不能界住 $L_f$)**.
(1) $J_f = \begin{pmatrix}0&M\\0&0\end{pmatrix}$: $\text{tr}(J_f)=0$ 但 $\|J_f\|_{\text{op}}=M$ (SDR-RF 失效).
(2) $J_f = M \cdot I$: $\text{antisym}(J_f)=0$ 但 $\|J_f\|_{\text{op}}=M$ (CFR 失效).
(3) $\|J_f\|_F^2 = \sum_i \sigma_i^2 \geq \sigma_{\max}^2 = \|J_f\|_{\text{op}}^2$ (ISLR-RF 有效). $\square$

**引理 1.7 (RF 速度场雅可比)**. $v_\theta = (x_t - f_\theta)/t$, $J_v = (I - J_f)/t$, $L_v \leq (1+L_f)/t$.

**命题 1.8' (理想 RF 下的 $J_f$ 行为, R1 反馈 3 修正)**.
(a) 若网络 "已知" GT ($\hat{x}_0 \equiv x_0$, 与 $x_t$ 无关), 则 $J_f = 0$, $\mathcal{L}_{\text{ISLR}} = 0$. 但此情形仅在训练时 (有 GT 监督) 成立, 推理时不可达.
(b) 若网络从 $x_t$ 推断 $x_0$ ($\hat{x}_0 = g(x_t)$), 则理想 $J_f = \partial g / \partial x_t$ 依赖于 $t$:
   - $t \approx 0$ (低噪声): $x_t \approx x_0$, $g(x_t) \approx x_t$, $J_f \approx I$ (恒等映射, 而非 0!)
   - $t \approx 1$ (纯噪声): $x_t \approx \varepsilon$, $g$ 无法提取 $x_0$, $J_f \approx 0$
   - 中间 $t$: $J_f$ 非零, 用于去噪
(c) **ISLR 的目标调整为**: 约束 $J_f$ 在 $t \approx 1$ (噪声端) 接近 0, 在中间 $t$ 适度, 在 $t \approx 0$ 允许非零 (但接近 $I$ 以避免输出漂移).

### 2.3 ISLR-RF 正则化目标 (R1 反馈 3 修正: $t$-dependent 加权)

$$\mathcal{L}_{\text{ISLR}} = \lambda_0 \cdot \mathbb{E}_{x_t, t}\!\left[\lambda(t) \cdot \left\|\nabla_{x_t} f_\theta(x_t, t)\right\|_F^2\right] = \lambda_0 \cdot \mathbb{E}\!\left[\lambda(t) \cdot \sum_{i=1}^{4}\sum_{j=1}^{4}\left(\frac{\partial f_{\theta,i}}{\partial x_{t,j}}\right)^2\right]$$

其中 $\lambda(t) = t^p$ ($p \geq 1$, 默认 $p=2$), 设计为:
- $t \approx 1$ (纯噪声): $\lambda(t) = 1$ (强正则, $J_f \to 0$ 合理, 因无法从噪声推断 $x_0$)
- $t \approx 0$ (低噪声): $\lambda(t) = 0.1$ 量级 (弱正则, 允许 $J_f \approx I$)
- 中间 $t$: $\lambda(t)$ 平滑过渡

总损失: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{giou}} + \mathcal{L}_{\text{ISLR}}$.

### 2.4 Grönwall 误差传播 ($t$ 范围修正, R1 反馈 5)

**定理 2.3' (Grönwall 误差传播, $t$ 范围修正)**. 设 DPM-Solver++ 4 步采样的有效 $t$ 范围为 $[t_{\min}, 1]$, $t_{\min} = 0.25$. 则累积误差界为:

$$\|x(t_{\min}) - \tilde{x}(t_{\min})\| \leq \|x(1) - \tilde{x}(1)\| \cdot e^{(1+L_f)(1-t_{\min})} = \|x(1) - \tilde{x}(1)\| \cdot e^{0.75(1+L_f)}$$

ISLR 使 $L_f \to L_f'$, Grönwall 指数从 $e^{0.75(1+L_f)}$ 降至 $e^{0.75(1+L_f')}$. 若 $L_f' = 0.5 L_f$ ($L_f \approx 1$), 改善约 31% (指数下降), 对应 seed std 下降约 15%.

### 2.5 Rademacher 复杂度收紧 (R1 反馈 1 修正, Sokolić et al. 2017)

**命题 3.3' (Rademacher 复杂度收紧, 基于 Sokolić et al. 2017)**. 对函数类 $\mathcal{F} = \{f_\theta : \mathbb{E}[\|J_f\|_F^2] \leq M\}$, 其经验 Rademacher 复杂度满足

$$\hat{\mathcal{R}}_n(\mathcal{F}) \leq O\left(\frac{B_x \cdot \sqrt{M}}{\sqrt{n}}\right)$$

其中 $B_x$ 是输入半径, $M = \mathbb{E}[\|J_f\|_F^2]$ 是 ISLR 正则化目标.

ISLR 通过降低 $M$, 直接收紧 Rademacher 复杂度界. 在 $n=2274$ 的小数据集上, 收紧比例与 $M$ 的下降成线性关系 (Rademacher 界线性 $\sqrt{M}$ 依赖, 非 Bartlett 界的平方关系), 但理论链条严格.

**注 2.1 (Bartlett 界间接论证, 保留但削弱)**. Bartlett-Foster-Telgarsky (2017) 的谱裕度界 $\text{gen gap} \leq O(\prod_i \|W_i\|_2 \cdot R \cdot \sqrt{\text{depth}}/\sqrt{n})$ 针对权重矩阵谱范数乘积. ISLR 正则化输入雅可比 $\|J_f\|_F$, 通过链式法则间接约束 $\prod_i \|W_i\|_2$, 但不直接界住每一层. 严格的 Bartlett 界收紧需要逐层谱归一化 (LSCR-RF 方向的范畴). ISLR 与 LSCR-RF 互补.

---

## 3. 与当前架构的关系

KaryoFlow 的 `DiffusionDetHead` ([head.py:31](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/head.py)) 包含 6 级 cascade head, `q_sample` ([rectified_flow.py:36](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py)) 实现 $x_t = (1-t)x_0 + t\varepsilon$, `RFDPMSolverMultistep` ([rectified_flow.py:114](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py)) 4 步推理.

ISLR-RF 嵌入在 `head.py` 的 `loss()` 方法 (L598) 中, 在 `self.criterion(outputs, targets, t=t)` (L684) 之后附加 ISLR 正则项. **推理时无改动**: DPM-Solver++ 4 步、Top-K 剪枝、box_renewal、NMS 全部不变, NFE=24.

---

## 4. 与任务特性的匹配论证

### 4.1 $d=4$ 低维: 精确 VJP 计算甜区

$J_f \in \mathbb{R}^{4 \times 4}$, 通过 4 次 VJP (每次 1 次反传) 精确计算全部 16 个偏导数, $O(d^2)=O(16)$ 闭式 Frobenius 范数. 图像生成 ($d \sim 10^5$) 和 COCO 检测 ($d=320$) 不可行, 必须 Hutchinson 估计. **染色体检测 $d=4$ 是精确计算的唯一甜区**.

### 4.2 $K \approx 46$ 密集: 方差降低 (R2 N2 修正)

batch 内 $K \approx 46$ 个 proposal 的雅可比范数取均值, 正则信号方差降低. **R1 反馈 2 修正后强制子采样 $K_{\text{islr}}=16$**, 实际方差降低 $\sqrt{K_{\text{islr}}} = \sqrt{16} = 4\times$ (而非原 $\sqrt{46} \approx 6.8\times$). 4× 方差降低仍有效, 但任务匹配论点强度下降.

### 4.3 24 类不平衡: 稀有类雅可比更大 (R1 反馈 4 降级为假设)

稀有类 (Y 染色体, 1803 vs 7000) 有效训练样本少, 网络可能通过增大输入雅可比补偿样本不足 (**此为待验证假设, 非已结论**, R1 反馈 4 删除了原 WANT@ICML 不可验证引用). ISLR 均匀正则所有类, 但若稀有类雅可比基线更高, 绝对减小量更大. Phase 1 实验中将实测 per-class $\|J_f\|_F$ 验证此假设.

### 4.4 小数据集 $n=2274$: Rademacher 界收紧 (R1 反馈 1 修正)

原 Bartlett 界论证 ("$L_f$ 减小 50% 等效数据量增加 $4\times$") 已删除, 改为 Sokolić Rademacher 复杂度论证 (§2.5). Rademacher 界 $\hat{\mathcal{R}}_n \propto \sqrt{M}/\sqrt{n}$ 收紧, $M$ 下降 50% 时 Rademacher 界收紧 $\sqrt{0.5} \approx 0.71$ 倍, 泛化 gap 改善约 29% (而非 Bartlett 的 50%).

### 4.5 RF 直线轨迹: 与 TFR 正交分解 $\eta_{\text{str}}$

$\frac{d\hat{x}_0}{dt} = \underbrace{\partial_t \hat{x}_0}_{\text{TFR}} + \underbrace{J_f \cdot v_\theta}_{\text{ISLR-RF}}$, 两者正交分解 $\eta_{\text{str}}$, 可叠加.

---

## 5. 与已证伪方向的严格区分

| 已证伪方向 | 失败机制 | ISLR-RF 区分 |
|-----------|---------|------------|
| ScaleConditionedRF (mAP 0.741) | 改前向路径, 训推不一致 | 不改路径, 训推一致 |
| FBM variants | 改噪声分布 | 不改噪声 |
| CBS / Seesaw Loss | 类别重加权/重采样 | 输入雅可比正则, 类无关 |
| CFM | 改训练目标 | 仅附加正则项 |
| h_velocity_loss | velocity 分量加权, $1/t^2$ 梯度放大 | 输入雅可比全范数正则, **不直接涉及 $1/t$ 因子** (R1 反馈: 已补充) |
| ITO | 推理时优化 | 训练-only, 推理零开销 |
| scale_aware_loss | 推理尺度不可靠 | 不依赖推理尺度, 但同样面临 "正则化过强损害精度" 的风险 |

与已有保守设计: TFR (时间轴 $\partial\hat{x}_0/\partial t$) 正交; SDR-RF ($\text{tr}(J_f)^2$ 散度) 被 ISLR 包含; CFR ($\|\text{antisym}(J_f)\|_F^2$ 旋度) 被 ISLR 包含; LSCR-RF (权重谱范数, 参数空间) 互补. **ISLR-RF 是唯一直接界住 $L_f$ 的方案**.

---

## 6. 文献依据

### 6.1 核心理论文献

1. **Lipschitz (1886)** — Lipschitz 连续性原始定义
2. **Grönwall (1919)** — ODE 误差指数传播界
3. **Bartlett, Foster & Telgarsky (2017, NeurIPS)** — 谱裕度泛化界 (R1 反馈 1 修正: 仅作间接论证, 主论证切换到 Sokolić)
4. **Sokolić et al. (2017, IEEE TSP, arXiv:1705.09267)** — 输入梯度正则化 + Jacobian-based Rademacher 复杂度界 (R1 反馈 1 新增主论证)
5. **Rahaman et al. (2019, ICML)** — 频率原理 / 谱偏置
6. **Drucker & Le Cun (1992, IEEE TNN)** — 输入梯度正则化 (double backprop)
7. **Finlay & Oberman (2019, arXiv:1905.11468, ICLR 2020)** — 可扩展输入梯度正则化, 避免 double backprop (R1 反馈 7 新增, ISLR-RF 的工程先例)
8. **Varga et al. (2018, arXiv:1710.06968)** — 梯度正则化对判别模型的有效性验证 (R1 反馈 7 新增)
9. **Liu, Gong & Liu (2022, ICLR)** — Rectified Flow 直线轨迹
10. **Lu et al. (2022, NeurIPS)** — DPM-Solver++ 截断误差

### 6.2 扩散/流模型中的输入雅可比正则化先例 (R1 反馈 7 新增)

**为何 ISLR 在扩散检测中尚未被尝试**:
1. 图像生成 ($d \sim 10^5$): 精确 VJP 不可行, 必须 Hutchinson 估计, 方差大; Finlay & Oberman (2019) 为此设计避免二阶图的方案
2. COCO 检测 ($d=320$): 4 VJP × 320 = 1280 次反传, 开销过高
3. **染色体检测 $d=4$ 是精确 VJP 的唯一甜区**: 4 次 VJP, $O(d^2)=O(16)$ 闭式 Frobenius, 无需 Hutchinson, 无估计方差

ISLR-RF 是首次将输入雅可比精确正则化应用于扩散检测 (基于 $d=4$ 低维优势), 与 Finlay & Oberman (2019) 在 ImageNet 上的可扩展方案形成对比: 前者精确低维, 后者近似高维.

### 6.3 已删除文献

- ~~Yang et al. 2024, WANT@ICML~~ (R1 反馈 4: 无法验证, 已删除)

### 6.4 降级文献

- **Picard-Lindelöf (1890)** (R1 反馈 6: 降级为附录, 非核心理论基础). 价值在于保证 RF ODE 解存在唯一, 是任何求解器 (含 DPM-Solver++) 的理论基础, 但不直接指导正则化设计.

---

## 7. 代码集成方案

### 7.1 修改文件清单

| 文件 | 改动 | 推理影响 | 代码量 |
|------|------|---------|--------|
| `ldmdet/core/head.py` | `loss()` 增加 `_compute_islr`; 新增方法 | 无 | ~80 行 |
| `ldmdet/criterion/criterion.py` | `__init__` 增加 ISLR 配置 | 无 | ~20 行 |
| `experiments/configs/.../karyoflow_islr.py` | `use_islr=True, islr_lambda_0=0.01, islr_t_power=2.0` | 无 | ~30 行 |

### 7.2 核心实现伪代码 (R1 反馈 2 修正: 强制子采样 + $t$-dependent 加权)

```python
# ldmdet/core/head.py

def _compute_islr(self, x_t, t, x0_pred_fn, K_islr=16, t_power=2.0):
    """ISLR: ||J_f||_F^2 via 4 VJPs for d=4, with mandatory proposal subsampling
    and t-dependent weighting (R1 反馈 2 + 3 修正).

    Args:
        x_t: [B, K, 4] 输入 (raw cxcywh)
        t: [B] 时间步
        x0_pred_fn: callable, x_t -> x0_pred
        K_islr: int, 子采样 proposal 数 (默认 16, 强制)
        t_power: float, t-dependent 加权幂次 (默认 2.0)
    Returns:
        islr_loss: 标量, 加权 ||J_f||_F^2 的 batch 均值
    """
    bs, K, d = x_t.shape  # d=4
    assert d == 4

    # 强制子采样: 从 K≈46 中随机选 K_islr=16 个 proposals (R1 反馈 2 修正)
    if K > K_islr:
        idx = torch.randperm(K, device=x_t.device)[:K_islr]
        x_t_sel = x_t[:, idx].reshape(-1, d)  # [bs * K_islr, 4]
        t_sel = t.repeat_interleave(K_islr)  # [bs * K_islr]
    else:
        x_t_sel = x_t.reshape(-1, d)
        t_sel = t.repeat_interleave(K)
    x_t_sel = x_t_sel.detach().requires_grad_(True)  # [N, 4]

    x0_pred = x0_pred_fn(x_t_sel)  # [N, 4]

    # 4 VJPs: grad of (e_i^T x0_pred) w.r.t. x_t = i-th row of J_f
    jac_norm_sq = torch.zeros(x_t_sel.shape[0], device=x_t_sel.device)
    for i in range(d):
        e_i = torch.zeros_like(x0_pred)
        e_i[:, i] = 1.0
        grad_i = torch.autograd.grad(
            (e_i * x0_pred).sum(dim=1), x_t_sel,
            create_graph=True, retain_graph=True,
        )[0]  # [N, 4]
        jac_norm_sq += (grad_i ** 2).sum(dim=1)

    # t-dependent 加权 (R1 反馈 3 修正): t 大 (噪声端) 强正则, t 小 (数据端) 弱正则
    t_weights = t_sel ** t_power  # [N]
    weighted_jac_norm_sq = jac_norm_sq * t_weights  # [N]

    return weighted_jac_norm_sq.mean()

# In loss():
if self.use_islr and self.training:
    losses['loss_islr'] = self.islr_lambda_0 * self._compute_islr(
        x_t, t, x0_pred_fn,
        K_islr=self.islr_K,  # 默认 16
        t_power=self.islr_t_power,  # 默认 2.0
    )
```

### 7.3 Fallback 方案 (R1 反馈 2 新增)

| 方案 | VJP 次数 | 开销 | 方差 | 适用场景 |
|------|---------|------|------|---------|
| 主方案: 4 VJP 精确 | 4 | +80~200% | 0 (精确) | $d=4$ 甜区, Phase 0 验证开销可接受 |
| Fallback 1: Hutchinson 1-sample | 1 | +20~50% | $O(1/d)$ 量级 (依赖 $J$ 奇异值分布) | Phase 0 开销 > 100% |
| Fallback 2: Finlay-Oberman 有限差分 | 0 (避免二阶图) | +10~30% | 有限差分误差 | AMP 不稳定 |

**Hutchinson 方差说明 (R2 N3 修正)**: Hutchinson 估计器 $\hat{s} = \|z^\top J\|^2$ ($z \sim \mathcal{N}(0, I_d)$) 是 $\|J\|_F^2$ 的无偏估计, 其方差 $\text{Var}(\hat{s}) = 2 \sum_{i,j} J_{ij}^4 + 4 \sum_{i \neq k, j} J_{ij}^2 J_{kj}^2$ 依赖于 $J$ 的具体结构, **非简单的 $2/d^2$**. 对于 $d=4$, 若 $J$ 接近正交矩阵, 方差可能小于 12.5%; 若 $J$ 有大奇异值, 方差可能更大. 典型情况下 $O(1/d)$ 量级.

---

## 8. 预期收益分析

| 指标 | Baseline | ISLR-RF 预期 | 提升 | 依据 |
|------|:-:|:-:|:-:|------|
| mAP@50 | 0.863 | 0.864~0.868 | $+0.001 \sim +0.005$ | Rademacher 收紧 + ODE 适定性 (R1 反馈 1 下调) |
| mAP@75 | 0.974 | 0.976~0.980 | $+0.002 \sim +0.006$ | Grönwall 误差传播抑制 (R1 反馈 5 下调) |
| Y 染色体 AP | 0.781 | 0.781~0.791 | $+0.000 \sim +0.010$ | 小类雅可比更大 (待 Phase 1 验证, R1 反馈 4 下调) |
| $\eta_{\text{str}}$ | ~0.15 | ~0.13~0.14 | $-7 \sim 13\%$ | $J_f \downarrow \Rightarrow \eta_{\text{str}} \downarrow$ (R1 反馈 5 下调) |
| 推理 seed std | ~0.003 | ~0.0025~0.003 | $-10 \sim 20\%$ | $L_v \downarrow \Rightarrow$ Grönwall 指数 $\downarrow$ (R1 反馈 5 下调) |
| NFE | 24 | 24 | 0 | 推理零改动 |

---

## 9. 风险分析 (R1 反馈 2 + 3 修正)

| 风险 | 概率 | 影响 | 缓解 | Fallback |
|------|:-:|:-:|------|---------|
| R1: 4× 反传 + 二阶图开销 | **高** | **高** | 强制 $K_{\text{islr}}=16$ + Phase 0 微基准 | Hutchinson / Finlay-Oberman |
| R2: $\lambda$ 过大 | 中 | 高 | 网格搜索 $\lambda_0$ + 监控 $\mathcal{L}_{\text{det}}$ | $\lambda_0=0$ (baseline) |
| R3: double-backprop AMP 不稳定 | 中 | 中 | float32 计算 + 禁用 grad scaler | 有限差分近似 |
| R4: 与 TFR 梯度冲突 | 低 | 中 | 监控梯度余弦 + PCGrad | 不叠加 TFR |
| R5: 仅末级 head 不够 | 中 | 低 | 消融: 末级 vs 全部 6 级 | 正则所有级 |
| **R6: 过度正则化损害去噪 (R1 反馈 3 新增)** | **中** | **高** | $t$-dependent 加权 $\lambda(t) = \lambda_0 t^p$; 监控 $\mathcal{L}_{\text{det}}$ 不退化 | $\lambda_0=0$ (baseline) |
| **R7: 超参数 proliferation (R2 N1 新增)** | 中 | 中 | Phase 1 固定 $p=2$ 仅调 $\lambda_0$; Phase 2 在最优 $\lambda_0^*$ 附近做 $p$ 一维消融 | 默认 $p=2$ |

---

## 10. 实验计划 (R1 反馈 2 新增 Phase 0 微基准)

### Phase 0: 微基准测试 (0.5 天, R1 反馈 2 新增)

**目标**: 实测 4 VJP 的训练时间增量, 确认开销可接受.

**方法**: 用 `torch.autograd.functional.jacobian` 在小 batch (B=1, K=4) 上实测单步时间, 与 baseline 对比.

**判据**:
- 开销 < 100%: 进入 Phase 1 (主方案 4 VJP)
- 开销 ≥ 100%: 降级为 Hutchinson 1-sample (Fallback 1) 或 Finlay-Oberman (Fallback 2)
- 同时实测 Hutchinson 1-sample 在 $d=4$ 下的实际方差 (R2 N3 剩余问题)

**SwanLab run**: `islr_phase0_microbench`

### Phase 1: 单 seed 验证 (3 天)

**配置**: $\lambda_0=0.01$, $p=2$, 末级 head, $K_{\text{islr}}=16$.

**判据**: mAP $\geq 0.862$ (不掉点), $\eta_{\text{str}}$ 下降 $>5\%$.

**额外诊断 (R2 N4 剩余问题)**: 实测 $\|J_f(t)\|_F$ 曲线, 验证命题 1.8' 的 "理想 $J_f$ 在 $t \approx 0$ 为 $I$、$t \approx 1$ 为 $0$" 论断. 实测 per-class $\|J_f\|_F$ 验证稀有类雅可比更大假设.

### Phase 2: 消融实验 (3 天)

**实验矩阵**:
- $\lambda_0 \in \{0.001, 0.003, 0.01, 0.03, 0.1\}$ (5 配置)
- 在最优 $\lambda_0^*$ 附近做 $p \in \{1, 2, 4\}$ 一维消融 (3 配置, R2 N1 修正: 不做 2D 网格)
- 正则级数 (末级 vs 全部 6 级)
- TFR 叠加正交性验证

### Phase 3: 3-seed 验证 (5 天)

**配置**: 最优 $(\lambda_0^*, p^*)$, 3 seeds, 150 epochs.

**判据**: mean mAP $> 0.865$, std $< 0.003$.

### Phase 4: 论文级表格 + 诊断图 (3 天)

$\eta_{\text{str}}$ 曲线, seed 方差, 梯度分析, per-class $\|J_f\|_F$.

---

## 11. R2 评审剩余问题 (供后续 R3 参考)

1. **超参数搜索策略 (R2 N1, 中等)**: $(\lambda_0, p)$ 的 2D 网格搜索在 3-seed 验证下实验量过大, 需设计更高效的搜索策略. 已修订为 Phase 1 固定 $p=2$ 仅调 $\lambda_0$, Phase 2 在最优 $\lambda_0^*$ 附近做 $p$ 一维消融.
2. **Hutchinson fallback 方差 (R2 N3, 轻微)**: 需在 Phase 0 实测 Hutchinson 1-sample 估计在 $d=4$ 下的实际方差, 验证是否可接受. 已修订方差声明为 "取决于 $J$ 的奇异值分布, 典型情况下 $O(1/d)$ 量级".
3. **与 LSCR-RF 的互补性 (R2 剩余)**: ISLR 约束输入雅可比, LSCR-RF 约束权重谱范数, 二者互补. 若 LSCR-RF 也被采纳, 需验证二者叠加是否冲突或协同.
4. **命题 1.8' 的实验验证 (R2 N4)**: 理想 $J_f$ 在 $t \approx 0$ 为 $I$、$t \approx 1$ 为 $0$ 的论断应在 Phase 1 实测 $\|J_f(t)\|_F$ 曲线验证.
5. **Rademacher 界的常数因子 (R2 N5)**: $\hat{\mathcal{R}}_n(\mathcal{F}) \leq O(B_x \sqrt{M}/\sqrt{n})$ 中, $B_x$ (输入半径) 在 raw cxcywh 空间的具体值未给出, 影响定量收益预估.
6. **Rademacher 界线性 vs Bartlett 界平方 (R2 N4)**: Rademacher 界线性收紧弱于 Bartlett 界平方收紧, 小数据集收益论点进一步削弱. 应在论文中诚实标注.

---

## 12. 自评 (修订后)

### 12.1 评分 (10 分制, R2 修正后)

| 维度 | 分数 | 理由 |
|------|------|------|
| 理论严谨性 | 7.8 | Bartlett→Sokolić 切换正确, Rademacher 界与 Jacobian 正则化直接对应; 命题 1.8' 修正深刻; Grönwall $t$ 范围补充. 扣分: Rademacher 界线性收紧弱于 Bartlett 平方, 小数据集收益论点削弱 |
| 文献验证 | 7.5 | WANT 删除正确; Sokolić (2017) WebSearch 验证通过; Finlay-Oberman 和 Varga 补充合理. 扣分: Hutchinson 方差声明不严格 |
| 与 FALSIFIED 区分 | 8.5 | 区分清晰, h_velocity_loss 区分已加强 |
| 实现可行性 | 6.8 | 开销估算修正诚实 (+80~200%); Phase 0 微基准是好实践; 强制 $K_{\text{islr}}=16$ 是合理妥协. 扣分: 超参数 proliferation ($\lambda_0, p$), 方差降低论证削弱 |
| 任务匹配 | 7.0 | t-dependent 加权解决过度正则化, 但暴露了 "理想 $J_f$ 在 $t \approx 0$ 为 $I$ 而非 $0$" 的更深问题 — ISLR 的 "理想 RF 下 $J_f = 0$" 论断仅在 $t \approx 1$ 成立, 任务匹配论点需重新定位为 "噪声端 Lipschitz 正则化" |
| 预期收益 | 6.5 | 收益预期诚实下调 (mAP +0.001~+0.005); Rademacher 界提供严格但较弱的理论支撑 |
| 风险可控 | 7.5 | 7 个风险均有缓解和 fallback; t-dependent 加权缓解过度正则化; Phase 0 微基准控制开销风险 |
| 论文叙事 | 7.5 | "唯一直接界住 $L_f$" 仍成立; Rademacher 界论证更严格; t-dependent 加权增加技术深度 |

**综合评分**: 7.4/10

### 12.2 核心优势保留

- **唯一直接界住 $L_f$**: 全 Frobenius 范数 $\|J_f\|_F^2 \geq \|J_f\|_{\text{op}}^2$, 仍成立
- **$d=4$ 精确 VJP 甜区**: 仍成立, 但需 $K_{\text{islr}}=16$ 子采样控制开销
- **推理零改动**: 仍成立
- **与 TFR 正交可叠加**: 仍成立

### 12.3 核心弱点承认

- Bartlett 界论证错误 (已切换到 Rademacher, 但线性收紧弱于平方)
- 4 VJP 开销估算不足 (已修正, 强制子采样)
- 过度正则化风险未讨论 (已引入 t-dependent 加权)
- 预期收益基于过强理论界, 实际收益不确定 (已下调)
- 超参数 proliferation ($\lambda_0, p$ 两参数)
- "理想 $J_f = 0$" 仅在 $t \approx 1$ 成立, 任务匹配论点需重新定位

### 12.4 投稿建议

1. **TMI 正文**: "基于 Rademacher 复杂度的输入雅可比正则化" 作为训练正则化贡献. 强调:
   - 唯一直接界住 $L_f$ (命题 1.5);
   - $d=4$ 精确 VJP 甜区 (任务特性结合);
   - t-dependent 加权避免过度正则化 (命题 1.8' 深刻概念修正);
   - Rademacher 复杂度收紧 (理论严格但较弱, 诚实标注).

2. **arXiv companion**: 定理详细证明, Phase 0 微基准完整数据, Phase 2 完整消融表, per-class $\|J_f\|_F$ 曲线, Hutchinson 方差实测.

3. **创新点叙事**: "首次将输入雅可比精确正则化应用于 RF 检测, 基于 $d=4$ 低维优势以 4 次 VJP 精确计算全 Frobenius 范数, 通过 Sokolić Rademacher 复杂度界严格收紧泛化界, 并引入 t-dependent 加权避免过度正则化."

---

## 附录 A: 符号表

| 符号 | 含义 |
|------|------|
| $x_t \in \mathbb{R}^4$ | RF 轨迹状态 (cxcywh) |
| $f_\theta(x_t, t)$ | x0-prediction 网络 (6 级 cascade head) |
| $J_f = \nabla_{x_t} f_\theta \in \mathbb{R}^{4 \times 4}$ | 输入雅可比 |
| $L_f = \sup \|J_f\|_{\text{op}} \leq \sup \|J_f\|_F$ | Lipschitz 常数 |
| $v_\theta = (x_t - f_\theta)/t$ | RF 速度场 |
| $L_v \leq (1+L_f)/t$ | 速度场 Lipschitz 常数 |
| $\lambda(t) = t^p$ ($p=2$ 默认) | t-dependent 加权函数 |
| $\lambda_0$ | ISLR 基础正则强度 (默认 0.01) |
| $\mathcal{L}_{\text{ISLR}} = \lambda_0 \cdot \mathbb{E}[\lambda(t) \cdot \|J_f\|_F^2]$ | ISLR 正则项 |
| $\eta_{\text{str}} = \|D_1\|/\|\hat{x}_0\|$ | 轨迹直线度诊断 |
| $K_{\text{islr}}=16$ | 强制子采样 proposal 数 |
| $n=2274$, $K\approx 46$, $d=4$ | 数据集参数 |
| $t_{\min}=0.25$ | DPM-Solver++ 4 步采样最小有效 $t$ |
| $B_x$ | 输入半径 (raw cxcywh 空间) |
| $M = \mathbb{E}[\|J_f\|_F^2]$ | Rademacher 界参数 |

---

## 附录 B: Picard-Lindelöf 定理 (R1 反馈 6 降级)

**Picard-Lindelöf ODE 适定性定理 (1894)**. 考虑 ODE $\dot{x} = f(x, t)$, $x(t_0) = x_0$. 若 $f$ 在某区域 $\mathcal{R}$ 上连续且关于 $x$ 满足 Lipschitz 条件 (Lipschitz 常数 $L$), 则 ODE 在 $\mathcal{R}$ 内存在唯一解.

**与 ISLR-RF 的关系**: Picard-Lindelöf 保证 RF ODE $dx/dt = v_\theta(x, t)$ 在 $v_\theta$ Lipschitz 时解存在唯一, 这是任何求解器 (包括 DPM-Solver++) 的**理论基础**. ISLR 通过降低 $L_v$ 增强适定性, 间接惠及 DPM-Solver++ 的数值稳定性. **但 Picard-Lindelöf 不直接指导正则化设计, 故降级为附录**.

---

*文档结束. ISLR-RF 基于 Lipschitz (1886) + Grönwall (1919) + Sokolić Rademacher 复杂度 (2017), 从输入雅可比 Frobenius 范数推导出唯一直接界住 Lipschitz 常数的正则化方案. 与 TFR (时间)、SDR-RF (散度)、CFR (旋度) 正交/包含, 可叠加. $d=4$ 精确 VJP (强制 $K_{\text{islr}}=16$ 子采样), t-dependent 加权避免过度正则化, 推理零改动. 预期 mAP +0.001~+0.005, mAP@75 +0.002~+0.006.*
