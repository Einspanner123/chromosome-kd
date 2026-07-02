# 实验脉络算法示意图 — 图片解读

本目录包含 `EXPERIMENT_LINEAGE.md` 中各改进的底层算法可视化示意图，共 7 个面板，覆盖从 DiffusionDet DDPM 根基线到 OT Flow 耦合的完整改进脉络。

- **拼版**: `experiment_lineage_schematics.png` (EN) / `experiment_lineage_schematics_zh.png` (中文)
- **单图**: `panel_{a..g}_<name>.png` (EN) / `panel_{a..g}_<name>_zh.png` (中文)
- **生成脚本**: `generate_algorithm_schematics.py`（`python generate_algorithm_schematics.py` 一次性生成全部 16 张 PNG）

> **基线**: DiffusionDet DDPM = 0.729 ± 0.003 → RF+Heun+AdaLN = 0.746 ± 0.001（chromo 数据集, DiffusionDet 默认 aug, 3 seeds）

---

## 各面板解读

### (a) DiffusionDet DDPM — 根基线

**文件**: `panel_a_ddpm.png` | **ΔmAP**: 0.729 ± 0.003

**论文出处**:
- DiffusionDet: Chen et al., "DiffusionDet: Diffusion Model for Object Detection", ICCV 2023
- DDPM: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- DDIM: Song et al., "Denoising Diffusion Implicit Models", ICLR 2021

**图意**: 离散马尔可夫链 $x_0 \leftrightarrow x_1 \leftrightarrow \cdots \leftrightarrow x_T$，上方灰色箭头为前向加噪 $q(x_t \mid x_{t-1})$，下方橙色箭头为反向去噪 $p_\theta(x_{t-1} \mid x_t)$。

**解读**:
- DiffusionDet 采用 DDIM 采样器，**1 步推理**（`sampling_timesteps=1`），即从纯噪声 $x_T$ 直接预测检测结果，NFE=1
- 时间条件使用 `scale_shift` 方式（平移+缩放正弦编码）
- **步数对齐验证**（绿色高亮）: DDIM 4 步（4 NFE）= 0.729 ± 0.004，DDIM 8 步（8 NFE）= 0.729 ± 0.003——**加步数不提升**。DDPM 模型针对 1 步推理训练，多步 DDIM 无法利用额外计算量
- 这是所有改进的对比基准：后续 RF+Heun 的 +0.017 是**纯算法贡献**（步数对齐后 Δ 仍为 +0.017），非计算量增量

---

### (b) RF + Heun + Shifted + AdaLN-Zero — 主要贡献

**文件**: `panel_b_rf_heun.png` | **ΔmAP**: +0.017（主要贡献）

**论文出处**:
- Rectified Flow: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023
- Shifted schedule: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3), ICML 2024
- AdaLN-Zero: Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), 2023
- Heun 求解器: 经典二阶 Runge-Kutta 数值方法（非论文首创）
- **本文组合**: 将上述方法首次统一应用于扩散检测框架

**图意**: 三部分组成——
1. **上部**: Rectified Flow 连续 ODE 轨迹 $x_1 \to x_0$（$t: 1 \to 0$），橙色虚线为 Heun 二阶积分的 Euler 预测点 + 中点校正
2. **左下**: Shifted 噪声调度曲线 $\sigma(t)$，对比线性调度，shift 参数 $s=3$ 使中间时刻信噪比更高
3. **右下**: AdaLN-Zero 模块，调制公式 $h = (1+\gamma)x + \beta$，$\gamma$ 零初始化（$\sim \mathcal{N}(0, 10^{-5})$）

**解读**:
- **Rectified Flow** 将离散马尔可夫链替换为连续 ODE，轨迹被"拉直"，降低采样复杂度
- **Heun 求解器**（2 阶 Runge-Kutta）：每步 2 次 NFE（Euler 预测 + 中点校正），4 步共 8 NFE，比 Euler 1 阶精度更高
- **Shifted schedule**：$s=3$ 的对数正态偏移使模型在中间时刻（$t \approx 0.5$）分配更多训练信号，对检测任务中中等尺度目标更友好
- **AdaLN-Zero**：自适应 LayerNorm 的 $\gamma$ 初始化为 0，使残差路径初始为恒等映射，训练初期不破坏主干特征，随训练逐步激活
- ⚠️ **步数对齐已验证**: DDIM 4 步 = 0.729 = DDIM 1 步，加步数不提升 DDPM。+0.017 是纯算法贡献（RF 公式 + Heun + shifted + AdaLN），非计算量增量

---

### (c) DPM-Solver++ — 效率改进（非精度改进）

**文件**: `panel_c_dpm_solver.png` | **ΔmAP**: 同质，NFE -37%

**论文出处**:
- DPM-Solver++: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Models", NeurIPS 2022
- 本文将其从图像生成迁移到 RF 检测采样，验证效率提升

**图意**: 三部分组成——
1. **上部**: $\hat{x}_0^{\text{pred}}(t)$ 历史预测点（橙色），多项式插值连接，绿色双向箭头标注 order=2 配对（用 $t_{n-1}, t_n$ 两点预测 $t_{n+1}$）
2. **中部**: $t$ 轴上 6 个步长格点 $t_0 \dots t_6$
3. **下部**: 半线性精确积分公式 + 实验结论

**解读**:
- DPM-Solver++ 是**多步法**（multistep）：每步仅需 1 次 NFE（复用历史预测），而 Heun 每步需 2 次 NFE
- **核心公式** $x_{n+1} = \frac{t_{n+1}}{t_n} x_n + (1 - \frac{t_{n+1}}{t_n}) \hat{x}_n + \phi_1 D_1$：线性部分精确积分，非线性部分用多项式校正
- **关键结论（绿色高亮）**: 4 步（5 NFE）即可达到 Heun 4 步（8 NFE）的同等 mAP（0.746），**NFE 降低 37%**
- 6 步（7-8 NFE）时 +0.001，边际提升可忽略——DPM-Solver++ 的价值在**效率**而非精度
- 步数对齐（4 步 vs 4 步）时 Δ=+0.000；NFE 对齐（8 NFE）时 Δ=+0.001

---

### (d) Hard OT Coupling — 边际收益

**文件**: `panel_d_hard_ot.png` | **ΔmAP**: +0.001（边际）

**论文出处**:
- Minibatch OT 耦合: Pooladian et al., "Multisample Flow Matching: Straightening Flows with Minibatch Couplings", ICML 2023
- Hard OT（双射配对）为 minibatch OT 的确定性特例

**图意**: 上行蓝色圆点为 $x_0$（数据样本），下行灰色圆点为 $x_1$（噪声样本，经置换打乱顺序），橙色箭头为双射配对。底部为 OT 目标公式。

**解读**:
- **Hard OT（硬配对）**: 在 minibatch 内求解最优传输，得到双射（bijective）配对 $\pi$，使 $\sum_i c(x_0^{(i)}, x_1^{(\pi_i)})$ 最小
- 相比随机配对（$x_1$ 与任意 $x_0$ 配对），OT 配对使训练轨迹更短、更直，理论上加速收敛
- 但实践中 **ΔmAP 仅 +0.001**：chromo 数据集的检测框分布相对集中，随机配对的次优性被 RF 的轨迹拉直能力补偿，OT 的边际收益有限
- 公式移至底部避免被交叉配对箭头遮挡

---

### (e) Sinkhorn Stochastic OT — 边际收益

**文件**: `panel_e_sinkhorn.png` | **ΔmAP**: +0.002（边际）

**论文出处**:
- Sinkhorn OT: Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", NeurIPS 2013
- 随机 OT 耦合: Pooladian et al., "Multisample Flow Matching: Straightening Flows with Minibatch Couplings", ICML 2023

**图意**: 左侧为耦合矩阵 $P \in \Sigma$ 的热力图（YlOrBr 配色，绿色方框标注随机采样的配对），右侧为随机配对示意（线宽 $\propto P_{ij}$）。

**解读**:
- **Sinkhorn OT**: 在 Hard OT 基础上加入熵正则化 $\varepsilon H(P)$，使耦合矩阵 $P$ 从硬配对（0/1）变为软概率分布
- **随机采样**: 每次训练从 $P$ 中随机采样配对（stochastic），增加训练多样性
- $\varepsilon=5.0$ 为熵正则化强度：越大越接近均匀随机配对，越小越接近 Hard OT
- **ΔmAP +0.002** 略优于 Hard OT（+0.001）：随机性带来轻度数据增强效果
- 但总体仍属边际——OT 耦合在检测任务上的收益不如生成任务显著

---

### (f) Focal Loss γ=3 — 分类损失调整

**文件**: `panel_f_focal.png` | **ΔmAP**: +0.004

**论文出处**:
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), ICCV 2017
- 本文仅调整超参数 $\gamma: 2 \to 3$，非方法首创

**图意**: Focal Loss 曲线对比 $\gamma=1$（等价 CE）/ $\gamma=2$（默认）/ $\gamma=3$（本文），横轴为预测概率 $p$，纵轴为损失值。橙色标注 $\gamma=3$ 在低 $p$ 区域的放大效果。

**解读**:
- **Focal Loss** 公式 $\mathcal{L} = -\alpha(1-p)^\gamma \log(p)$，$\gamma$ 控制难样本（低 $p$）的权重放大程度
- $\gamma=3$ 比 $\gamma=2$ 进一步压制易样本（高 $p$）的梯度贡献，使模型更关注难分类的染色体实例
- **ΔmAP +0.004** 是所有边际改进中收益最大的——chromo 数据集存在类间相似度高（如近端着丝粒 vs 中着丝粒染色体）的问题，Focal $\gamma=3$ 有效缓解了难样本的分类错误
- 箭头标注：在 $p=0.15$ 处，$\gamma=3$ 的损失约为 $\gamma=2$ 的 1.5 倍，梯度信号显著增强

---

### (g) OT Flow Coupling (E4.2) — OT Flow 耦合有效

**文件**: `panel_g_ot_flow.png` | **ΔmAP**: +0.005

**论文出处**:
- 非线性轨迹思路: Pooladian et al., "Multisample Flow Matching: Straightening Flows with Minibatch Couplings", ICML 2023
- Sinkhorn OT 配对: Cuturi, NeurIPS 2013
- **本文首创**: 尺度调制噪声调度 $\sigma(t, s) = t^{\kappa(s)}$，按目标尺度 $s$ 动态调节噪声增长速率

**图意**: 三部分组成——
1. **上部**: 不同尺度目标的非线性轨迹——蓝色直线为大目标（$\kappa \approx 1$），橙色曲线为小目标（$\kappa > 1$，噪声增长更快）
2. **左下**: 尺度调制函数 $\kappa(s)$ 曲线，$s$ 为目标尺度，$\kappa$ 随 $s$ 减小而增大
3. **右下**: OT Flow 耦合说明框

**解读**:
- **核心创新**: 噪声调度不再全局固定，而是**按目标尺度 $s$ 调制**——$\sigma(t, s) = t^{\kappa(s)}$
- $\kappa(s) = 1 + \lambda \cdot \frac{s_{\max} - s}{s_{\max}}$：小目标（$s$ 小）的 $\kappa > 1$，噪声增长更快，迫使模型在更早时刻学习小目标的精细结构
- **Sinkhorn OT 配对**: 同时使用 OT 耦合（逐 batch 传输方案），使训练轨迹与尺度调制协同
- **ΔmAP +0.005**：OT Flow 在所有边际改进中收益最高，验证了"小目标需要更快噪声增长"的假设
- 实验编号 E4.2：仅使用 OT Flow 耦合（非线性轨迹），不含 SCRF 的完整 scale-conditioned 机制

---

## 文件清单

| 面板 | 英文 | 中文 | ΔmAP |
|---|---|---|---|
| 拼版 | experiment_lineage_schematics.png | _zh.png | — |
| (a) DDPM | panel_a_ddpm.png | panel_a_ddpm_zh.png | 0.729 (根, 步数对齐) |
| (b) RF+Heun | panel_b_rf_heun.png | panel_b_rf_heun_zh.png | +0.017 (步数对齐) |
| (c) DPM-Solver++ | panel_c_dpm_solver.png | panel_c_dpm_solver_zh.png | NFE -37% |
| (d) Hard OT | panel_d_hard_ot.png | panel_d_hard_ot_zh.png | +0.001 |
| (e) Sinkhorn OT | panel_e_sinkhorn.png | panel_e_sinkhorn_zh.png | +0.002 |
| (f) Focal γ=3 | panel_f_focal.png | panel_f_focal_zh.png | +0.004 |
| (g) OT Flow | panel_g_ot_flow.png | panel_g_ot_flow_zh.png | +0.005 |

> BoxRefineNet（原 Direction D）因 ΔmAP ≈ 0（持平 baseline）已从图中移除。

---

## Baseline vs SOTA 对比分析图

除算法示意图外，本目录还包含 4 组定量对比图，从召回率、准确率及混淆矩阵三个维度对比 DDPM baseline 与 RF+Heun SOTA。

- **数据源**: `experiments/analysis/baseline_vs_sota_results.json`（3 seeds × 2 模型，步数对齐 4-step）
- **生成脚本**: `generate_comparison_figures.py`
- **数据收集脚本**: `experiments/analysis/baseline_vs_sota.py`

### 数据汇总 (3-seed 均值)

| 指标 | DDPM (baseline) | RF+Heun (SOTA) | Δ |
|------|:---:|:---:|:---:|
| **mAP** (IoU=0.5:0.95) | 0.728 | 0.746 | **+0.018** |
| AP50 | 0.922 | 0.941 | +0.019 |
| AP75 | 0.818 | 0.834 | +0.016 |
| AP_s (small) | 0.479 | 0.513 | **+0.035** |
| AP_m (medium) | 0.723 | 0.740 | +0.017 |
| AP_l (large) | 0.645 | 0.651 | +0.006 |
| **AR@100** | 0.787 | 0.807 | **+0.020** |
| AR_s (small) | 0.550 | 0.620 | **+0.070** |
| AR_m (medium) | 0.776 | 0.794 | +0.018 |
| AR_l (large) | 0.667 | 0.689 | +0.022 |
| Det Precision (IoU=0.5, score=0.3) | 0.947 | 0.901 | -0.046 |
| Det Recall (IoU=0.5, score=0.3) | 0.944 | 0.964 | +0.020 |
| Det F1 | 0.946 | 0.931 | -0.014 |

### 1. 每类 AP 柱状图

**文件**: `comparison_per_class_ap.png` / `comparison_per_class_ap_zh.png`

**图意**: 24 个染色体类别的 AP (IoU=0.5:0.95) 分组柱状图，灰色为 DDPM baseline，橙色为 RF+Heun SOTA，3-seed 均值 ± 标准差。每类上方标注 Δ 值（绿色正/红色负），虚线为各自 mAP 均值。

**解读**:
- RF+Heun 在**全部 24 类**上均有提升（无任何类别退化），验证了改进的全局一致性
- 提升最大的类别: Y (+0.037)、A2 (+0.026)、C7 (+0.025)、E18 (+0.025)——其中 Y 是最难的类别 (AP 最低 0.594→0.631)，增益最大
- 提升最小的类别: F20 (+0.003)、F19 (+0.006)——这两类本身 AP 已较高，改进空间有限
- 增益分布在所有染色体组 (A-G, X, Y) 上均匀存在，说明 mAP +0.018 不是个别类别驱动，而是算法层面的整体提升
- 标准差普遍较小（<0.01），说明 3-seed 结果稳定

### 2. 混淆矩阵热力图

**文件**: `comparison_confusion_matrix.png` / `comparison_confusion_matrix_zh.png`

**图意**: 三面板热力图——左: DDPM 混淆矩阵（行归一化），中: RF+Heun 混淆矩阵（行归一化），右: Δ 矩阵 (SOTA − baseline)。行 = GT 类别，列 = 预测类别，对角线 = 分类正确率 (recall)。配置: IoU=0.5, score=0.3, 3-seed 均值。

**解读**:
- **对角线（分类准确率）**: RF+Heun 的对角线整体更亮（更高 recall），多数类别改善（如 B4 +0.035, C9 +0.033, E18 +0.034, X +0.031）
- **个别类别在固定阈值下退化**: C8 (-0.003)、G22 (-0.006)、Y (-0.035) 在 score=0.3 下分类准确率略降——RF+Heun 生成更多候选框，Y 等难类别的误分类增多。但 Y 的 AP（阈值无关）仍提升 +0.037，说明模型判别能力真正提升，固定阈值下的退化是 P/R 权衡的体现
- **Δ 面板**: 绿色表示 SOTA 改善，红色表示退化。对角线以绿色为主，非对角区域的红色斑点反映 RF+Heun 在个别类别间的误分类增加
- 行归一化使每行和为 1（含背景列），可直接比较各类的 recall 分布

### 3. 聚合 P/R 表格

**文件**: `comparison_pr_table.png` / `comparison_pr_table_zh.png`

**图意**: 聚合指标对比表格，分三组——COCO 准确率 (mAP/AP50/AP75/AP_s/AP_m/AP_l)、COCO 召回率 (AR@100/AR_s/AR_m/AR_l)、检测 P/R/F1 (混淆矩阵, IoU=0.5, score=0.3)。绿色 Δ = 改善，红色 = 退化。

**解读**:
- **COCO 指标全面提升**: mAP +0.018，AP/AR 在所有尺度上均正增长
- **小目标收益最大**: ΔAP_s=+0.035, ΔAR_s=+0.070——RF+Heun 的连续 ODE 轨迹和 shifted schedule 对小目标检测帮助最大
- **大目标收益最小**: ΔAP_l=+0.006——大目标本身的定位难度低，改进空间有限
- **固定阈值 P/R 权衡**: RF+Heun 的 Det Precision 下降 0.046 但 Recall 上升 0.020——RF+Heun 生成更多预测 (~2x)，在固定 score=0.3 下 FP 增多。但 mAP（遍历全 PR 曲线，阈值无关）仍提升 +0.018，说明模型的**判别能力**真正提升，而非简单地降低阈值

### 4. 每类检测 P/R 柱状图

**文件**: `comparison_per_class_pr.png` / `comparison_per_class_pr_zh.png`

**图意**: 上下两子图分别为每类 Precision 和 Recall 分组柱状图（混淆矩阵, IoU=0.5, score=0.3, 3-seed 均值）。

**解读**:
- **Recall 子图**: RF+Heun 在多数类别上 recall 更高，尤其 Y、D13-D15 等难类别
- **Precision 子图**: DDPM 在多数类别上 precision 略高——因为 DDPM 预测数少 (~67K)，FP 绝对数少
- 这一 P/R 权衡是 RF+Heun 生成更多候选框的直接结果，在实际部署中可通过调高 score 阈值来平衡

---

## 完整文件清单

### 算法示意图

| 面板 | 英文 | 中文 | ΔmAP |
|---|---|---|---|
| 拼版 | experiment_lineage_schematics.png | _zh.png | — |
| (a) DDPM | panel_a_ddpm.png | panel_a_ddpm_zh.png | 0.729 (根, 步数对齐) |
| (b) RF+Heun | panel_b_rf_heun.png | panel_b_rf_heun_zh.png | +0.017 (步数对齐) |
| (c) DPM-Solver++ | panel_c_dpm_solver.png | panel_c_dpm_solver_zh.png | NFE -37% |
| (d) Hard OT | panel_d_hard_ot.png | panel_d_hard_ot_zh.png | +0.001 |
| (e) Sinkhorn OT | panel_e_sinkhorn.png | panel_e_sinkhorn_zh.png | +0.002 |
| (f) Focal γ=3 | panel_f_focal.png | panel_f_focal_zh.png | +0.004 |
| (g) OT Flow | panel_g_ot_flow.png | panel_g_ot_flow_zh.png | +0.005 |

### Baseline vs SOTA 对比图

| 图表 | 英文 | 中文 | 说明 |
|---|---|---|---|
| 每类 AP | comparison_per_class_ap.png | _zh.png | 24 类 AP 柱状图 |
| 混淆矩阵 | comparison_confusion_matrix.png | _zh.png | 3 面板热力图 (DDPM/SOTA/Δ) |
| 聚合 P/R | comparison_pr_table.png | _zh.png | COCO AP/AR + 检测 P/R/F1 |
| 每类 P/R | comparison_per_class_pr.png | _zh.png | 每类 Precision/Recall |

> BoxRefineNet（原 Direction D）因 ΔmAP ≈ 0（持平 baseline）已从图中移除。
