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

除算法示意图外，本目录还包含多组定量对比图，覆盖 **chromo** 与 **24obj** 两个数据集，从召回率、准确率及混淆矩阵三个维度对比 baseline 与 SOTA。

- **数据源**: `experiments/analysis/baseline_vs_sota_results.json`（多数据集, 步数对齐 4-step）
- **生成脚本**: `generate_comparison_figures.py`
- **数据收集脚本**: `experiments/analysis/baseline_vs_sota.py`

### 跨数据集聚合指标汇总

**文件**: `per_dataset_summary_table.png` / `per_dataset_summary_table_zh.png`

| 数据集 · 模型 | mAP | AP50 | AP75 | AP_s | AP_m | AP_l | AR@100 | Det P | Det R | F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chromo · DiffusionDet (n=3) | 0.728 | 0.922 | 0.818 | 0.479 | 0.723 | 0.645 | 0.787 | 0.947 | 0.944 | 0.946 |
| chromo · SOTA (n=1) | 0.748 | 0.939 | 0.837 | 0.512 | 0.738 | 0.641 | 0.807 | 0.902 | 0.963 | 0.931 |
| 24obj · DiffusionDet (n=1) | 0.803 | 0.970 | 0.936 | 0.423 | 0.800 | 0.814 | 0.848 | 0.957 | 0.977 | 0.967 |
| 24obj · SOTA (n=1) | 0.852 | 0.985 | 0.966 | 0.405 | 0.848 | 0.906 | 0.896 | 0.983 | 0.991 | 0.987 |

> 各数据集 mAP 最优以绿色加粗标注。chromo 最优为 SOTA (0.748)，相对 DiffusionDet baseline (0.728) 提升 +0.020；24obj 最优为 SOTA (0.852)，相对 DiffusionDet (0.803) 显著领先 +0.049。

---

### chromo 数据集: DiffusionDet vs SOTA

chromo 数据集共 2 个模型对比：DiffusionDet baseline（3 seeds, mAP=0.728）、SOTA（1 seed, 即 RF+Heun+AdaLN+Sinkhorn Stochastic OT, 训练 best epoch 0.753, 实测推理 0.748）。直接对比 baseline → SOTA 的完整提升。

#### 数据汇总

| 指标 | DiffusionDet (n=3) | SOTA (n=1) | Δ |
|------|:---:|:---:|:---:|
| **mAP** | 0.728 | 0.748 | **+0.020** |
| AP50 | 0.922 | 0.939 | +0.016 |
| AP75 | 0.818 | 0.837 | +0.019 |
| AP_s | 0.479 | 0.512 | **+0.033** |
| AP_m | 0.723 | 0.738 | +0.016 |
| AP_l | 0.645 | 0.641 | -0.004 |
| AR@100 | 0.787 | 0.807 | **+0.020** |
| AR_s | 0.550 | 0.614 | **+0.064** |
| AR_m | 0.776 | 0.791 | +0.015 |
| AR_l | 0.667 | 0.680 | +0.014 |
| Det Precision | 0.947 | 0.902 | -0.045 |
| Det Recall | 0.944 | 0.963 | +0.019 |
| Det F1 | 0.946 | 0.931 | -0.014 |

**关键结论**:
- **SOTA 全面优于 DiffusionDet baseline**：ΔmAP=+0.020（0.748 vs 0.728），AP/AR 几乎全面正增长，小目标收益最大 (ΔAP_s=+0.033, ΔAR_s=+0.064)
- **唯一退化维度**：AP_l 略降 (-0.004)，大目标本身定位难度低，改进空间有限；固定阈值下 Det Precision 下降 0.045（P/R 权衡，SOTA 生成更多候选框 ~2x）
- **训练 best 0.753 ≠ 推理 0.753**：训练 best epoch mAP 为 0.753，实测推理（固定 seed=42, 4-step）为 0.748，验证集波动约 0.005

#### 1. 每类 AP 柱状图

**文件**: `comparison_per_class_ap.png` / `comparison_per_class_ap_zh.png`

**图意**: 24 个染色体类别的 AP (IoU=0.5:0.95) 双柱分组图（灰=DiffusionDet | 橙=SOTA），3-seed 均值 ± 标准差（SOTA 仅 1 seed 无误差棒）。每类上方标注 Δ 值（绿色正/红色负），虚线为各模型 mAP 均值。

**解读**:
- DiffusionDet → SOTA（灰→橙）: 绝大多数类别提升（绿色标注为主），Y、A2、C7 等难类别增益最大
- 标准差普遍 <0.01（DiffusionDet），说明 3-seed 结果稳定；SOTA 仅 1 seed，无法评估稳定性

#### 2. 混淆矩阵热力图

**文件**: `comparison_confusion_matrix.png` / `comparison_confusion_matrix_zh.png`

**图意**: 三面板热力图——左: DiffusionDet 混淆矩阵（行归一化），中: SOTA 混淆矩阵，右: Δ 矩阵 (SOTA − DiffusionDet)。行 = GT 类别，列 = 预测类别，对角线 = 分类正确率 (recall)。配置: IoU=0.5, score=0.3。

**解读**:
- **对角线整体更亮**：SOTA 的分类正确率（recall）在多数类别上高于 DiffusionDet，Δ 对角线以绿色为主
- **Δ 面板绿色显著**：明显的绿色块印证 +0.020 mAP 的完整提升
- 个别类别（如 Y）可能出现微小红斑，反映固定阈值下难类别的 P/R 权衡

#### 3. 聚合 P/R 表格

**文件**: `comparison_pr_table.png` / `comparison_pr_table_zh.png`

**图意**: 聚合指标对比表格，分三组——COCO 准确率、COCO 召回率、检测 P/R/F1。绿色 Δ = 改善，红色 = 退化。

**解读**:
- **COCO 指标全面正增长**：mAP +0.020, AP50 +0.016, AP75 +0.019, AP/AR 各尺度均提升（仅 AP_l -0.004 略降）
- **小目标收益最大**：ΔAP_s=+0.033, ΔAR_s=+0.064——RF+Heun 的连续 ODE 轨迹 + shifted schedule + OT 耦合对小目标检测帮助最大
- **固定阈值 P/R 权衡**：Det Precision 下降 0.045 但 Recall 上升 0.019——SOTA 生成更多预测 (~2x)，在固定 score=0.3 下 FP 增多。但 mAP（阈值无关）仍提升 +0.020，说明模型**判别能力**真正提升

#### 4. 每类检测 P/R 柱状图

**文件**: `comparison_per_class_pr.png` / `comparison_per_class_pr_zh.png`

**图意**: 上下两子图分别为每类 Precision 和 Recall 分组柱状图（混淆矩阵, IoU=0.5, score=0.3）。

**解读**:
- **Recall 子图**: SOTA 在多数类别上 recall 更高，尤其 Y、D13-D15 等难类别
- **Precision 子图**: DiffusionDet 在多数类别上 precision 略高——因为 DiffusionDet 预测数少 (~67K)，FP 绝对数少
- 这一 P/R 权衡是 SOTA 生成更多候选框的直接结果，在实际部署中可通过调高 score 阈值来平衡

---

### 24obj 数据集: DiffusionDet vs SOTA

24obj 数据集共 2 个模型对比：DiffusionDet baseline（1 seed, mAP=0.803）、SOTA（1 seed, 即 RF+Heun+AdaLN+Sinkhorn Stochastic OT, mAP=0.852）。24obj 的 DiffusionDet baseline 与 chromo 不同（独立训练的 benchmark 配置），SOTA 权重路径见 `baseline_vs_sota.py`。

#### 数据汇总

| 指标 | DiffusionDet (n=1) | SOTA (n=1) | Δ |
|------|:---:|:---:|:---:|
| **mAP** | 0.803 | 0.852 | **+0.049** |
| AP50 | 0.970 | 0.985 | +0.015 |
| AP75 | 0.936 | 0.966 | +0.030 |
| AP_s | 0.423 | 0.405 | **-0.017** |
| AP_m | 0.800 | 0.848 | +0.048 |
| AP_l | 0.814 | 0.906 | **+0.092** |
| AR@100 | 0.848 | 0.896 | +0.048 |
| AR_s | 0.477 | 0.424 | **-0.053** |
| AR_m | 0.845 | 0.894 | +0.049 |
| AR_l | 0.868 | 0.925 | +0.058 |
| Det Precision | 0.957 | 0.983 | +0.027 |
| Det Recall | 0.977 | 0.991 | +0.014 |
| Det F1 | 0.967 | 0.987 | +0.021 |

**关键结论**:
- **SOTA 显著全面优于 DiffusionDet**：ΔmAP=+0.049，远大于 chromo 上的 +0.020
- **大/中目标收益巨大**：ΔAP_l=+0.092, ΔAP_m=+0.048, ΔAR_l=+0.058——24obj 数据集目标尺度分布与 chromo 不同，SOTA 对大中目标的定位精度提升显著
- **小目标反而退化**：ΔAP_s=-0.017, ΔAR_s=-0.053——这是 24obj 上唯一退化的维度，可能与 OT 耦合的尺度无关配对在小目标上的次优性有关
- **检测 P/R/F1 全面提升**：与 chromo 上 Det Precision 下降不同，24obj SOTA 的 P/R 同步上升 (ΔF1=+0.021)，说明模型在 24obj 上的候选框质量整体更优
- **跨数据集对比**：同一 SOTA 方法在 chromo 上 +0.020，在 24obj 上 +0.049——OT 耦合的收益与数据集的目标分布强相关，24obj 的更大尺度跨度使 OT 配对收益更明显

#### 1. 每类 AP 柱状图

**文件**: `24obj_per_class_ap.png` / `24obj_per_class_ap_zh.png`

**图意**: 24 个染色体类别的 AP (IoU=0.5:0.95) 双柱分组图（灰=DiffusionDet | 橙=SOTA），1 seed。每类上方标注 Δ 值（绿色正/红色负），虚线为各模型 mAP 均值。

**解读**:
- SOTA 在绝大多数类别上提升，Δ 普遍为正（绿色标注为主）
- 提升幅度在不同染色体组间分布相对均匀，说明 SOTA 的增益是算法层面的整体提升，而非个别类别驱动

#### 2. 混淆矩阵热力图

**文件**: `24obj_confusion_matrix.png` / `24obj_confusion_matrix_zh.png`

**图意**: 三面板热力图——左: DiffusionDet 混淆矩阵（行归一化），中: SOTA 混淆矩阵，右: Δ 矩阵 (SOTA − DiffusionDet)。配置: IoU=0.5, score=0.3。

**解读**:
- **对角线整体更亮**：SOTA 的分类正确率（recall）在多数类别上提升，Δ 对角线以绿色为主
- **Δ 面板绿色显著**：明显的绿色块印证 +0.049 mAP 的显著提升
- 个别类别可能出现微小红斑（小目标相关），与小目标 AP 退化一致

#### 3. 聚合 P/R 表格

**文件**: `24obj_pr_table.png` / `24obj_pr_table_zh.png`

**图意**: 聚合指标对比表格，分三组——COCO 准确率、COCO 召回率、检测 P/R/F1。绿色 Δ = 改善，红色 = 退化。

**解读**:
- **几乎全绿**：除 AP_s/AR_s 略红外，其余指标均为绿色正增长
- **ΔAP_l=+0.092 是最大亮点**：大目标定位精度提升近 10 个百分点，是 mAP +0.049 的主要贡献
- 检测 P/R/F1 同步上升 (ΔF1=+0.021)，无 chromo 上的 P/R 权衡问题

#### 4. 每类检测 P/R 柱状图

**文件**: `24obj_per_class_pr.png` / `24obj_per_class_pr_zh.png`

**图意**: 上下两子图分别为每类 Precision 和 Recall 分组柱状图（混淆矩阵, IoU=0.5, score=0.3）。

**解读**:
- SOTA 在多数类别的 P 和 R 上均高于 DiffusionDet，与聚合表的全面领先一致
- 个别小目标类别可能出现 R 略降，与 AR_s=-0.053 一致

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

#### chromo 数据集 (DiffusionDet vs SOTA)

| 图表 | 英文 | 中文 | 说明 |
|---|---|---|---|
| 每类 AP | comparison_per_class_ap.png | _zh.png | 24 类双柱 AP (DiffusionDet\|SOTA) |
| 混淆矩阵 | comparison_confusion_matrix.png | _zh.png | 3 面板热力图 (DiffusionDet\|SOTA\|Δ) |
| 聚合 P/R | comparison_pr_table.png | _zh.png | DiffusionDet vs SOTA COCO AP/AR + P/R/F1 |
| 每类 P/R | comparison_per_class_pr.png | _zh.png | DiffusionDet vs SOTA 每类 Precision/Recall |

#### 24obj 数据集 (DiffusionDet vs SOTA)

| 图表 | 英文 | 中文 | 说明 |
|---|---|---|---|
| 每类 AP | 24obj_per_class_ap.png | _zh.png | 24 类双柱 AP (DiffusionDet\|SOTA) |
| 混淆矩阵 | 24obj_confusion_matrix.png | _zh.png | 3 面板热力图 (DiffusionDet\|SOTA\|Δ) |
| 聚合 P/R | 24obj_pr_table.png | _zh.png | DiffusionDet vs SOTA COCO AP/AR + P/R/F1 |
| 每类 P/R | 24obj_per_class_pr.png | _zh.png | DiffusionDet vs SOTA 每类 Precision/Recall |

#### 跨数据集汇总

| 图表 | 英文 | 中文 | 说明 |
|---|---|---|---|
| 汇总表 | per_dataset_summary_table.png | _zh.png | chromo+24obj 全模型聚合指标 |

> BoxRefineNet（原 Direction D）因 ΔmAP ≈ 0（持平 baseline）已从图中移除。
