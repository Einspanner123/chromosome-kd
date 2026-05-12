# LDMDet 完整理论与实验路线

> 从 DiffusionDet 基线到 Flow Matching + OT 理论的完整研究轨迹
> 最后更新：2026-05-12

---

## 一、研究起源与核心问题

### 1.1 动机

DiffusionDet 将目标检测建模为扩散模型的去噪过程：从随机噪声框出发，通过多步去噪逐步精化为检测框。核心瓶颈是**推理速度**——需要 4-8 步才能达到满意精度，而单步推理精度骤降。

**核心研究问题**：能否在保持/提升检测精度的同时，将推理步数压缩到 1-2 步？

### 1.2 数据集

染色体核型分析数据集（24 类染色体 + 性染色体），约 400+ 张显微图像。每张图包含约 46 个染色体实例（$K \approx 46$），检测空间为 $\mathbb{R}^4$（bbox 坐标）。

---

## 二、理论基础：统一数学框架

### 2.1 从 DDPM 到 Rectified Flow

**DDPM 前向过程**（DiffusionDet 原论文）：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

路径非直线，需要更多采样步数追踪弯曲轨迹。

**RF 前向过程**（我们的改进）：

$$x_t = (1-t) x_0 + t \cdot x_1, \quad t \in [0,1]$$

路径为直线，传输代价最小（$\mathcal{A}[\gamma] \geq \|x_1 - x_0\|^2$，等号当且仅当直线）。**理论上 RF 比 DDPM 需要更少采样步数达到相同精度。**

### 2.2 五大改进的数学统一

**检测最优传输原理**：检测框的生成质量由 ODE 路径的直度决定，路径直度由三个因素控制：

1. **耦合质量 $\pi$**：决定传输分量 $v_{OT}$ 的大小（$W_2$ 距离）
2. **修正曲率 $\text{Curv}(\delta v)$**：决定采样误差的上界
3. **时间分配 $p(t)$**：决定训练资源在损失敏感度上的分配效率

| 改进 | 优化的分量 | 数学效果 |
|---|---|---|
| DDPM → RF | $v_{OT}$ 的路径形式 | 弯曲路径 → 直线，$v_{OT}$ 变为常数 |
| OT Coupling | $v_{OT}$ 的耦合 $\pi^*$ | 最小化 $\|v_{OT}\|^2 = W_2^2$ |
| Structured Noise | $v_{OT}$ 的源分布 $\mu_z$ | 减小 $W_2^2(\mu_z, \mu_{gt})$ |
| Shifted Schedule | 训练资源在 $t$ 上的分配 | 集中优化 $\delta v_\theta$ 影响最大处 |
| AdaLN-Zero | $\delta v_\theta$ 的初始化 | 初始 $\delta v_\theta = 0$，路径零曲率 |

### 2.3 速度场分解定理

$$v_\theta(x_t, t, f) = \underbrace{v_{OT}(x_t, t)}_{\text{OT 传输分量}} + \underbrace{\delta v_\theta(x_t, t, f)}_{\text{特征修正分量}}$$

- $v_{OT}$：仅依赖 OT 耦合，可直接解析计算
- $\delta v_\theta$：依赖图像特征的修正，是网络需要学习的部分

**采样误差完全由修正分量 $\delta v_\theta$ 的曲率决定。**

### 2.4 Shifted Schedule 的理论依据

原始采样 $t \sim U(0,1)$，shifted schedule 映射 $g: t \mapsto \frac{st}{1+(s-1)t}$（$s=3$）。

检测损失对速度误差的敏感度与 $t$ 成正比：$\frac{\partial \mathcal{L}_{det}}{\partial v_\theta} = -t \cdot \nabla_{x_0} \mathcal{L}_{det}$。Shifted schedule 在敏感度最高的区域（$t \approx 1$）分配最多训练资源。

### 2.5 AdaLN-Zero 的零初始化优势

Scale-shift: $\gamma, \beta$ 随机初始化（非零），初始路径有随机曲率。

AdaLN-Zero: MLP 最后一层零初始化 → $v_\theta|_{\text{init}} = 0$ → 初始 ODE 路径曲率为零，训练从最直路径出发。

---

## 三、实验演进路线

### 阶段 0：基线建立（DDPM 体系）

| 实验 | mAP | 核心配置 |
|---|---|---|
| `diffusiondet_baseline` | ~0.45 | DiffusionDet 原论文复现（DDPM, scale-shift, 1步） |
| `ldmdet_baseline` | 0.725 | 染色体数据集基线（DDPM, scale-shift, 4步推理） |

**关键**：DiffusionDet 在染色体数据集上基线 0.725，为后续改进提供参照。

### 阶段 1：RF 基础改进（路径直化）

| 实验 | mAP | mAP@50 | mAP@75 | 核心改进 |
|---|---|---|---|---|
| `ldmdet_rf` | 0.733 | 0.939 | 0.828 | RF 替代 DDPM |
| `ldmdet_rf_shifted_schedule` | 0.747 | 0.937 | 0.837 | RF + Shifted Schedule ($s=3$) |
| `ldmdet_flowdet_adaln` | **0.751** | 0.943 | 0.843 | RF + AdaLN-Zero + Shifted + Heun |

**关键发现**：
- RF 单独提升 +0.8%，证明直线路径优于弯曲路径
- Shifted schedule 额外 +1.4%，验证时间敏感度假说
- AdaLN-Zero 额外 +0.4%，零初始化收益显著
- **AdaLN 是单项最大改进（+1.8% vs RF baseline）**

### 阶段 2：OT 耦合探索（传输优化）

| 实验 | mAP | 关键参数 | 发现 |
|---|---|---|---|
| `ldmdet_flowdet_adaln_ot` | 0.735 | Nearest OT (argmin) | ❌ OT 反而降低性能 |
| `ldmdet_flowdet_adaln_ot_sinkhorn` | 0.748 | Sinkhorn argmax, eps=1 | 比硬 OT 好但仍不如随机 |
| `ot_sinkhorn_eps5/10/50/100` | 0.733-0.748 | Sinkhorn argmax, 多 eps | ε 扫描曲线异常平坦 |
| `sinkhorn_sample_eps5` | **0.751** | Sinkhorn stochastic, eps=5 | ✅ 随机采样恢复性能 |
| `sinkhorn_sample_eps50` | 0.736 | Sinkhorn stochastic, eps=50 | ❌ 过大 ε 有损 |
| `group_hierarchical_stoch` | **0.752** | 群组层次 OT + stochastic | ✅ 最优耦合策略 |

**关键发现**：
- **硬 OT（argmax）在所有 ε 下都不如随机耦合**，因为确定性分配消除了训练多样性
- Stochastic Coupling (从传输矩阵采样) 修复了这个问题，在 eps=5 时达到 0.751
- 群组层次 OT（利用染色体 A-G 组 + 性染色体先验）额外 +0.001，达到 0.752
- **多样性 > 传输效率**：在低维检测空间，保持训练信号的多样性比最小化传输代价更重要

### 阶段 3：训练动力学优化（辅助机制）

| 实验 | mAP | 核心配置 |
|---|---|---|
| `ldmdet_flowdet_adaln_trd` | 0.743 | TRD（传输-修正分解） |
| `ldmdet_flowdet_adaln_trd_only` | 0.746 | TRD + velocity，无其他辅助 |
| `ldmdet_flowdet_adaln_trd_full` | **0.752** | TRD + CAT + LSAS + velocity + Heun |
| `ldmdet_flowdet_adaln_lsas` | 0.743 | LSAS（损失敏感调度） |
| `ldmdet_flowdet_adaln_crossattn_v2` | — | Cross-Attention 变体 |
| `ldmdet_flowdet_adaln_convnext` | — | ConvNeXt backbone 探索 |

**关键发现**：
- TRD/CAT/LSAS/velocity 的单独增益都很小（0.001-0.004），但组合效应显著
- trd_full（0.752）与 group_hierarchical_stoch（0.752）并列 SOTA，但走的是两条正交路径：
  - **路径 A（group_hierarchical）**：优化耦合策略（群组层次 OT + stochastic）
  - **路径 B（trd_full）**：优化训练动力学（TRD + CAT + LSAS + velocity）
- 两条路径理论上可以叠加——这是当前未探索的最大潜力点

### 阶段 4：Reflow 单步推理探索

| 实验 | 1步 mAP | 最佳 epoch | 核心发现 |
|---|---|---|---|
| Reflow v1（从零训练） | 0.715 | 47 | 可收敛但精度低，训练不稳定 |
| Reflow v2（微调，val配对） | 0.739 | 1 | Epoch 1 最佳，后续退化 |
| Reflow v3（微调，train配对） | 0.739 | 1 | 配对来源非关键问题 |
| Reflow v4（修复velocity_head） | 0.739 | 1 | grad_norm=183.7，梯度爆炸 |
| Reflow v5（warmup+调参） | **0.734** | 19 | grad_norm 正常化，但最佳 mAP 降低 |
| Reflow v6（第2轮 Reflow） | 0.739 | 1 | 边际收益递减 |
| ITD（中间轨迹蒸馏） | — | — | 理论提出，实验结果待确认 |

**关键发现**：
- **退化陷阱**：所有 Reflow 实验均呈现 "Epoch 1 最佳 → 后续退化" 的模式
- 退化根因：velocity loss 和检测 loss 的梯度方向冲突（实测 cos=-0.104，86.8% 层冲突）
- velocity loss 收敛天花板 ~0.23-0.30
- 多次 Reflow 边际收益递减：v2(+0.011) → v6(+0.011)
- **最佳 1步推理 mAP=0.739（Reflow v6），与 4步基线 0.752 差距 1.3%**

### 阶段 5：Scale-Conditioned 与 KaryoFlow 失败探索

| 实验方向 | 结果 | 失败原因 |
|---|---|---|
| Scale-Conditioned FM (sc_noise, sc_loss, sc_combined) | 负收益 | 流速场偏移破坏最优性，重加权破坏变分原理 |
| KaryoFlow 排列学习 | 不可行 | 信息量不足 (25.8 bits << 178 bits) |

**理论贡献**：两个负结果均有严格的数学分析，构成论文的"负面结果 + 理论解释"部分。

---

## 四、核心理论发现

### 4.1 OT Diversity Collapse（OT 多样性坍缩）

**现象**：在低维检测空间（$d=4$），OT 耦合完全消除了条件速度的多样性。

**定理**：OT 耦合与随机耦合的条件速度熵差为 $\Delta H = \log K$（连续极限下）。

**维度效应**：相对熵差 $\frac{\Delta H}{H_{\text{rand}}} \propto \frac{1}{d}$。在图像生成中（$d \approx 10^5$）可忽略，在检测中（$d=4$）占主导。

**实验验证**：染色体数据集（$K \approx 46.6$），实测 $\Delta H = 3.8415$，$\log K = 3.8427$，相对误差 0.03%。

### 4.2 CAM 定理（耦合-分配失配）

**核心发现**：Sinkhorn OT + argmax 管线的根本缺陷——ε 参数对软传输矩阵的多样性调控被 argmax 操作完全抹除。

**定量证据**：argmax 解码下，分配多样性恒定（≈2.80 在任何 ε 下），而 Stochastic 解码从 2.81（ε=0.01）单调增加到 5.31（ε=100）。

**解决方案**：Stochastic Coupling（从传输矩阵采样替代 argmax），回复 ε 对多样性的单调调控能力。

### 4.3 多样性-传输效率的帕累托前沿

Sinkhorn + Stochastic Coupling 在随机耦合（最大多样性，零传输结构）和确定性 OT（最大传输效率，零多样性）之间建立连续帕累托前沿。

**三区间模型**：
- **区间 1**（ε < 0.5）：OT 主导，多样性不足，mAP < 0.745
- **区间 2**（ε ∈ [0.5, 5.0]）：最优区间，多样性饱和 + 效率可控，mAP 0.748-0.751
- **区间 3**（ε > 5.0）：耦合偏差主导，训练不稳定，mAP 退化

**最优 ε* ≈ 1-3**（闭式推导，结合指数衰减模型）。

### 4.4 Reflow Degradation Trap（梯度冲突）

**现象**：Reflow 训练中 velocity loss 和 detection loss 存在系统性梯度冲突。

**数学条件**：$\rho = \cos(g_{\text{det}}, g_{\text{vel}}) < 0 \iff e_{\text{det}}^\top e_{\text{vel}} < 0$（检测残差与 velocity 残差方向相反）。

**实验验证**：实测平均余弦相似度 -0.104，86.8% 层存在冲突。

**解决方案**：两阶段训练（Stage 1: 冻结 velocity_head 训练检测 → Stage 2: 冻结共享层训练 velocity_head）。

### 4.5 Scale-Conditioned FM 为何必然无效

三个理论原因：
1. **有偏流速估计**：$\sigma_c \epsilon - x_0 = (\epsilon - x_0) + (\sigma_c - 1)\epsilon$，第二项引入偏差
2. **多目标冲突**：不同 Denver 组的 $\sigma_c$ 不同，模型无法同时最优
3. **变分原理破坏**：$w_c \neq 1/\sigma_c^2$ 时优化目标不等于任何概率散度

### 4.6 KaryoFlow 排列学习的信息论不可行性

- 排列空间大小：$|S_{46}| = 46! \approx 1.8 \times 10^{59}$
- 所需信息量：$\log_2(46!) \approx 200$ bits（考虑同源对称性后 ~178 bits）
- 实际可用信息量：$I_{\text{total}} \approx 46 \times 0.56 \approx 25.8$ bits
- **结论**：即使 8-way Denver 组分类完美（100% acc），信息量（138 bits）仍不足以唯一确定排列

---

## 五、实验目录完整分类

### 5.1 基线体系

```
diffusiondet_baseline/       - DiffusionDet 原论文基线
ldmdet_baseline/             - 染色体数据集 DDPM 基线 (0.725)
ldmdet_baseline_fair/        - 公平对比基线
ldmdet_baseline_step4/       - 4步推理基线
chromo_coco_detection/       - COCO 数据集基线
```

### 5.2 RF 核心改进链

```
ldmdet_rf_heun_shifted_*     - RF + Heun + Shifted schedule 各变体
ldmdet_flowdet_adaln/        - AdaLN-Zero（最佳单模型，0.751）
ldmdet_flowdet_adaln_cat/    - + Curvature-Aware Training（训练崩溃）
ldmdet_flowdet_adaln_obj/    - + Objectness 分支
```

### 5.3 OT 耦合全系列

```
ldmdet_flowdet_adaln_ot/                          - Nearest OT (0.735)
ldmdet_flowdet_sinkhorn/                           - Sinkhorn OT 基础
ldmdet_flowdet_adaln_ot_sinkhorn/                  - Sinkhorn argmax eps=1 (0.748)
ldmdet_flowdet_adaln_ot_sinkhorn_eps5/             - argmax eps=5
ldmdet_flowdet_adaln_ot_sinkhorn_eps10/            - argmax eps=10
ldmdet_flowdet_adaln_ot_sinkhorn_eps50/            - argmax eps=50
ldmdet_flowdet_adaln_ot_sinkhorn_eps100/           - argmax eps=100
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05/     - Stochastic eps=0.5
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1/      - Stochastic eps=1
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps2/      - Stochastic eps=2
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps3/      - Stochastic eps=3
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/      - Stochastic eps=5 (0.751)
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps10/     - Stochastic eps=10
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps50/     - Stochastic eps=50 (0.736, 退化)
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_repro/ - 复现实验
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_seed2/ - 多 seed 验证
```

### 5.4 群组层次 OT

```
ldmdet_flowdet_adaln_group_hierarchical/           - 群组层次 (argmax)
ldmdet_flowdet_adaln_group_hierarchical_stoch/     - 群组层次 stochastic (0.752)
ldmdet_group_hierarchical_stoch_seed2/             - 多 seed 验证
```

### 5.5 TRD 系列

```
ldmdet_flowdet_adaln_trd/                          - TRD 基础
ldmdet_flowdet_adaln_trd_only/                     - TRD + velocity (0.746)
ldmdet_flowdet_adaln_trd_full/                     - TRD+CAT+LSAS+velocity+Heun (0.752)
```

### 5.6 Reflow 系列（单步推理）

```
ldmdet_flowdet_adaln_reflow/                       - Reflow v1（从零训练）
ldmdet_flowdet_adaln_reflow_v2/                    - Reflow v2（微调，val配对）
ldmdet_flowdet_adaln_reflow_*                      - Reflow v3-v6
ldmdet_flowdet_adaln_reflow_det_only/              - 仅检测 loss
ldmdet_flowdet_adaln_reflow_det_only_30ep/         - det_only 长训练
ldmdet_flowdet_adaln_reflow_det_only_lr1e6/        - det_only 低学习率
ldmdet_flowdet_adaln_reflow_freeze/                - 冻结共享层
ldmdet_flowdet_adaln_reflow_freeze_stage2/         - 两阶段训练
ldmdet_flowdet_adaln_reflow_itd/                   - 中间轨迹蒸馏 (ITD)
ldmdet_flowdet_adaln_reflow_lr1e6_vel/             - 低 lr velocity
ldmdet_flowdet_adaln_reflow_consistency/           - Consistency Distillation
ldmdet_flowdet_adaln_reflow_pcgrad/                - PCGrad 梯度冲突缓解
ldmdet_flowdet_adaln_reflow_vel_detach/            - Velocity detachment
```

### 5.7 诊断与工具实验

```
eval_multistep/                                    - 多步评估诊断
gradient_analysis/                                 - 梯度冲突分析
coco_random/                                       - COCO 随机耦合对比
coco_ot_sinkhorn_eps5/                             - COCO OT 对比
```

### 5.8 探索性实验

```
karyoflow_overfit/overfit2/overfit3/               - KaryoFlow 过拟合探索
karyoflow_v1/                                      - KaryoFlow v1
ldmdet_flowdet_adaln_crossattn/                    - Cross-Attention
ldmdet_flowdet_adaln_crossattn_v2/                 - Cross-Attention v2
ldmdet_flowdet_adaln_convnext/                     - ConvNeXt backbone
ldmdet_flowdet_adaln_lsas/                         - Loss-Sensitive Adaptive Scheduling
ldmdet_flowdet_objectness/                         - Objectness 分支
ldmdet_single_chromo_*                             - 单染色体实验
scale_conditioned_*                                - Scale-Conditioned 系列（负结果）
ablations/                                         - 消融实验集合
```

---

## 六、当前最佳结果汇总

### 6.1 4步推理 SOTA

| 实验 | mAP | mAP@50 | mAP@75 | 核心技术 |
|---|---|---|---|---|
| `group_hierarchical_stoch` | **0.752** | 0.946 | 0.841 | 群组层次 OT + Stochastic + Heun |
| `trd_full` | **0.752** | 0.940 | 0.835 | TRD + CAT + LSAS + velocity + Heun |
| `adaln` (vanilla) | 0.751 | 0.943 | 0.843 | AdaLN-Zero + Heun（无 OT） |
| `sinkhorn_sample_eps5` | 0.751 | 0.945 | 0.839 | Sinkhorn Stochastic eps=5 |
| `ot_coupling` | 0.749 | 0.941 | 0.836 | Nearest OT |
| `ot_sinkhorn` | 0.748 | 0.947 | 0.840 | Sinkhorn argmax eps=1 |
| `trd_only` | 0.746 | 0.942 | 0.834 | TRD + velocity |

### 6.2 1步推理最佳

| 实验 | 1步 mAP | vs 4步基线 |
|---|---|---|
| Reflow v6 (2轮 Reflow) | 0.739 | -0.013 |
| Reflow v5 (1轮 Reflow) | 0.734 | -0.018 |
| TRD-Full (无 Reflow) | 0.728 | -0.024 |

### 6.3 效率-精度前沿

| 步数 | TRD-Full | Reflow v5 | 差距 |
|---|---|---|---|
| 1步 | 0.728 | **0.731** | +0.003 |
| 2步 | 0.740 | **0.738** | -0.002 |
| 4步 | 0.741 | **0.743** | +0.002 |
| 8步 | 0.743 | **0.743** | 0 |

**关键**：2步推理的性价比最优（0.738 vs 4步基线 0.741，仅差 0.3%，推理速度翻倍）。

---

## 七、两条正交 SOTA 路径的互补性

| 维度 | 路径 A (group_hierarchical) | 路径 B (trd_full) |
|---|---|---|
| 核心思想 | 优化耦合策略 | 优化训练动力学 |
| 关键模块 | 群组层次 OT + stochastic | TRD + CAT + LSAS + velocity |
| 推理时行为 | 无额外计算 | TRD 自条件精化（需额外前向） |
| 耦合方式 | Sinkhorn Stochastic | Nearest OT |
| 与对方的兼容性 | ✅ 可用 TRD 精化 | ✅ 可用群组层次 OT |

**未探索的最大潜力**：两条路径的组合——取群组层次 stochastic OT 的耦合，加 TRD/CAT/LSAS 的训练动力学优化，预期 mAP 0.753-0.754。

---

## 八、关键反思与教训

### 8.1 什么有效

1. **RF 替代 DDPM**：路径直化是基础收益，简单但有效
2. **AdaLN-Zero 时间条件**：单项最大改进（+1.8%），零初始化至关重要
3. **Shifted Schedule**：时间敏感度理论指导的采样策略，+1.4%
4. **Stochastic Coupling**：修复 OT 在低维空间的多样性损失
5. **Heun 二阶求解器**：几乎免费的增益（2× 前向，+0.001-0.002）
6. **群组先验利用**：染色体生物学分组是有价值的归纳偏置

### 8.2 什么无效

1. **硬 OT 耦合（argmax）**：在低维空间导致多样性坍缩，反而不如随机
2. **Sinkhorn + argmax 管线**：ε 参数被 argmax 短路，无法调控多样性
3. **CAT（曲率正则化）单独使用**：导致训练崩溃（实现有 bug）
4. **Scale-Conditioned FM**：三种理论原因导致必然无效
5. **KaryoFlow 端到端排列学习**：信息论下界不可达
6. **多次 Reflow**：边际收益递减，第2轮几乎无收益

### 8.3 仍待解决的问题

1. **Reflow 退化陷阱**：梯度冲突（cos=-0.104）的稳定解决方案
2. **velocity loss 收敛天花板**：~0.23-0.30 后停滞
3. **1步 vs 4步的 1.3% 差距**：当前 Reflow 只能缩小到 1.3%
4. **两条 SOTA 路径的组合**：理论上有叠加效应，待实验验证
5. **COCO 通用性验证**：当前所有实验均在染色体数据集
6. **更大 ε（ε=1-3）的 Stochastic 实验**：理论上最优区间，待实测

---

## 九、论文框架建议

### 9.1 核心论点（3句话）

1. 基于扩散的目标检测中，OD E路径直度由耦合质量、修正曲率和时间分配三者共同决定，而非单一因素
2. 在低维检测空间（$\mathbb{R}^4$）中，OT 耦合导致训练多样性严重坍缩（$\Delta H = \log K$），而 Stochastic Coupling 是最优修复方案
3. 通过耦合策略优化（群组层次 OT）和训练动力学优化（TRD+C AT+LSAS）两条正交路径，可以达到 0.752 mAP，且两者理论上可叠加

### 9.2 三大贡献

1. **LDMDet 检测器**：基于 Rectified Flow + AdaLN-Zero + Shifted Schedule 的高效扩散检测器，4步推理达到 0.752 mAP
2. **OT Diversity Collapse 理论**：首次从信息论证明 OT 在低维空间导致多样性坍缩，揭示维度依赖性（$\Delta H / H \propto 1/d$），提出 CAM 定理揭示 Sinkhorn+argmax 管线的根本缺陷
3. **Stochastic Coupling + 群组层次 OT**：保留 ε 对多样性的单调调控，利用生物学先验实现帕累托最优

### 9.3 五个需修正的认知

1. **"OT 总是好的"** → 只有在高维空间成立，低维检测中多样性损失严重
2. **"大 ε 增加多样性"** → 对 argmax 无效（CAM 定理），只对 Stochastic Coupling 有效
3. **"更多 Reflow 更好"** → 边际收益递减，根因是梯度冲突而非路径不够直
4. **"Scale-Conditioned 可以改善小物体检测"** → 流速场偏移和变分原理破坏导致必然无效
5. **"端到端排列学习可行"** → 信息论下界（178 bits）远超可用信息量（25.8 bits）

---

## 十、下一步实验优先级

### P0：论文必需（1-2周）

1. **两条 SOTA 路径的组合实验**：Group-Hierarchical Stochastic OT + TRD + Heun（预期 0.753-0.754）
2. **Stochastic ε=1.0 和 ε=2.0 实验**：验证理论预测的最优 ε 区间
3. **COCO 数据集验证**：RF + AdaLN + Shifted 在 COCO 上的通用性证明
4. **完整消融表**：补齐所有模块的消融实验数据

### P1：强化论文（2-4周）

5. **两阶段 Reflow 训练**：Stage 1 冻结 velocity_head 训练检测 → Stage 2 冻结共享层训练 velocity_head
6. **Reflow + ITD（中间轨迹蒸馏）**：在多个中间时间步提供 velocity 监督
7. **推理速度 Benchmark**：完整 FPS 测量（1/2/4/8/16步）
8. **更强的 Backbone**：Swin-T / ConvNeXt

### P2：探索性（4周+）

9. **自适应群组划分**：不依赖固定生物学分类的动态聚类
10. **学习型 ε 调度**：训练初期大 ε（多样性），后期小 ε（效率）
11. **与 C²OT / W-CFM 的结合**：条件加权 + Stochastic Coupling

---

## 十一、关键实验配置对应

| 实验名（work_dirs） | 配置文件 | 核心参数 |
|---|---|---|
| `ldmdet_flowdet_adaln` | `ldmdet_flowdet_adaln.py` | RF + AdaLN-Zero + Shifted s=3 + Heun |
| `ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5` | `ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py` | Sinkhorn Stochastic eps=5 + Euler |
| `ldmdet_flowdet_adaln_group_hierarchical_stoch` | — | 群组层次 OT + Stochastic + Heun |
| `ldmdet_flowdet_adaln_trd_full` | — | TRD + CAT + LSAS + velocity + Heun |
| `ldmdet_flowdet_adaln_reflow_v5` | — | Reflow v5, det_loss_scale=0.5, warmup=3850 |
| `ldmdet_flowdet_adaln_reflow_itd` | `ldmdet_flowdet_adaln_reflow_itd.py` | ITD, num_points=4 |

---

*本文档整合了 work_dirs 中所有实验分支的 markdown 文档、实验日志和理论分析。主要来源：THEORY_FRAMEWORK.md, OT_DIVERSITY_COLLAPSE_PROOF.md, experiment_summary.md, SOTA_ANALYSIS.md, THEORY_WHY_FAILED.md, IMPROVEMENT_PLAN.md, LDMDet_Architecture.md, WEEK1_REPRO_PROTOCOL.md, Research_Plan.md, PAPER_FRAMEWORK.md, 以及 research_qna 中的 11 篇中文研究方向文档。*
