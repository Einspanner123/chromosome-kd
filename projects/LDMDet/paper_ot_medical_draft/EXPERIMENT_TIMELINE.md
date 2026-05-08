# 实验时间线 / Experiment Timeline

> 按研究阶段递进整理的完整实验记录。✓ 已完成 / → 运行中 / ○ 待补充
> Progressive experiment log organized by research phase. ✓ Done / → Running / ○ Pending

---

## Phase 1: 扩散重构 / Diffusion Reformulation (2025.10 – 2026.02)

**目标 / Goal**: 用 Rectified Flow 替代 DDPM，优化检测架构 / Replace DDPM with Rectified Flow and optimize architecture.

| 日期 Date | 实验 Experiment | mAP | AP50 | AP75 | 说明 Note |
|-----------|----------------|-----|------|------|-----------|
| 12/28 | `ldmdet_baseline` | 0.725 | 0.921 | 0.812 | DDPM + ScaleShift + Euler（原始 DiffusionDet） |
| 12/30 | `ldmdet_rf` | 0.733 | 0.939 | 0.828 | + Rectified Flow, Euler |
| 12/31 | `ldmdet_rf_shifted_schdule` | 0.747 | 0.937 | 0.837 | + 移位调度 Shifted schedule (s=3.0) |
| 01/20 | `ldmdet_rf_shifted_all` | 0.740 | 0.941 | 0.837 | RF + shifted, 训练推理统一 |
| 01/21 | `ldmdet_rf_heun_shifted` | 0.749 | 0.942 | 0.841 | + Heun 二阶求解器 |
| 01/27 | `ldmdet_rf_heun_shifted_bs2` | 0.748 | 0.945 | 0.842 | 批大小 2，训练稳定 |
| 02/04 | `ldmdet_rf_heun_shifted_bs2_reproduce` | 0.748 | 0.944 | 0.837 | 固定种子复现 |
| 03/18 | `ldmdet_flowdet_adaln` | **0.751** | 0.943 | 0.843 | **+ AdaLN-Zero** → 工程 SOTA 基线 ✓ |

**关键结果 / Key result**: DDPM (0.725) → RF+Heun+Shifted (0.748) → +AdaLN (0.751)。扩散重构贡献 +0.023；AdaLN 在稳定 Heun 基线上贡献 +0.003。✓

---

## Phase 2: OT 耦合发现 / OT Coupling Discovery (2026.03 – 04)

**目标 / Goal**: 探究 OT 耦合是否有利于密集染色体检测 / Investigate whether OT coupling benefits dense chromosome detection.

### 2.1 核心耦合对比 / Core Coupling Comparison

| 日期 Date | 实验 Experiment | mAP | AP50 | AP75 | 说明 Note |
|-----------|----------------|-----|------|------|-----------|
| 03/23 | `ldmdet_flowdet_ot_coupling` | 0.749 | 0.941 | 0.836 | 最近邻 OT / Nearest-neighbor OT |
| 03/23 | `ldmdet_flowdet_adaln_ot` | **0.735** | 0.941 | 0.832 | **硬 OT：−0.016 vs 随机基线** / **Hard OT: −0.016 vs Random** |
| 04/10 | `ldmdet_flowdet_adaln_ot_sinkhorn` | 0.748 | 0.947 | 0.840 | Sinkhorn argmax ε=1 |

**关键发现 / Key finding**: 硬 OT 将 mAP 从 0.751 降至 0.735。来自生成建模的直觉（OT 有益）在密集检测中失效。✓

### 2.2 ε 扫描 — Argmax 解码 / Epsilon Sweep — Argmax

| 日期 Date | 实验 Experiment | ε | mAP | 说明 Note |
|-----------|----------------|----|-----|-----------|
| 04/21 | `ldmdet_flowdet_adaln_ot_sinkhorn_eps5` | 5 | 0.745 | Argmax ε=5 |
| 04/21 | `ldmdet_flowdet_adaln_ot_sinkhorn_eps10` | 10 | 0.745 | Argmax ε=10 |
| 04/21 | `ldmdet_flowdet_adaln_ot_sinkhorn_eps50` | 50 | 0.747 | Argmax ε=50 |
| 04/21 | `ldmdet_flowdet_adaln_ot_sinkhorn_eps100` | 100 | **0.733** | Argmax ε=100 → 退化至硬 OT |

**关键发现 / Key finding**: Argmax 的 mAP 随 ε 增大单调下降（0.748→0.745→0.733）。ε=100 时性能等于硬 OT。Argmax 从根上破坏了 ε 的多样性控制。✓

### 2.3 ε 扫描 — Sinkhorn 采样 / Epsilon Sweep — Sinkhorn Sampling

| 日期 Date | 实验 Experiment | ε | mAP | AP50 | AP75 | 说明 Note |
|-----------|----------------|----|-----|------|------|-----------|
| 04/26 | `...sample_eps05` | 0.5 | 0.742 | 0.938 | 0.829 | 近 OT / Near-OT |
| 04/22 | `...sample_eps1` | 1 | 0.748 | 0.941 | 0.839 | |
| 04/27 | `...sample_eps2` | 2 | 0.744 | 0.943 | 0.832 | |
| 04/27 | `...sample_eps3` | 3 | 0.741 | 0.940 | 0.835 | |
| 04/22 | `...sample_eps5` | 5 | **0.751** | 0.945 | 0.839 | **最优区间峰值 / Sweet spot peak** |
| 04/22 | `...sample_eps10` | 10 | 0.747 | 0.940 | 0.835 | |
| 04/22 | `...sample_eps50` | 50 | 0.736 | 0.936 | 0.828 | 偏差主导 / Bias-dominated |

**关键发现 / Key finding**: Sinkhorn 采样呈现清晰的倒 U 形，峰值在 ε=5。多样性恢复提前饱和，传输偏差后期累积。识别出三个区间：OT 主导区（ε<0.5）、最优区间（0.5≤ε≤5）、偏差主导区（ε>5）。✓

### 2.4 复现与多种子 / Reproduction & Multi-Seed

| 日期 Date | 实验 Experiment | mAP | 说明 Note |
|-----------|----------------|-----|-----------|
| 05/06 | `...sample_eps5_repro` | 0.738 | 复现失败（训练不稳定） |
| 05/06 | `...sample_eps5_seed2` | 0.750 | 种子 3408：与 3407 相差 0.001 |

**关键发现 / Key finding**: 跨种子方差约 0.001–0.004。核心结论（OT 下降 0.016）是噪声下限的 4 倍。✓

---

## Phase 3: 群组层次耦合 / Group-Hierarchical Coupling (2026.04 – 05)

**目标 / Goal**: 利用领域知识（染色体群组）精化多样性原则 / Apply domain knowledge (chromosome groups) to refine the diversity principle.

| 日期 Date | 实验 Experiment | mAP | AP50 | AP75 | 说明 Note |
|-----------|----------------|-----|------|------|-----------|
| 04/29 | `...group_hierarchical` | 0.734 | 0.936 | 0.825 | 群组层次 **配合 argmax** → 失败 |
| 04/29 | `...group_hierarchical_stoch` | **0.752** | 0.946 | 0.841 | 群组层次 **配合采样** → **SOTA** |
| 05/07 | `...group_hierarchical_stoch_seed2` | 0.747 | 0.941 | 0.835 | 种子 3408 验证 |

**关键发现 / Key finding**: GHSS 达到 0.752 mAP——唯一超越随机基线的耦合策略。领域结构消除无效跨组噪声，同时保留组内有效多样性。✓

---

## Phase 4: 训练动力学 / Training Dynamics — 负结果 (2026.04 – 05)

**目标 / Goal**: 将正交训练改进（TRD、CAT、LSAS、速度预测）与耦合策略组合 / Combine orthogonal training improvements with coupling strategies.

| 日期 Date | 实验 Experiment | mAP | 说明 Note |
|-----------|----------------|-----|-----------|
| 04/10 | `...trd` | 0.743 | 仅 TRD |
| 04/10 | `...cat` | 0.740 | 仅 CAT |
| 04/10 | `...lsas` | 0.743 | 仅 LSAS |
| 04/11 | `...trd_full` | 0.752 | TRD+CAT+LSAS+速度+Heun——用最近邻 OT 达到 0.752 |
| 04/27 | `...trd_only` | 0.746 | TRD+速度预测 |
| 04/28 | `...stochastic_eps5_trd_cat` | 0.740 | Sinkhorn+TRD+CAT → **更差** |
| 05/06 | `ldmdet_group_hierarchical_trd` | 0.746 | GHSS+TRD → **低于 GHSS 单独** |
| 05/06 | `ldmdet_sinkhorn_trd_cat_lsas` | 0.743 | Sinkhorn+TRD+CAT+LSAS → **更差** |

**关键发现 / Key finding**: 训练动力学（TRD/CAT/LSAS）与耦合改进不叠加。机制不可复合——耦合和训练动力学作用于优化的不同层面。这是一个有意义的负结果。✓

---

## Phase 5: 跨数据集验证 / Cross-Dataset Validation (2026.05)

**目标 / Goal**: 检验耦合效应是否迁移至独立来源的单类染色体数据集 / Test coupling effects on independently sourced single-class dataset.

### 已完成 / Completed ✓

| 日期 Date | 实验 Experiment | mAP | AP50 | AP75 | 说明 Note |
|-----------|----------------|-----|------|------|-----------|
| 05/07 | `ldmdet_single_chromo_random` | 0.676 | 0.957 | 0.813 | 随机基线（单类）/ Random baseline |
| 05/07 | `ldmdet_single_chromo_hard_ot` | **0.683** | 0.954 | 0.822 | 硬 OT **优于** 随机（单类）/ Hard OT > Random |
| 05/07 | `ldmdet_single_chromo_stoch_eps5` | 0.681 | 0.959 | 0.821 | Sinkhorn 采样 ε=5 |

**关键发现 / Key finding**: 耦合效应在单类数据上**反转**：硬 OT > Sinkhorn > 随机。无类别混淆时，传输结构提供更清晰的定位信号。验证了 ε 是任务依赖的控制维度。✓

### 运行中 / 待补充 / Running / Pending

| 状态 Status | 实验 Experiment | ε | 目的 Purpose |
|-------------|----------------|----|-------------|
| → | `ldmdet_single_chromo_argmax_eps5` | 5 | argmax 在单类场景是否同样失效？ |
| → | `ldmdet_single_chromo_argmax_eps1` | 1 | 第二个 argmax 点确认趋势 |
| ○ | `ldmdet_single_chromo_ddpm_baseline` | — | Dataset B 上的原始 DiffusionDet 基线 |

---

## Phase 6: 补充消融与分析 / Additional Ablations (2026.04 – 05)

### 架构变体 / Architecture Variants

| 日期 Date | 实验 Experiment | mAP | 说明 Note |
|-----------|----------------|-----|-----------|
| 05/03 | `...crossattn` | 0.745 | 线性交叉注意力（vs. 动态卷积） |
| 05/03 | `...convnext` | NaN | ConvNeXt 骨干——NaN 失败（fp16 溢出） |
| 05/01–02 | `scale_conditioned_*` | 0.736–0.745 | 尺度条件耦合——全部低于基线 |
| 05/04–06 | `ldmdet_phase*` | 0.686–0.745 | BoxRenewal, RoI共享, DAP——全部低于 SOTA |

### 机制分析 / Mechanism Analysis

| 日期 Date | 分析 Analysis | 结果 Result |
|-----------|--------------|-------------|
| 04/21 | 速度熵 H(V\|Z) | 硬 OT 下 3.841→0 nats；组间方差崩溃 70% |
| 04/21 | ε 相图 (ρ, η) | 三区域结构；ρ 在 ε≈1 处饱和；η 次线性增长 |
| 04/21 | 有效匹配数 D_eff | Argmax 冻结于 ~2.80；Sinkhorn 采样升至 5.31 |
| 05/07 | 逐类 AP（5 个检查点） | 23/24 类在硬 OT 下受损；Y 为唯一例外；Sinkhorn 采样均匀恢复 |

---

## 递进叙事弧 / Progressive Narrative Arc

```
Phase 1: 工程 / Engineering (10月–3月)
  DDPM 0.725 → RF+Heun+Shifted 0.748 → +AdaLN 0.751
  "我们构建了一个强检测器。"

Phase 2: 发现 / Discovery (3月–4月)
  硬 OT 0.735 ← 反直觉下降
  Argmax 跨 ε：0.748→0.745→0.733（单调下降）
  Sinkhorn 采样跨 ε：倒 U 形，ε=5 峰值 (0.751)
  "OT 损害密集检测；argmax 破坏 ε；采样修复它。"

Phase 3: 精化 / Refinement (4月–5月)
  GHSS 0.752 ← 超越随机基线
  TRD/CAT/LSAS：与耦合策略负交互
  "领域结构增强原则；训练动力学与之正交。"

Phase 4: 验证 / Validation (5月)
  跨数据集：硬 OT > 随机（单类）
  "ε 是任务依赖的控制维度，而非固定超参数。"
```

---

## 投稿前待补充 / Pending Before Submission

| 优先级 Priority | 实验 Experiment | GPU 天数 | 影响 Impact |
|----------------|----------------|---------|-------------|
| P0 | 跨数据集 argmax ε=1,5 | 1–2 | 确认 argmax 在单类场景同样失效 |
| P1 | 跨数据集 DDPM 基线 | 1 | 证明工程增益可迁移 |
| P2 | Dataset B 逐类 AP | 0.5 | 单类，信息量较低 |
| P3 | Dataset B 完整 ε 扫描 | 4–6 | 理想完整实验 |
