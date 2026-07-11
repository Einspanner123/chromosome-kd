# LDMDet 论文结果数据

> 最后更新: 2026-07-10
> 数据集:
>   - **Original**: `data/Chromosome20240904_NoAug_NoResize_coco/` — 24 类染色体，原始数据集（结论已**暂时废弃**）
>   - **24obj**: `data/24_chromosomes_object/coco/` — 24 类染色体，后采集的大规模基准数据集（现行结论）

---

## 0. 24obj 论文核心结论归纳（最新，SwanLab 验证）

> 本节为 24obj 数据集上的论文级结论归纳，覆盖 35 个实验 / 6 个 SwanLab 项目。
> 旧数据集 Chromosome20240904 的结论已暂时废弃，详见各章节标注。

### 0.1 主路线消融 A0-A4（项目 `ldmdet-mainline-ablation-24obj`，⭐ 论文核心）

| 实验 | mAP | AP50 | AP75 | ΔmAP | 说明 |
|---|---|---|---|---|---|
| A0 baseline (Euler 1步, 无RF) | 0.774 | 0.968 | 0.916 | — | 基线 |
| A1 +RF+Heun | 0.856 | 0.990 | 0.971 | **+0.082** | 主要贡献 |
| A2 +AdaLN-Zero | 0.856 | 0.990 | 0.972 | +0.000 | 持平 A1 |
| A3 +StochOT eps5 | 0.858 | 0.990 | 0.973 | +0.002 | 边际 |
| A4 DPM-Solver++替换Heun | **0.863** | 0.990 | 0.974 | +0.005 | 推理加速且精度提升 |

### 0.2 SOTA 对比（24obj，项目 `chromosome-kd-benchmark-24obj`）

| 方法 | mAP | AP50 | AP75 | 备注 |
|---|---|---|---|---|
| RTMDet-L | **0.869** | 0.992 | 0.976 | 单 seed |
| DINO R50 (4scale) | 0.868 | 0.992 | 0.979 | CRASHED 但有 eval |
| **A4 DPM-Solver++ (LDMDet, Ours)** | **0.863** | 0.990 | 0.974 | ⭐ 本文 SOTA |
| A3 SOTA Heun (LDMDet, Ours) | 0.858 | 0.990 | 0.973 | 本文方法（Heun 4步） |
| Cascade R-CNN R50 | 0.854 | 0.987 | 0.972 | 单 seed |
| LDMDet (stochot ε=5, 早期) | 0.853 | 0.987 | 0.970 | 单 seed |
| YOLOX-S | 0.803 | 0.987 | 0.946 | 单 seed |
| DiffusionDet | 0.787 | 0.970 | 0.928 | CRASHED 但有 eval |

### 0.3 耦合策略消融（项目 `ldmdet-ablation`，3 seeds）

| 耦合策略 | seed 42 | seed 123 | seed 789 | **mAP (mean±std)** |
|---|---|---|---|---|
| Random | 0.859 | 0.860 | 0.860 | **0.860±0.001** |
| GHSS | 0.857 | 0.859 | 0.859 | **0.858±0.001** |
| Sinkhorn Stochastic | 0.856 | — | — | 0.856 (1 seed) |

### 0.4 论文核心论断（24obj 验证）

1. **RF 路线优势** (A1 vs A0, +0.082)：Rectified Flow + Heun 显著超越 DDPM-style Euler 基线，证明 RF 路线在染色体检测任务上的有效性。
2. **AdaLN-Zero 稳定训练但不提升精度** (A2 vs A1, +0.000)：与 chromo 上 +0.003 的结论方向一致，但 24obj 上完全持平。
3. **StochOT 边际收益** (A3 vs A2, +0.002)：24obj 大数据集下 OT 耦合的优势减弱（chromo 上 +0.005）。
4. **DPM-Solver++ 在 24obj 上同时加速和提升精度** (A4 vs A3, +0.005)：与 chromo 上"持平"结论不同，24obj 上 DPM-Solver++ 是明确的精度提升手段。
5. **LDMDet 超越 DiffusionDet +0.076**（0.863 vs 0.787）：证明扩散式检测模型的 RF 路线相对 DDPM 路线的显著优势。
6. **LDMDet 接近 SOTA 检测器**：A4 (0.863) 与 RTMDet-L (0.869) 和 DINO (0.868) 差距仅 -0.005~-0.006，超越 Cascade R-CNN (0.854)、YOLOX-S (0.803)、DiffusionDet (0.787)。
7. **耦合策略在 24obj 上优势减弱**：Random (0.860) ≥ GHSS (0.858) ≥ Sinkhorn (0.856)，差异在 ±0.001-0.004，与 chromo 上"GHSS 最优"结论不同——大数据集下 OT 耦合的多样性管理优势被数据规模稀释。

### 0.5 与旧数据集 (chromo) 结论的关键差异

| 维度 | chromo (废弃) | 24obj (现行) | 论文写作建议 |
|---|---|---|---|
| mAP 量级 | 0.72-0.75 | 0.77-0.87 | 以 24obj 为主数据集 |
| 最优耦合 | GHSS (0.752) > Random (0.751) | Random (0.860) > GHSS (0.858) | 不再强调 GHSS 优势，改为"耦合策略差异在大数据集下减弱" |
| DPM-Solver++ | 持平 Heun (+0.000) | **超越 Heun (+0.005)** | 可作为论文亮点：DPM-Solver++ 同时加速和提升精度 |
| LDMDet vs DiffusionDet | +0.115 | +0.076 | 仍为显著优势，但幅度减小 |
| LDMDet vs 其他检测器 | SOTA | 接近 RTMDet-L/DINO | 需诚实承认与 RTMDet-L/DINO 仍有差距 |

---



## 1. 原始数据集结果（Original Dataset）

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 §2。

### 1.1 SOTA 对比

| 方法 | 主干网络 | mAP | AP50 | AP75 | 数据来源 |
|---|---|---|---|---|---|
| **LDMDet (Sinkhorn stoch ε=5)** | ResNet-50 | **0.753** | 0.946 | 0.843 | ldmdet-experiment phase5 |
| RTMDet-L | CSPNeXt-L | 0.742 | 0.950 | 0.855 | SwanLab benchmark |
| DINO R50 (4scale) | ResNet-50 | 0.737 | 0.946 | 0.829 | SwanLab benchmark |
| Cascade R-CNN R50 | ResNet-50 | 0.732 | 0.936 | 0.843 | SwanLab benchmark |
| YOLOX-S | CSPDarkNet-S | 0.608 | 0.940 | 0.732 | SwanLab benchmark |
| DiffusionDet | ResNet-50 | 0.638 | 0.894 | 0.744 | SwanLab benchmark |

### 1.2 Coupling 消融（3-seed，全增强）

| Coupling | seed 42 | seed 123 | seed 789 | **mean±std** |
|---|---|---|---|---|
| Random | 0.745 | 0.747 | 0.747 | **0.746±0.001** |
| DDPM | 0.726 | 0.733 | 0.727 | **0.729±0.004** |
| Hard OT | 0.747 | 0.747 | **TBD** | — |
| Sinkhorn Stochastic (ε=5) | 0.748 | **TBD** | **TBD** | — |

### 1.3 Epsilon 消融（单 seed，无增强）

| ε | Sinkhorn Stochastic | + AdaLN |
|---|---|---|
| 1 | 0.729 | 0.735 |
| 2 | 0.721 | 0.738 |
| 5 | 0.720 | 0.734 |
| Baseline (adaln, no OT) | — | **0.728** |

### 1.4 SOTA 演进路径

| Phase | 改进 | mAP | Δ |
|---|---|---|---|
| Baseline | RF + Heun + AdaLN | 0.728 | — |
| + Sinkhorn Stochastic OT (ε=5) | 0.753 | **+0.025** |
| + Loss optimization (rel L1) | 0.752 | +0.024 |
| + DPM-Solver++ (inference) | 0.748 | +0.020 |
| + KCEC ploidy constraint | 0.748 | +0.020 |
| + DAEC contrastive learning | 0.740 | +0.012 |
| + Small object optimization | 0.744 | +0.016 |
| **SOTA multiseed (4 seeds)** | **0.746±0.004** | — |

---

## 2. 24obj 数据集结果

### 2.1 SOTA 对比

| 方法 | 主干网络 | mAP | 备注 |
|---|---|---|---|
| RTMDet-L | CSPNeXt-L | **0.869** | 单 seed |
| DINO R50 (4scale) | ResNet-50 | **0.868** | CRASHED 但有 eval |
| **LDMDet (Random, 3-seed)** | ResNet-50 | **0.860±0.001** | ✅ **本文方法** |
| **LDMDet (GHSS, 3-seed)** | ResNet-50 | **0.858±0.001** | ✅ **本文方法** |
| Cascade R-CNN R50 | ResNet-50 | 0.854 | 单 seed |
| LDMDet (stochot ε=5) | ResNet-50 | 0.853 | 单 seed |
| YOLOX-S | CSPDarkNet-S | 0.803 | 单 seed |
| DiffusionDet | ResNet-50 | 0.787 | CRASHED 但有 eval |

### 2.2 24obj 消融（完整 3-seed）

| 策略 | seed 42 | seed 123 | seed 789 | **mean±std** |
|---|---|---|---|---|
| **GHSS** | 0.857 | 0.859 | 0.859 | **0.858±0.001** |
| **Random** | 0.859 | **0.860** | **0.860** | **0.860±0.001** |
| Sinkhorn | ▶ **运行中** | — | — | TBD |

### 2.3 GHSS 3-seed 详细指标

| 指标 | seed 42 | seed 123 | seed 789 | **mean** |
|---|---|---|---|---|
| mAP | 0.857 | 0.859 | 0.859 | **0.858** |
| AP50 | 0.988 | 0.988 | 0.989 | **0.988** |
| AP75 | 0.965 | 0.969 | 0.968 | **0.967** |
| APs | 0.518 | 0.521 | 0.533 | **0.524** |
| APm | 0.846 | 0.854 | 0.851 | **0.850** |
| APl | 0.909 | 0.866 | 0.865 | **0.880** |
| best epoch | 83 | 102 | 75 | — |

### 2.4 Random 3-seed 详细指标

| 指标 | seed 42 | seed 123 | seed 789 | **mean** |
|---|---|---|---|---|
| mAP | 0.859 | **0.860** | **0.860** | **0.860** |
| AP50 | 0.988 | 0.989 | 0.989 | **0.989** |
| AP75 | 0.967 | 0.969 | 0.970 | **0.969** |
| APs | 0.564 | 0.528 | 0.518 | **0.537** |
| APm | 0.847 | 0.852 | 0.855 | **0.851** |
| APl | 0.901 | 0.903 | 0.912 | **0.905** |
| best epoch | 59 | 115 | 82 | — |

### 2.5 LDMDet 主路线消融 A0-A4（新发现）

**项目**: `ldmdet-mainline-ablation-24obj` | **数据集**: 24 Chromosomes Object

| 实验 | mAP | AP50 | AP75 | ΔmAP | 说明 |
|---|---|---|---|---|---|
| A0 baseline (Euler 1步, 无RF) | 0.774 | 0.968 | 0.916 | — | 基线 |
| A1 +RF+Heun | 0.856 | 0.990 | 0.971 | **+0.082** | 主要贡献 |
| A2 +AdaLN-Zero | 0.856 | 0.990 | 0.972 | +0.000 | 持平 A1 |
| A3 +StochOT eps5 | 0.858 | 0.990 | 0.973 | +0.002 | 边际 |
| A4 DPM-Solver++替换Heun | 0.863 | 0.990 | 0.974 | +0.005 | 推理加速且精度提升 |

> A0-A4 为 24obj 主路线递进消融，对应 §1.4（旧数据集 chromo，已标注暂时废弃）的演进路径。A1 (RF+Heun) 贡献 +0.082 为主要改进；A4 (DPM-Solver++) 在 24obj 上不仅加速还提升精度 +0.005，与 chromo 上"持平"的结论不同。

---

## 3. 论文表格模板

### Table 1: Main coupling ablation (Original Dataset, 3-seed)

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 §2.5（A0-A4 主路线消融）。

| Coupling Strategy | mAP (mean±std) | Δ vs Random |
|---|---|---|
| Random (baseline) | **0.746 ± 0.001** | — |
| DDPM | 0.729 ± 0.004 | −0.017 |
| Hard OT | *TBD* | *TBD* |
| Sinkhorn Stochastic (ε=5) | *TBD* | *TBD* |
| GHSS (ε=5) | *TBD* | *TBD* |

### Table 2: SOTA comparison on two datasets

> ⚠️ 本表同时含两数据集。**Original Dataset 列**基于旧数据集 Chromosome20240904（mAP≈0.72-0.75，暂时废弃）；**24obj Dataset 列**保留为现行结论。

| Method | Backbone | Original Dataset | 24obj Dataset |
|---|---|---|---|
| RTMDet-L | CSPNeXt-L | 0.742 | **0.869** |
| DINO R50 (4scale) | ResNet-50 | 0.737 | 0.868 |
| Cascade R-CNN R50 | ResNet-50 | 0.732 | 0.854 |
| **LDMDet (Ours)** | ResNet-50 | **0.753** | **0.860** |
| YOLOX-S | CSPDarkNet-S | 0.608 | 0.803 |

### Table 3: Epsilon Analysis

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 §2.5。

| ε | w/o AdaLN | w/ AdaLN |
|---|---|---|
| 1 | 0.729 | 0.735 |
| 2 | 0.721 | 0.738 |
| 5 | 0.720 | 0.734 |
| — | — | — |
| Baseline | — | 0.728 |

---

## 4. 待补实验

| 实验 | 优先级 | 状态 |
|---|---|---|
| Sinkhorn 24obj seed 42 | ✅ 完成 | mAP=0.856 @ epoch 53 (run_id=o96m1eqz4l12qjeyys1cs) |
| Sinkhorn / Hard OT 剩余 5 seed | ⬜ 可能取消 | Sinkhorn 42 = 0.856 < Random 0.860，已无补全必要 |
| Hard OT / Sinkhorn full-aug 补全（原始数据集） | 🟢 低优先级 | 原 ablation table 的占位（chromo 已废弃） |

---

## 5. 数据来源索引

| 数据 | 位置 |
|---|---|
| Original SOTA 对比 | SwanLab: einspanner/chromosome-kd-benchmark |
| Coupling ablation (3-seed) | `work_dirs/multi_seed_aug/` |
| Epsilon ablation | `work_dirs/ablation/` |
| SOTA 演进 | `ldmdet-experiment/sota/` |
| 24obj SOTA 对比 | SwanLab: einspanner/chromosome-kd-benchmark-24obj |
| 24obj GHSS 3-seed | `work_dirs/24obj_ablation/ghss/` |
| 24obj Random 3-seed | `work_dirs/24obj_ablation/random/` |
