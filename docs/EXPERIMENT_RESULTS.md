# LDMDet 实验结果汇总（论文数据版本）

> 生成日期: 2026-06-25 | 最近更新: 2026-07-10
> 最新: ✅ 24obj 完整实验汇总完成（SwanLab 35 实验 / 6 项目），新发现 A0-A4 主路线消融
> 数据来源: work_dirs/ (本地 checkpoint) + SwanLab (云端指标) + ldmdet-experiment/ (归档)
> 数据集: 见各表备注

---

## 0. 24obj 核心结论归纳（最新，SwanLab 验证）

> 本节为 24obj 数据集（`data/24_chromosomes_object/coco/`，5000 张图 / 24 类）的最终结论归纳。
> 旧数据集 Chromosome20240904（mAP≈0.72-0.75）的结论已**暂时废弃**，详见各章节的废弃标注。

### 0.1 主路线消融 A0-A4（项目 `ldmdet-mainline-ablation-24obj`，⭐ 核心）

| 实验 | mAP | AP50 | AP75 | ΔmAP | 说明 |
|---|---|---|---|---|---|
| A0 baseline (Euler 1步, 无RF) | 0.774 | 0.968 | 0.916 | — | 基线 |
| A1 +RF+Heun | 0.856 | 0.990 | 0.971 | **+0.082** | 主要贡献 |
| A2 +AdaLN-Zero | 0.856 | 0.990 | 0.972 | +0.000 | 持平 A1 |
| A3 +StochOT eps5 | 0.858 | 0.990 | 0.973 | +0.002 | 边际 |
| A4 DPM-Solver++替换Heun | **0.863** | 0.990 | 0.974 | +0.005 | 推理加速且精度提升 |

**核心发现**：
1. **A1 (RF+Heun) 是主要贡献** (+0.082)，从 0.774 跃升到 0.856
2. **A2 (AdaLN-Zero) 持平 A1** (+0.000)，稳定训练但不提升精度
3. **A3 (StochOT) 边际收益** (+0.002)
4. **A4 (DPM-Solver++) 在 24obj 上不仅加速还提升精度 +0.005**（与 chromo 上"持平"结论不同）

### 0.2 SOTA 对比（项目 `chromosome-kd-benchmark-24obj`）

| 方法 | mAP | AP50 | AP75 | 备注 |
|---|---|---|---|---|
| RTMDet-L | **0.869** | 0.992 | 0.976 | 单 seed |
| DINO R50 (4scale) | 0.868 | 0.992 | 0.979 | CRASHED 但有 eval |
| **A4 DPM-Solver++ (LDMDet)** | **0.863** | 0.990 | 0.974 | ⭐ 本文方法（主路线 SOTA） |
| A3 SOTA Heun (LDMDet) | 0.858 | 0.990 | 0.973 | 本文方法（Heun 4步） |
| LDMDet (Random, 3-seed) | 0.860±0.001 | 0.989 | 0.969 | 本文方法（耦合消融） |
| LDMDet (GHSS, 3-seed) | 0.858±0.001 | 0.988 | 0.967 | 本文方法（耦合消融） |
| Cascade R-CNN R50 | 0.854 | 0.987 | 0.972 | 单 seed |
| LDMDet (stochot ε=5) | 0.853 | 0.987 | 0.970 | 单 seed（早期实验） |
| Sinkhorn Stochastic (LDMDet) | 0.856 | — | — | 单 seed (42) |
| YOLOX-S | 0.803 | 0.987 | 0.946 | 单 seed |
| DiffusionDet | 0.787 | 0.970 | 0.928 | CRASHED 但有 eval |

**核心发现**：
1. **A4 (LDMDet, mAP=0.863) 超越 Cascade R-CNN (0.854) 和 DiffusionDet (0.787)，接近 RTMDet-L (0.869) 和 DINO (0.868)**
2. **LDMDet 相对 DiffusionDet 提升 +0.076**（0.863 vs 0.787），证明 RF 路线相对 DDPM 的优势
3. 耦合策略对比：Random (0.860) > GHSS (0.858) > Sinkhorn Stochastic (0.856)，差异在 ±0.001-0.004 量级

### 0.3 耦合策略消融（项目 `ldmdet-ablation`，3 seeds）

| 耦合策略 | seed 42 | seed 123 | seed 789 | **mAP (mean±std)** |
|---|---|---|---|---|
| Random | 0.859 | 0.860 | 0.860 | **0.860±0.001** |
| GHSS | 0.857 | 0.859 | 0.859 | **0.858±0.001** |
| Sinkhorn Stochastic | 0.856 | — | — | 0.856 (1 seed) |

**核心发现**：24obj 上 **Random ≥ GHSS ≥ Sinkhorn Stochastic**，与 chromo 上"GHSS 最优"结论不同；24obj 大数据集下 OT 耦合的优势减弱。

### 0.4 其他 24obj 实验（项目 `ldmdet-breakthrough` / `ldmdet-frontier-directions` / `few-shot-benchmark`）

| 实验 | mAP | 状态 | 项目 |
|---|---|---|---|
| SC-RF (自条件化RF) | 0.857 | RUNNING | ldmdet-breakthrough |
| h_velocity_loss | 0.856 | CRASHED | ldmdet-frontier-directions |
| n_cascade_e2e | 0.684 | FINISHED | ldmdet-frontier-directions |
| h_cfm_velocity | ~0 | FAILED | ldmdet-frontier-directions |
| FBM CrossAttn (few-shot 源预训练) | 0.857 | CRASHED | few-shot-benchmark |
| FBM SimpleGate (few-shot 源预训练) | 0.677 | CRASHED | few-shot-benchmark |

**核心发现**：
1. **SC-RF (0.857) 与 A3 (0.858) 持平**，自条件化 RF 未带来显著提升
2. **n_cascade_e2e (0.684) 显著退化**，端到端级联策略失败
3. **FBM CrossAttn 源预训练 (0.857) 与 A3 持平**，但 SimpleGate (0.677) 严重退化

### 0.5 与旧数据集 (chromo) 结论的关键差异

| 维度 | chromo (废弃) | 24obj (现行) |
|---|---|---|
| mAP 量级 | 0.72-0.75 | 0.77-0.87 |
| 最优耦合 | GHSS (0.752) > Random (0.751) | Random (0.860) > GHSS (0.858) > Sinkhorn (0.856) |
| DPM-Solver++ | 持平 Heun (+0.000) | **超越 Heun (+0.005)** |
| LDMDet vs DiffusionDet | +0.115 (0.753 vs 0.638) | +0.076 (0.863 vs 0.787) |
| LDMDet vs 其他检测器 | SOTA | 接近 RTMDet-L/DINO，超越 Cascade/YOLOX/DiffusionDet |

---

## 1. 实验设置

### 1.1 数据集

| 名称 | 路径 | 类别数 | 说明 |
|---|---|---|---|
| **Chromosome20240904** | `data/Chromosome20240904_NoAug_NoResize_coco/` | 24 (A1~Y) | 原始染色体检测数据集，COCO 格式 |
| **24 Chromosomes Object** | `data/24_chromosomes_object/coco/` | 24 (A1~Y) | 后采集的 24obj 基准数据集，COCO 格式 |
| **Selfmake Chromosome v2** | `data/selfmake_chromosome202250604_NoResizeNoAug/` | 24 (A1~Y) | 自制染色体数据集 (2022)，chromo v2 用 |

### 1.2 训练超参（所有 LDMDet 实验）

| 参数 | 值 |
|---|---|
| Backbone | ResNet-50 (torchvision 预训练) |
| Neck | FPN (256通道, 4 levels) |
| Proposals | 500 |
| Transformer heads | 6 |
| Deep supervision | ✅ (5 aux heads) |
| Optimizer | AdamW, lr=5e-5, wd=1e-4 |
| LR schedule | Linear warmup 5ep + CosineAnnealing |
| Max epochs | 150 |
| Loss | Focal (cls, 2.0) + L1 (bbox, 5.0) + GIoU (2.0) |
| Matcher | Hungarian (FocalCost 2.0 + L1Cost 5.0 + GIoUCost 2.0) |
| Diffusion | Rectified Flow, shifted schedule (shift=3.0) |
| Solver | Heun (2nd order), 4 sampling steps |
| Batch size | 4 (full-aug), 2 (no-aug) |

### 1.3 数据增强（full-aug 实验）

| 增强 | 配置 |
|---|---|
| RandomFlip | prob=0.5 |
| Multi-scale resize | (480~800) × 1333 |
| Random crop | (384, 600) crop + resize |
| CLAHE | prob=0.5, clip_limit=2.0, tile=8×8 |
| SmallObjectCopyPaste | area_thr=2500, max_pasted=8 |

---

## 2. 主消融实验：Coupling 策略对比

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 24obj 相关章节。

**数据集**: Chromosome20240904 | **增强**: 全增强 | **种子**: 42, 123, 789 | **BS**: 4

| Coupling 策略 | seed=42 | seed=123 | seed=789 | **mAP (mean±std)** | 状态 |
|---|---|---|---|---|---|
| **Random** (baseline) | 0.745 | 0.747 | 0.747 | **0.746 ± 0.001** | ✅ 完成 |
| **DDPM** | 0.726 | 0.733 | 0.727 | **0.729 ± 0.004** | ✅ 完成 |
| **Hard OT** | 0.747 | 0.747 | **TBD** | **TBD** | ⚠️ 缺 seed 789 |
| **Sinkhorn Stochastic** (ε=5) | 0.748 | **TBD** | **TBD** | **TBD** | ❌ 缺 seed 123, 789 |
| **GHSS** (ε=5) | **TBD** | **TBD** | **TBD** | **TBD** | ❌ 全部缺失 |
| **Sinkhorn Argmax** (ε=5) | **TBD** | **TBD** | **TBD** | **TBD** | ❌ 全部缺失 |

### 2.1 待补实验清单

| # | 实验 | 命令 |
|---|------|------|
| 1 | GHSS × 3 seeds | `python experiments/runners/train.py experiments/configs/ldmdet/ghss.py --seed <N> --gpu-id <G> --work-dir work_dirs/multi_seed_aug/ghss/seed_<N>` |
| 2 | Sinkhorn Stochastic seed 123, 789 | `python experiments/runners/train.py experiments/configs/ldmdet/sinkhorn_stochastic.py --seed <N> --gpu-id <G> --work-dir work_dirs/multi_seed_aug/sinkhorn_stochastic/seed_<N>` |
| 3 | Hard OT seed 789 | `python experiments/runners/train.py experiments/configs/ldmdet/hard_ot.py --seed 789 --gpu-id <G> --work-dir work_dirs/multi_seed_aug/hard_ot/seed_789` |
| 4 | Sinkhorn Argmax × 3 seeds | `python experiments/runners/train.py experiments/configs/ldmdet/sinkhorn_argmax.py --seed <N> --gpu-id <G> --work-dir work_dirs/multi_seed_aug/sinkhorn_argmax/seed_<N>` |

---

## 3. 无增强基线

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 24obj 相关章节。

**数据集**: Chromosome20240904 | **增强**: 无 | **种子**: 42, 123, 789 | **BS**: 2 (默认)

| Coupling | seed=42 | seed=123 | seed=789 | **mAP (mean±std)** | 状态 |
|---|---|---|---|---|---|
| **Random** | 0.695 | 0.691 | 0.700 | **0.695 ± 0.004** | ✅ 完成 |
| **Hard OT** | 0.705 | 0.703 | 0.707 | **0.705 ± 0.002** | ✅ 完成 |

> BS=4 variant: Random 0.693 / 0.679 / 0.691 → **0.688 ± 0.006** (亦有完整数据)

---

## 4. SOTA 对比（24obj 基准数据集）

**数据集**: 24 Chromosomes Object | **增强**: 各模型自带

| 方法 | Backbone | mAP | AP50 | AP75 | APs | APm | APl | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| **RTMDet-L** | CSPNeXt-L | **0.869** | — | — | — | — | — | 单 seed |
| **DINO R50 (4scale)** | ResNet-50 | **0.868** | — | — | — | — | — | CRASHED 但有 eval |
| **LDMDet (Random, 3-seed)** | ResNet-50 | **0.860±0.001** | 0.989 | 0.969 | 0.537 | 0.851 | 0.905 | ✅ 本文 |
| **LDMDet (GHSS, 3-seed)** | ResNet-50 | **0.858±0.001** | 0.988 | 0.967 | 0.524 | 0.850 | 0.880 | ✅ 本文 |
| **Cascade R-CNN R50** | ResNet-50 | **0.854** | — | — | — | — | — | 单 seed |
| **LDMDet (RF+AdaLN+stochot)** | ResNet-50 | **0.853** | — | — | — | — | — | 单 seed |
| **YOLOX-S** | CSPDarkNet-S | **0.803** | — | — | — | — | — | 单 seed |
| **DiffusionDet** | ResNet-50 | **0.787** | — | — | — | — | — | CRASHED 但有 eval |

> 注: CRASHED 仍有数据的实验来自 SwanLab 上的 benchmark-24obj 项目。部分模型的详细 AP50/75/s/m/l 待补全。

### 4.1 LDMDet 主路线消融 A0-A4（24obj, 新发现）

**项目**: `ldmdet-mainline-ablation-24obj` | **数据集**: 24 Chromosomes Object

| 实验 | mAP | AP50 | AP75 | ΔmAP | 说明 |
|---|---|---|---|---|---|
| A0 baseline (Euler 1步, 无RF) | 0.774 | 0.968 | 0.916 | — | 基线 |
| A1 +RF+Heun | 0.856 | 0.990 | 0.971 | **+0.082** | 主要贡献 |
| A2 +AdaLN-Zero | 0.856 | 0.990 | 0.972 | +0.000 | 持平 A1 |
| A3 +StochOT eps5 | 0.858 | 0.990 | 0.973 | +0.002 | 边际 |
| A4 DPM-Solver++替换Heun | 0.863 | 0.990 | 0.974 | +0.005 | 推理加速且精度提升 |

> A0-A4 为 24obj 主路线递进消融。A1 (RF+Heun) 贡献 +0.082 为主要改进；A4 (DPM-Solver++) 在 24obj 上不仅加速还提升精度 +0.005（与 chromo 上持平结论不同）。

---

## 5. Epsilon 消融（随机 vs Sinkhorn 采样强度）

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 24obj 相关章节。

**数据集**: Chromosome20240904 | **种子**: 单 seed | **耦合**: Sinkhorn stochastic

### 5.1 无 AdaLN 版本

| ε | best mAP | 状态 |
|---|---|---|
| 1 | **0.729** | ✅ EarlyStop @ ep120 |
| 2 | **0.721** | ✅ EarlyStop @ ep89 |
| 5 | **0.720** | ⚠️ 中断 @ ep64 |

### 5.2 带 AdaLN 版本

| ε | best mAP | 状态 |
|---|---|---|
| 1 | **0.735** | ✅ EarlyStop @ ep106 |
| 2 | **0.738** | ✅ EarlyStop @ ep136 |
| 5 | **0.734** | ✅ EarlyStop @ ep120 |

### 5.3 纯随机 baseline

| 版本 | best mAP | 状态 |
|---|---|---|
| adaln (no OT) | **0.728** | ✅ EarlyStop @ ep121 |
| sota_group_hier_stoch | **0.741** | ✅ FINISHED |

---

## 6. SOTA 演进历程（原始数据集）

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 24obj 相关章节。

**数据集**: Chromosome20240904 | **单 seed**

| Phase | 改进点 | best mAP | AP50 | AP75 | 相比 baseline 提升 |
|---|---|---|---|---|---|
| — | **Baseline**: Random + RF + Heun + AdaLN | **0.728** | — | — | — |
| phase5 | + Sinkhorn Stochastic OT (ε=5) | **0.753** | 0.943 | 0.843 | **+0.025** |
| phase7_loss | + 损失函数优化 (rel L1) | **0.752** | 0.946 | 0.843 | +0.024 |
| phase7_smallobj | + 小目标检测优化 | **0.744** | 0.943 | 0.833 | +0.016 |
| phase8_kcec | + KCEC 倍性约束 (slot×2) | **0.748** | 0.944 | 0.843 | +0.020 |
| phase8_daec | + DAEC 对比学习 | **0.740** | 0.937 | 0.825 | +0.012 |
| phase9_dpm | + DPM-Solver++ (推理加速) | **0.748** | 0.945 | 0.836 | +0.020 |
| **SOTA multiseed** | 综合: 4 seeds | **0.746±0.004** | — | — | — |

---

## 7. Multi-seed SOTA 最终结果

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 24obj 相关章节。

**数据集**: Chromosome20240904

| seed | best mAP |
|---|---|
| 42 | 0.740 |
| 123 | 0.749 |
| 456 | 0.746 |
| 1000 | 0.749 |
| **mean** | **0.746 ± 0.004** |

---

## 8. Chromo 24obj 结果

**数据集**: 24 Chromosomes Object

### 8.1 GHSS 3-seed

| seed | best mAP | best epoch | AP50 | AP75 | 状态 |
|---|---|---|---|---|---|
| 42 | **0.857** | 83 | 0.988 | 0.965 | ✅ EarlyStop @ 113 |
| 123 | **0.859** | 102 | 0.988 | 0.969 | ✅ EarlyStop @ 132 |
| 789 | **0.859** | 75 | 0.989 | 0.968 | ✅ EarlyStop @ 105 |
| **mean** | **0.858±0.001** | — | — | — | ✅ **完成** |

### 8.2 Random 3-seed

| seed | best mAP | best epoch | AP50 | AP75 | 状态 |
|---|---|---|---|---|---|
| 42 | **0.859** | 59 | 0.988 | 0.967 | ✅ EarlyStop @ 89 |
| 123 | **0.860** | 115 | 0.989 | 0.969 | ✅ EarlyStop @ 145 |
| 789 | **0.860** | 82 | 0.989 | 0.970 | ✅ EarlyStop @ 112 |
| **mean** | **0.860±0.001** | — | — | — | ✅ **完成** |

### 8.3 Sinkhorn

| seed | best mAP | best epoch | AP50 | AP75 | 状态 |
|---|---|---|---|---|---|
| 42 | **0.856** | 53 | — | — | ✅ FINISHED |
| 123/789 | — | — | — | — | ⬜ 视结果决定 |

> SwanLab: project=ldmdet-ablation, experiment_name=chromo_24obj_sinkhorn_seed42, run_id=o96m1eqz4l12qjeyys1cs

### 8.4 Hard OT

| seed | 状态 |
|---|---|
| 42/123/789 | ⬜ 已取消，Sinkhorn 42 结果后决定 |

### 8.5 LDMDet 主路线消融 A0-A4（新发现）

**项目**: `ldmdet-mainline-ablation-24obj` | **数据集**: 24 Chromosomes Object

| 实验 | mAP | AP50 | AP75 | ΔmAP | 说明 |
|---|---|---|---|---|---|
| A0 baseline (Euler 1步, 无RF) | 0.774 | 0.968 | 0.916 | — | 基线 |
| A1 +RF+Heun | 0.856 | 0.990 | 0.971 | **+0.082** | 主要贡献 |
| A2 +AdaLN-Zero | 0.856 | 0.990 | 0.972 | +0.000 | 持平 A1 |
| A3 +StochOT eps5 | 0.858 | 0.990 | 0.973 | +0.002 | 边际 |
| A4 DPM-Solver++替换Heun | 0.863 | 0.990 | 0.974 | +0.005 | DPM-Solver++ 替换 Heun |

> A0-A4 主路线消融为 24obj 数据集上的新发现，对应 chromo 数据集上的旧主路线消融（见 §6，已标注暂时废弃）。A1 (RF+Heun) 贡献 +0.082 为主要改进；A4 (DPM-Solver++) 在 24obj 上不仅加速还提升精度 +0.005，与 chromo 上"持平"的结论不同。

### 8.6 其他 24obj 实验

| 实验 | mAP | 状态 |
|---|---|---|
| SC-RF (自条件化RF) | 0.857 | RUNNING |
| h_velocity_loss | 0.856 | CRASHED |
| n_cascade_e2e | 0.684 | FINISHED |
| FBM CrossAttn (few-shot) | 0.857 | CRASHED |

---

## 9. 论文表格模板

### Table 1: Main ablation — Coupling strategies

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 24obj 相关章节。

| Coupling | mAP (3 seeds) | Δ vs Random |
|---|---|---|
| Random (baseline) | **0.746 ± 0.001** | — |
| DDPM | 0.729 ± 0.004 | −0.017 |
| Hard OT | **TBD** (2/3, best 0.747) | TBD |
| Sinkhorn Stochastic (ε=5) | **TBD** (1/3, best 0.748) | TBD |
| GHSS (ε=5) | **TBD** | TBD |
| Sinkhorn Argmax (ε=5) | **TBD** | TBD |

### Table 2: SOTA comparison on 24obj benchmark

| Method | mAP |
|---|---|
| RTMDet-L | **0.869** |
| DINO R50 (4scale) | 0.868 |
| Cascade R-CNN R50 | 0.854 |
| **LDMDet (RF+AdaLN+stochot eps5)** | 0.853 |
| YOLOX-S | 0.803 |

### Table 3: Historical SOTA progression

> ⚠️ **暂时废弃**：以下结论基于旧数据集 Chromosome20240904（mAP≈0.72-0.75），24obj 数据集上的结论已更新，见 24obj 相关章节。

| Method | mAP | Δ vs baseline |
|---|---|---|
| Baseline (RF+Heun+adaln) | 0.728 | — |
| + Sinkhorn Stochastic OT | 0.753 | +0.025 |
| + DPM-Solver++ (4-step inference) | 0.748 | +0.020 |
| + KCEC ploidy constraint | 0.748 | +0.020 |
| + Loss optimization | 0.752 | +0.024 |

---

## 附录：Checkpoint 位置

### 完成的 3-seed 实验 checkpoints

```
work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth
work_dirs/multi_seed_aug/rf_heun_adaln/seed_123/best_coco_bbox_mAP_epoch_101.pth
work_dirs/multi_seed_aug/rf_heun_adaln/seed_789/best_coco_bbox_mAP_epoch_75.pth
work_dirs/multi_seed_aug/ddpm/seed_42/best_coco_bbox_mAP_epoch_79.pth
work_dirs/multi_seed_aug/ddpm/seed_123/best_coco_bbox_mAP_epoch_66.pth
work_dirs/multi_seed_aug/ddpm/seed_789/best_coco_bbox_mAP_epoch_87.pth
```

### 不完整的实验

```
work_dirs/multi_seed_aug/hard_ot/seed_42/best_coco_bbox_mAP_epoch_65.pth   (0.747)
work_dirs/multi_seed_aug/hard_ot/seed_123/best_coco_bbox_mAP_epoch_67.pth  (0.747)
work_dirs/multi_seed_aug/sinkhorn_stochastic/seed_42/best_coco_bbox_mAP_epoch_51.pth (0.748)
```
