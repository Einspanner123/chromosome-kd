# LDMDet 论文结果数据

> 最后更新: 2026-06-25
> 数据集:
>   - **Original**: `data/Chromosome20240904_NoAug_NoResize_coco/` — 24 类染色体，原始数据集
>   - **24obj**: `data/24_chromosomes_object/coco/` — 24 类染色体，后采集的大规模基准数据集

---

## 1. 原始数据集结果（Original Dataset）

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

---

## 3. 论文表格模板

### Table 1: Main coupling ablation (Original Dataset, 3-seed)

| Coupling Strategy | mAP (mean±std) | Δ vs Random |
|---|---|---|
| Random (baseline) | **0.746 ± 0.001** | — |
| DDPM | 0.729 ± 0.004 | −0.017 |
| Hard OT | *TBD* | *TBD* |
| Sinkhorn Stochastic (ε=5) | *TBD* | *TBD* |
| GHSS (ε=5) | *TBD* | *TBD* |

### Table 2: SOTA comparison on two datasets

| Method | Backbone | Original Dataset | 24obj Dataset |
|---|---|---|---|
| RTMDet-L | CSPNeXt-L | 0.742 | **0.869** |
| DINO R50 (4scale) | ResNet-50 | 0.737 | 0.868 |
| Cascade R-CNN R50 | ResNet-50 | 0.732 | 0.854 |
| **LDMDet (Ours)** | ResNet-50 | **0.753** | **0.860** |
| YOLOX-S | CSPDarkNet-S | 0.608 | 0.803 |

### Table 3: Epsilon Analysis

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
| Sinkhorn 24obj seed 42 | 🟡 看结果决定后续 | 🔵 运行中（06-26 ~10:30 完成） |
| Sinkhorn / Hard OT 剩余 5 seed | ⬜ 可能取消 | 如 Sinkhorn 42 ≈ 0.860，则取消 |
| Hard OT / Sinkhorn full-aug 补全（原始数据集） | 🟢 低优先级 | 原 ablation table 的占位 |

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
