# 实验脉络关系总览

> 本文档梳理所有实验的递进关系,明确真正的 baseline,识别废弃/错误实验。
> 更新时间: 2026-06-29

## 一、数据集分组 (关键!)

实验涉及 **3 个不同数据集**,不可跨数据集对比 mAP:

| 数据集 | 路径 | 用途 | 实验组 |
|--------|------|------|--------|
| **新数据集 (主线)** | `data/Chromosome20240904_NoAug_NoResize_coco/` | 当前研究主线 | ddpm, rf_heun, OT, sota, bottleneck, direction |
| 旧数据集 | `data/24_chromosomes_object/coco/` | 早期实验 | 24obj_ablation, merged_ablation, ldmdet_rf_heun_adaln_stochot_eps5 |
| 多数据集 | `data/selfmake_chromosome202250604_NoResizeNoAug/` | 跨数据集测试 | multi_dataset |

> ⚠️ 之前错误地将旧数据集 ghss (0.857) 作为新数据集 baseline 对照。**所有新数据集实验应与 DDPM baseline (0.729) 或 RF baseline (0.746) 比较。**

## 二、实验递进树 (新数据集主线)

```
DiffusionDet (原始 DDPM)  ← 真正的根 baseline
│  config: experiments/configs/baselines/diffusiondet_ddpm.py
│  result: multi_seed_aug/ddpm = 0.729 ± 0.003 (3 seeds)  [含数据增强]
│  SwanLab: project='ldmdet-ablation', exp='diffusiondet_ddpm_seed{42,789,123}'
│
├─→ + RF + Heun + Shifted Schedule + AdaLN-Zero  (DDPM → Rectified Flow)
│      config: experiments/configs/ldmdet/rf_heun_adaln.py
│      result: multi_seed_aug/rf_heun_adaln = 0.746 ± 0.001 (3 seeds)  [+0.017]
│      SwanLab: project='ldmdet-ablation', exp='rf_heun_adaln_seed{42,789,123}'
│      关键改动: diffusion_type=rectified_flow, solver=heun, rf_schedule=shifted, time_conditioning=adaln_zero
│      │
│      ├─→ + Hard OT Coupling
│      │      result: multi_seed_aug/hard_ot = 0.747 ± 0.000 (2 seeds)  [+0.001, 边际]
│      │      SwanLab: project='ldmdet-ablation', exp='hard_ot_seed{42,789,123}'
│      │
│      ├─→ + Sinkhorn Stochastic OT
│      │      result: multi_seed_aug/sinkhorn_stochastic = 0.748 (1 seed)  [+0.002, 边际]
│      │      SwanLab: project='ldmdet-ablation', exp='sinkhorn_stochastic_seed42'
│      │
│      ├─→ + GHSS Coupling
│      │      result: multi_seed_aug/ghss = ? (未完成/无结果)
│      │      SwanLab: project='ldmdet-ablation', exp='ghss_seed42'
│      │
│      └─→ SOTA (Sinkhorn Stochastic + aug + ot_coupling=True)
│             config: work_dirs/sota_seed{42,123,456,1000}/sota_seed*.py (动态生成)
│             result: sota_seed* = 0.742 ± 0.008 (5 runs, best 0.749)  [+0.003, 高方差]
│             SwanLab: project='chromosome-kd-multiseed', exp='sota_seed{42,123,456,1000}'
│             ⚠ SOTA 平均(0.742)反而低于 sinkhorn_stochastic(0.748),高方差(0.727-0.749)
│
├─→ Bottleneck 瓶颈分析 (基于 SOTA 模型)
│      config: experiments/configs/bottleneck/*.py
│      result: 见 work_dirs/bottleneck/bottleneck_report.json
│      分析文档: experiments/configs/bottleneck/EXPERIMENT_ANALYSIS.md
│      │
│      ├─→ focal_gamma_3 = 0.750  [+0.038 vs rf_heun 0.712, +0.004 vs SOTA 0.746]
│      ├─→ focal_gamma_1_5 = 0.747
│      ├─→ scale_aware_loss = 0.742
│      ├─→ relative_l1_loss = 0.740
│      ├─→ high_cls_weight = 0.739
│      ├─→ high_giou_weight = 0.737
│      ├─→ no_box_renewal = 0.730  (消融: 移除 box_renewal)
│      ├─→ class_balanced_sampling = FAILED (SIGKILL)
│      └─→ proposals_100 = FAILED (SIGKILL)
│
└─→ Direction 方向实验 A-F (基于 rf_heun_adaln,无 OT)
       config: experiments/configs/ldmdet/direction_*.py
       base: rf_heun_adaln.py (mAP=0.712, 无 aug, 无 OT)
       ⚠ 注意: direction 实验基于 rf_heun_adaln (无 aug), 应与 0.712 比较
       │
       ├─→ D (BoxRefineNet) = 0.747  [+0.035 vs 0.712]  ✓ 完成
       ├─→ B (DecoupledHead) = 运行中 (epoch 16, 当前 0.672)
       ├─→ C (Morphology+Contrastive) = 排队
       ├─→ F (StructuredPrior+100 proposals) = 排队
       ├─→ A (P1+Deformable) = 排队
       └─→ E (ClassBalanced) = 排队
```

## 三、关键结论

### 1. 真正的 Baseline
- **根 baseline**: DiffusionDet DDPM = **0.729** (含 aug, 3 seeds)
  - 位置: `multi_seed_aug/ddpm/`
  - SwanLab: `ldmdet-ablation/diffusiondet_ddpm_seed*`
- **改进 baseline**: RF+Heun+AdaLN = **0.746** (含 aug, 3 seeds)
  - 位置: `multi_seed_aug/rf_heun_adaln/`
  - SwanLab: `ldmdet-ablation/rf_heun_adaln_seed*`

### 2. 各部件贡献
| 改进 | ΔmAP | 评价 |
|------|------|------|
| DDPM → RF+Heun+Shifted+AdaLN | **+0.017** | 🟢 主要贡献 |
| + Hard OT Coupling | +0.001 | 🟠 边际收益 |
| + Sinkhorn Stochastic OT | +0.002 | 🟠 边际收益 |
| + SOTA (ot_coupling=True) | +0.003 (高方差) | 🟠 边际但高方差 |

### 3. 之前的错误对照 (已修正)
| 错误对照 | mAP | 问题 |
|---------|-----|------|
| ghss (旧数据集) | 0.857 | ❌ 旧数据集,不可对照 |
| rf_heun_adaln multi_seed (无 aug) | 0.712 | ❌ 缺少 aug,低于 DDPM baseline |

## 四、SwanLab 项目映射

| SwanLab Project | 实验组 | 说明 |
|-----------------|--------|------|
| `ldmdet-ablation` | ddpm, rf_heun, OT, hard_ot, sinkhorn, 24obj, merged | 主消融实验 |
| `chromosome-kd-multiseed` | sota_seed* | SOTA 多种子 |
| `chromosome-kd` | direction_exps | 方向实验 (新) |

## 五、废弃/错误实验清理方案

### A. 旧数据集实验 (不可对比, 建议清理 checkpoint 释放空间)
| 目录 | 大小 | 数据集 | mAP | 清理建议 |
|------|------|--------|-----|---------|
| `work_dirs/24obj_ablation/` | 23G | 旧 24_chromosomes_object | 0.857-0.860 | 🗑️ 删除 checkpoint,保留 scalars.json |
| `work_dirs/merged_ablation/` | 3.3G | 旧 24_chromosomes_object | 0.806-0.849 | 🗑️ 删除 checkpoint,保留 scalars.json |
| `work_dirs/ldmdet_rf_heun_adaln_stochot_eps5/` | 1.9G | 旧 24_chromosomes_object | 0.853 | 🗑️ 删除 checkpoint,保留 scalars.json |

### B. 无 aug 旧 baseline (被 multi_seed_aug 取代)
| 目录 | 大小 | mAP | 清理建议 |
|------|------|-----|---------|
| `work_dirs/multi_seed/rf_heun_adaln/` | ~5G | 0.712 | 🗑️ 删除 (被 multi_seed_aug 取代) |
| `work_dirs/multi_seed/rf_heun_adaln_bs4/` | ~5G | 0.702 | 🗑️ 删除 (bs4 实验) |
| `work_dirs/multi_seed/hard_ot/` | ~5G | 0.705 | 🗑️ 删除 (无 aug 版) |

### C. 旧 ablation (无 aug, 被 multi_seed_aug 取代)
| 目录 | 大小 | mAP | 清理建议 |
|------|------|-----|---------|
| `work_dirs/ablation/adaln*` `stochot*` | 13G | 0.720-0.738 | 🗑️ 删除 (无 aug, 结论已被取代) |

### D. 失败/调试实验
| 目录 | 大小 | mAP | 清理建议 |
|------|------|-----|---------|
| `work_dirs/debug_train_final/` | 1.9G | 0.525 | 🗑️ 删除 |
| `work_dirs/debug_train_timm/` | 8.1M | - | 🗑️ 删除 |
| `work_dirs/optim_test_v2/` | 1.8G | - | 🗑️ 删除 |
| `work_dirs/bottleneck/ablation/high_giou_weight/20260624_234713/` | - | 0.523 | 🗑️ 删除 (failed run) |
| `work_dirs/bottleneck/ablation/no_box_renewal/20260623_183224/` | - | 0.635 | 🗑️ 删除 (failed early) |

### E. 被取代的方向实验
| 目录 | 大小 | mAP | 清理建议 |
|------|------|-----|---------|
| `work_dirs/scheme_a_dinov2_s/` | 3.9G | 0.502-0.676 | 🗑️ 删除 (DINOv2 backbone 全失败) |
| `work_dirs/scheme_E_bifpn/` | 1.9G | 0.744 | 🗑️ 删除 (BiFPN 未采用) |
| `work_dirs/stability/warm_restart_*` | 6.7G | 0.714-0.728 | 🗑️ 删除 (无 aug, 被 SOTA 取代) |
| `work_dirs/cspnext_l_rf_heun_adaln_stochot/` | 2.2G | 0.730 | 🗑️ 删除 (CSPNeXt, 无 aug) |

### F. 冗余 checkpoint (保留 best, 删除中间 epoch)
| 目录 | 大小 | 问题 | 清理建议 |
|------|------|------|---------|
| `work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v3/` | **156G** | 保留所有 epoch_*.pth | 🗑️ 删除非 best checkpoint |
| `work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v2/` | **96G** | 同上 | 🗑️ 删除非 best checkpoint |
| `work_dirs/chromogen_phase1/` | **103G** | 生成模型,保留所有 checkpoint | ⚠️ 确认是否需要后清理 |

### 清理预估空间释放
- A. 旧数据集: ~28G
- B. 无 aug baseline: ~15G
- C. 旧 ablation: ~13G
- D. 失败实验: ~4G
- E. 被取代方向: ~15G
- F. 冗余 checkpoint: **~250G+**
- **总计可释放: ~325G**

## 六、保留的核心实验 (不可清理)

| 目录 | 说明 | 重要性 |
|------|------|--------|
| `work_dirs/multi_seed_aug/ddpm/` | DDPM 根 baseline | ⭐⭐⭐ |
| `work_dirs/multi_seed_aug/rf_heun_adaln/` | RF 改进 baseline | ⭐⭐⭐ |
| `work_dirs/multi_seed_aug/hard_ot/` | OT 消融 | ⭐⭐ |
| `work_dirs/multi_seed_aug/sinkhorn_stochastic/` | OT 消融 | ⭐⭐ |
| `work_dirs/sota_seed*/` | SOTA 最终结果 | ⭐⭐⭐ |
| `work_dirs/bottleneck/` | 瓶颈分析 (保留 best + report) | ⭐⭐⭐ |
| `work_dirs/direction_exps/` | 方向实验 (进行中) | ⭐⭐⭐ |

## 七、Direction 实验的正确对照

Direction 实验基于 `rf_heun_adaln.py` (无 aug, 无 OT, mAP=0.712):
- ✅ 正确对照: rf_heun_adaln multi_seed = **0.712**
- ❌ 错误对照: ghss (旧数据集) = 0.857
- ❌ 错误对照: multi_seed_aug/rf_heun_adaln (有 aug) = 0.746

> Direction 实验缺少数据增强,这是设计选择 (隔离变量),但意味着绝对值偏低。
> 若要追求 SOTA,应将有效方向叠加到 SOTA config (含 aug + OT) 上。
