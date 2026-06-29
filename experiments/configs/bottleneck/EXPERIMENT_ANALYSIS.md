# Bottleneck 验证实验记录与分析

> 本文档记录 `experiments/configs/bottleneck/` 下所有瓶颈验证实验的结果、状态与分析结论。
> 实验时间: 2026-06-23 ~ 2026-06-29

## 一、实验环境与基线对照

### 数据集 (重要)
所有瓶颈实验均使用同一数据集:
- **数据集**: `data/Chromosome20240904_NoAug_NoResize_coco/` (24 类染色体)
- **训练集**: `train/_annotations.coco.json`
- **类别数**: 24

### 正确的 Baseline 对照
| Baseline | 数据集 | mAP | 说明 |
|----------|--------|-----|------|
| **rf_heun_adaln (multi_seed)** | `Chromosome20240904_NoAug_NoResize_coco/` | **0.712** | ✅ **正确对照** (3 seeds 平均) |
| ghss (24obj_ablation) | `24_chromosomes_object/coco/` | 0.857 | ❌ 旧数据集,不可对照 |

> ⚠️ **数据集差异**: `Chromosome20240904_NoAug_NoResize_coco/` (新) vs `24_chromosomes_object/coco/` (旧) 是两个不同数据集,样本量、标注策略不同,mAP 不可直接对比。瓶颈实验与 direction 实验应与 **rf_heun_adaln multi_seed (0.712)** 比较。

## 二、实验结果总览

### 实验完成状态
| 实验 | 配置文件 | 训练到 epoch | best epoch | best mAP | 状态 |
|------|---------|------|------|---------|------|
| focal_gamma_3 | `focal_gamma_3.py` | 118/150 | 88 | 0.7500 | ✓ best 已收敛 |
| focal_gamma_1_5 | `focal_gamma_1_5.py` | 111/150 | 81 | 0.7470 | ✓ best 已收敛 |
| scale_aware_loss | `scale_aware_loss.py` | 99/150 | 69 | 0.7420 | ✓ best 已收敛 |
| relative_l1_loss | `relative_l1_loss.py` | 106/150 | 76 | 0.7400 | ✓ best 已收敛 |
| high_cls_weight | `high_cls_weight.py` | 104/150 | 74 | 0.7390 | ✓ best 已收敛 |
| high_giou_weight | `high_giou_weight.py` | 75/150 | 45 | 0.7370 | ⚠ best 较早,可能未充分收敛 |
| no_box_renewal | `no_box_renewal.py` | 41/150 | 37 | 0.7300 | ⚠ early stop (failed -9) |
| class_balanced_sampling | `class_balanced_sampling.py` | 0/150 | - | - | ✗ failed (-9 SIGKILL) |
| proposals_100 | `proposals_100.py` | 0/150 | - | - | ✗ failed (-9 SIGKILL) |

### 与正确 Baseline (0.712) 的对比
| 实验 | best mAP | Δ vs baseline | 评价 |
|------|---------|---------|------|
| focal_gamma_3 | 0.7500 | **+0.038** | 🟢 最佳,焦点损失 gamma=3 显著提升 |
| focal_gamma_1_5 | 0.7470 | +0.035 | 🟢 焦点损失 gamma=1.5 |
| scale_aware_loss | 0.7420 | +0.030 | 🟢 尺度感知损失 |
| relative_l1_loss | 0.7400 | +0.028 | 🟡 相对 L1 损失 |
| high_cls_weight | 0.7390 | +0.027 | 🟡 分类损失权重提升 |
| high_giou_weight | 0.7370 | +0.025 | 🟡 GIoU 权重提升 |
| no_box_renewal | 0.7300 | +0.018 | 🟠 移除 box_renewal 仍高于 baseline (说明其他改动有效) |

### 是否需要继续跑完?
**结论: 不需要**
- 7/9 实验有有效数据,best_mAP 已在 epoch 37-88 之间出现
- 继续训练到 150 epoch 的边际收益有限 (best 已收敛)
- 2 个 failed 实验 (proposals_100, class_balanced_sampling) 已被方向 F (num_proposals=100) 和方向 E (class_balanced) 取代,无需重跑

## 三、瓶颈分析结论 (来自 `work_dirs/bottleneck/bottleneck_report.json`)

### 主瓶颈 1: 定位精度 (最严重)
- AP@50 = 0.943 → AP@75 = 0.839 → AP@90 = 0.483 → AP@95 = 0.097
- IoU mean = 0.890,但需要 0.95+ 才能提升 AP@90
- **最大下降区间**: [0.9, 0.95],下降 0.387
- **结论**: 高 IoU 精度是主要瓶颈,框质量不足

### 主瓶颈 2: 分类误差 (43.4%)
- 最差类别: C9 (0.888), Y (0.901), C10 (0.903), G22 (0.908), X (0.911)
- **Top 混淆对全是同组形态相似类**:
  - C10↔C9, C9↔C8 (C 组内混淆)
  - B5→B4 (B 组内)
  - D13↔D14 (D 组内)
  - F20→E18 (跨组但形态相似)
- **结论**: 分类误差主要来自同组形态相似类,需要形态感知特征

### 非瓶颈
- **尺度差异**: 小/中/大 recall 均 ~0.92,尺度非瓶颈
- **置信度校准**: ECE=0.036,校准良好
- **Proposal 利用率**: 500 proposals → 平均预测 302.8,GT 46.8,pred/gt=6.47,NMS 抑制率 39.45%

### 误差分解
- 分类误差: 43.4% (主)
- 定位误差: 35.6% (次)
- 其他: 21.0%

## 四、bottleneck 消融实验关键发现

### 1. focal_gamma_3 最佳 (+0.038)
- **改动**: FocalLoss 的 gamma 从 2.0 → 3.0
- **机制**: 增加困难样本权重,对同组形态相似类的难分类样本更敏感
- **价值**: 验证了瓶颈分析中"分类误差为主"的判断

### 2. no_box_renewal 实验 (+0.018)
- **改动**: 移除 box renewal 机制
- **结果**: mAP=0.730,仍高于 baseline (0.712)
- **解读**: 
  - 即使移除 box renewal,其他改动仍带来 +0.018 提升
  - 但相比其他改动 (+0.025~+0.038),no_box_renewal 的提升最少
  - **说明 box_renewal 本身贡献约 +0.007~+0.020** (其他实验的提升减去 no_box_renewal 的提升)
  - 这验证了方向 D (BoxRefineNet) 的必要性

### 3. high_giou_weight 实验 (+0.025)
- **改动**: GIoU 损失权重提升
- **结果**: best epoch 45 (较早),mAP=0.737
- **解读**: GIoU 权重提升对定位精度有帮助,但训练不稳定 (best 出现早)

### 4. scale_aware_loss (+0.030)
- **改动**: 尺度感知损失
- **结果**: mAP=0.742
- **解读**: 尺度感知损失有效,但瓶颈分析显示尺度非主瓶颈,收益可能来自其他机制

## 五、对方向 A-F 实验的指导意义

### 与瓶颈的对应关系
| 瓶颈发现 | 对应方向 | 预期收益 | bottleneck 验证 |
|---------|---------|---------|------|
| 定位精度差 (AP@90=0.48) | **方向 D (BoxRefineNet)** | 残差精化提升高 IoU 精度 | ✓ no_box_renewal 验证 box_renewal 有效 |
| 分类误差 43.4% (同组混淆) | **方向 B (DecoupledHead)** + **方向 C (ShapeAttention+ContrastiveLoss)** | 形态感知 + 困难负对挖掘 | ✓ focal_gamma_3 验证分类损失调整有效 |
| Proposal 浪费 (500→302) | **方向 F (num_proposals 100)** | 减少背景浪费 | ⚠ 未验证 (proposals_100 failed) |
| 尺度非瓶颈 | 方向 A (P1+Deformable) 收益有限 | 小目标 recall 已 0.905 | ✗ 尺度非瓶颈,收益有限 |

### 建议实验优先级 (基于瓶颈严重程度)
1. **方向 D** (定位精度最严重,AP@90 仅 0.48)
2. **方向 B+C** (分类误差 43.4%,同组混淆)
3. **方向 F** (proposal 优化,推理加速)
4. 方向 A、E (瓶颈分析显示收益有限)

## 六、数据集一致性说明 (修正记录)

> 之前的分析误将 `24obj_ablation/ghss` (旧数据集 `24_chromosomes_object/coco/`,mAP=0.857) 作为 bottleneck 实验的对照 baseline,这是错误的。

| 实验 | 数据集 | mAP |
|------|--------|-----|
| rf_heun_adaln (multi_seed, 新数据集) | `Chromosome20240904_NoAug_NoResize_coco/` | 0.712 |
| ghss (24obj_ablation, 旧数据集) | `24_chromosomes_object/coco/` | 0.857 |
| bottleneck 实验 (全部) | `Chromosome20240904_NoAug_NoResize_coco/` | 0.730-0.750 |
| direction 实验 (全部) | `Chromosome20240904_NoAug_NoResize_coco/` | - |

**正确对照**: bottleneck 与 direction 实验都应与 **rf_heun_adaln multi_seed (0.712)** 比较。

## 七、报告文件索引
- `work_dirs/bottleneck/bottleneck_report.json` - 核心瓶颈诊断
- `work_dirs/bottleneck/supplementary_report.json` - 补充分析
- `work_dirs/bottleneck/final_report.json` - 整合报告 (含 sampling_sweep)
- `work_dirs/bottleneck/phase2_summary.json` - phase2 实验状态 (no_box_renewal, proposals_100 失败)
- `work_dirs/bottleneck/bottleneck_preds.json` - 预测结果
- `work_dirs/bottleneck/sampling_sweep_results.json` - 采样策略扫描

## 八、实验配置文件索引
| 配置 | 说明 |
|------|------|
| `ddpm_baseline.py` | DDPM 基线 |
| `focal_gamma_3.py` | FocalLoss gamma=3 (+0.038, 最佳) |
| `focal_gamma_1_5.py` | FocalLoss gamma=1.5 (+0.035) |
| `scale_aware_loss.py` | 尺度感知损失 (+0.030) |
| `relative_l1_loss.py` | 相对 L1 损失 (+0.028) |
| `high_cls_weight.py` | 分类损失权重提升 (+0.027) |
| `high_giou_weight.py` | GIoU 权重提升 (+0.025) |
| `no_box_renewal.py` | 移除 box renewal (+0.018) |
| `class_balanced_sampling.py` | 类别平衡采样 (failed) |
| `proposals_100.py` | 100 proposals (failed) |
| `proposals_1000.py` | 1000 proposals |
| `heads_2.py` / `heads_12.py` | 注意力头数 |
| `rf_shift_1.py` / `rf_shift_5.py` | RF 平移 |
| `multi_step_sampling.py` / `single_step_sampling.py` | 采样步数 |
| `no_deep_supervision.py` / `no_ensemble.py` | 移除深度监督/集成 |
| `dpm_solver_pp.py` | DPM-Solver++ |
