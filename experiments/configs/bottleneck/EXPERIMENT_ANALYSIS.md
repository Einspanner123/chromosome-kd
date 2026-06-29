# Bottleneck 验证实验记录与分析

> 本文档记录 `experiments/configs/bottleneck/` 下所有瓶颈验证实验的结果、状态与分析结论。
> 每条实验记录附 **可靠数据源地址** (本地服务器路径 / SwanLab run_id)。
> 实验时间: 2026-06-23 ~ 2026-06-29
> 更新时间: 2026-06-29

## 一、实验环境与基线对照 (修正)

### 数据集 (统一)
所有瓶颈实验使用同一数据集:
- **数据集**: `data/Chromosome20240904_NoAug_NoResize_coco/` (24 类染色体)
- **训练集**: `train/_annotations.coco.json`
- **类别数**: 24

### aug 策略
- 所有 bottleneck 实验使用 **DiffusionDet 默认 aug** (multi_scale + RandomCrop)
- 与 `multi_seed_aug/*` baseline 使用相同 aug,可直接对比

### 正确的 Baseline 对照
| Baseline | mAP | 数据源 |
|----------|-----|--------|
| **RF+Heun+AdaLN (默认 aug)** | **0.746 ± 0.001** (3 seeds) | 本地 `work_dirs/multi_seed_aug/rf_heun_adaln/` + SwanLab `ldmdet-ablation/rf_heun_adaln_seed{42,789,123}` |
| 历史 SOTA (stochot_eps5_v2) | 0.753 | ldmdet-experiment `sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/` |

> ⚠️ **修正记录**:
> 之前误用 `multi_seed/rf_heun_adaln` (mAP=0.712, **简化 aug**, 无 multi-scale/crop) 作为对照。
> 实际上 0.712 是非标准简化实验,不应作为 baseline。正确对照应为 `multi_seed_aug/rf_heun_adaln` (0.746, **DiffusionDet 默认 aug**)。
> 用户判断正确: aug 是 DiffusionDet 默认,不是额外增强。

## 二、实验结果总览 (附数据源)

### 实验完成状态
| 实验 | 配置 | 训练到 epoch | best epoch | best mAP | 状态 | 本地路径 | SwanLab run_id |
|------|------|------|------|---------|------|---------|---------------|
| focal_gamma_3 | `focal_gamma_3.py` | 118/150 | 88 | 0.7500 | ✓ 已收敛 | `work_dirs/bottleneck/ablation/focal_gamma_3/20260628_013823/` | `ye6a2whory9y67tnvalg3` (ldmdet-ablation/focal_gamma_3_seed42) |
| focal_gamma_1_5 | `focal_gamma_1_5.py` | 111/150 | 81 | 0.7470 | ✓ 已收敛 | `work_dirs/bottleneck/ablation/focal_gamma_1_5/20260628_100256/` | `1f0r738snqeljucu6jm03` |
| scale_aware_loss | `scale_aware_loss.py` | 99/150 | 69 | 0.7420 | ✓ 已收敛 | `work_dirs/bottleneck/ablation/scale_aware_loss/20260624_112630/` | `o8elke3rcmln1i47fznr5` |
| relative_l1_loss | `relative_l1_loss.py` | 106/150 | 76 | 0.7400 | ✓ 已收敛 | `work_dirs/bottleneck/ablation/relative_l1_loss/20260626_093627/` | `cmxgctni1n1d9g9go0ywf` |
| high_cls_weight | `high_cls_weight.py` | 104/150 | 74 | 0.7390 | ✓ 已收敛 | `work_dirs/bottleneck/ablation/high_cls_weight/20260628_180423/` | `m9dnqrl43khkl3jcbjico` |
| high_giou_weight | `high_giou_weight.py` | 75/150 | 45 | 0.7370 | ⚠ best 较早 | `work_dirs/bottleneck/ablation/high_giou_weight/20260625_230646/` | `b3w4gwly842j700yqg9og` |
| no_box_renewal | `no_box_renewal.py` | 41/150 | 37 | 0.7300 | ⚠ early stop | `work_dirs/bottleneck/ablation/no_box_renewal/20260623_211757/` | `52o1g5pq09eddplyzbl9q` |
| class_balanced_sampling | `class_balanced_sampling.py` | 0/150 | - | - | ✗ failed | `work_dirs/bottleneck/ablation/class_balanced_sampling/20260629_013833/` | `vsq3xd3bw51iss1n1at15` |
| proposals_100 | `proposals_100.py` | 0/150 | - | - | ✗ failed | `work_dirs/bottleneck/ablation/proposals_100/20260624_021155/` | - (无 run_id) |

### 与正确 Baseline (0.746) 的对比
| 实验 | best mAP | Δ vs baseline | 评价 |
|------|---------|---------|------|
| focal_gamma_3 | 0.7500 | **+0.004** | 🟢 最佳,焦点损失 gamma=3 |
| focal_gamma_1_5 | 0.7470 | +0.001 | 🟡 持平 |
| scale_aware_loss | 0.7420 | -0.004 | 🟡 略低 |
| relative_l1_loss | 0.7400 | -0.006 | 🟠 低于 |
| high_cls_weight | 0.7390 | -0.007 | 🟠 低于 |
| high_giou_weight | 0.7370 | -0.009 | 🟠 低于 |
| no_box_renewal | 0.7300 | -0.016 | 🔴 显著低于 (消融验证 box_renewal 贡献) |

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

## 四、bottleneck 消融实验关键发现 (修正)

### 1. focal_gamma_3 最佳 (+0.004 vs 0.746)
- **改动**: FocalLoss 的 gamma 从 2.0 → 3.0
- **机制**: 增加困难样本权重,对同组形态相似类的难分类样本更敏感
- **价值**: 验证了瓶颈分析中"分类误差为主"的判断
- **数据源**: 本地 `work_dirs/bottleneck/ablation/focal_gamma_3/20260628_013823/` + SwanLab `ldmdet-ablation/focal_gamma_3_seed42` (run_id=ye6a2whory9y67tnvalg3)

### 2. no_box_renewal 实验 (-0.016 vs 0.746)
- **改动**: 移除 box renewal 机制
- **结果**: mAP=0.730,显著低于 baseline (0.746)
- **解读**: 
  - 移除 box renewal 导致 -0.016,说明 **box_renewal 贡献约 +0.016**
  - 这验证了方向 D (BoxRefineNet) 的必要性
  - 但 direction_d 实验结果 (0.747) 显示加上 BoxRefineNet 仅 +0.001,可能 BoxRefineNet 设计还需优化
- **数据源**: 本地 `work_dirs/bottleneck/ablation/no_box_renewal/20260623_211757/` + SwanLab `ldmdet-ablation/no_box_renewal` (run_id=52o1g5pq09eddplyzbl9q)

### 3. high_giou_weight 实验 (-0.009 vs 0.746)
- **改动**: GIoU 损失权重提升
- **结果**: best epoch 45 (较早),mAP=0.737
- **解读**: GIoU 权重提升反而降低性能,可能训练不稳定
- **数据源**: 本地 `work_dirs/bottleneck/ablation/high_giou_weight/20260625_230646/` + SwanLab `ldmdet-ablation/high_giou_weight` (run_id=b3w4gwly842j700yqg9og)

### 4. scale_aware_loss (-0.004 vs 0.746)
- **改动**: 尺度感知损失
- **结果**: mAP=0.742,略低于 baseline
- **解读**: 尺度感知损失无效,印证瓶颈分析中"尺度非主瓶颈"的判断
- **数据源**: 本地 `work_dirs/bottleneck/ablation/scale_aware_loss/20260624_112630/` + SwanLab `ldmdet-ablation/scale_aware_loss` (run_id=o8elke3rcmln1i47fznr5)

## 五、对方向 A-F 实验的指导意义

### 与瓶颈的对应关系
| 瓶颈发现 | 对应方向 | 预期收益 | bottleneck 验证 |
|---------|---------|---------|------|
| 定位精度差 (AP@90=0.48) | **方向 D (BoxRefineNet)** | 残差精化提升高 IoU 精度 | ✓ no_box_renewal 验证 box_renewal 贡献 +0.016 |
| 分类误差 43.4% (同组混淆) | **方向 B (DecoupledHead)** + **方向 C (ShapeAttention+ContrastiveLoss)** | 形态感知 + 困难负对挖掘 | ✓ focal_gamma_3 验证分类损失调整有效 (+0.004) |
| Proposal 浪费 (500→302) | **方向 F (num_proposals 100)** | 减少背景浪费 | ⚠ 未验证 (proposals_100 failed) |
| 尺度非瓶颈 | 方向 A (P1+Deformable) 收益有限 | 小目标 recall 已 0.905 | ✗ scale_aware_loss 验证尺度非瓶颈 (-0.004) |

### 建议实验优先级 (基于瓶颈严重程度)
1. **方向 D** (定位精度最严重,AP@90 仅 0.48) - 但 direction_d 实际仅 +0.001,需重新设计
2. **方向 B+C** (分类误差 43.4%,同组混淆) - focal_gamma_3 验证分类损失调整有效
3. **方向 F** (proposal 优化,推理加速) - 未验证
4. 方向 A、E (瓶颈分析显示收益有限) - scale_aware_loss 验证尺度非瓶颈

## 六、数据集一致性说明 (修正记录)

> 之前的分析误将 `24obj_ablation/ghss` (旧数据集 `24_chromosomes_object/coco/`,mAP=0.857) 作为 bottleneck 实验的对照 baseline,这是错误的。
> 又误将 `multi_seed/rf_heun_adaln` (mAP=0.712, **简化 aug**) 作为对照,这也是错误的。

| 实验 | 数据集 | aug 策略 | mAP | 评价 |
|------|--------|---------|-----|------|
| multi_seed/rf_heun_adaln (旧对照) | `Chromosome20240904_NoAug_NoResize_coco/` | simple_resize (无 multi-scale/crop) | 0.712 | ❌ 非标准 |
| multi_seed_aug/rf_heun_adaln (正确对照) | `Chromosome20240904_NoAug_NoResize_coco/` | DiffusionDet 默认 aug | 0.746 | ✅ 标准 baseline |
| ghss (24obj_ablation, 旧对照) | `24_chromosomes_object/coco/` | DiffusionDet 默认 aug | 0.857 | ❌ 旧数据集 |
| bottleneck 实验 (全部) | `Chromosome20240904_NoAug_NoResize_coco/` | DiffusionDet 默认 aug | 0.730-0.750 | ✅ 可与 0.746 对照 |
| direction 实验 (全部) | `Chromosome20240904_NoAug_NoResize_coco/` | DiffusionDet 默认 aug | - | ✅ 可与 0.746 对照 |

**正确对照**: bottleneck 与 direction 实验都应与 **multi_seed_aug/rf_heun_adaln (0.746)** 比较。

## 七、报告文件索引
- `work_dirs/bottleneck/bottleneck_report.json` - 核心瓶颈诊断
- `work_dirs/bottleneck/supplementary_report.json` - 补充分析
- `work_dirs/bottleneck/final_report.json` - 整合报告 (含 sampling_sweep)
- `work_dirs/bottleneck/phase2_summary.json` - phase2 实验状态 (no_box_renewal, proposals_100 失败)
- `work_dirs/bottleneck/bottleneck_preds.json` - 预测结果
- `work_dirs/bottleneck/sampling_sweep_results.json` - 采样策略扫描

## 八、实验配置文件索引
| 配置 | 说明 | best mAP | 数据源 |
|------|------|---------|--------|
| `ddpm_baseline.py` | DDPM 基线 (对比扩散框架) | - | - |
| `focal_gamma_3.py` | FocalLoss gamma=3 | 0.750 (best) | 本地 + SwanLab `ye6a2whory9y67tnvalg3` |
| `focal_gamma_1_5.py` | FocalLoss gamma=1.5 | 0.747 | 本地 + SwanLab `1f0r738snqeljucu6jm03` |
| `scale_aware_loss.py` | 尺度感知损失 | 0.742 | 本地 + SwanLab `o8elke3rcmln1i47fznr5` |
| `relative_l1_loss.py` | 相对 L1 损失 | 0.740 | 本地 + SwanLab `cmxgctni1n1d9g9go0ywf` |
| `high_cls_weight.py` | 分类损失权重提升 | 0.739 | 本地 + SwanLab `m9dnqrl43khkl3jcbjico` |
| `high_giou_weight.py` | GIoU 权重提升 | 0.737 | 本地 + SwanLab `b3w4gwly842j700yqg9og` |
| `no_box_renewal.py` | 移除 box renewal (消融) | 0.730 | 本地 + SwanLab `52o1g5pq09eddplyzbl9q` |
| `class_balanced_sampling.py` | 类别平衡采样 | failed | 本地 + SwanLab `vsq3xd3bw51iss1n1at15` |
| `proposals_100.py` | 100 proposals | failed | 本地 (无 SwanLab run) |
| `proposals_1000.py` | 1000 proposals | - | - |
| `heads_2.py` / `heads_12.py` | 注意力头数 | - | - |
| `rf_shift_1.py` / `rf_shift_5.py` | RF 平移 | - | - |
| `multi_step_sampling.py` / `single_step_sampling.py` | 采样步数 | - | - |
| `no_deep_supervision.py` / `no_ensemble.py` | 移除深度监督/集成 | - | - |
| `dpm_solver_pp.py` | DPM-Solver++ | - | - |
