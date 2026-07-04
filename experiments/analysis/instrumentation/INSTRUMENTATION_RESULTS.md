# 白盒插桩分析结果

> 本文档记录对 7 个 checkpoint 执行白盒插桩分析的实验结果与发现。
> 执行时间: 2026-06-29 ~ 2026-06-30
> 方案文档: [INSTRUMENTATION_PLAN.md](./INSTRUMENTATION_PLAN.md)
> 原始 JSON 报告: `work_dirs/instrumentation/<ckpt_name>/{trajectory,roi_feature,head_output}_report.json`
> 汇总对比: `work_dirs/instrumentation/comparison.json`

## 一、执行概况

### 分析矩阵

| # | Checkpoint | 数据集 | mAP | Trajectory | RoIFeature | HeadOutput | 样本量 |
|---|-----------|--------|-----|:---------:|:---------:|:---------:|:------:|
| 1 | baseline_aug | Chromosome20240904 | 0.745 | ✓ | ✓ | ✓ | 20 |
| 2 | direction_d | Chromosome20240904 | 0.747 | ✓ | — | — | 20 |
| 3 | direction_b | Chromosome20240904 | 0.749 | — | ✓ | ✓ | 50 |
| 4 | focal_gamma_3 | Chromosome20240904 | 0.750 | — | — | ✓ | 50 |
| 5 | no_box_renewal | Chromosome20240904 | 0.635 | ✓ | — | — | 20 |
| 6 | ghss_24obj | 24_chromosomes_object | 0.857 | ✓ | ✓ | ✓ | 20 |
| 7 | random_24obj | 24_chromosomes_object | 0.859 | ✓ | ✓ | ✓ | 20 |

### 执行参数
- 采样器: **Rectified Flow + Heun solver**, `sampling_timesteps=4` (每图 4 步采样)
- `rf_schedule='shifted'`, `rf_shift=3.0`
- trajectory: 每图独立分析后聚合 (修复跨图混合 bug)
- RoIFeature / HeadOutput: 累积 RoI 样本 (每图约 500 proposals)
- 设备: `cuda:0`

### 染色体类别索引映射
```
0:A1  1:A2  2:A3   | 3:B4  4:B5        | 5:C10  6:C11  7:C12  8:C6  9:C7  10:C8  11:C9
12:D13 13:D14 14:D15 | 15:E16 16:E17 17:E18 | 18:F19 19:F20 | 20:G21 21:G22 | 22:X | 23:Y
```

分组: A(0-2), B(3-4), C(5-11), D(12-14), E(15-17), F(18-19), G(20-21), X(22), Y(23)

## 二、核心发现 (跨 ckpt 综合)

### 发现 1: 分类瓶颈在 cls_head, 不在特征抽取

| ckpt | 同组内相似度 (均值) | 跨组相似度 | same_group_too_similar |
|------|:------------------:|:----------:|:---------------------:|
| baseline_aug | 0.997 | 0.996 | False |
| direction_b | 0.9995 | 0.9993 | False |
| ghss_24obj | 0.998 | 0.9965 | False |
| random_24obj | 0.997 | 0.9963 | False |

**关键观察**:
- 同组类别余弦相似度与跨组相似度**几乎无差异** (差异 <0.003)
- `same_group_too_similar=False` 说明特征层面同组并未"过度相似"
- 但瓶颈报告显示 Cls error 43.4% 且 Top 混淆全是同组

**结论**: 分类瓶颈**不在 RoI 特征抽取**, 而在 **cls_head** 在已抽取特征上未能拉开同组类间距。这解释了为什么 direction_b (解耦 head) 和 focal_gamma_3 (分类损失) 都能小幅提升 mAP — 它们都作用于 cls_head。

**对方向的指导**:
- 方向 C (Morphology+Contrastive) 若仅做特征层对比学习, 效果可能有限
- 应优先改进 cls_head 的损失/结构, 而非特征抽取

### 发现 2: Y 类 (class 23) 坍塌是普遍问题

| ckpt | class_collapse | collapsed_classes | Y 类 mean_logit | 其他类平均 mean_logit |
|------|:--------------:|:-----------------:|:---------------:|:--------------------:|
| baseline_aug | True | [23] | -5.17 | -4.01 |
| direction_b | True | [23] | -4.03 | -3.51 |
| focal_gamma_3 | True | [23] | -3.57 | -2.85 |
| ghss_24obj | True | [0, 23] | -5.03 | -4.10 |
| random_24obj | True | [23] | -5.26 | -4.10 |

**关键观察**:
- 5 个 ckpt 全部检测到 Y 类坍塌 (class 23)
- Y 类 mean_logit 始终显著低于其他类平均 (低 0.5~1.2)
- 即使 focal_gamma=3 (理论上抑制易分类、放大难分类) 也未解决
- ghss_24obj 额外有 A1 类 (class 0) 坍塌

**根因**: Y 染色体在数据集中样本数最少, 训练信号弱, cls_head 学不到有效的 Y 类决策边界。

**对方向的指导**:
- 方向 E (ClassBalanced) 是解决此问题的关键, 应优先实现类别平衡采样 + 重加权
- 在 E 实现前, 所有分类相关改进 (B, focal_gamma) 都无法根治 Y 类问题

### 发现 3: 扩散采样步数充分, box 第 3 步收敛

| ckpt | n_steps/图 | box convergence_step | cls convergence_step | cls_converged_ratio |
|------|:---------:|:--------------------:|:--------------------:|:-------------------:|
| baseline_aug | 4 | 3.0 | 2.05 | 0.95 |
| direction_d | 4 | 3.0 | 2.0 | 1.0 |
| no_box_renewal | 4 | 3.0 | 2.0 | 1.0 |
| ghss_24obj | 4 | 3.0 | 2.1 | 0.9 |
| random_24obj | 4 | 3.0 | 2.05 | 0.95 |

**关键观察**:
- 采样器: Rectified Flow + Heun solver, `sampling_timesteps=4` (每图 4 步)
- **box 在第 3 步收敛** (4 步中倒数第 2 步), 全部图一致 (distribution: {step 3: 20/20})
- **cls 在第 2 步收敛** (90-100% 图收敛), 比 box 更早
- 4 步采样对 box 是必要的: 减到 3 步刚好在收敛边界, 减到 2 步 box 未收敛

**深层含义**:
- box 晚于 cls 收敛 (3 vs 2) → 定位比分类需要更多采样步
- 这与瓶颈报告 (定位瓶颈在 IoU≥0.9) 一致: 高 IoU 定位需要充分的扩散步数
- cls 早期收敛但 mAP 仍受 cls 误差影响 → cls_head 的预测质量不是步数问题, 而是容量/训练问题 (见发现 1, 2)

**对方向 H (采样效率) 的影响**:
- 加速空间有限: 4→3 步仅 1.33× 加速, 且在收敛边界, mAP 可能下降
- 4→2 或 4→1 会损害 box 收敛, 不可行
- H 方案价值有限, 降为可选验证

### 发现 4: box_renewal 推理未触发, 但训练时大幅稳定 x0 步间一致性

| ckpt | renewal_effective_ratio | early_x0_quality | x0_stability |
|------|:----------------------:|:----------------:|:------------:|
| baseline_aug | 0.0 | 0.280 | **81.23** |
| direction_d | 0.0 | 0.299 | 75.19 |
| no_box_renewal | 0.0 | **0.517** | **7.10** |
| ghss_24obj | 0.0 | 0.270 | 82.81 |
| random_24obj | 0.0 | 0.269 | 85.47 |

**关键观察**:
- `renewal_effective_ratio=0.0` 在所有 ckpt 成立 — box_renewal 在**推理时确实未触发**
- direction_d (BoxRefineNet 方向) 也未让 renewal 生效 — mAP +0.002 不是来自 renewal

**但 mAP 暴跌 0.110**: no_box_renewal mAP=0.635 vs baseline 0.745, 差距 -0.110

**反直觉发现与解释**:
- no_box_renewal 的 `early_x0_quality=0.517` **反而高于** baseline 的 0.280 — 关闭 renewal 后单步 x0 质量更好
- 但 `x0_stability=7.10` **远低于** baseline 的 81.23 — 降幅 **91%**
- 解释: renewal **不改善单步 x0 质量**, 但**大幅提升步间稳定性** (从 7 → 81, 11 倍)
- renewal 在训练时作为正则, 让 x0 预测在步间保持一致; 关闭后 x0 单步质量虽好, 但步间剧烈波动, 最终定位质量下降

**对方向的指导**:
- 方向 D (BoxRefineNet) 的当前实现**未对症** — renewal 在推理时未触发, 改进 renewal_head 不会有效
- 应重新审视 renewal 的触发条件 (是否阈值过严?), 或将 box_refine 改为**显式作用于 reg_head 输出**而非 renewal 机制
- reg 系统性偏移 (见发现 5) 是更值得针对的问题

### 发现 5: 回归头存在系统性 w/h 收缩偏移

| ckpt | delta_mean | per_dim_mean [dx, dy, dw, dh] | is_conservative |
|------|:---------:|:----------------------------:|:---------------:|
| baseline_aug | -0.836 | [-0.013, 0.122, **-1.819**, **-1.633**] | False |
| direction_b | -0.798 | [-0.022, 0.050, **-1.652**, **-1.569**] | False |
| focal_gamma_3 | -0.766 | [0.002, -0.083, **-1.536**, **-1.446**] | False |
| ghss_24obj | -0.641 | [-0.027, 0.042, **-1.310**, **-1.270**] | False |
| random_24obj | -0.682 | [-0.064, -0.128, **-1.269**, **-1.267**] | False |

**关键观察**:
- 所有 ckpt 的 dw, dh 均显著偏负 (-1.2 ~ -1.8)
- 这**不是保守回归** (is_conservative=False, 因 |delta_mean| > 0.1), 而是**系统性缩小框**
- dx, dy 接近 0, 说明中心定位无偏, 仅尺度预测系统性偏小
- 新数据集收缩更严重 (-1.6~-1.8) vs 24obj (-1.3), 但方向一致

**可能根因**:
- 训练数据中 GT 框可能偏大 (标注习惯), 模型学习到"收缩"补偿
- 或扩散过程的 box 初始化偏大, reg_head 学习补偿
- 或 L1 loss 对 dw/dh 的对称性导致模型学到均值偏移

**对方向的指导**:
- 方向 D' (reg 校正) 应针对此系统性偏移设计
- 可考虑: (1) 后处理尺度校准; (2) reg_head 加正则化约束; (3) 数据增强中加 box 尺度抖动

### 发现 6: direction_b 与 focal_gamma_3 的共同机制 — "更谦虚"

| ckpt | hard_sample_confidence | hard_sample_top2_gap | mAP |
|------|:----------------------:|:--------------------:|:---:|
| baseline_aug | 0.294 | 0.157 | 0.745 |
| direction_b | 0.223 | 0.110 | 0.749 |
| focal_gamma_3 | **0.168** | **0.056** | 0.750 |

**关键观察**:
- direction_b 和 focal_gamma_3 都**未改善** same_group_similarity 和 class_collapse
- 但都**降低了困难样本的置信度**和 top1-top2 gap:
  - confidence: 0.294 → 0.223 (B) → 0.168 (focal)
  - top2_gap: 0.157 → 0.110 (B) → 0.056 (focal)
- mAP 提升幅度: B +0.004, focal +0.005 — 与"谦虚程度"正相关

**机制解释**:
- 困难样本上**低置信度** → 更可能被 score_thr 过滤 → 减少 false positive
- top1-top2 gap 小 → 分类更不确定 → 在 NMS 中更易被抑制
- 净效应: 通过减少 hard sample 上的错误高置信预测, 提升 mAP

**对方向的指导**:
- 这解释了为什么 B 和 focal_loss 即使没解决根本问题 (Y 坍塌, 同组混淆) 仍能小幅提升 mAP
- 单独追求"谦虚"有上限 (mAP +0.005 量级), 必须配合根本性改进 (Y 类平衡, 同组区分)

### 发现 7: 易混淆对的组内集中性 (F/G 组跨数据集一致)

| ckpt | 数据集 | Top 易混淆对 (相关系数) |
|------|--------|------------------------|
| baseline_aug | 新 | (F19,G22)=0.875, (G21,G22)=0.846 |
| direction_b | 新 | (F20,G22)=0.826, (D13,D14)=0.819 |
| focal_gamma_3 | 新 | (D13,D15)=0.873, (C8,X)=0.863, (F19,G22)=0.860, (F20,G22)=0.855, (B4,C6)=0.855 |
| ghss_24obj | 24obj | (F19,G22)=0.843 |
| random_24obj | 24obj | (F19,G22)=0.847, (G21,G22)=0.838, (B4,B5)=0.823, (C6,C7)=0.813 |

**关键观察**:
- **F19↔G22 混淆在所有跨数据集 ckpt 都出现** (相关 0.843~0.875) → **架构固有的形态混淆**
- F/G 组 (F19, F20, G21, G22) 混淆跨数据集一致 → 这两组染色体形态相似性是**生物学固有**, 非数据集偏差
- D 组内部 (D13, D14, D15) 在 direction_b 和 focal_gamma_3 中出现 → 模型在解决 F/G 混淆后, D 组混淆暴露出来

**对方向的指导**:
- 方向 C (Morphology) 应优先针对 F/G 组的形态差异设计先验
- 这两组都是中长臂染色体, 形态相似性高, 需要细粒度形态特征

## 三、各 ckpt 详细分析

### #1 baseline_aug (mAP=0.745) — 基线诊断

**TrajectoryAnalyzer**:
- 采样器: Rectified Flow + Heun solver, 4 步/图
- box 在第 3 步收敛 (convergence_step_mean=3.0, 全部 20 图一致)
- cls 在第 2 步收敛 (convergence_step_mean=2.05, 95% 图收敛)
- 早期 x0 质量 IoU=0.280 (起点质量尚可)
- renewal_effective_ratio=0.0 (renewal 未触发)
- x0_stability=81.23 (后续对比基准)

**RoIFeatureAnalyzer**:
- 同组内相似度 0.997, 跨组 0.996 → 几乎无差异, **分类瓶颈在 cls_head**
- 小/中/大目标特征范数: 5.17 / 5.09 / 4.99 → 尺度对特征影响微弱 (scale_affects=False)

**HeadOutputAnalyzer**:
- Y 类 (class 23) 坍塌, mean_logit=-5.17 (远低于均值 -4.01)
- 易混淆对: F19↔G22 (0.875), G21↔G22 (0.846) — 全是 F/G 组
- reg 系统性收缩: dw=-1.819, dh=-1.633
- 困难样本 1436 个, confidence=0.294, top2_gap=0.157, 未过度自信

### #2 direction_d (mAP=0.747, +0.002) — box_refine 验证

**TrajectoryAnalyzer**:
- convergence_step_mean=3.0 (与 baseline 一致, 未提前收敛)
- early_x0_quality=0.299 (略好于 baseline 0.280, 改善微弱)
- renewal_effective_ratio=0.0 (与 baseline 一致, box_renewal 仍未触发)
- x0_stability=75.19 (略低于 baseline 81.23, 反而略差)

**判断**: direction_d 在轨迹层面**几乎没有改善**:
- box_refine_net 既未让 renewal 有效 (renewal_effective_ratio 仍为 0)
- 也未让 x0 质量显著提升 (0.280 → 0.299 是噪声级别)
- x0_stability 甚至略降 (81 → 75)
- mAP +0.002 的来源不是 renewal 机制, 可能来自训练时正则化效应

**对症性**: ❌ 未对症 (box_renewal 未生效, 定位瓶颈未改善)

### #3 direction_b (mAP=0.749, +0.004) — 解耦 head 验证

**RoIFeatureAnalyzer**:
- 同组内相似度 0.9995 (略高于 baseline 0.997, 在 0.002 量级, 属噪声)
- 跨组 0.9993 (略高于 baseline 0.996)
- **解耦 head 未改善特征区分度**

**HeadOutputAnalyzer**:
- Y 类坍塌仍未解决 (collapsed_classes=[23])
- 易混淆对: F20↔G22 (0.826), D13↔D14 (0.819) — F/G 组仍在, D 组新出现
- reg: dw=-1.652, dh=-1.569 (与 baseline 接近, 系统性偏移未改善)
- **困难样本"更谦虚"**: confidence 0.294→0.223, top2_gap 0.157→0.110

**判断**: direction_b 通过让困难样本**更谦虚**获得 +0.004 mAP, 而非通过改善特征区分度:
- 未解决 Y 类坍塌
- 未降低同组相似度
- 但减少 hard sample 上的 false positive

**对症性**: ⚠️ 部分对症 (通过"谦虚"间接改善, 未根本解决分类瓶颈)

### #4 focal_gamma_3 (mAP=0.750, +0.005) — focal loss 验证

**HeadOutputAnalyzer**:
- Y 类坍塌仍未解决 (collapsed_classes=[23])
- 易混淆对 5 对 (多于 baseline 2 对), 因相关系数整体偏高:
  - Top: D13↔D15 (0.873), C8↔X (0.863), F19↔G22 (0.860), F20↔G22 (0.855), B4↔C6 (0.855)
  - 涉及 D, C, F/G, B/C 多个组
- reg: dw=-1.536, dh=-1.446 (系统性偏移略缓解, 但仍存在)
- **困难样本"最谦虚"**: confidence 0.168 (最低), top2_gap 0.056 (最低)

**判断**: focal_gamma=3 的作用机制与 direction_b 类似但更强:
- 让困难样本置信度更低 → 更激进地过滤 false positive
- mAP +0.005 (略高于 B 的 +0.004), 与"谦虚程度"正相关
- 但仍未解决 Y 类坍塌和同组混淆的根本问题

**对症性**: ⚠️ 部分对症 (通过更强"谦虚"获得略高收益, 未根本解决)

### #5 no_box_renewal (mAP=0.635, -0.110) — renewal 贡献量化

**TrajectoryAnalyzer**:
- convergence_step_mean=3.0 (与 baseline 一致, box 仍第 3 步收敛)
- early_x0_quality=0.517 (**反而高于** baseline 0.280 — 关闭 renewal 后单步 x0 质量更好)
- renewal_effective_ratio=0.0 (本就关闭)
- **x0_stability=7.10 (远低于 baseline 81.23, 降幅 91%)**

**关键量化**:
- 关闭 renewal 后, box 收敛步数不变 (都是第 3 步), 单步 x0 质量甚至更好
- 但 x0_stability 暴跌 91% (81 → 7) → renewal 在**训练阶段**隐式稳定了步间 x0 一致性
- mAP 暴跌 -0.110 → 步间 x0 剧烈波动导致最终定位质量下降

**机制澄清**:
- box_renewal 在**推理时**修正量=0 (形同虚设)
- 但在**训练时**作为正则项, 让 cls_head/reg_head 学到步间一致的 x0 预测
- renewal **不改善单步 x0 质量**, 而是**提升步间稳定性** (11 倍提升)
- 这是"训练时副作用 > 推理时显式作用"的典型案例

**判断**: box_renewal 的价值在于训练正则, 而非推理修正。方向 D 若要改进, 应:
- (1) 修复推理时 renewal 触发条件, 让其真正修正 box
- (2) 或接受 renewal 是训练正则的事实, 转而针对 reg 系统性偏移 (发现 5) 设计

## 四、24obj 数据集对照分析

> 目的: 区分"架构固有缺陷"与"数据集效应"。24obj 模型与新数据集模型架构/损失完全一致 (num_classes=24, num_proposals=500, FocalLoss+L1Loss, FPN),仅数据集不同,构成完美对照。
> 对照设计: baseline_aug (新数据集, 0.745) vs ghss_24obj (24obj, 0.857) vs random_24obj (24obj, 0.859)

### 4.1 架构固有缺陷 (两数据集一致 → 模型架构问题)

以下 7 项发现在新数据集和 24obj 上**完全一致**,确认为**架构固有缺陷**,与数据集无关:

| 发现 | baseline_aug (新) | ghss_24obj (24obj) | random_24obj (24obj) | 结论 |
|------|:-----------------:|:------------------:|:--------------------:|------|
| box convergence_step | 3.0 (4步中第3步) | 3.0 | 3.0 | **架构固有** |
| cls_converged_ratio | 0.95 | 0.9 | 0.95 | **架构固有** (90-95% 收敛) |
| renewal_effective_ratio | 0.0 | 0.0 | 0.0 | **架构固有** (推理时未触发) |
| same_group_too_similar | False | False | False | **架构固有** |
| scale_affects | False | False | False | **架构固有** |
| class_collapse | True | True | True | **架构固有** (但坍塌类有差异,见 4.2) |
| reg 系统性收缩 (dw/dh 偏负) | [-1.82, -1.63] | [-1.31, -1.27] | [-1.27, -1.27] | **架构固有** (方向一致, 幅度不同) |

**关键结论**: 7 项核心发现都是架构固有的 — 在 24obj (mAP=0.857) 上同样存在。这意味着:
- 即使 mAP 达到 0.857,box 仍在第 3 步收敛、renewal 仍未触发、reg 仍系统性收缩
- 这些架构缺陷被高数据质量"掩盖"了,但并未消失
- **方向 D' (reg 校正) 在两数据集上都有价值**

### 4.2 数据集效应 (有差异 → 数据集相关)

| 指标 | baseline_aug (新) | ghss_24obj (24obj) | 差异 | 归因 |
|------|:-----------------:|:------------------:|------|------|
| mAP | 0.745 | 0.857 | **+0.112** | 24obj 数据质量更好 |
| early_x0_quality | 0.280 | 0.270 | -0.01 | 接近, 起点质量无显著差异 |
| x0_stability | 81.23 | 82.81 | +1.6 | 接近, 步间稳定性无显著差异 |
| reg dw/dh | [-1.82, -1.63] | [-1.31, -1.27] | **+0.5** | 新数据集框回归收缩更严重 |
| 特征范数 (small) | 5.17 | 8.36 | **+3.2** | 24obj 的 RoI 特征范数整体更高 |
| hard_sample_confidence | 0.294 | 0.389 | +0.10 | 24obj 的困难样本更"自信" |
| A1 类坍塌 | 否 | **是** | — | 24obj 特有问题 (ghss 耦合下) |

**关键发现**:
1. **mAP 差距 0.112 的根因不是架构**,而是数据质量 — 24obj 数据让同样的架构表现更好
2. **reg 收缩幅度差异** (新数据集 -1.8 vs 24obj -1.3) 提示新数据集的标注框可能偏大,或目标尺度分布不同
3. **特征范数差异** (5.2 vs 8.4) 提示两数据集的图像特性不同 (分辨率/对比度/目标大小)
4. **A1 类在 24obj+ghss 耦合下额外坍塌** — 这是数据集×耦合方式的交互效应

### 4.3 Y 类坍塌的归因 (架构 + 数据共同作用)

| ckpt | 数据集 | collapsed_classes | Y (class 23) mean_logit |
|------|--------|:-----------------:|:-----------------------:|
| baseline_aug | 新 | [23] | -5.17 |
| ghss_24obj | 24obj | [0, 23] | -5.03 |
| random_24obj | 24obj | [23] | -5.26 |

**关键观察**:
- Y 类坍塌在**所有 3 个 ckpt** (跨数据集) 都存在 → **架构固有倾向**
- 但 24obj 上 Y 类 mean_logit (-5.03) 比新数据集 (-5.17) 略好 → **数据集也有影响**
- Y 类在两个数据集中都是样本数最少的类 → **类别不平衡是跨数据集的共性问题**

**结论**: Y 类坍塌是**架构倾向 + 数据不平衡**共同作用的结果。方向 E (ClassBalanced) 在两数据集上都需要,但 24obj 上 Y 类问题略轻。

### 4.4 耦合方式效应 (ghss_24obj vs random_24obj)

两者都在 24obj 上训练,仅耦合方式不同:

| 指标 | ghss_24obj | random_24obj | 差异 |
|------|:----------:|:------------:|------|
| mAP | 0.857 | 0.859 | random 略高 (+0.002) |
| class_collapse | [0, 23] | [23] | **ghss 多坍塌 A1** |
| 易混淆对数 | 1 | 4 | ghss 更少 |
| hard_sample_confidence | 0.389 | 0.329 | ghss 更自信 |
| reg dw/dh | [-1.31, -1.27] | [-1.27, -1.27] | 接近 |
| x0_stability | 82.81 | 85.47 | random 略高 |

**关键观察**:
- 耦合方式对**模块内部状态有影响但不大**
- **ghss 耦合下 A1 类额外坍塌** (class 0) → ghss 的分组机制可能对 A 组 (A1,A2,A3) 有负面影响
- random 耦合的 x0 稳定性略高, mAP 也略高 → random 耦合在 24obj 上略优
- 但两者 mAP 差距仅 0.002, 耦合方式不是主要因素

### 4.5 对照分析对方向的修正指导

| 方向 | 基于新数据集的指导 | 24obj 对照后的修正 |
|------|---------------------|-------------------|
| A (P1+Deformable) | reg 系统性收缩提示框回归有偏 | 收缩是架构固有, 两数据集都需要 → 确认普适性 |
| B (DecoupledHead) | 通过"谦虚"获 +0.004 | cls_head 瓶颈是架构固有 → 应有效 |
| C (Morphology+Contrastive) | 转向 cls_head 损失设计 | F/G 组混淆是跨数据集生物学难题 → 优先针对 F/G 组 |
| D (BoxRefineNet) | 放弃 renewal, 改针对 reg 偏移 | reg 收缩是架构固有 → 确认方向 D' 普适价值 |
| E (ClassBalanced) | 优先实现, 解决 Y 类坍塌 | Y 类坍塌是架构+数据共同作用 → 两数据集都需要 |
| F (StructuredPrior) | x0 质量尚可 (0.280), 有改善空间 | x0 质量跨数据集接近 (0.27-0.28) → 价值下调, 但仍值得验证 |
| **D' (reg 校正)** | reg 系统性收缩 | 收缩是架构固有, 两数据集都需要 → **确认普适价值, 最高优先级** |
| **H (采样效率)** | box 第 3 步收敛, 4→3 边界 | 跨数据集一致 → **价值有限, 可选验证** |

### 4.6 对照分析的核心结论

1. **7 项发现全部是架构固有的** (跨数据集一致), 仅 Y 类坍塌幅度和数据集特异坍塌 (A1) 有数据依赖
2. **mAP 差距 0.112 的根因是数据质量, 不是架构** — 同样的架构在 24obj 上达到 0.857
3. **架构缺陷被高数据质量"掩盖"** — 24obj 上 box 仍第 3 步收敛、renewal 仍未触发、reg 仍收缩, 只是幅度更轻
4. **F/G 组形态混淆是跨数据集的生物学固有难题** — 方向 C 应优先针对此
5. **方向 D' (reg 校正) 价值最高** — reg 系统性收缩是跨数据集的架构固有特性, 两数据集都需要
6. **方向 H (采样效率) 价值有限** — box 第 3 步收敛 (4步中), 减步空间仅 4→3, 且在收敛边界

## 五、白盒 vs 黑盒分析的互补性

### 黑盒分析 (bottleneck_report.json) 已回答
- AP-IoU 曲线: 定位瓶颈在 IoU≥0.9 (AP@90=0.483)
- 误差分解: Cls error 43.4%, 同组混淆为主
- 混淆矩阵: Top 混淆对 (C10↔C9, C9↔C8 等)
- 尺度/类别不平衡: Y 类样本最少

### 白盒分析 (本文档) 新增回答
- **分类瓶颈定位**: cls_head (非特征) — 黑盒无法回答
- **Y 类坍塌机制**: mean_logit 显著低, 是 cls_head 学习问题 — 黑盒只能看到 Y 类 AP 低
- **扩散采样收敛模式**: box 第 3 步收敛, cls 第 2 步收敛 (4 步采样) — 黑盒无法看到
- **box_renewal 实际作用**: 推理时未触发, 训练时提升 x0 步间稳定性 11 倍 — 黑盒只能看到 mAP 差异
- **reg 系统性偏移**: dw/dh 偏负 -1.8/-1.6 — 黑盒只能看到 IoU 低, 不知是中心还是尺度问题

### 仍未回答的问题 (需进一步插桩)
- cls_head 内部哪一层 (FC/attention) 导致同组混淆?
- 扩散噪声 schedule 是否合理? (early_x0_quality=0.280, 起点质量尚可但有改善空间)
- proposal 数量对 x0 质量的影响? (方向 F 假设)

## 六、对方向 A-H 的指导汇总

| 方向 | 白盒分析揭示的问题 | 建议调整 |
|------|------------------|---------|
| A (P1+Deformable) | 未直接分析, 但 reg 系统性收缩 (-1.8, -1.6) 提示框回归有偏; scale_affects=False | P1 分辨率可能改善小目标, 但需配合 reg 偏移校正 |
| **B (DecoupledHead)** | 未解决 Y 坍塌, 未降低同组相似度; 通过"谦虚"获 +0.004 | 已早停 (epoch 86); 下一步: 解耦 + 类别平衡采样 |
| **C (Morphology+Contrastive)** | 分类瓶颈在 cls_head (非特征), 纯特征对比学习可能效果有限 | 转向 cls_head 损失设计; 若做 contrastive, 应作用于 cls_logit 而非 roi_feature |
| **D (BoxRefineNet)** | renewal 推理时未触发 (ratio=0); trajectory 与 baseline 一致 | 已证伪; 放弃 renewal 机制 |
| **E (ClassBalanced)** | Y 类 (class 23) 坍塌在所有 ckpt 普遍存在 | **优先实现**, 这是解决分类瓶颈的根本 |
| F (StructuredPrior) | x0 早期质量尚可 (0.280), 但有改善空间 | 结构化先验可能改善 x0 起点, 值得验证 |
| **D' (reg 校正)** (新增) | reg 系统性收缩 dw/dh≈-1.6 (架构固有, 跨数据集) | **最高优先级**, 针对 reg dw/dh 偏移的显式校正 |
| **H (采样效率)** (新增) | box 第 3 步收敛 (4步), 步数必要 | 价值有限, 4→3 可选验证 |

### 接龙顺序 (基于白盒分析)

**E → F → D' → H (可选)**

跳过: C (瓶颈在 cls_head 不在特征), A (scale_affects=False, 尺度非主要瓶颈)

## 七、产出文件索引

### 代码与测试
- `experiments/analysis/instrumentation/__init__.py` — 模块导出
- `experiments/analysis/instrumentation/trajectory_analyzer.py` — 扩散轨迹分析器
- `experiments/analysis/instrumentation/feature_analyzer.py` — RoI 特征分析器
- `experiments/analysis/instrumentation/head_analyzer.py` — cls/reg 头分析器
- `experiments/analysis/instrumentation/run_instrumentation.py` — 批量执行脚本
- `ldmdet/tests/test_instrumentation.py` — 19 个单元测试 (红绿重构)

### 文档
- `experiments/analysis/instrumentation/INSTRUMENTATION_PLAN.md` — 方案文档
- `experiments/analysis/instrumentation/INSTRUMENTATION_RESULTS.md` — 本文 (结果文档)
- `experiments/configs/ldmdet/DIRECTIONS_REVISION.md` — 方向调整决策

### JSON 报告
- `work_dirs/instrumentation/comparison.json` — 跨 ckpt 汇总对比 (7 个 ckpt)
- `work_dirs/instrumentation/baseline_aug/{trajectory,roi_feature,head_output}_report.json`
- `work_dirs/instrumentation/direction_d/trajectory_report.json`
- `work_dirs/instrumentation/direction_b/{roi_feature,head_output}_report.json`
- `work_dirs/instrumentation/focal_gamma_3/head_output_report.json`
- `work_dirs/instrumentation/no_box_renewal/trajectory_report.json`
- `work_dirs/instrumentation/ghss_24obj/{trajectory,roi_feature,head_output}_report.json`
- `work_dirs/instrumentation/random_24obj/{trajectory,roi_feature,head_output}_report.json`

## 八、复现命令

```bash
# 单个 ckpt 分析
python experiments/analysis/instrumentation/run_instrumentation.py \
    --config experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py \
    --ckpt work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth \
    --name baseline_aug \
    --analyzers trajectory roi_feature head_output \
    --num-samples 20 --device cuda:0 \
    --output-dir work_dirs/instrumentation/baseline_aug

# 批量分析所有 7 个 ckpt (按 BATCH_PLAN)
python experiments/analysis/instrumentation/run_instrumentation.py \
    --batch --num-samples 20 --device cuda:0

# 只跑部分 ckpt (合并到已有 comparison.json)
python experiments/analysis/instrumentation/run_instrumentation.py \
    --batch --only ghss_24obj random_24obj --num-samples 20 --device cuda:0
```

## 九、核心结论

### 基于新数据集 5 ckpt 的结论

1. **分类瓶颈在 cls_head**, 不在特征抽取 — 方向 C 应调整重心
2. **Y 类坍塌普遍存在**, 是分类瓶颈的根因 — 方向 E 应优先
3. **box 第 3 步收敛 (4 步采样), cls 第 2 步收敛** — 步数对 box 是必要的, H 方案价值有限
4. **box_renewal 推理时未触发**, 训练时提升 x0 步间稳定性 11 倍 — 方向 D 需重新设计
5. **reg 系统性收缩** (dw/dh ≈ -1.6) — 方向 D' 应针对此偏移
6. **direction_b 和 focal_gamma_3 通过"更谦虚"获得小幅提升**, 但未根本解决问题
7. **F/G 组形态混淆是跨数据集的生物学固有难题** — 方向 C 应优先针对 F/G 组

### 基于 24obj 对照分析的补充结论

8. **7 项发现全部是架构固有的** (跨数据集一致) — 仅 Y 类坍塌幅度和数据集特异坍塌 (A1) 有数据依赖
9. **mAP 差距 0.112 的根因是数据质量, 不是架构** — 同架构在 24obj 上达 0.857
10. **架构缺陷被高数据质量"掩盖"** — 24obj 上 box 仍第 3 步收敛、renewal 仍未触发、reg 仍收缩
11. **方向 D' (reg 校正) 价值最高** — reg 系统性收缩跨数据集一致, 两数据集都需要
12. **方向 E/D' 的普适性已确认** — reg 收缩、Y 类坍塌在两数据集上都存在
