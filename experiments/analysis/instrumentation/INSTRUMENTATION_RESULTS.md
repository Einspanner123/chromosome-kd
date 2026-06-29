# 白盒插桩分析结果

> 本文档记录对 7 个 checkpoint 执行白盒插桩分析的实验结果与发现。
> 执行时间: 2026-06-29 (初始 5 ckpt) / 2026-06-30 (追加 24obj 对照 2 ckpt)
> 方案文档: [INSTRUMENTATION_PLAN.md](./INSTRUMENTATION_PLAN.md)
> 原始 JSON 报告: `work_dirs/instrumentation/<ckpt_name>/{trajectory,roi_feature,head_output}_report.json`
> 汇总对比: `work_dirs/instrumentation/comparison.json`

## 一、执行概况

### 分析矩阵执行情况

| # | Checkpoint | 数据集 | mAP | Trajectory | RoIFeature | HeadOutput | 状态 |
|---|-----------|--------|-----|:---------:|:---------:|:---------:|:----:|
| 1 | baseline_aug | Chromosome20240904 | 0.745 | ✓ | ✓ | ✓ | 完成 |
| 2 | direction_d | Chromosome20240904 | 0.747 | ✓ | — | — | 完成 |
| 3 | direction_b | Chromosome20240904 | 0.749 | — | ✓ | ✓ | 完成 |
| 4 | focal_gamma_3 | Chromosome20240904 | 0.750 | — | — | ✓ | 完成 |
| 5 | no_box_renewal | Chromosome20240904 | 0.635 | ✓ | — | — | 完成 |
| 6 | ghss_24obj | 24_chromosomes_object | 0.857 | ✓ | ✓ | ✓ | 完成 |
| 7 | random_24obj | 24_chromosomes_object | 0.859 | ✓ | ✓ | ✓ | 完成 |

### 执行参数
- 每个 ckpt 采样 **50 张验证图**
- RoIFeature / HeadOutput 累积约 **25000 个 RoI 样本** (50 图 × ~500 proposals)
- Trajectory 采集 **200 步采样轨迹** (与训练时 `num_steps=200` 一致)
- 设备: `cuda:0`
- 数据集: `Chromosome20240904_NoAug_NoResize_coco/` (与训练一致)

### 染色体类别索引映射 (cat_names)
```
0:A1  1:A2  2:A3   | 3:B4  4:B5        | 5:C10  6:C11  7:C12  8:C6  9:C7  10:C8  11:C9
12:D13 13:D14 14:D15 | 15:E16 16:E17 17:E18 | 18:F19 19:F20 | 20:G21 21:G22 | 22:X | 23:Y
```

## 二、核心发现 (跨 ckpt 综合)

### 发现 1: 分类瓶颈定位 — cls_head 而非特征提取

| ckpt | 同组内相似度 (均值) | 跨组相似度 | same_group_too_similar | 结论 |
|------|:------------------:|:----------:|:---------------------:|------|
| baseline_aug | 0.999 | 0.9988 | False | 瓶颈在 cls_head |
| direction_b | 0.999 | 0.9993 | False | 瓶颈在 cls_head |

**关键观察**:
- 同组类别余弦相似度 (0.999) 与跨组相似度 (0.9988~0.9993) **几乎无差异**
- `same_group_too_similar=False` 说明特征层面同组并未"过度相似"
- 但瓶颈报告显示 Cls error 43.4% 且 Top 混淆全是同组 (C8↔C9, C10↔C9 等)

**结论**: 分类瓶颈**不在 RoI 特征抽取**, 而在 **cls_head** 在已抽取特征上未能拉开同组类间距。这解释了为什么 direction_b (解耦 head) 和 focal_gamma_3 (分类损失) 都能小幅提升 mAP — 它们都作用于 cls_head。

**对方向的指导**:
- 方向 C (Morphology+Contrastive) 若仅做特征层对比学习, 效果可能有限
- 应优先改进 cls_head 的损失/结构, 而非特征抽取

### 发现 2: Y 类 (class 23) 坍塌是普遍问题

| ckpt | class_collapse | collapsed_classes | Y 类 mean_logit | 其他类平均 mean_logit |
|------|:--------------:|:-----------------:|:---------------:|:--------------------:|
| baseline_aug | True | [23] | -5.19 | -4.15 |
| direction_b | True | [23] | -4.03 | -3.51 |
| focal_gamma_3 | True | [23] | -3.57 | -2.85 |

**关键观察**:
- 三个 ckpt 全部检测到 Y 类坍塌 (class 23)
- Y 类 mean_logit 始终显著低于其他类平均 (低 0.5~1.0)
- 即使 focal_gamma=3 (理论上抑制易分类、放大难分类) 也未解决

**根因**: Y 染色体在数据集中样本数最少 (见 `EXPERIMENT_ANALYSIS.md` 类别不平衡分析), 训练信号弱, cls_head 学不到有效的 Y 类决策边界。

**对方向的指导**:
- 方向 E (ClassBalanced) 是解决此问题的关键, 应优先实现类别平衡采样 + 重加权
- 在 E 实现前, 所有分类相关改进 (B, focal_gamma) 都无法根治 Y 类问题

### 发现 3: 扩散采样步数充分, box 在第 3 步收敛

> 修正说明: 初版文档因 trajectory 分析 bug (跨图混合) 曾错误报告 "n_steps=200, box第1步收敛"。修复后 (每图独立分析) 真实结果如下。详见附录 A。

| ckpt | n_steps/图 | box convergence_step | cls convergence_step | cls_converged |
|------|:---------:|:--------------------:|:--------------------:|:-------------:|
| baseline_aug | 4 | 3.0 | 2.05 | 0.95 |
| direction_d | 4 | 3.0 | 2.0 | 1.0 |
| no_box_renewal | 4 | 3.0 | 2.0 | 1.0 |

**关键观察**:
- 采样器: Rectified Flow + Heun solver, `sampling_timesteps=4` (每图 4 步)
- **box 在第 3 步收敛** (4 步中倒数第 2 步), 不是第 1 步
- **cls 在第 2 步收敛** (90-100% 图收敛), 比 box 更早
- 4 步采样对 box 是必要的: 减到 3 步刚好在收敛边界, 减到 2 步 box 未收敛

**深层含义**:
- box 晚于 cls 收敛 (3 vs 2) → 定位比分类需要更多采样步
- 这与瓶颈报告 (定位瓶颈在 IoU≥0.9) 一致: 高 IoU 定位需要充分的扩散步数
- cls 早期收敛但 mAP 仍受 cls 误差影响 → cls_head 的预测质量不是步数问题, 而是容量/训练问题 (见发现 1, 2)

**对方向 H (采样效率) 的影响**:
- 加速空间有限: 4→3 步仅 1.33× 加速, 且在收敛边界, mAP 可能下降
- 4→2 或 4→1 会损害 box 收敛, 不可行
- **H 方案价值大幅降低**, 从"高优先级"降为"可选验证"

### 发现 4: box_renewal 推理未触发, 但训练时大幅稳定 x0

| ckpt | renewal_effective_ratio | early_x0_quality | x0_stability |
|------|:----------------------:|:----------------:|:------------:|
| baseline_aug | 0.0 | 0.280 | **81.23** |
| direction_d | 0.0 | 0.299 | 75.19 |
| no_box_renewal | 0.0 | **0.517** | **7.10** |

**关键观察**:
- `renewal_effective_ratio=0.0` 在所有 ckpt 成立 — box_renewal 在**推理时确实未触发** (此结论可靠)
- direction_d (BoxRefineNet) 也未让 renewal 生效 — mAP +0.002 不是来自 renewal

**但 mAP 暴跌 0.110**: no_box_renewal mAP=0.635 vs baseline 0.745, 差距 -0.110

**反直觉发现与解释**:
- no_box_renewal 的 `early_x0_quality=0.517` **反而高于** baseline 的 0.280 — 关闭 renewal 后单步 x0 质量更好
- 但 `x0_stability=7.10` **远低于** baseline 的 81.23 — 降幅 **91%** (初版 bug 数据误报 49%)
- 解释: renewal **不改善单步 x0 质量**, 但**大幅提升步间稳定性** (从 7 → 81, 11 倍)
- renewal 在训练时作为正则, 让 x0 预测在步间保持一致; 关闭后 x0 单步质量虽好, 但步间剧烈波动, 最终定位质量下降

**对方向的指导**:
- 方向 D (BoxRefineNet) 的当前实现**未对症** — renewal 在推理时未触发, 改进 renewal_head 不会有效
- 应重新审视 renewal 的触发条件 (是否阈值过严?), 或将 box_refine 改为**显式作用于 reg_head 输出**而非 renewal 机制
- reg 系统性偏移 (见发现 5) 是更值得针对的问题

### 发现 5: 回归头存在系统性 w/h 收缩偏移

| ckpt | delta_mean | per_dim_mean [dx, dy, dw, dh] | is_conservative |
|------|:---------:|:----------------------------:|:---------------:|
| baseline_aug | -0.805 | [-0.008, 0.127, **-1.730**, **-1.608**] | False |
| direction_b | -0.798 | [-0.022, 0.050, **-1.652**, **-1.569**] | False |
| focal_gamma_3 | -0.766 | [0.002, -0.083, **-1.536**, **-1.446**] | False |

**关键观察**:
- 所有 ckpt 的 dw, dh 均显著偏负 (-1.4 ~ -1.7)
- 这**不是保守回归** (is_conservative=False, 因为 |delta_mean| > 0.1), 而是**系统性缩小框**
- dx, dy 接近 0, 说明中心定位无偏, 但尺度预测系统性偏小

**可能根因**:
- 训练数据中 GT 框可能偏大 (标注习惯), 模型学习到"收缩"偏差
- 或扩散过程的 box 初始化偏大, reg_head 学习补偿

**对方向的指导**:
- 方向 D (BoxRefineNet) 应针对此系统性偏移设计, 而非 renewal 机制
- 可考虑: (1) 数据增强中加 box 尺度抖动; (2) reg_head 的 dw/dh 加正则化约束; (3) 后处理时尺度校准

### 发现 6: direction_b 与 focal_gamma_3 的共同机制 — "更谦虚"

| ckpt | hard_sample_count | hard_sample_confidence | hard_sample_top2_gap | mAP |
|------|:-----------------:|:----------------------:|:--------------------:|:---:|
| baseline_aug | 3599 | 0.309 | 0.171 | 0.745 |
| direction_b | 3543 | 0.223 | 0.110 | 0.749 |
| focal_gamma_3 | 3904 | **0.168** | **0.056** | 0.750 |

**关键观察**:
- direction_b 和 focal_gamma_3 都**未改善** same_group_similarity 和 class_collapse
- 但都**降低了困难样本的置信度**和 top1-top2 gap:
  - confidence: 0.309 → 0.223 (B) → 0.168 (focal)
  - top2_gap: 0.171 → 0.110 (B) → 0.056 (focal)
- mAP 提升幅度: B +0.004, focal +0.005 — 与"谦虚程度"正相关

**机制解释**:
- 困难样本上**低置信度** → 更可能被 score_thr 过滤 → 减少 false positive
- top1-top2 gap 小 → 分类更不确定 → 在 NMS 中更易被抑制
- 净效应: 通过减少 hard sample 上的错误高置信预测, 提升 mAP

**对方向的指导**:
- 这解释了为什么 B 和 focal_loss 即使没解决根本问题 (Y 坍塌, 同组混淆) 仍能小幅提升 mAP
- 单独追求"谦虚"有上限 (mAP +0.005 量级), 必须配合根本性改进 (Y 类平衡, 同组区分)
- 方向 C 的对比学习若能降低同组 confidence gap, 可能产生类似"谦虚"效应

### 发现 7: 易混淆类对的组内集中性

| ckpt | Top 易混淆对 (相关系数) | 涉及组 |
|------|------------------------|--------|
| baseline_aug | (F19,G22)=0.880, (G21,G22)=0.828, (F19,G21)=0.803 | F/G 组 |
| direction_b | (F20,G22)=0.826, (D13,D14)=0.819 | F/G, D 组 |
| focal_gamma_3 | (D13,D15)=0.873, (C8,X)=0.863, (F19,G22)=0.860, (F20,G22)=0.855, (B4,C6)=0.855, (D13,D14)=0.848, (C11,C12)=0.847, (F19,G21)=0.838, (D14,D15)=0.838, (C10,C9)=0.828 | F/G, D, C, B/C |

**关键观察**:
- F/G 组混淆 (F19, F20, G21, G22) 在所有 ckpt 中都出现 → 形态最相似的染色体对
- focal_gamma_3 检测到更多混淆对 (10 对 vs baseline 3 对), 但这不是 focal 表现更差, 而是相关系数整体偏高, 更多对越过 0.8 阈值
- D 组内部 (D13, D14, D15) 在 direction_b 和 focal_gamma_3 中新出现 → 可能是模型在解决 F/G 混淆后, D 组混淆暴露出来

**对方向的指导**:
- 方向 C (Morphology) 应优先针对 F/G 组和 D 组的形态差异设计先验
- 这两组都是中长臂染色体, 形态相似性高, 需要细粒度形态特征

## 三、各 ckpt 详细分析

### #1 baseline_aug (mAP=0.745) — 基线诊断

**TrajectoryAnalyzer** (修复后, 每图独立分析):
- 采样器: Rectified Flow + Heun solver, 4 步/图
- box 在第 3 步收敛 (convergence_step_mean=3.0, 全部 20 图一致)
- cls 在第 2 步收敛 (convergence_step_mean=2.05, 95% 图收敛)
- 早期 x0 质量 IoU=0.280 (起点质量尚可)
- renewal_effective_ratio=0.0 (renewal 未触发)
- x0_stability=81.23 (后续对比基准)

**RoIFeatureAnalyzer**:
- 同组内相似度 0.999, 跨组 0.9988 → 几乎无差异, **分类瓶颈在 cls_head**
- 小/中/大目标特征范数: 5.02 / 4.93 / 4.80 → 尺度对特征影响微弱 (scale_affects=False)

**HeadOutputAnalyzer**:
- Y 类 (class 23) 坍塌, mean_logit=-5.19 (远低于均值 -4.15)
- 易混淆对: F19↔G22 (0.880), G21↔G22 (0.828), F19↔G21 (0.803) — 全是 F/G 组
- reg 系统性收缩: dw=-1.730, dh=-1.608
- 困难样本 3599 个, confidence=0.309, top2_gap=0.171, 未过度自信

### #2 direction_d (mAP=0.747, +0.002) — box_refine 验证

**TrajectoryAnalyzer** (修复后):
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
- 同组内相似度 0.9996 (略高于 baseline 0.9991, 反而更相似!)
- 跨组 0.9993 (略高于 baseline 0.9988)
- **解耦 head 未改善特征区分度** (甚至略恶化, 但在 0.001 量级, 属噪声)

**HeadOutputAnalyzer**:
- Y 类坍塌仍未解决 (collapsed_classes=[23])
- 易混淆对: F20↔G22 (0.826), D13↔D14 (0.819) — F/G 组仍在, D 组新出现
- reg: dw=-1.652, dh=-1.569 (与 baseline 接近, 系统性偏移未改善)
- **困难样本"更谦虚"**: confidence 0.309→0.223, top2_gap 0.171→0.110

**判断**: direction_b 通过让困难样本**更谦虚**获得 +0.004 mAP, 而非通过改善特征区分度:
- 未解决 Y 类坍塌
- 未降低同组相似度
- 但减少 hard sample 上的 false positive

**对症性**: ⚠️ 部分对症 (通过"谦虚"间接改善, 未根本解决分类瓶颈)

### #4 focal_gamma_3 (mAP=0.750, +0.005) — focal loss 验证

**HeadOutputAnalyzer**:
- Y 类坍塌仍未解决 (collapsed_classes=[23])
- 易混淆对 10 对 (远多于 baseline 3 对), 但这是因相关系数整体偏高, 非表现更差:
  - Top: D13↔D15 (0.873), C8↔X (0.863), F19↔G22 (0.860), F20↔G22 (0.855), B4↔C6 (0.855)
  - 涉及 D, C, F/G, B/C 多个组
- reg: dw=-1.536, dh=-1.446 (系统性偏移略缓解, 但仍存在)
- **困难样本"最谦虚"**: confidence 0.168 (最低), top2_gap 0.056 (最低)
- hard_sample_count 3904 (最多, 因更多样本被识别为 hard)

**判断**: focal_gamma=3 的作用机制与 direction_b 类似但更强:
- 让困难样本置信度更低 → 更激进地过滤 false positive
- mAP +0.005 (略高于 B 的 +0.004), 与"谦虚程度"正相关
- 但仍未解决 Y 类坍塌和同组混淆的根本问题

**对症性**: ⚠️ 部分对症 (通过更强"谦虚"获得略高收益, 未根本解决)

### #5 no_box_renewal (mAP=0.635, -0.110) — renewal 贡献量化

**TrajectoryAnalyzer** (修复后):
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

## 四、对方向 A-F 的指导汇总

| 方向 | 白盒分析揭示的问题 | 建议调整 |
|------|------------------|---------|
| A (P1+Deformable) | 未直接分析, 但 reg 系统性收缩 (-1.7, -1.6) 提示框回归有偏 | P1 分辨率可能改善小目标, 但需配合 reg 偏移校正 |
| **B (DecoupledHead)** | 未解决 Y 坍塌, 未降低同组相似度; 通过"谦虚"获 +0.004 | 下一步: 解耦 + 类别平衡采样 (解决 Y 坍塌) |
| **C (Morphology+Contrastive)** | 分类瓶颈在 cls_head (非特征), 纯特征对比学习可能效果有限 | 转向 cls_head 损失设计; 若做 contrastive, 应作用于 cls_logit 而非 roi_feature |
| **D (BoxRefineNet)** | renewal 推理时未触发 (delta=0); reg 系统性收缩 (-1.7, -1.6) | 放弃 renewal 机制, 改为针对 reg dw/dh 偏移的显式校正 |
| **E (ClassBalanced)** | Y 类 (class 23) 坍塌在所有 ckpt 普遍存在 | 优先实现, 这是解决分类瓶颈的根本 |
| F (StructuredPrior) | x0 早期质量尚可 (0.280), 但有改善空间; renewal 稳定 x0 步间一致性 | 结构化先验可能改善 x0 起点, 值得验证 |

### 新方向建议

**方向 H: 扩散采样效率优化** (价值已下调)
- 依据 (修正后): box 第 3 步收敛 (4 步中), cls 第 2 步收敛; 步数对 box 是必要的
- 思路:
  - (H1) 减少采样步数 4→3, 验证 mAP 是否保持 (仅 1.33× 加速, 且在收敛边界)
  - (H2) 非对称采样: box 早停 + cls 继续精化 (但 cls 比 box 更早收敛, 意义不大)
- 预期收益: 有限 (最多 1.33× 加速), mAP 可能下降
- **结论: H 方案价值大幅降低, 降为可选验证**

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
- **reg 系统性偏移**: dw/dh 偏负 -1.7/-1.6 — 黑盒只能看到 IoU 低, 不知是中心还是尺度问题

### 仍未回答的问题 (需进一步插桩)
- cls_head 内部哪一层 (FC/attention) 导致同组混淆?
- 扩散噪声 schedule 是否合理? (early_x0_quality=0.280, 起点质量尚可但有改善空间)
- proposal 数量对 x0 质量的影响? (方向 F 假设)

## 六、24obj 数据集对照分析

> 目的: 区分"架构固有缺陷"与"数据集效应"。24obj 模型与新数据集模型架构/损失完全一致 (num_classes=24, num_proposals=500, FocalLoss+L1Loss, FPN),仅数据集不同,构成完美对照。
> 对照设计: baseline_aug (新数据集, 0.745) vs ghss_24obj (24obj, 0.857) vs random_24obj (24obj, 0.859)

### 6.1 架构固有缺陷 (两数据集一致 → 模型架构问题)

以下 7 项发现在新数据集和 24obj 上**完全一致**,确认为**架构固有缺陷**,与数据集无关 (trajectory 数据为修复后真实值):

| 发现 | baseline_aug (新) | ghss_24obj (24obj) | random_24obj (24obj) | 结论 |
|------|:-----------------:|:------------------:|:--------------------:|------|
| box convergence_step | 3.0 (4步中第3步) | 3.0 | 3.0 | **架构固有** |
| cls_converged_ratio | 0.95 | 0.9 | 0.95 | **架构固有** (90-95% 收敛) |
| renewal_effective_ratio | 0.0 | 0.0 | 0.0 | **架构固有** (推理时未触发) |
| same_group_too_similar | False | False | False | **架构固有** |
| scale_affects | False | False | False | **架构固有** |
| class_collapse | True | True | True | **架构固有** (但坍塌类有差异,见 6.2) |
| reg 系统性收缩 (dw/dh 偏负) | [-1.73, -1.61] | [-1.34, -1.29] | [-1.28, -1.28] | **架构固有** (方向一致, 幅度不同) |

**关键结论**: 前文 7 项核心发现中,除"Y 类坍塌"和"direction_b/focal 谦虚效应"外,其余都是架构固有的 — 在 24obj (mAP=0.857) 上同样存在。这意味着:
- 即使 mAP 达到 0.857,box 仍在第 3 步收敛、renewal 仍未触发、reg 仍系统性收缩
- 这些架构缺陷被高数据质量"掩盖"了,但并未消失
- **方向 D' (reg 校正) 在两数据集上都有价值**; H (采样效率) 价值有限 (4→3 边界)

### 6.2 数据集效应 (有差异 → 数据集相关)

| 指标 | baseline_aug (新) | ghss_24obj (24obj) | 差异 | 归因 |
|------|:-----------------:|:------------------:|------|------|
| mAP | 0.745 | 0.857 | **+0.112** | 24obj 数据质量更好 |
| early_x0_quality | 0.280 | 0.270 | -0.01 | 接近, 起点质量无显著差异 |
| x0_stability | 81.23 | 82.81 | +1.6 | 接近, 步间稳定性无显著差异 |
| reg dw/dh | [-1.73, -1.61] | [-1.34, -1.29] | **+0.4** | 新数据集框回归收缩更严重 |
| 特征范数 (small) | 5.02 | 7.01 | **+2.0** | 24obj 的 RoI 特征范数整体更高 |
| hard_sample_confidence | 0.309 | 0.389 | +0.08 | 24obj 的困难样本更"自信" |
| A1 类坍塌 | 否 | **是** | — | 24obj 特有问题 (ghss 耦合下) |

**关键发现**:
1. **mAP 差距 0.112 的根因不是架构**,而是数据质量 — 24obj 数据让同样的架构表现更好
2. **reg 收缩幅度差异** (新数据集 -1.7 vs 24obj -1.3) 提示新数据集的标注框可能偏大,或目标尺度分布不同
3. **特征范数差异** (5.0 vs 7.0) 提示两数据集的图像特性不同 (分辨率/对比度/目标大小)
4. **A1 类在 24obj+ghss 耦合下额外坍塌** — 这是数据集×耦合方式的交互效应

### 6.3 Y 类坍塌的归因 (架构 + 数据共同作用)

| ckpt | 数据集 | collapsed_classes | Y (class 23) mean_logit |
|------|--------|:-----------------:|:-----------------------:|
| baseline_aug | 新 | [23] | -5.19 |
| ghss_24obj | 24obj | [0, 23] | -5.07 |
| random_24obj | 24obj | [23] | -5.31 |

**关键观察**:
- Y 类坍塌在**所有 3 个 ckpt** (跨数据集) 都存在 → **架构固有倾向**
- 但 24obj 上 Y 类 mean_logit (-5.07) 比新数据集 (-5.19) 略好 → **数据集也有影响**
- Y 类在两个数据集中都是样本数最少的类 → **类别不平衡是跨数据集的共性问题**

**结论**: Y 类坍塌是**架构倾向 + 数据不平衡**共同作用的结果。方向 E (ClassBalanced) 在两数据集上都需要,但 24obj 上 Y 类问题略轻。

### 6.4 易混淆对的跨数据集一致性

| ckpt | 数据集 | Top 易混淆对 (相关系数) |
|------|--------|------------------------|
| baseline_aug | 新 | (F19,G22)=0.880, (G21,G22)=0.828, (F19,G21)=0.803 |
| ghss_24obj | 24obj | (F19,G22)=0.835 |
| random_24obj | 24obj | (F19,G22)=0.825, (G21,G22)=0.823, (B4,B5)=0.810 |

**关键观察**:
- **F19↔G22 混淆在所有 3 个 ckpt 都出现** (相关 0.825~0.880) → **架构固有的形态混淆**
- F/G 组 (F19, F20, G21, G22) 混淆跨数据集一致 → 这两对染色体形态相似性是**生物学固有**, 非数据集偏差
- 24obj 上混淆对更少 (ghss 仅 1 对, random 3 对) vs 新数据集 (3 对) → 新数据集可能引入了额外混淆

**对方向 C 的指导**: F/G 组的形态混淆是跨数据集的生物学难题,方向 C (Morphology) 应优先针对 F/G 组的形态差异设计先验,这在两数据集上都有效。

### 6.5 耦合方式效应 (ghss_24obj vs random_24obj)

两者都在 24obj 上训练,仅耦合方式不同:

| 指标 | ghss_24obj | random_24obj | 差异 |
|------|:----------:|:------------:|------|
| mAP | 0.857 | 0.859 | random 略高 (+0.002) |
| class_collapse | [0, 23] | [23] | **ghss 多坍塌 A1** |
| 易混淆对数 | 1 | 3 | ghss 更少 |
| hard_samples | 3976 | 3728 | ghss 更多 |
| hard_sample_confidence | 0.389 | 0.352 | ghss 更自信 |
| reg dw/dh | [-1.34, -1.29] | [-1.28, -1.28] | 接近 |
| x0_stability | 116.69 | 121.30 | random 略高 |

**关键观察**:
- 耦合方式对**模块内部状态有影响但不大**
- **ghss 耦合下 A1 类额外坍塌** (class 0) → ghss 的分组机制可能对 A 组 (A1,A2,A3) 有负面影响
- random 耦合的 x0 稳定性略高, mAP 也略高 → random 耦合在 24obj 上略优
- 但两者 mAP 差距仅 0.002, 耦合方式不是主要因素

### 6.6 对照分析对方向的修正指导

| 方向 | 原指导 (基于新数据集) | 24obj 对照后的修正 |
|------|---------------------|-------------------|
| A (P1+Deformable) | reg 系统性收缩提示框回归有偏 | 收缩是架构固有, 两数据集都需要 → **确认普适性** |
| B (DecoupledHead) | 通过"谦虚"获 +0.004 | 未测 24obj, 但 cls_head 瓶颈是架构固有 → **应有效** |
| C (Morphology+Contrastive) | 转向 cls_head 损失设计 | F/G 组混淆是跨数据集生物学难题 → **优先针对 F/G 组** |
| D (BoxRefineNet) | 放弃 renewal, 改针对 reg 偏移 | reg 收缩是架构固有 → **确认方向 D 普适价值** |
| E (ClassBalanced) | 优先实现, 解决 Y 类坍塌 | Y 类坍塌是架构+数据共同作用 → **两数据集都需要** |
| F (StructuredPrior) | x0 早期质量极差 | x0 质量尚可 (0.27-0.28), 跨数据集接近 → **价值下调, 但仍值得验证** |
| **H (采样效率)** | box 第 1 步收敛 | box 第 3 步收敛 (4步), 加速空间有限 (4→3) → **价值大幅下调** |
| **D' (reg 校正)** | reg 系统性收缩 | 收缩是架构固有, 两数据集都需要 → **确认普适价值** |

### 6.7 对照分析的核心结论

1. **前文 7 项发现中, 6 项是架构固有的** (跨数据集一致), 仅"Y 类坍塌幅度"和"direction_b/focal 谦虚效应"有数据集依赖
2. **mAP 差距 0.112 的根因是数据质量, 不是架构** — 同样的架构在 24obj 上达到 0.857
3. **架构缺陷被高数据质量"掩盖"** — 24obj 上 box 仍第 3 步收敛、renewal 仍未触发、reg 仍收缩, 只是幅度更轻
4. **F/G 组形态混淆是跨数据集的生物学固有难题** — 方向 C 应优先针对此
5. **方向 D' (reg 校正) 价值最高** — reg 系统性收缩是跨数据集的架构固有特性, 两数据集都需要
6. **方向 H (采样效率) 价值有限** — box 第 3 步收敛 (4步中), 减步空间仅 4→3, 且在收敛边界

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
    --config experiments/configs/ldmdet/rf_heun_adaln.py \
    --ckpt work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth \
    --analyzers trajectory roi_feature head_output \
    --num-samples 50 --device cuda:0 \
    --output-dir work_dirs/instrumentation/baseline_aug

# 批量分析所有 7 个 ckpt (按 BATCH_PLAN)
python experiments/analysis/instrumentation/run_instrumentation.py \
    --batch --num-samples 50 --device cuda:0

# 只跑 24obj 对照 2 个 ckpt (合并到已有 comparison.json)
python experiments/analysis/instrumentation/run_instrumentation.py \
    --batch --only ghss_24obj random_24obj --num-samples 50 --device cuda:0
```

## 九、核心结论

### 基于新数据集 5 ckpt 的结论

1. **分类瓶颈在 cls_head**, 不在特征抽取 — 方向 C 应调整重心
2. **Y 类坍塌普遍存在**, 是分类瓶颈的根因 — 方向 E 应优先
3. **box 第 3 步收敛 (4 步采样), cls 第 2 步收敛** — 步数对 box 是必要的, H 方案价值有限
4. **box_renewal 推理时未触发**, 训练时提升 x0 步间稳定性 11 倍 — 方向 D 需重新设计
5. **reg 系统性收缩** (dw/dh ≈ -1.6) — 方向 D' 应针对此偏移
6. **direction_b 和 focal_gamma_3 通过"更谦虚"获得小幅提升**, 但未根本解决问题
7. **白盒分析补充了黑盒分析的 5 个盲区**, 为方向 A-F 提供了更精准的指导

### 基于 24obj 对照分析的补充结论

8. **前 7 项发现中 6 项是架构固有的** (跨数据集一致) — 仅 Y 类坍塌幅度和数据集特异坍塌有数据依赖
9. **mAP 差距 0.112 的根因是数据质量, 不是架构** — 同架构在 24obj 上达 0.857
10. **架构缺陷被高数据质量"掩盖"** — 24obj 上 box 仍第 3 步收敛、renewal 仍未触发、reg 仍收缩
11. **F/G 组形态混淆是跨数据集的生物学固有难题** — 方向 C 应优先针对 F/G 组
12. **方向 D' (reg 校正) 价值最高** — reg 系统性收缩跨数据集一致, 两数据集都需要
13. **方向 E/D' 的普适性已确认** — reg 收缩、Y 类坍塌在两数据集上都存在

## 附录 A: Trajectory 分析 bug 修正说明

### Bug 描述
初版 `run_instrumentation.py` 的 `run_trajectory_analysis` 函数把所有验证图的采样步混入同一个 `TrajectoryCollector`:
- 每图返回 trajectory (长度 = `sampling_timesteps` = 4)
- 对每图的 4 步逐条 `collector.record()`, 50 图累积 200 条记录
- `TrajectoryAnalyzer.n_steps = len(trajectory)` = 200, 误把"总记录数"当"采样步数"

更严重的是 `analyze_box_evolution` 用 `self.trajectory[-1]` (第 50 张图第 4 步) 当 final, 遍历 200 条算 IoU — 前 196 条是其他图/其他步的框, IoU 计算无物理意义。

### 修复方案
改为每图独立 collector + analyzer, 再聚合统计量:
- 每图单独 `TrajectoryAnalyzer.full_report()` (n_steps=4, 正确)
- 跨图聚合: `convergence_step_mean`, `cls_converged_ratio`, `early_x0_quality_mean` 等

### 受影响的结论 (已修订)
| 原结论 (错误) | 修订后 (正确) |
|--------------|--------------|
| n_steps=200 | n_steps=4 |
| box 第 1 步收敛 | box 第 3 步收敛 |
| cls 跑完 200 步未收敛 | cls 第 2 步收敛, 90-95% 图收敛 |
| early_x0_quality=0.003 (极差) | early_x0_quality=0.280 (尚可) |
| x0_stability 降幅 49% | x0_stability 降幅 91% |
| H 方案 200× 加速 | H 方案最多 1.33× 加速 |

### 不受影响的结论
基于 RoIFeature 和 HeadOutput 的发现 (1, 2, 5, 6, 7) 完全不受此 bug 影响, 仍然可靠。
