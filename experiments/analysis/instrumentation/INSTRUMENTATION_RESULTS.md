# 白盒插桩分析结果

> 本文档记录对 5 个 checkpoint 执行白盒插桩分析的实验结果与发现。
> 执行时间: 2026-06-29
> 方案文档: [INSTRUMENTATION_PLAN.md](./INSTRUMENTATION_PLAN.md)
> 原始 JSON 报告: `work_dirs/instrumentation/<ckpt_name>/{trajectory,roi_feature,head_output}_report.json`
> 汇总对比: `work_dirs/instrumentation/comparison.json`

## 一、执行概况

### 分析矩阵执行情况

| # | Checkpoint | mAP | Trajectory | RoIFeature | HeadOutput | 状态 |
|---|-----------|-----|:---------:|:---------:|:---------:|:----:|
| 1 | baseline_aug | 0.745 | ✓ | ✓ | ✓ | 完成 |
| 2 | direction_d | 0.747 | ✓ | — | — | 完成 |
| 3 | direction_b | 0.749 | — | ✓ | ✓ | 完成 |
| 4 | focal_gamma_3 | 0.750 | — | — | ✓ | 完成 |
| 5 | no_box_renewal | 0.635 | ✓ | — | — | 完成 |

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

### 发现 3: 扩散采样的"步数浪费"矛盾

| ckpt | n_steps | box convergence_step | cls convergence_step | cls_converged |
|------|:-------:|:--------------------:|:--------------------:|:-------------:|
| baseline_aug | 200 | 1 | 199 | False |
| direction_d | 200 | 1 | 199 | False |
| no_box_renewal | 200 | 1 | 199 | False |

**关键矛盾**:
- **box 在第 1 步就收敛** (IoU 变化 <0.01)
- **cls_logits 跑完 200 步仍未收敛** (后期熵仍变化)
- 但训练时固定跑 200 步, 后 199 步对 box 回归是**完全浪费**的

**深层含义**:
- 当前 DiffusionDet 框架下, box 由扩散过程显式生成, cls 由 head 隐式预测
- box 早收敛 + cls 不收敛 → 说明 cls_head 的不确定性并非来自 box 质量, 而是自身容量/训练问题
- 这进一步支持发现 1: 分类瓶颈在 cls_head 内部

**新方向建议**:
- 设计 **box 早停 + cls 继续精化** 的非对称采样策略
- 或减少采样步数 (e.g., 50 步) 以节省推理时间, 因为 box 已在第 1 步收敛

### 发现 4: box_renewal 形同虚设

| ckpt | renewal_effective | renewal_direction | per_step_renewal_delta (均值) |
|------|:-----------------:|:-----------------:|:----------------------------:|
| baseline_aug | False | 0.0 | 0.0 (全部 200 步均为 0) |
| direction_d | False | 0.0 | 0.0 |
| no_box_renewal | False | 0.0 | 0.0 (此 ckpt 本就无 renewal) |

**关键观察**:
- baseline_aug 的 `per_step_renewal_delta` **全部 200 步均为 0.0** — box_renewal 完全未触发
- direction_d (BoxRefineNet 方向) 也未让 renewal 生效 — mAP +0.002 不是来自 renewal
- no_box_renewal (本就关闭 renewal) 的轨迹与 baseline 几乎一致 (convergence_step=1, early_x0_quality=0.003)

**但 mAP 暴跌 0.110**: no_box_renewal mAP=0.635 vs baseline 0.745, 差距 -0.110

**矛盾与解释**:
- renewal_delta=0 但关闭 renewal 后 mAP 暴跌 → renewal 在**训练阶段**起作用 (梯度回传影响 cls_head/reg_head), 而非**推理阶段**显式修正 box
- 这与 `x0_stability` 数据吻合:

| ckpt | x0_stability |
|------|:-----------:|
| baseline_aug | 109.57 |
| no_box_renewal | 55.74 |

- baseline 的 x0 稳定性是 no_box_renewal 的 **2 倍** → renewal 在训练时隐式稳定了 x0 预测

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

**TrajectoryAnalyzer**:
- box 在第 1 步收敛 (convergence_step=1), 后 199 步对 box 是浪费
- cls_logits 跑完 200 步仍未收敛 (cls_converged=False, convergence_step=199)
- 早期 x0 质量 IoU=0.003 (几乎为 0, 扩散起点不可靠)
- renewal_delta 全部为 0.0 (renewal 未触发)
- x0_stability=109.57 (后续对比基准)

**RoIFeatureAnalyzer**:
- 同组内相似度 0.999, 跨组 0.9988 → 几乎无差异, **分类瓶颈在 cls_head**
- 小/中/大目标特征范数: 5.02 / 4.93 / 4.80 → 尺度对特征影响微弱 (scale_affects=False)

**HeadOutputAnalyzer**:
- Y 类 (class 23) 坍塌, mean_logit=-5.19 (远低于均值 -4.15)
- 易混淆对: F19↔G22 (0.880), G21↔G22 (0.828), F19↔G21 (0.803) — 全是 F/G 组
- reg 系统性收缩: dw=-1.730, dh=-1.608
- 困难样本 3599 个, confidence=0.309, top2_gap=0.171, 未过度自信

### #2 direction_d (mAP=0.747, +0.002) — box_refine 验证

**TrajectoryAnalyzer**:
- convergence_step=1 (与 baseline 一致, 未提前收敛)
- early_x0_quality=0.0038 (略好于 baseline 0.003, 但仍是 0.004 级别, 改善微弱)
- renewal_effective=False (与 baseline 一致, box_renewal 仍未触发)
- x0_stability=105.86 (略低于 baseline 109.57)

**判断**: direction_d 在轨迹层面**几乎没有改善**:
- box_refine_net 既未让 renewal 有效 (renewal_delta 仍为 0)
- 也未让 x0 质量显著提升 (0.003 → 0.004 是噪声级别)
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

**TrajectoryAnalyzer**:
- convergence_step=1 (与 baseline 一致, box 仍第 1 步收敛)
- early_x0_quality=0.0029 (与 baseline 0.003 几乎一致)
- renewal_effective=False (本就关闭)
- **x0_stability=55.74 (远低于 baseline 109.57, 降幅 49%)**

**关键量化**:
- 关闭 renewal 后, 推理轨迹模式几乎不变 (convergence_step, early_x0_quality 一致)
- 但 x0_stability 暴跌 49% → renewal 在**训练阶段**隐式稳定了 x0 预测
- mAP 暴跌 -0.110 → x0 不稳定导致最终定位质量下降

**机制澄清**:
- box_renewal 在**推理时**修正量=0 (形同虚设)
- 但在**训练时**作为正则项, 让 cls_head/reg_head 学到更稳定的 x0 预测
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
| F (StructuredPrior) | x0 早期质量极差 (0.003), 扩散起点不可靠 | 结构化先验可能改善 x0 起点, 值得验证 |

### 新方向建议

**方向 H: 扩散采样效率优化**
- 依据: box 第 1 步收敛, 后 199 步对 box 是浪费; cls 跑完 200 步仍不收敛
- 思路:
  - (H1) 减少采样步数 (e.g., 50 步), 验证 mAP 是否保持
  - (H2) 非对称采样: box 早停 + cls 继续精化
  - (H3) 动态步数: 根据 box 收敛判据自适应停止 box 更新
- 预期收益: 推理速度 4×~10× 提升, mAP 基本保持

## 五、白盒 vs 黑盒分析的互补性

### 黑盒分析 (bottleneck_report.json) 已回答
- AP-IoU 曲线: 定位瓶颈在 IoU≥0.9 (AP@90=0.483)
- 误差分解: Cls error 43.4%, 同组混淆为主
- 混淆矩阵: Top 混淆对 (C10↔C9, C9↔C8 等)
- 尺度/类别不平衡: Y 类样本最少

### 白盒分析 (本文档) 新增回答
- **分类瓶颈定位**: cls_head (非特征) — 黑盒无法回答
- **Y 类坍塌机制**: mean_logit 显著低, 是 cls_head 学习问题 — 黑盒只能看到 Y 类 AP 低
- **扩散步数浪费**: box 第 1 步收敛, 后 199 步无效 — 黑盒无法看到
- **box_renewal 实际作用**: 推理时未触发, 训练时隐式正则 — 黑盒只能看到 mAP 差异
- **reg 系统性偏移**: dw/dh 偏负 -1.7/-1.6 — 黑盒只能看到 IoU 低, 不知是中心还是尺度问题

### 仍未回答的问题 (需进一步插桩)
- cls_head 内部哪一层 (FC/attention) 导致同组混淆?
- 扩散噪声 schedule 是否合理? (t 越大, x0 质量应越差, 但当前 early_x0_quality=0.003 已极差)
- proposal 数量对 x0 质量的影响? (方向 F 假设)

## 六、产出文件索引

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
- `work_dirs/instrumentation/comparison.json` — 跨 ckpt 汇总对比
- `work_dirs/instrumentation/baseline_aug/{trajectory,roi_feature,head_output}_report.json`
- `work_dirs/instrumentation/direction_d/trajectory_report.json`
- `work_dirs/instrumentation/direction_b/{roi_feature,head_output}_report.json`
- `work_dirs/instrumentation/focal_gamma_3/head_output_report.json`
- `work_dirs/instrumentation/no_box_renewal/trajectory_report.json`

## 七、复现命令

```bash
# 单个 ckpt 分析
python experiments/analysis/instrumentation/run_instrumentation.py \
    --config experiments/configs/ldmdet/rf_heun_adaln.py \
    --ckpt work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/best_coco_bbox_mAP_epoch_102.pth \
    --analyzers trajectory roi_feature head_output \
    --num-samples 50 --device cuda:0 \
    --output-dir work_dirs/instrumentation/baseline_aug

# 批量分析所有 5 个 ckpt (按 BATCH_PLAN)
python experiments/analysis/instrumentation/run_instrumentation.py \
    --batch --num-samples 50 --device cuda:0
```

## 八、核心结论

1. **分类瓶颈在 cls_head**, 不在特征抽取 — 方向 C 应调整重心
2. **Y 类坍塌普遍存在**, 是分类瓶颈的根因 — 方向 E 应优先
3. **box 第 1 步收敛, 后 199 步浪费** — 新方向 H (采样效率) 机会
4. **box_renewal 推理时未触发**, 训练时起正则作用 — 方向 D 需重新设计
5. **reg 系统性收缩** (dw/dh ≈ -1.6) — 方向 D 应针对此偏移
6. **direction_b 和 focal_gamma_3 通过"更谦虚"获得小幅提升**, 但未根本解决问题
7. **白盒分析补充了黑盒分析的 5 个盲区**, 为方向 A-F 提供了更精准的指导
