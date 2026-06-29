# 白盒插桩分析方案

> 本文档记录基于瓶颈实验,对模型模块内在问题进行白盒插桩分析的方案。
> 分析时间: 2026-06-29
> 分析器位置: `experiments/analysis/instrumentation/`

## 一、背景

### 已有的黑盒瓶颈分析 (输出层面)
`work_dirs/bottleneck/bottleneck_report.json` + `supplementary_report.json` 已覆盖:
- AP-IoU 曲线、多 IoU 误差分解 (TIDE 风格)
- 混淆矩阵、每类准确率、Top 混淆对
- 尺度/密度/类别不平衡分析
- 置信度校准、框质量分布
- Proposal 利用率

**局限**: 这些全是"黑盒"分析 — 只看最终预测 vs GT,无法回答"**为什么**同组类别会混淆"、 "**为什么**高 IoU 定位差"、 "**哪一步**扩散采样出了问题"。

### 白盒插桩分析的目的
通过 forward hook 采集模块中间状态,将瓶颈定位到**具体模块**,回答:
1. 分类瓶颈在**特征提取**还是**cls_head**?
2. 定位瓶颈在**扩散过程**还是**reg_head**?
3. box_renewal 的**实际贡献机制**是什么?

## 二、三个插桩分析器

### 1. TrajectoryAnalyzer (扩散采样轨迹)
**采集方式**: `head.predict(return_trajectory=True)` (零代码改动)

| 分析项 | 回答的问题 |
|--------|-----------|
| `analyze_box_evolution` | 哪一步导致定位误差? box 是否随步数收敛? |
| `analyze_cls_convergence` | 分类置信度是否随 t 收敛? 还是震荡? |
| `analyze_x0_quality` | 每步 x0 预测质量? 早期 x0 是否可靠? |
| `analyze_renewal` | box_renewal 实际修正了多少? 是否有效? |

### 2. RoIFeatureAnalyzer (RoI 特征区分度)
**采集方式**: forward hook on `roi_extractor`

| 分析项 | 回答的问题 |
|--------|-----------|
| `analyze_same_group_similarity` | 同组类别特征余弦相似度 → **特征问题 vs 头问题** |
| `analyze_scale_feature_norm` | 小目标特征范数是否偏弱? |
| `analyze_error_type_separability` | TP/Cls/Loc 在特征空间是否可分? |

### 3. HeadOutputAnalyzer (cls/reg 头输出)
**采集方式**: forward hook on `cls_head` / `reg_head`

| 分析项 | 回答的问题 |
|--------|-----------|
| `analyze_cls_distribution` | 类别坍塌? 易混淆类对? |
| `analyze_reg_distribution` | 回归是否保守 (delta 太小)? |
| `analyze_gradient_ratio` | cls vs reg 哪个头主导? |
| `analyze_hard_samples` | 困难样本是否过度自信? |

## 三、checkpoint 选择

### 数据集一致性
所有 ckpt 均使用 `Chromosome20240904_NoAug_NoResize_coco/` (新数据集, 24 类)。

### 选择的 5 个 ckpt

| # | Checkpoint | mAP | 角色 |
|---|-----------|-----|------|
| 1 | `multi_seed_aug/rf_heun_adaln/seed_42` | 0.745 | **baseline** (aug, 最佳基线) |
| 2 | `direction_exps/direction_d_box_refine` | 0.747 | 方向 D (box_refine) |
| 3 | `direction_exps/direction_b_decoupled_head` | 0.749 | 方向 B (decoupled) |
| 4 | `bottleneck/ablation/focal_gamma_3` | 0.750 | bottleneck 最佳 (分类损失) |
| 5 | `bottleneck/ablation/no_box_renewal` | 0.635 | bottleneck (无 renewal, 对比) |

### 不跑的 ckpt 及原因
| Checkpoint | 原因 |
|-----------|------|
| baseline 无 aug (0.712) | aug baseline (0.745) 更强,作为对比基准更合理 |
| SOTA (0.740) | 架构可能不同,白盒分析针对 ldmdet |
| high_cls_weight/scale_aware_loss | 与 focal_gamma_3 机制类似,#4 已覆盖 |
| 其他 ablation | 无法回答核心问题 |

## 四、分析矩阵: ckpt × 分析器

| # | Checkpoint | Trajectory | RoIFeature | HeadOutput | 目的 |
|---|-----------|:---------:|:---------:|:---------:|------|
| 1 | baseline_aug | ✓ | ✓ | ✓ | 建立白盒诊断**基线**,定位问题根因 |
| 2 | direction_d | ✓ | — | — | 验证 box_refine 是否改善**轨迹收敛**和**renewal** |
| 3 | direction_b | — | ✓ | ✓ | 验证解耦是否改善**同组区分度**和**cls 输出** |
| 4 | focal_gamma_3 | — | — | ✓ | 验证 focal gamma 如何改善**类别坍塌/易混淆对** |
| 5 | no_box_renewal | ✓ | — | — | 量化 **box_renewal 的实际贡献** |

## 五、每个分析的预期产出

### #1 baseline_aug 基线诊断
**TrajectoryAnalyzer**:
- x0 早期质量 (IoU 期望 <0.5 → 扩散起点不可靠)
- 收敛步 (期望后 1/3 → 前期浪费)
- renewal 修正幅度 (期望 >0.001 → 有效)

**RoIFeatureAnalyzer**:
- 同组相似度 (C8/C9/C10 期望 >0.9 → 特征问题,方向 C 对症)
- 小目标特征范数 (期望 small/large <0.7 → 尺度影响特征)

**HeadOutputAnalyzer**:
- 类别坍塌 (期望有 → 某些类从不被预测)
- 易混淆类对 (期望与瓶颈报告的 Top 混淆对一致)
- 回归保守 (期望 delta_mean <0.1 → 高 IoU 定位不足)
- 困难样本过度自信 (期望 confidence >0.7 → 需要困难样本损失)

### #2 direction_d vs baseline (TrajectoryAnalyzer)
- box_refine 是否让 renewal 修正幅度增大?
- box_refine 是否让 x0 质量提升?
- box_refine 是否让收敛步提前?
- **判断**: D 是否对症定位瓶颈

### #3 direction_b vs baseline (RoIFeature + HeadOutput)
- 解耦后同组相似度是否降低?
- 解耦后易混淆类对是否减少?
- 解耦后困难样本置信度是否降低?
- **判断**: B 是否对症分类瓶颈

### #4 focal_gamma_3 vs baseline (HeadOutput)
- focal gamma=3 后类别坍塌是否缓解?
- 易混淆类对相关系数是否降低?
- **判断**: 分类损失调整的作用机制,指导方向 C 设计

### #5 no_box_renewal vs baseline (TrajectoryAnalyzer)
- 无 renewal 时轨迹是否发散?
- 无 renewal 时 x0 质量是否更差?
- **量化**: box_renewal 对最终 mAP 的贡献机制

## 六、染色体分组映射 (用于同组相似度分析)

```python
CHROMOSOME_GROUPS = {
    0: [0, 1, 2],        # A 组 (A1, A2, A3)
    1: [3, 4],            # B 组 (B4, B5)
    2: [5, 6, 7, 8, 9, 10, 11],  # C 组 (C6-C12)
    3: [12, 13, 14],      # D 组 (D13-D15)
    4: [15, 16, 17],      # E 组 (E16-E18)
    5: [18, 19],          # F 组 (F19, F20)
    6: [20, 21],          # G 组 (G21, G22)
    7: [22],              # X
    8: [23],              # Y
}
```

## 七、执行计划

1. 编写运行脚本 `run_instrumentation.py`
2. 对 5 个 ckpt 按分析矩阵执行
3. 每个 ckpt 生成 JSON 报告到 `work_dirs/instrumentation/<ckpt_name>/`
4. 汇总对比,生成 `INSTRUMENTATION_RESULTS.md`

## 八、产出文件索引
- 方案文档: `experiments/analysis/instrumentation/INSTRUMENTATION_PLAN.md` (本文档)
- 运行脚本: `experiments/analysis/instrumentation/run_instrumentation.py`
- 结果文档: `experiments/analysis/instrumentation/INSTRUMENTATION_RESULTS.md`
- JSON 报告: `work_dirs/instrumentation/<ckpt_name>/{trajectory,roi_feature,head_output}_report.json`
