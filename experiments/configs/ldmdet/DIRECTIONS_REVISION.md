# 方向调整决策 (基于白盒插桩分析)

> 决策时间: 2026-06-30 (E 失败后更新)
> 依据: [INSTRUMENTATION_RESULTS.md](../../analysis/instrumentation/INSTRUMENTATION_RESULTS.md) 的 13 项核心发现 + E (ClassBalanced) 实验结果
> 背景: 白盒插桩分析揭示了 6 项架构固有缺陷 (跨数据集一致); E (ClassBalanced) 实验失败 (+0.001), 需重新评估方向

## 一、接龙顺序调整

### 原顺序 (基于黑盒分析)
D (BoxRefineNet) → B (DecoupledHead) → C (Morphology) → F (StructuredPrior) → A (P1+Deformable) → E (ClassBalanced)

### 调整后顺序 (E 失败后, 2026-06-30 更新)
**D' (reg 校正) → F (StructuredPrior) → G (LAMFPN, 可选) → H (采样效率, 可选)**

跳过: E (已证伪, +0.001), C (瓶颈在 cls_head 不在特征), A (scale_affects=False, 尺度非主要瓶颈), B (已早停)

> 演进历史:
> - v1 (trajectory bug 前): E → F → H → D'
> - v2 (trajectory bug 后): E → F → D' → H (H 降为可选, 因 box 第 3 步收敛)
> - v3 (E 失败后, 当前): D' → F → G → H (D' 升为 P0, E 暂缓)

### 各方向价值评估

| 方向 | 状态 | 白盒诊断 | 价值 | 决策 |
|------|------|---------|------|------|
| D (BoxRefineNet) | 完成 +0.002 | ❌ renewal 推理时 delta=0 未触发; trajectory 与 baseline 一致 | 已证伪 | 无需继续 |
| B (DecoupledHead) | 早停 at epoch 86, +0.004 | ⚠️ 未解决 Y 坍塌; 仅"更谦虚"获小幅提升 | 价值有限 | **已停** |
| C (Morphology) | 跳过 | ⚠️ 瓶颈在 cls_head (same_group_too_similar=False); ShapeAttention 作用于特征层 | 价值存疑 | **跳过** |
| A (P1+Deformable) | 跳过 | ⚠️ scale_affects=False (尺度对特征影响微弱) | 价值存疑 | **跳过** |
| F (StructuredPrior) | 排队 | ✅ 针对 early_x0_quality=0.280 (有改善空间); renewal 稳定 x0 步间一致性 | 有价值 | **保留** |
| E (ClassBalanced) | ❌ 失败 (+0.001) | ✅ 直接针对 Y 类坍塌根因, 但 ClassBalancedDataset 未解决 | 已证伪 | **暂缓, 待 E2-E5** |
| **H (采样效率)** | 新增 (价值下调) | ⚠️ box 第 3 步收敛 (4步中), 减步空间仅 4→3 | 价值有限 | **可选验证** |
| **D' (reg 校正)** | 新增 | ✅ reg 系统性收缩 dw/dh≈-1.6 (架构固有, 跨数据集) | 高价值 | **新增** |

### 调整理由

1. **停 B**: epoch 86, best at 74, 后 12 epoch 零提升 → 已收敛; 白盒显示 +0.004 来自"谦虚"副作用, 未解决 Y 坍塌和同组混淆
2. **跳 C**: 白盒证实分类瓶颈在 cls_head 决策边界 (same_group_too_similar=False), 不在特征抽取; ShapeAttention 作用于特征层, 但特征已足够区分
3. **跳 A**: 白盒显示 scale_affects=False (小/中/大目标特征范数接近 5.0/4.9/4.8), 尺度非主要瓶颈
4. **E 已证伪**: ClassBalancedDataset (oversample_thr=0.5) 仅 +0.001, 在噪声范围内; 训练曲线波动大 (0.60-0.75); 失败根因: 每图全类存在, 图像级过采样无法解决 instance 级不平衡, 且未触及 cls_head 决策边界 (详见第五章)
5. **新增 H (价值已下调)**: box 第 3 步收敛 (4 步中), 减步空间仅 4→3 (1.33× 加速), 且在收敛边界 — 从"高优先级"降为"可选验证"
6. **D' 升为 P0**: reg 系统性收缩 dw/dh≈-1.6 是架构固有 (24obj 上 -1.3, 方向一致), 应针对此偏移设计 (区别于原 D 的 renewal 机制); E 失败后, D' 成为最可能产生稳定收益的方向

> 修正说明: 初版基于 trajectory 分析 bug 曾误判 "box 第 1 步收敛, H 方案 200× 加速"。修复后真实结果为 box 第 3 步收敛 (4 步采样), H 价值大幅下调。详见 [INSTRUMENTATION_RESULTS.md 附录 A](../../analysis/instrumentation/INSTRUMENTATION_RESULTS.md)。

## 二、方向 H: 扩散采样效率优化 (价值已下调)

### 白盒依据 (修复后真实数据)
- **发现 3 (修正)**: box 在第 3 步收敛 (convergence_step_mean=3.0, 4 步中倒数第 2 步)
- cls 在第 2 步收敛 (cls_converged_ratio=0.9-1.0), 比 box 更早
- 4 步采样对 box 是必要的: 减到 3 步刚好在收敛边界, 减到 2 步 box 未收敛
- 跨数据集一致 (24obj 也是第 3 步收敛), 确认为架构固有

### 已确认: 实际采样 4 步 (非 200)
- 配置 `sampling_timesteps=4`, 采样器 Rectified Flow + Heun solver
- 初版报告 n_steps=200 是 bug (跨图混合), 已修复

### 子方向设计

#### H1: 减少采样步数 (价值有限)
- **做法**: 修改配置 `sampling_timesteps` 从 4 → 3 → 2
- **实现**: 仅改配置, 无需改代码
- **验证**: 对比 mAP, 找到 mAP 不降的最小步数
- **预期 (修正)**:
  - 4→3: box 刚好在收敛边界, mAP 可能保持, 仅 1.33× 加速
  - 4→2: box 未收敛, mAP 可能下降
  - 4→1: box 远未收敛, mAP 肯定下降
- **结论**: 加速空间有限, 降为可选验证

#### H2: 非对称采样 (不适用)
- **做法**: 修改 `predict` 方法, box 在前 K 步更新, 后续步仅更新 cls_logits
- **问题**: 修复后数据显示 cls (第 2 步) 比 box (第 3 步) 更早收敛 → "cls 继续精化"无意义
- **结论**: 放弃 H2

#### H3: 动态步数 (复杂度高, 收益不确定)
- 略

### 实施计划
1. **H1 可选** (仅改配置): 若 E/F/D' 之间有空闲, 跑 `sampling_timesteps=3` 验证 mAP 是否保持
2. H2/H3 放弃

### 配置文件
```python
# experiments/configs/ldmdet/direction_h_sampling_efficiency.py
_base_ = ['./rf_heun_adaln.py']
model = dict(
    bbox_head=dict(
        sampling_timesteps=3,  # H1: 从 4 减到 3 (收敛边界)
    ),
)
```

## 三、方向 D': 回归偏移校正

### 白盒依据
- **发现 5**: reg 系统性收缩, dw/dh ≈ -1.6 (baseline), -1.3 (24obj), -1.4 (focal_gamma_3)
- **发现 8 (24obj 对照)**: 收缩方向跨数据集一致 (都偏负), 确认为架构固有
- 这**不是保守回归** (is_conservative=False, 因 |delta_mean| > 0.1), 而是系统性缩小框
- dx, dy 接近 0, 说明中心定位无偏, 仅尺度预测偏小

### 根因推测
- 训练数据中 GT 框可能偏大 (标注习惯), 模型学习到"收缩"补偿
- 或扩散过程的 box 初始化偏大, reg_head 学习补偿
- 或 L1 loss 对 dw/dh 的对称性导致模型学到均值偏移

### 子方向设计

#### D'1: 后处理尺度校准 (最简, 优先实施)
- **做法**: 推理时对预测框的 w/h 加校准因子 (e.g., ×1.16 补偿 -1.6 偏移)
- **实现**: 修改 predict 方法的后处理, 或加 test_pipeline 后处理
- **校准因子推导**: 从白盒分析的 per_dim_mean 反推
  - dw=-1.6 → exp(-1.6)≈0.20, 即预测 wh 是 GT 的 ~20%? 不对
  - 实际: delta 是 reg_head 输出, 需看 box 编码方式
  - 若 delta 直接是 log 空间偏移: 校准因子 = exp(1.6) ≈ 4.95? 需确认编码
- **预期**: 简单校准可能改善 mAP, 但全局校准不适应类别/尺度差异
- **风险**: 校准因子可能因类别/尺度不同而异, 全局校准可能损害部分类

#### D'2: reg_head 正则化 (训练时约束) — ✅ 已实施
- **做法**: 训练时加 reg loss 项, 约束**正样本** dw/dh 均值接近 0
- **实现**:
  - `single_head.py`: 保存 `_last_bboxes_deltas` (reg_head 输出)
  - `criterion.py`: 保存 `_last_indices` (matcher 正样本匹配)
  - `head.py`: 用 fg_masks 提取正样本 delta, 计算 `loss_reg_bias = weight * (dw_mean^2 + dh_mean^2)`
  - 诊断: 同时记录 `fg_dw_mean`, `fg_dh_mean`, `bg_dw_mean`, `bg_dh_mean` (detached, 仅 log)
- **配置**: `direction_d_prime_reg_calibration.py`, `reg_bias_weight=0.05`
- **box 编码确认**: `pred_w = proposal_w * exp(dw)`, dw=-1.8 意味着 pred_w ≈ proposal_w * 16%
- **预期**: 减少正样本系统性偏移, 提升高 IoU 定位精度 (mAP75)
- **复杂度**: 中等

> **D'2 bug 修复记录**:
> - v1 (commit da1c2015): 对所有 proposal (含背景) 计算 reg_bias_loss → epoch 5 降到 0, 但可能是背景 proposal 补偿
> - v2 (当前): 只对正样本 (matcher 匹配) 计算, 添加 fg/bg 诊断统计 → 确保正样本偏移真正消除

#### D'3: 数据增强 (box 尺度抖动)
- **做法**: 训练时对 GT 框加随机尺度抖动 (e.g., ±10%), 打破系统性偏移
- **实现**: 修改 train_pipeline, 加 ScaleJitter transform
- **预期**: 增强尺度鲁棒性, 减少偏移
- **风险**: 可能影响小目标检测

#### D'4: box 参数化改进
- **做法**: 改用不同的 box 编码 (e.g., 直接预测 xyxy 而非 cxcywh + delta)
- **实现**: 修改 box 编码/解码逻辑
- **复杂度**: 高 (涉及训练和推理)

### 实施计划
1. **D'2 已实施** (训练时正样本正则, 当前版本 v2): reg_bias_weight=0.05
2. D'1 (后处理校准) 暂缓: -1.8 含背景 proposal, 校准因子推导不确定; D'2 让模型自适应
3. D'3/D'4 作为备选

### D'2 插桩充分性评估

| 验证目标 | 插桩方式 | 充分性 |
|---------|---------|:------:|
| 正样本 dw/dh 偏移消除 | 训练 log `fg_dw_mean`, `fg_dh_mean` 趋近 0 | ✅ |
| 背景 proposal 未补偿 | 训练 log `bg_dw_mean`, `bg_dh_mean` 不异常偏正 | ✅ |
| loss_reg_bias 正确计算 | 训练 log `loss_reg_bias` 非零且递减 | ✅ |
| mAP 提升 | CocoMetric `coco/bbox_mAP` | ✅ |
| 高 IoU 定位改善 | CocoMetric `coco/bbox_mAP_75` | ✅ |
| 训练后正样本 delta 验证 | 白盒 HeadOutputAnalyzer per_dim_mean | ✅ (训练后) |
| 不损害分类 | CocoMetric per-class AP + 白盒 cls_collapse | ✅ |

### 配置文件
```python
# experiments/configs/ldmdet/direction_d_prime_reg_calibration.py
# D'1: 后处理尺度校准
_base_ = ['./rf_heun_adaln.py']
model = dict(
    bbox_head=dict(
        reg_calibration=dict(  # 新增后处理校准
            type='RegCalibration',
            wh_scale=1.16,  # 校准因子, 需根据编码方式推导
        ),
    ),
)
```

## 四、接龙实验状态跟踪

| 方向 | 状态 | mAP | best_epoch | Δ | 备注 |
|------|------|-----|-----------|---|------|
| baseline (rf_heun_adaln) | 基线 | 0.745 | 102 | — | multi_seed_aug, 早停于 132 |
| D (BoxRefineNet) | ✅ 完成 | 0.747 | 55 | +0.002 | renewal 未生效 |
| B (DecoupledHead) | ⏹ 早停 | 0.749 | 74 | +0.004 | "谦虚"效应, epoch 86 停 |
| E (ClassBalanced) | ❌ 失败 | 0.746 | 55 | **+0.001** | ClassBalancedDataset 失败, epoch 85 早停 |
| F (StructuredPrior) | ⏳ 待跑 | — | — | — | 前置 box_stats.pt 已就绪, 可直接跑 |
| D' (reg 校正) | ⏳ 规划中 | — | — | — | **最高优先级** (E 失败后) |
| H (采样效率) | ⏳ 可选 | — | — | — | H1 仅改配置 4→3 (价值有限) |
| G (LAMFPN) | ⏳ 备选 | — | — | — | 白盒未直接评估 neck 价值 |

## 五、E (ClassBalanced) 失败分析

### 5.1 实验结果

- **配置**: `ClassBalancedDataset(oversample_thr=0.5)` 包装 CocoDataset
- **best mAP=0.746 at epoch 55** (baseline 0.745, **仅 +0.001, 在噪声范围内**)
- 早停于 epoch 85 (patience=30, best at 55)
- 训练曲线波动剧烈: mAP 在 0.60-0.75 之间震荡 (epoch 6=0.602, 22=0.721, 47=0.743, 55=0.746, 75=0.733, 85=0.732)

### 5.2 失败根因分析

1. **过采样破坏训练稳定性**:
   - ClassBalancedDataset 通过重复采样少数类图像来平衡类别分布
   - 但每张染色体图像都包含 46 条染色体 (各类都有), 过采样某类等于过采样整张图
   - 实际效果是放大了少数类图像的权重, 而非真正增加少数类样本的多样性
   - 导致训练梯度方向不稳定, mAP 震荡

2. **未触及 cls_head 决策边界**:
   - 白盒发现 1 明确: 分类瓶颈在 cls_head, 不在数据分布
   - 单纯过采样不改变 cls_head 的学习难度, Y 类决策边界仍未有效建立
   - mean_logit=-5.17 (Y 类坍塌) 是 cls_head 学习问题, 非数据量问题

3. **每图全类存在的特殊性**:
   - 染色体核型图像每张都包含 24 类 (除异常样本)
   - 传统检测的类别不平衡解决方案 (基于图像级过采样) 在此场景失效
   - 真正的不平衡在 instance 级 (Y 类框数少), 但 ClassBalancedDataset 只在图像级平衡

### 5.3 E 方向的后续

E1 (ClassBalancedDataset) 已证伪, 但 Y 类坍塌问题仍需解决。可探索的替代方案:
- **E2 (logit adjustment)**: 在 cls loss 中根据类频率加 log 先验, 直接调整决策边界
- **E3 (类别加权 loss)**: 对 Y 类的 cls loss 加权 (而非过采样数据)
- **E4 (Y 类数据增强)**: 对含 Y 类的图像做 copy-paste / mixup (但需保持核型完整)
- **E5 (decoupled representation)**: 先学特征再学分类器, 解耦 cls_head 的学习

> 决策: 当前不再继续 E 子方向, 转 D' (reg 校正) 和 F (StructuredPrior), 理由:
> - D' 针对 reg 系统性收缩 (架构固有, 跨数据集一致), 最可能产生稳定收益
> - F 针对 early_x0_quality=0.280 (有改善空间), 前置已就绪
> - E 的 Y 类坍塌问题需更深入方法 (E2-E5), 暂缓

## 六、方向优先级重评 (E 失败后)

### 6.1 优先级调整

| 优先级 | 方向 | 理由 |
|:------:|------|------|
| **P0** | **D' (reg 校正)** | reg 系统性收缩是架构固有 (跨数据集一致), D'1 后处理校准最简, 最可能产生稳定收益 |
| P1 | F (StructuredPrior) | 前置已就绪, 针对 early_x0_quality; 但白盒显示 24obj 上 x0 质量接近, 价值下调 |
| P2 | G (LAMFPN) | 替换 neck, 白盒未直接评估; 但 scale_affects=False 提示特征层可能非瓶颈 |
| P3 | H (采样效率) | box 第 3 步收敛 (4步中), 减步空间仅 4→3, 价值有限 |
| 暂缓 | E2-E5 (Y 类替代方案) | 需更深入设计, 待 D'/F/G 结果后决定 |

### 6.2 新接龙顺序

**D' (P0) → F (P1) → G (P2, 可选) → H (P3, 可选)**

跳过: E (已证伪), C (瓶颈在 cls_head 不在特征), A (scale_affects=False), B (已早停, +0.004 来自副作用)

## 七、复现命令

```bash
# D'2 (在跑, v2 正样本版本)
python experiments/runners/train.py experiments/configs/ldmdet/direction_d_prime_reg_calibration.py \
    --work-dir work_dirs/direction_exps/direction_d_prime_reg_calibration --seed 42 --gpu-id 0

# F (待跑, 前置已就绪)
python experiments/runners/train.py experiments/configs/ldmdet/direction_f_structured_prior.py \
    --work-dir work_dirs/direction_exps/direction_f_structured_prior --seed 42 --gpu-id 0

# H1 (仅改配置, 无需训练, 用已有 ckpt 推理)
# 待 H 配置创建后补充
```

## 八、插桩充分性评估

> 评估目的: 确认在跑和待跑实验的插桩能否真实准确反映设计预期和模型能力。
> 评估维度: (1) 设计预期验证 (2) 训练中实时监控 (3) 训练后白盒验证

### 8.1 方向 D'2 (reg bias 正则化) — 🔄 在跑

**设计预期**: 消除正样本 reg_head 输出的 dw/dh 系统性偏负 (-1.8), 提升高 IoU 定位精度

| 验证目标 | 插桩方式 | 充分性 | 说明 |
|---------|---------|:------:|------|
| 正样本 dw/dh 偏移消除 | 训练 log `fg_dw_mean`/`fg_dh_mean` | ✅ | v2 新增, 实时监控正样本趋势 |
| 背景 proposal 未补偿 | 训练 log `bg_dw_mean`/`bg_dh_mean` | ✅ | v2 新增, 检测背景是否异常偏正 |
| loss_reg_bias 正确计算 | 训练 log `loss_reg_bias` | ✅ | v1 降到 0 发现 bug, v2 修复 |
| mAP 提升 | CocoMetric `coco/bbox_mAP` | ✅ | 黑盒验证 |
| 高 IoU 定位改善 | CocoMetric `coco/bbox_mAP_75` | ✅ | reg 偏移主要影响高 IoU |
| 训练后正样本 delta | 白盒 HeadOutputAnalyzer | ✅ | 训练后运行 |
| 不损害分类 | per-class AP + 白盒 cls_collapse | ✅ | 训练后运行 |

**v1 bug 教训**: 对所有 proposal (含背景) 计算 reg_bias_loss → epoch 5 降到 0 → 实际是背景 proposal 偏正补偿, 正样本偏移未消除。**v2 修复**: 只对正样本计算 + 添加 fg/bg 诊断统计。

**结论**: ✅ 插桩充分 (v2 修复后)

### 8.2 方向 F (StructuredPrior) — ⏳ 待跑

**设计预期**: 改善 early_x0_quality (baseline=0.280), 提升 x0 预测质量

| 验证目标 | 插桩方式 | 充分性 | 说明 |
|---------|---------|:------:|------|
| early_x0_quality 提升 | 白盒 TrajectoryAnalyzer | ✅ | 训练后运行 |
| x0_stability 保持/提升 | 白盒 TrajectoryAnalyzer | ✅ | 训练后运行 |
| mAP 提升 | CocoMetric | ✅ | 黑盒验证 |
| structured_prior 模块有效 | 白盒 trajectory 对比 | ✅ | 与 baseline trajectory 对比 |
| 训练中 x0 质量监控 | ❌ 无实时监控 | ⚠️ | 训练中无法看到 x0 质量, 需训练后分析 |

**结论**: ⚠️ 训练后插桩充分, 训练中无实时监控 (可接受 — x0 质量需完整采样流程, 训练中监控开销大)

### 8.3 方向 G (LAMFPN) — ⏳ 可选

**设计预期**: 改进 neck 特征提取

| 验证目标 | 插桩方式 | 充分性 | 说明 |
|---------|---------|:------:|------|
| 特征质量提升 | 白盒 RoIFeatureAnalyzer | ✅ | 训练后运行 |
| mAP 提升 | CocoMetric | ✅ | 黑盒验证 |
| scale_affects 改善 | 白盒 RoIFeatureAnalyzer | ✅ | 但白盒显示 scale_affects=False, 方向价值存疑 |

**结论**: ✅ 插桩充分 (但方向本身价值存疑 — 白盒显示特征层非瓶颈)

### 8.4 方向 H (采样效率) — ⏳ 可选

**设计预期**: 减少采样步数 4→3, mAP 保持

| 验证目标 | 插桩方式 | 充分性 | 说明 |
|---------|---------|:------:|------|
| mAP 保持 | CocoMetric | ✅ | 无需训练, 仅改配置推理 |
| box 收敛步数 ≤3 | 白盒 TrajectoryAnalyzer | ✅ | 验证减步是否在收敛边界 |
| 采样速度提升 | 推理时间对比 | ✅ | 直接测量 |

**结论**: ✅ 插桩充分

### 8.5 总结

| 方向 | 插桩充分性 | 需额外插桩? | 说明 |
|------|:---------:|:----------:|------|
| D'2 (在跑) | ✅ 充分 (v2) | 否 | v1 发现 bug 并修复, v2 有完整 fg/bg 诊断 |
| F (待跑) | ✅ 充分 | 否 | 训练后白盒分析覆盖所有验证目标 |
| G (可选) | ✅ 充分 | 否 | 但方向价值存疑 (白盒显示特征非瓶颈) |
| H (可选) | ✅ 充分 | 否 | 无需训练, 推理验证即可 |

**关键改进**: D'2 v1 的 reg_bias_loss bug 通过训练日志分析发现 (loss_reg_bias 在 epoch 5 降到 0 异常), 证明**训练中实时插桩**对早期发现问题至关重要。v2 添加的 `fg_dw_mean`/`bg_dw_mean` 诊断统计将确保正样本偏移真正消除。
