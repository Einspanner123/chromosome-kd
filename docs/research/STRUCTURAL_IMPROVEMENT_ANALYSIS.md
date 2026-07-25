# 模型结构优化分析：瓶颈诊断与改进方案

> **创建日期**: 2026-07-22
> **更新日期**: 2026-07-22 (加入实测诊断数据)
> **目标**: 从模型结构本身出发，通过插桩诊断识别瓶颈模块，提出符合染色体检测叙事的结构改进方案
> **基线**: KaryoFlow A4 (RF + DPM-Solver++, mAP=0.864, 24obj, checkpoint best_epoch_117)
> **核心原则**: 改进必须 (1) 从模型结构出发而非推理优化, (2) 与染色体检测任务特性深度结合, (3) **用实测数据验证而非理论推断**
>
> **⚠️ 重要**: 本文档第一版 (2026-07-22 上午) 的瓶颈分析全部基于代码阅读和理论推断, 未经实验验证。
> 用户指出这一问题后, 作者编写了 [structural_diagnosis.py](../../experiments/analysis/structural_diagnosis.py) 并在 A4 checkpoint 上跑了 500 张验证图的推理诊断。
> **实测结果推翻了 6 个瓶颈假设中的 2 个**, 显著修正了改进方案的优先级。下方 §零·实测诊断结果 记录了真实数据。

---

## 零、实测诊断结果 (500 张验证图, A4 checkpoint)

> 数据来源: `experiments/analysis/structural_diagnosis.py` + `work_dirs/diagnosis/structural_diagnosis_v2.json`
> 方法: PyTorch forward hooks 提取 6 级 cascade head 内部的真实激活值/权重

### 假设 vs 实测对照表

| 瓶颈假设 | 理论推断结论 | 实测验证结果 | 判定 |
|----------|------------|------------|------|
| 1. RoI 空间结构被压平 | "7×7 空间信息被丢弃" | 空间方差 0.68-0.77 (高), DynamicConv 后降至 0.33-0.39; **D1 消融**: 抹平空间→mAP 0.863→0.009 (Δ=-0.854, 灾难性崩溃) | ✅ **已消融验证** (空间信息至关重要, DynamicConv 有效提取而非丢失) |
| 2. 时间条件化形同虚设 | "scale≈1, shift≈0" | AdaLN alpha \|α\|=0.47-0.81, std=0.61-1.0 | ❌ **推翻** (时间条件化活跃且自适应) |
| 3. 跨提案交互断层 | "self_attn 发散, 无法解耦重叠" | head0 熵=6.08 (近均匀, max=6.21), head2-3 熵~5.0 | ⚠️ 部分支持 (head0 近均匀, head2-3 有聚焦) |
| 4. 尺度-类别未耦合 | "尺寸→类别先验未被利用" | inter_class_log_area_std=0.589, A1→Y 单调递减 | ❌ **推翻** (模型已隐式学到强尺寸→类别映射) |
| 5. 级联头同质化 | "6 头冗余" | head0 修正最大 (reg std 0.94), 后级递减但非完全相同 | ⚠️ 部分支持 (有角色分化但不充分) |
| 6. box_renewal 纯随机 | 代码事实 | 代码事实 | ✅ 确认 (非诊断项) |

### D1 实测: RoI 空间信息利用率

```
head0: roi_spatial_var=0.248  dynconv_out_diversity=0.166  (输入纯噪声, 方差低)
head1: roi_spatial_var=0.748  dynconv_out_diversity=0.360  压缩比 0.48×
head2: roi_spatial_var=0.768  dynconv_out_diversity=0.368  压缩比 0.48×
head3: roi_spatial_var=0.772  dynconv_out_diversity=0.333  压缩比 0.43×
head4: roi_spatial_var=0.684  dynconv_out_diversity=0.392  压缩比 0.57×
head5: roi_spatial_var=0.740  dynconv_out_diversity=0.333  压缩比 0.45×
```

**解读**: RoI 7×7 特征确实有高空间方差 (0.68-0.77), DynamicConv squeeze 后降至 0.33-0.39 (压缩比 ~0.45×)。这证明空间信息**存在且被压缩**。

### D1 消融实验结果 (2026-07-22, 零成本推理验证)

> 脚本: `experiments/analysis/d1_roi_ablation.py` | 结果: `work_dirs/diagnosis/d1_roi_ablation.json`
> 方法: 在 `roi_extractor` 输出上注册 forward hook, 将 7×7 特征做空间平均池化后广播回 7×7 (所有 49 位置值相同, 空间信息完全抹平, 维度不变, DynamicConv 预训练权重完整加载)

| 指标 | Baseline (7×7 原始) | Ablation (空间抹平) | Δ |
|------|-------------------|-------------------|---|
| mAP | 0.863 | **0.009** | **-0.854** |
| AP50 | 0.988 | 0.048 | -0.940 |
| AP75 | 0.972 | 0.000 | -0.972 |
| Latency(ms) | 155.4 | 149.9 | -5.5 |

**所有 24 类全部崩溃到近零** (per-class AP 0.001-0.029), 模型完全失效。

**结论**: 7×7 空间结构对当前架构**至关重要** — DynamicConv 的 `num_output = feat_channels × pooler_resolution² = 256×49 = 12544` 维线性层主动处理 49 个空间位置的特征, 抹平空间后输入变为严重分布偏移 (12544 维中仅 256 维有效), 导致灾难性崩溃。

**对 M1 的指导**:
- 空间信息**不是"丢失"而是"有效提取"** — DynamicConv 已充分利用 7×7 空间结构
- M1 应**增强**现有空间编码 (添加染色体形态学先验: 着丝粒位置、臂长比、弯曲度), 而非**重建**空间编码
- M1 必须作为**并行分支**或**零初始化残差**接入, 不能破坏现有已验证有效的空间通路
- 风险提示: 灾难性崩溃部分源于训练/推理分布偏移 (模型从未见过空间抹平输入), 但 Δ=-0.854 的极端程度仍证明空间通路是模型的核心依赖

### D2 实测: 时间条件化强度 (AdaLN-Zero 模式)

```
head0: |alpha|=0.469  std=0.607  (时间条件化已生效, 非零初始化状态)
head1: |alpha|=0.576  std=0.720
head2: |alpha|=0.709  std=0.855
head3: |alpha|=0.778  std=0.989  (最强)
head4: |alpha|=0.807  std=0.996  (最强)
head5: |alpha|=0.669  std=0.862
```

**解读**: AdaLN-Zero 的零初始化在训练中已被完全克服。alpha 值 0.47-0.81 且 std 0.6-1.0 说明时间条件化**是活跃且自适应的** — 模型学到了根据时间步 t 调整每层的调制强度。alpha 从 head0 到 head4 **递增** (0.47→0.81), 说明后级 cascade head 更依赖时间条件化 (因为后级处理更精细的框, 需要更准确的时间感知)。

**A2 实验 (+0.000 mAP) 的真实原因**: 不是时间条件化无效, 而是 AdaLN-Zero 相对 scale_shift **无额外增益** — 两种时间条件化方式效果相当。这与"时间条件化形同虚设"是完全不同的结论。

### D3 实测: 级联头贡献度

```
head0: cls_std=1.818  reg_std=0.936  (最大修正: 粗定位)
head1: cls_std=1.596  reg_std=0.531
head2: cls_std=1.551  reg_std=0.360
head3: cls_std=1.548  reg_std=1.093  (reg 异常高, 可能负责特定尺度调整)
head4: cls_std=1.451  reg_std=0.841
head5: cls_std=1.405  reg_std=0.450  (最小修正: 精细分类)
```

**解读**: cls_std 从 head0 (1.82) 到 head5 (1.41) **单调递减**, 说明后级预测更确定 (logits 更集中)。head0 做最大修正 (reg std 0.94), 符合"前级粗定位"角色。但各头**不完全冗余** — head3 的 reg_std 异常高 (1.09), 可能有特定角色。等权 deep_supervision (全部 weight=1.0) 可能不是最优, 但头间确实有自然分化。

### D4-fix 实测: 尺度-类别相关性 (基于实际预测框面积)

```
inter_class_log_area_std = 0.589  (类间尺寸差异显著)
inter_class_log_area_range = 2.045  (A1 到 Y 的 log 面积跨度)

A1: log_area=9.68  w=143.6 h=133.3  (n=1600)  ← 最大
A2: log_area=9.63  w=136.5 h=130.2  (n=1588)
A3: log_area=9.43  w=113.8 h=124.7  (n=1600)
B4: log_area=9.29  w=113.0 h=110.8  (n=1598)
B5: log_area=9.24  w=106.3 h=111.6  (n=1573)
C6: log_area=9.18  w=105.9 h=105.5  (n=1583)
C7: log_area=9.07  w=95.3  h=103.4  (n=1549)
C8: log_area=8.91  w=93.3  h=90.2   (n=1565)
C9: log_area=8.84  w=83.8  h=92.8   (n=1551)
C10: log_area=8.82 w=89.8  h=86.6   (n=1559)
C11: log_area=8.83 w=91.0  h=84.2   (n=1597)
C12: log_area=8.84 w=88.0  h=88.0   (n=1595)
D13: log_area=8.52 w=73.8  h=76.2   (n=1595)
D14: log_area=8.47 w=75.7  h=70.7   (n=1586)
D15: log_area=8.42 w=71.5  h=71.5   (n=1560)
E16: log_area=8.29 w=65.3  h=66.5   (n=1574)
E17: log_area=8.29 w=67.9  h=63.4   (n=1611)
E18: log_area=8.15 w=60.3  h=61.7   (n=1619)
F19: log_area=7.96 w=56.0  h=54.4   (n=1523)
F20: log_area=7.96 w=56.7  h=53.4   (n=1564)
G21: log_area=7.64 w=47.8  h=45.9   (n=1552)
G22: log_area=7.77 w=50.6  h=50.1   (n=1543)
X:  log_area=8.87 w=91.3  h=88.2   (n=1116)  ← 中等 (与 C 组相当)
Y:  log_area=7.72 w=47.6  h=50.2   (n=397)   ← 最小, 且样本量最少
```

**解读**: 模型预测的框尺寸与类别之间有**强相关性** (inter_class_log_area_std=0.589)。尺寸从 A1 (log_area=9.68) 到 Y (7.72) **单调递减**, 完全符合 ISCN 生物学分组 (A>B>C>D>E>F>G/Y)。这表明:

1. **模型已隐式学到尺寸→类别映射** — 通过共享 fc_feature → cls_head/reg_head 的间接路径, 无需显式耦合
2. **M2 (尺度-类别耦合头) 的预期收益降低** — 既然模型已经学到了这个先验, 显式耦合的边际收益可能有限
3. **Y 染色体检测样本量最少 (n=397 vs 其他类 ~1500-1600)** — 印证 Y 的低召回率 (mAP=0.780), 问题在 proposal 生成而非分类
4. **G21 和 Y 尺寸相近** (log_area 7.64 vs 7.72) — 尺寸无法区分这两类, 需要形态学特征 (着丝粒位置), 支持 M1 (形态感知 RoI 编码器)

### D5 实测: 自注意力模式

```
head0: entropy=6.084  topk10_cov=0.075  ← 近乎均匀 (max=log(500)=6.21)
head1: entropy=5.444  topk10_cov=0.249
head2: entropy=5.028  topk10_cov=0.375  ← 最聚焦
head3: entropy=4.973  topk10_cov=0.392  ← 最聚焦
head4: entropy=5.544  topk10_cov=0.198
head5: entropy=5.387  topk10_cov=0.218
```

**解读**:
- **head0 的注意力近乎均匀** (熵 6.08 vs 最大 6.21, top-10 仅覆盖 7.5%) — 第一级 cascade head 处理纯噪声框时, 自注意力没有学到有意义的提案关系。这是一个真实发现。
- **head2-3 有一定聚焦** (熵~5.0, top-10 覆盖 37-39%) — 中间级头学到了一些提案关系, 但仍相当发散 (均匀分布的 top-10 覆盖率应为 10/500=2%, 实际 37-39% 说明有一定聚焦)。
- **注意力模式非单调** — head4-5 比	head2-3 更发散, 可能因为后级框已收敛到不同区域, 注意力分散到多个目标。

**对 M3 (重叠感知注意力) 的影响**: head0 的均匀注意力确实说明自注意力在噪声输入时无效, 但 head2-3 表明中间级头学到了一些关系。M3 的收益可能集中在 head0-1, 但这些头处理的是噪声框, 重叠解耦的意义不大。**M3 的优先级应降低**。

### 实测结论: 改进方案优先级修正

| 方案 | 原优先级 | 实测后修正优先级 | 修正原因 |
|------|---------|---------------|---------|
| M1 形态感知 RoI 编码器 | ⭐最高 | ⭐最高 (不变, 但策略调整) | D1 消融证实空间信息至关重要 (Δ=-0.854), DynamicConv 已有效提取 → M1 应**增强**而非重建; D4 显示 G21/Y 尺寸相近需形态区分 |
| M2 尺度-类别耦合头 | ⭐高 | ↓ 中 (降低) | D4-fix 推翻: 模型已隐式学到强尺寸→类别映射, 显式耦合边际收益有限 |
| M3 重叠感知注意力 | 中 | ↓ 低 (降低) | D5 显示 head2-3 已有聚焦, head0 均匀但处理噪声框无重叠解耦意义 |
| M4 级联头角色分化 | 中 | ↑ 中-高 (提升) | D3 显示头间有自然分化但不充分, 等权 deep_supervision 可优化 |
| M5 学习式 renewal | 低 | 低 (不变) | D4 显示 Y 样本量最少 (n=397), 可能是 renewal 问题, 但收益不确定 |

---

## 〇、现有架构全景

### 0.1 数据流总览

```
Image → ResNet-50 (frozen_stages=1) → FPN [P2,P3,P4,P5] (256ch)
                                              │
                    ┌─────────────────────────┘
                    ▼
    DiffusionDetHead.forward(features, bboxes, t)
    │
    ├─ time_mlp: SinusoidalPosEmb(256) → Linear(256→1024) → SiLU → Linear(1024→1024)
    │   产出 time_emb [bs, 1024], 被 6 个级联头共享
    │
    └─ for i in 0..5 (6 级级联头):
         │  (可选 PHTR: time_emb_i = time_emb * scale_i + shift_i)
         │
         SingleDiffusionDetHead.forward(features, curr_bboxes, proposals, roi_extractor, time_emb_i)
         │
         ├─ RoIAlign: features + bboxes → roi_features [N, 256, 7, 7]
         │   (多尺度: 按 RoI 面积分配到 P2-P5, finest_scale=56)
         │
         ├─ proposals = mean(roi_features) if None  [bs, N, 256]
         │
         ├─ ① self_attn: MultiheadAttention(256, 8 heads)
         │   proposals [N, bs, 256] → attn → +residual → norm1
         │   (提案间自注意力, 500×500 attention matrix)
         │
         ├─ ② DynamicConv (inst_interact):
         │   proposals → dynamic_layer → conv params [N, 2×256×64]
         │   roi_features [N, 49, 256] → bmm(param1) → norm1 → relu
         │                           → bmm(param2) → norm2 → relu
         │                           → reshape → out_layer → norm3 → relu  [1, N, 256]
         │   → +residual → norm2
         │   (提案条件化动态卷积, 每个 proposal 独立处理自己的 7×7 RoI)
         │
         ├─ ③ FFN: Linear(256→2048) → ReLU → dropout → Linear(2048→256)
         │   → +residual → norm3
         │
         ├─ ④ 时间条件化 (scale_shift 模式):
         │   time_emb [1024] → time_mlp(SiLU→Linear(1024→512)) → (scale, shift) [各 256]
         │   fc_feature = fc_feature * (scale + 1) + shift
         │   (仅末端调制, 不作用于 self_attn/DynamicConv/FFN 内部)
         │
         ├─ cls_head: 1×(Linear+LN+ReLU) → Linear(256→24)    分类
         └─ reg_head: 3×(Linear+LN+ReLU) → Linear(256→4)     回归 (cxcywh deltas)

         curr_bboxes = pred_bboxes.detach()  (cascade_detach, 梯度隔离)
```

### 0.2 推理循环 (predict)

```
x_raw = randn(bs, 500, 4)  扩散空间噪声框
for step_idx, (t_curr, t_next) in time_pairs (4 步):
    cls_logits, pred_bboxes, x0_raw = _forward_at_t(features, x_raw, t_curr)
        └─ 运行全部 6 级级联头, 仅取最后一级输出
    ensemble_results.append((cls_logits, pred_bboxes))  集成
    x_raw = dpm_solver.step(x_raw, x0_raw, t_curr, step_idx)  DPM-Solver++ 推进
    if box_renewal:
        x_raw = apply_box_renewal(x_raw, cls_logits, x0_pred, t_curr)
            └─ 低置信度框 (< score_thr) 替换为 randn (或 VGAR: α·x0 + (1-α)·randn)
post_process: concat(ensemble_results) → batched_nms
```

### 0.3 训练循环 (loss)

```
t = sample_t(bs)  均匀采样 [0,1] (RF)
x_noisy = rf.q_sample(x_start, noise, t)  前向扩散
curr_bboxes = raw_to_xyxy(x_noisy)
all_cls_logits, all_pred_bboxes = forward(features, curr_bboxes, t)  全部 6 级级联头
criterion(outputs, targets, t):
    主损失: 最后一级 cascade head 的 cls + bbox + giou
    aux 损失: aux_0..aux_4 (前 5 级), 全部 weight=1.0 (无衰减!)
```

### 0.4 Per-class AP 现状 (A3 checkpoint, 24obj 验证集)

| 组 | 类别 | mAP | 趋势 |
|----|------|-----|------|
| A (最大) | A1-A3 | 0.906-0.916 | 最好 |
| B | B4-B5 | 0.905-0.908 | |
| C (亚中) | C6-C12 | 0.875-0.904 | 内部混淆 |
| D (近中) | D13-D15 | 0.846-0.856 | |
| E | E16-E18 | 0.834-0.853 | |
| F (小) | F19-F20 | 0.818-0.820 | |
| G (最小常) | G21-G22 | **0.787-0.788** | 最差 |
| X | X | 0.879 | |
| Y (最小) | Y | **0.781** | 最差 |

**关键规律**: 染色体尺寸与 mAP 强正相关 (A→G/Y, 0.916→0.781)。最小的 G21/G22 和 Y 是主要瓶颈。

---

## 一、架构瓶颈分析

### 瓶颈 1: RoI 空间结构在编码早期被压平 (Shape Information Loss)

**代码位置**: [single_head.py:287-289](../../ldmdet/core/single_head.py#L287-L289) → [dynamic_conv.py:183-200](../../ldmdet/core/dynamic_conv.py#L183-L200)

```python
# single_head.py: RoI 特征 [N, 256, 7, 7] 被展平为 [49, N, 256]
roi_features = roi_features.view(bs * num_boxes, self.feat_channels, -1).permute(2, 0, 1)

# dynamic_conv.py: 7×7 空间结构被 squeeze 成 1×256
features = features.transpose(0, 1)  # (N, 49, 256)
# ... bmm + norm + relu ...
features = features.reshape(features.size(0), -1)  # (N, 49*256) → out_layer → (N, 256)
```

**问题**: RoIAlign 提取的 7×7 空间特征图编码了染色体的**臂长比、着丝粒位置、弯曲度**等形态信息——这正是 24 类细粒度判别的核心依据。但 DynamicConv 在第一步就将 7×7 展平并 squeeze 到 256 维，空间结构在编码早期就丢失。

**染色体叙事**: 染色体核型分析的临床标准 (ISCN) 正是基于形态学分组: A 组 (大, 近中着丝粒)、C 组 (中等, 亚中着丝粒, 仅靠带纹区分)、G 组 (小, 近端着丝粒)。当前架构在 RoI 编码阶段就压平了这些形态线索，迫使 cls_head 从已丢失空间信息的 256 维特征中重建判别——这在 C 组内部 (C6-C12, mAP 0.875-0.904) 和 G/Y 组 (mAP 0.781-0.788) 造成的损失最大。

**与已有方向的关系**: 方向 C1 (形态感知分类) 已规划 `ShapeAttention` 模块并预留了 hook ([single_head.py:277-278](../../ldmdet/core/single_head.py#L277-L278)), 但**从未实现**。现有 hook 为空 (`shape_attention=None`), 需要填充。

### 瓶颈 2: 时间条件化过晚且过浅 (Late & Shallow Time Conditioning)

**代码位置**: [single_head.py:411-439](../../ldmdet/core/single_head.py#L411-L439)

```python
# scale_shift 模式: time_emb 仅在 FFN 之后调制最终特征
fc_feature = obj_features.transpose(0, 1).reshape(bs * num_boxes, -1)
scale_shift = self.time_mlp(time_emb)
scale, shift = scale_shift.chunk(2, dim=1)
fc_feature = fc_feature * (scale + 1) + shift  # ← 唯一的时间注入点
```

**问题**: time_emb (1024 维, 编码当前噪声水平 t) 只在 self_attn → DynamicConv → FFN **全部完成后**才调制特征。这意味着三个子层在**不知道当前噪声水平**的情况下处理特征。AdaLN-Zero (A2, +0.000 mAP) 尝试了逐层调制但零收益——但 A2 是**均匀地**对所有子层施加 AdaLN, 可能因调制过强反而干扰了已学好的特征。

**染色体叙事**: RF 的速度场 $v = x_1 - x_0$ 恒定, 但沿轨迹的不同位置 $t$ 需要不同的处理策略:
- 早期 (t→1): proposals 是纯噪声, 应侧重**粗定位** (regression)
- 后期 (t→0): proposals 接近 GT, 应侧重**细粒度分类** (classification)

当前架构的 6 个级联头**完全同构**, 在所有 $t$ 下执行相同的 self_attn → DynamicConv → FFN 流程, 没有随 $t$ 自适应调整处理重心。

### 瓶颈 3: 跨提案交互仅作用在压平特征上, 无法解耦重叠 (Overlap Blindness)

**代码位置**: [single_head.py:417](../../ldmdet/core/single_head.py#L417) (self_attn) vs [dynamic_conv.py:177-201](../../ldmdet/core/dynamic_conv.py#L177-L201) (DynamicConv)

**问题**: 两条交互路径存在信息断层:
- **self_attn**: 在 proposals (256 维压平特征) 上做 500×500 自注意力——提案间交互, 但**没有空间 RoI 信息**
- **DynamicConv**: 每个提案**独立地**用自己的 RoI 7×7 特征做动态卷积——有空间信息, 但**提案间无交互**

当两条染色体重叠时, 它们的 RoI 区域有共享像素。DynamicConv 独立处理每个 RoI, 无法让重叠提案"协商"哪些像素属于自己; self_attn 能让提案交互, 但此时 RoI 空间信息已被压平, 无法定位重叠区域。

**染色体叙事**: 染色体中期相图像中 97.8% 的图存在框重叠 (IoU>0)。虽然高 IoU (>0.5) 的框对仅占 9.4% 的图, 但即使低 IoU 重叠也会导致 RoI 特征污染。当前架构的"提案交互无空间信息 + 空间处理无提案交互"断层, 使重叠解耦依赖后续 NMS 被动处理, 而非在特征层面主动解耦。

### 瓶颈 4: 分类与回归头完全独立, 未利用尺度-类别强先验 (Scale-Class Decoupling)

**代码位置**: [single_head.py:365-366](../../ldmdet/core/single_head.py#L365-L366)

```python
def _predict(self, fc_feature, bboxes, bs, num_boxes):
    class_logits = self.cls_head(fc_feature)        # 分类: 256→24
    pred_bboxes = self._predict_bboxes(fc_feature)  # 回归: 256→4
    # 两者独立, 无信息交换
```

**问题**: cls_head 和 reg_head 从同一个 fc_feature 分支, 但**互不通信**。染色体检测有一个极强的生物学先验: **尺寸 → 类别**。A1 最大 (~10μm), Y 最小 (~2μm), 尺寸跨度 5×。这个先验在当前架构中完全没有被显式利用。

**数据证据**: per-class AP 与尺寸强正相关 (A1 0.916 → Y 0.781, Δ=0.135)。最小的 G21/G22 和 Y 是性能瓶颈。如果 reg_head 预测的尺寸信息能反馈给 cls_head, 模型可以排除尺寸不匹配的类别候选, 直接缩小搜索空间。

**与已有方向的关系**: 方向 B (解耦分类定位) 追求的是**分离**两条路径以减少梯度竞争。这里提出的是**反向**思路: 在末端**耦合** (尺寸→类别先验), 两者不矛盾——解耦在共享主干层面, 耦合在预测头层面。

### 瓶颈 5: 级联头角色同质化, 无特征传递 (Cascade Homogeneity)

**代码位置**: [head.py:418-422](../../ldmdet/core/head.py#L418-L422), [criterion.py:73-81](../../ldmdet/criterion/criterion.py#L73-L81)

**问题**:
1. 6 个级联头**完全同构** (deepcopy), 没有角色分化
2. `cascade_detach=True`: 每级只接收上一级的 `pred_bboxes.detach()`, **不传递特征**
3. deep_supervision 的 6 个头损失**等权** (weight=1.0, 无衰减), 没有"后级更重要"的先验

**问题本质**: 级联头的初衷是"逐级精修"——前级粗定位, 后级细分类。但同构 + 等权 + 无特征传递的设计让 6 个头学到的功能高度重叠, 部分头可能是冗余的。

**染色体叙事**: 染色体检测的级联精修有明确的生物学对应:
- 前级 (head 0-1): 处理高噪声框, 应侧重**分离重叠 + 粗定位** (把 500 个噪声框收敛到 ~46 条染色体附近)
- 后级 (head 4-5): 处理低噪声框, 应侧重**细粒度分类** (C 组带纹区分)

当前架构无法实现这种角色分化。

### 瓶颈 6: box_renewal 使用纯随机噪声, 无空间先验 (Unlearned Renewal)

**代码位置**: [sampling.py:231-252](../../ldmdet/diffusion/sampling.py#L231-L252)

```python
for i in range(bs):
    keep = scores[i] > self.score_thr  # 高置信度保留
    num_renew = (~keep).sum()
    if num_renew > 0:
        noise = torch.randn(num_renew, 4, device=device)  # ← 纯随机!
        x_raw_new[i, ~keep] = noise
```

**问题**: 低置信度框被替换为**纯随机噪声**, 完全忽略图像内容。但染色体中期相铺展有空间结构: 染色体均匀分布在视野中, 不会堆叠在一个角落。纯随机 renewal 可能将新框放在空白区域, 浪费 proposal 容量。

**染色体叙事**: 中期相铺展图像中, 染色体覆盖区域有一定规律性 (避免极端聚集/分散)。一个轻量的空间先验网络可以从 FPN 特征预测"哪里可能有染色体", 引导 renewal 框落在高概率区域, 提升小目标 (G/Y) 的召回率。

---

## 二、插桩诊断计划

在实施结构改进前, 先通过 Probe 插桩**量化验证**每个瓶颈的严重程度。所有探针设计为**非侵入式** (零开销 when disabled), 复用现有 `probe.record_tensor_stats` / `probe.record_inference_tensor_stats` 接口。

### 诊断 1: RoI 空间信息利用率 (验证瓶颈 1)

**目标**: 量化 7×7 RoI 空间结构在 DynamicConv 前后的信息损失

**插桩位置**: [single_head.py](../../ldmdet/core/single_head.py) `_forward_scale_shift` 方法

**新增探针**:
```python
# 在 roi_features 展平前, 记录空间方差
# roi_features: [N, 256, 7, 7] → 计算每通道空间 std
spatial_var = roi_features.var(dim=[2, 3]).mean()  # 空间方差均值
probe.record_tensor_stats('single_head/roi_spatial_var_before', spatial_var)

# 在 DynamicConv 输出后, 记录特征多样性
# inst_out: [1, N, 256] → 计算 proposal 间方差
proposal_diversity = inst_out.squeeze(0).var(dim=0).mean()
probe.record_tensor_stats('single_head/proposal_diversity_after_dynconv', proposal_diversity)
```

**诊断逻辑**: 如果 `roi_spatial_var_before` 高 (空间信息丰富) 但对分类贡献低 (移除空间维度后 mAP 不降), 说明空间信息确实被浪费; 反之说明空间信息不重要。

**零成本验证方案**: 临时将 RoI 7×7 替换为全局平均池化 (1×1), 跑推理对比 mAP。如果 mAP 不降, 证明 7×7 空间结构对当前架构无贡献, 是改进空间。

### 诊断 2: 时间条件化有效性 (验证瓶颈 2)

**目标**: 量化 scale_shift 在不同 $t$ 下的调制强度

**插桩位置**: [single_head.py:435-438](../../ldmdet/core/single_head.py#L435-L438)

**新增探针**:
```python
scale_shift = self.time_mlp(time_emb)
scale, shift = scale_shift.chunk(2, dim=1)
# 记录 scale 偏离 1 的程度, shift 偏离 0 的程度
probe.record_tensor_stats('single_head/time_scale_deviation', (scale - 1).abs().mean())
probe.record_tensor_stats('single_head/time_shift_magnitude', shift.abs().mean())
# 记录 scale/shift 与 t 的相关性 (按 t 分桶)
```

**诊断逻辑**: 如果 `time_scale_deviation ≈ 0` 且 `time_shift_magnitude ≈ 0`, 说明 time conditioning 形同虚设 (scale≈1, shift≈0 即恒等变换)。如果调制强度随 $t$ 无变化, 说明时间条件化未学到 $t$-adaptive 行为。

### 诊断 3: 级联头贡献度分布 (验证瓶颈 5)

**目标**: 量化每级 cascade head 对最终预测的贡献, 识别冗余头

**插桩位置**: 已有 `cascade/head{i}/cls_logits` 和 `cascade/head{i}/pred_bboxes` 探针, 需增加**头间差异**度量

**新增探针** (在 [head.py:362-424](../../ldmdet/core/head.py#L362-L424) 循环内):
```python
# 相邻头间的框变化量 (归一化)
if i > 0:
    box_delta = (pred_bboxes - prev_bboxes).norm(dim=-1)
    box_scale = (pred_bboxes[..., 2:] - pred_bboxes[..., :2]).norm(dim=-1)
    relative_delta = (box_delta / box_scale.clamp(min=1e-6)).mean()
    probe.record_inference_scalar(f'cascade/head{i}/box_delta_vs_prev', relative_delta)

    # 分类 logit 变化量
    cls_delta = (cls_logits - prev_logits).abs().mean()
    probe.record_inference_scalar(f'cascade/head{i}/cls_delta_vs_prev', cls_delta)
```

**诊断逻辑**: 如果 head 3-5 的 `box_delta_vs_prev` 和 `cls_delta_vs_prev` 趋近于 0, 说明后级头是冗余的 (输入≈输出), 可以削减或重新分配角色。结合 IO4 (级联头提前退出) 的退出统计可以交叉验证。

### 诊断 4: 尺度-类别相关性 (验证瓶颈 4)

**目标**: 量化模型预测的框尺寸与类别之间的隐式相关性

**插桩位置**: 推理后处理阶段

**新增探针** (在 [head.py](../../ldmdet/core/head.py) predict 的 ensemble 收集后):
```python
# 对最终预测, 统计每类的平均框面积
scores = torch.sigmoid(cls_logits).max(-1)
for cls_id in range(num_classes):
    mask = (cls_logits.argmax(-1) == cls_id) & (scores > 0.3)
    if mask.any():
        areas = (pred_bboxes[..., 2] - pred_bboxes[..., 0]) * \
                (pred_bboxes[..., 3] - pred_bboxes[..., 1])
        probe.record_inference_scalar(
            f'pred/class{cls_id}_mean_area', areas[mask].mean().item()
        )
```

**诊断逻辑**: 如果模型已隐式学到尺度→类别映射 (A1 面积最大, Y 最小), 则显式耦合的边际收益有限; 如果相关性弱 (同类面积方差大), 则显式耦合有较大改进空间。对照 GT 的尺寸分布可以计算"模型预测尺寸的准确率"。

### 诊断 5: 自注意力模式分析 (验证瓶颈 3)

**目标**: 分析 self_attn 的注意力分布, 判断是否聚焦于重叠邻居

**插桩位置**: [single_head.py:208-216](../../ldmdet/core/single_head.py#L208-L216) `_self_attn` 方法

**新增探针** (需要修改 `_sdpa_self_attn` 返回 attention weights, 或用 `nn.MultiheadAttention` 路径):
```python
# 注意力熵: 高=发散, 低=聚焦
attn_probs = F.softmax(attn_weights, dim=-1)  # [bs, num_heads, N, N]
attn_entropy = -(attn_probs * (attn_probs + 1e-8).log()).sum(-1).mean()
probe.record_tensor_stats('single_head/attn_entropy', attn_entropy)

# Top-k 注意力覆盖率: 前 k 个注意力权重之和
topk_cov = attn_probs.topk(10, dim=-1)[0].sum(-1).mean()
probe.record_tensor_stats('single_head/attn_topk10_coverage', topk_cov)
```

**诊断逻辑**: 如果注意力熵高 (发散) 且 top-k 覆盖率低, 说明 self_attn 没有学到有意义的提案关系, 重叠解耦能力弱。

---

## 三、结构改进方案 (按优先级)

### 方案 M1: 形态感知 RoI 编码器 (Morphology-Aware RoI Encoder) — ⭐ 最高优先级

**对应瓶颈**: 1 (RoI 空间结构丢失)
**对应已有方向**: 方向 C1 (已规划未实现)
**染色体叙事强度**: ★★★★★
**D1 消融验证**: ✅ 现有 7×7 + DynamicConv 空间编码至关重要 (抹平→mAP 0.009)。M1 的零初始化残差设计 (`roi_features + morph_emb`) 完美契合 D1 结论 — 初始状态不改变 A4 行为, 训练中逐步增强形态编码, 不破坏已验证有效的空间通路。

#### 3.1.1 设计

在 RoIAlign 之后、DynamicConv 之前插入一个**轻量空间形态编码器**, 显式保留并增强 7×7 RoI 的空间结构:

```python
class MorphologyAwareRoIEncoder(nn.Module):
    """形态感知 RoI 编码器: 在 7×7 RoI 特征上提取染色体形态线索.

    染色体形态学判据 (ISCN 标准):
    - 臂长比 (p/q): 着丝粒位置 → 近中/亚中/近端着丝粒
    - 整体尺寸: A>G>Y, 5× 跨度
    - 弯曲度: 部分类别有特征性弯曲

    设计: 用方向解耦卷积分别捕获水平 (臂长方向) 和垂直 (着丝粒方向) 形态,
    再融合为增强的 RoI 特征. 残差连接确保不破坏预训练.
    """
    def __init__(self, channels=256, reduction=4):
        super().__init__()
        # 水平方向 (臂长比): (7,1) 卷积沿宽方向扫描
        self.h_conv = nn.Conv2d(channels, channels // reduction, (7, 1))
        # 垂直方向 (着丝粒): (1,7) 卷积沿高方向扫描
        self.v_conv = nn.Conv2d(channels, channels // reduction, (1, 7))
        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(channels // reduction * 2, channels, 1),
            nn.LayerNorm([channels, 1, 1]),  # 逐通道
            nn.SiLU(),
        )
        # 零初始化最后一层, 确保加载预训练时输出≡0 (残差恒等)
        nn.init.zeros_(self.fuse[0].weight)
        nn.init.zeros_(self.fuse[0].bias)

    def forward(self, roi_features):
        # roi_features: [N, C, 7, 7]
        h_feat = self.h_conv(roi_features)   # [N, C//r, 1, 7]
        v_feat = self.v_conv(roi_features)   # [N, C//r, 7, 1]
        # 广播到 7×7
        h_feat = h_feat.expand(-1, -1, 7, -1)
        v_feat = v_feat.expand(-1, -1, -1, 7)
        morph_emb = self.fuse(torch.cat([h_feat, v_feat], dim=1))
        return roi_features + morph_emb  # 残差
```

#### 3.1.2 接线方式

填充现有 `shape_attention` hook ([single_head.py:277-278](../../ldmdet/core/single_head.py#L277-L278)):
```python
# single_head.py forward() 中已有:
if self.shape_attention is not None:
    roi_features = self.shape_attention(roi_features)
# 只需在配置中传入 MorphologyAwareRoIEncoder 实例
```

#### 3.1.3 零初始化保证

`fuse` 最后一层零初始化 → 初始 `morph_emb≡0` → `roi_features + 0 = roi_features` → 加载 A3 预训练权重时行为不变, 训练初期梯度通过残差路径流回, 逐步学到形态增强。

#### 3.1.4 预期收益

- C 组 (C6-C12, 亚中着丝粒, mAP 0.875-0.904): 臂长比是主要区分依据, 方向解耦卷积直接编码
- G/Y 组 (最小, mAP 0.781-0.788): 尺寸虽小但着丝粒位置 (近端) 仍是判别特征
- 参数开销: ~2×(256×64×7) + 128×256 ≈ 265K 参数 (<总参数 0.5%)

#### 3.1.5 插桩验证

- 诊断 1 (RoI 空间信息): 对比有无 M1 的 `roi_spatial_var` → mAP 关系
- 消融: 禁用 h_conv / v_conv 分别测试, 验证方向解耦的必要性
- 可视化: 对 C 组染色体绘制 `morph_emb` 的激活图, 确认着丝粒位置被编码

#### 3.1.6 BF16 实验结果与 fuse 权重分析 (2026-07-23)

> ⚠ **BF16 条件下的初步结果**, 需 FP32 复现确认。BF16 本身导致 A4 掉点 -0.038, 是主要"退化"来源。

**实验设置**: workstation A5000 24GB, BF16 AMP (`amp_dtype='bfloat16'`), 从 A4 best checkpoint 微调 30 epoch, lr=1e-5, seed 42。
**配置**: [m1_morphology_aware_24obj_ws.py](../../experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj_ws.py)

**mAP 结果**:

| 配置 | mAP | 说明 |
|------|-----|------|
| A4 (FP32) | 0.863 | 基线 (best@ep117) |
| A4+BF16 | 0.825 | **BF16 掉点 -0.038** (零成本 eval 诊断) |
| M1+BF16 (best@ep1) | 0.818 | 全程 0.811-0.818 波动, 30 epoch 未改善 |
| **M1 vs A4+BF16** | **-0.007** | noise 范围但偏负面 |

**Per-class AP 对比 (M1+BF16 ep1 vs A4+BF16, 均 BF16)**:

| 类别 | A4+BF16 | M1+BF16 | Δ | 说明 |
|------|---------|---------|------|------|
| A1 | 0.890 | 0.887 | -0.003 | |
| A2 | 0.884 | 0.884 | 0.000 | 唯一持平 |
| A3 | 0.874 | 0.870 | -0.004 | |
| B4 | 0.874 | 0.867 | -0.007 | |
| B5 | 0.874 | 0.872 | -0.002 | |
| C6 | 0.870 | 0.864 | -0.006 | C 组 (亚中着丝粒) |
| C7 | 0.864 | 0.854 | -0.010 | |
| C8 | 0.850 | 0.843 | -0.007 | |
| C9 | 0.842 | 0.835 | -0.007 | |
| C10 | 0.844 | 0.840 | -0.004 | |
| C11 | 0.835 | 0.831 | -0.004 | |
| C12 | 0.855 | 0.842 | **-0.013** | C 组退化最严重 |
| D13 | 0.815 | 0.813 | -0.002 | |
| D14 | 0.815 | 0.809 | -0.006 | |
| D15 | 0.807 | 0.801 | -0.006 | |
| E16 | 0.814 | 0.803 | -0.011 | |
| E17 | 0.804 | 0.788 | **-0.016** | 退化最严重 |
| E18 | 0.788 | 0.776 | -0.012 | |
| F19 | 0.766 | 0.761 | -0.005 | |
| F20 | 0.780 | 0.765 | **-0.015** | |
| G21 | 0.737 | 0.732 | -0.005 | G/Y 组 (尺寸相近) |
| G22 | 0.739 | 0.729 | -0.010 | |
| X | 0.843 | 0.840 | -0.003 | |
| Y | 0.726 | 0.714 | -0.012 | |

**Per-class 关键发现**: **所有 24 类全部退化, 没有一类改善**。退化最严重: E17 (-0.016)、F20 (-0.015)、C12 (-0.013)、E18/Y (-0.012)。M1 预期受益的 C 组 (亚中着丝粒, 臂长比判别) 和 G/Y 组 (尺寸相近需形态区分) 均未改善, 反而退化。

**fuse 权重分析 (关键发现)**:

提取 M1+BF16 训练后 6 个 cascade head 的 h_conv/v_conv 权重:

```
head0: fuse_norm=0.2150  h_conv_norm=4.6536  v_conv_norm=4.6471  h/v=1.001
  h_energy (高度/臂长比方向): min=0.0118  max=0.0119  ratio=1.01  std=0.0000
  v_energy (宽度/着丝粒方向): min=0.0118  max=0.0119  ratio=1.01  std=0.0000
(所有 6 head 一致, ep1 fuse_norm=0.006-0.007, ep30 fuse_norm=0.238-0.260)
```

**结论**: h_conv 和 v_conv 权重沿空间维度**完全均匀** (ratio=1.01, std=0.0000) — M1 **未学到方向性形态信息**。morph_emb 退化为常数偏置, 不是有意义的形态编码。fuse 层确实在学习 (norm 从 0.006 增长到 0.238), 但学到的是无方向性的常数偏置。

**根因分析**:
1. **零初始化 fuse 的梯度瓶颈**: fuse 最后一层零初始化 → 初始 morph_emb≡0 → 梯度通过残差路径流回 fuse, 但 h_conv/v_conv 的梯度信号极弱 (需穿过零初始化的 fuse 层), 导致方向卷积权重无法分化
2. **BF16 加剧梯度噪声**: BF16 精度下, 本就微弱的方向梯度进一步被噪声淹没
3. **设计缺陷**: 方向解耦卷积 (7,1)+(1,7) 的感受野与 7×7 RoI 尺寸相同, 可能缺乏足够的空间上下文来区分方向

**待办**: ~~FP32 复现~~ → ✅ 已完成, 见下方 §3.1.7

#### 3.1.7 FP32 复现结果 (2026-07-24, ✅ 已完成)

> **核心结论**: M1 FP32 mAP=0.862, 与 A4 (0.863) **统计上持平** (Δ=-0.001)。BF16 实验完全误导 — BF16 导致 -0.044 虚假退化。但 h_conv/v_conv 在 FP32 下**仍然均匀**, 确认是设计问题而非精度问题。

**实验设置**: 本地 A6000 49GB, FP32 (无 amp_dtype), 从 A4 best checkpoint 微调 30 epoch, lr=2e-5 (2× BF16 实验), 1 epoch warmup, seed 42。
**配置**: [m1_morphology_aware_24obj_fp32.py](../../experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj_fp32.py)
**显存**: 37487 MiB (FP32, A6000 49GB 可行)

**mAP 结果**:

| 配置 | mAP | Δ vs A4 | 说明 |
|------|-----|---------|------|
| A4 (FP32) | 0.863 | — | 基线 |
| A4+BF16 | 0.825 | -0.038 | BF16 本身掉点 |
| M1 (BF16) | 0.818 | -0.045 | BF16 虚假退化 |
| **M1 (FP32)** | **0.862** | **-0.001** | **统计上持平！BF16 误导** |

**mAP 轨迹**: ep1=0.854 → ep3=0.860 → ep19=**0.862 (best)** → ep30=0.859。全程稳定在 0.852-0.862, 无崩溃。BF16 实验 ep1=0.818 且 30 epoch 未改善, FP32 首个 epoch 即超过 BF16 best。

**fuse 权重分析 (FP32 best@ep19 vs BF16 ep30)**:

| 指标 | M1 FP32 (ep19) | M1 BF16 (ep30) | 结论 |
|------|---------------|----------------|------|
| fuse norm | 0.36-0.43 | 0.215 | FP32 增长更大 (lr 2×) |
| h_conv ratio (空间) | 1.01-1.02 | 1.01 | **仍均匀** |
| h_conv std (空间维度) | 0.0001 | 0.0000 | **仍均匀** |
| v_conv ratio (空间) | 1.02 | 1.01 | **仍均匀** |

→ **h_conv/v_conv 即使在 FP32 下仍然完全均匀** — 确认是**设计问题**, 不是精度问题。fuse 层学到的是常数偏置, 不是方向性形态编码。

**Per-class AP (M1 FP32 best@ep19 vs A4 best@ep117)**:

| 类别 | A4 | M1 FP32 | Δ | 说明 |
|------|-----|---------|------|------|
| A1 | 0.911 | 0.909 | -0.002 | |
| A2 | 0.910 | 0.909 | -0.001 | |
| A3 | 0.904 | 0.903 | -0.001 | |
| B4 | 0.905 | 0.906 | **+0.001** | ✓ |
| B5 | 0.906 | 0.906 | 0.000 | |
| C6 | 0.903 | 0.902 | -0.001 | |
| C7 | 0.897 | 0.897 | 0.000 | |
| C8 | 0.888 | 0.883 | -0.005 | 最大退化 |
| C9 | 0.879 | 0.881 | **+0.002** | ✓ |
| C10 | 0.881 | 0.878 | -0.003 | |
| C11 | 0.872 | 0.872 | 0.000 | |
| C12 | 0.890 | 0.889 | -0.001 | |
| D13 | 0.856 | 0.852 | -0.004 | |
| D14 | 0.853 | 0.851 | -0.002 | |
| D15 | 0.844 | 0.841 | -0.003 | |
| E16 | 0.853 | 0.853 | 0.000 | |
| E17 | 0.842 | 0.841 | -0.001 | |
| E18 | 0.832 | 0.832 | 0.000 | |
| F19 | 0.817 | 0.820 | **+0.003** | ✓ 小目标改善 |
| F20 | 0.819 | 0.818 | -0.001 | |
| G21 | 0.789 | 0.788 | -0.001 | |
| G22 | 0.787 | 0.787 | 0.000 | |
| X | 0.883 | 0.886 | **+0.003** | ✓ 性染色体改善 |
| Y | 0.780 | 0.779 | -0.001 | |

→ **4 类改善** (B4, C9, F19, X), **6 类持平**, **14 类轻微退化** (max -0.005)。比 BF16 (24 类全退化) 和方向 C (3 改善/20 退化) 显著更接近 A4。

**根因分析 (更新)**:
1. **零初始化 fuse 的梯度瓶颈** (确认): FP32 下 fuse norm 增长更大 (0.36 vs 0.215), 但 h_conv/v_conv 仍均匀 → 梯度瓶颈是设计固有问题, 与精度无关
2. ~~BF16 加剧梯度噪声~~ (证伪): BF16 不是 h_conv/v_conv 均匀的原因, FP32 下同样均匀
3. **设计缺陷** (确认): 方向解耦卷积 (7,1)+(1,7) 的梯度通过零初始化 fuse 后, 空间维度的差异被抹平 → 需重新设计

**M1 当前状态**: **中性** — 不伤害性能 (Δ=-0.001 持平), 但也不帮助 (fuse 退化为常数偏置)。需重新设计才能实现真正的形态感知编码。

**改进方向** (若后续启动 M1-v2):
- 非零初始化 fuse (如小常数初始化 0.01, 打破梯度瓶颈)
- 显式形态先验注入 (臂长比/面积作为输入, 而非依赖卷积发现)
- 注意力机制替代方向卷积 (self-attention 自然捕获空间关系)

---

### 方案 M2: 尺度-类别耦合预测头 (Scale-Class Coupled Head) — ⭐ 高优先级

**对应瓶颈**: 4 (分类回归独立, 未利用尺寸→类别先验)
**染色体叙事强度**: ★★★★★ (最强染色体特异性)

#### 3.2.1 设计

在 cls_head 和 reg_head 之间建立**单向耦合**: reg_head 预测的尺寸信息反馈给 cls_head, 作为分类的先验条件。

```python
class ScaleClassCoupledHead(nn.Module):
    """尺度-类别耦合预测头.

    生物学先验: 染色体尺寸 → 类别有强映射关系.
    A1≈10μm (最大) → Y≈2μm (最小), 尺寸跨度 5×.
    24 类按尺寸排序: A1>A2>A3>B4>B5>C6>...>G21>G22>Y

    设计:
    1. reg_head 先预测 box deltas → 解码得预测尺寸 (w, h)
    2. 尺寸 → 尺寸嵌入 (size_emb, 64 维)
    3. size_emb 注入 cls_head 的中间特征

    零初始化: size_emb 投影层零初始化, 确保预训练兼容.
    """
    def __init__(self, feat_channels=256, num_classes=24, size_emb_dim=64):
        super().__init__()
        # 回归头 (与现有 reg_head 结构一致)
        self.reg_head = self._build_reg_head(feat_channels, num_reg_convs=3)

        # 尺寸嵌入: 将 (w, h) → size_emb
        # 用 sinusoidal + MLP 编码连续尺寸
        self.size_mlp = nn.Sequential(
            nn.Linear(2, size_emb_dim),  # (w, h) → size_emb
            nn.SiLU(),
            nn.Linear(size_emb_dim, size_emb_dim),
        )

        # 尺寸→分类的耦合投影 (零初始化)
        self.size_to_cls = nn.Linear(size_emb_dim, feat_channels)
        nn.init.zeros_(self.size_to_cls.weight)
        nn.init.zeros_(self.size_to_cls.bias)

        # 分类头 (与现有 cls_head 结构一致, 但接受尺寸条件)
        self.cls_head = self._build_cls_head(feat_channels, num_cls_convs=1, num_classes=num_classes)

    def forward(self, fc_feature, bboxes):
        # 1. 先回归
        bboxes_deltas = self.reg_head(fc_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes)

        # 2. 提取预测尺寸
        pred_w = (pred_bboxes[..., 2] - pred_bboxes[..., 0]).clamp(min=1e-6)
        pred_h = (pred_bboxes[..., 3] - pred_bboxes[..., 1]).clamp(min=1e-6)
        # 对数尺度 (尺寸跨度过大, 用 log 压缩)
        size = torch.stack([pred_w.log(), pred_h.log()], dim=-1)  # [N, 2]
        size_emb = self.size_mlp(size)  # [N, 64]

        # 3. 尺寸条件注入分类特征 (零初始化时 size_to_cls(size_emb)≡0)
        cls_feature = fc_feature + self.size_to_cls(size_emb)

        # 4. 分类
        class_logits = self.cls_head(cls_feature)
        return class_logits, pred_bboxes
```

#### 3.2.2 接线方式

替换 [single_head.py:364-371](../../ldmdet/core/single_head.py#L364-L371) 的 `_predict` 方法:
```python
# 原始: cls_head 和 reg_head 独立
# 改为: ScaleClassCoupledHead, reg 先行, 尺寸反馈 cls
```

#### 3.2.3 零初始化保证

`size_to_cls` 零初始化 → 初始 `size_emb` 贡献为 0 → cls_feature = fc_feature → 与原始 cls_head 行为一致。训练中梯度通过 `size_to_cls` 流回 `size_mlp`, 逐步学到尺寸→类别映射。

#### 3.2.4 预期收益

- G21/G22 (mAP 0.787-0.788) 和 Y (mAP 0.781): 尺寸最小, 与其他类差异显著, 尺寸先验可大幅缩小候选空间
- C 组内部 (C6-C12): 尺寸相近但仍有差异, 尺寸先验可辅助分界
- 推理零额外开销 (仅一次 2→64→256 的轻量 MLP)

#### 3.2.5 插桩验证

- 诊断 4 (尺度-类别相关性): 对比有无 M2 的 per-class AP, 重点关注 G/Y 组
- 消融: 随机打乱 size_emb (破坏尺寸信息), 验证尺寸先验的贡献
- 分析: 可视化 `size_to_cls` 权重, 确认模型学到尺寸→类别的映射方向

---

### 方案 M3: 重叠感知跨提案注意力 (Overlap-Aware Cross-Proposal Attention) — 中优先级

**对应瓶颈**: 3 (提案交互无空间信息, 空间处理无提案交互)
**染色体叙事强度**: ★★★★☆

#### 3.3.1 设计

在 self_attn 之后、DynamicConv 之前插入一个**IoU 感知的跨提案注意力**, 让重叠提案交换 RoI 空间信息:

```python
class OverlapAwareAttention(nn.Module):
    """重叠感知跨提案注意力.

    问题: self_attn 在压平的 256 维特征上交互, 无空间信息;
          DynamicConv 在 7×7 RoI 上独立处理, 无提案交互.
    方案: 对高 IoU 提案对, 交换 RoI 特征的注意力信息.

    计算:
    1. 计算提案对 IoU 矩阵 [N, N]
    2. 对每个提案, 用 RoI 特征与其他高 IoU 提案的 RoI 特征做 cross-attention
    3. 仅对 IoU > threshold 的提案对施加 (稀疏注意力, 控制计算量)
    """
    def __init__(self, channels=256, num_heads=4, iou_threshold=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(channels, num_heads)
        self.norm = nn.LayerNorm(channels)
        self.iou_threshold = iou_threshold
        # 零初始化输出投影, 确保残差恒等
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, proposals, roi_features, bboxes):
        # proposals: [N, bs, 256] (self_attn 后)
        # roi_features: [49, N, 256] (7×7 展平)
        # bboxes: [bs, N, 4] xyxy

        # 计算 IoU 矩阵 (仅对重叠提案对做 cross-attn)
        # ... (稀疏化: 仅 top-k 高 IoU 邻居)
        # cross_attn_out, _ = self.cross_attn(proposals, roi_features_mean, roi_features)
        # proposals = proposals + cross_attn_out
        return self.norm(proposals)
```

#### 3.3.2 计算量控制

500×500 的全 cross-attention 过重。稀疏化策略:
- 仅对 IoU > 0.1 的提案对计算 (染色体数据中仅 2.6% 框对有重叠)
- 每个提案仅与 top-10 高 IoU 邻居交互
- 实际计算量 ≈ 500×10 = 5000 对 (vs 全量 250000 对)

#### 3.3.3 预期收益与风险

- **收益**: 重叠染色体 RoI 特征污染的主动解耦, 可能提升高重叠图像的 AP
- **风险**: 染色体数据高 IoU (>0.5) 框对仅占 9.4% 的图, 收益可能有限
- **建议**: 先用诊断 5 确认 self_attn 注意力模式后再决定是否实施

---

### 方案 M4: 级联头角色分化 (Cascade Role Differentiation) — 中优先级

**对应瓶颈**: 5 (级联头同质化, 无特征传递)
**染色体叙事强度**: ★★★☆☆

#### 3.4.1 设计

**方案 A (轻量, 推荐先试)**: 级联头损失权重衰减
```python
# criterion.py: deep_supervision 的 aux 损失加权
# 当前: 全部 weight=1.0
# 改为: 线性衰减 1.0 → 0.2 (head 0 最小, head 5 最大)
aux_weights = torch.linspace(0.2, 1.0, num_heads)  # [0.2, 0.36, 0.52, 0.68, 0.84, 1.0]
# 或指数衰减: 0.5^i (更激进)
```

**方案 B (结构, 需重训练)**: 级联头间特征传递
```python
# head.py forward(): 在 cascade 间传递特征 (非 detach)
# curr_feature = head_i(features, curr_bboxes, curr_proposals, ...)
# curr_proposals = curr_feature  # 传递特征 (非 detach)
# curr_bboxes = pred_bboxes.detach()  # 框仍 detach
```

**方案 C (结构, 需重训练)**: 前级头回归侧重, 后级头分类侧重
```python
# 前 3 级: reg_head 3 convs (强回归), cls_head 1 conv (弱分类)
# 后 3 级: reg_head 1 conv (弱回归), cls_head 3 convs (强分类)
# 通过不同配置的 single_head 实现
```

#### 3.4.2 插桩验证

- 诊断 3 (级联头贡献度): 用 `box_delta_vs_prev` 和 `cls_delta_vs_prev` 确定哪些头冗余
- 方案 A 可零成本验证 (仅改损失权重, 复用 A3 checkpoint 微调 10 epoch)

---

### 方案 M5: 学习式空间先验 renewal (Learned Spatial Prior Renewal) — 低优先级

**对应瓶颈**: 6 (box_renewal 纯随机)
**染色体叙事强度**: ★★★☆☆

#### 3.5.1 设计

```python
class SpatialPriorRenewal(nn.Module):
    """从 FPN 特征预测染色体空间分布, 引导 box_renewal.

    中期相铺展: 染色体均匀分布在视野中.
    用 FPN 特征 → 小型 U-Net → 空间热度图 → 采样 renewal 框位置.
    """
    # 轻量: 仅 3 层 1×1 卷积, 从 FPN P3 (1/8 分辨率) 预测热度图
    # renewal 时从热度图高分区域采样框中心, 而非纯随机
```

#### 3.5.2 优先级说明

此方案需要额外训练一个空间先验网络, 实现复杂度较高且收益不确定 (renewal 主要影响召回, 对 mAP 的提升有限)。建议在 M1-M4 验证后再考虑。

---

## 四、实现路线图

### Phase 0: 插桩诊断 (零成本, 1 天) — ✅ 已完成

1. **实现诊断 1-5 的 Probe 探针** (修改 single_head.py, head.py, criterion.py)
2. **在 A4 checkpoint 上跑推理诊断** (无需重训练, 仅前向) — `experiments/analysis/structural_diagnosis.py`
3. **收集诊断数据**, 确认瓶颈优先级:
   - ✅ 诊断 1: RoI 空间信息是否被浪费? → **D1 消融验证**: 空间信息至关重要 (Δ=-0.854), DynamicConv 有效提取
   - ✅ 诊断 2: 时间条件化是否有效? → **D2**: 活跃且自适应 (|α|=0.47-0.81), 非瓶颈
   - ✅ 诊断 3: 哪些级联头冗余? → **D3**: 头间有自然分化但不充分, head0 修正最大
   - ✅ 诊断 4: 模型是否已隐式学到尺寸→类别? → **D4**: 已学到强映射 (std=0.589), M2 边际收益低
   - ✅ 诊断 5: self_attn 注意力模式如何? → **D5**: head0 近均匀, head2-3 有聚焦, M3 优先级降低

### Phase 1: M1 形态感知 RoI 编码器 (微调, 2-3 天) — 代码就绪 ✅, 待训练

> **D1 消融指导**: 现有 7×7 + DynamicConv 空间编码已验证有效 (抹平则 mAP 崩溃至 0.009)。
> M1 必须作为**并行增强分支**接入, 不能破坏现有空间通路。

1. ✅ 实现 `MorphologyAwareRoIEncoder` (填充 `shape_attention` hook), 作为**零初始化残差分支**叠加在 RoI 特征上 — `ldmdet/core/morphology_encoder.py`
2. ✅ 单元测试: 零初始化恒等性 (确保初始状态不改变 A4 行为)、方向解耦正确性 — 15 测试全通过
3. ⏳ 从 **A4 checkpoint** 微调 (~30 epoch, lr=1e-5, 零初始化保证快速收敛) — 配置 `m1_morphology_aware_24obj.py` 就绪
4. ⏳ 消融: h_conv only / v_conv only / both
5. ⏳ 重点观察 C 组和 G/Y 组 per-class AP 变化 (G21/Y 尺寸相近需形态区分)

### Phase 2: M2 尺度-类别耦合头 (微调, 2-3 天)

1. 实现 `ScaleClassCoupledHead` (替换 `_predict` 方法)
2. 单元测试: 零初始化恒等性、尺寸嵌入正确性
3. 从 A3 checkpoint 微调
4. 消融: 随机打乱 size_emb (验证尺寸先验贡献)
5. 重点观察 G/Y 组 per-class AP 变化

### Phase 3: M4 级联头角色分化 (微调, 1-2 天)

1. 方案 A (损失权重衰减): 零代码改动, 仅改 criterion 配置
2. 从 A3 checkpoint 微调, 对比等权 vs 衰减
3. 如有效, 尝试方案 B (特征传递) 或方案 C (结构分化)

### Phase 4: M3 重叠感知注意力 (重训练, 3-5 天)

1. 仅在诊断 5 确认 self_attn 注意力发散后实施
2. 实现 `OverlapAwareAttention` (稀疏 IoU 感知)
3. 从 A3 checkpoint 微调
4. 重点观察高重叠图像的 AP 变化

### Phase 5: M1+M2 联合 (重训练, 3 天)

1. 组合 M1 + M2 (两者正交, 可叠加)
2. 从 A3 checkpoint 微调
3. 如有增益, 跑 3 seeds 确认稳定性

---

## 五、与论文叙事的整合

### 5.1 创新点定位

| 方案 | 论文贡献定位 | 染色体叙事 |
|------|-------------|-----------|
| M1 | 形态感知编码: 将 ISCN 形态学标准嵌入 RoI 编码 | 24 类判别依赖臂长比/着丝粒位置, 现有架构在编码早期压平空间结构 |
| M2 | 尺度-类别耦合: 利用染色体尺寸→类别的生物学先验 | A1→Y 尺寸跨度 5×, 尺寸是强类别先验, 当前架构未利用 |
| M3 | 重叠解耦: IoU 感知跨提案注意力 | 97.8% 图像有重叠, 现有架构提案交互与空间处理断层 |
| M4 | 级联角色分化: 粗定位→细分类的生物学对应 | 级联精修对应核型分析的从粗到细流程 |

### 5.2 与三层次研究目标的对应

- **初级目标** (与 SOTA 相当): M1+M2 有望将 mAP 从 0.863 推至 0.87+, 接近/超越 DINO (0.868)
- **进阶目标** (任务特性结合): M1-M4 均深度结合染色体检测特性 (形态学/尺寸先验/重叠/级联精修)
- **最高级目标** (可扩展性): M1 (形态感知) 可扩展到细胞检测 (细胞形态判别); M2 (尺度-类别耦合) 可扩展到任何尺寸-类别有强相关的检测任务 (遥感目标: 舰船尺寸→类型)

### 5.3 与已有方向的关系

- **M1 ⊃ 方向 C1**: M1 是方向 C1 的具体实现 (ShapeAttention 已规划未实现)
- **M2 ⊥ 方向 B**: 方向 B 追求解耦 (减少梯度竞争), M2 追求末端耦合 (利用先验), 两者正交
- **M3 ⊃ 方向 Q1**: 方向 Q1 是训练时遮挡增广 (数据层), M3 是结构层重叠解耦, 可叠加
- **M4 ⊃ 方向 C (step-aware)**: 方向 C 让头感知 solver step, M4 让头有不同角色, 互补

---

## 六、风险与缓解

| 风险 | 缓解 |
|------|------|
| M1 零初始化导致不收敛 | 残差连接 + 零初始化是标准做法 (AdaLN-Zero, ControlNet), 训练初期梯度正常流回 |
| M2 尺寸预测不准导致 cls 误导 | 零初始化 + 渐进学习; 训练初期 size_emb 贡献为 0, 逐步学到有用映射 |
| M3 计算量过大 | 稀疏化 (仅 top-k 高 IoU 邻居), 染色体数据重叠框对仅 2.6% |
| M4 方案 B 破坏梯度隔离 | 方案 A (损失权重) 零风险先行验证, 确认有效后再尝试方案 B |
| 所有方案增益在噪声范围内 | 3 seeds 验证 + per-class AP 分析 (G/Y 组是敏感指标) |
| 微调过拟合 (小数据集) | 冻结 backbone, 仅微调 head; 早停 (patience=30) |
