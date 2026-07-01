# 方向 I: 长尾少样本类别平衡

> 创建时间: 2026-07-01
> 前置: D+B 组合实验证伪 (best=0.743 < baseline=0.744, 见 DIRECTIONS_REVISION.md 2.5.1 节)
> 核心目标: 解决 Y 类 (及少数尾部类) 的 cls_head 预测坍塌问题

---

## 一、问题背景

### 1.1 现象

白盒插桩分析 ([INSTRUMENTATION_RESULTS.md](../../analysis/instrumentation/INSTRUMENTATION_RESULTS.md)) 显示:
- Y 类在 cls_head 输出趋近于 0 (预测崩溃)
- `same_group_too_similar=False`: 瓶颈在 cls_head 决策边界, 不在特征抽取
- 跨数据集一致 (24obj 数据集同样存在), 确认为架构固有缺陷

### 1.2 已尝试方案及失败原因

| 方案 | 改动维度 | 结果 | 失败根因 |
|------|---------|------|---------|
| E (ClassBalancedDataset) | 数据采样 (图像级 oversample) | +0.001 (噪声) | 不改变 softmax 中头类对尾类的负梯度压制 |
| B (DecoupledHead) | head 结构 (cls/reg 双分支) | +0.004 (谦虚副作用) | 未触及 cls_head 内部 Y 类的负梯度压制与权重范数坍塌 |
| D+B (组合) | head 结构 + 框精化残差 | -0.001 (证伪) | 两者都不触及 cls_head 损失函数和分类器参数化 |

### 1.3 根因诊断

文献共识 (BAGS CVPR'20, Seesaw Loss CVPR'21, EQL v2 CVPR'21) 指出, 尾部类预测坍塌有两个根因:

**根因 1: 负梯度压制**
- softmax 交叉熵中, 头类 (如 A1, A2 等高频类) 作为负样本时, 对 Y 类产生强负梯度
- Y 类的正梯度信号 (来自少量正样本) 被头类的负梯度淹没
- 结果: Y 类分类器权重在训练中持续被压低

**根因 2: 分类器权重范数坍塌**
- 由于正样本少, Y 类的 cls_head 权重范数远小于头类
- 即使 Y 类的特征质量良好, softmax 输出仍趋近于 0 (因为 logits = W·x, |W_Y| << |W_head|)
- 这不是特征学习问题, 而是分类器参数化问题

**关键认知**: 之前两次尝试 (E/B) 都未触及这两个根因:
- E 改数据频率 → 不改梯度流向
- B 改 head 结构 → 不改损失函数和分类器参数化

---

## 二、方案设计

基于文献调研, 设计三个子方向, 按对症度排序。三者位于不同维度, 可独立验证后逐步叠加。

### 2.1 I-1: Seesaw Loss + Normalized Classifier (损失+分类器参数化)

**论文**: [Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)](https://arxiv.org/abs/2008.10032)
**来源**: LVIS 2020 竞赛冠军方案

**核心思想**:

1. **Seesaw 惩罚**: 给定当前样本属于类 i, 对负样本类 j 的损失惩罚按累计样本数比值动态衰减:
   ```
   S_ij = (N_j / N_i)^p,  p=0.8
   ```
   - 当 N_j >> N_i (头类 j 对尾类 i): S_ij << 1, 大幅衰减头类对尾类的负梯度
   - 当 N_j ≈ N_i (同类): S_ij ≈ 1, 保持正常惩罚
   - 误分类时补充惩罚 (self-calibrated): 如果尾类被误分为头类, 恢复正常惩罚以学习区分

2. **Normalized Classifier**: 分类器权重 W 和特征 x 都 L2 归一化:
   ```
   logits = τ · (W̃ · x̃) + b,  τ=20
   ```
   - 消除 |W_Y| << |W_head| 的范数偏差
   - 让分类决策基于余弦相似度而非点积绝对值

**对症度**: ★★★★★
- 直接修复根因 1 (Seesaw 衰减负梯度) + 根因 2 (归一化修复权重范数)
- 与 E/B/D 完全正交 (改损失函数 + 分类器参数化)

**实现计划**:
- 替换 `criterion.loss_cls` 从 PurePyTorchFocalLoss → SeesawLoss
- 替换 `single_head.cls_head` 最后一层为 normalized linear (L2 归一化权重 + 特征)
- 维护累计样本计数 buffer (每个 epoch 更新)
- 配置参数: `p=0.8`, `τ=20`

**关键代码改动**:
- `ldmdet/core/criterion.py`: 新增 SeesawLoss 类
- `ldmdet/core/single_head.py`: cls_head 归一化
- `experiments/configs/ldmdet/direction_i_seesaw.py`: 新配置

**预期收益**:
- Y 类 AP 显著提升 (LVIS rare 类 +14-18 AP, 但 24 类场景增幅会小于 LVIS 1203 类)
- 整体 mAP 预期 +0.005~0.010 (保守估计, 因 24 类不平衡程度远低于 LVIS)

**风险**:
- 24 类的不平衡程度远低于 LVIS (1203 类), 收益可能缩水
- 归一化分类器可能影响头类精度 (需调 τ)

---

### 2.2 I-2: Decoupled Representation & Classifier (训练时序解耦)

**论文**: [Decoupling Representation and Classifier for Long-Tailed Recognition (ICLR 2020)](https://arxiv.org/abs/1910.09217)

**核心思想**:

> ⚠️ **关键澄清**: 这与已试过的 DecoupledHead (方向 B) 完全不同。
> - DecoupledHead 解耦的是: cls head 和 reg head 的**结构** (空间维度)
> - Kang 解耦的是: 表征学习和分类器学习的**时序** (时间维度)

**两阶段训练**:
- **阶段 1 (表征学习)**: 用实例平衡采样 (正常采样) 训练全模型 150 epochs。数据不平衡对学高质量表征不是问题
- **阶段 2 (分类器重训练)**: 冻结 backbone + neck + head 的特征提取部分, 只重训 cls_head:
  - 方式 A (cRT): 用 class-balanced 采样重训 cls_head 几个 epoch
  - 方式 B (τ-norm): 对 cls_head 权重按各类样本数做 τ 归一化 (无需重训)
  - 方式 C (NCM): 用各类特征均值作为分类中心 (无需训练)

**对症度**: ★★★★
- 修复根因 2 (分类器权重范数): 阶段 2 专门校正分类器偏差
- 部分修复根因 1: 阶段 2 的 class-balanced 采样缓解正样本不足
- 不改损失函数, 与 I-1 正交可叠加

**实现计划**:
- 阶段 1: 复用现有 baseline 训练 (无需改动)
- 阶段 2: 加载 baseline ckpt, 冻结 `backbone` + `neck` + `head_series[*].{inst_interact, self_attn, linear1, linear2, ...}`, 只保留 `cls_head` 可训练
- 用 ClassBalancedDataset 采样训练 cls_head 5-10 epochs
- 配置: `direction_i_decoupled_cls.py`

**预期收益**:
- 单独: +0.003~0.005 (Kang 论文中 cRT 在 ImageNet-LT 提升 5-6%)
- 叠加 I-1: 阶段 2 用 Seesaw Loss 重训 cls_head, 预期叠加增益

**风险**:
- 扩散检测器的 cls_head 与分类任务不同 (RoI 特征上的分类), 需验证冻结后特征质量
- 阶段 2 的 class-balanced 采样仍可能过拟合 Y 类 (样本太少)

---

### 2.3 I-3: Copy-Paste / 扩散生成 Y 类样本 (数据根因修复)

**论文**:
- [Simple Copy-Paste (CVPR 2021)](https://arxiv.org/abs/2012.07177)
- [X-Paste (ICML 2023)](https://arxiv.org/abs/2210.11335) — LVIS 长尾类 +6.8 box AP
- [DiffuLT (NeurIPS 2024)](https://openreview.net/forum?id=Kcsj9FGnKR) — 仅用数据集自身训练扩散模型生成尾类样本
- [AD-Det DCC (2025)](https://arxiv.org/abs/2504.05601) — 动态类平衡 Copy-Paste

**核心思想**:

1. **Copy-Paste (简单版)**: 收集训练集中所有 Y 类实例 (含 bbox), 随机粘贴到其他训练图中
   - 实例级组合 (非图像级重复), 多样性远高于 oversample
   - 需要实例 mask (或用 bbox 裁剪)

2. **扩散生成 (进阶版)**: 复用项目的扩散 backbone, 生成 Y 类合成样本
   - DiffuLT: 仅用长尾数据集自身训练扩散模型, 生成"近似分布内 (AID)"样本
   - 项目已有扩散模型, 几乎零额外架构成本
   - 生成样本后用 Copy-Paste 粘贴到训练图

**对症度**: ★★★★
- 从数据根因入手, 直接增加 Y 类正样本数量
- 与 I-1/I-2 完全正交, 可叠加

**实现计划**:
- 收集 Y 类实例库 (从训练集提取 bbox + 特征)
- 实现 Copy-Paste 增强 (在 DataLoader 的 transform 中)
- 可选: 用扩散模型生成 Y 类合成实例

**预期收益**:
- Copy-Paste: +0.003~0.008 (LVIS rare 类 +3.6 mask AP)
- 扩散生成: +0.005~0.010 (X-Paste +6.8 box AP)

**风险**:
- Copy-Paste 需要 mask (染色体项目可能只有 bbox, 需用 bbox 裁剪替代)
- 扩散生成质量不确定, 可能引入噪声样本
- 实现复杂度较高

---

## 三、实施路径

### 3.1 分阶段验证

```
阶段 1: I-1 (Seesaw + NormCls)        ← 最对症, 优先
    ↓ 独立验证
阶段 2: I-2 (Decoupled 两阶段训练)      ← 叠加, 阶段 2 用 Seesaw
    ↓ 叠加验证
阶段 3: I-3 (Copy-Paste / 扩散生成)    ← 若 Y 类样本 <10 才需要
```

### 3.2 控制变量

每个子方向独立验证, 与 baseline (rf_heun_adaln, seed=42) 对比:
- 训练: seed=42, 150 epochs, EarlyStopping patience=30
- 推理: test.py, seed=42, Heun 4步 (与 2.4 节 baseline 对齐)
- 指标: mAP, mAP_50, mAP_75, mAP_s, per-class AP (关注 Y 类 AP 变化)

### 3.3 决策阈值

- mAP +0.005 以上: 成功, 进入下一阶段叠加
- mAP ±0.003: 噪声范围, 需多 seed 验证
- mAP -0.005 以下: 失败, 停止

---

## 四、与已有方向的关系

### 4.1 失败方向回顾

| 方向 | 改动维度 | 是否触及根因 |
|------|---------|:---:|
| E (ClassBalanced) | 数据采样 (图像级) | ❌ 不改梯度/权重 |
| B (DecoupledHead) | head 结构 (cls/reg) | ❌ 不改损失/分类器 |
| D+B (组合) | head 结构 + 框精化 | ❌ 不改损失/分类器 |
| **I-1 (Seesaw)** | **损失 + 分类器参数化** | ✅✅ |
| **I-2 (Decoupled Repr)** | **训练时序** | ✅ |
| **I-3 (Copy-Paste)** | **数据 (实例级)** | ✅ |

### 4.2 与 E (ClassBalanced) 的关键区别

E 失败是因为图像级 oversample 不改变 softmax 梯度流向。I 方向从三个不同维度切入:
- I-1: 梯度级 (Seesaw 衰减) + 分类器级 (归一化)
- I-2: 时序级 (先表征后分类器)
- I-3: 实例级 (Copy-Paste vs 图像级重复)

### 4.3 与 H (采样效率) 的关系

H 已成功 (Euler/DPM++ 4步 = 0.745), 但仅优化推理效率, 不改训练。I 方向改训练, 与 H 完全正交, 可叠加使用。

---

## 五、论文参考

### 5.1 损失函数级别

| 方法 | 年份 | 核心创新 | 链接 |
|------|------|---------|------|
| EQL v1 | CVPR 2019 | 忽略头类对尾类的负梯度 | https://arxiv.org/abs/1903.05147 |
| EQL v2 | CVPR 2021 | 平衡正负梯度比例 | https://arxiv.org/abs/2012.08548 |
| **Seesaw Loss** | **CVPR 2021** | **动态衰减 + 归一化分类器** | https://arxiv.org/abs/2008.10032 |
| LDAM + DRW | NeurIPS 2019 | 尾类大 margin + 延迟重加权 | https://arxiv.org/abs/1906.07413 |
| BAGS | CVPR 2020 | 分组 softmax 消除头尾竞争 | https://arxiv.org/abs/2006.10408 |
| IGAM Loss | ICLR 2025 | 类别信息量动态 margin | https://arxiv.org/abs/2502.03852 |

### 5.2 表征学习级别

| 方法 | 年份 | 核心创新 | 链接 |
|------|------|---------|------|
| **Decoupled Repr** | **ICLR 2020** | **冻结表征, 重训分类器** | https://arxiv.org/abs/1910.09217 |
| OLTR | CVPR 2019 | 记忆增强元嵌入 | https://liuziwei7.github.io/projects/LongTail.html |
| ProCo | TPAMI 2024 | vMF 分布在线对比学习 | https://github.com/LeapLabTHU/ProCo |

### 5.3 数据增强级别

| 方法 | 年份 | 核心创新 | 链接 |
|------|------|---------|------|
| Copy-Paste | CVPR 2021 | 实例级粘贴增强 | https://arxiv.org/abs/2012.07177 |
| X-Paste | ICML 2023 | CLIP+SD 生成实例粘贴 | https://arxiv.org/abs/2210.11335 |
| **DiffuLT** | **NeurIPS 2024** | **数据集自身训练扩散模型生成尾类** | https://openreview.net/forum?id=Kcsj9FGnKR |
| AD-Det DCC | 2025 | 动态类平衡 Copy-Paste | https://arxiv.org/abs/2504.05601 |

### 5.4 扩散检测器相关

| 方法 | 年份 | 核心创新 | 链接 |
|------|------|---------|------|
| RaTrack | Electronics 2024 | 扩散模型对齐尾类生成 | https://www.mdpi.com/2079-9292/13/23/4693 |
| DiffuLT | NeurIPS 2024 | 长尾数据集自身扩散生成 | https://openreview.net/forum?id=Kcsj9FGnKR |

---

## 六、风险评估

### 6.1 主要风险

1. **24 类不平衡程度低于 LVIS**: 染色体 24 类的不平衡程度远低于 LVIS (1203 类), 收益可能缩水
   - 缓解: 关注 per-class AP 而非仅 mAP, Y 类 AP 提升即使整体 mAP 变化小也有价值

2. **归一化分类器影响头类**: Normalized Classifier 可能让头类精度下降
   - 缓解: 调 τ 参数, 可能需要 τ=10~30 的网格搜索

3. **扩散生成质量**: 生成的 Y 类样本可能不符合真实分布
   - 缓解: 质量过滤策略 (FID, IS), 或仅用 Copy-Paste (不生成)

4. **阶段 2 冻结后特征退化**: 扩散检测器的特征质量依赖端到端训练
   - 缓解: 阶段 2 只冻结 backbone+neck, 不冻结 head 的特征提取部分

### 6.2 备选方案

若 I-1/I-2/I-3 均未达预期:
- **BAGS (分组 softmax)**: 将 24 类按样本数分 3-4 桶, Y 单独成桶, 消除与头类竞争
- **EQL v2**: 正负梯度比例平衡, 比 Seesaw 更直接但实现更复杂
- **LDAM + DRW**: 尾类大 margin + 延迟重加权, 实现最简

---

## 七、总结

方向 I 是基于 D+B 失败后的根因分析, 转向直接修复 cls_head 的损失函数和分类器参数化。与之前所有方向 (D/B/E/F/G/H) 的改动维度不同, 是唯一直接对症"负梯度压制 + 权重范数坍塌"两个根因的方案。

**优先级**: I-1 (Seesaw + NormCls) > I-2 (Decoupled 两阶段) > I-3 (Copy-Paste/扩散生成)
**预期**: I-1 单独 +0.005~0.010, 三者叠加上限 +0.015~0.020
