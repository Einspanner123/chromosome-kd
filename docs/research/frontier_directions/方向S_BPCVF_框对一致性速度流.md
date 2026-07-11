# 方向 S：BPCVF — Box Pair Consistency Velocity Flow (框对一致性速度流)

> **状态：方案设计阶段 (未实验)**
>
> **基线**：a3_full_sota, mAP=0.858 (RF + Heun 4步 + AdaLN-Zero + StochasticOT ε=5 + box_renewal)
>
> **核心思想**：利用染色体核型中的同源染色体配对先验，在扩散速度场中引入配对一致性正则化。模型仍预测 x_0 (保持级联 Head 架构不变)，但通过推导速度场施加同源对一致性约束，使速度场在配对维度上更平滑，减少配对染色体的预测方差。
>
> **关键区分**：BPCVF **不直接预测速度** (区别于已证伪的 CFM/方向 H)，而是**约束推导速度场**。这是本方向可行性的核心依据，详见 §4。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [理论依据](#2-理论依据)
3. [详细方案设计](#3-详细方案设计)
4. [深入可行性分析 — 与 CFM 证伪的关系](#4-深入可行性分析--与-cfm-证伪的关系)
5. [与 KCEC 失败的对比](#5-与-kcec-失败的对比)
6. [风险评估](#6-风险评估)
7. [预期收益分析](#7-预期收益分析)
8. [实现路线图](#8-实现路线图)
9. [与已证伪方向的系统对比](#9-与已证伪方向的系统对比)
10. [参考文献](#10-参考文献)

---

## 1. 背景与动机

### 1.1 核型配对先验

染色体检测的本质是核型分析 (Karyotyping)。根据 ISCN 国际标准，人类核型由 46 条染色体组成，构成 23 对同源染色体：

```
核型结构 (24obj 数据集, 24 类):
  A组: A1×2, A2×2, A3×2        — 6条, 大型, 中着丝粒
  B组: B4×2, B5×2              — 4条, 大型, 亚中着丝粒
  C组: C6-C12×2 (各2条)        — 14条, 中型, 中着丝粒
  D组: D13×2, D14×2, D15×2     — 6条, 中型, 近端着丝粒
  E组: E16×2, E17×2, E18×2     — 6条, 小型, 中/亚中着丝粒
  F组: F19×2, F20×2            — 4条, 短型, 中着丝粒
  G组: G21×2, G22×2            — 4条, 短型, 近端着丝粒
  性染色体: X (×1或2), Y (×0或1)
```

**关键先验**：
- 同源染色体对 (如两条 A1) 在大小、形态、带型上高度相似
- 核型分析的临床操作本身就是"配对"——将形态最相似的同类染色体配对排列
- 每张图像中，同类染色体恰好出现 2 条 (非性染色体)，构成天然的配对监督信号

### 1.2 当前模型的盲区

当前 a3_full_sota 基线将 46 条染色体作为**独立目标**检测：

- **架构** (`head.py`): 6 级级联 Head，每级对 500 个 proposals 独立做 delta 回归 (`single_head.py: _predict_bboxes → apply_deltas`)，框间无显式关系建模
- **损失** (`criterion.py`): L1(5.0) + GIoU(2.0) + Focal(2.0)，每个 proposal 独立计算损失，无配对约束
- **耦合** (`ot_flow_coupling.py`): StochasticOT ε=5 将 noise proposals 与 GT 配对，但耦合策略只影响 (x_t, target) 配对，不传播到表征层 (KCEC 已证实)

**问题**：同源染色体对的预测是独立的，无法利用"两条 A1 应大小相似"这一强先验。当其中一条预测偏差较大时 (异常框)，模型无法通过配对关系进行修正。

### 1.3 速度场结构化的动机

当前 RF 路径中，采样器使用推导速度场进行去噪：

```python
# rectified_flow.py:42-45 — 推导速度
def get_velocity(self, x_t, x_0_pred, t):
    return (x_t - x_0_pred) / torch.clamp(t, min=1e-5)

# rectified_flow.py:55-61 — Euler 采样步
def step(self, x_t, x_0_pred, t_curr, t_next, velocity=None):
    dt = t_next - t_curr
    return x_t + dt * velocity
```

速度场 `v(x_t, t) = (x_t - x_0_pred) / t` 是模型预测的间接产物。对于同源染色体对，由于 GT 大小/形态相似 (x_start 相似)，其**理想速度场在配对维度上应具有一致性结构**。当前模型没有显式约束这一结构，速度场在配对维度上是"自由"的。

**BPCVF 的动机**：将核型配对先验注入速度场，通过 PairWise 一致性损失显式约束同源对的推导速度，使速度场在配对维度上更平滑，从而：
1. 减少配对染色体的预测方差 (两条 A1 的预测更一致)
2. 提供异常框的纠正信号 (偏离配对的预测被惩罚)
3. 不改变级联 Head 的预测目标 (仍预测 x_0)，保持架构兼容性

---

## 2. 理论依据

### 2.1 RF 速度场理论

Rectified Flow 定义直线路径前向扩散 (`rectified_flow.py:37-40`)：

$$x_t = (1-t) \, x_0 + t \, x_1, \quad t \in [0, 1]$$

其中 $x_0$ 是数据 (GT box, 扩散空间)，$x_1$ 是噪声。对应的速度场 (velocity) 为：

$$v = x_1 - x_0 = \frac{x_t - x_0}{t}$$

采样时，模型预测 $\hat{x}_0$，推导速度 $\hat{v} = (x_t - \hat{x}_0) / t$，通过 ODE 积分 (Euler/Heun/DPM-Solver++) 从 $t=1$ (噪声) 演化到 $t=0$ (数据)。

### 2.2 同源对速度一致性的数学推导

**设定**：考虑一对同源染色体 $(i, j)$ (如两条 A1)，其在扩散空间的 GT 表示为 $x_0^{(i)}$ 和 $x_0^{(j)}$。

**前提 1 (GT 相似性)**：同源染色体大小/形态相似，在扩散空间中：

$$\|x_0^{(i)} - x_0^{(j)}\| \ll \|x_0^{(i)}\|$$

即 $x_0^{(i)} \approx x_0^{(j)}$ (相对量级)。

**前提 2 (噪声独立性)**：在当前 RF 实现中 (`rectified_flow.py:32-33`)，每个 proposal 的噪声 $x_1$ 独立采样：

$$x_1^{(i)} \sim \mathcal{N}(0, I), \quad x_1^{(j)} \sim \mathcal{N}(0, I), \quad \text{独立}$$

**目标速度**：

$$v^{(i)} = x_1^{(i)} - x_0^{(i)}, \quad v^{(j)} = x_1^{(j)} - x_0^{(j)}$$

目标速度差：

$$v^{(i)} - v^{(j)} = (x_1^{(i)} - x_1^{(j)}) - (x_0^{(i)} - x_0^{(j)})$$

由前提 1，$x_0^{(i)} - x_0^{(j)} \approx 0$，故：

$$v^{(i)} - v^{(j)} \approx x_1^{(i)} - x_1^{(j)} \sim \mathcal{N}(0, 2I)$$

**关键发现 1**：目标速度差 $\|v^{(i)} - v^{(j)}\|^2 \approx 2d$ ($d=4$ 维，期望值 $\approx 8$)，**并非小量**。直接约束 $\|v^{(i)} - v^{(j)}\|^2 \to 0$ 在目标层面是错误的——它等于要求 $x_1^{(i)} \approx x_1^{(j)}$，即要求独立噪声相同，这是不可能的。

### 2.3 正确的约束：残差速度一致性

模型预测 $\hat{x}_0^{(i)}$ 和 $\hat{x}_0^{(j)}$，推导速度：

$$\hat{v}^{(i)} = \frac{x_t^{(i)} - \hat{x}_0^{(i)}}{t}, \quad \hat{v}^{(j)} = \frac{x_t^{(j)} - \hat{x}_0^{(j)}}{t}$$

定义**速度残差** (预测速度与目标速度之差)：

$$\Delta v^{(i)} = \hat{v}^{(i)} - v^{(i)} = \frac{x_t^{(i)} - \hat{x}_0^{(i)}}{t} - \frac{x_t^{(i)} - x_0^{(i)}}{t} = \frac{x_0^{(i)} - \hat{x}_0^{(i)}}{t}$$

同理：

$$\Delta v^{(j)} = \frac{x_0^{(j)} - \hat{x}_0^{(j)}}{t}$$

**残差速度一致性约束**：

$$\mathcal{L}_{\text{pair}} = \|\Delta v^{(i)} - \Delta v^{(j)}\|^2 = \frac{1}{t^2} \left\| (x_0^{(i)} - \hat{x}_0^{(i)}) - (x_0^{(j)} - \hat{x}_0^{(j)}) \right\|^2$$

展开：

$$\mathcal{L}_{\text{pair}} = \frac{1}{t^2} \left\| (x_0^{(i)} - x_0^{(j)}) - (\hat{x}_0^{(i)} - \hat{x}_0^{(j)}) \right\|^2$$

由前提 1 ($x_0^{(i)} \approx x_0^{(j)}$)：

$$\mathcal{L}_{\text{pair}} \approx \frac{1}{t^2} \left\| \hat{x}_0^{(i)} - \hat{x}_0^{(j)} \right\|^2$$

**关键发现 2**：残差速度一致性约束在数学上等价于 (在 $x_0^{(i)} \approx x_0^{(j)}$ 近似下)：

$$\boxed{\mathcal{L}_{\text{pair}} \approx \frac{1}{t^2} \|\hat{x}_0^{(i)} - \hat{x}_0^{(j)}\|^2}$$

这是一个**带时间加权 $\frac{1}{t^2}$ 的预测一致性损失**，要求模型对同源对的预测 $\hat{x}_0$ 保持一致。时间加权 $\frac{1}{t^2}$ 的物理含义：
- $t \to 0$ (靠近数据): 权重 $\to \infty$，强约束预测精度——此时速度场对 $\hat{x}_0$ 极其敏感
- $t \to 1$ (靠近噪声): 权重 $\to 1$，弱约束——此时预测本身不确定，一致性要求放松

### 2.4 与 StochasticOT 耦合的协同

当前 SOTA 使用 StochasticOT ε=5 (`ot_flow_coupling.py`)，其核心机制是：

```python
# ot_flow_coupling.py:103-110 — Stochastic 解码
if coupling_mode == 'multinomial':
    matched_idx = torch.multinomial(transport_col.t().clamp_min(1e-10), 1)
```

StochasticOT 通过从 Sinkhorn 传输矩阵中**随机采样** (而非 argmax)，恢复了 ε-依赖的多样性单调性 (定理 4.3)。其增益来源是训练信号多样性，而非配对结构。

BPCVF 与 StochasticOT 正交且互补：
- **StochasticOT** 约束 (noise, GT) 配对的多样性 → 影响训练信号分布
- **BPCVF** 约束 (prediction_i, prediction_j) 的一致性 → 影响输出空间结构
- 两者作用在不同环节：耦合在 `_build_training_targets` (数据准备)，BPCVF 在 `loss` (损失计算)

**协同效应**：StochasticOT 保证了同源对的 GT 分配具有多样性 (不会总是将同一 GT 分配给所有 proposals)，而 BPCVF 在此基础上约束预测的一致性。多样性保证模型见过多种配对模式，一致性保证输出空间的结构化。

---

## 3. 详细方案设计

### 3.1 配对识别算法

#### 3.1.1 训练时配对 (有 GT 标签)

训练时，GT 标签直接提供同源配对信息。利用 `_build_training_targets` 中已计算的 `matched_gt_indices` (`head.py:443-446`)：

```python
def _identify_homologous_pairs(self, matched_gt_indices, gt_labels, fg_mask):
    """识别同源染色体对。

    Args:
        matched_gt_indices: [N] 每个 proposal 匹配的 GT 索引
        gt_labels: [M] GT 类别标签
        fg_mask: [N] 正样本 mask

    Returns:
        pairs: List[(int, int)] 同源 proposal 对的索引
    """
    pairs = []
    # 按类别分组 proposal 索引
    class_to_props = {}
    for prop_idx in fg_mask.nonzero(as_tuple=True)[0]:
        gt_idx = matched_gt_indices[prop_idx]
        cls = gt_labels[gt_idx].item()
        class_to_props.setdefault(cls, []).append(prop_idx.item())

    # 每个类别内，将 proposals 两两配对
    for cls, prop_indices in class_to_props.items():
        if len(prop_indices) < 2:
            continue
        # 简单策略: 按序两两配对 (proposal 0-1, 2-3, ...)
        # 更优策略: 按 x_t 相似度配对 (见下文)
        for k in range(0, len(prop_indices) - 1, 2):
            pairs.append((prop_indices[k], prop_indices[k + 1]))

    return pairs
```

**配对策略选择**：

| 策略 | 描述 | 复杂度 | 适用性 |
|------|------|--------|--------|
| 按序配对 | 同类 proposals 按索引顺序两两配对 | O(N) | 简单，但可能配错 |
| x_t 相似度配对 | 同类 proposals 按 x_t 距离做匈牙利匹配 | O(N³) | 最优，但计算贵 |
| GT 索引配对 | 匹配到不同 GT 实例的同类 proposals 配对 | O(N) | 推荐，天然对应同源对 |

**推荐策略 (GT 索引配对)**：同一类别的两个不同 GT 实例 (如 GT#0 和 GT#1 都是 A1) 天然构成同源对。将匹配到 GT#0 的 proposals 和匹配到 GT#1 的 proposals 进行配对：

```python
def _gt_index_pairing(self, matched_gt_indices, gt_labels, fg_mask):
    """基于 GT 索引的同源配对——同一类别不同 GT 实例的 proposals 配对。"""
    # 收集每个 (class, gt_idx) 的 proposal 列表
    gt_to_props = {}
    for prop_idx in fg_mask.nonzero(as_tuple=True)[0]:
        gt_idx = matched_gt_indices[prop_idx].item()
        gt_to_props.setdefault(gt_idx, []).append(prop_idx.item())

    # 找同类不同 GT 实例对
    gt_classes = {gt_idx: gt_labels[gt_idx].item() for gt_idx in gt_to_props}
    pairs = []
    gt_indices = list(gt_to_props.keys())
    for i in range(len(gt_indices)):
        for j in range(i + 1, len(gt_indices)):
            gi, gj = gt_indices[i], gt_indices[j]
            if gt_classes[gi] == gt_classes[gj]:  # 同类 = 同源
                # 从两组中各取一个配对 (取置信度最高的或最近的)
                # 简化: 各取第一个
                pairs.append((gt_to_props[gi][0], gt_to_props[gj][0]))
    return pairs
```

#### 3.1.2 推理时配对 (无 GT 标签)

推理时无 GT 标签，需基于预测结果进行配对：

```python
def _infer_homologous_pairs(self, pred_boxes, pred_scores, pred_labels):
    """推理时基于形态相似性的同源配对。

    Args:
        pred_boxes: [N, 4] xyxy (NMS 后)
        pred_scores: [N] 置信度
        pred_labels: [N] 预测类别

    Returns:
        pairs: List[(int, int)] 配对索引
    """
    pairs = []
    for cls in pred_labels.unique():
        cls_mask = pred_labels == cls
        cls_indices = cls_mask.nonzero(as_tuple=True)[0]
        if len(cls_indices) < 2:
            continue

        # 取 top-2 作为同源对候选
        cls_scores = pred_scores[cls_indices]
        top2 = cls_scores.topk(2).indices
        pairs.append((cls_indices[top2[0]].item(),
                      cls_indices[top2[1]].item()))
    return pairs
```

### 3.2 速度一致性损失函数

基于 §2.3 的推导，BPCVF 损失作用于**推导速度的残差一致性**。实现位置在 `head.py: loss()` 方法中，在 criterion 计算之后追加：

```python
def _bpcvf_loss(
    self,
    all_pred_bboxes,    # [num_heads, bs, N, 4] xyxy 归一化
    targets,             # List[InstanceData]
    matched_gt_indices,  # List[Tensor] 每个 image 的 GT 匹配索引
    t,                   # [bs] 扩散时间
    img_metas,
):
    """BPCVF: 同源对残差速度一致性损失。

    数学等价 (§2.3): L_pair ≈ (1/t²) ||x0_pred_i - x0_pred_j||²
    """
    bs = len(targets)
    total_loss = 0.0
    num_pairs = 0

    for h in range(all_pred_bboxes.shape[0]):  # 遍历级联 Head
        for i in range(bs):
            # 获取当前 image 当前 head 的预测
            pred_boxes_h = all_pred_bboxes[h, i]  # [N, 4] 归一化 xyxy

            # 获取 matcher 的匹配结果 (正样本 mask + GT 索引)
            # 从 criterion 缓存中读取
            fg_mask = self._last_fg_masks[i]      # [N] bool
            matched_idx = matched_gt_indices[i]    # [N] long

            # 识别同源对 (§3.1.1)
            pairs = self._gt_index_pairing(
                matched_idx, targets[i].labels, fg_mask
            )

            if len(pairs) == 0:
                continue

            # 转为 cxcywh 计算一致性 (尺度不变)
            pred_cxcywh = bbox_xyxy_to_cxcywh(pred_boxes_h)  # [N, 4]

            for (pi, pj) in pairs:
                # 残差速度一致性 ≈ (1/t²) ||x0_pred_i - x0_pred_j||²
                # 注意: 这里用 x0_pred 的 cxcywh 差异, 已含 1/t² 加权
                diff = pred_cxcywh[pi] - pred_cxcywh[pj]  # [4]
                total_loss = total_loss + (diff ** 2).sum()
                num_pairs += 1

    if num_pairs == 0:
        return all_pred_bboxes.sum() * 0.0

    # 时间加权: 1/t² (§2.3), clamp 防止 t→0 时爆炸
    t_weight = (1.0 / (t.clamp(min=0.1) ** 2)).mean()  # 标量
    return self.bpcvf_weight * t_weight * total_loss / num_pairs
```

**损失项配置**：

```python
# 在 head.py __init__ 中新增参数
bpcvf_weight: float = 0.5,       # BPCVF 损失权重
bpcvf_t_clamp: float = 0.1,      # t 下限 clamp (防 1/t² 爆炸)
bpcvf_mode: str = 'residual',    # 'residual' (推荐) 或 'naive' (消融)
```

**总损失**：

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{L1}} + \mathcal{L}_{\text{GIoU}}}_{\text{现有损失 (criterion.py)}} + \lambda_{\text{bpcvf}} \cdot \mathcal{L}_{\text{pair}}$$

其中 $\lambda_{\text{bpcvf}}$ 控制配对一致性的强度。建议初始值 0.5 (介于 L1=5.0 和 GIoU=2.0 之间的一半量级)，并通过消融实验调优。

### 3.3 推理时配对约束 (后处理)

推理时，在 NMS 后追加配对一致性后处理：

```python
def _pair_consistency_postprocess(self, results, img_metas):
    """配对一致性后处理: 修正偏离配对的异常框。

    策略: 对同源对中置信度较低的框, 用高置信度框的尺寸做 soft 修正。
    """
    for i, result in enumerate(results):
        if len(result.bboxes) < 2:
            continue

        pairs = self._infer_homologous_pairs(
            result.bboxes, result.scores, result.labels
        )

        for (pi, pj) in pairs:
            si, sj = result.scores[pi], result.scores[pj]
            # 低置信度框向高置信度框做尺寸 soft 对齐
            if si > sj:
                strong, weak = pi, pj
            else:
                strong, weak = pj, pi

            box_strong = bbox_xyxy_to_cxcywh(result.bboxes[strong])
            box_weak = bbox_xyxy_to_cxcywh(result.bboxes[weak])

            # 只修正尺寸 (w, h), 不修正位置 (cx, cy)
            # 尺寸做加权平均, 权重 = 置信度比
            w_ratio = sj / (si + sj + 1e-8)
            box_weak[2:] = (1 - w_ratio) * box_weak[2:] + w_ratio * box_strong[2:]

            # 写回
            result.bboxes[weak] = bbox_cxcywh_to_xyxy(box_weak.unsqueeze(0)).squeeze(0)

    return results
```

**注意**：推理时配对约束是**可选的**后处理，不改变模型本身。Phase 1 先验证训练时损失的效果，Phase 3 再加推理后处理。

---

## 4. 深入可行性分析 — 与 CFM 证伪的关系

### 4.1 CFM 证伪回顾

**CFM (Conditional Flow Matching, 方向 H)** 的核心改动是让模型**直接预测速度** $v_\theta$ 而非 $x_0$：

```python
# single_head.py:305-312 — CFM 路径 (predict_velocity=True)
def _predict(self, fc_feature, bboxes, bs, num_boxes):
    class_logits = self.cls_head(fc_feature)
    velocity = self.reg_head(fc_feature)  # ← 直接输出速度, 不经 apply_deltas
    return class_logits, velocity, ...

# head.py:199-210 — CFM 级联转换
if self.use_cfm:
    # x0 = x_t - t * v_pred (从速度反推 x0)
    x0 = self._x_raw - t_view * pred_bboxes  # pred_bboxes 此时是 velocity
    x0 = torch.clamp(x0, min=-10.0, max=10.0)
    next_bboxes = self._sampler.raw_to_xyxy(x0, img_metas)
```

**CFM 失败原因 (mAP=0.823, -0.035 vs 基线 0.858)**：

1. **级联 Head 1-5 输入 ≈ x_0，丢失 x_noise 信息**：CFM 路径中，每级 Head 预测 velocity 后立即反推 $x_0 = x_t - t \cdot v$，并将 $x_0$ (而非 $x_t$) 作为下一级 Head 的输入 (`head.py:213`)。由于级联 Head 的逐级精化，Head 1 之后的输入已接近 $x_0$，**扩散时间 $t$ 的条件化信息被丢失**——模型看到的不再是"加噪到时刻 $t$ 的框"，而是"已去噪的干净框"，导致后续 Head 的速度预测失去意义。

2. **delta 回归被绕过**：非 CFM 路径中，`_predict_bboxes` 调用 `apply_deltas` (`single_head.py:389-394`)，以当前框为锚点做 delta 回归。CFM 路径中，velocity 直接从 `reg_head` 输出 (`single_head.py:307`)，**不使用当前框作为锚点**，级联精化的归纳偏置被破坏。

3. **velocity MSE 损失与 L1+GIoU 损失的梯度冲突**：CFM 同时计算 `loss_v = F.mse_loss(all_pred_output, v_target)` (`head.py:274-277`) 和从速度反推 x0 的 L1+GIoU 损失。两个损失对 `reg_head` 的梯度方向不一致 (MSE 约束速度精度，L1+GIoU 约束框精度)，导致训练不稳定。

### 4.2 BPCVF 与 CFM 的本质区别

**核心区分**：BPCVF **不改变预测目标**，模型仍预测 $x_0$ (经 `apply_deltas` delta 回归)，只在损失中追加配对一致性约束。

| 维度 | CFM (已证伪) | BPCVF (本方案) |
|------|-------------|----------------|
| **预测目标** | 速度 $v_\theta$ (改 `reg_head` 输出语义) | $x_0$ (不变, 仍走 `apply_deltas`) |
| **级联输入** | $x_0 = x_t - t \cdot v$ (反推, 丢 $x_t$ 信息) | $x_0$ (delta 回归结果, 不反推) |
| **级联锚点** | 无 (velocity 不用 box anchor) | 有 (`apply_deltas` 以当前框为锚) |
| **速度来源** | 模型直接输出 (`reg_head`) | 从 $x_0$ 推导: $v = (x_t - \hat{x}_0)/t$ |
| **损失形式** | MSE($v_\theta$, $v_{\text{target}}$) (逐 proposal) | $\frac{1}{t^2}\|\hat{x}_0^{(i)} - \hat{x}_0^{(j)}\|^2$ (PairWise) |
| **梯度冲突** | MSE vs L1+GIoU (对 `reg_head` 冲突) | 一致性 vs L1+GIoU (对 `reg_head` 协同) |
| **信息流** | 速度→x0→级联 (信息截断) | x0→级联 (信息保持), 速度仅用于损失 |

**梯度冲突分析 (关键)**：

CFM 的梯度冲突根因：`reg_head` 同时被 `loss_v` (MSE on velocity) 和 `loss_bbox` (L1 on x0, 其中 x0 = x_t - t*v) 约束。对 `reg_head` 参数 $\theta$：

$$\frac{\partial \mathcal{L}_v}{\partial \theta} = 2(v_\theta - v_{\text{target}}) \cdot \frac{\partial v_\theta}{\partial \theta}$$

$$\frac{\partial \mathcal{L}_{\text{bbox}}}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{L1}}}{\partial \hat{x}_0} \cdot \frac{\partial \hat{x}_0}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{L1}}}{\partial \hat{x}_0} \cdot (-t) \cdot \frac{\partial v_\theta}{\partial \theta}$$

两个梯度方向相反 ($v_\theta$ 增大 → $\mathcal{L}_v$ 增大但 $\hat{x}_0$ 减小 → $\mathcal{L}_{\text{L1}}$ 变化方向取决于 $\hat{x}_0$ vs $x_0$)，产生冲突。

BPCVF 的梯度结构：`reg_head` 输出仍是 $x_0$ 的 delta，一致性损失 $\mathcal{L}_{\text{pair}} \approx \frac{1}{t^2}\|\hat{x}_0^{(i)} - \hat{x}_0^{(j)}\|^2$ 的梯度：

$$\frac{\partial \mathcal{L}_{\text{pair}}}{\partial \theta} = \frac{2}{t^2}(\hat{x}_0^{(i)} - \hat{x}_0^{(j)}) \cdot \left(\frac{\partial \hat{x}_0^{(i)}}{\partial \theta} - \frac{\partial \hat{x}_0^{(j)}}{\partial \theta}\right)$$

而 L1 损失的梯度 $\frac{\partial \mathcal{L}_{\text{L1}}}{\partial \theta} \propto (\hat{x}_0 - x_0) \cdot \frac{\partial \hat{x}_0}{\partial \theta}$。

**协同性**：当 $\hat{x}_0^{(i)}$ 偏大而 $\hat{x}_0^{(j)}$ 偏小时 (同源对预测发散)，L1 损失分别将两者拉向各自的 GT (方向可能相反)，而一致性损失将两者拉近 (方向一致)。如果两个 GT 相似 ($x_0^{(i)} \approx x_0^{(j)}$)，则 L1 和一致性损失方向**协同**——都让 $\hat{x}_0^{(i)}$ 和 $\hat{x}_0^{(j)}$ 趋向相似的值。这是 BPCVF 梯度不冲突的理论依据。

### 4.3 为什么 BPCVF 约束速度场而非直接预测速度

**根本原因**：在级联 Head 架构中，$x_0$ 预测 (经 delta 回归) 是**架构的归纳偏置**，不能被绕过。

1. **delta 回归的锚点效应**：`apply_deltas` (`single_head.py:396-422`) 以当前框为锚点，通过 learned deltas 做尺寸/位置修正。这利用了级联精化的先验——每级 Head 在上一级的基础上微调。CFM 的 velocity 预测绕过了这一机制，导致级联失效。

2. **x_t 信息的保持**：非 CFM 路径中，级联 Head 的输入是 $\hat{x}_0$ (上一级的预测)，这仍然携带了 $x_t$ 的信息 (因为 $\hat{x}_0$ 是从 $x_t$ 预测的)。CFM 路径中，$x_0 = x_t - t \cdot v$ 是显式反推，$x_t$ 被消去，后续 Head 无法感知"当前在轨迹的哪个位置"。

3. **速度场约束的信息瓶颈**：BPCVF 将速度作为**约束目标**而非**预测目标**。模型仍预测 $x_0$，但通过 $\hat{v} = (x_t - \hat{x}_0)/t$ 推导速度，在损失空间施加 PairWise 约束。这种"预测 $x_0$、约束 $v$"的解耦设计，保留了架构归纳偏置，同时注入了速度场结构化先验。

**类比**：这类似于谱归一化 (Spectral Normalization) 的思想——不改变模型的预测目标，而是通过约束中间量的性质 (Lipschitz 连续性) 来正则化模型行为。BPCVF 约束的是速度场的**配对平滑性**。

### 4.4 数值稳定性分析

BPCVF 损失包含 $\frac{1}{t^2}$ 加权，当 $t \to 0$ 时可能爆炸。分析：

- **RF 采样时间网格** (`rectified_flow.py:101-103`): `timesteps = linspace(1.0, 0.0, N+1)`，包含 $t=0$ 端点
- **训练时 $t$ 采样** (`head.py:415-423`): `t = torch.rand((bs,))`，$t \in (0, 1)$，不会精确为 0
- **但 $t$ 可以很小** (如 $t = 0.001$)，此时 $\frac{1}{t^2} = 10^6$，梯度爆炸

**缓解措施**：
1. **$t$ 下限 clamp**: `t_weight = 1.0 / (t.clamp(min=0.1) ** 2)`，限制最大权重为 100
2. **软加权**: 用 $\frac{1}{(t + \epsilon)^2}$ 替代 $\frac{1}{t^2}$，$\epsilon = 0.05$
3. **指数衰减**: 用 $e^{\alpha(1-t)}$ 替代 $\frac{1}{t^2}$，避免奇点

推荐方案 1 (clamp)，因为 clamp 后的 $\frac{1}{t^2}$ 在 $t > 0.1$ 时仍保持物理意义 (靠近数据时强约束)。

---

## 5. 与 KCEC 失败的对比

### 5.1 KCEC 回顾

KCEC (Karyotype-Constrained Entropic Coupling, Phase 8) 试图在 Sinkhorn OT 耦合中注入倍性约束 (col_mass 非均匀边际)，强制每类恰好匹配 2 条。所有 7 个实验均失败 (mAP=0.744 vs SOTA 0.753, -0.009)。

**KCEC 失败根因 (诊断结论)**：
1. **信息截断**: 耦合层 (Sinkhorn OT) 的约束不传播到 Transformer 表征层。Transformer 只看到 `(x_t, target)` 配对，无法感知"这个配对存在是因为倍性约束"
2. **同源表征缺失**: 诊断显示 A1 同源对的类内余弦相似度 = 0.8495，A1 vs G22 类间 = 0.8408，比值 = **1.01**。模型**没有同源等价性的内部概念**

### 5.2 BPCVF 如何避免 KCEC 的陷阱

| 维度 | KCEC (已失败) | BPCVF (本方案) |
|------|--------------|----------------|
| **约束位置** | 耦合层 (Sinkhorn col_mass) | 损失层 (PairWise loss) |
| **信息传播** | 耦合→配对→(截断)→Transformer | 预测→损失→梯度→Head 权重 |
| **梯度可达性** | 不可达 (耦合是 `@torch.no_grad()`) | 可达 (损失直接反传到 `reg_head`) |
| **表征影响** | 无 (Transformer 看不到耦合约束) | 有 (梯度直接塑造 `reg_head` 权重) |
| **同源概念** | 无 (诊断 ratio=1.01) | 显式注入 (配对损失强制同源一致性) |

**关键区别**：KCEC 在 `@torch.no_grad()` 的耦合层施加约束 (`matcher.py:38`, `ot_flow_coupling.py` 整个 `couple` 方法无梯度)，约束信号无法反传到模型参数。BPCVF 在 `loss()` 中施加约束，梯度直接流向 `head_series` 的 `reg_head` 和 `cls_head` 参数，**从信息论上保证了约束信号到达表征层**。

**形式化论证**：

KCEC 的信息流：
```
Sinkhorn col_mass 约束 → 传输矩阵 T → matched_idx → (x_t, target) 配对 → Transformer
                        ↑ 截断点 (no_grad)                    ↓
                        约束信号无法越过此点         Transformer 只看到配对结果
```

BPCVF 的信息流：
```
PairWise 一致性损失 ← 预测 x0_pred ← reg_head(fc_feature) ← Transformer ← RoI features
       ↓ 梯度反传
   reg_head 参数更新 ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
```

BPCVF 的约束信号通过梯度反传**完整到达**表征层，不存在 KCEC 的信息截断问题。

### 5.3 KCEC 诊断的启示

KCEC 诊断发现"模型没有同源等价性的内部概念" (ratio=1.01)。BPCVF 通过配对一致性损失，**显式地将同源等价性注入模型**：

- 损失 $\mathcal{L}_{\text{pair}} \propto \|\hat{x}_0^{(i)} - \hat{x}_0^{(j)}\|^2$ 直接惩罚同源对预测的发散
- 梯度迫使 `reg_head` 对相似特征 (同源染色体) 产生相似预测
- 这等价于在特征空间中施加**同源等变约束** (homologous equivariance)

**预期效果**：BPCVF 训练后，同源对的类内余弦相似度应显著高于类间 (ratio > 1.05)，表明模型学到了同源等价性概念。这可作为 BPCVF 有效性的诊断指标。

---

## 6. 风险评估

### 6.1 风险 1：配对识别错误 (中风险)

**风险描述**：训练时，matcher 的动态 Top-K 分配 (`matcher.py:286-312`) 可能将同类不同 GT 的 proposals 错误配对。例如，将匹配到 A1#0 的 proposal 与另一个也匹配到 A1#0 的 proposal 配对 (而非 A1#1)，导致一致性约束错误。

**影响**：错误配对会引入错误的一致性约束，可能降低而非提高预测精度。

**缓解方案**：
1. **GT 索引配对策略** (§3.1.1 推荐)：严格基于 `matched_gt_indices` 区分不同 GT 实例，确保配对来自不同 GT
2. **置信度过滤**：只对高置信度正样本 (score > 0.5) 施加配对约束，减少噪声配对
3. **配对数量限制**：每类最多 K 对 (如 K=5)，避免过多低质量配对
4. **消融实验**：对比不同配对策略 (按序 vs GT 索引 vs x_t 相似度) 的效果

### 6.2 风险 2：$\frac{1}{t^2}$ 加权的数值不稳定 (中风险)

**风险描述**：当 $t$ 较小时 (如 $t < 0.05$)，$\frac{1}{t^2} > 400$，配对一致性损失可能主导总损失，导致训练不稳定。这类似于 CFM 的梯度冲突问题——某个损失项过大，破坏训练平衡。

**影响**：训练早期 mAP 震荡，或晚期 mAP 平台化。

**缓解方案**：
1. **$t$ 下限 clamp** (§4.4 方案 1)：`t.clamp(min=0.1)`，限制最大权重为 100
2. **损失归一化**：将 BPCVF 损失除以当前 batch 的 L1 损失，保持相对量级稳定
3. **Warmup**：前 5 epoch 设 $\lambda_{\text{bpcvf}} = 0$，之后线性增长到目标值
4. **梯度裁剪**：对 BPCVF 损失的梯度单独裁剪 (max_norm=1.0)

### 6.3 风险 3：过约束导致同质化 (高风险)

**风险描述**：BPCVF 强制同源对预测一致，可能导致模型将同源对预测为**完全相同**的框 (退化解)，丧失定位多样性。这类似于 GAN 中的模式坍塌 (mode collapse)——模型为满足一致性约束，牺牲了个体定位精度。

**影响**：同源对的两条染色体被预测为重叠框，mAP 下降 (尤其是 AP75 和 AR)。

**缓解方案**：
1. **软一致性**：用 $\|\hat{x}_0^{(i)} - \hat{x}_0^{(j)}\|^2$ (L2) 而非 $\|\hat{x}_0^{(i)} - \hat{x}_0^{(j)}\|_1$ (L1)，L2 允许小差异存在
2. **只约束尺寸，不约束位置**：同源对大小应相似，但位置不同。约束 $\|(w_i, h_i) - (w_j, h_j)\|^2$ 而非全 4 维
3. **权重衰减**：设较小的 $\lambda_{\text{bpcvf}}$ (如 0.1)，让 L1+GIoU 主导定位，BPCVF 只起正则化作用
4. **监控指标**：训练中监控同源对的预测 IoU，若 > 0.9 (近乎重叠) 则降低 $\lambda_{\text{bpcvf}}$

### 6.4 风险 4：与 StochasticOT 的交互异常 (低风险)

**风险描述**：StochasticOT ε=5 的随机采样可能导致同类 proposals 被分配到差异较大的 GT 实例 (虽然同类，但大小/位置差异大)，此时配对一致性约束不合理。

**影响**：不一致的配对信号，削弱 BPCVF 效果。

**缓解方案**：
1. **配对前验证 GT 相似度**：只对 $\|x_0^{(i)} - x_0^{(j)}\| < \delta$ 的同源对施加约束
2. **自适应权重**：根据 GT 相似度调整配对损失权重 (相似度高→权重大)
3. **与 StochasticOT 解耦**：BPCVF 的配对基于 GT 标签 (同源) 而非耦合结果，不受 StochasticOT 随机性影响

### 6.5 风险 5：推理时配对后处理的副作用 (低风险)

**风险描述**：推理时配对后处理 (§3.3) 修正异常框时，可能将正确定位的框错误修正 (假阳性配对)。

**影响**：AR 下降 (漏检增加)。

**缓解方案**：
1. **Phase 3 再启用**：先验证训练时损失的效果 (Phase 1-2)，推理后处理作为可选增强
2. **保守修正**：只修正尺寸 (w, h)，不修正位置 (cx, cy)，且修正幅度不超过 20%
3. **置信度门控**：只对置信度差异 > 0.3 的配对施加修正

---

## 7. 预期收益分析

### 7.1 基于现有增益的定量外推

**参考增益数据**：

| 改进项 | mAP 增益 | 机制 | 与 BPCVF 的关系 |
|--------|---------|------|----------------|
| box_renewal | +0.016 | 采样后框更新 | 推理时改进, 与 BPCVF 正交 |
| StochasticOT ε=5 | SOTA 耦合 | 训练信号多样性 | 与 BPCVF 正交 (§2.4) |
| AdaLN-Zero | SOTA 条件化 | 时间条件化 | 与 BPCVF 正交 |
| KCEC (失败) | -0.009 | 耦合层约束 | BPCVF 避免了其信息截断问题 |
| Consistency Loss (失败) | -0.032 | 共享层一致性损失 | BPCVF 在输出空间, 非共享层 |

**外推逻辑**：

1. **正向上界**：BPCVF 注入了领域先验 (同源配对)，类似于 KCEC 的目标但机制不同。如果 BPCVF 成功注入同源等价性 (KCEC 失败的点)，预期增益应**超过 KCEC 的目标增益** (KCEC 设计目标 +0.005-0.010)。上界估计: **+0.008-0.012 mAP**

2. **中性预期**：BPCVF 是输出空间的轻量正则化，类似于 GIoU 损失 (输出空间, +0.016 增益中的组成部分)。但 BPCVF 只作用于同源对 (约 23 对 / 500 proposals ≈ 9.2% 的 proposals)，影响范围有限。中性估计: **+0.003-0.006 mAP**

3. **下界 (失败情景)**：如果配对识别错误或 $\frac{1}{t^2}$ 加权不稳定，BPCVF 可能像 Consistency Loss 一样引入梯度冲突。下界估计: **-0.005-0.010 mAP**

### 7.2 分类别预期收益

BPCVF 对不同类别的预期收益不同：

| 类别组 | 同源对区分难度 | 预期收益 | 理由 |
|--------|--------------|---------|------|
| A, B, F, G | 低 (组内类别少, 形态差异大) | +0.002-0.004 | 配对识别准确, 一致性约束有效 |
| D, E | 中 (组内 3 类, 形态相近) | +0.004-0.008 | 同源对形态相似, 一致性约束价值大 |
| C (6-12, X) | 高 (7 类, 形态极相似) | +0.006-0.012 | C 组是同组混淆重灾区, BPCVF 收益最大 |

**整体 mAP 增益估计**: **+0.003-0.008 mAP** (中性预期)

### 7.3 与 mAP_90 的关系

当前 mAP_90 (AP@IoU=0.90) 是更严格的精度指标。BPCVF 的配对一致性约束直接提升高 IoU 阈值下的精度 (两条同源染色体大小一致 → IoU 更高)。预期 mAP_90 增益 > mAP 增益：

- mAP 增益: +0.003-0.008
- mAP_90 增益: +0.005-0.012 (高 IoU 下收益更大)

### 7.4 成功判据

| 指标 | 基线 (a3_full_sota) | BPCVF 目标 | 判定 |
|------|-------------------|-----------|------|
| mAP | 0.858 | ≥ 0.861 | +0.003 为最小有效增益 |
| mAP_90 | (待测) | +0.005 | 高 IoU 增益更显著 |
| 同源对预测方差 | (待测) | -20% | 核心诊断指标 |
| 同源特征 ratio | 1.01 (KCEC 诊断) | ≥ 1.05 | 表征层同源等价性 |
| 训练稳定性 | 稳定 | 无震荡 | mAP 不出现 > 0.02 回落 |

---

## 8. 实现路线图

### 8.1 分 Phase 实施计划

#### Phase 1: 最小可行验证 (1 周)

**目标**：验证 BPCVF 损失的基本可行性和训练稳定性

**改动**：
1. `head.py`: 新增 `_bpcvf_loss` 方法和 `bpcvf_weight` 参数
2. `head.py: loss()`: 在 criterion 损失后追加 BPCVF 损失
3. 配对识别: 实现 GT 索引配对 (§3.1.1)
4. 损失: 残差速度一致性 (§3.2), $\lambda_{\text{bpcvf}} = 0.5$, $t$ clamp=0.1

**验证**：
- 训练 50 epoch, 观察 mAP 曲线稳定性
- 监控同源对预测方差和特征 ratio
- 消融: $\lambda_{\text{bpcvf}} \in \{0.1, 0.5, 1.0\}$

**代码改动清单**：

```
ldmdet/core/head.py:
  - __init__: 新增 bpcvf_weight, bpcvf_t_clamp, bpcvf_mode 参数
  - loss(): 新增 _bpcvf_loss 调用
  - _bpcvf_loss(): 新增方法 (§3.2 代码)
  - _gt_index_pairing(): 新增方法 (§3.1.1 代码)
  - _identify_homologous_pairs(): 新增方法 (§3.1.1 代码)

experiments/configs/ldmdet/directions/frontier_directions/:
  - s_bpcvf_phase1.py: 新增配置 (bpcvf_weight=0.5)
```

#### Phase 2: 消融与调优 (2 周)

**目标**：找到最优配置，验证各组件贡献

**实验矩阵**：

| 实验 | 变量 | 预期 |
|------|------|------|
| S1-baseline | bpcvf_weight=0 | 对照组 (= a3_full_sota) |
| S1-w0.1 | bpcvf_weight=0.1 | 弱约束 |
| S1-w0.5 | bpcvf_weight=0.5 | 中约束 (推荐) |
| S1-w1.0 | bpcvf_weight=1.0 | 强约束 |
| S2-size_only | 只约束 (w,h) | §6.3 缓解 |
| S2-full | 约束全 4 维 | 对照 |
| S3-t_clamp0.1 | t clamp=0.1 | §4.4 方案 1 |
| S3-t_clamp0.2 | t clamp=0.2 | 更保守 |
| S4-naive | naive 速度一致性 (错误版) | 消融: 验证残差 vs naive |
| S4-residual | 残差速度一致性 (正确版) | 对照 |

**关键消融**：S4-naive vs S4-residual 验证 §2.2-2.3 的理论分析——naive 版本 (直接约束 $\|v^{(i)} - v^{(j)}\|^2$) 应该失败或无增益，残差版本应有正收益。

#### Phase 3: 推理时配对约束 (1 周)

**目标**：验证推理后处理 (§3.3) 的增量收益

**改动**：
1. `head.py: predict()`: 在 post_process 后追加 `_pair_consistency_postprocess`
2. 实现 `_infer_homologous_pairs` (§3.1.2)
3. 实现 `_pair_consistency_postprocess` (§3.3)

**验证**：
- 对比有无推理后处理的 mAP
- 监控 AR 变化 (§6.5 风险)

#### Phase 4: 与 DPM-Solver++ 的兼容性验证 (1 周)

**目标**：验证 BPCVF 与 DPM-Solver++ 采样器 (`rectified_flow.py:92-158`) 的兼容性

**改动**：无代码改动，仅配置切换

**验证**：
- BPCVF 训练的模型 + DPM-2 8步采样 → mAP
- 对比 a3_full_sota + DPM-2 8步 (离线 0.755) 的增益

### 8.2 代码改动总览

```
ldmdet/
├── core/
│   └── head.py                          # 修改: +_bpcvf_loss, +配对识别, +参数
├── criterion/
│   └── (无改动)                          # BPCVF 在 head.py 中实现, 不改 criterion
├── diffusion/
│   └── (无改动)                          # rectified_flow.py 不变
└── coupling/
    └── (无改动)                          # StochasticOT 不变

experiments/configs/ldmdet/directions/frontier_directions/
├── s_bpcvf_phase1.py                    # 新增: Phase 1 配置
├── s_bpcvf_ablation_w.py                # 新增: 权重消融
├── s_bpcvf_ablation_mode.py             # 新增: naive vs residual 消融
└── s_bpcvf_phase3_infer.py              # 新增: Phase 3 推理后处理配置

tests/
└── test_direction_s_bpcvf.py            # 新增: 单元测试
```

**改动量估计**：~200 行新增代码 (head.py), ~100 行配置, ~100 行测试。总计 ~400 行。

### 8.3 测试计划

```python
class TestBPCVF:
    def test_pair_identification_correct(self):
        """同源对识别: 同类不同 GT 实例的 proposals 正确配对"""

    def test_residual_velocity_consistency(self):
        """残差速度一致性: L_pair ≈ (1/t²)||x0_pred_i - x0_pred_j||²"""

    def test_naive_velocity_inconsistency(self):
        """naive 速度一致性: ||v_i - v_j||² ≈ 2d (非小量, 验证 §2.2)"""

    def test_t_clamp_stability(self):
        """t→0 时 1/t² 被 clamp, 梯度不爆炸"""

    def test_gradient_flow_to_reg_head(self):
        """BPCVF 损失梯度到达 reg_head (区别于 KCEC)"""

    def test_no_cascade_break(self):
        """级联 Head 输入仍是 x0 (非速度), 架构不变 (区别于 CFM)"""

    def test_pair_count(self):
        """配对数量合理: ~23 对 / image (46 条染色体)"""

    def test_homologous_feature_ratio(self):
        """训练后同源特征 ratio > 1.05 (KCEC 诊断指标改善)"""
```

---

## 9. 与已证伪方向的系统对比

### 9.1 对比表

| 维度 | CFM (方向 H, 已证伪) | KCEC (Phase 8, 已失败) | Consistency Loss (Phase 2, 已失败) | **BPCVF (本方案)** |
|------|---------------------|----------------------|-----------------------------------|-------------------|
| **mAP** | 0.823 (-0.035) | 0.744 (-0.009) | 0.721 (-0.032) | 待验证 |
| **失败/风险根因** | 级联输入丢失 x_t; delta 回归被绕过 | 耦合层信息截断; 无同源表征 | 共享层梯度冲突 | 配对识别错误; 1/t² 不稳定 |
| **约束位置** | 预测目标 (reg_head 输出) | 耦合层 (Sinkhorn col_mass) | 共享层 (特征一致性) | **损失层 (PairWise 输出约束)** |
| **梯度可达性** | 可达 (但方向冲突) | 不可达 (no_grad) | 可达 (但与检测损失冲突) | **可达且协同** (§4.2) |
| **架构改动** | 大 (改 _predict, 级联转换) | 中 (改耦合) | 小 (加特征损失) | **小 (只加损失)** |
| **归纳偏置** | 破坏 (绕过 delta 回归) | 不影响 | 不影响 | **保持 (仍走 apply_deltas)** |
| **领域先验** | 无 | 有 (倍性约束) | 无 | **有 (同源配对)** |
| **信息瓶颈** | 速度→x0 反推截断 | 耦合→Transformer 截断 | 无截断 (但冲突) | **无截断 (输出空间约束)** |

### 9.2 核心区分：BPCVF 不是 CFM

**最常见的误解**：BPCVF 和 CFM 都涉及"速度"，因此 BPCVF 可能重蹈 CFM 覆辙。

**澄清**：

1. **CFM 直接预测速度** (`single_head.py: predict_velocity=True, _predict 返回 velocity`)。这改变了 `reg_head` 的输出语义，破坏了 `apply_deltas` 的 delta 回归机制，且级联 Head 接收反推的 $x_0$ (丢失 $x_t$)。

2. **BPCVF 约束推导速度**。模型仍预测 $x_0$ (`predict_velocity=False`, 走 `_predict_bboxes → apply_deltas`)，速度是从 $x_0$ 推导的: $\hat{v} = (x_t - \hat{x}_0)/t$。BPCVF 只在损失空间约束推导速度的配对一致性，不改变预测目标、级联输入流、delta 回归机制。

3. **形式化区分**：
   - CFM: $\hat{v} = \text{reg\_head}(\phi)$, $\hat{x}_0 = x_t - t \cdot \hat{v}$ (预测→反推)
   - BPCVF: $\hat{x}_0 = \text{apply\_deltas}(\text{reg\_head}(\phi), \text{box})$, $\hat{v} = (x_t - \hat{x}_0)/t$ (预测→推导→约束)

4. **信息流区分**：
   - CFM: 速度是**预测的中间量**，参与级联输入计算 → 速度误差影响级联
   - BPCVF: 速度是**损失的中间量**，不参与级联输入 → 速度约束只影响梯度

### 9.3 BPCVF 不是 KCEC

**最常见的误解**：BPCVF 和 KCEC 都注入核型配对先验，因此 BPCVF 可能重蹈 KCEC 覆辙。

**澄清**：

1. **KCEC 在耦合层施加约束** (Sinkhorn col_mass)，耦合层是 `@torch.no_grad()` 的数据准备环节，约束信号无法反传到模型参数。模型表征层完全感知不到配对约束。

2. **BPCVF 在损失层施加约束** (PairWise loss)，损失直接参与梯度计算，梯度流向 `reg_head` 和 `cls_head` 参数。模型表征层被强制学习同源等价性。

3. **KCEC 的诊断结论 ("模型没有同源等价性概念") 恰恰是 BPCVF 要解决的问题**——BPCVF 通过梯度显式注入同源等价性，预期训练后 ratio > 1.05 (KCEC 诊断 ratio = 1.01)。

### 9.4 BPCVF 不是 Consistency Loss

**Phase 2 的 Consistency Loss** 在共享层施加特征一致性约束 (velocity_head 特征与 detection 特征一致)，导致与检测损失的梯度冲突 (mAP=0.721, -0.032)。

**BPCVF 的区别**：
1. BPCVF 约束在**输出空间** (预测框的 cxcywh)，而非共享特征层
2. BPCVF 的梯度与 L1+GIoU **协同** (§4.2 分析)，而非冲突
3. BPCVF 只作用于**同源对** (~9% 的 proposals)，而非全部 proposals

---

## 10. 参考文献

### 10.1 核心理论

- **Rectified Flow**: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023). https://arxiv.org/abs/2209.03003
  - RF 直线路径理论, velocity = x_1 - x_0 的定义

- **Conditional Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023). https://arxiv.org/abs/2302.00482
  - CFM 理论框架, OT-CFM 配对策略

- **Stochastic Interpolants**: Albergo & Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions" (2023). https://arxiv.org/abs/2303.08797
  - 随机插值理论, RF 的广义框架

### 10.2 检测与扩散

- **DiffusionDet**: Zhang et al., "DiffusionDet: Diffusion Model for Object Detection" (ICCV 2023). https://arxiv.org/abs/2211.09788
  - 扩散检测基础架构

- **FlowDet**: Baty et al., "FlowDet: Unifying Object Detection and Generative Transport Flows" (2025). https://arxiv.org/abs/2512.16771
  - CFM 检测, COCO +3.6% AP

### 10.3 配对与一致性

- **Consistency Models**: Song et al., "Consistency Models" (ICML 2023). https://arxiv.org/abs/2303.01469
  - 自一致性约束理论

- **Graph R-CNN**: Yang et al., "Graph R-CNN for Scene Graph Generation" (ECCV 2018). https://arxiv.org/abs/1808.07172
  - 关系建模, 框间交互

- **OT-CFM**: Tong et al., "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" (2023). https://arxiv.org/abs/2302.00482
  - OT 配对提升 CFM

### 10.4 染色体核型

- **ISCN 2024**: International System for Human Cytogenomic Nomenclature
  - 染色体核型国际标准, 46 条 23 对

- **Deep learning for karyotyping**: 多篇综述
  - 染色体自动核型分析的深度学习方法

### 10.5 项目内文档

- [方向 H: FlowMatching 检测](方向H_FlowMatching检测.md) — CFM 证伪记录
- [方向 K: 核型结构化生成](方向K_核型结构化生成.md) — GNN 核型建模
- [方向间关系与纠正说明](方向间关系与纠正说明.md) — 方向间依赖与冲突
- [THEORY_FRAMEWORK.md](../theory/THEORY_FRAMEWORK.md) Appendix C — KCEC 理论框架
- [OT_DIVERSITY_COLLAPSE_PROOF.md](../theory/OT_DIVERSITY_COLLAPSE_PROOF.md) §12 — StochasticOT 理论

---

## 附录 A：关键代码位置索引

| 代码位置 | 文件 | 行号 | 说明 |
|---------|------|------|------|
| RF 前向扩散 | `ldmdet/diffusion/rectified_flow.py` | 20-40 | `q_sample`: x_t = (1-t)x_0 + t·x_1 |
| RF 速度推导 | `ldmdet/diffusion/rectified_flow.py` | 42-45 | `get_velocity`: v = (x_t - x_0_pred)/t |
| RF Euler 采样 | `ldmdet/diffusion/rectified_flow.py` | 47-61 | `step`: x_next = x_t + dt·v |
| RF Heun 采样 | `ldmdet/diffusion/rectified_flow.py` | 63-89 | `heun_step`: 二阶采样 |
| DPM-Solver++ | `ldmdet/diffusion/rectified_flow.py` | 92-158 | `RFDPMSolverMultistep` |
| 级联 Head 前向 | `ldmdet/core/head.py` | 174-229 | `forward`: 6 级级联, cascade_detach |
| 训练损失 | `ldmdet/core/head.py` | 235-311 | `loss`: CFM 路径 vs 非 CFM 路径 |
| CFM 速度损失 | `ldmdet/core/head.py` | 268-309 | `use_cfm=True` 时 MSE(v_pred, v_target) |
| 训练目标构建 | `ldmdet/core/head.py` | 425-453 | `_build_training_targets`: 耦合+扩散 |
| 耦合调用 | `ldmdet/core/head.py` | 455-462 | `_couple_single_image` |
| SingleHead 预测 | `ldmdet/core/single_head.py` | 303-319 | `_predict`: CFM(velocity) vs 非 CFM(x0) |
| delta 回归 | `ldmdet/core/single_head.py` | 389-422 | `apply_deltas`: 框锚点 delta 回归 |
| AdaLN-Zero | `ldmdet/core/single_head.py` | 321-357 | `_forward_adaln_zero` |
| StochasticOT 耦合 | `ldmdet/coupling/ot_flow_coupling.py` | 59-114 | `couple`: Sinkhorn + multinomial |
| 随机耦合 | `ldmdet/coupling/random.py` | 13-25 | `RandomCoupling` |
| 动态 Top-K 匹配 | `ldmdet/criterion/matcher.py` | 286-312 | `_dynamic_k_matching` |
| 损失计算 | `ldmdet/criterion/criterion.py` | 53-94 | `forward` + `_get_loss` |
| L1+GIoU+Focal | `ldmdet/criterion/losses.py` | 10-102 | 三个损失函数 |

---

## 附录 B：naive vs residual 速度一致性消融设计

为验证 §2.2-2.3 的理论分析，设计对照实验：

**Naive 版本 (理论错误)**：
```python
# 直接约束预测速度的配对一致性
v_pred_i = (x_t_i - x0_pred_i) / t
v_pred_j = (x_t_j - x0_pred_j) / t
loss_naive = (v_pred_i - v_pred_j).pow(2).sum()
# 理论分析: loss_naive ≈ ||x_noise_i - x_noise_j||² ≈ 8 (常数, 无信息)
```

**Residual 版本 (理论正确)**：
```python
# 约束残差速度的配对一致性
v_residual_i = (x0_gt_i - x0_pred_i) / t
v_residual_j = (x0_gt_j - x0_pred_j) / t
loss_residual = (v_residual_i - v_residual_j).pow(2).sum()
# 理论分析: loss_residual ≈ (1/t²)||x0_pred_i - x0_pred_j||² (有意义)
```

**预期结果**：
- Naive 版本: mAP ≤ 基线 (无信息约束, 等价于随机扰动)
- Residual 版本: mAP > 基线 (有效正则化)

如果 Naive 版本也有效 (mAP > 基线)，则说明理论分析有误，需重新审视 §2.2 的推导。

---

> **文档版本**: v1.0
>
> **撰写日期**: 2026-07-12
>
> **作者**: LDMDet 研究团队
>
> **审核状态**: 待实验验证
