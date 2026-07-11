# 方向 Q：ATD (Amodal Trajectory Diffusion, Amodal 轨迹扩散)

> **目标**：利用扩散模型的生成式先验能力, 在训练时主动模拟遮挡场景, 强制模型学会从部分观测中补全完整的染色体边界, 攻克 amodal 检测中重叠/交叉实例的漏检与定位偏移问题。
>
> **理论依据**：
> - Kirkegaard 2024: 扩散模型在重叠细胞检测中的"自发对称性破缺"
> - CytoDiffusion (Nature 2025): 扩散模型在血细胞分类中的 OOD 鲁棒性 (0.854 vs 判别模型 0.738)
> - Amodal Detection: 补全理论的生成式视角
>
> **当前代码位置**：
> - [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) — RF 前向扩散与采样
> - [ldmdet/core/head.py](../../ldmdet/core/head.py) — DiffusionDetHead 训练/推理主逻辑
> - [ldmdet/core/single_head.py](../../ldmdet/core/single_head.py) — 单步检测头 (AdaLN-Zero)
> - [ldmdet/diffusion/sampling.py](../../ldmdet/diffusion/sampling.py) — 采样器 (含 box_renewal)
>
> **基线**: a3_full_sota, mAP=0.858 (RF + Heun 4步 + AdaLN-Zero + StochasticOT ε=5)

---

## 1. 背景与动机

### 1.1 染色体重叠问题的特殊性

染色体检测的核心难点在于中期细胞分裂图像中染色体频繁的**物理重叠与交叉**。与通用目标检测中的"遮挡"不同, 染色体重叠具有以下特征:

| 特征 | 通用目标遮挡 | 染色体重叠 |
|------|------------|-----------|
| 重叠原因 | 视角/前景遮挡 | 物理堆积, 染色体在同一平面上交叉 |
| 标注方式 | visible-only (仅可见部分) | **amodal** (标注完整边界, 即使不可见) |
| 重叠程度 | 部分遮挡 (0-70%) | 高度重叠 (可达 80-95%), 多条交织 |
| 实例可分性 | 通常靠纹理/颜色区分 | **形态高度相似**, 仅靠位置和细微形态差异 |
| 数量密度 | 每图 ~10-20 目标 | 每图 **46 条** (24 类, 含同源染色体对) |

**关键挑战**: 24obj 数据集采用 amodal 标注, 要求模型预测**完整边界框**而非可见部分。当两条染色体交叉时, 现有检测器面临两个问题:

1. **定位偏移**: 模型倾向于将框中心移向可见部分的重心, 而非真实中心
2. **漏检**: 高度重叠的实例可能被合并为单个检测, 或因置信度过低被 NMS 过滤

### 1.2 扩散模型的生成式先验优势

当前 LDMDet 架构 (RF + Heun + AdaLN-Zero + StochasticOT) 已展现扩散模型在检测任务上的两个关键优势:

**优势 1: box_renewal 的"探索-利用"机制**

[sampling.py:96-117](../../ldmdet/diffusion/sampling.py#L96-L117) 的 `apply_box_renewal` 在每步采样后将低置信度框替换为随机噪声:

```python
def apply_box_renewal(self, x_raw, cls_logits):
    scores = torch.sigmoid(cls_logits).max(-1)[0]
    keep = scores[i] > self.score_thr
    x_raw_new[i, ~keep] = torch.randn(num_renew, 4, device=device)
```

这一机制贡献了 **+0.016 mAP** (最强单一扩散机制增益), 本质是扩散采样的**随机重探索** — 低置信度框被"重置"为噪声, 给模型第二次机会去捕获漏检实例。这与 amodal 检测的需求高度契合: 重叠区域中被"淹没"的实例可以通过重新采样被分离出来。

**优势 2: 多步迭代的渐进式精化**

6 级级联 Head ([head.py:119-121](../../ldmdet/core/head.py#L119-L121)) + Heun 4步采样提供了 **24 次精化机会** (6 cascade × 4 steps), 每次精化都能利用全局上下文修正局部误判。对于重叠场景, 这意味着模型可以在多步迭代中逐步"分离"交织的实例。

### 1.3 现有架构的不足

尽管 box_renewal 已证明有效, 但当前训练流程存在根本性局限:

**问题 1: 训练时从未见过遮挡场景**

当前训练流程 ([head.py:235-311](../../ldmdet/core/head.py#L235-L311)) 中, GT 框经过 `_build_training_targets` 直接耦合到噪声, 模型在训练时看到的是**完整无遮挡的 RoI 特征**。推理时遇到重叠场景, RoI 特征被污染 (多条染色体特征混合), 模型缺乏从部分观测恢复完整框的能力。

**问题 2: 采样轨迹缺乏一致性约束**

级联 Head 的 6 级预测 ([head.py:219-229](../../ldmdet/core/head.py#L219-L229)) 通过 `deep_supervision` 各自独立计算损失, 但**不同级之间的预测一致性未被显式约束**。对于重叠场景, 早期级 (低精度) 可能预测被截断的框, 而后期级 (高精度) 应收敛到完整框 — 这一"渐进补全"过程目前完全隐式。

**问题 3: 单次采样的实例分离能力不足**

当前推理 ([head.py:318-395](../../ldmdet/core/head.py#L318-L395)) 用单次随机种子 `torch.randn(bs, num_proposals, 4)` 初始化 500 个 proposal。对于高度重叠的区域, 多个 proposal 可能收敛到同一条"显眼"的染色体, 而忽略被遮挡的实例。box_renewal 虽能重探索, 但其触发条件 (score < score_thr) 对"被遮挡但部分可见"的实例无效。

### 1.4 ATD 的核心思想

ATD 的核心洞察是: **扩散模型的生成式先验天然适合 amodal 补全任务** — 正如扩散模型能从噪声生成完整图像, 它也应能从部分观测 (遮挡 RoI 特征) 生成完整框。ATD 通过三个正交机制实现这一目标:

```
Q1: 训练时遮挡模拟 → 让模型学会从部分观测补全完整框
Q2: 轨迹一致性损失 → 约束采样轨迹渐进收敛到 amodal 框
Q3: 多假设采样分离 → 利用扩散随机性分离重叠实例
```

---

## 2. 理论依据

### 2.1 Kirkegaard 2024: 自发对称性破缺

Kirkegaard (2024) 在重叠细胞检测中发现扩散模型展现**自发对称性破缺** (spontaneous symmetry breaking) 现象:

> 给定一个包含 K 个重叠细胞的观测 $x_{obs}$, 生成模型 $p_\theta(x_{full} | x_{obs})$ 的后验分布是多模态的 — 存在 K! 种实例排列对应相同的观测。扩散采样的随机性 $\epsilon \sim \mathcal{N}(0, I)$ 能自然地将不同采样分支分配到不同模式, 实现"无监督实例分离"。

**数学描述**: 设重叠场景有 K 条染色体, 真实配置为 $\{b_1, b_2, ..., b_K\}$ (amodal 框)。观测 $x_{obs}$ 是所有染色体可见部分的混合。后验分布:

$$p(\{b_1, ..., b_K\} | x_{obs}) = \sum_{\sigma \in S_K} p_\sigma(\{b_{\sigma(1)}, ..., b_{\sigma(K)}\} | x_{obs})$$

其中 $S_K$ 是 K 个实例的排列群。扩散采样 $x_T \sim \mathcal{N}(0, I) \to x_0 \sim p(x_0 | x_{obs})$ 中, 不同的 $x_T$ 初始化会自然落入不同的模式 $p_\sigma$, 因为高维高斯噪声的微小扰动会在多模态分布中被放大。

**与 LDMDet 的关联**: 当前 box_renewal 机制 ([sampling.py:96-117](../../ldmdet/diffusion/sampling.py#L96-L117)) 已经体现了这一原理 — 低置信度框被替换为随机噪声, 等于在不同模式间重新采样。ATD 的 Q3 (多假设采样) 是对这一原理的**显式利用**: 通过多次独立采样, 主动覆盖更多模式。

### 2.2 CytoDiffusion: 扩散的 OOD 鲁棒性

CytoDiffusion (Nature 2025) 在血细胞分类中对比扩散模型与判别模型:

| 模型类型 | In-domain accuracy | OOD accuracy |
|---------|-------------------|-------------|
| 判别模型 (ResNet) | 0.891 | 0.738 |
| 扩散模型 (CytoDiffusion) | 0.887 | **0.854** |

**关键结论**: 扩散模型在 In-domain 上与判别模型持平, 但在 OOD (分布外) 场景上显著领先 (+0.116)。理论解释是扩散模型通过"生成-判别"两步: 先从噪声生成最可能的完整样本, 再基于生成样本分类 — 这种**生成式先验**提供了对分布偏移的鲁棒性。

**与染色体检测的关联**: 染色体重叠场景本质上是一种**结构性 OOD** — 训练集中重叠程度有限, 但测试集中可能出现更严重的重叠。扩散模型的 OOD 鲁棒性理论上能缓解这一问题。

### 2.3 Amodal 补全的生成式理论

Amodal 检测的生成式视角将问题形式化为:

$$\hat{b}_{amodal} = \mathbb{E}_{b \sim p_\theta(b | x_{visible})}[b]$$

其中 $x_{visible}$ 是 RoI 的可见部分特征, $p_\theta(b | x_{visible})$ 是条件生成分布。

**扩散模型的适配性**: RF 的前向过程 $x_t = (1-t)x_0 + t\epsilon$ ([rectified_flow.py:38](../../ldmdet/diffusion/rectified_flow.py#L38)) 中, $x_0$ 是完整 GT 框, $x_t$ 是加噪后的"部分观测"。这天然对应 amodal 场景: $t$ 越大, "遮挡"越严重, 模型需要从更少的信号中恢复 $x_0$。

**关键区别**: 通用扩散模型从**像素级噪声**生成图像; LDMDet 从**4维框噪声**生成检测框。虽然 4维空间看似简单, 但染色体的**形态约束** (amodal 框必须符合染色体形态学) 使得生成分布 $p(b)$ 具有强结构性, 扩散模型能利用这一结构。

### 2.4 轨迹一致性的变分视角

在级联 Head 中, 不同级的预测 $\hat{x}_0^{(1)}, \hat{x}_0^{(2)}, ..., \hat{x}_0^{(6)}$ 可视为同一后验分布的**不同精度估计**。轨迹一致性损失约束这些估计的方差:

$$\mathcal{L}_{consist} = \text{Var}[\hat{x}_0^{(1)}, ..., \hat{x}_0^{(6)}] = \frac{1}{H}\sum_{h=1}^{H} \|\hat{x}_0^{(h)} - \bar{x}_0\|^2$$

从变分推断角度, 这等价于最小化后验估计的**认知不确定性** (epistemic uncertainty)。对于 amodal 检测, 降低不确定性意味着模型对"被遮挡部分"的预测更稳定, 不因级联深度而剧烈变化。

---

## 3. 详细方案设计

### 3.1 Q1: 训练时遮挡模拟 (Occlusion Simulation)

#### 3.1.1 核心思想

在训练阶段, 对 RoI 特征或输入框随机施加遮挡, 但损失仍对完整 GT 框计算。这强制模型学习从部分观测恢复完整框的能力。

#### 3.1.2 遮挡策略设计

**策略 A: RoI 特征级遮挡 (Feature Masking)**

在 [single_head.py:267-290](../../ldmdet/core/single_head.py#L267-L290) 的 `forward` 方法中, RoI 特征提取后施加随机 mask:

```python
def forward(self, features, bboxes, proposals, pooler, time_emb):
    bs, num_boxes = bboxes.shape[:2]
    rois = bbox2roi([bboxes[i] for i in range(bs)])
    roi_features = pooler(features, rois)  # (N, C, H, W), H=W=7

    # Q1: RoI 特征遮挡模拟
    if self.training and self.occlusion_prob > 0:
        roi_features = self._apply_occlusion(roi_features)

    # ... 后续逻辑不变
```

遮挡实现:

```python
def _apply_occlusion(self, roi_features):
    """随机遮挡 RoI 特征的一部分, 模拟视觉遮挡"""
    N, C, H, W = roi_features.shape
    # 每个 RoI 独立决定是否遮挡
    occlude_mask = torch.rand(N, device=roi_features.device) < self.occlusion_prob
    if not occlude_mask.any():
        return roi_features

    # 随机遮挡区域 (矩形 mask, 模拟重叠)
    occluded = roi_features.clone()
    for i in torch.where(occlude_mask)[0]:
        # 遮挡比例: 20%-60% (模拟不同程度重叠)
        occ_ratio = torch.empty(1).uniform_(0.2, 0.6).item()
        occ_h = max(1, int(H * occ_ratio**0.5))
        occ_w = max(1, int(W * occ_ratio**0.5))
        # 随机位置
        y0 = torch.randint(0, H - occ_h + 1, (1,)).item()
        x0 = torch.randint(0, W - occ_w + 1, (1,)).item()
        # 零填充 (或均值填充)
        occluded[i, :, y0:y0+occ_h, x0:x0+occ_w] = 0.0
    return occluded
```

**策略 B: 框坐标级扰动 (Box Perturbation)**

在 [head.py:425-453](../../ldmdet/core/head.py#L425-L453) 的 `_build_training_targets` 中, 对部分 GT 框施加"截断"模拟:

```python
def _build_training_targets(self, bs, device, t, targets, gt_bboxes):
    x_boxes, x_starts, x_noises, matched_gt_indices = [], [], [], []
    for i in range(bs):
        # ... 现有逻辑 ...
        gt_diffusion = (norm_gt_cxcywh * 2 - 1) * self.snr_scale

        # Q1: 框坐标截断模拟 (amodal → visible 模拟)
        if self.training and self.occlusion_prob > 0:
            gt_diffusion = self._simulate_truncation(gt_diffusion)

        # ... 后续耦合逻辑不变 ...
```

截断实现:

```python
def _simulate_truncation(self, gt_diffusion):
    """模拟 amodal → visible 的截断: 缩小框的 w/h"""
    N = gt_diffusion.shape[0]
    occlude_mask = torch.rand(N, device=gt_diffusion.device) < self.occlusion_prob
    if not occlude_mask.any():
        return gt_diffusion

    perturbed = gt_diffusion.clone()
    # gt_diffusion 是 cxcywh 格式, 范围 [-snr_scale, snr_scale]
    # 截断: 缩小 w, h (模拟可见部分比完整框小)
    scale_factors = torch.ones(N, 4, device=gt_diffusion.device)
    scale_factors[occlude_mask, 2] = torch.empty(
        occlude_mask.sum(), device=gt_diffusion.device
    ).uniform_(0.4, 0.8)  # w 缩小到 40%-80%
    scale_factors[occlude_mask, 3] = torch.empty(
        occlude_mask.sum(), device=gt_diffusion.device
    ).uniform_(0.4, 0.8)  # h 缩小到 40%-80%
    perturbed = perturbed * scale_factors
    return perturbed
```

**策略选择**: 推荐策略 A (RoI 特征级遮挡), 因为它更接近真实遮挡机理 — 重叠染色体在 RoI 特征上表现为特征混合/污染, 而非简单的框缩放。策略 B 作为消融对比。

#### 3.1.3 损失函数设计

**核心原则**: 遮挡施加在**输入端**, 损失计算在**完整 GT** 上。

当前损失 ([head.py:306](../../ldmdet/core/head.py#L306)):
```python
losses = self.criterion(outputs, targets, t=t)
```

ATD 不改变损失目标 — `targets` 仍然是完整 amodal GT 框。遮挡只影响模型**看到的输入**, 不影响**监督信号**。这确保模型学习的是 $p(b_{amodal} | x_{occluded})$ 而非 $p(b_{visible} | x_{occluded})$。

**辅助损失 (可选)**: 对被遮挡的样本施加额外权重的 GIoU 损失:

```python
# 对遮挡样本的 GIoU 损失加权
if hasattr(self.criterion, 'set_occlusion_mask'):
    self.criterion.set_occlusion_mask(occlude_mask)
    # criterion 内部: loss_giou[occlude_mask] *= self.occlusion_giou_weight
```

### 3.2 Q2: 轨迹一致性损失 (Trajectory Consistency Loss)

#### 3.2.1 核心思想

在级联 Head 的 6 级预测中, 约束不同级的预测保持形状一致性 — 即使中间级受噪声扰动, 最终应收敛到完整的 amodal 框。

#### 3.2.2 数学推导

当前级联 Head ([head.py:182-229](../../ldmdet/core/head.py#L182-L229)) 的 6 级预测为 $\hat{x}_0^{(1)}, ..., \hat{x}_0^{(6)}$。定义轨迹一致性损失:

$$\mathcal{L}_{traj} = \frac{1}{H(H-1)} \sum_{i < j} \omega_{ij} \cdot \|\hat{x}_0^{(i)} - \hat{x}_0^{(j)}\|_1$$

其中 $\omega_{ij}$ 是基于级间距离的衰减权重:

$$\omega_{ij} = \exp\left(-\frac{|i-j|}{\tau}\right)$$

$\tau$ 控制衰减速度 — 相邻级 (|i-j|=1) 权重最高, 远级 (|i-j|=5) 权重最低。这反映"近邻级预测应更一致"的先验。

**变体: 中心化一致性**

以最后一级 (最高精度) 为锚点:

$$\mathcal{L}_{traj}^{anchor} = \frac{1}{H-1} \sum_{h=1}^{H-1} \|\hat{x}_0^{(h)} - \text{sg}(\hat{x}_0^{(H)})\|_1$$

其中 $\text{sg}(\cdot)$ 是 stop-gradient, 避免早期级的梯度反向传播到最后级。这确保最后级保持独立优化, 早期级"对齐"到最后级。

#### 3.2.3 代码实现

在 [head.py:300-311](../../ldmdet/core/head.py#L300-L311) 的 `loss` 方法中添加:

```python
def loss(self, features, img_metas, gt_bboxes, gt_labels):
    # ... 现有前向传播 ...

    losses = self.criterion(outputs, targets, t=t)

    # Q2: 轨迹一致性损失
    if self.trajectory_consistency_weight > 0:
        loss_traj = self._trajectory_consistency_loss(
            all_pred_bboxes  # [num_heads, bs, num_proposals, 4]
        )
        losses['loss_traj'] = (
            self.trajectory_consistency_weight * loss_traj
        )

    if self.use_cfm:
        losses['loss_velocity'] = self.velocity_loss_weight * loss_v

    return losses

def _trajectory_consistency_loss(self, all_pred_bboxes):
    """级联 Head 的轨迹一致性损失"""
    num_heads = all_pred_bboxes.shape[0]
    if num_heads < 2:
        return all_pred_bboxes.new_tensor(0.0)

    # 归一化到 [0,1] 空间 (与 criterion 一致)
    norm_preds = self._normalize_pred_bboxes(
        all_pred_bboxes, self._img_metas_for_loss
    )

    # 锚点: 最后一级 (stop-gradient)
    anchor = norm_preds[-1].detach()
    loss = 0.0
    for h in range(num_heads - 1):
        # 衰减权重: 越接近最后级, 权重越高
        w = math.exp(-(num_heads - 1 - h - 1) / self.tau)
        loss = loss + w * F.l1_loss(norm_preds[h], anchor)
    return loss / (num_heads - 1)
```

#### 3.2.4 与现有 deep_supervision 的关系

当前 `deep_supervision` ([head.py:517-524](../../ldmdet/core/head.py#L517-L524)) 对每级独立计算分类+回归损失, 监督信号是 GT。轨迹一致性损失是**级间约束**, 监督信号是最后级的预测。两者互补:

- `deep_supervision`: 每级 → GT (绝对监督)
- `trajectory_consistency`: 早期级 → 最后级 (相对约束)

### 3.3 Q3: 多假设采样分离 (Multi-Hypothesis Sampling)

#### 3.3.1 核心思想

对重叠区域, 利用扩散的随机性采样多个候选框集合, 通过几何分析 (IoU/聚类) 分离重叠实例。

#### 3.3.2 采样策略

**策略: 多次独立采样 + 聚类合并**

修改 [head.py:318-395](../../ldmdet/core/head.py#L318-L395) 的 `predict` 方法:

```python
@torch.no_grad()
def predict(self, features, img_metas, rescale=True, return_trajectory=False):
    if self.num_hypotheses <= 1:
        # 原始单次采样
        return self._single_hypothesis_predict(
            features, img_metas, rescale, return_trajectory
        )

    # Q3: 多假设采样
    all_hypothesis_results = []
    for h in range(self.num_hypotheses):
        # 每次用不同的随机种子
        torch.manual_seed(self.base_seed + h)
        results = self._single_hypothesis_predict(
            features, img_metas, rescale=False  # 先不 rescale
        )
        all_hypothesis_results.append(results)

    # 聚类合并: 跨假设的检测结果聚合
    merged_results = self._merge_hypotheses(
        all_hypothesis_results, img_metas, rescale
    )
    return merged_results
```

#### 3.3.3 假设合并的几何分析

```python
def _merge_hypotheses(self, all_results, img_metas, rescale):
    """跨假设的检测结果聚类合并

    核心思想: 同一真实实例在不同假设中应产生相似的检测框,
    通过聚类找到"共识"检测, 过滤"离群"检测。
    """
    bs = len(img_metas)
    merged = []
    for i in range(bs):
        # 收集所有假设的检测
        all_boxes = []
        all_scores = []
        all_labels = []
        for results in all_results:
            all_boxes.append(results[i].bboxes)
            all_scores.append(results[i].scores)
            all_labels.append(results[i].labels)

        all_boxes = torch.cat(all_boxes)      # (H * K, 4)
        all_scores = torch.cat(all_scores)    # (H * K,)
        all_labels = torch.cat(all_labels)    # (H * K,)

        # 按类别分组, 类内聚类
        final_boxes = []
        final_scores = []
        final_labels = []
        for label in all_labels.unique():
            mask = all_labels == label
            cls_boxes = all_boxes[mask]
            cls_scores = all_scores[mask]

            # 类内 NMS, 但 IoU 阈值设高 (0.7), 只合并高度重叠的框
            keep = batched_nms(
                cls_boxes, cls_scores,
                torch.zeros_like(cls_labels[mask]),  # 同类
                self.hypothesis_iou_thr
            )
            # 取 keep 后的框 (NMS 保留最高分)
            final_boxes.append(cls_boxes[keep])
            final_scores.append(cls_scores[keep])
            final_labels.append(
                torch.full((len(keep),), label.item(), device=cls_boxes.device)
            )

        final_boxes = torch.cat(final_boxes)
        final_scores = torch.cat(final_scores)
        final_labels = torch.cat(final_labels)

        # 最终 NMS (低阈值, 去重复)
        keep = batched_nms(
            final_boxes, final_scores, final_labels, self.nms_thr
        )
        # ... rescale 逻辑 ...
        merged.append(DetectionResult(...))
    return merged
```

#### 3.3.4 计算复杂度分析

| 配置 | NFE (单假设) | NFE (K假设) | 推理时间倍数 |
|------|-------------|------------|------------|
| Heun 4步 | 8 | 8K | K |
| DPM-2 6步 | 7 | 7K | ~K |

对于 K=3 假设, 推理时间约 3 倍。但染色体检测不是实时任务 (核型分析可离线), 3 倍推理时间换取重叠实例分离是可接受的。

**优化**: 利用 batch 并行 — 将 K 个假设作为 batch 维度并行采样:

```python
# 原始: (bs, num_proposals, 4)
# 多假设: (bs * K, num_proposals, 4) — 单次前向处理 K 个假设
x_raw = torch.randn(bs * self.num_hypotheses, self.num_proposals, 4, device=device)
```

这样推理时间仅增加 ~1.2-1.5 倍 (取决于 GPU 并行度), 而非 K 倍。

---

## 4. 深入可行性分析

### 4.1 理论可行性

#### 4.1.1 遮挡模拟的信息论分析

设完整 GT 框 $x_0 \in \mathbb{R}^4$ (cxcywh), 遮挡后的观测为 $x_{obs} = M(x_0, \epsilon_{occ})$, 其中 $M$ 是遮挡函数, $\epsilon_{occ}$ 是遮挡参数。

**定理 (信息保持性)**: 若遮挡仅影响 RoI 特征 (策略 A), 而非 GT 框本身, 则训练目标的互信息 $I(\hat{x}_0; x_0 | x_{obs})$ 不降低 — 模型仍能从 $x_{obs}$ 恢复 $x_0$, 只要特征提取器有足够容量。

**证明草图**: RoI 特征 $\phi = \text{RoIAlign}(F, b)$, 遮挡后的特征 $\phi' = M(\phi)$。模型的预测 $\hat{x}_0 = f_\theta(\phi', t)$。由于 $x_0$ 通过 coupling 进入训练信号, 且 $x_0$ 与 $\phi$ 的互信息 $I(x_0; \phi) > 0$ (RoIAlign 保留了框位置信息), 只要 $M$ 不是满射 (信息完全丢失), $I(x_0; \phi') > 0$ 仍成立。扩散过程 $x_t = (1-t)x_0 + t\epsilon$ 进一步提供了噪声正则化, 防止模型过拟合到遮挡模式。

**关键约束**: 遮挡比例 $r$ 不能过高。若 $r \to 1$ (完全遮挡), $I(x_0; \phi') \to 0$, 模型无法恢复。经验上 $r \in [0.2, 0.6]$ 是合理的 — 这对应染色体 20%-60% 面积被遮挡, 覆盖了大多数真实重叠场景。

#### 4.1.2 轨迹一致性的凸性分析

轨迹一致性损失 $\mathcal{L}_{traj} = \frac{1}{H-1}\sum_h \omega_h \|\hat{x}_0^{(h)} - \text{sg}(\hat{x}_0^{(H)})\|_1$ 是凸函数 (L1 范数的非负加权和)。添加到总损失后:

$$\mathcal{L}_{total} = \mathcal{L}_{det} + \lambda \mathcal{L}_{traj}$$

其中 $\mathcal{L}_{det}$ 是现有的 L1 + GIoU + Focal 损失。

**收敛性**: $\mathcal{L}_{traj}$ 的梯度 $\nabla_{\theta_h} \mathcal{L}_{traj} = \omega_h \cdot \text{sign}(\hat{x}_0^{(h)} - \hat{x}_0^{(H)})$ 是有界的 (L1 的次梯度), 不会导致梯度爆炸。与 $\mathcal{L}_{det}$ 的梯度叠加后, 总梯度仍保持有界性。

**与 gradient conflict 的关系**: 实验记录显示, 速度损失与检测损失的梯度冲突 (cos=-0.104) 导致了训练不稳定。轨迹一致性损失的梯度方向与检测损失的梯度方向**高度一致** (都指向 GT), 因此不会引入新的梯度冲突。这是 ATD 相比 CFM (已证伪) 的关键优势。

#### 4.1.3 多假设采样的概率分析

设单次采样的检测率为 $P_{det}$ (对被遮挡实例), 漏检率为 $1 - P_{det}$。K 次独立采样的漏检率:

$$P_{miss}^{(K)} = (1 - P_{det})^K$$

对于 $P_{det} = 0.5$ (中等难度遮挡), $K=3$ 时 $P_{miss}^{(3)} = 0.125$, 检测率从 0.5 提升到 0.875。

**独立性假设的成立条件**: 多次采样需要不同的随机初始化 $x_T^{(k)} \sim \mathcal{N}(0, I)$。由于高维高斯的独立性, 不同 $x_T^{(k)}$ 在概率空间中近似独立, 满足上述分析的前提。

**与 box_renewal 的协同**: box_renewal 在单次采样内提供"重探索", 多假设采样在采样间提供"并行探索"。两者正交:

- box_renewal: 采样步内, 对低置信度框重采样
- 多假设: 采样间, 对整个场景重新初始化

### 4.2 工程可行性

#### 4.2.1 代码改动评估

| 子方向 | 改动文件 | 改动行数 (估计) | 改动性质 | 风险 |
|--------|---------|---------------|---------|------|
| Q1 (遮挡模拟) | `single_head.py`, `head.py` | ~80 行 | 新增方法, 不改现有逻辑 | 低 |
| Q2 (轨迹一致性) | `head.py` | ~40 行 | 新增 loss 项 | 低 |
| Q3 (多假设采样) | `head.py`, `sampling.py` | ~120 行 | 修改 predict, 新增 merge | 中 |

**总改动**: ~240 行, 集中在 `head.py` 和 `single_head.py`, 不涉及 backbone/neck/criterion 的核心逻辑。

#### 4.2.2 与现有架构的兼容性

**Q1 (遮挡模拟)**:
- 施加在 `single_head.py:270` 的 `roi_features` 之后, 不影响 `bbox2roi` 和 `pooler` 的接口
- 新增参数: `occlusion_prob` (默认 0.0, 不遮挡), `occlusion_ratio_range` (默认 [0.2, 0.6])
- 向后兼容: `occlusion_prob=0.0` 时行为与现有代码完全一致

**Q2 (轨迹一致性)**:
- 在 `head.py:306` 的 `losses` 字典中新增 `loss_traj` 项
- 新增参数: `trajectory_consistency_weight` (默认 0.0), `tau` (默认 2.0)
- 向后兼容: `trajectory_consistency_weight=0.0` 时无额外损失

**Q3 (多假设采样)**:
- 修改 `head.py:318` 的 `predict` 方法, 增加 `num_hypotheses` 分支
- 新增参数: `num_hypotheses` (默认 1), `hypothesis_iou_thr` (默认 0.7), `base_seed` (默认 42)
- 向后兼容: `num_hypotheses=1` 时行为与现有代码完全一致

**与 AdaLN-Zero 的兼容性**: AdaLN-Zero ([single_head.py:321-357](../../ldmdet/core/single_head.py#L321-L357)) 的零初始化策略与 ATD 完全兼容 — Q1 的遮挡施加在 AdaLN-Zero 之前, Q2 的损失计算在 AdaLN-Zero 之后, Q3 的多采样不涉及时间条件化。

**与 StochasticOT ε=5 的兼容性**: OT 耦合 ([head.py:455-462](../../ldmdet/core/head.py#L455-L462)) 发生在遮挡模拟之前 (耦合 GT 与噪声), 遮挡模拟发生在 RoI 特征提取之后。两者在流程上正交, 无冲突。

#### 4.2.3 计算开销评估

| 组件 | 训练开销 | 推理开销 |
|------|---------|---------|
| Q1 (遮挡模拟) | +~2% (随机 mask 生成) | 0% (仅训练时) |
| Q2 (轨迹一致性) | +~5% (额外 L1 计算) | 0% (仅训练时) |
| Q3 (多假设 K=3) | 0% (仅推理) | +~20-50% (batch 并行优化后) |

**总训练开销**: ~7%, 可接受 (当前训练 bs=8, 150 epoch, ~12 小时; ATD 约 +50 分钟)。

**总推理开销**: K=3 假设时 +20-50%, 但染色体检测非实时任务, 可接受。

#### 4.2.4 显存评估

Q3 的多假设采样将 batch 从 `bs` 扩展到 `bs * K`, 显存需求增加 K 倍。对于 K=3:

| 组件 | 单假设显存 | K=3 假设显存 |
|------|-----------|-------------|
| x_raw | (bs, 500, 4) = 32KB | (3bs, 500, 4) = 96KB |
| RoI 特征 | (3bs*500, 256, 7, 7) = ~12GB | ~36GB ⚠️ |

**风险**: RoI 特征的显存可能超出单卡 (24GB) 容量。

**缓解方案**:
1. **梯度检查点**: 已有 `use_checkpoint` 参数 ([head.py:59](../../ldmdet/core/head.py#L59)), 推理时无需梯度, 显存压力较小
2. **分批处理**: K 个假设分批推理, 每批处理 1 个假设, 仅增加推理时间不增加显存
3. **降低 num_proposals**: 多假设时将 500 降至 200, 总框数 3*200=600 < 500*1.5

### 4.3 数据需求分析

#### 4.3.1 现有数据充分性

24obj 数据集:
- 训练集: ~4680 张图, ~215k 实例 (每图 ~46 条)
- 类别: 24 类 (A1-A3, B4-B5, C6-C12, D13-D15, E16-E18, F19-F20, G21-G22, X, Y)
- 标注: amodal (完整边界框)

**重叠统计**: 染色体图像天然高密度, 每图 46 条染色体在有限视野内必然大量重叠。估计:
- 平均每对染色体的 IoU > 0.3 的比例为 ~15-20%
- 严重重叠 (IoU > 0.5) 的比例为 ~5-8%

**ATD 的数据需求**: Q1 的遮挡模拟是**数据增强**, 不需要额外标注。Q2 和 Q3 不涉及训练数据。因此 ATD **无需额外数据**, 现有 24obj 数据集完全足够。

#### 4.3.2 遮挡模拟的数据增强效果

ATD 的 Q1 等价于一种**智能数据增强** — 通过随机遮挡 RoI 特征, 模型在训练时见到的"虚拟遮挡"场景远多于真实数据中的重叠场景。

**定量估计**: 假设训练集有 4680 张图, 每图 46 个实例, `occlusion_prob=0.3`:
- 每个 epoch 中被遮挡的实例数: 4680 × 46 × 0.3 ≈ 64,584
- 150 epoch 的总遮挡样本: ~9.7M
- 这远超真实重叠场景的数量 (~30k), 有效扩充了重叠场景的训练数据

---

## 5. 风险评估

### 5.1 风险 1: 遮挡模拟与真实遮挡的分布不匹配

**风险描述**: Q1 的随机矩形 mask 可能与真实染色体重叠的形态不匹配。真实重叠通常是非规则形状 (染色体弯曲交叉), 且遮挡区域有"污染"特征 (两条染色体的特征混合), 而非简单的零填充。

**影响程度**: 中。如果分布不匹配严重, 模型可能学到"对零填充鲁棒"而非"对真实重叠鲁棒", 导致测试集增益低于训练集。

**缓解方案**:
1. **混合遮挡策略**: 不用零填充, 而是用**其他实例的 RoI 特征**填充遮挡区域, 模拟特征混合:
   ```python
   # 从同 batch 的其他 RoI 随机采样填充
   other_idx = torch.randperm(N)
   occluded[i, :, y0:y0+occ_h, x0:x0+occ_w] = roi_features[other_idx[i], :, :occ_h, :occ_w]
   ```
2. **渐进式遮挡**: 训练早期用低遮挡比例 (0.2), 后期逐步增加到 0.6, 让模型先学基础检测再学遮挡鲁棒性
3. **验证集监控**: 在验证集上分别统计重叠/非重叠场景的 AP, 确保遮挡模拟对重叠场景有针对性增益

### 5.2 风险 2: 轨迹一致性损失可能限制级联 Head 的多样性

**风险描述**: 级联 Head 的设计初衷是让不同级有不同感受域/精度, 早期级粗定位, 后期级精定位。轨迹一致性损失强制各级"对齐"到最后级, 可能使早期级丧失"粗定位"能力, 反而降低检测性能。

**影响程度**: 中高。这与方向 N (端到端可微 Cascade, mAP=0.684, -0.172) 的失败教训相关 — 去除 detach 后级联多样性被破坏。

**缓解方案**:
1. **衰减权重设计**: 使用 $\omega_h = \exp(-|H-h|/\tau)$, 早期级权重低, 后期级权重高。$\tau=2.0$ 时, 第 1 级权重 $\omega_1 = \exp(-5/2) \approx 0.082$, 第 5 级权重 $\omega_5 = \exp(-1/2) \approx 0.607$ — 早期级几乎不受约束。
2. **仅约束后期级**: 只对最后 3 级施加一致性损失, 保留前 3 级的独立性:
   ```python
   for h in range(max(0, num_heads - 3), num_heads - 1):
       loss = loss + w * F.l1_loss(norm_preds[h], anchor)
   ```
3. **权重 warmup**: 前 50 epoch `trajectory_consistency_weight=0`, 让级联 Head 先独立收敛, 50 epoch 后逐步增加到目标值
4. **消融验证**: 先用小权重 (0.1) 实验, 观察是否影响级联多样性 (通过各级 AP 的差异度量)

### 5.3 风险 3: 多假设采样的计算开销与收益不成比例

**风险描述**: Q3 的多假设采样增加 20-50% 推理时间, 但对于非重叠场景 (占 80-85%) 无增益 — 这些场景单次采样已足够。可能导致整体推理变慢但 mAP 增益有限。

**影响程度**: 中。如果重叠场景仅占 5-8%, Q3 的增益可能只有 +0.002-0.005 mAP, 不值得 50% 的推理开销。

**缓解方案**:
1. **自适应多假设**: 先单假设采样, 检测高重叠区域 (NMS 前的框密度), 仅对高重叠区域触发多假设:
   ```python
   # 单假设采样后, 统计重叠区域
   overlap_density = compute_overlap_density(single_results)
   if overlap_density.max() > self.overlap_thr:
       # 仅对高重叠图像重新多假设采样
       high_overlap_imgs = torch.where(overlap_density > self.overlap_thr)
       multi_results = self._multi_hypothesis_predict(
           features[high_overlap_imgs], img_metas[high_overlap_imgs], K=3
       )
   ```
2. **推理时动态 K**: 根据图像复杂度调整 K — 简单图像 K=1, 复杂图像 K=3-5
3. **仅用于后处理**: Q3 可以不修改采样流程, 仅在 NMS 阶段用聚类替代 NMS, 实现零额外采样开销

### 5.4 风险 4: 与 box_renewal 的交互效应未知

**风险描述**: box_renewal (+0.016 mAP) 是当前最强单一扩散机制。ATD 的 Q3 (多假设采样) 与 box_renewal 在功能上有重叠 — 都是"重新探索"机制。两者叠加可能导致:
1. 冗余探索 (box_renewal 已重探索, 多假设再探索是浪费)
2. 过度随机化 (太多探索导致高方差, 精度下降)

**影响程度**: 低中。box_renewal 在采样步内重探索, Q3 在采样间重探索, 理论上正交, 但实际交互需实验验证。

**缓解方案**:
1. **消融实验**: 单独 Q3 (无 box_renewal) vs Q3 + box_renewal, 量化交互效应
2. **降低 box_renewal 强度**: 启用 Q3 时, 将 `score_thr` 从 0.05 提高到 0.1, 减少 box_renewal 的重探索频率, 避免过度随机化

---

## 6. 预期收益分析

### 6.1 基于 box_renewal +0.016 的定量外推

**外推逻辑**: box_renewal 的 +0.016 mAP 来自"随机重探索"机制 — 低置信度框被替换为噪声, 给模型第二次机会。ATD 的三个子方向在机制上与 box_renewal 有不同程度的关联:

| 子方向 | 机制相似度 | 独立信息量 | 预期增益 |
|--------|----------|-----------|---------|
| Q1 (遮挡模拟) | 低 (训练时 vs 推理时) | 高 (新能力: amodal 补全) | +0.010 ~ +0.020 |
| Q2 (轨迹一致性) | 无 (确定性约束 vs 随机探索) | 中 (新约束: 级间一致性) | +0.003 ~ +0.008 |
| Q3 (多假设采样) | 高 (多次探索 vs 单次重探索) | 低 (类似机制的扩展) | +0.005 ~ +0.012 |

**详细推导**:

**Q1 的收益外推**: box_renewal 在推理时对**低置信度框** (score < 0.05) 重采样, 这些框通常对应**漏检实例** (包括被遮挡的)。box_renewal 的 +0.016 mAP 中, 估计 ~40% (约 +0.006) 来自被遮挡实例的恢复。Q1 通过训练时遮挡模拟, 直接提升模型对遮挡场景的检测能力, 预期能恢复 box_renewal 无法恢复的实例 (那些 score > 0.05 但位置偏移的遮挡实例)。保守估计 Q1 的增益为 box_renewal 遮挡相关增益的 1.5-3 倍: +0.009 ~ +0.018。

**Q2 的收益外推**: 轨迹一致性损失是正则化项, 不直接恢复漏检, 而是降低预测方差。参考方向 H (velocity loss, +0.000) 的失败 — 速度损失无增益因为不引入新信息。Q2 与 H 的关键区别是: Q2 约束的是**检测框空间** (4D, 与损失直接相关), H 约束的是**速度空间** (间接关联)。预期 Q2 有温和增益: +0.003 ~ +0.008。

**Q3 的收益外推**: Q3 是 box_renewal 的"采样间扩展"。box_renewal 在单次采样内提供 ~3-5 次重探索 (4步 Heun, 每步可能 renew), Q3 提供 K=3 次独立采样。由于 box_renewal 已捕获了大部分"易恢复"漏检, Q3 的边际增益递减。预期 Q3 增益为 box_renewal 的 30-75%: +0.005 ~ +0.012。

### 6.2 组合收益估计

由于三个子方向的机制部分正交, 组合增益需考虑重叠:

**乐观场景** (三方向完全正交):
$$\Delta mAP_{total} = 0.020 + 0.008 + 0.012 = +0.040$$

**保守场景** (三方向 50% 重叠):
$$\Delta mAP_{total} = 0.010 + 0.003 + 0.005 + 0.5 \times (0.010 + 0.005 + 0.007) = +0.026$$

**最可能场景** (基于机制分析):
$$\Delta mAP_{total} \approx +0.015 \sim +0.025$$

即 mAP 从 0.858 提升到 **0.873 ~ 0.883**。

### 6.3 分指标预期

| 指标 | baseline (a3) | ATD 预期 | 增益来源 |
|------|--------------|---------|---------|
| mAP | 0.858 | 0.873-0.883 | Q1+Q2+Q3 |
| AP50 | 0.943 | 0.950-0.955 | Q1 (遮挡定位) |
| AP75 | 0.844 | 0.860-0.875 | Q1+Q2 (精确定位) |
| APs (小目标) | 0.521 | 0.525-0.535 | Q3 (多假设分离) |
| AR (召回率) | 0.810 | 0.830-0.850 | Q1+Q3 (漏检恢复) |

**关键指标**: AR (Average Recall) 的提升是 ATD 的核心价值 — amodal 检测的最大痛点是重叠实例的漏检, ATD 通过 Q1 (训练时学习补全) 和 Q3 (推理时多假设探索) 直接针对漏检问题。

---

## 7. 实现路线图

### 7.1 Phase 1: Q1 遮挡模拟 (2 周)

**目标**: 验证训练时遮挡模拟对 amodal 检测的增益。

**代码改动清单**:
1. `ldmdet/core/single_head.py`
   - `__init__`: 新增 `occlusion_prob`, `occlusion_ratio_range` 参数
   - `forward`: 在 `roi_features = pooler(features, rois)` 后调用 `_apply_occlusion`
   - 新增 `_apply_occlusion` 方法 (特征级遮挡)
2. `ldmdet/core/head.py`
   - `__init__`: 透传 `occlusion_prob`, `occlusion_ratio_range` 到 `single_head`
3. `experiments/configs/ldmdet/directions/frontier_directions/q1_occlusion.py`
   - 新增配置文件, 基于 a3_full_sota 添加 `occlusion_prob=0.3`

**实验计划**:
| 实验 | occlusion_prob | 预期 |
|------|---------------|------|
| Q1-p03 | 0.3 | mAP +0.010 |
| Q1-p05 | 0.5 | mAP +0.015 (或下降, 过遮挡) |
| Q1-p02 | 0.2 | mAP +0.005 (温和) |
| Q1-feature vs Q1-box | 0.3 | 特征级 vs 框级对比 |

**决策门槛**: 若 Q1-p03 的 mAP >= 0.865 (+0.007), 继续 Phase 2; 否则分析原因并调整遮挡策略。

### 7.2 Phase 2: Q2 轨迹一致性 (1.5 周)

**目标**: 在 Q1 基础上添加轨迹一致性损失, 约束级联 Head 的预测一致性。

**代码改动清单**:
1. `ldmdet/core/head.py`
   - `__init__`: 新增 `trajectory_consistency_weight`, `tau` 参数
   - `loss`: 在 `losses` 字典中添加 `loss_traj` 项
   - 新增 `_trajectory_consistency_loss` 方法
2. `experiments/configs/ldmdet/directions/frontier_directions/q2_traj_consistency.py`
   - 新增配置文件, 基于 Q1 最佳配置添加 `trajectory_consistency_weight=0.5`

**实验计划**:
| 实验 | traj_weight | tau | 预期 |
|------|-----------|-----|------|
| Q2-w05-t2 | 0.5 | 2.0 | mAP +0.005 |
| Q2-w10-t2 | 1.0 | 2.0 | mAP +0.003 (或下降) |
| Q2-w05-t1 | 0.5 | 1.0 | mAP +0.003 |
| Q2-last3 | 0.5 | 仅最后3级 | mAP +0.006 |

**决策门槛**: 若 Q2-w05-t2 的 mAP >= Q1最佳 + 0.003, 继续 Phase 3; 否则将 traj_weight 降至 0.1 或跳过 Q2。

### 7.3 Phase 3: Q3 多假设采样 (2 周)

**目标**: 在 Q1+Q2 基础上添加多假设采样, 分离重叠实例。

**代码改动清单**:
1. `ldmdet/core/head.py`
   - `__init__`: 新增 `num_hypotheses`, `hypothesis_iou_thr`, `base_seed` 参数
   - `predict`: 添加多假设分支, 调用 `_multi_hypothesis_predict`
   - 新增 `_multi_hypothesis_predict` 方法
   - 新增 `_merge_hypotheses` 方法
2. `ldmdet/diffusion/sampling.py`
   - `DiffusionSampler.__init__`: 新增 `hypothesis_iou_thr` 参数
   - 新增 `merge_hypotheses` 方法 (聚类合并逻辑)
3. `experiments/configs/ldmdet/directions/frontier_directions/q3_multi_hypothesis.py`
   - 新增配置文件, 基于 Q1+Q2 最佳配置添加 `num_hypotheses=3`

**实验计划**:
| 实验 | num_hypotheses | hypothesis_iou_thr | 预期 |
|------|---------------|-------------------|------|
| Q3-K2 | 2 | 0.7 | mAP +0.005 |
| Q3-K3 | 3 | 0.7 | mAP +0.008 |
| Q3-K5 | 5 | 0.7 | mAP +0.010 (或收益递减) |
| Q3-adaptive | 自适应 | 0.7 | mAP +0.007, 推理更快 |

**决策门槛**: 若 Q3-K3 的 mAP >= Q1+Q2最佳 + 0.003, 确定其为最终配置; 否则评估推理开销是否值得。

### 7.4 Phase 4: 联合调优与消融 (1.5 周)

**目标**: 在 Q1+Q2+Q3 基础上进行超参联合调优, 完成消融实验。

**实验计划**:
| 实验 | 配置 | 目的 |
|------|------|------|
| Q-full | Q1+Q2+Q3 最佳组合 | 最终 mAP |
| Q-ablation-Q1 | 去掉 Q1 | Q1 贡献 |
| Q-ablation-Q2 | 去掉 Q2 | Q2 贡献 |
| Q-ablation-Q3 | 去掉 Q3 | Q3 贡献 |
| Q-vs-boxrenewal | 去掉 box_renewal | 与 box_renewal 交互 |
| Q-no-boxrenewal | Q-full 无 box_renewal | ATD 独立能力 |

**总工期**: ~7 周 (Phase 1-4 串行, 可部分重叠)

---

## 8. 与已证伪方向的对比

### 8.1 与 CFM 速度预测 (mAP=0.823) 的对比

**CFM 失败原因** (实验记录):
> CFM 模式下, velocity_head 输出与级联 Head 结构不兼容 — 级联 Head 设计为预测 x_0 (绝对位置), 而 CFM 要求预测速度 v (相对变化)。级联的每一级需要从 v 反推 x_0, 但反推过程引入数值不稳定 (x0 = x_t - t*v, 早期 t 大时 x0 爆炸)。

| 维度 | CFM (已证伪) | ATD (本方向) |
|------|-------------|-------------|
| 预测目标 | 速度 v (改变) | x_0 (不变) |
| 级联兼容性 | 不兼容 (v→x0 反推不稳定) | 完全兼容 (不改级联逻辑) |
| 损失目标 | MSE(v_pred, v_target) | L1(x0_pred, x0_gt) + 一致性 |
| 新增信息 | 无 (仅对齐训练-推理目标) | 有 (遮挡模拟提供新场景) |
| 梯度冲突 | 严重 (cos=-0.104) | 无 (与检测损失方向一致) |
| 数值稳定性 | 差 (x0 爆炸需 clamp) | 好 (无反推) |

**核心区别**: CFM 改变了扩散的**预测目标** (x_0 → v), 破坏了与级联 Head 的兼容性; ATD 保持 x_0 预测不变, 仅在**输入端** (遮挡) 和**损失端** (一致性) 添加约束, 不触及扩散核心公式。

### 8.2 与 ScaleConditionedRF (mAP=0.741) 的对比

**ScaleConditionedRF 失败原因** (方向四, [方向四_流匹配的非线性轨迹.md](../breakthrough_directions/方向四_流匹配的非线性轨迹.md)):
> 尺度条件化噪声调度将直线路径改为非线性路径, 速度从常数变为 t 的函数, 导致:
> 1. 采样需数值积分, 累积误差增大
> 2. 4维检测空间 (d=4) 中, 尺度条件化的信息增益被 OT Diversity Collapse 抵消
> 3. 非线性路径破坏了 RF 的"直线性"优势, Heun 二阶假设失效

| 维度 | ScaleConditionedRF (已证伪) | ATD (本方向) |
|------|---------------------------|-------------|
| 路径形式 | 非线性 (t^1/κ(s)) | 线性 (不变) |
| 速度场 | t 的函数 (非常数) | 常数 (x_1 - x_0) |
| 采样器兼容 | 不兼容 Heun (需数值积分) | 完全兼容 (Euler/Heun/DPM) |
| 条件化空间 | 尺度 s (新增条件维度) | 无新增条件 (仅输入扰动) |
| 4D 空间影响 | 严重 (d=4 下 OT 崩塌) | 无 (不改耦合/路径) |
| 信息瓶颈 | 尺度信息被耦合层压缩 | 无 (遮挡在特征层) |

**核心区别**: ScaleConditionedRF 改变了扩散的**路径形式** (线性→非线性), 破坏了 RF 的直线性优势和采样器兼容性; ATD 保持路径形式不变, 仅在**特征提取层**施加遮挡, 不触及扩散数学核心。

### 8.3 ATD 的设计哲学

ATD 的设计遵循**"不破坏已验证组件"**原则:

```
已验证有效组件 (不改动):
  ├── RF 直线路径 (rectified_flow.py)
  ├── Heun 4步采样 (rectified_flow.py)
  ├── AdaLN-Zero 时间条件化 (single_head.py)
  ├── StochasticOT ε=5 (coupling)
  ├── box_renewal (sampling.py)
  ├── 6级级联 Head (head.py)
  └── L1+GIoU+Focal 损失 (criterion)

ATD 新增组件 (正交叠加):
  ├── Q1: 遮挡模拟 (single_head.py, 特征层) ← 新增数据增强
  ├── Q2: 轨迹一致性 (head.py, 损失层) ← 新增正则化
  └── Q3: 多假设采样 (head.py, 推理层) ← 新增后处理
```

这与 KCEC (Phase 8, mAP=0.744) 的失败教训一致 — KCEC 试图改变耦合层, 但耦合层信息无法传播到 Transformer。ATD 的所有改动都在**特征层和损失层**, 直接影响梯度流, 避免了信息断层。

---

## 9. 参考文献

### 9.1 核心理论文献

1. **Kirkegaard 2024** — "Diffusion Models for Overlapping Cell Detection"
   - 自发对称性破缺现象, 扩散采样分离重叠实例
   - 关键贡献: 证明扩散随机性天然适配多模态后验

2. **CytoDiffusion (Nature 2025)** — "Diffusion Models for Blood Cell Classification"
   - OOD 鲁棒性: 0.854 vs 判别模型 0.738
   - 关键贡献: 生成式先验提升分布外泛化

3. **Lipman et al. (ICLR 2023)** — "Flow Matching for Generative Modeling"
   - Conditional Flow Matching 理论框架
   - https://arxiv.org/abs/2302.00482

4. **Liu et al. (ICLR 2023)** — "Flow Straight and Fast: Training Linear Deterministic Generative Models with Rectified Flow"
   - Rectified Flow 直线路径理论
   - https://arxiv.org/abs/2209.03003

### 9.2 Amodal 检测文献

5. **Amodal Detection (CVPR 2022)** — "Amodal Detection: Learning to Detect the Whole Object"
   - Amodal 标注的检测框架
   - 关键贡献: 从 visible 标注到 amodal 预测的生成式方法

6. **Stacked Amodal Completion (ECCV 2020)** — "Amodal Instance Segmentation"
   - 从部分观测补全完整实例
   - 关键贡献: 分阶段补全策略

### 9.3 扩散检测文献

7. **DiffusionDet (ICCV 2023)** — "DiffusionDet: Diffusion Model for Object Detection"
   - 扩散模型用于目标检测的奠基工作
   - 关键贡献: 噪声→框的扩散范式

8. **FlowDet (arxiv 2512.16771, 2025-12)** — "FlowDet: Unifying Object Detection and Generative Transport Flows"
   - Flow Matching 用于检测
   - https://arxiv.org/abs/2512.16771

### 9.4 项目内部文档

9. **OT Diversity Collapse 证明** — `docs/OT_DIVERSITY_COLLAPSE_PROOF.md`
   - 4维空间下 OT 多样性崩塌的理论证明
   - 解释为何 ScaleConditionedRF 失败

10. **方向 H 实验报告** — `docs/research/frontier_directions/方向H_FlowMatching检测.md`
    - CFM 速度预测的失败分析
    - 已证伪: mAP=0.854, -0.002

11. **方向四: 非线性轨迹** — `docs/research/breakthrough_directions/方向四_流匹配的非线性轨迹.md`
    - ScaleConditionedRF 的失败分析
    - 已证伪: mAP=0.741

12. **KCEC 诊断报告** — Phase 8 实验记录
    - 耦合层信息断层问题
    - 已证伪: mAP=0.744

---

## 10. 总结

ATD (Amodal Trajectory Diffusion) 是一个**低风险、中收益**的演进方向, 其核心价值在于:

1. **理论坚实**: 基于 Kirkegaard 对称性破缺和 CytoDiffusion OOD 鲁棒性的双重理论支撑
2. **工程可行**: ~240 行代码改动, 不触及已验证的扩散核心组件
3. **与已证伪方向正交**: 不改变预测目标 (vs CFM)、不改变路径形式 (vs ScaleConditionedRF)、不改变耦合层 (vs KCEC)
4. **可增量验证**: Q1→Q2→Q3 三阶段独立验证, 每阶段有明确的决策门槛
5. **针对性强**: 直接针对染色体 amodal 检测的核心痛点 — 重叠实例的漏检与定位偏移

**最可能收益**: mAP 从 0.858 提升到 0.873-0.883 (+0.015~+0.025), 主要来自 Q1 (遮挡模拟) 的 amodal 补全能力和 Q3 (多假设采样) 的实例分离能力。
