# DiffuDETR box 坐标空间转换 Bug 分析报告

> 日期: 2026-07-29
> 对比: 原仓库 (MBadran2000/DiffuDETR) vs 我们的复现实现 (projects/diffudetr/)
> 问题: 训练崩溃 `boxes1 x2<y2`, grad_norm=inf, mAP=0.000, loss_giou≈35

---

## 1. 原仓库的完整 box 坐标链

### 1.1 数据流概览

```
GT (xyxy 像素)
  → prepare_targets: /image_size + xyxy→cxcywh  →  [0,1] cxcywh  (用于 loss target)
  → apply_box_noise: [0,1]→[-2,2]→q_sample→clamp→[-2,2]→[0,1]  →  [0,1] cxcywh  (用于 transformer 输入)
  → transformer (DINO refinement: inverse_sigmoid + sigmoid)     →  [0,1] cxcywh  (预测输出)
  → criterion: pred [0,1] vs target [0,1]                         →  L1 + GIoU
```

### 1.2 逐步详解

#### 步骤 1: GT 准备 — `prepare_targets` (dino_diffu_det_noise.py:881)

```python
gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy  # xyxy 归一化 [0,1]
gt_boxes = box_xyxy_to_cxcywh(gt_boxes)                          # cxcywh 归一化 [0,1]
# new_targets (用于 loss): boxes 在 [0,1] cxcywh
# new_targets_diffusion (用于扩散): GT + 填充 [0.5,0.5,0.5,0.5], 仍在 [0,1] cxcywh
```

**关键: loss target 保持在 [0,1] cxcywh, 不进入扩散空间。**

#### 步骤 2: 扩散加噪 — `apply_box_noise` (dino_diffu_det_noise.py:397)

```python
def apply_box_noise(self, boxes, t):  # boxes 输入 [0,1]
    noise = torch.randn_like(boxes)
    gt_boxes = (boxes * 2. - 1.) * self.scale       # [0,1] → [-2,2] (扩散空间)
    x = self.q_sample(x_start=gt_boxes, t=t, noise=noise)  # x_t 在 [-2,2]
    x = torch.clamp(x, min=-2, max=2)                 # clamp 到 [-2,2]
    x = ((x / self.scale) + 1) / 2.                  # [-2,2] → [0,1] (转回归一化!)
    return diff_boxes, noise  # diff_boxes 在 [0,1], noise 是原始 randn
```

**关键: `apply_box_noise` 的输入和输出都在 [0,1] cxcywh。扩散空间 [-2,2] 仅在函数内部短暂使用。**

#### 步骤 3: Transformer 输入

```python
# forward (line 493, 512):
noisy_queries, init_query_points, ... = self.process_targets(targets, new_targets_diffusion)
query_embeds_diffusion = (noisy_queries, init_query_points)
# init_query_points = diff_boxes, 在 [0,1] cxcywh
```

**Transformer 接收的 reference points 在 [0,1] 归一化空间, 不在扩散空间。**

#### 步骤 4: 预测输出 — DINO 迭代精修 (dino_diffu_det_noise.py:534-549)

```python
for lvl in range(inter_states.shape[0]):
    reference = init_reference if lvl == 0 else inter_references[lvl - 1]  # [0,1]
    reference = inverse_sigmoid(reference)          # [0,1] → inverse sigmoid 空间
    tmp = self.bbox_embed[lvl](inter_states[lvl])   # MLP 预测 offset
    tmp += reference                                 # DINO 迭代精修: offset + inverse_sigmoid(ref)
    outputs_coord = tmp.sigmoid()                    # sigmoid → 保证输出在 (0,1)
```

**关键: 原仓库使用 DINO 的迭代精修机制 — `sigmoid(bbox_embed(feat) + inverse_sigmoid(reference))`。sigmoid 保证输出始终在 (0,1), 且梯度有界 (最大 0.25)。**

#### 步骤 5: Loss 计算 — `diffu_criterion.py`

```python
# criterion.forward (line 363):
indices = self.matcher(outputs_without_aux, targets)  # Hungarian 匹配

# num_boxes 归一化 (line 370):
num_boxes = sum(len(t["labels"]) for t in targets)  # 总 GT 数量 (跨 batch)

# loss_boxes (line 257):
src_boxes = outputs["pred_boxes"][idx]               # [0,1] (sigmoid 输出)
target_boxes = torch.cat([t["boxes"][i] for ...])    # [0,1] (from prepare_targets)
loss_bbox = F.l1_loss(src_boxes, target_boxes, ...)   # L1 在 [0,1] 空间
loss_giou = 1 - torch.diag(generalized_box_iou(
    box_cxcywh_to_xyxy(src_boxes),                    # [0,1] → xyxy [0,1]
    box_cxcywh_to_xyxy(target_boxes)
))
# SNR 加权
loss_bbox = loss_bbox * t_weight.unsqueeze(1).repeat(1, 4)
loss_giou = loss_giou * t_weight
losses["loss_bbox"] = loss_bbox.sum() / num_boxes     # ÷ num_boxes (总 GT 数)
losses["loss_giou"] = loss_giou.sum() / num_boxes     # ÷ num_boxes (总 GT 数)
```

### 1.3 原仓库关键设计总结

| 环节 | 空间 | 说明 |
|------|------|------|
| GT for loss | [0,1] cxcywh | `prepare_targets` 不进入扩散空间 |
| GT for diffusion | [0,1] → [-2,2] → [0,1] | `apply_box_noise` 内部转换, 输入输出都是 [0,1] |
| Transformer 输入 | [0,1] cxcywh | reference points 在归一化空间 |
| bbox_embed 输出 | inverse_sigmoid 空间 | 预测的是 offset, 不是 x0 |
| 最终预测 | [0,1] cxcywh | `sigmoid(offset + inverse_sigmoid(ref))` 保证 |
| L1 loss | [0,1] cxcywh | pred 和 target 都在 [0,1] |
| GIoU loss | xyxy [0,1] | `box_cxcywh_to_xyxy` 转换后计算 |
| 归一化 | ÷ num_boxes | 总 GT 数量 (典型 ~50) |

---

## 2. 我们的实现的 box 坐标链

### 2.1 数据流概览

```
GT (xyxy 像素)
  → detector: /image_size + xyxy→cxcywh + (x*2-1)*scale  →  [-2,2] cxcywh  (扩散空间)
  → _prepare_x_start: GT[-2,2] + randn 填充                →  [-2,2] cxcywh  (x_start)
  → q_sample(x_start, t)                                   →  [-2,2] cxcywh  (x_t)
  → transformer(x_t): 内部 clamp→[0,1] 仅用于位置编码       →  特征
  → bbox_embed(feat) = 直接预测 x0                          →  任意值 (无 sigmoid!)
  → diffusion_to_norm: clamp[-2,2]→[0,1]                    →  [0,1] cxcywh  (pred)
  → diffusion_to_norm(GT[-2,2])                             →  [0,1] cxcywh  (target)
  → criterion: pred [0,1] vs target [0,1]                  →  L1 + GIoU (÷ batch_size!)
```

### 2.2 逐步详解

#### 步骤 1: GT 准备 — `diffudetr_detector.py:loss()`

```python
gt_cxcywh = box_xyxy_to_cxcywh(gt_bboxes)                    # cxcywh 像素
gt_cxcywh_norm = gt_cxcywh / norm_scale                       # [0,1] cxcywh
gt_diffusion = self.gt_to_diffusion_space(gt_cxcywh_norm, scale)  # (x*2-1)*2 → [-2,2]
# 传给 forward_train 的 gt_boxes_list 在 [-2,2] 扩散空间
```

#### 步骤 2: x_start 准备 — `diffudetr_head.py:_prepare_x_start()`

```python
x_start = torch.randn(B, N, 4, device=device)  # 填充: N(0,1) 扩散空间
# GT ([-2,2]) 放到随机位置
x_start[i, positions] = gt_boxes  # GT 在 [-2,2]
# x_start 在 [-2,2] 扩散空间
```

#### 步骤 3: 前向扩散

```python
x_t = self.scheduler.q_sample(x_start, t)  # x_t 在 [-2,2] 扩散空间
```

#### 步骤 4: Transformer — `transformer.py:forward()`

```python
# noisy_boxes (x_t) 在 [-2,2], transformer 内部转换为 [0,1] 仅用于位置编码:
ref = (noisy_boxes.clamp(-s, s) / s + 1.0) / 2.0  # [-2,2] → [0,1] 用于 sine_embed
query_pos = self.ref_point_head(get_sine_pos_embed(ref, ...))
# 但 transformer 不做 box 预测, 只输出特征
```

#### 步骤 5: 预测 — `diffudetr_head.py:_run_heads()`

```python
pred_offset = self.bbox_embed[i](feat)   # MLP 直接输出, 无 sigmoid!
pred_boxes = pred_offset                  # 被当作扩散空间的 x0
# pred_boxes 是无约束的 MLP 输出, 可以为任意值
```

#### 步骤 6: 空间转换 — `diffudetr_head.py:forward_train()`

```python
pred_boxes_norm_list = [self.diffusion_to_norm(pb) for pb in pred_boxes_list]
# diffusion_to_norm(x) = (x.clamp(-2,2) / 2 + 1) / 2  →  [0,1]

gt_boxes_norm = self.diffusion_to_norm(gt_boxes_diff)  # [-2,2] → [0,1]
```

#### 步骤 7: Loss — `criterion.py`

```python
# HungarianMatcher.forward (line 169):
cost_giou = -generalized_box_iou(
    box_cxcywh_to_xyxy(out_bbox),    # pred [0,1] → xyxy
    box_cxcywh_to_xyxy(tgt_bbox)     # target [0,1] → xyxy
)

# loss_boxes (line 331-332):
loss_bbox = loss_bbox.sum() / pred_boxes.shape[0]   # ÷ B (batch_size!) ← BUG!
loss_giou = loss_giou.sum() / pred_boxes.shape[0]   # ÷ B (batch_size!) ← BUG!

# loss_labels (line 280):
loss_ce = loss_ce.mean(1).sum() / pred_logits.shape[0]  # ÷ B ← BUG!
```

### 2.3 我们的关键设计总结

| 环节 | 空间 | 说明 |
|------|------|------|
| GT for loss | [-2,2] → [0,1] | detector 先转 [-2,2], head 再转回 [0,1] (多一次往返) |
| GT for diffusion | [-2,2] | 直接在扩散空间, x_start = GT[-2,2] + randn 填充 |
| Transformer 输入 | [-2,2] cxcywh | x_t 在扩散空间 (内部转 [0,1] 仅用于位置编码) |
| bbox_embed 输出 | 任意值 (无约束) | 直接预测 x0, 无 sigmoid, 无 inverse_sigmoid |
| 最终预测 | clamp→[0,1] | `diffusion_to_norm` 用 clamp, 超范围梯度为 0 |
| L1 loss | [0,1] cxcywh | pred 和 target 都在 [0,1] |
| GIoU loss | xyxy [0,1] | `box_cxcywh_to_xyxy` 转换后计算 |
| 归一化 | ÷ batch_size B | **错误!** 应为 ÷ num_boxes (总 GT 数) |

---

## 3. 差异点和 Bug 分析

### Bug 1 (直接致命): Loss 归一化除以 batch_size 而非 num_boxes

**严重程度: 致命 — 直接导致梯度爆炸和训练崩溃**

| | 原仓库 | 我们的实现 |
|---|---|---|
| 分类 loss 归一化 | `÷ num_boxes` (总 GT 数 ≈50) | `÷ pred_logits.shape[0]` (= B ≈2) |
| 回归 loss 归一化 | `÷ num_boxes` | `÷ pred_boxes.shape[0]` (= B) |
| GIoU loss 归一化 | `÷ num_boxes` | `÷ pred_boxes.shape[0]` (= B) |

**影响**: 所有 loss 被放大 ~25 倍 (num_boxes/B ≈ 50/2)。

- 正常 `loss_giou` ≈ 1.5 → 我们的 ≈ 37.5 (与报告的 ≈35 吻合!)
- `weight_dict` 中 `loss_giou: 2.0` → 加权后 ≈ 75
- 梯度爆炸 → `grad_norm = inf`
- 权重更新后变为 NaN/Inf
- 下一次前向传播: `pred_boxes` 含 NaN
- `diffusion_to_norm(NaN)` = NaN (clamp 对 NaN 无效)
- `box_cxcywh_to_xyxy(NaN)` = NaN
- `NaN >= NaN` = False → **断言 `boxes1 x2<y2` 失败!**

**代码位置**:
- `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/criterion.py:280` — `loss_ce` ÷ `pred_logits.shape[0]`
- `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/criterion.py:331` — `loss_bbox` ÷ `pred_boxes.shape[0]`
- `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/criterion.py:332` — `loss_giou` ÷ `pred_boxes.shape[0]`

原仓库对应: `/tmp/diffu_criterion.py:273,297` — `loss_bbox.sum() / num_boxes`, `loss_giou.sum() / num_boxes`

### Bug 2 (深层架构): 直接预测 x0 vs DINO 迭代精修

**严重程度: 高 — 即使修复归一化也会导致训练不稳定**

**原仓库** (DINO 迭代精修):
```python
reference = inverse_sigmoid(reference)        # [0,1] → inverse sigmoid
tmp = self.bbox_embed[lvl](inter_states)      # MLP 预测 offset
tmp += reference                               # offset + inverse_sigmoid(ref)
outputs_coord = tmp.sigmoid()                 # sigmoid → 保证 (0,1), 梯度有界
```

**我们的实现** (直接 x0 预测):
```python
pred_offset = self.bbox_embed[i](feat)       # MLP 直接输出, 无约束
pred_boxes = pred_offset                       # 当作扩散空间 x0
# 后续 diffusion_to_norm 用 clamp 映射到 [0,1]
```

**差异影响**:

| 方面 | 原仓库 (sigmoid) | 我们的实现 (clamp) |
|------|-------------------|---------------------|
| 输出范围 | (0, 1) 自然有界 | 任意值, 靠 clamp 截断 |
| 梯度 | sigmoid 梯度最大 0.25, 始终有梯度 | clamp 在 [-2,2] 范围外梯度为 **0** |
| 训练稳定性 | 稳定, 梯度有界 | 不稳定, 容易出现死神经元 |
| 参考点一致性 | reference 和 output 都在 [0,1] | reference 在 [0,1], output 在扩散空间 |

当 MLP 输出超出 [-2,2] 时, `clamp` 的梯度为 0, 导致梯度无法回传, 模型无法自我纠正。这在训练初期 (bbox_embed 初始化为 0) 不明显, 但随着训练进行会逐渐恶化。

**代码位置**:
- `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/diffudetr_head.py:263-266` — `_run_heads` 中直接用 MLP 输出作为 pred_boxes

### Bug 3 (架构差异): Transformer 接收的 box 空间不同

**严重程度: 中 — 影响模型语义一致性**

| | 原仓库 | 我们的实现 |
|---|---|---|
| Transformer 输入 | [0,1] cxcywh (reference points) | [-2,2] cxcywh (x_t 扩散空间) |
| 内部转换 | 不需要 (直接用 [0,1]) | `clamp→[0,1]` 仅用于 sine 位置编码 |
| 语义 | reference points 是归一化坐标 | x_t 是扩散噪声框, 语义不同 |

原仓库的 transformer 在 [0,1] 空间操作, reference points 直接用于 deformable attention 采样。我们的 transformer 接收 [-2,2] 的 x_t, 内部转 [0,1] 仅用于位置编码, 但 box 本身的语义是扩散噪声而非归一化坐标。

### Bug 4 (次要): GT 坐标的冗余往返转换

**严重程度: 低 — 不直接导致 bug, 但增加了数值误差和代码复杂度**

原仓库: GT 保持 [0,1], 扩散仅在 `apply_box_noise` 内部往返。
我们的实现: GT 先 `detector` 转 [-2,2], 再在 `head` 转回 [0,1]。

```
原仓库:  GT[0,1] → apply_box_noise([0,1]→[-2,2]→[0,1]) → transformer[0,1] → sigmoid → [0,1]
我们:    GT[0,1] → detector→[-2,2] → head→diffusion_to_norm→[0,1] → transformer[0,1] → clamp → [0,1]
```

### Bug 5 (次要): SNR loss_weight 方向验证

原仓库的 `lvlb_weights` 公式 (两个实现一致):
```python
lvlb_weights = 0.5 * sqrt(alphas_cumprod) / (2.0 - alphas_cumprod)
```

注意: `2. * 1 - alphas_cumprod` = `2.0 - alphas_cumprod` (Python 运算符优先级, 不是 `2*(1-)`)

- t=0: alphas_cumprod≈1, weight ≈ 0.5 (高 SNR, 低噪声 → 高权重)
- t=999: alphas_cumprod≈0, weight ≈ 0 (低 SNR, 高噪声 → 低权重)

这意味着高 timestep (噪声大) 的 loss 被降权。这在两个实现中是一致的, 不是 bug。

---

## 4. 修复建议

### 4.1 修复 1 (必须立即修复): Loss 归一化

**文件**: `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/criterion.py`

在 `SetCriterion.forward` 中计算 `num_boxes` 并传递给各 loss 函数:

```python
def forward(self, pred_logits_list, pred_boxes_list, targets, time_steps, loss_weight):
    # 计算 num_boxes (对齐原仓库)
    num_boxes = sum(len(t['labels']) for t in targets)
    num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                device=pred_logits_list[-1].device)
    num_boxes = torch.clamp(num_boxes, min=1).item()
    ...
```

在 `loss_boxes` 中:
```python
# 修改前 (BUG):
loss_bbox = loss_bbox.sum() / pred_boxes.shape[0]  # ÷ B
loss_giou = loss_giou.sum() / pred_boxes.shape[0]  # ÷ B

# 修改后:
loss_bbox = loss_bbox.sum() / num_boxes  # ÷ num_boxes (总 GT 数)
loss_giou = loss_giou.sum() / num_boxes
```

在 `loss_labels` 中:
```python
# 修改前 (BUG):
loss_ce = loss_ce.mean(1).sum() / pred_logits.shape[0]  # ÷ B

# 修改后:
loss_ce = loss_ce.mean(1).sum() / num_boxes  # ÷ num_boxes
```

需要将 `num_boxes` 作为参数传递给 `loss_boxes` 和 `loss_labels` (对齐原仓库 `get_loss` 签名)。

### 4.2 修复 2 (强烈建议): 改用 DINO 迭代精修

**文件**: `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/diffudetr_head.py`

将 `_run_heads` 改为 DINO 风格的迭代精修:

```python
def _run_heads(self, inter_states, time_emb, init_reference):
    """DINO 迭代精修: sigmoid(bbox_embed(feat) + inverse_sigmoid(ref))."""
    pred_logits_list = []
    pred_boxes_list = []
    reference = init_reference  # [B, N, 4] in [0,1]

    for i in range(self.num_layers):
        feat = inter_states[i]
        # 分类
        feat_cls = self.class_time_embed[i](feat, time_emb)
        pred_logits = self.class_embed[i](feat_cls)
        # 回归: DINO 迭代精修
        pred_offset = self.bbox_embed[i](feat)
        # sigmoid(offset + inverse_sigmoid(reference)) → 保证 [0,1]
        pred_boxes = (pred_offset + inverse_sigmoid(reference)).sigmoid()
        pred_logits_list.append(pred_logits)
        pred_boxes_list.append(pred_boxes)
        reference = pred_boxes  # 下一层以当前预测为 reference
    return pred_logits_list, pred_boxes_list
```

需要同时修改 `forward_train`:
- 将 x_t 从 [-2,2] 转换到 [0,1] 后传入 transformer (对齐原仓库 `apply_box_noise` 的输出)
- 不再需要 `diffusion_to_norm` 转换 pred_boxes (sigmoid 已保证 [0,1])
- GT target 保持 [0,1] (在 detector 中不转扩散空间, 或转后再转回)

### 4.3 修复 3 (建议): 对齐 transformer 输入空间

**文件**: `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/transformer.py` 和 `diffudetr_head.py`

将 transformer 接收的 box 从 [-2,2] 改为 [0,1]:

```python
# forward_train 中:
x_t_diffusion = self.scheduler.q_sample(x_start, t)  # [-2,2]
# 转换为 [0,1] 传入 transformer (对齐原仓库 apply_box_noise 输出)
x_t_norm = self.diffusion_to_norm(x_t_diffusion)  # [0,1]
inter_states = self.transformer(multi_level_feats, x_t_norm, t, scale=self.scale)
```

Transformer 中移除内部的 `diffusion_to_norm` 调用 (因为输入已经是 [0,1])。

### 4.4 修复 4 (建议): 简化 GT 坐标链

**文件**: `/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/diffudetr_detector.py`

在 detector 中保持 GT 为 [0,1] (不转扩散空间), 在 head 内部按原仓库方式处理:

```python
# detector.loss() 中:
gt_cxcywh_norm = gt_cxcywh / norm_scale  # [0,1]
# 不再转为扩散空间, 直接传 [0,1]
gt_boxes_list.append(gt_cxcywh_norm)

# head.forward_train() 中:
# 在 _prepare_x_start 内部将 [0,1] → [-2,2] (对齐 apply_box_noise)
# targets 直接用 [0,1] (不需要 diffusion_to_norm 转换)
```

### 4.5 修复优先级

| 优先级 | 修复项 | 预期效果 |
|--------|--------|----------|
| P0 (必须) | 4.1 Loss 归一化 ÷num_boxes | 消除梯度爆炸, loss_giou 从 ~35 降到 ~1.5 |
| P1 (强烈建议) | 4.2 DINO 迭代精修 (sigmoid) | 保证输出 [0,1], 梯度有界, 训练稳定 |
| P2 (建议) | 4.3 Transformer 输入 [0,1] | 语义一致性, 对齐原仓库 |
| P3 (可选) | 4.4 简化 GT 坐标链 | 减少冗余转换, 降低出错风险 |

### 4.6 快速验证方案

仅修复 P0 (loss 归一化) 后, 可通过以下指标验证:

1. `loss_giou` 应从 ~35 降到 ~1.5
2. `grad_norm` 应从 inf 降到 < 100
3. 断言 `boxes1 x2<y2` 不再触发 (因为不再产生 NaN)
4. mAP 应从 0.000 开始有非零值 (可能很低, 但不再是 0)

若 P0 修复后仍有训练不稳定 (loss 震荡), 再修复 P1 (sigmoid 迭代精修)。

---

## 5. 关键代码位置索引

### 原仓库 (已下载到 /tmp/)

| 文件 | 关键行 | 说明 |
|------|--------|------|
| `/tmp/diffu_criterion.py:273,297` | `loss.sum() / num_boxes` | 正确的 loss 归一化 |
| `/tmp/diffu_criterion.py:363` | `indices = self.matcher(...)` | criterion 内部调用 matcher |
| `/tmp/diffu_criterion.py:370` | `num_boxes = sum(len(t["labels"])...)` | num_boxes 计算 |
| `/tmp/diffu_det_noise.py:397-416` | `apply_box_noise` | [0,1]→[-2,2]→[0,1] 往返转换 |
| `/tmp/diffu_det_noise.py:541-549` | `inverse_sigmoid + sigmoid` | DINO 迭代精修 |
| `/tmp/diffu_det_noise.py:881-910` | `prepare_targets` | GT 保持 [0,1] |

### 我们的实现

| 文件 | 关键行 | 说明 |
|------|--------|------|
| `criterion.py:280` | `÷ pred_logits.shape[0]` | **BUG: loss_ce 归一化** |
| `criterion.py:331` | `÷ pred_boxes.shape[0]` | **BUG: loss_bbox 归一化** |
| `criterion.py:332` | `÷ pred_boxes.shape[0]` | **BUG: loss_giou 归一化** |
| `criterion.py:56` | `assert boxes1 x2<y2` | 断言失败位置 |
| `diffudetr_head.py:263-266` | `pred_boxes = pred_offset` | **缺少 sigmoid/inverse_sigmoid** |
| `diffudetr_head.py:174-181` | `diffusion_to_norm` | clamp 转换 (梯度问题) |
| `diffudetr_head.py:314` | `diffusion_to_norm(pred_boxes)` | pred 转换 |
| `diffudetr_head.py:324` | `diffusion_to_norm(gt_boxes)` | GT 转换 |
| `diffudetr_detector.py:98` | `gt_to_diffusion_space` | GT [0,1]→[-2,2] |
| `transformer.py:508` | `ref = (noisy_boxes.clamp...)/2` | 内部转 [0,1] (仅位置编码) |

---

## 6. 结论

训练崩溃的根本原因是 **loss 归一化错误** (`÷ batch_size` 代替 `÷ num_boxes`), 导致所有 loss 放大约 25 倍, 梯度爆炸 (grad_norm=inf), 权重变为 NaN, 最终 GIoU 断言因 NaN 而失败。

深层架构差异 (直接 x0 预测 vs DINO sigmoid 迭代精修) 虽不是直接崩溃原因, 但会导致 clamp 梯度截断问题, 即使修复归一化后仍可能影响长期训练稳定性。

**建议修复顺序: 先修复 P0 (loss 归一化) 验证崩溃消除, 再修复 P1 (sigmoid 迭代精修) 确保训练稳定收敛。**
