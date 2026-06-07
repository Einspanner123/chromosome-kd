# LDMDetDiT mAP=0 训练失败与噪声框问题分析

> 评价对象: `projects/LDMDetDiT/`
> 现象: 训练过程中 mAP 始终为 0；test/可视化输出中噪声框、低质量框居多。
> 分析方式: 仅基于当前代码实现审计，不依赖当前运行环境。

---

## 一、当前问题的总体判断

当前问题不应首先归因于训练轮数不足、DINOv3 backbone 不适配，或环境未配置。代码中已经存在足以解释现象的推理链路错误和训练目标错位风险。

最直接的结论是：

1. **推理阶段会把 box renewal 产生的随机框或先验框加入最终输出。**
2. **后处理没有按 `score_thr` 过滤低分框。**
3. **`regression_mode='direct'` 的配置注释与真实代码语义不一致；当前 direct 实际是 RF velocity prediction。**
4. **Rectified Flow 训练分布、推理初始化分布、box renewal 分布不一致。**
5. **Deformable attention 的采样位置与 bbox 尺度脱钩，图像特征对框去噪的约束较弱。**

这些问题叠加后，即使模型局部开始学习，评估阶段也会被大量低质量 proposal 淹没，表现为 mAP 长期为 0。

---

## 二、关键代码脉络

### 2.1 配置层：当前使用的是 RF + direct + renewal + ensemble

配置文件 `projects/LDMDetDiT/configs/ldmdet_dit.py` 中的关键设置为：

```python
num_proposals=100
num_heads=3
num_blocks=3
sampling_timesteps=6
diffusion_type='rectified_flow'
solver_type='heun'
box_renewal=True
use_ensemble=True
prediction_mode='x0'
regression_mode='direct'
box_init_mode='spatial_prior'
ot_coupling=True
ot_matcher='sinkhorn'
```

这意味着当前模型不是普通 DETR 检测头，而是：

```text
高斯/先验 box latent
  -> Rectified Flow 采样
  -> DiT head 预测 velocity 或 x0
  -> 多步采样
  -> box renewal
  -> ensemble 拼接
  -> NMS 后输出
```

如果采样链路中任一步将未去噪的框加入结果，最终输出就会显著偏向噪声框。

---

## 三、首要根因：box renewal 后的噪声框被加入 ensemble

### 3.1 当前代码行为

在 `mods/dit_head.py` 的 RF 推理分支中，采样更新后立刻执行 renewal：

```python
if self.box_renewal:
    x_raw = self._apply_box_renewal(x_raw, cls_logits)

if self.regression_mode == 'direct':
    x_raw_normed = self._raw_cxcywh_to_normed_xyxy(x_raw)
    x_raw_img = self._normed_to_img(x_raw_normed, img_metas)
else:
    x_raw_img = self._raw_to_xyxy(x_raw, img_metas)

if self.use_ensemble:
    ensemble_results.append((cls_logits, x_raw_img))
```

这里的顺序有严重问题。

`_apply_box_renewal()` 的语义是：低分 proposal 被替换成新的初始 proposal，让下一步采样继续探索。这些替换框不是当前模型预测结果，而是随机框或 anchor prior。

但当前实现把 renewal 后的 `x_raw` 转成检测框并加入 `ensemble_results`。因此最终输出中会包含大量没有被当前步模型去噪的框。

### 3.2 为什么会产生噪声框居多

`_apply_box_renewal()` 内部逻辑：

```python
scores = torch.sigmoid(cls_logits).max(-1)[0]
keep = scores[i] > self.score_thr
if keep.sum() < self.min_keep:
    _, topk_idx = scores[i].topk(min(self.min_keep, scores.shape[1]))
    keep[topk_idx] = True
num_renew = (~keep).sum()
```

当前 `min_keep=60`，`num_proposals=100`。如果分类头尚未学好，大量 proposal 分数接近背景；此时至少保留 60 个，其余最多 40 个会被 renewal。对于 `box_init_mode='spatial_prior'`，这些低分 proposal 会被替换成随机选取的 anchor prior；否则会被替换成标准高斯噪声。

随后这些替换后的框直接进入 ensemble。于是可视化中看到的不是单纯“模型预测差”，而是**推理代码主动把探索用的初始框输出为检测结果**。

### 3.3 分数与框错位

更严重的是，append 时使用的是 renewal 前的 `cls_logits`：

```python
ensemble_results.append((cls_logits, x_raw_img))
```

但 `x_raw_img` 已经包含 renewal 后的新框。也就是说，一部分框的位置已经换了，分数却仍来自旧框。这会破坏 AP 所依赖的 score-ranking，使 mAP 更容易归零。

---

## 四、第二根因：后处理没有 score threshold

配置里有：

```python
test_cfg=dict(
    use_nms=True,
    score_thr=0.01,
    ...
)
```

但 `DiTDiffusionDetHead._post_process()` 中只做 NMS：

```python
final_scores = all_scores
final_bboxes = all_bboxes
final_labels = all_labels
if self.use_nms:
    keep = batched_nms(final_bboxes, final_scores, final_labels, self.nms_thr)
    final_scores = final_scores[keep]
    final_bboxes = final_bboxes[keep]
    final_labels = final_labels[keep]
```

没有执行：

```python
keep = final_scores > self.score_thr
```

因此低分 proposal 仍然会参与最终结果。NMS 只能去掉高度重叠框，不能清除大量位置分散的噪声框。对于 RF 采样和 ensemble，噪声框位置天然分散，NMS 的清理能力非常有限。

这解释了两个现象：

1. test 可视化中低质量框很多。
2. mAP 为 0，因为正确框即使存在，也可能被大量低质量框和错误 score 排序淹没。

---

## 五、第三根因：direct 模式的真实语义是 velocity prediction

配置中有：

```python
prediction_mode='x0'
regression_mode='direct'  # 注释声称 sigmoid 直接预测(cx,cy,w,h)
```

但 `mods/dit_single_head.py` 中实际实现为：

```python
if self.regression_mode == 'direct':
    return self.reg_head(fc_feature)
```

注释明确写着：

```python
# v-prediction 模式: reg_head 直接输出 velocity v
# v = x_noise - x_start (RF 理论速度)
# 训练时: loss = MSE(v_pred, v_target)
```

因此当前 direct 并不是“直接预测归一化 bbox”，而是：

```text
reg_head 输出 v_pred
x0_raw = x_t - v_pred * t
pred_bbox = raw_to_xyxy(x0_raw)
```

训练时又额外加入：

```python
loss_vel = mse(v_pred, x_noise - x_start)
```

该设计本身可以成立，但有两个现实风险：

1. 配置与注释误导排查。看配置会以为模型在直接预测 bbox，实际它在学速度场。
2. 如果 velocity 没学好，`x0_raw` 会接近当前 noisy state，输出自然接近噪声框。

在 mAP=0 的情况下，应优先检查诊断指标：

```text
diag_v_pred_std
diag_v_target_std
diag_v_mae_cx/cy/w/h
diag_x0_mae_cx/cy/w/h
diag_normed_pred_min/max/std
diag_valid_w_ratio
diag_valid_h_ratio
```

如果 `diag_v_pred_std` 长期远小于 `diag_v_target_std`，说明模型几乎没有学到速度场，输出会停留在噪声附近。

---

## 六、第四根因：RF 训练分布与推理 renewal 分布不一致

训练阶段，RF 噪声为标准高斯：

```python
noise = torch.randn(self.num_proposals, 4, device=device)
x_noisy, _ = self.rf.q_sample(x_start, x_noise=noise, t=t[i : i + 1])
```

推理初始化在 RF 模式下也是标准高斯：

```python
return torch.randn(bs, self.num_proposals, 4, device=device)
```

但 renewal 在 `box_init_mode='spatial_prior'` 下使用 anchor prior：

```python
anchor_raw = (anchors * 2 - 1) * self.snr_scale
x_raw_new[i, ~keep] = anchor_raw[replace_idx]
```

这导致推理中间状态混入训练时没有显式建模的分布。模型训练的是从高斯噪声到 GT 的 RF 路径，但推理中途把部分样本替换成 anchor raw，再继续用同一个速度场去积分。

如果这些 anchor raw 仅作为下一步探索状态，还可以接受；但当前代码又把 renewal 后的 anchor raw 直接加入输出，因此分布错位被进一步放大。

---

## 七、第五根因：Deformable Attention 采样与 bbox 尺度脱钩

当前 `mods/deformable_attn.py` 中采样位置为：

```python
offsets = offsets.tanh() * 0.1
sampling_locations = ref_points_expanded + offsets
```

这意味着每个 proposal 的采样半径是固定的全图归一化尺度，和 bbox 的宽高无关。

对染色体检测，这会带来三个问题：

1. 细长小目标的采样点可能大量落到目标外部。
2. 大目标和小目标使用同一半径，不符合框内特征积分的直觉。
3. `bbox_coords` 只通过 reference center 进入 cross-attention，框宽高没有直接控制采样区域。

文档 `IMPLEMENTATION_MATH_REFACTOR.md` 中提出过 variance-aware sampling，即：

```python
scaled_offsets = sampling_offsets * box_wh.unsqueeze(1)
sample_points = box_centers.unsqueeze(1) + scaled_offsets
```

但当前代码没有落实。这不会单独导致 mAP=0，但会显著削弱模型从图像特征中纠正噪声框的能力。

---

## 八、建议修复方案

### Phase 0：先让评估输出可信

目标：先排除“评估阶段主动输出噪声框”的问题。

#### 方案 0.1：临时关闭 renewal 和 ensemble

在配置中先设置：

```python
box_renewal=False
use_ensemble=False
```

预期现象：

1. test 可视化中的噪声框数量应明显减少。
2. 如果 mAP 仍为 0，但输出框更少、更稳定，说明训练确实未学会；如果 mAP 有非零迹象，说明此前主要被推理链路污染。

这是最小侵入消融，不改代码。

#### 方案 0.2：后处理增加 score threshold

在 `_post_process()` 中，NMS 前加入：

```python
score_keep = final_scores > self.score_thr
final_scores = final_scores[score_keep]
final_bboxes = final_bboxes[score_keep]
final_labels = final_labels[score_keep]
```

如果过滤后没有框，返回空 `DetectionResult` 即可，不应强行保留低分框参与评估。

### Phase 1：修正 renewal 与 ensemble 顺序

正确逻辑应该是：

```text
当前 x_raw
  -> model forward 得到 cls_logits, pred_bboxes, x0_raw
  -> 用当前预测结果加入 ensemble
  -> 用 x0_raw / step 后状态更新 x_raw
  -> renewal 只影响下一步 x_raw
```

也就是说，RF 分支应改为：

```python
cls_logits, pred_bboxes, x0_raw, v_pred = self._forward_at_t(...)

# 先保存当前模型预测，不保存 renewal 后状态
if self.use_ensemble:
    ensemble_results.append((cls_logits, pred_bboxes))

# 再积分到下一步
x_raw = self.rf.heun_step(...) or self.rf.step(...)

# 最后 renewal，仅作为下一步输入
if self.box_renewal:
    x_raw = self._apply_box_renewal(x_raw, cls_logits)
```

如果希望 ensemble 保存每一步积分后的 `x_raw`，也必须在 renewal 之前保存：

```python
x_raw_for_output = x_raw.clone()
...
if self.box_renewal:
    x_raw = self._apply_box_renewal(x_raw, cls_logits)
ensemble_results.append((cls_logits, x_raw_for_output_img))
```

禁止把 renewal 后的新 proposal 当成当前检测结果。

### Phase 2：统一 direct/velocity 语义

当前配置中的注释必须改掉，否则后续实验会持续误判。

推荐改为：

```python
prediction_mode='velocity'
regression_mode='direct'
```

或者至少修改注释：

```python
regression_mode='direct',  # direct 模式下 reg_head 输出 RF velocity v，不是 sigmoid bbox
```

同时建议在代码中加入断言或日志：

```python
if self.diffusion_type == 'rectified_flow' and self.regression_mode == 'direct':
    assert self.prediction_mode in ('x0', 'velocity')
```

更干净的设计是拆成两种模式：

1. `regression_mode='velocity'`: reg_head 输出 `v_pred`
2. `regression_mode='bbox_direct'`: reg_head 输出归一化 bbox 或 raw x0

不要继续让 `direct` 同时承载两套语义。

### Phase 3：让训练目标更容易收敛

当前 RF velocity target 为：

```python
v_target = x_noise - x_start
```

其尺度可能比较大，尤其 `snr_scale=2.0` 下，`x_start` 在 `[-2,2]`，`x_noise` 是标准高斯。建议检查：

```text
diag_v_target_mean/std
diag_v_pred_mean/std
diag_v_mae_*
```

如果 `v_pred_std` 长期接近 0，可尝试：

1. 临时提高 `loss_vel` 权重到 2 或 5。
2. 降低分类 loss 权重，例如 `loss_cls=2.0`。
3. 暂时关闭 aux detection loss，只保留最后 head 的 detection loss 与 velocity loss。
4. 先训练 bbox-only sanity check，确认 `loss_vel` 和 `diag_x0_mae_*` 会下降。

### Phase 4：修复 Deformable Attention 的尺度感知采样

需要让采样偏移与当前 bbox 尺度相关。

一种实现方向：

1. 将 `bbox_coords` 或 `box_wh` 传入 `MultiScaleDeformableAttention.forward()`。
2. 计算归一化宽高：

```python
box_wh = (bbox_coords[..., 2:] - bbox_coords[..., :2]).clamp(min=1e-4)
```

3. offset 改为相对 box wh：

```python
offsets = offsets.tanh()
scaled_offsets = offsets * box_wh[:, :, None, None, None, :] * offset_scale
sampling_locations = ref_points_expanded + scaled_offsets
```

`offset_scale` 可从 0.5 或 1.0 起做消融。

这样每个 proposal 的 cross-attention 更像在框附近做可学习积分，而不是在全图固定半径内乱采样。

### Phase 5：重新审视 matching

当前 matcher：

```python
center_radius=5.0
candidate_topk=12
```

对训练初期较宽松，有助于产生正样本，但也可能让低质量预测参与监督。建议在推理链路修好后再调整：

1. `center_radius`: 5.0 -> 2.5 或 3.0
2. `candidate_topk`: 12 -> 6 或 8
3. 或实现 center radius 退火：前期宽松，后期收紧。

不要在 Phase 0/1 前先改 matcher，否则容易掩盖推理链路 bug。

---

## 九、建议验证顺序

### Step 1：不改代码的快速消融

配置：

```python
box_renewal=False
use_ensemble=False
score_thr=0.05
```

观察：

```text
test 输出框数量
噪声框比例
score 分布
diag_normed_pred_min/max/std
diag_valid_w_ratio / diag_valid_h_ratio
```

如果噪声框显著减少，说明 renewal/ensemble 是主因。

### Step 2：增加后处理阈值

加入 score filtering 后重新评估。

观察：

```text
每张图最终 prediction 数量
最高分框是否靠近 GT
mAP 是否从 0 变成极小非零
```

### Step 3：修正 renewal 顺序

恢复：

```python
box_renewal=True
use_ensemble=True
```

但确保 ensemble 不接收 renewal 后的新 proposal。

观察：

```text
use_ensemble=True 是否优于 False
box_renewal=True 是否优于 False
```

如果打开后再次变差，说明 renewal 策略仍需重新设计。

### Step 4：bbox-only sanity check

临时弱化分类：

```text
loss_cls weight 降低
只看 loss_vel、loss_bbox、loss_giou 和 x0_mae 是否下降
```

如果 bbox 仍不能下降，说明 velocity target 或 cross-attention 提供的信息不足。

### Step 5：尺度感知 deformable attention 消融

对比：

```text
固定 offsets: offsets.tanh() * 0.1
尺度 offsets: offsets.tanh() * box_wh * scale
```

观察训练前几 epoch 的：

```text
loss_vel
loss_bbox
diag_x0_mae_*
valid_w/h ratio
召回率
```

---

## 十、优先级列表

| 优先级 | 问题 | 操作 | 预期影响 |
|---|---|---|---|
| P0 | renewal 后噪声框进入 ensemble | 先关闭 `box_renewal/use_ensemble`，再修顺序 | 立刻减少噪声输出 |
| P0 | 无 score filtering | `_post_process()` 加 `score_thr` | 减少低分框污染评估 |
| P1 | direct 语义混乱 | 改注释/命名，明确 velocity 模式 | 降低排查误判 |
| P1 | score 与 renewal 框错位 | ensemble 保存 renewal 前预测 | 修复 AP 排序 |
| P2 | RF 分布不一致 | renewal 只作为下一步输入，或 RF 模式禁用 spatial prior renewal | 提升采样一致性 |
| P2 | deformable 采样不随框尺度 | offsets 乘 bbox wh | 增强图像特征约束 |
| P3 | matcher 过宽 | center_radius/topk 消融或退火 | 改善正样本质量 |

---

## 十一、当前建议的最小代码修复集

第一批只做三件事：

1. `_post_process()` 加 score threshold。
2. RF 推理中，ensemble append 移到 renewal 之前，或 append 当前 `pred_bboxes` 而不是 renewal 后 `x_raw_img`。
3. 修改配置注释，明确 `direct` 是 velocity prediction。

第一批不建议同时修改：

1. backbone
2. neck
3. matcher
4. RoPE
5. optimizer

原因是当前最明显的 bug 在推理输出链路。先让评估结果可信，再判断训练本身是否失败。

---

## 十二、最终结论

当前 mAP=0 与 test 噪声框居多，有明确代码级解释：

```text
分类头未稳定
  -> 大量低分 proposal 触发 renewal
  -> renewal 生成随机/anchor 框
  -> 当前代码把 renewal 后框加入 ensemble
  -> 后处理不做 score threshold
  -> 大量低质量框进入评估和可视化
  -> score 与 bbox 还可能错位
  -> mAP 长期为 0
```

因此，当前首要任务不是继续加复杂结构，而是修复评估/推理链路，使输出真正代表模型预测。只有在 renewal、ensemble、score filtering 修复后，mAP 仍为 0，才应进一步集中处理 velocity target、cross-attention 采样和 matcher。
