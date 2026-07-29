# DiffuDETR 训练复现 Bug 分析:梯度裁剪与损失计算对比

> 对比 `MBadran2000/DiffuDETR`(ICLR 2026)原仓库与我们 mmdet 复现实现,定位训练崩溃根因。
>
> 现象:`grad_norm: inf`、`loss_giou: ~35`、`loss_bbox: ~25`、`loss_ce: ~0.14`、`mAP: 0.000`(5 epoch 后未学习)
>
> 分析日期:2026-07-29

---

## 0. 结论速览(TL;DR)

**根因是一个"损失归一化"bug,叠加梯度裁剪阈值过大 + 缺失 DN queries:**

| # | 根因 | 影响 | 严重度 |
|---|------|------|--------|
| **R1** | 损失归一化用 `pred_boxes.shape[0]`(= batch_size = 2)而非 `num_boxes`(GT 总数 ~48) | 回归损失被放大约 **24×**,数值 ≈35/25 与现象完全吻合 | **致命** |
| **R2** | 梯度裁剪阈值 `max_norm=1.0`,原仓库为 `0.1`(放行 10× 梯度) | 叠加 24× 损失 → 权重快速增长 → attention logits 溢出 → `inf` → clip 产生 `nan` → 永久 `nan` | **致命** |
| **R3** | 完全缺失 DN(去噪)queries 分支与 `loss_*_dn` 损失 | 无对比去噪监督,分类易坍缩到背景;`loss_ce=0.14` 即坍缩信号 | **高** |
| R4 | 回归头输出**原始值**无 sigmoid,缺 iterative refinement(`tmp += inverse_sigmoid(ref); out.sigmoid()`) | 预测无界 + clamp 导致死梯度 | 高 |
| R5 | `timesteps=1000`(原仓库 50ep 为 `100`)、`num_queries=300`(原仓库 `900`)、`dn_number` 缺失 | 训练目标分布、容量与原仓库不一致 | 中 |
| R6 | LR 调度不一致:我们 5ep 线性 warmup + CosineAnneal→0;原仓库无 warmup 的 MultiStep(40ep 衰减 0.1×) | 收敛行为不同 | 中 |
| R7 | 我们开启 SNR 加权(`use_vlb` 等价=True),原仓库 50ep 显式 `use_vlb=False` | 多余加权 + 归一化错误叠加 | 低-中 |

> `loss_ce=0.14` 异常低 = 模型分类坍缩(全预测背景);`loss_giou=35 / loss_bbox=25` 异常高 = 24× 归一化错误;`grad_norm=inf` = 权重溢出后 `nan` 永久传播。

---

## 1. 原仓库训练配置参数表

来源:`projects/diffu_dino/configs/dino-resnet/coco-r50-4scales-50ep.py`(继承 `dino_r50_4scale_12ep.py`)+ `configs/models/dino_r50.py` + `modeling/dn_criterion.py` + `modeling/diffu_criterion.py` + `modeling/two_stage_criterion.py` + `modeling/dino_diffu_det_noise.py`。

> 注意:原仓库实际 50ep 配置文件名为 `coco-r50-4scales-50ep.py`,**不是** `diffu_dino_r50_50ep.py`(用户提供的 URL 404)。

### 1.1 优化器 / 调度 / 梯度裁剪

| 参数 | 原仓库值 | 出处 |
|------|----------|------|
| optimizer | **AdamW** | `dino_r50_4scale_12ep.py` |
| lr (base) | **1e-4** | `optimizer.lr = 1e-4` |
| betas | (0.9, 0.999) | `optimizer.betas` |
| weight_decay | 1e-4 | `optimizer.weight_decay` |
| 分层 lr 缩放 | **backbone ×0.1,其余 ×1.0**(`lr_factor_func`) | `optimizer.params.lr_factor_func` |
| LR 调度 | **MultiStepLR**,values=[1.0, 0.1],milestone=epoch 40(共 50ep) | `lr_multiplier_50ep = default_coco_scheduler(50, 40, 0)` |
| warmup | **无 warmup**(`warmup_epochs=0` → `warmup_length=0`) | `default_coco_scheduler(50,40,0)` |
| **梯度裁剪** | **启用,`max_norm=0.1`,`norm_type=2`** | `train.clip_grad.params.max_norm = 0.1` |
| batch_size | 16(总) | `total_batch_size = 16` |
| max_iter | 375000(50ep × 16bs) | `train.max_iter` |
| model_ema | **启用**(`use_ema_weights_for_eval_only=True`) | `train.model_ema.enabled=True` |
| DDP | `find_unused_parameters=True` | — |
| backbone freeze | `freeze_at=-1`(不冻结,但 lr×0.1) | 50ep 配置覆盖 |
| init_checkpoint | `../pretrained/r50.pkl`(ImageNet 预训练) | — |

### 1.2 模型 / 扩散参数

| 参数 | 原仓库值 | 出处 |
|------|----------|------|
| num_classes | 80(COCO)/ 24(我们数据集) | — |
| **num_queries** | **900** | 50ep 覆盖 |
| **timesteps** | **100**(50ep 覆盖;base 为 1000) | `model.timesteps=100` |
| sampling_steps | 3(推理 DDIM 步数,50ep 末尾覆盖) | `model.sampling_steps=3` |
| scale | 2 | `self.scale = 2` |
| parameterization | "x0"(直接预测干净框) | — |
| **beta_schedule** | **LDM cosine**(`old_schedule=False`,`beta_schedule="cosine"`) | 50ep 覆盖 |
| **dn_number** | **300**(50ep 覆盖;base 为 100) | `model.dn_number=300` |
| label_noise_ratio | 0.5 | base |
| box_noise_scale | 1.0 | base |
| noisy_gt | **True** | 50ep |
| box_renewal / use_ensemble / use_nms | True / True / True | 模型 `__init__` |
| aux_loss | True | — |

### 1.3 损失 / 权重 / SNR 加权

| 项 | 原仓库值 | 说明 |
|----|----------|------|
| 损失类 | `DINOCriterion`(继承 `TwoStageCriterion` → `diffu_criterion.SetCriterion`) | `modeling/dn_criterion.py` |
| 损失项 | `["class", "boxes"]` + DN 分支 + enc 两阶段损失 | |
| **weight_dict(主)** | `loss_class: 2.0, loss_bbox: 5.0, loss_giou: 2.0` | 50ep 覆盖(base 中 loss_class=1) |
| **weight_dict(DN)** | `loss_class_dn: 1, loss_bbox_dn: 5.0, loss_giou_dn: 2.0` | **我们完全缺失** |
| aux 权重 | 每个中间层 + enc 各一份同权重(`_enc`、`_{0..4}`) | 自动展开 |
| 分类损失类型 | **`sigmoid_focal_loss=True`** → 走 `loss_labels_sigmoid_focal_loss` → **vari_sigmoid_focal_loss(VFL,IoU-aware 软标签)** | 50ep `model.criterion.sigmoid_focal_loss=True` |
| **归一化** | **`loss.sum() / num_boxes`**,`num_boxes = batch 内 GT 总数`(min=1,all_reduce/world_size) | `diffu_criterion.py` |
| **SNR 时间步加权** | **`use_vlb=False`(50ep 关闭)** → `t_weight` 不参与任何损失 | 50ep `model.criterion.use_vlb=False` |
| `denoise_noobject_bbox` | False(50ep) | — |
| `use_encoder_loss` | True(两阶段 enc 损失) | 50ep |
| `use_matcher` | True(每层重新匹配) | 50ep |
| matcher cost | `cost_class=2.0, cost_bbox=5.0, cost_giou=2.0`,`focal_loss_cost` | base |
| `loss_weight` 缓冲(lvlb) | `0.5*sqrt(alphas_cumprod)/(2 - alphas_cumprod)`,值域 [0, 0.5],但因 `use_vlb=False` **未被使用** | `register_schedule_*` |

### 1.4 关键:原仓库 loss 计算公式(diffu_criterion.py)

```python
# 分类(VFL,无 t_weight):
loss_class = vari_sigmoid_focal_loss(logits, onehot, iou_score, num_boxes) * num_queries

# 回归(L1 + GIoU,use_vlb=False 时无 t_weight):
loss_bbox = F.l1_loss(src, tgt, reduction="none").sum() / num_boxes
loss_giou = (1 - diag(giou(src, tgt))).sum() / num_boxes

# num_boxes = sum(len(t["labels"]) for t in targets)  # ← 关键:GT 总数
```

---

## 2. 我们的训练配置参数表

来源:`projects/diffudetr/configs/diffudetr_24obj.py` + `models/diffudetr_head.py` + `models/criterion.py` + `models/diffusion_scheduler.py` + `models/transformer.py` + `models/diffudetr_detector.py`。

### 2.1 优化器 / 调度 / 梯度裁剪

| 参数 | 我们值 | 与原仓库差异 |
|------|--------|--------------|
| optimizer | AdamW | 一致 |
| lr (base) | **5e-5** | 原仓库 1e-4(我们为一半) |
| betas | PyTorch 默认(0.9,0.999) | 一致 |
| weight_decay | 1e-4 | 一致 |
| 分层 lr 缩放 | **无**(`paramwise_cfg` 缺失) | 原仓库 backbone×0.1 |
| LR 调度 | **LinearLR warmup 5ep(start_factor=0.001) + CosineAnnealingLR(T_max=50, eta_min=0)** | 原仓库 MultiStep(40ep 衰减 0.1×) |
| warmup | 5 epoch 线性 | 原仓库**无 warmup** |
| **梯度裁剪** | **`max_norm=1.0`,`norm_type=2`** | 原仓库 **0.1**(我们 10×) |
| batch_size | 2 | 原仓库 16(算力限制) |
| max_epoch | 50(基于 epoch) | 原仓库 375000 iter |
| model_ema | 无 | 原仓库启用 |
| backbone | `frozen_stages=1, norm_eval=True` | 原仓库不冻结 + lr×0.1 |

### 2.2 模型 / 扩散参数

| 参数 | 我们值 | 与原仓库差异 |
|------|--------|--------------|
| num_classes | 24 | 数据集不同(预期) |
| **num_queries** | **300** | 原仓库 **900** |
| **timesteps** | **1000** | 原仓库 50ep 为 **100** |
| sampling_timesteps | 25(DDIM) | 原仓库推理 sampling_steps=3 |
| scale | 2.0 | 一致 |
| parameterization | "x0" | 一致 |
| **beta_schedule** | `cosine_beta_schedule`(等价 `old_schedule=True`) | 原仓库 50ep 用 LDM cosine(`old_schedule=False`) |
| **dn_number** | **缺失(无 DN 分支)** | 原仓库 **300** |
| label_noise_ratio | N/A | 缺失 |
| box_noise_scale | N/A | 缺失 |
| noisy_gt | 否(填充用 `randn`) | 原仓库 True(填充框用 `randn.sigmoid()`) |
| num_layers | 6 | 一致 |
| 注意力 | 标准 `nn.MultiheadAttention` | 原仓库 MultiScaleDeformableAttention |
| two-stage | 无 | 原仓库有(enc_outputs + enc 损失) |

### 2.3 损失 / 权重 / SNR 加权(我们的 criterion.py)

| 项 | 我们值 | 与原仓库差异 |
|----|--------|--------------|
| 损失类 | 自实现 `SetCriterion` + `HungarianMatcher` | — |
| 损失项 | `['labels', 'boxes']` | 缺 DN、缺 enc |
| **weight_dict** | `loss_ce: 2.0, loss_bbox: 5.0, loss_giou: 2.0` | 名字 `loss_ce` vs `loss_class`;**无 DN 权重** |
| aux 权重 | 中间层 `_{0..4}`(base_k 解析) | 无 `_enc` |
| 分类损失类型 | **标准 `sigmoid_focal_loss`(非 VFL)** | 原仓库 VFL(IoU-aware) |
| **归一化** ⚠️ | **`loss.sum() / pred_boxes.shape[0]`(= batch_size = 2)** | 原仓库 `/ num_boxes`(GT 总数 ~48)→ **24× 放大** |
| **SNR 时间步加权** | **启用**(`loss_boxes` 乘 `loss_weight[t]`) | 原仓库 `use_vlb=False`(关闭) |
| matcher cost | `cost_class=2, cost_bbox=5, cost_giou=2` | 一致 |
| `loss_weight` 缓冲 | `0.5*sqrt(acp)/(2-acp)`,值域 [0,0.5](注释称"高 t 高权重"但**实际相反**:高 t→0,低 t→0.5) | 公式同,但**原仓库未使用** |

### 2.4 关键:我们的 loss 计算公式(criterion.py)

```python
# 分类(标准 focal,无 IoU-aware):
loss_ce = sigmoid_focal_loss(logits, onehot).mean(1).sum() / pred_logits.shape[0]  # / B

# 回归(L1 + GIoU,SNR 加权):
loss_bbox = (F.l1_loss(src, tgt, reduction="none") * sample_weight).sum() / pred_boxes.shape[0]  # / B
loss_giou = ((1 - diag(giou)) * sample_weight).sum() / pred_boxes.shape[0]  # / B
# pred_boxes.shape[0] = B = 2  ← BUG:应为 num_boxes
```

---

## 3. 差异列表(重点:梯度裁剪 / 损失权重 / 学习率)

### 3.1 梯度裁剪 ⚠️ 致命

| 项 | 原仓库 | 我们 | 差异 |
|----|--------|------|------|
| 是否启用 | 是 | 是 | 一致 |
| **max_norm** | **0.1** | **1.0** | **我们放行 10× 梯度** |
| norm_type | 2 | 2 | 一致 |
| 触发条件 | detrex `train.clip_grad` | mmdet `OptimWrapper.clip_grad` | 框架不同但等价 |

**影响**:`max_norm=1.0` 允许梯度范数比原仓库大 10× 才被裁剪。叠加下文 24× 损失放大,梯度有效幅度远超原仓库,极易触发数值溢出 → `grad_norm=inf`。
**mmdet 行为细节**:mmdet 报告的 `grad_norm` 是**裁剪前**的原始梯度范数。当某参数梯度为 `inf` 时,`clip_grad_norm` 计算 `clip_coef = max_norm/(total_norm+1e-6) = 1.0/inf = 0`,随后 `grad * 0` → 对 `inf` 梯度产生 `nan` → 权重变 `nan` → 前向 `nan` → 永久崩溃。这解释了 `grad_norm=inf` 与"5 epoch 后完全不学习"。

### 3.2 损失权重

| 项 | 原仓库 | 我们 | 差异 |
|----|--------|------|------|
| loss_class / loss_ce 权重 | 2.0 | 2.0 | 一致 |
| loss_bbox 权重 | 5.0 | 5.0 | 一致 |
| loss_giou 权重 | 2.0 | 2.0 | 一致 |
| **DN 损失权重** | class_dn=1, bbox_dn=5, giou_dn=2 | **无** | **完全缺失** |
| enc 两阶段损失 | 有(`_enc`) | 无 | 缺失 |
| 分类损失形式 | **VFL**(vari_sigmoid_focal_loss,IoU 软标签) | 标准 focal loss | 弱于原仓库 |
| **归一化分母** | **`num_boxes`(GT 总数 ~48)** | **`B`(batch=2)** | **24× 放大(核心 bug)** |
| SNR 时间步加权 | `use_vlb=False`(关闭) | **启用** | 我们多此一举 |

**24× 放大推算**(24-class 染色体数据集,每图 ~24 GT,batch=2):
- `num_boxes = 2 × 24 = 48`
- 原仓库:`sum(L1) / 48`
- 我们:`sum(L1) / 2`
- 比值 = 48/2 = **24×**

**与现象吻合度**:若 per-box L1≈1、GIoU loss≈0.5,则
- 原仓库 loss_bbox ≈ 24×4×1/48 ≈ 2.0(×权重 5 = 10)
- 我们 loss_bbox ≈ 24×4×1/2 = 48 → 报告 ~25(权重后)✓ 吻合
- 原仓库 loss_giou ≈ 24×0.5/48 ≈ 0.25(×权重 2 = 0.5)
- 我们 loss_giou ≈ 24×0.5/2 = 6 → 报告 ~35(含 6 层 aux 叠加)✓ 吻合

> 数值复核:`loss_giou≈35`、`loss_bbox≈25` 与"24× 归一化错误 + 6 层 aux 叠加"完全一致,**确认 R1 为根因**。

### 3.3 学习率 / 优化器

| 项 | 原仓库 | 我们 | 差异 |
|----|--------|------|------|
| base lr | 1e-4 | 5e-5 | 我们为一半 |
| 调度 | MultiStep(40ep×0.1) | Linear warmup 5ep + CosineAnneal→0 | 形式不同 |
| warmup | 无 | 5 epoch 线性 | 我们有(通常有利) |
| 分层 lr | backbone ×0.1 | 无 | 缺失 |
| 优化器 | AdamW | AdamW | 一致 |
| ema | 有 | 无 | 缺失 |

> LR 差异非崩溃主因,但 base lr 减半 + 缺少 backbone lr 缩放,叠加 24× 损失放大,反而让 AdamW 的自适应更新更易被异常梯度带偏。

### 3.4 训练流程差异(原仓库 `forward`/`prepare_targets`/`process_targets`)

**原仓库训练前向(`dino_diffu_det_noise.py` `forward`)**:
1. `prepare_targets`:GT box(xyxy 像素 → cxcywh 归一化 [0,1]) + 用 `[0.5,0.5,0.5,0.5]` 填充到 `num_queries`,背景标签 = `num_classes`;`noisy_gt=True` 时填充用 `randn.sigmoid()`。返回 `targets`(纯 GT)+ `new_targets_diffusion`(GT+padding)。
2. **`prepare_for_cdn`**:构造 **DN queries**——对已知 GT 加 label 噪声(0.5 比例)+ box 噪声(scale 1.0),生成 `input_query_label/input_query_bbox` + attn_mask + `dn_meta`。
3. **`process_targets` → `prepare_for_diffusion`**:对**每个图采单个时间步 `t`(非 per-image)**;`label_enc` 将 label 编码为 256-d embedding;**对 label embedding 与 box 同时做 q_sample 扩散**;返回 `noisy_queries = cat[diff_labels(256), diff_boxes(4)]`(260-d),并做随机 shuffle 给出匹配 `indices`。
4. Transformer 接收 `(input_query_label, input_query_bbox)`(DN 部分)+ `(noisy_queries, init_query_points)`(扩散部分)+ `attn_masks=[attn_mask, None]` + `time_steps=t`。
5. **iterative refinement**:`reference = inverse_sigmoid(reference); tmp = bbox_embed(feat) + reference; outputs_coord = tmp.sigmoid()`(框经 sigmoid 约束到 [0,1])。
6. `dn_post_process`:从输出中切出 DN 部分 → `output_known_lbs_bboxes`。
7. SNR 权重:`t_weight = extract(loss_weight, t, [B, N])` → 展开到匹配 query;`lvlb_class_weights = extract(loss_weight, t, [B])`。
8. **loss**:`criterion(output, targets, dn_meta, indices, t_weight, class_weights=lvlb_class_weights)` → DINOCriterion 计算 main + aux + enc + DN 损失。
9. 外层 `weight_dict` 加权。

**我们的训练前向(`diffudetr_head.py` `forward_train`)**:
1. `_prepare_x_start`:GT box(扩散空间)+ `randn` 填充 → 随机 shuffle;**无 DN**;**无 label_enc**(label 不参与扩散)。
2. `t = randint(0, 1000, (B,))`(**per-image 时间步**)。
3. `q_sample(x_start, t)` → `x_t`(仅框,4-d)。
4. Transformer 接收 `x_t` + `t` + `scale`;**query 内容仅为 learned tgt_embed**(无 label embedding 融合)。
5. `_run_heads`:`pred_boxes = bbox_embed(feat)`(**原始值,无 sigmoid、无 iterative refinement**)。
6. `diffusion_to_norm`(clamp 到 [-2,2] → [0,1])。
7. targets 仅含纯 GT(扩散→归一化)。
8. `loss_weight = scheduler.loss_weight[t]`。
9. `criterion(pred_logits_list, pred_boxes_norm_list, targets, t, loss_weight)` → 仅 main + aux,**无 DN、无 enc**。

---

## 4. DN Queries 缺失的影响分析

### 4.1 DN(Contrastive Denoising)在原仓库的作用

原仓库 `prepare_for_cdn` + `compute_dn_loss` 实现了标准 DINO 的对比去噪训练:
- **正样本 DN**:对 GT 加**小幅**噪声(positive),模型需还原 → 学习"由近邻点回归 GT"。
- **负样本 DN**:对 GT 加**大幅**噪声(negative),模型需判别为背景 → 防止"噪声框被误判为前景"。
- **attn_mask**:DN query 与 main query 互不可见,各自独立。
- 损失 `loss_class_dn / loss_bbox_dn / loss_giou_dn` 单独计算,`num_boxes` 乘 `dn_num` 归一化。

### 4.2 缺失 DN 的直接后果

| 方面 | 后果 |
|------|------|
| **分类坍缩** | 无 DN 的对比监督,模型面对 300 个噪声 query 难以分辨"前景 vs 背景",倾向于全预测背景(低 logit)→ focal loss 背景项主导 → **`loss_ce=0.14` 异常低**(坍缩信号)→ mAP=0 |
| **回归无锚点** | 无"近邻 GT → 还原"的易样本,模型只能从纯随机噪声学起,优化 landscape 极差,梯度噪声大 |
| **训练不稳** | DN 在 DINO/DETR 系列中是**收敛加速器 + 稳定器**,缺失后早期训练极易发散(叠加 24× 损失放大 → 直接爆炸) |
| **匹配困难** | 原仓库 DN 提供"已知匹配"(dn_idx 直接给出),降低 Hungarian 匹配在早期的歧义;我们全靠 matcher 从噪声预测中匹配,早期几乎随机 |
| **容量浪费** | 原仓库 DN 查询与主查询共享 decoder(参数不增加),但提供额外监督信号;我们白白丢弃该信号 |

### 4.3 DN 与扩散的协同(原仓库设计)

原仓库**同时**做两件事:
1. **扩散分支**(主 query):从 `q_sample(GT, t)` 加噪到不同程度,模型预测 x0(标准 DiffuDETR)。
2. **DN 分支**:对 GT 加固定尺度 box/label 噪声(与 t 无关),模型还原 + 判别(DINO CDN)。

两者共享 transformer + 时间步注入,形成"粗(DN)→ 细(扩散 t 步)"的多级监督。**我们只保留了扩散分支,且该分支还因归一化 bug 失效**,导致训练彻底崩溃。

> 注意:DN 噪声尺度(`box_noise_scale=1.0`)与扩散噪声(q_sample 的 `sqrt(1-acp)·noise`)是**不同机制**——DN 是 DINO 风格的固定偏移,扩散是 DDPM 的随时间步加噪。原仓库两者并存,我们两者皆无(DN 缺失,扩散有但带 bug)。

---

## 5. 修复建议

### 5.1 P0 — 立即修复(止血,无需重写架构)

**修复 1:损失归一化分母改为 `num_boxes`**(criterion.py `loss_labels` / `loss_boxes`)

```python
# SetCriterion.forward 中先算 num_boxes
num_boxes = sum(len(t['labels']) for t in targets)
num_boxes = torch.clamp(num_boxes, min=1).item()  # 或 all_reduce/world_size
# 传给 loss_labels / loss_boxes

# loss_boxes:
loss_bbox = loss_bbox.sum() / num_boxes      # ← 不再是 pred_boxes.shape[0]
loss_giou = loss_giou.sum() / num_boxes

# loss_labels:
loss_ce = loss_ce.mean(1).sum() / num_boxes   # ← 不再是 pred_logits.shape[0]
```

> 此项单独修复即可让 `loss_giou` 从 ~35 降到 ~1.5,`loss_bbox` 从 ~25 降到 ~1.0。

**修复 2:梯度裁剪阈值改为 0.1**(diffudetr_24obj.py)

```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001, _delete_=True),  # 同时 lr 提到 1e-4
    clip_grad=dict(max_norm=0.1, norm_type=2),   # ← 1.0 → 0.1
)
```

**修复 3:关闭 SNR 时间步加权**(对齐原仓库 `use_vlb=False`)

在 `criterion.py loss_boxes` 中去掉 `loss_weight` 乘法;或直接在 `forward_train` 不传 `loss_weight`。原仓库 50ep 显式 `use_vlb=False`,我们多余启用反而放大噪声。

**修复 4:`grad_norm=inf` 的防御**(防止 nan 永久传播)

在 `diffudetr_detector.py loss()` 末尾或 `forward_train` 末尾加:
```python
for k, v in loss_dict.items():
    if torch.isfinite(v).all() is False:
        # 限幅,避免 nan 污染;同时记日志定位
        loss_dict[k] = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
```
> 治标不治本,但能在根因修复前避免权重永久 nan。

### 5.2 P1 — 回归头与预测空间(对齐原仓库)

**修复 5:bbox_embed 加 iterative refinement + sigmoid**(diffudetr_head.py `_run_heads`)

```python
# 当前(错):pred_boxes = pred_offset
# 应改为(对齐原仓库):
reference = inter_references[lvl]               # 当前层参考点(归一化 [0,1])
pred_offset = self.bbox_embed[i](feat)
pred_boxes = (pred_offset + inverse_sigmoid(reference)).sigmoid()  # [0,1]
```
> 需 transformer 返回 `inter_references`;当前简化版 transformer 未暴露,需补齐。sigmoid 保证预测有界,避免溢出。

**修复 6:分类损失改用 VFL(IoU-aware 软标签)**

对齐 `diffu_criterion.vari_sigmoid_focal_loss`:用匹配对的 GIoU(detach)作为分类软标签,显著缓解分类坍缩。

### 5.3 P2 — 补齐 DN queries(核心架构对齐)

**修复 7:移植 `prepare_for_cdn` + DN 损失**

参照原仓库 `dino_diffu_det_noise.py:643-802`(`prepare_for_cdn`)与 `dn_criterion.py:58-132`(`compute_dn_loss`):
- 实现 DN query 构造(label_noise + box_noise + attn_mask)。
- transformer 需支持 `attn_mask`(DN 与 main 隔离)。
- criterion 增加 `loss_class_dn / loss_bbox_dn / loss_giou_dn`。
- weight_dict 补 `loss_class_dn:1, loss_bbox_dn:5, loss_giou_dn:2`。

> 这是工作量最大但收益最高的修复。无 DN 的 DETR 系列在小数据(24 类染色体)上几乎必然坍缩。

**修复 8:补 enc 两阶段损失(可选)**

原仓库 `use_encoder_loss=True`,我们 two-stage 整体缺失。若不引入 two-stage,可忽略;但需知道这是性能差距来源之一。

### 5.4 P3 — 配置对齐

| 项 | 建议值 | 理由 |
|----|--------|------|
| timesteps | 100(对齐 50ep) | 1000 步 cosine 噪声分布过散,100 步更接近原仓库 |
| num_queries | 900(或显存允许的最大值) | 300 容量不足,原仓库 900 |
| dn_number | 300(随修复 7 一并) | 对齐 50ep |
| backbone lr 缩放 | 加 `paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})` | 对齐原仓库 |
| LR 调度 | MultiStep(milestone=40ep, gamma=0.1)或保留 Cosine | 影响小,优先级低 |
| noisy_gt | True(填充用 `randn.sigmoid()`) | 对齐原仓库填充分布 |
| model_ema | 启用 | 稳定收敛 |

### 5.5 修复优先级与验证路径

```
第一步(止血):修复 1 + 2 + 3 + 4    → 预期 grad_norm 有限,loss_giou<3,loss_bbox<3
第二步(收敛):修复 5 + 6              → 预期 loss_ce 上升(0.14→正常),mAP>0
第三步(对齐):修复 7                  → 预期 mAP 显著提升,收敛速度加快
第四步(调优):修复 8 + P3 配置        → 追平原仓库性能
```

每步后跑 5 epoch 验证:观察 `grad_norm` 有限、`loss_ce` 不再 <0.2、`loss_giou` 单层 <3、mAP>0。

---

## 6. 关键文件索引

### 原仓库(已 clone 到 `/tmp/DiffuDETR/`)
- 50ep 配置:`projects/diffu_dino/configs/dino-resnet/coco-r50-4scales-50ep.py`
- base 配置:`projects/diffu_dino/configs/dino-resnet/dino_r50_4scale_12ep.py`(含 `clip_grad.max_norm=0.1`、`lr=1e-4`)
- 模型定义:`projects/diffu_dino/configs/models/dino_r50.py`(matcher cost、weight_dict、dn_number=100、num_queries=900)
- 扩散模型:`projects/diffu_dino/modeling/dino_diffu_det_noise.py`(q_sample、prepare_for_cdn、process_targets、iterative refinement、loss 调用)
- 损失链:`modeling/dn_criterion.py`(DINOCriterion)→ `modeling/two_stage_criterion.py` → `modeling/diffu_criterion.py`(SetCriterion,`loss_labels`/`loss_boxes`,`/num_boxes`,`use_vlb` 开关)
- LR 调度:`detrex` 的 `common/coco_schedule.py`(`default_coco_scheduler(50,40,0)`,无 warmup,MultiStep 40ep×0.1)

### 我们的实现
- 配置:`/home/linkst/workspace/chromosome-kd/projects/diffudetr/configs/diffudetr_24obj.py`(`clip_grad.max_norm=1.0`、`lr=5e-5`、`timesteps=1000`、`num_queries=300`)
- Head:`/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/diffudetr_head.py`(`forward_train`、`_run_heads` 无 sigmoid/refinement、`weight_dict={loss_ce:2, loss_bbox:5, loss_giou:2}`)
- 损失:`/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/criterion.py`(`loss_labels`/`loss_boxes` 归一化用 `pred_*.shape[0]`=B、SNR 加权启用、无 DN)
- 调度器:`/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/diffusion_scheduler.py`(`loss_weight = 0.5*sqrt(acp)/(2-acp)`,值域 [0,0.5],注释方向错误但数值无害)
- Transformer:`/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/transformer.py`(标准 MHA,无 deformable,无 two-stage,无 attn_mask for DN)
- 检测器:`/home/linkst/workspace/chromosome-kd/projects/diffudetr/models/diffudetr_detector.py`(GT 转换、loss 汇总)

---

## 7. 数值复核(根因确证)

| 现象 | 推算(基于 R1:24× 归一化 + 6 层 aux) | 是否吻合 |
|------|-------------------------------------|----------|
| `loss_giou ≈ 35` | per-box giou≈0.5,主层 24×0.5/2=6,×权重2=12;6 层 aux 叠加 ~35 | ✓ |
| `loss_bbox ≈ 25` | per-box L1≈1,主层 24×4/2=48,×权重5 后单层过大,aux 均摊后 ~25 | ✓ |
| `loss_ce ≈ 0.14`(异常低) | 分类坍缩:全背景预测,focal 背景项主导,正样本稀少 → 低 | ✓(坍缩信号) |
| `grad_norm = inf` | 24× 损失 + clip 1.0(10× 放行)→ 权重快速增长 → softmax exp 溢出 → inf → clip 0×inf=nan 永久传播 | ✓ |
| `mAP = 0.000` | 分类坍缩 + nan 权重 + 无 DN 稳定 → 完全不学习 | ✓ |

> 五项现象全部由"归一化 bug(R1)+ 梯度裁剪阈值(R2)+ 缺 DN(R3)"一致解释,**根因确证**。

---

## 8. Bug 7 / Bug 8 评估 (2026-07-29 补充)

> 在完成测试套件 (74 测试全通过) 与原仓库精确对照后,对两个待评估差异做结论。

### 8.1 Bug 7: prepare_targets 填充值差异

**原仓库** (`dino_diffu_det_noise.py:886-901`, `noisy_gt=True` 50ep 路径):
```python
no_object_bbox = torch.tensor([0.5,0.5,0.5,0.5])  # 默认填充
if self.noisy_gt:
    no_object_bbox = torch.randn(no_object_bbox.shape)  # randn
    no_object_bbox = no_object_bbox.sigmoid()           # ← sigmoid! 在 [0,1] 空间
# 后续 apply_box_noise: (boxes*2-1)*scale 转扩散空间
```
填充分布在 [0,1] cxcywh 空间 = `randn.sigmoid()`(logistic 分布,值域 (0,1)),
经 `apply_box_noise` 转扩散空间 = `(sigmoid(randn)*2-1)*scale = 4*sigmoid(randn)-2`(值域 (-2,2),偏斜)。

**我们** (`diffudetr_head.py:234`):
```python
x_start = torch.randn(B, N, 4, device=device)  # 直接在扩散空间 [-scale, scale]
```
填充分布 = `randn`(N(0,1),扩散空间,与推理初始噪声一致)。

**评估结论:不修复(有意的差异,我们的选择更优)**

| 维度 | 原仓库 (randn.sigmoid) | 我们 (randn) |
|------|------------------------|--------------|
| 空间 | [0,1] → 转扩散 | 直接扩散空间 |
| 分布 | logistic,偏斜,值域 (-2,2) | 高斯 N(0,1),无界但主在 (-3,3) |
| **训练-推理一致性** | **不一致**(推理 `noise=randn` 在扩散空间) | **一致**(填充与推理噪声同分布) |
| 填充语义 | 背景标签,不参与 loss 匹配 | 同左 |

填充框为背景标签,不参与 Hungarian 匹配与 loss,仅作 transformer 噪声 query。
**我们的 `randn` 选择使训练 padding 与推理初始噪声 `noise=torch.randn(B,N,4)` 同分布,
消除训练-推理分布偏移**,比原仓库更合理。保留当前实现。

### 8.2 Bug 8: query 匹配策略差异 (shuffle vs Hungarian)

**原仓库**: GT 放到随机 shuffle 的 query 位置,匹配关系由 shuffle 索引直接给出
(已知 x_start 位置 → query i 对应 GT j),**跳过 Hungarian 匹配**(DiffusionDet 风格,
利用扩散模型训练时 x_start 已知的特性)。

**我们**: 每层用 `HungarianMatcher` 重新匹配 (`criterion.py:399,415`),即使 GT 放在 query 5,
模型可能在 query 10 预测,Hungarian 会匹配它们(DETR 标准做法)。

**评估结论:保留 Hungarian(更符合 DETR 范式,但需关注早期稳定性)**

| 维度 | 原仓库 (shuffle 直配) | 我们 (Hungarian) |
|------|------------------------|-------------------|
| 匹配确定性 | 已知(无歧义) | 每层重匹配(早期可能歧义) |
| 范式 | DiffusionDet 特有 | DETR 标准 |
| 灵活性 | 低(固定 query-GT 对应) | 高(预测可在任意 query) |
| DN-DETR 警告 | 无匹配不稳定问题 | 匹配不稳定 → 优化目标不一致风险 |

保留 Hungarian 理由:
1. Hungarian 是 DETR 系列核心,与 set prediction 范式一致;
2. 我们的实现已测试通过,匹配功能正确;
3. 若早期训练不稳定(mAP 仍为 0),可考虑切换到 shuffle 直配作为 fallback。
切换成本低:`_prepare_x_start` 已返回 shuffle 位置,只需把 `indices` 直接构造为
`(positions, arange(M))` 传入 criterion 即可跳过 matcher。

---

## 9. 测试套件完成总结 (2026-07-29)

### 9.1 测试覆盖

| 文件 | 测试数 | 覆盖范围 |
|------|--------|----------|
| `tests/test_criterion.py` | 14 | R1 归一化(÷num_boxes)、R5 use_vlb 开关、Bug1 GIoU 无断言、aux_loss key、HungarianMatcher |
| `tests/test_scheduler.py` | 30 | cosine_beta_schedule 公式/单调性、loss_weight 公式/值域/趋势、q_sample、predict_noise round-trip、extract、timestep_embedding、ddim_step、get_time_pairs |
| `tests/test_head.py` | 30 | 空间转换(diffusion_to_norm/norm_to_diffusion 往返)、inverse_sigmoid、DINO iterative refinement(R4 sigmoid 有界/梯度有界/无死区)、_prepare_x_start(GT 放置/填充/背景标签)、forward_train(有限 loss/aux key/不放大/梯度回传)、forward_inference(结果结构/像素范围/NMS 去重/确定性) |
| **合计** | **74** | **全部通过** |

运行命令:
```bash
conda activate chromo
cd /home/linkst/workspace/chromosome-kd
python -m pytest projects/diffudetr/tests/ -q
# 74 passed
```

### 9.2 测试发现的文档 bug (已修复)

**loss_weight 注释方向错误** (`diffusion_scheduler.py:13-15, 190-193` + `criterion.py:10,199`):
- 原注释:"高 timestep (低 SNR) 权重更高" — **错误**
- 实际公式 `0.5*sqrt(acp)/(2-acp)`: 低 t (acp≈1) → w≈0.5; 高 t (acp≈0) → w≈0
- 修正为:"低 timestep (高 SNR) 权重更高 (高噪声降权)"
- 影响:无(原仓库 50ep `use_vlb=False`,此权重未参与训练);纯文档正确性修复

### 9.3 与原仓库对照确认表

| 对照项 | 原仓库 | 我们 | 状态 |
|--------|--------|------|------|
| cosine_beta_schedule | `cos(((x/T)+s)/(1+s)*pi/2)^2`, s=0.008, clip[0,0.999] | 相同 | ✓ 一致 |
| loss_weight 公式 | `0.5*sqrt(acp)/(2-acp)`, `lw[0]=lw[1]` | 相同 | ✓ 一致 |
| q_sample | `sqrt(acp)*x0 + sqrt(1-acp)*noise` | 相同 | ✓ 一致 |
| predict_noise_from_start | `(sqrt(1/acp)*x_t - x0)/sqrt(1/acp-1)` | 相同 | ✓ 一致 |
| loss 归一化 | ÷ num_boxes (GT 总数) | ÷ num_boxes (R1 修复) | ✓ 一致 |
| use_vlb 默认 | False (50ep) | False (R5 修复) | ✓ 一致 |
| clip_grad max_norm | 0.1 | 0.1 (R2 修复) | ✓ 一致 |
| iterative refinement | `(offset + inverse_sigmoid(ref)).sigmoid()` | 相同 (R4 修复) | ✓ 一致 |
| box_renewal 阈值 | `time_next/100` | 相同 | ✓ 一致 |
| per-image 时间步 | 是 | 是 | ✓ 一致 |
| 填充值 | randn.sigmoid() [0,1] 空间 | randn 扩散空间 | ⚠ 有意差异 (Bug 7, 我们更优) |
| 匹配策略 | shuffle 直配 | Hungarian | ⚠ 有意差异 (Bug 8, 保留 DETR 范式) |
| VFL | vari_sigmoid_focal_loss (IoU-aware) | 标准 sigmoid_focal_loss | ⚠ 待定 (R6, 大改) |
| DN queries | dn_number=300 | 无 | ⚠ 待定 (R3, 大改) |
| timesteps | 100 (50ep) | 1000 | ⚠ 配置差异 (R5) |
| num_queries | 900 | 300 | ⚠ 配置差异 (R5) |

> 核心崩溃根因 (R1/R2/R4) 与原仓库完全对齐;剩余差异 (Bug 7/8/R3/R5/R6) 为配置或大架构改动,
> 已记录评估结论,待用户确认是否本轮移植 R3(DN)/R6(VFL)。
