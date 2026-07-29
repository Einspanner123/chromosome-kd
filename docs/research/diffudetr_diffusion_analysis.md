# DiffuDETR 扩散核心逻辑技术报告

> 基于 DiffuDETR 官方仓库 (MBadran2000/DiffuDETR, main 分支) 三个核心文件
> 与 KaryoFlow (`ldmdet/diffusion/`) 的对比分析。
>
 - 源文件 1: `projects/diffu_dino/modeling/dino_diffu_det_noise.py` (1192 行)
 - 源文件 2: `projects/diffu_dino/modeling/diffu_criterion.py` (416 行, 原始 deprecated 版)
 - 源文件 3: `layers_diffu_detr/denoising.py` (269 行)
>
 - 对照文件 A: `ldmdet/diffusion/sampling.py` (KaryoFlow)
 - 对照文件 B: `ldmdet/diffusion/rectified_flow.py` (KaryoFlow)
 - 对照文件 C: `ldmdet/diffusion/noise_schedule.py` (KaryoFlow)

---

## 1. 扩散调度 (dino_diffu_det_noise.py)

### 1.1 cosine_beta_schedule (独立函数, L82-92)

标准 Nichol & Dhariwal (2021) 余弦调度, 与 KaryoFlow `cosine_noise_schedule` 完全一致:

```python
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]   # 归一化使 ac[0]=1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
```

### 1.2 make_beta_schedule (L58-80, LDM 风格)

返回 numpy 数组, 支持 4 种调度: `linear` / `cosine` / `sqrt_linear` / `sqrt`。
`linear` 用 `linspace(start^0.5, end^0.5, T)^2` (平方使两端更平滑)。

### 1.3 register_schedule_old vs register_schedule_ldm

两个调度注册函数均向 module 注册相同的 buffer 集合 (`betas`, `alphas_cumprod`,
`sqrt_alphas_cumprod`, `sqrt_recip_alphas_cumprod`, `posterior_variance`,
`posterior_mean_coef1/2`, `loss_weight` 等), 但实现差异显著:

| 维度 | register_schedule_old (L258-295) | register_schedule_ldm (L297-349) |
|---|---|---|
| beta 来源 | `cosine_beta_schedule` (torch, float64) | `make_beta_schedule` (numpy, 支持 linear/cosine/sqrt) |
| dtype | 全程 torch.float64 | 经 `to_torch = partial(torch.tensor, dtype=float32)` 转 float32 |
| posterior_variance | `betas * (1 - ac_prev) / (1 - ac)` (标准 DDPM) | `(1-v)*betas*(1-ac_prev)/(1-ac) + v*betas` (v_posterior 混合) |
| parameterization | 仅隐式 x0 | 显式支持 `eps` / `x0` (L339-345) |
| lvlb_weights | `0.5 * sqrt(ac) / (2 - ac)` | x0: 同左; eps: `betas^2 / (2 * posterior_var * alphas * (1-ac))` |
| 入口 | `old_schedule=True` (默认) | `old_schedule=False` |

> **关键细节**: `lvlb_weights` 公式 `0.5 * np.sqrt(alphas_cumprod) / (2. * 1 - alphas_cumprod)`
> 中的 `2. * 1` 实为 `2.0` (常量乘 1), 因此分母是 `2 - ac` 而非标准 LVLB 的
> `2 * (1 - ac)`。这是疑似笔误: 标准 x0 参数化 LVLB 权重应为
> `0.5 * sqrt(SNR) / (1 + SNR)` (SNR = ac/(1-ac)), 等价于 `0.5 * sqrt(ac) * sqrt(1-ac)`。
> 当前实现 `0.5 * sqrt(ac) / (2 - ac)` 在 ac→1 时趋 0.5, ac→0 时趋 0,
> 形状与标准 LVLB 相近但数值不同。`lvlb_weights[0] = lvlb_weights[1]` 处理 t=0 奇点。

### 1.4 q_sample / predict_start_from_noise / predict_noise_from_start

三个核心运算 (L351-369), 均通过 `extract(a, t, x_shape)` 按 t 索引并广播:

```python
def q_sample(self, x_start, t, noise=None):
    # 前向加噪: x_t = sqrt(ac_t) * x_0 + sqrt(1-ac_t) * noise
    return sqrt_ac_t * x_start + sqrt_1mac_t * noise

def predict_start_from_noise(self, x_t, t, noise):
    # 反推 x_0: x_0 = (x_t - sqrt(1-ac) * noise) / sqrt(ac)
    return sqrt_recip_ac_t * x_t - sqrt_recipm1_ac_t * noise

def predict_noise_from_start(self, x_t, t, x0):
    # 反推 noise: noise = (x_t - sqrt(ac) * x_0) / sqrt(1-ac)
    return (sqrt_recip_ac_t * x_t - x0) / sqrt_recipm1_ac_t
```

> KaryoFlow `sampling.py` L503-515 的 `predict_noise_from_start` 实现完全相同
> (用 `load_buffer` 替代 `extract`, 逻辑等价)。

### 1.5 parameterization="x0" 的含义

`self.parameterization = "x0"` (L232, L243) 表示模型直接预测 **干净数据 x_0**
(而非噪声 eps)。具体体现:

- bbox_embed 输出经 sigmoid 得到 `outputs_coord` (归一化 [0,1] cxcywh), 即视为 x_0 预测;
- 推理时通过 `predict_noise_from_start(reference_points, t, x_start)` 由 (x_t, x_0) 反推噪声,
  再代入 DDIM 更新公式;
- `loss_weight` (lvlb_weights) 按 x0 参数化公式计算。

### 1.6 扩散空间映射 (self.scale = 2)

`__init__` 中 `self.scale = 2` (L249)。映射函数:

```python
# [0,1] cxcywh → [-scale, scale] 扩散空间 (GT 加噪前)
gt_boxes = (boxes * 2. - 1.) * self.scale      # [0,1] → [-1,1] → [-2,2]

# 扩散空间 → [0,1] (q_sample 后还原)
x = torch.clamp(x, min=-1*self.scale, max=self.scale)
x = ((x / self.scale) + 1) / 2.                # [-2,2] → [-1,1] → [0,1]
```

> **与 KaryoFlow 一致**: `sampling.py` `xyxy_to_raw` 用 `(x0 * 2 - 1) * self.snr_scale`
> (snr_scale 默认 2.0), `raw_to_xyxy` 用 `((raw/snr_scale)+1)/2`。差异仅在坐标格式:
> DiffuDETR 全程 cxcywh; KaryoFlow raw 空间用 cxcywh, 图像空间用 xyxy。

---

## 2. 训练流程

### 2.1 prepare_for_diffusion 完整逻辑 (L1134-1173)

```python
def prepare_for_diffusion(self, gt_boxes, labels, num_gt, is_train):
    # (1) 时间步采样: shape=(1,), 即每张图一个 t, 图内所有 query 共享
    t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()

    # (2) 拼接 label_embed 与 box 为 query
    queries = torch.cat([labels, gt_boxes], dim=1)         # [N, embed+4]

    # (3) 标签噪声 (独立 randn)
    noise_labels = torch.randn(labels.shape, device=self.device)

    # (4) box 加噪: apply_box_noise 内部完成 [0,1]→[-scale,scale] 映射 + q_sample + 还原
    diff_boxes, noise = self.apply_box_noise(gt_boxes, t)   # noise 未被后续使用

    # (5) 标签加噪: 把 label embedding 当连续向量做 q_sample
    diff_labels = self.q_sample(x_start=labels, t=t, noise=noise_labels)

    # (6) 拼接加噪 query
    noisy_queries = torch.cat([diff_labels, diff_boxes], dim=1)

    # (7) 固定随机 shuffle 匹配 (跳过 Hungarian)
    shuffled_indices = torch.randperm(diff_labels.size(0))
    diff_labels = diff_labels[shuffled_indices]
    diff_boxes   = diff_boxes[shuffled_indices]
    shuffled_indices = torch.argsort(shuffled_indices)     # 逆置换
    # indices[0] = 逆置换后前 num_gt 个 (即原 GT 在新位置的索引)
    # indices[1] = [0, 1, ..., num_gt-1] (原 GT 目标索引)
    indices = shuffled_indices[:num_gt], torch.arange(diff_labels.size(0))[:num_gt]
    return noisy_queries, diff_boxes, queries, t, indices
```

**关键点逐项解读**:

| 要点 | 实现 | 说明 |
|---|---|---|
| t 采样粒度 | `torch.randint(0, T, (1,))` | **每图一个 t** (非每 query, 非每 batch)。`process_targets` 按图循环调用, 每图独立采 t |
| box → 扩散空间 | `(boxes*2-1)*scale`, scale=2 | [0,1] cxcywh → [-2,2] |
| box 加噪 | `apply_box_noise` 内 `q_sample` + clamp + 还原 | clamp 到 [-scale, scale] 后映射回 [0,1] |
| label 加噪 | `q_sample(x_start=label_embed, t, noise=randn)` | 把 label embedding 当连续向量扩散 (非常规做法) |
| shuffle 匹配 | `randperm` + `argsort` 取逆 | **预计算 indices, 跳过 Hungarian**: 每个 GT query 经随机置换后落到新位置, 该位置 ↔ 原 GT 目标 |
| noisy_gt 参数 | L247, L899-901 | True 时 no-object padding box 用 `randn().sigmoid()` (随机 [0,1]) 替代默认 `[0.5,0.5,0.5,0.5]` |

**shuffle 匹配的数学含义**:
- 加噪前 layout: `[gt_0, gt_1, ..., gt_{n-1}, noobj_0, ..., noobj_{K-n-1}]`
- `shuffled_indices[i]` = 原 index 为 i 的 query 现在的位置;
- `argsort(shuffled_indices)[k]` = 原 GT k 现在所在的新位置;
- `indices = (新位置[:num_gt], [0..num_gt-1])` → query 在新位置 `new_pos[k]` 监督原 GT `k`。
- no-object query 不出现在 indices[0], 即不参与正样本监督。

### 2.2 label_enc 编码 (L180, L1183)

```python
self.label_enc = nn.Embedding(num_classes + 1, embed_dim)   # +1 给 no-object 类
# process_targets 中:
labels = self.label_enc(labels.squeeze())   # [num_queries] → [num_queries, embed_dim]
```

`+1` 索引 (num_classes) 用于 no-object 类。no-object query 的 label 经 embedding 后
作为连续向量参与 `q_sample` 加噪。

### 2.3 损失计算与 SNR 加权 (L563-589)

```python
# 按 t 提取 lvlb_weights (每图一个标量)
lvlb_class_weights = extract(self.loss_weight, t, [batch_size])                # [bs]
# 按 t 提取 lvlb_weights 并广播到 num_queries, 再按 indices 收集到匹配 query
t_weight = extract(self.loss_weight, t, [batch_size, self.num_queries])      # [bs, Q]
t_weight = [t_weight[i].expand(len(indices[i][0])) for i in range(batch_size)]
t_weight = torch.cat(t_weight)                                               # [总匹配 query 数]

loss_dict = self.criterion(output, targets, dn_meta, indices, t_weight,
                           class_weights=lvlb_class_weights)
```

**SNR 加权公式** (register_schedule_old, L278):

```
loss_weight[t] = 0.5 * sqrt(alphas_cumprod[t]) / (2 - alphas_cumprod[t])
```

- ac→1 (t 小, 低噪声): weight → 0.5 (大权重, 强调干净预测)
- ac→0 (t 大, 高噪声): weight → 0 (小权重, 高噪声步不强调)
- `loss_weight[0] = loss_weight[1]` 处理 t=0 奇点

> 与标准 LVLB (`0.5 * sqrt(SNR)/(1+SNR)`) 形状相近但数值不同, 疑似 `2 - ac` 应为
> `2*(1-ac)` 的笔误。**KaryoFlow 不使用 LVLB 加权**, 训练用 RF velocity MSE,
> 推理用 SHTS 自适应时间网格 (基于 SNR 导数)。

### 2.4 criterion 接收预计算 indices (跳过 Hungarian)

**重要发现**: 提供的 `diffu_criterion.py` 是 **原始 deprecated 版本** (文件 docstring
L14-18 明示: "This is the original implementation of SetCriterion which will be
deprecated in the next version. We keep it here because our modified Criterion module
is still under test.")。

该原始版 `forward` (L350-399) **仍运行 Hungarian 匹配**:

```python
def forward(self, outputs, targets, indices, t_weight, return_indices=False, **kwargs):
    ...
    indices = self.matcher(outputs_without_aux, targets)   # L363: 覆盖传入的 indices!
    ...
```

但调用方 (dino_diffu_det_noise.py L589) 传参顺序为:

```python
self.criterion(output, targets, dn_meta, indices, t_weight, class_weights=...)
#                    outputs  targets  dn_meta  indices  t_weight
```

若按原始 forward 签名映射: `dn_meta→indices 参数`, `indices→t_weight 参数`,
`t_weight→return_indices 参数` — 这会导致 `t_weight.unsqueeze(1)` 在 list 上崩溃。
**因此实际使用的是 modified 版** (本仓库未提供), 其签名应为
`forward(self, outputs, targets, dn_meta, indices, t_weight, **kwargs)`,
**直接使用预计算的 shuffle indices, 跳过 Hungarian matcher**。

| 损失项 | 公式 (L229-266) | t_weight 应用 |
|---|---|---|
| loss_bbox (L1) | `F.l1_loss(src_boxes, target_boxes)` | `* t_weight.unsqueeze(1).repeat(1,4)` 当 `use_vlb=True` |
| loss_giou | `1 - diag(generalized_box_iou(...))` | `* t_weight` 当 `use_vlb=True` |
| loss_class (focal) | `sigmoid_focal_loss(...)` | `* t_weight.unsqueeze(1)` 当 `use_vlb=True` (L51-53) |
| loss_bbox_noobject | `L1(noobj_pred, 0.5)` | `* class_weights.unsqueeze(1)` (用 lvlb_class_weights) |

### 2.5 DN queries 与 Diffusion queries 损失分离

`dn_post_process` (L804-816) 按 `padding_size = single_padding * dn_num` 切分:

```python
padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
output_known_class = outputs_class[:, :, :padding_size, :]   # DN 部分
output_known_coord = outputs_coord[:, :, :padding_size, :]
outputs_class = outputs_class[:, :, padding_size:, :]        # Diffusion 部分
outputs_coord = outputs_coord[:, :, padding_size:, :]
dn_metas["output_known_lbs_bboxes"] = {"pred_logits": ..., "pred_boxes": ...}
```

DN 损失用 `dn_metas["output_known_lbs_bboxes"]` 单独计算; Diffusion 损失用切分后的
`outputs_class/coord`。两者通过 attention mask (`prepare_for_cdn` L771-795) 在
transformer 内部隔离 (DN 组间互不可见, match query 不可见 DN)。

---

## 3. 推理流程 (DDIM Sampling, L914-1132)

### 3.1 时间序列构建 (sampling_steps=25)

```python
times = torch.linspace(0, total_timesteps - 1, steps=sampling_timesteps + 1)
# T=1000, S=25 → 26 个整数点 [0, 40, 80, ..., 960, 999] (int 取整)
times = list(reversed(times.int().tolist()))   # [999, 960, ..., 40, 0]
time_pairs = list(zip(times[:-1], times[1:]))  # [(999,960), (960,920), ..., (40,0)]
```

- 共 S=25 个 (time, time_next) 对;
- **不含 (0, -1) 步** (与 KaryoFlow `build_time_pairs` 用 `linspace(-1, T-1, ...)` 不同);
- 最后一步 (40, 0): alpha_next = alphas_cumprod[0] = 1.0 (余弦归一化), sigma=0, c=0,
  `x_next = sqrt(1)*x0 + 0 = x0` (完全去噪)。

### 3.2 初始噪声生成 (两阶段提议 + 最大噪声, L927-956)

```python
# (1) t=0 编码图像, 取两阶段提议 init_reference
t = torch.zeros((N,), device=self.device, dtype=torch.long)
(memory, mask_flatten, spatial_shapes, level_start_index,
 valid_ratios, init_reference, target_unact) = self.transformer(..., is_train=False)

# (2) query 内容: 可学习 embedding 或 encoder target
query = self.transformer.tgt_embed.weight[None].repeat(N, self.num_queries, 1)  # 若 learnt_init_query
# 或 query = target_unact

# (3) 对两阶段提议加最大噪声 (t = T-1)
t = torch.full((N,), self.num_timesteps - 1, device=self.device, dtype=torch.long)
_, reference_points = self.apply_box_noise(init_reference, t=t)   # q_sample 至最大噪声
noise = torch.clamp(reference_points, min=-1*self.scale, max=self.scale)
noise = ((noise / self.scale) + 1) / 2                            # 还原 [0,1]
init_reference_points = noise.float()
```

**两阶段提议 + 最大噪声**: 不是从纯 randn 开始, 而是先跑 encoder 得到两阶段 proposal
(`init_reference`), 再 `q_sample(x0=proposal, t=T-1)` 加最大噪声作为起点。
这保留了 encoder 的语义先验, 仅在最大噪声步扰动。

### 3.3 DDIM 更新公式 (L1019-1037)

```python
# 模型预测 x0 (bbox_start in [0,1]) → 映射到扩散空间
bbox_start = outputs_coord[-1]
x_start = (bbox_start * 2 - 1.) * self.scale
x_start = torch.clamp(x_start, min=-1*self.scale, max=self.scale)

# 由 (x_t, x0) 反推噪声
pred_noise_bbox = self.predict_noise_from_start(reference_points, t, x_start)

# DDIM 系数 (eta=0 → 确定性 DDIM)
alpha = self.alphas_cumprod[time]
alpha_next = self.alphas_cumprod[time_next]
sigma = eta * ((1 - alpha/alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()  # eta=0 → 0
c = (1 - alpha_next - sigma**2).sqrt()                                          # = sqrt(1-ac_next)

# 更新: x_{t-1} = sqrt(ac_next) * x0 + c * pred_noise  (+ sigma*randn if eta>0)
reference_points = bbox_start * alpha_next.sqrt() + c * pred_noise_bbox
```

> **注意**: 更新公式用 `bbox_start` (模型预测的 x0, [0,1] 空间) 而非 `x_start`
> (扩散空间 x0)。这是疑似 bug: 应该用 `x_start * alpha_next.sqrt()`。
> `bbox_start * alpha_next.sqrt()` 把 [0,1] 量级的 x0 与扩散空间量级的 pred_noise
> 混在一起。后续 clamp + 还原 [0,1] 会部分修正, 但量纲不一致。
> 实际效果上 `reference_points` 在每步都被 clamp 到 [-scale,scale] 再还原, 可能
> 掩盖了这个量纲问题。

### 3.4 box_renewal 自适应阈值 (L1038-1061)

```python
if self.box_renewal:
    threshold = 0.5                        # 基础值 (被覆盖)
    threshold = time_next / 100            # v1: 自适应, 随 t 递减
    # threshold = (time + time_next) / 200  # v2 (注释)
    assert threshold >= 0 and threshold <= 1
    score_per_image = score_per_image.sigmoid()
    value, _ = torch.max(score_per_image, -1)
    keep_idx = value > threshold           # 高置信 query 保留
    # 低置信 query 重置为纯随机噪声
    new_ref = torch.randn_like(init_reference_points)
    new_ref[keep_idx] = init_reference_points[keep_idx]
    reference_points = new_ref
    # 同步更新 [0,1] 空间的 init_reference_points
    new_ref = clamp(new_ref, -scale, scale); new_ref = ((new_ref/scale)+1)/2
    new_ref[keep_idx] = init_reference_points[keep_idx]
    init_reference_points = new_ref
```

**阈值策略**: `threshold = time_next / 100`。
- 早期步 (time_next 大, 如 960): threshold = 9.6 > 1 → 所有 query 的 score 都 < threshold
  → **全部重置为随机噪声** (激进探索);
- 后期步 (time_next 小, 如 40): threshold = 0.4 → 高置信 query 保留, 低置信重置;
- 最后步 (time_next=0): threshold=0 → 所有 score>0 的 query 保留。

> **潜在 bug**: `assert threshold <= 1` 在 T=1000、S=25 时, 前 ~9 步 time_next > 100,
> threshold > 1, assert 会失败。要么配置上 T 较小 (如 T=100), 要么该 assert 是后加的
> 未触发。移植时需注意归一化 threshold 或用 `time_next / total_timesteps`。

### 3.5 ensemble 收集与 NMS 后处理 (L1062-1118)

```python
# 每步收集预测 (final=False 跳过 scale, 返回原始坐标)
if self.use_ensemble and self.sampling_timesteps > 1:
    box_pred, scores, labels = self.inference(outputs_class[-1], outputs_coord[-1],
                                              images.image_sizes, final=False)
    ensemble_coord.append(box_pred)   # [N_img, 4]
    ensemble_score.append(scores)
    ensemble_label.append(labels)

# 采样结束后合并所有步的预测
box_pred_per_image = torch.cat(ensemble_coord, dim=0)   # [S * N_img, 4]
scores_per_image = torch.cat(ensemble_score, dim=0)
labels_per_image = torch.cat(ensemble_label, dim=0)

# NMS (阈值 0.8)
if self.use_nms:
    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.8)
    ...保留 keep...

# Top-K 选择 (select_box_nums_for_evaluation=300)
sorted_indices = torch.argsort(scores_per_image, descending=True)
selected_indices = sorted_indices[:300]
```

**Ensemble 策略**: 跨所有采样步收集全部预测 → NMS(0.8) 去重 → 按分数取 top-300。
无权重加权, 各步预测等权合并。

---

## 4. 与 KaryoFlow 的对比

### 4.1 差异对比总表

| 维度 | DiffuDETR | KaryoFlow |
|---|---|---|
| **扩散范式** | DDPM (cosine schedule, T=1000) | Rectified Flow (直线 ODE, t∈[0,1]) |
| **参数化** | x0 (模型直接预测干净 box) | x0 (模型预测 x0, 由 RF 推速度) |
| **扩散空间映射** | `(box*2-1)*2`, [-2,2] cxcywh | `(x0*2-1)*snr_scale`, [-2,2] cxcywh (一致) |
| **采样器** | DDIM (eta=0, 确定性, 25 步) | DDIM / DPM-Solver++ (2-3 阶) / Heun / Euler, 可选 SHTS 网格 |
| **时间序列** | `linspace(0, T-1, S+1).int()` 整数步, 不含 (0,-1) | RF: `linspace(1,0,S+1)` 连续 t; DDPM: `linspace(-1,T-1,S+1)` 含 (0,-1) |
| **初始噪声** | 两阶段 proposal + `q_sample(t=T-1)` | (采样器决定, 通常纯 randn 或两阶段 proposal) |
| **box_renewal 阈值** | `time_next/100` (随 t 递减, 可能 >1) | 固定 `score_thr` + `min_keep` topk 兜底 + VGAR 时间自适应 alpha |
| **renewal 噪声** | 纯 `randn` (无 x0 引导) | 纯 `randn` 或 VGAR: `alpha(t)*x0_pred + (1-alpha(t))*randn` |
| **ensemble** | 收集所有步预测 → NMS(0.8) → top-300 | 收集所有步预测 → NMS(可配置) → rescale; + PCSE 核型一致性选优 |
| **损失加权** | LVLB: `0.5*sqrt(ac)/(2-ac)`, 每图一 t | RF velocity MSE; 推理用 SHTS (SNR 导数网格) 自适应步分配 |
| **t 采样粒度** | 每图一个 t (图内 query 共享) | (训练在 head 模块, 未在 sampling.py 中体现) |
| **query 匹配** | 固定随机 shuffle (跳过 Hungarian) | (训练匹配在 head 模块) |
| **label 处理** | label_enc embedding 作为连续向量 q_sample 加噪 | (未在采样文件体现) |
| **跨级联更新** | 无 | CCBR: `apply_inter_head_renewal` 软 renewal (alpha*pred + (1-alpha)*perturbed) |
| **per-dim 策略** | 无 | `RFDPMSolverPerDim`: h 维 1 阶 Euler, cx/cy/w 维 2 阶 DPM-Solver++ |

### 4.2 关键差异深度解读

**(1) 扩散空间映射 — 完全一致**

两者均用 `(x*2-1)*scale` (scale=2) 把归一化 [0,1] cxcywh 映射到 [-2,2],
反向用 `((x/scale)+1)/2` 还原。可直接互换。

**(2) 采样器 — DDIM vs DPM-Solver++**

- DiffuDETR 用确定性 DDIM (eta=0), 单步更新:
  `x_{t-1} = sqrt(ac_next)*x0 + sqrt(1-ac_next)*pred_noise`;
- KaryoFlow DDIM (`ddim_step`) 公式相同但 eta 可配;
  主力是 `RFDPMSolverMultistep` (2-3 阶), 用 x0_history 多项式插值积分 RF ODE:
  `x_next = (t_next/t_n)*x + (1 - t_next/t_n)*x0 + phi1*D1 + phi2*D2`;
- KaryoFlow 还支持 `RFDPMSolverAdaptive` (static/eta_threshold 自适应阶次)
  和 `RFDPMSolverPerDim` (per-dim 阶数分配)。

**(3) box_renewal 阈值 — 自适应递减 vs 固定+VGAR**

- DiffuDETR: `threshold = time_next/100`, 早期全重置 (threshold>1 时全随机),
  后期逐步收紧。**无 min_keep 兜底**, 可能全部重置导致信息丢失;
- KaryoFlow: 固定 `score_thr` + `min_keep` (topk 兜底, 保证至少保留 K 个),
  可选 VGAR (`alpha(t) = 0.2 + 0.6*sigmoid(5*(0.5-t))`, 早期探索后期利用),
  renewal 噪声保留 x0_pred 方向而非纯随机。

**(4) ensemble — 简单合并 vs 合并+PCSE**

- DiffuDETR: 所有步预测 cat → NMS(0.8) → top-300, 无假设评分;
- KaryoFlow: 所有步预测 cat → NMS(可配) → rescale, 额外有 `KaryotypeScorer`
  (PCSE) 用核型先验 (数量/配对/形态/性染色体) 对假设评分选优。

**(5) 损失加权 — LVLB vs velocity MSE + SHTS**

- DiffuDETR: 训练用 LVLB 权重 `0.5*sqrt(ac)/(2-ac)` 加权 L1/GIoU/focal,
  每图一个 t, 同图 query 共享权重;
- KaryoFlow: 训练用 RF velocity MSE (未在采样文件体现), 推理用 SHTS
  (基于 SNR 导数的自适应时间网格) 优化采样步分布, 不涉及 LVLB 权重。

---

## 5. 实现细节参数表

### 5.1 DIFFUSION_DINO.__init__ 关键参数 (L121-152)

| 参数 | 默认值 | 作用 |
|---|---|---|
| `timesteps` | 1000 | 训练扩散步数 T |
| `sampling_steps` | 25 | 推理 DDIM 采样步数 S |
| `beta_schedule` | 'linear' | LDM 调度类型 (仅 old_schedule=False 时生效) |
| `old_schedule` | True | True 用 register_schedule_old (cosine), False 用 register_schedule_ldm |
| `noisy_gt` | False | True 时 no-object padding box 用 randn().sigmoid() |
| `v_posterior` | 0.0 | posterior variance 混合系数 (仅 ldm 调度) |
| `linear_start` | 1e-4 | linear beta 起点 |
| `linear_end` | 2e-2 | linear beta 终点 |
| `cosine_s` | 8e-3 | cosine 调度偏移 |
| `scale` | 2 (硬编码 L249) | 扩散空间尺度, box 映射到 [-scale, scale] |
| `ddim_sampling_eta` | 0.0 (L230) | DDIM 随机性, 0=确定性 |
| `box_renewal` | True (L252) | 推理时低置信 query 重置 |
| `use_ensemble` | True (L253) | 跨步 ensemble |
| `use_nms` | True (L254) | ensemble 后 NMS |
| `dn_number` | 100 | DN query 组数 |
| `label_noise_ratio` | 0.2 | DN 标签噪声概率 |
| `box_noise_scale` | 1.0 | DN box 噪声尺度 |
| `select_box_nums_for_evaluation` | 300 | 最终 top-K 输出数 |
| `parameterization` | "x0" (L232) | 模型预测目标 |

### 5.2 推理 NMS / ensemble 参数

| 参数 | 值 | 位置 |
|---|---|---|
| ensemble NMS 阈值 | 0.8 | L1081 (硬编码) |
| box_renewal threshold 公式 | `time_next/100` | L1043 (v1) |
| box_renewal threshold v2 | `(time+time_next)/200` | L1044 (注释) |
| box_renewal 基础值 | 0.5 | L1040 (被覆盖) |

---

## 6. 移植到 mmdet 框架的注意事项

### 6.1 调度注册

- mmdet 的 `BaseModel` 不自带 `register_buffer`, 需在 detector `__init__` 中显式
  `self.register_buffer('alphas_cumprod', ...)` 或独立封装为 `NoiseSchedule` 模块
  (KaryoFlow 已有 `noise_schedule.py` 可复用);
- `loss_weight` 用 `persistent=False` (L280, L348), 移植时若用 mmdet 的
  `load_state_dict` 需注意该 buffer 不被保存;
- `cosine_beta_schedule` 与 KaryoFlow `cosine_noise_schedule` 完全一致, 可直接复用。

### 6.2 扩散空间映射

- DiffuDETR 全程用 cxcywh 归一化坐标; mmdet 默认 xyxy 绝对坐标。需在
  `data_preprocessor` 或 `get_targets` 中转换;
- 映射 `(box*2-1)*scale` 与 KaryoFlow `xyxy_to_raw` 一致, 可直接复用
  `DiffusionSampler.xyxy_to_raw` / `raw_to_xyxy`。

### 6.3 时间步采样

- DiffuDETR **每图一个 t** (`torch.randint(0, T, (1,))`), mmdet 的 batch 训练中
  需在 `loss` 方法内按图循环采 t, 或改为每 batch 一个 t (简化但损失分辨率);
- `t_weight` (lvlb_weights) 按 t 索引并广播到 query, 需在 `loss_by_feat` 中实现。

### 6.4 query 匹配 (跳过 Hungarian)

- DiffuDETR 的 shuffle 匹配 (randperm + argsort 逆置换) 是预计算 indices,
  **跳过 Hungarian matcher**。mmdet 的 `Assigner` 体系基于 Hungarian
  (`MaskHungarianAssigner` 等), 需新增 `DiffusionShuffleAssigner` 或在
  `loss_by_feat` 中直接传入预计算 indices, 绕过 assigner;
- **重要**: 提供的 `diffu_criterion.py` 是 deprecated 原始版 (仍跑 Hungarian),
  实际使用的 modified 版未在仓库中。移植时需自行实现 "跳过 matcher" 逻辑:
  在 criterion forward 中直接用传入的 indices, 不调用 `self.matcher(...)`。

### 6.5 DDIM 采样器

- DiffuDETR DDIM 更新公式 `x_{t-1} = sqrt(ac_next)*x0 + c*pred_noise` (eta=0, c=sqrt(1-ac_next))
  与 KaryoFlow `ddim_step` 一致, 可直接复用 `DiffusionSampler.ddim_step`;
- **注意量纲 bug**: DiffuDETR L1032 用 `bbox_start` ([0,1] 空间) 而非
  `x_start` (扩散空间) 乘 `alpha_next.sqrt()`。移植时应改为 `x_start * alpha_next.sqrt()`
  以保证量纲一致。KaryoFlow `ddim_step` (L421-423) 用 `x0` (扩散空间) 是正确的。

### 6.6 box_renewal

- DiffuDETR threshold = `time_next/100` 在 T=1000 时前几步 >1 导致 assert 失败。
  移植时应归一化为 `time_next / total_timesteps` 或用 KaryoFlow 的固定
  `score_thr` + `min_keep` 策略;
- DiffuDETR 无 `min_keep` 兜底, 可能全部重置。建议移植时引入 KaryoFlow 的
  `min_keep` topk 兜底机制 (L233-238);
- 可选引入 KaryoFlow VGAR (`alpha(t)*x0_pred + (1-alpha(t))*randn`) 替代纯随机
  renewal, 保留 x0 方向信息。

### 6.7 ensemble / NMS

- DiffuDETR ensemble NMS 阈值 0.8 硬编码 (L1081), top-300 硬编码。
  mmdet 的 `test_cfg` 应参数化;
- 可选引入 KaryoFlow PCSE (`KaryotypeScorer`) 对染色体任务做核型一致性后处理。

### 6.8 DN 与 Diffusion 损失分离

- DN query 与 diffusion query 通过 attention mask 隔离, 损失在
  `dn_post_process` 中切分。mmdet 需在 `loss_by_feat` 中维护 `dn_meta`
  (single_padding, dn_num), 切分 DN 部分单独计算 DN loss;
- `label_enc = nn.Embedding(num_classes+1, embed_dim)` 的 +1 给 no-object 类,
  移植时注意 num_classes 语义。

### 6.9 label 扩散 (非常规)

- DiffuDETR 把 label embedding 当连续向量做 `q_sample` 加噪 (L1159)。
  这与常规分类损失 (focal on discrete labels) 不兼容 — 加噪后的 label embedding
  如何参与损失需明确。移植时若不需要 label 扩散, 可只用 box 扩散, label 走
  常规 DN 噪声 (`apply_label_noise` 随机替换类别)。

### 6.10 复用建议

| DiffuDETR 组件 | KaryoFlow 可复用模块 |
|---|---|
| `cosine_beta_schedule` | `noise_schedule.cosine_noise_schedule` (一致) |
| `q_sample` / `predict_noise_from_start` | `sampling.predict_noise_from_start` + 自行实现 q_sample |
| 扩散空间映射 `(x*2-1)*scale` | `DiffusionSampler.xyxy_to_raw` / `raw_to_xyxy` |
| DDIM 采样 | `DiffusionSampler.ddim_step` (修正量纲后) |
| box_renewal | `DiffusionSampler.apply_box_renewal` (带 min_keep + VGAR) |
| ensemble + NMS | `DiffusionSampler.post_process` |
| LVLB 加权 | 无对应 (KaryoFlow 用 RF velocity loss), 需自行实现 |

---

## 7. 关键文件路径

**DiffuDETR (已下载分析, 未保留本地副本)**:
- `projects/diffu_dino/modeling/dino_diffu_det_noise.py` (1192 行, 核心检测器)
- `projects/diffu_dino/modeling/diffu_criterion.py` (416 行, deprecated 原始 criterion)
- `layers_diffu_detr/denoising.py` (269 行, DN query 生成)

**KaryoFlow (本地)**:
- `/home/linkst/workspace/chromosome-kd/ldmdet/diffusion/sampling.py` (655 行, 采样器)
- `/home/linkst/workspace/chromosome-kd/ldmdet/diffusion/rectified_flow.py` (474 行, RF + DPM-Solver++)
- `/home/linkst/workspace/chromosome-kd/ldmdet/diffusion/noise_schedule.py` (32 行, 余弦调度)

---

## 8. 总结

DiffuDETR 的扩散检测核心可概括为:

1. **DDPM x0 参数化 + cosine 调度**, box 映射到 [-2,2] cxcywh 扩散空间;
2. **训练**: 每图采一个 t, GT box 经 q_sample 加噪; label embedding 也作连续向量
   加噪; 用固定随机 shuffle 匹配 (跳过 Hungarian); LVLB 权重 `0.5*sqrt(ac)/(2-ac)`
   按 t 加权 L1/GIoU/focal 损失;
3. **推理**: 两阶段 proposal + 最大噪声 q_sample 起步; 25 步确定性 DDIM;
   box_renewal 用 `time_next/100` 自适应阈值重置低置信 query; 跨步 ensemble
   + NMS(0.8) + top-300。

**移植到 mmdet/KaryoFlow 的核心改动点**:
- 修正 DDIM 更新量纲 (用扩散空间 x0 而非 [0,1] bbox_start);
- box_renewal 阈值归一化 + 引入 min_keep 兜底;
- 实现 shuffle 预匹配 Assigner (跳过 Hungarian);
- DN/Diffusion 损失切分;
- 可选: 用 KaryoFlow DPM-Solver++ 替换 DDIM, 用 VGAR 替换纯随机 renewal,
  用 PCSE 增强 ensemble。
