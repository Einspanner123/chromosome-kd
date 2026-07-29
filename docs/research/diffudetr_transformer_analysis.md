# DiffuDETR Transformer 与 Timestep 注入逻辑技术报告

> 源仓库: https://github.com/MBadran2000/DiffuDETR
> 分析文件: `transformer.py` / `dino_transformer.py` / `bbox_embedd.py` / `attention.py` / `multi_scale_deform_attn.py` / `ldm/util.py`
> 对比对象: KaryoFlow `ldmdet/core/single_head.py` + `ldmdet/core/dynamic_conv.py`
> 生成日期: 2026-07-28

---

## 目录

1. [BaseTransformerLayer 结构与数据流](#1-basetransformerlayer-结构与数据流)
2. [DINOTransformer 结构与数据流](#2-dinotransformer-结构与数据流)
3. [TimeStepBlock (FiLM 调制) 实现](#3-timestepblock-film-调制实现)
4. [Attention 实现](#4-attention-实现)
5. [Timestep 注入完整路径](#5-timestep-注入完整路径)
6. [与 KaryoFlow 差异对比](#6-与-karyoflow-差异对比)
7. [移植到 mmdet 框架的注意事项](#7-移植到-mmdet-框架的注意事项)

---

## 1. BaseTransformerLayer 结构与数据流

> 文件: `layers_diffu_detr/transformer.py`

### 1.1 类结构图

```
BaseTransformerLayer(nn.Module)
├── __init__(attn, ffn, norm, time_step_embed, time_step_embed_query, operation_order)
│   ├── self.attentions: ModuleList   # 按 operation_order 中的 self_attn/cross_attn 顺序填充
│   ├── self.ffns: ModuleList          # 按 ffn 出现次数深拷贝
│   ├── self.norms: ModuleList         # 按 norm 出现次数深拷贝
│   ├── self.time_step_embed: Module   # 单个模块，当前未使用（cross_attn 中对 reference_points 的注入被注释）
│   └── self.time_step_embeds: ModuleList  # time_step_embed_query 的深拷贝，数量 = num_attn + num_ffns
│
└── forward(query, key, value, query_pos, key_pos, t, attn_masks, ...)
    ├── norm_index, attn_index, ffn_index, t_index = 0  # 四个独立游标
    └── for layer in operation_order:
        ├── "self_attn":  [若 t≠None] time_step_embeds[t_index](query, t) → t_index++ → attentions[attn_index](...)
        ├── "norm":       norms[norm_index](query)
        ├── "cross_attn": [若 t≠None] time_step_embeds[t_index](query, t) → t_index++ → attentions[attn_index](...)
        └── "ffn":        [若 t≠None] time_step_embeds[t_index](query, t) → t_index++ → ffns[ffn_index](...)
```

### 1.2 operation_order 定义

约束为 `{"self_attn", "norm", "cross_attn", "ffn"}` 的子集：

| 使用场景 | operation_order | num_attn | num_ffn | time_step_embeds 数量 |
|---------|-----------------|----------|---------|---------------------|
| Encoder | `("self_attn", "norm", "ffn", "norm")` | 1 | 1 | 2 |
| Decoder | `("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")` | 2 | 1 | 3 |

### 1.3 time_step_embed 与 time_step_embed_query 的区别

| 属性 | `time_step_embed` | `time_step_embed_query` |
|------|-------------------|------------------------|
| 类型 | 单个 nn.Module | 单个 nn.Module（构造时传入） |
| 用途 | 原设计用于 cross_attn 中调制 reference_points | 用于在每个 attn/ffn 前调制 query |
| 实际状态 | **未使用**（cross_attn 中相关代码被注释掉，见 forward 第 202-204 行） | **生效**：构造时被深拷贝 `num_attn + num_ffn` 次存入 `self.time_step_embeds`，原字段置 None |
| 调用对象 | reference_points（key 的参考点） | query（解码器查询特征） |

关键代码：
```python
# 构造时：time_step_embed_query 被复制为 ModuleList
if time_step_embed_query is not None:
    self.time_step_embeds = nn.ModuleList()
    for _ in range(num_attn + num_ffns):
        self.time_step_embeds.append(copy.deepcopy(time_step_embed_query))
    self.time_step_embed_query = None  # 原字段清空
```

### 1.4 t_index 递增逻辑

`t_index` 是一个**层内游标**，初始为 0。每当 `t is not None` 且执行到 `self_attn` / `cross_attn` / `ffn` 之一时，先调用 `self.time_step_embeds[t_index]`，然后 `t_index += 1`。

以 Decoder 的 operation_order 为例（`num_attn + num_ffn = 3`）：

```
self_attn  → time_step_embeds[0] → t_index=1
norm       → (无注入)
cross_attn → time_step_embeds[1] → t_index=2
norm       → (无注入)
ffn        → time_step_embeds[2] → t_index=3
norm       → (无注入)
```

每个 `time_step_embeds[i]` 是独立的 TimeStepBlock 实例（深拷贝），参数不共享。

### 1.5 "no_queries" 参数的作用（DN queries 分离处理）

`no_queries` 通过 `**kwargs` 传入，是一个 list `[num_dn_queries, num_diffusion_queries]`。

**背景**：DN (Denoising) queries 与 diffusion queries 拼接在同一个 query 序列中，但只有 diffusion queries 需要时间步条件化，DN queries 不需要。

**逻辑**（在 self_attn / cross_attn / ffn 中均相同）：
```python
if "no_queries" in kwargs:
    # 切分：前 no_queries[0] 个是 DN queries（不注入 t），其余是 diffusion queries（注入 t）
    x = query[:, kwargs['no_queries'][0]:, :]   # diffusion 部分
    x = self.time_step_embeds[t_index](x, t)     # 仅对 diffusion 部分做 FiLM
    y = query[:, :kwargs['no_queries'][0], :]    # DN 部分，保持不变
    query = torch.cat([y, x], 1)                 # 重新拼接
else:
    query = self.time_step_embeds[t_index](query, t)  # 全部注入
```

**触发条件**：当 `DINOTransformer.cdn_with_timestep = False` 时，DINOTransformer.forward 会在 kwargs 中设置 `no_queries`。若 `cdn_with_timestep = True`，则 DN queries 也接收 timestep 注入（走 else 分支）。

### 1.6 timestep 注入位置总结

| 子模块 | 注入时机 | 注入对象 | 注入方式 |
|--------|---------|---------|---------|
| self_attn | **进入 self_attn 之前** | query | `time_step_embeds[t_index](query, t)` |
| cross_attn | **进入 cross_attn 之前** | query | `time_step_embeds[t_index](query, t)` |
| ffn | **进入 ffn 之前** | query | `time_step_embeds[t_index](query, t)` |
| norm | 不注入 | — | — |

注入发生在每个子模块**之前**（pre-conditioning），而非之后。注意力/FFN 模块本身**不感知** timestep，timestep 仅通过前置的 FiLM 调制改变 query 特征分布。

---

## 2. DINOTransformer 结构与数据流

> 文件: `projects/diffu_dino/modeling/dino_transformer.py`

### 2.1 整体类结构

```
DINOTransformer(nn.Module)
├── encoder: DINOTransformerEncoder   # 基于 MultiScaleDeformableAttention 的编码器
├── decoder: DINOTransformerDecoder   # 混合 MultiheadAttention + MultiScaleDeformableAttention
├── level_embeds: Parameter           # [num_levels, embed_dim]
├── enc_output: Linear(embed_dim → embed_dim)
├── enc_output_norm: LayerNorm
├── tgt_embed: Embedding              # learnt init query（两种模式）
├── time_embed: Sequential            # MLP: embed_dim → 4*embed_dim → 4*embed_dim
├── tgt_embed_new: bool               # 是否使用单向量广播模式
└── cdn_with_timestep: bool           # DN queries 是否也接收 timestep 注入
```

### 2.2 与标准 DINO Transformer 的区别

| 方面 | 标准 DINO | DiffuDETR (DINOTransformer) |
|------|----------|------------------------------|
| Decoder query 来源 | learnt init query 或 two-stage proposal | learnt init query + **diffusion noised queries** 拼接 |
| Reference points 来源 | two-stage proposal boxes | **diffusion noised boxes** + DN box queries 拼接 |
| Timestep 条件化 | 无 | **每层 decoder 注入 timestep（FiLM）** |
| ClassEmbed | 无 timestep | **ClassEmbed 含 TimeStepBlock** |
| time_embed MLP | 无 | `embed_dim → 4*embed_dim → 4*embed_dim` |
| forward 签名 | 标准 | 多了 `query_embeds_diffusion`, `time_steps`, `is_train` |
| 推理分支 | 与训练一致 | **提前返回**（`not is_train` 时返回 encoder 中间结果） |
| DN queries 处理 | 标准 DN | 可选是否对 DN 注入 timestep（`cdn_with_timestep`） |

### 2.3 time_embed MLP 结构

```python
time_embed_dim = self.embed_dim * 4   # 256 * 4 = 1024
self.time_embed = nn.Sequential(
    linear(self.embed_dim, time_embed_dim),   # 256 → 1024
    nn.SiLU(),
    linear(time_embed_dim, time_embed_dim),    # 1024 → 1024
)
```

**注意**：输出维度是 `4 * embed_dim`（1024），而非 `embed_dim`（256）。这与下游 `TimeStepBlock` 的 `emb_channels = 4 * embed_dim` 对应。

### 2.4 forward 签名

```python
def forward(
    self,
    multi_level_feats,        # 多尺度特征图列表
    multi_level_masks,        # padding masks
    multi_level_pos_embeds,   # 位置编码
    query_embed,              # tuple: (DN_label_queries, DN_box_queries)，可为 None
    attn_masks,               # DN 注意力掩码
    query_embeds_diffusion,   # tuple: (noisy_queries, noised_boxes) — 扩散噪声输入
    time_steps,               # 原始时间步索引（标量/整数），尚未编码
    is_train=True,            # 训练/推理分支标志
    **kwargs,
)
```

### 2.5 训练 vs 推理分支

**训练分支**（`is_train=True`）：
1. 编码器处理多尺度特征 → memory
2. 两阶段提案生成 → top-k reference_points + target_unact
3. 拼接 DN queries + diffusion queries
4. timestep 编码 → time_embed MLP
5. 解码器处理（含 timestep 注入）
6. 返回中间状态和参考点

**推理分支**（`is_train=False`）：
```python
if not is_train:
    return memory, mask_flatten, spatial_shapes, level_start_index, \
           valid_ratios, reference_points, target_unact.detach()
```
在两阶段提案生成后**提前返回**，不执行解码器。这表明推理时的扩散去噪循环在外部驱动，每次调用 transformer 只处理单步。

### 2.6 DN queries 与 diffusion queries 的拼接方式

```python
# 1. 解包 diffusion 输入
noisy_queries, reference_points = query_embeds_diffusion[0].float(), query_embeds_diffusion[1].float()
# 注意：这里 reference_points 被 diffusion noised boxes 覆盖

# 2. 拼接 reference_points: [DN_box_queries.sigmoid() | diffusion_noised_boxes]
if query_embed[1] is not None:   # DN_box_queries
    reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)

# 3. 初始化 target（diffusion queries 的内容特征）
if self.learnt_init_query:
    if self.tgt_embed_new:
        target = self.tgt_embed.weight[None].repeat(bs, self.two_stage_num_proposals, 1)
    else:
        target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
else:
    target = target_unact.detach()   # 两阶段提案特征

# 4. 拼接 target: [DN_label_queries | tgt_embed_queries]
if query_embed[0] is not None:   # DN_label_queries
    if not self.cdn_with_timestep:
        kwargs['no_queries'] = [query_embed[0].shape[1], target.shape[1]]
    target = torch.cat([query_embed[0], target], 1)
```

**最终拼接结果**：

```
target          = [ DN_label_queries | tgt_embed_queries ]
reference_points = [ DN_box_queries.sigmoid() | diffusion_noised_boxes ]
                   └── DN 部分 ──┘   └── diffusion 部分 ──┘
```

### 2.7 attn_mask 的生成和作用

`attn_masks` 从外部传入（由 DN 模块生成）。作用：
- **DN queries 内部**：部分可见（group-wise，防止不同 group 互相看到）
- **DN → diffusion**：阻止 DN queries 看到 diffusion queries
- **diffusion → DN**：阻止 diffusion queries 看到 DN queries
- **diffusion → diffusion**：全可见

这确保 DN 去噪任务与扩散去噪任务在 query 交互上隔离，避免信息泄漏。

### 2.8 两阶段提案如何初始化 diffusion

```python
# 1. 编码器输出 → 生成提案
output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

# 2. 编码器端预测类别和坐标
enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

# 3. 选 top-k 作为两阶段提案
topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]
topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
reference_points = topk_coords_unact.detach().sigmoid()   # 两阶段提案的参考点

# 4. 提取提案特征
target_unact = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
```

两阶段提案提供：
- `reference_points`：扩散 query 的初始参考点（如果 `learnt_init_query=True` 则被替换为 learnt embedding）
- `target_unact`：当 `learnt_init_query=False` 时作为 diffusion query 的初始内容特征

**注意**：diffusion 的噪声 boxes（`query_embeds_diffusion[1]`）在拼接时**覆盖**了两阶段 reference_points（两阶段的 reference_points 仅在 `not is_train` 的推理返回中使用）。

---

## 3. TimeStepBlock (FiLM 调制) 实现

> 文件: `projects/diffu_dino/modeling/bbox_embedd.py`

### 3.1 类结构与 forward 数据流

```
TimeStepBlock(nn.Module)
├── __init__(channels, emb_channels, out_channels=256, dims=1, dropout=0.2, use_scale_shift_norm=True)
│   └── self.emb_layers = Sequential(SiLU, Linear(emb_channels → 2*out_channels))
│       # 注意：out_layers / out_norm 被注释掉，原 LDM 的 conv 投影未使用
│
└── forward(x, time_embed)
    ├── emb_out = emb_layers(time_embed)           # [bs, 2*out_ch]
    ├── while len(emb_out.shape) < len(x.shape):   # 广播到 query 维度
    │       emb_out = emb_out[:, None]             # [bs, 1, 2*out_ch]
    ├── scale, shift = emb_out.chunk(2, dim=-1)    # 各 [bs, 1, out_ch]
    ├── h = x * (1 + scale) + shift                # FiLM 调制
    └── return x + h                               # 残差连接
```

### 3.2 FiLM 调制的具体实现

**emb_layers 结构**：
```
input: time_embed  [bs, emb_channels]  (emb_channels = 4*embed_dim = 1024)
  │
  ├─ SiLU
  └─ Linear(emb_channels → 2*out_channels)   (2*256 = 512)
output: [bs, 2*out_channels]
```

**调制公式**：
```
scale, shift = chunk(emb_out, 2, dim=-1)     # 各 [bs, 1, out_ch]
h = x * (1 + scale) + shift                  # 标准 FiLM
output = x + h                               # 残差
```

展开后：`output = x + (x * (1 + scale) + shift) = x * (2 + scale) + shift`

**与标准 FiLM 的差异**：
- 标准 FiLM：`output = x * (1 + scale) + shift`（直接调制）
- DiffuDETR：`output = x + (x * (1 + scale) + shift)`（调制 + 残差）

额外的残差使 timestep 调制更温和（identity 占比更大），且原 LDM 中的 `LayerNorm + SiLU + Dropout + zero_conv` 后处理被全部注释掉，是一个高度简化的版本。

### 3.3 BBoxEmbed 中 TimeStepBlock 的使用情况

```python
class BBoxEmbed(nn.Module):
    def __init__(self, embed_dim, time_embed_channels):
        self.pred = MLP(embed_dim, embed_dim, 4, 3)
        self.norm = nn.LayerNorm(embed_dim)
        self.time_step_embed = TimeStepBlock(
            channels=embed_dim,
            emb_channels=time_embed_channels,
        )

    def forward(self, x, time_embed=None):
        # x = self.time_step_embed(x, time_embed)   ← 被注释掉
        # x = self.pred(self.norm(x))                ← 被注释掉
        x = self.pred(x)                            # 实际执行：无 timestep，无 norm
        return x
```

**结论**：BBoxEmbed **定义了** TimeStepBlock 但**未使用**。bbox 回归不接收 timestep 条件化。`self.norm` 也被旁路。

### 3.4 ClassEmbed 中 TimeStepBlock 的使用情况

```python
class ClassEmbed(nn.Module):
    def __init__(self, embed_dim, time_embed_channels, num_classes):
        self.pred = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)
        self.time_step_embed = TimeStepBlock(
            channels=embed_dim,
            emb_channels=4 * embed_dim,   # 注意：硬编码为 4*embed_dim
        )

    def forward(self, x, time_embed=None):
        if time_embed is not None:
            x = self.time_step_embed(x, time_embed)   # 生效：先 FiLM 调制
        # x = self.pred(self.norm(x))                   ← 被注释掉
        x = self.pred(x)                               # 再分类
        return x
```

**结论**：ClassEmbed **使用** TimeStepBlock。当 `time_embed` 不为 None 时，先对特征做 FiLM 调制，再送入分类头。`self.norm` 同样被旁路。

**注意维度不一致**：BBoxEmbed 的 `emb_channels` 由构造参数 `time_embed_channels` 传入，而 ClassEmbed 的 `emb_channels` **硬编码为 `4 * embed_dim`**。需确保外部传入的 `time_embed` 维度匹配。

---

## 4. Attention 实现

> 文件: `layers_diffu_detr/attention.py` + `layers_diffu_detr/multi_scale_deform_attn.py`

### 4.1 MultiheadAttention（attention.py）

**与标准 `torch.nn.MultiheadAttention` 的区别**：

| 特性 | torch.nn.MHA | DiffuDETR MultiheadAttention |
|------|-------------|------------------------------|
| 位置编码 | 外部处理 | **内部注入**：`query = query + query_pos`，`key = key + key_pos` |
| 残差连接 | 无 | **identity + out**（identity 默认为 query） |
| key_pos 默认 | — | 若 `key_pos=None` 且 `query_pos.shape == key.shape`，则 `key_pos = query_pos` |
| 输出 | attn_output, attn_weights | **仅返回 attn_output**（取 `[0]`） |
| proj_drop | 无 | 有（构造参数 `proj_drop`） |

本质是 `torch.nn.MultiheadAttention` 的薄包装，增加 position embedding 加法和 identity shortcut。

### 4.2 MultiScaleDeformableAttention（multi_scale_deform_attn.py）

**存在**，位于单独文件 `multi_scale_deform_attn.py`，从 `layers_diffu_detr` 包导出。

```
MultiScaleDeformableAttention(nn.Module)
├── __init__(embed_dim=256, num_heads=8, num_levels=4, num_points=4, ...)
│   ├── sampling_offsets: Linear(embed_dim → num_heads * num_levels * num_points * 2)
│   ├── attention_weights: Linear(embed_dim → num_heads * num_levels * num_points)
│   ├── value_proj: Linear(embed_dim → embed_dim)
│   └── output_proj: Linear(embed_dim → embed_dim)
│
└── forward(query, key, value, identity, query_pos, key_padding_mask,
            reference_points, spatial_shapes, level_start_index, t, ...)
    ├── query = query + query_pos (if not None)
    ├── value = value_proj(value)
    ├── sampling_offsets = sampling_offsets(query).view(bs, nq, nh, nl, np, 2)
    ├── attention_weights = softmax(attention_weights(query))
    ├── sampling_locations = reference_points + sampling_offsets / offset_normalizer
    ├── [CUDA] MultiScaleDeformableAttnFunction.apply(value, spatial_shapes, ...)
    │   [CPU] multi_scale_deformable_attn_pytorch(value, ...)  # 使用 F.grid_sample
    └── output = output_proj(output)
```

**CUDA 自定义算子**：`MultiScaleDeformableAttnFunction`（继承 `torch.autograd.Function`），提供高效 CUDA 实现。

**PyTorch 回退**：`multi_scale_deformable_attn_pytorch` 使用 `F.grid_sample` 做 bilinear 采样，逐 level 计算后加权求和。

**timestep 注入情况**：forward 签名中有 `t` 参数，但相关代码被注释掉：
```python
# self.sampling_shift = TimeStepBlock(...)  ← 构造中被注释
if t is not None:
    # sampling_offsets = self.sampling_offsets(query)
    # sampling_offsets = self.sampling_shift(sampling_offsets, t).view(...)
    sampling_offsets = self.sampling_offsets(query).view(...)  # 实际：t 不影响
else:
    sampling_offsets = self.sampling_offsets(query).view(...)  # 与 t≠None 分支完全相同
```
**结论**：MultiScaleDeformableAttention **不使用** timestep。t 参数是预留接口，但采样偏移和注意力权重都不受 timestep 影响。

### 4.3 ConditionalSelfAttention / ConditionalCrossAttention

attention.py 还包含 Conditional-DETR 的条件注意力实现（内容/位置分离投影），但在 DiffuDETR 的 DINOTransformer 中**未被使用**（DINOTransformer 只导入 `MultiheadAttention` 和 `MultiScaleDeformableAttention`）。

---

## 5. Timestep 注入完整路径

从原始时间步标量到每层 decoder 的完整数据流：

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 原始输入                                                              │
│    time_steps: [bs]  (整数/浮点标量, 扩散时间步索引)                      │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. 正弦位置编码 (DINOTransformer.forward)                                │
│    time_steps = timestep_embedding(time_steps.float(), self.embed_dim,   │
│                                    repeat_only=False)                    │
│    → [bs, 256]  (sinusoidal embedding, half cos + half sin)             │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. time_embed MLP (DINOTransformer)                                     │
│    time_steps = self.time_embed(time_steps)                             │
│    Sequential: Linear(256→1024) → SiLU → Linear(1024→1024)             │
│    → [bs, 1024]  (= [bs, 4*embed_dim])                                  │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. 传入 decoder (DINOTransformerDecoder.forward)                        │
│    inter_states, inter_refs = self.decoder(..., t=time_steps, ...)      │
│    t: [bs, 1024]                                                         │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. 逐层注入 (BaseTransformerLayer.forward, 共 num_layers 层)             │
│    每层 t 重置为传入的 [bs, 1024]                                        │
│    每层内 t_index 从 0 递增                                              │
│                                                                          │
│    ┌── self_attn 前 ──────────────────────────────────────┐             │
│    │  query [bs, nq, 256] + t [bs, 1024]                  │             │
│    │  → time_step_embeds[0](query, t)                     │             │
│    │    emb_layers: SiLU → Linear(1024 → 512)             │             │
│    │    → [bs, 1, 512] → chunk → scale[bs,1,256], shift   │             │
│    │    h = query*(1+scale) + shift                       │             │
│    │    query = query + h   (FiLM + 残差)                  │             │
│    │  → MultiheadAttention(query, query, query, ...)      │             │
│    └──────────────────────────────────────────────────────┘             │
│                                                                          │
│    ┌── cross_attn 前 ─────────────────────────────────────┐             │
│    │  query [bs, nq, 256] + t [bs, 1024]                  │             │
│    │  → time_step_embeds[1](query, t)   (同上)             │             │
│    │  → MultiScaleDeformableAttention(query, memory, ...)  │             │
│    │    (注意：MSDA 内部不使用 t)                          │             │
│    └──────────────────────────────────────────────────────┘             │
│                                                                          │
│    ┌── ffn 前 ────────────────────────────────────────────┐             │
│    │  query [bs, nq, 256] + t [bs, 1024]                  │             │
│    │  → time_step_embeds[2](query, t)   (同上)             │             │
│    │  → FFN(query)                                         │             │
│    └──────────────────────────────────────────────────────┘             │
│                                                                          │
│    (no_queries 模式下，仅 query[:, dn_len:, :] 被注入，DN 部分跳过)      │
└──────────────────────────┬──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. 预测头阶段 (decoder 外部)                                             │
│    ClassEmbed: 若 time_embed≠None → time_step_embed(x, t) → pred(x)     │
│    BBoxEmbed:  直接 pred(x)，无 timestep                                │
│    (注意：预测头的 time_embed 是否传入取决于外部调用，decoder 内部未传)    │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键维度变化**：
```
raw t: [bs] → sinusoidal: [bs, 256] → MLP: [bs, 1024]
                                                    │
                    ┌───────────────────────────────┘
                    ▼ (广播到 query 维度)
  TimeStepBlock.emb_layers: [bs, 1024] → SiLU → Linear → [bs, 512]
                                                    │
                              chunk(2, dim=-1) ──────┤
                              ▼                     ▼
                         scale [bs,1,256]     shift [bs,1,256]
                                                    │
  query [bs, nq, 256] * (1 + scale) + shift → h     │
  output = query + h                                │
```

---

## 6. 与 KaryoFlow 差异对比

> KaryoFlow 文件: `ldmdet/core/single_head.py` + `ldmdet/core/dynamic_conv.py`

### 6.1 整体架构对比

| 方面 | DiffuDETR | KaryoFlow |
|------|-----------|-----------|
| 检测范式 | 端到端 query-based (DETR-like) | 两阶段 proposal-based (RoI-based) |
| 特征提取 | 多尺度可变形注意力 (MSDA) | RoIAlign + DynamicConv |
| Query 数量 | 固定 (two_stage_num_proposals, 默认 900) | 动态 (由 proposal 数量决定) |
| 扩散对象 | query 特征 + reference boxes (noised boxes) | bboxes (noised boxes) |
| 解码器层数 | 6 层 (可配置) | 单层 (SingleDiffusionDetHead) |
| 扩散循环 | 推理时外部循环，每次调用 transformer | 内部逐 T 步去噪 |

### 6.2 Timestep 注入方式对比

| 方面 | DiffuDETR (FiLM) | KaryoFlow (scale_shift) | KaryoFlow (adaln_zero) |
|------|-------------------|------------------------|------------------------|
| **调制公式** | `h = x*(1+s)+sh; out = x + h` | `out = x*(1+s) + sh` | `out = LN(x)*(1+γ)+β; out = x + α·sublayer(out)` |
| **MLP 结构** | SiLU → Linear(4C → 2C) | SiLU → Linear(4C → 2C) | SiLU → Linear(4C → 6C) |
| **输出维度** | 2C (scale+shift) | 2C (scale+shift) | 6C (γ1,β1,α1,γ2,β2,α2) |
| **残差** | 有 (x + h) | 无 | 有 (α 门控) |
| **初始化** | 默认 | 默认 | **zero-init** (α=0, 稳定训练) |
| **注入位置** | 每层 self_attn/cross_attn/ffn **之前** | 整个 transformer 块**之后**，预测头之前 | self_attn 和 ffn **之前** (AdaLN) |
| **注入频率** | 每层 3 次 (6 层 = 18 次) | 每步 1 次 | 每步 2 次 |
| **DN 分离** | 支持 (no_queries) | 不适用 (无 DN) | 不适用 (无 DN) |
| **门控机制** | 无 | 无 | **有** (α 系数，DiT 风格) |

**关键差异分析**：

1. **注入位置**：DiffuDETR 在每个子模块前注入（高频、细粒度），KaryoFlow scale_shift 模式在整块后注入（低频、粗粒度），KaryoFlow adaln_zero 在子模块前注入但仅有 2 个调制点。

2. **调制强度**：DiffuDETR 的 `x + h = x*(2+s) + sh` 使 identity 占比更大（系数 2），调制更温和；KaryoFlow scale_shift 的 `x*(1+s) + sh` 是标准 FiLM；adaln_zero 的 zero-init 使训练初期等价于无调制（α=0 时 sublayer 输出被屏蔽），逐步引入条件。

3. **预测头条件化**：DiffuDETR 的 ClassEmbed 接收 timestep，BBoxEmbed 不接收；KaryoFlow 的 cls_head 和 reg_head 都使用同一个条件化后的 fc_feature（即都间接受 timestep 影响）。

### 6.3 注意力机制对比

| 方面 | DiffuDETR | KaryoFlow |
|------|-----------|-----------|
| **Self-Attention** | `MultiheadAttention` (torch.nn.MHA 包装) | `nn.MultiheadAttention` + SDPA 加速可选 |
| **Cross-Attention** | `MultiScaleDeformableAttention` (可变形采样) | **无** (替代为 DynamicConv) |
| **Instance Interaction** | cross_attn (deformable) | `DynamicConv` (动态卷积) |
| **位置编码注入** | attention 内部 (query+query_pos) | 无 (RoI 特征已含空间信息) |
| **多尺度** | 支持 (num_levels, num_points) | 不支持 (单一 RoI 分辨率) |
| **计算复杂度** | O(num_query × num_points × num_levels) | O(num_proposal × dynamic_dim × P²) |

**DynamicConv vs MultiScaleDeformableAttention**：

```
DynamicConv:
  proposals (1,N,C) → dynamic_layer → params (N, num_params)
  roi_feats (P²,N,C) → bmm(param1) → norm → relu → bmm(param2) → norm → relu
  → reshape → out_layer → norm → relu → (1,N,C)

MSDA:
  query (bs,nq,C) → sampling_offsets Linear → offsets (bs,nq,nh,nl,np,2)
  query → attention_weights Linear → softmax → weights
  value → value_proj → grid_sample(sampling_locations) → weighted_sum → output_proj
```

- DynamicConv 是**提案驱动的动态卷积**：每个提案生成专属卷积参数，作用于 RoI 特征
- MSDA 是**参考点驱动的可变形采样**：每个 query 从多尺度特征图上采样多个点并加权

### 6.4 特征提取对比

| 方面 | DiffuDETR | KaryoFlow |
|------|-----------|-----------|
| **特征来源** | 多尺度 FPN 特征图 (flatten) | 多尺度 FPN 特征图 |
| **特征获取** | 可变形注意力从全图采样 | RoIAlign 池化固定区域 |
| **感受野** | 可学习采样点 (num_points × num_levels) | 固定 P×P (默认 7×7) |
| **空间信息** | 通过 reference_points + sampling_offsets | 通过 RoI 网格结构 |
| **参数量** | sampling_offsets + attention_weights + value_proj + output_proj | dynamic_layer + out_layer |

### 6.5 差异对比总表

| 维度 | DiffuDETR | KaryoFlow |
|------|-----------|-----------|
| timestep 编码 | sinusoidal + MLP (256→1024→1024) | sinusoidal + MLP (外部，4C 维度) |
| timestep 调制 | FiLM (TimeStepBlock) | scale_shift 或 AdaLN-Zero |
| 调制位置 | 每层子模块前 (3×6=18 次) | 整块后 (1 次) 或子模块前 (2 次) |
| 自注意力 | nn.MultiheadAttention | nn.MultiheadAttention + SDPA |
| 交叉注意力 | MultiScaleDeformableAttention | DynamicConv (动态卷积) |
| 特征提取 | 可变形采样 (无显式池化) | RoIAlign (显式池化) |
| DN 去噪 | 支持 (含 no_queries 隔离) | 不支持 |
| 两阶段提案 | 支持 (encoder top-k) | 不适用 (外部 proposal) |
| 预测头条件化 | ClassEmbed 有，BBoxEmbed 无 | cls/reg 共享条件化特征 |
| 残差结构 | 有 (x + h) | scale_shift 无，adaln_zero 有 |
| 训练稳定性 | 默认初始化 | adaln_zero 模式 zero-init |

---

## 7. 移植到 mmdet 框架的注意事项

### 7.1 MultiScaleDeformableAttention 的替代方案

DiffuDETR 使用 detrex 的 `MultiScaleDeformableAttention`（含 CUDA 自定义算子）。移植到 mmdet 有以下选项：

| 方案 | 来源 | 优点 | 缺点 |
|------|------|------|------|
| **A. mmcv 自带** | `mmcv.cnn.bricks.transformer.MultiScaleDeformableAttention` | 原生支持，无需额外依赖 | API 略有差异，需适配 forward 签名 |
| **B. mmdet 自带** | `mmdet.models.utils.transformer.` 相关 | 与 mmdet 检测器深度集成 | 可能需要从 DetrTransformerDecoder 适配 |
| **C. 直接移植** | 复制 `multi_scale_deform_attn.py` | 100% 兼容 | 需编译 CUDA 算子或使用 PyTorch 回退 |
| **D. PyTorch 回退** | 使用 `multi_scale_deformable_attn_pytorch` | 纯 PyTorch，无需编译 | 速度慢 (约 2-5x) |

**推荐方案 A**（mmcv 自带），注意以下适配：
- mmcv 版本的 `forward` 签名：`forward(query, key, value, identity, query_pos, key_pos, reference_points, key_padding_mask, spatial_shapes, level_start_index, **kwargs)`
- DiffuDETR 版本额外有 `t` 参数（虽未使用），可安全移除
- mmcv 版本同样有 CUDA 算子 + PyTorch 回退，行为一致

### 7.2 BaseTransformerLayer 的适配

DiffuDETR 的 `BaseTransformerLayer` 改自 mmcv，增加了 `time_step_embed` / `time_step_embed_query` / `t` 参数。移植选项：

| 方案 | 说明 |
|------|------|
| **A. 继承 mmcv 版本** | 子类化 `mmcv.cnn.bricks.transformer.BaseTransformerLayer`，重写 `forward` 加入 timestep 逻辑 |
| **B. 直接复制** | 复制 DiffuDETR 的 `transformer.py`，移除 detrex 依赖 |
| **C. 使用 mmdet TransformerLayerSequence** | 适配 mmdet 的 `DetrTransformerDecoder`，在每层间插入 timestep |

**推荐方案 B**（直接复制），因为 timestep 注入逻辑深度嵌入 forward 循环，子类化重写反而更复杂。注意：
- 移除 `from layers_diffu_detr import` 改为直接导入
- `**kwargs` 中的 `no_queries` 需保留（DN 分离依赖）
- `identity if self.pre_norm else None` 的 pre_norm 逻辑需保持

### 7.3 TimeStepBlock 的移植

TimeStepBlock 是自包含模块，移植简单：
- 依赖：`ldm.modules.diffusionmodules.util.linear` → 替换为 `nn.Linear`
- 依赖：`einops`（实际未使用，可移除 import）
- 维度：`emb_channels` 必须与外部 `time_embed` MLP 输出维度一致（`4 * embed_dim`）

### 7.4 timestep_embedding 的移植

DiffuDETR 使用 LDM 的 `timestep_embedding`（正弦编码）。移植选项：
- **直接复制** `timestep_embedding` 函数（约 20 行，纯 PyTorch）
- 或使用 mmdet/已有 KaryoFlow 中的等价实现（KaryoFlow 已有 sinusoidal embedding）

注意参数：`repeat_only=False`（使用 cos+sin），`max_period=10000`。

### 7.5 DINOTransformer 的适配

这是最复杂的部分。DINOTransformer 整合了：
1. 两阶段提案生成
2. DN queries 拼接
3. Diffusion queries 拼接
4. timestep 编码与注入
5. 推理/训练分支

**移植策略**：

| 组件 | mmdet 对应 | 适配要点 |
|------|-----------|---------|
| Encoder | `mmdet.models.utils.transformer.DetrTransformerEncoder` 或 DeformableDetrTransformerEncoder | 需替换 attention 为 mmcv MSDA |
| Decoder | `DeformableDetrTransformerDecoder` / `DinoTransformerDecoder` | 需加 timestep 注入 + no_queries |
| 两阶段提案 | `DeformableDETR` 的 `gen_encoder_output_proposals` | mmdet 已有等价实现 |
| DN queries | `DINO` 的 `get_contrastive_denoising_training_group` | mmdet 已有 DN 支持 |
| Diffusion queries | **无 mmdet 对应** | 需自行实现 noised boxes 生成与拼接 |
| time_embed MLP | **无 mmdet 对应** | 需自行添加 |
| 推理分支 | **无 mmdet 对应** | 需改造 detector 的 forward 支持 `is_train` 分支 |

### 7.6 具体迁移清单

1. **attention 层**：
   - `MultiheadAttention` → mmcv 版本（API 兼容）
   - `MultiScaleDeformableAttention` → mmcv 版本（移除 `t` 参数）
   - `ConditionalSelfAttention` / `ConditionalCrossAttention` → 不需要（DINO 不用）

2. **transformer 层**：
   - `BaseTransformerLayer` → 复制 DiffuDETR 版本，移除 detrex 依赖
   - `TransformerLayerSequence` → 复制（与 mmcv 版本基本一致）
   - `DINOTransformerEncoder` → 可用 mmdet DeformableDetrTransformerEncoder 替代
   - `DINOTransformerDecoder` → 需修改 mmdet DinoTransformerDecoder，加入 `t` 参数传递
   - `DINOTransformer` → 需较大改造，建议基于 mmdet DINO 改

3. **bbox_embedd 层**：
   - `TimeStepBlock` → 直接复制，`linear` 替换为 `nn.Linear`
   - `BBoxEmbed` → 直接复制（timestep 未实际使用，可简化）
   - `ClassEmbed` → 直接复制（timestep 生效）

4. **依赖清理**：
   - `from layers_diffu_detr import` → 逐个替换为 mmcv/mmdet 导入
   - `from detrex.utils import inverse_sigmoid` → `from mmdet.utils import inverse_sigmoid`
   - `from fairscale.nn.checkpoint import checkpoint_wrapper` → mmcv 的 `checkpoint_wrapper` 或 `torch.utils.checkpoint`
   - `from .ldm.modules.diffusionmodules.util import timestep_embedding, linear` → 复制函数或用等价实现

### 7.7 关键风险点

1. **no_queries 机制的 DN 兼容性**：mmdet 的 DN 实现将 DN queries 与正常 queries 拼接，需确认拼接顺序与 `no_queries[0]` 索引一致（DN 在前，diffusion 在后）。

2. **reference_points 维度**：DiffuDETR 的 reference_points 最后一个维度可为 2 或 4（MSDA 支持），mmdet 的 DINO 通常用 4（box）。需确认 MSDA 分支逻辑一致。

3. **推理分支的外部循环**：DiffuDETR 推理时 `is_train=False` 提前返回，扩散去噪循环在外部。需在 mmdet detector 层实现该循环，且每次调用 transformer 时传入不同的 `query_embeds_diffusion`（逐步去噪的 boxes）。

4. **time_embed 维度**：输出 `4 * embed_dim`（1024），而非 `embed_dim`（256）。TimeStepBlock 的 `emb_channels` 必须匹配，否则 `emb_layers` 的 Linear 维度报错。

5. **ClassEmbed emb_channels 硬编码**：`emb_channels = 4 * embed_dim` 硬编码，而 BBoxEmbed 由参数传入。移植时需统一为 `4 * embed_dim` 并确保外部 `time_embed` 输出匹配。

6. **BBoxEmbed 的 timestep 未启用**：源码中 BBoxEmbed 的 timestep 注入被注释。若移植后希望 bbox 也条件化，需取消注释并确认 `time_embed_channels` 维度正确。

7. **checkpoint_wrapper 依赖**：fairscale 的 `checkpoint_wrapper` 与 mmcv/torch 原生梯度检查点 API 不同，需替换并测试显存/速度。

---

## 附录：关键文件路径

### DiffuDETR（远程仓库）
- `layers_diffu_detr/transformer.py` — BaseTransformerLayer, TransformerLayerSequence
- `layers_diffu_detr/attention.py` — MultiheadAttention, ConditionalSelfAttention, ConditionalCrossAttention
- `layers_diffu_detr/multi_scale_deform_attn.py` — MultiScaleDeformableAttention (含 CUDA 算子)
- `layers_diffu_detr/__init__.py` — 包导出定义
- `projects/diffu_dino/modeling/dino_transformer.py` — DINOTransformer, DINOTransformerEncoder, DINOTransformerDecoder
- `projects/diffu_dino/modeling/bbox_embedd.py` — TimeStepBlock, BBoxEmbed, ClassEmbed
- `projects/diffu_dino/modeling/ldm/modules/diffusionmodules/util.py` — timestep_embedding, linear, zero_module

### KaryoFlow（本地）
- `/home/linkst/workspace/chromosome-kd/ldmdet/core/single_head.py` — SingleDiffusionDetHead (含 scale_shift / adaln_zero 两种条件化)
- `/home/linkst/workspace/chromosome-kd/ldmdet/core/dynamic_conv.py` — DynamicConv (动态卷积实例交互)

---

*报告结束*
