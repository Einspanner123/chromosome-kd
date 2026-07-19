# SC-RF: Self-Conditioned Rectified Flow for Dense Detection

> **方向类型**: 核心方法方向（训练改进，可独立投稿）
> **可叠加方向**: PD-RF（SC-RF 作为 teacher 提供更好的轨迹供 PD-RF 蒸馏）
> **目标会议**: MICCAI 2026 / IEEE TMI
> **预期增益**: +0.005~0.015 mAP（基于图像生成领域自条件化的一致性增益）
>
> **实验状态**: ❌ 已归档（2026-07-11，负增益证伪）
> - Best mAP = 0.860（Epoch 82/150，Early Stop at Epoch 112）
> - 相比 A4 baseline (0.862) **负增益 -0.002**，未达预期 +0.005~0.015
> - 自条件化机制本身实现正确（50% 激活、零初始化过渡、校正幅度增长均正常）
> - 详细复盘见 [第 8 节 实验结果与复盘](#8-实验结果与复盘)

---

## 1. 问题动机

### 1.1 当前局限性

当前 RF 检测器的单步预测 $x_0^{pred} = f_\theta(x_t)$ 是一次性映射：模型从中间状态 $x_t$ 直接预测目标 $x_0$，没有利用自身的预测历史。

在多步采样（Heun/DPM-Solver++）中，每一步的预测是独立的——step $k$ 的模型不知道 step $k-1$ 预测了什么。这浪费了迭代精炼（iterative refinement）的潜力。

### 1.2 核心洞察

**让模型条件化于自身的上一步预测 $\hat{x}_0^{prev}$，实现迭代精炼。**

- 标准 RF: $x_0^{pred} = f_\theta(x_t)$
- SC-RF: $x_0^{pred} = f_\theta(x_t, \hat{x}_0^{prev})$

模型从"给定 $x_t$ 预测 $x_0$"变为"给定 $x_t$ 和粗略预测 $\hat{x}_0^{prev}$，输出精炼预测"。这是一个残差学习问题：模型学习校正 $\Delta(x_t, \hat{x}_0^{prev}) = x_0 - \hat{x}_0^{prev}$，而非从头预测 $x_0$。

### 1.3 与已尝试方向的区别

| 方向 | 机制 | 结果 | SC-RF 的区别 |
|---|---|---|---|
| CC-RF | 条件化于 N（近常数） | 理论收益≈0 | 条件化于 $\hat{x}_0^{prev}$（高信息量） |
| TRD/CAT/LSAS | 添加辅助损失 | 梯度冲突 | 不添加辅助损失，仅修改前向传播 |
| Reflow | 两阶段重训练 | Epoch 1 后退化 | 单阶段训练，无额外损失 |

**关键**: SC-RF 不引入任何辅助损失，仅修改模型前向传播（增加输入），避免了梯度冲突问题。

---

## 2. 理论分析

> **定位声明**: 本节为动机分析（motivation analysis），非严格定理。自条件化是图像生成领域已验证的成熟技术（Chen et al., 2022），本方案将其迁移到检测 RF。以下分析提供理论动机，不声称为新颖理论贡献。

### 2.1 设定

- **标准 RF**: 学习 $f_\theta(x_t) \approx \mathbb{E}[X_0 | X_t]$
- **SC-RF**: 学习 $f_\theta(x_t, \hat{x}_0) \approx \mathbb{E}[X_0 | X_t, \hat{X}_0]$

其中 $\hat{X}_0$ 是模型在**上一步**的预测（推理时）或无梯度前向的预测（训练时）。

### 2.2 动机分析 1：条件化不增加熵

**性质**: $H(X_0 | X_t, \hat{X}_0) \leq H(X_0 | X_t)$

由条件互信息非负性 $I(X_0; \hat{X}_0 | X_t) \geq 0$ 直接得到。

**重要限定**: 
- **训练时** $\hat{X}_0 = f_\theta(X_t)$ 是 $X_t$ 的确定函数，$\sigma(\hat{X}_0) \subseteq \sigma(X_t)$，故 $I(X_0; \hat{X}_0 | X_t) = 0$，等式成立——条件化不提供额外信息。
- **推理时** $\hat{X}_0^{(k)}$ 来自不同时间步 $X_{t_{k-1}}$，非 $X_{t_k}$ 的函数，此时 $I(X_0; \hat{X}_0^{(k)} | X_{t_k})$ 可严格大于零。

此性质仅说明"条件化不损害"（不会增加条件熵），不直接保证"有增益"。增益的来源是推理时的跨时间步信息（见 2.3）和训练时的残差学习机制（见 2.4）。

### 2.3 动机分析 2：推理时的跨时间步信息

**推理时**，$\hat{X}_0^{(k)} = f_\theta(X_{t_{k-1}}, \hat{X}_0^{(k-1)})$ 来自**上一步**的中间状态 $X_{t_{k-1}}$，而非当前的 $X_{t_k}$。

设 $X_{t_1} \to X_{t_2}$ 为一步 Euler 求解器更新（$t_1 > t_2$）：
- $\hat{X}_0^{(1)} = f_\theta(X_{t_1}, 0)$
- $X_{t_2} = X_{t_1} + (t_2 - t_1) \cdot \frac{X_{t_1} - \hat{X}_0^{(1)}}{t_1}$（RF Euler step）

定义复合映射 $h(X_{t_1}) = X_{t_2} = S(X_{t_1}, f_\theta(X_{t_1}))$，其中 $S$ 是求解器。

**经验论据**（非严格证明）：$h: \mathbb{R}^4 \to \mathbb{R}^4$ 的非单射性依赖于 $f_\theta$ 和 $S$ 的具体形式。在实际系统中，以下因素使 $h$ 大概率非单射：
1. **box_renewal 随机性**：推理时 `apply_box_renewal` 对低置信度框重采样，引入额外随机性
2. **数值精度**：FP32 精度下，不同 $X_{t_1}$ 可能映射到相同 $X_{t_2}$
3. **模型非线性**：$f_\theta$ 含多层 attention + FFN，$h$ 的 Jacobian 行列式可零

当 $h$ 非单射时，$\hat{X}_0^{(1)}$ 携带了被 $h$ 丢弃的信息，$I(X_0; \hat{X}_0^{(1)} | X_{t_2}) > 0$ 可成立。

**严格证明的困难**：形式化需要分析 $h$ 的 Jacobian 性质，超出本方案范围。我们依赖以下经验事实支撑：图像生成领域自条件化在多步采样中一致改善质量（Chen et al., 2022; 多篇 RF 论文）。

### 2.4 训练时的残差学习视角

训练时 $\hat{X}_0 = f_\theta(X_t)$（无梯度），虽然 $\hat{X}_0$ 是 $X_t$ 的确定函数（信息论意义上零增益），但模型学习的是**残差函数**：

$$f_\theta(X_t, \hat{X}_0) = \hat{X}_0 + \Delta_\theta(X_t, \hat{X}_0)$$

其中 $\Delta_\theta$ 是校正项。这与 ResNet 的残差学习原理一致：学习 $x_0 - \hat{x}_0$ 比直接学习 $x_0$ 更容易，因为 $\hat{x}_0$ 已经是 $x_0$ 的粗略估计。

**关键**：残差学习是**优化景观**论证，与 2.2-2.3 的**信息论**论证是不同视角。即使训练时信息论增益为零，残差学习仍可改善优化：
- 模型学习纠正自身预测误差 $\Delta = X_0 - f_\theta(X_t)$
- 这是一个更平滑的优化目标（残差通常比原值更小、更集中）
- 零初始化保证 $\Delta_\theta$ 从 0 开始增长，不破坏已训练特征

### 2.5 训练-推理分布失配及缓解

**失配问题**：训练时 $\hat{X}_0^{train} = f_\theta(X_t)$（同一时间步，确定函数），推理时 $\hat{X}_0^{infer} = f_\theta(X_{t_{k-1}})$（不同时间步）。两者分布不同，模型可能未学会利用推理时的 $\hat{X}_0^{infer}$。

**缓解机制**：
1. **零初始化 + 50% 切换**：50% 训练样本使用 $\hat{X}_0 = 0$（无自条件化），模型必须学会在无 $\hat{X}_0$ 时也能工作。这保证即使推理时 $\hat{X}_0^{infer}$ 分布偏移，模型也有 fallback 路径。
2. **残差学习的跨分布泛化**：模型学习的是校正函数 $\Delta_\theta(X_t, \hat{X}_0)$。校正能力（"给定粗略预测，输出精炼预测"）是相对通用的技能，可跨分布迁移。
3. **渐进式启用**：训练初期 $\hat{X}_0^{train} = 0$（零初始化），模型先学会标准 RF；随着 $\hat{X}_0^{train}$ 逐渐非零，模型渐进学习利用自条件化。

**残留风险**：若推理时 $\hat{X}_0^{infer}$ 与训练时 $\hat{X}_0^{train}$ 差异过大，自条件化可能无效果。此时退化为标准 RF（$\hat{X}_0 = 0$ 路径），不会损害性能。

### 2.6 级联 Head 与自条件化的交互

当前架构有 6 个级联 head（`head_series`），每个 head 接收上一个 head 的 `pred_bboxes`（detached）作为输入，在**单时间步内**做迭代精炼。SC-RF 的 $\hat{x}_0^{prev}$ 是**跨时间步**的精炼信号。

**交互分析**：
- 级联 head 精炼：同一 $x_t$ 下，逐步改善预测质量（intra-step）
- SC-RF 精炼：跨 $x_t$ 间，利用历史预测提供额外信息（inter-step）
- 两者作用层面不同，互补而非冗余

**实现策略**：$\hat{x}_0^{prev}$ 注入到**每个**级联 head 的 proposal features 上。这允许每个 head 都利用历史信息，而非仅第一个 head。级联 head 的 detach 机制不影响 $\hat{x}_0^{prev}$（$\hat{x}_0^{prev}$ 来自无梯度前向，本身就是 detached 的）。

### 2.7 与 DPM-Solver++ 的兼容性

DPM-Solver++ 的半线性形式：

$$\frac{dx_t}{dt} - \frac{1}{t}x_t = -\frac{1}{t}x_0^{pred}(x_t, t, \hat{x}_0^{prev})$$

$\hat{x}_0^{prev}$ 作为 $x_0^{pred}$ 的额外输入，不影响指数积分器的数学结构。每步求解器调用模型时传入上一步的 $x_0^{pred}$ 作为 $\hat{x}_0^{prev}$，$O(\Delta t^3)$ 截断误差保持。

---

## 3. 架构设计

### 3.1 Self-Conditioning 注入点

在 `SingleDiffusionDetHead` 中，将 $\hat{x}_0^{prev}$ 投影到特征空间并加到 proposal features 上：

```
x0_prev (bs, num_proposals, 4) → Linear(4, feat_channels) → x0_prev_emb (bs, num_proposals, feat_channels)
proposals = proposals + x0_prev_emb  (在 self-attention 之前)
```

**零初始化**: `Linear(4, feat_channels)` 的 weight 和 bias 均零初始化，确保初始时 `x0_prev_emb = 0`，SC-RF 等价于标准 RF。

### 3.2 训练流程

```python
def loss(self, features, img_metas, gt_bboxes, gt_labels):
    # ... 构建 x_noisy, targets 等
    
    # Self-conditioning: 50% 概率使用
    if self.training and self.use_self_conditioning:
        if random.random() < self.self_conditioning_prob:
            with torch.no_grad():
                # 无梯度前向获取 x0_pred
                cls_logits, pred_bboxes, _ = self(features, curr_bboxes, t_input)
                # pred_bboxes 是 xyxy 格式, 转换为 raw 空间
                x0_pred_prev = self._sampler.xyxy_to_raw(pred_bboxes[-1], img_metas)
        else:
            x0_pred_prev = torch.zeros_like(x_noisy_batch)
    else:
        x0_pred_prev = torch.zeros_like(x_noisy_batch)
    
    # 带自条件化的前向
    all_cls_logits, all_pred_bboxes, _ = self(features, curr_bboxes, t_input, x0_pred_prev)
    # ... 计算 loss
```

### 3.3 推理流程

```python
def predict(self, features, img_metas, ...):
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
    x0_pred_prev = torch.zeros_like(x_raw)  # 第一步无历史
    
    for step_idx, (t_curr, t_next) in enumerate(time_pairs):
        cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
            features, x_raw, t_curr, img_metas, x0_pred_prev
        )
        x0_pred_prev = x0_raw  # 传给下一步
        # ... solver step
```

### 3.3.1 Heun 求解器兼容性

Heun 二阶步在中间点调用 `model_fn(x_next_euler, t_next)` 进行第二次模型求值。SC-RF 需在 `model_fn` 闭包中传递 `x0_prev`：

```python
# head.py predict 方法中 Heun 分支
elif self.solver_type == 'heun' and t_next > 0:
    def model_fn(x_tmp, t_tmp):
        # x0_prev 传递: 使用当前步的 x0_raw 作为 Heun 子步的 x0_prev
        _, _, x0_tmp = self._forward_at_t(features, x_tmp, t_tmp, img_metas, x0_pred_prev)
        return x0_tmp, None
    x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
```

Heun 子步中的 `x0_prev` 使用当前步的 `x0_raw`（即上一步模型的完整预测），保持与 Euler/DPM++ 一致的语义。

### 3.4 监控指标（SwanLab 插桩）

训练时记录：
- `sc_x0_prev_norm`: $\hat{x}_0^{prev}$ 投影后的 L2 范数（应从 0 逐渐增长）
- `sc_correction_magnitude`: $||x_0^{pred} - \hat{x}_0^{prev}||_2$（校正项大小）
- `sc_active_rate`: 训练中使用自条件化的比例（应≈0.5）

推理时记录：
- `sc_refinement_gain`: 多步推理中各步预测的变化量 $||x_0^{(k)} - x_0^{(k-1)}||_2$

---

## 4. 实现计划

### 4.1 代码修改清单

| 文件 | 修改内容 |
|---|---|
| `ldmdet/core/single_head.py` | 添加 `x0_prev_proj`（零初始化 Linear），`forward` 接受 `x0_prev` 参数 |
| `ldmdet/core/head.py` | `forward` 接受 `x0_prev` 并传给 `head_series`；`loss` 中实现 50% 自条件化训练；`predict`/`_forward_at_t` 传递上一步 $x_0$ |
| `experiments/configs/ldmdet/directions/sc_rf/` | 新建 SC-RF 配置 |
| `tests/unit/test_self_conditioning.py` | 新建单元测试 |

### 4.2 关键实现细节

**`SingleDiffusionDetHead.__init__` 新增**:
```python
if use_self_conditioning:
    self.x0_prev_proj = nn.Linear(4, feat_channels)
    nn.init.zeros_(self.x0_prev_proj.weight)
    nn.init.zeros_(self.x0_prev_proj.bias)
```

**`SingleDiffusionDetHead.forward` 修改**:
```python
def forward(self, features, bboxes, proposals, pooler, time_emb, x0_prev=None):
    # ... roi_features, proposals 提取
    if self.use_self_conditioning and x0_prev is not None:
        x0_prev_emb = self.x0_prev_proj(x0_prev)  # [bs, num_boxes, feat_channels]
        proposals = proposals + x0_prev_emb
    # ... 后续不变
```

**`DiffusionDetHead.forward` 修改**:
```python
def forward(self, features, bboxes, t, x0_prev=None):
    time_emb = self.time_mlp(t)
    # ...
    for head in self.head_series:
        result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb, x0_prev)
        # ...
```

### 4.3 测试计划

**单元测试** (`tests/unit/test_self_conditioning.py`):

1. `test_x0_prev_proj_zero_init`: `x0_prev_proj` 零初始化
2. `test_zero_x0_prev_equals_standard_rf`: $\hat{x}_0=0$ 时输出与标准 RF 一致
3. `test_nonzero_x0_prev_changes_output`: 非零 $\hat{x}_0$ 改变输出
4. `test_different_x0_prev_different_output`: 不同 $\hat{x}_0$ 产生不同预测
5. `test_gradient_flow_x0_prev_proj`: `x0_prev_proj` 梯度正常回传
6. `test_training_self_conditioning_toggle`: 50% 概率启/禁用自条件化
7. `test_inference_multi_step_refinement`: 多步推理中 $\hat{x}_0$ 正确传递
8. `test_dpm_solver_compatibility`: SC-RF + DPM-Solver++ 兼容
9. `test_heun_compatibility`: SC-RF + Heun 兼容
10. `test_residual_learning_property`: 校正项 $\Delta$ 随训练减小（需要模拟训练步）

### 4.4 配置设计

```python
# experiments/configs/ldmdet/directions/sc_rf/sc_rf_24obj.py
_base_ = ['../../ldmdet_rf_heun_shifted_bs8.py']
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='dpm_solver_pp',
        sampling_timesteps=4,
        single_head=dict(
            time_conditioning='adaln_zero',
            use_self_conditioning=True,
        ),
        coupling=dict(type='ot_flow', epsilon=5.0, num_iters=20, coupling_mode='multinomial'),
        use_self_conditioning=True,
        self_conditioning_prob=0.5,
    ),
)
```

---

## 5. 实验设计

### 5.1 主实验

| 实验 | 配置 | 预期 mAP | 目的 |
|---|---|---|---|
| A4 baseline | RF+AdaLN+StochOT+DPM++ (无 SC) | 0.862 | 基线 |
| **SC-RF** | A4 + self_conditioning | **0.867~0.877** | 验证自条件化增益 |

### 5.2 消融实验

| 消融 | 目的 |
|---|---|
| SC-RF vs A4 (24obj) | 自条件化的增益 |
| SC-RF prob=0.0/0.25/0.5/0.75/1.0 | 最优自条件化概率 |
| SC-RF + Heun vs SC-RF + DPM++ | 自条件化与求解器的交互 |
| SC-RF 1-step vs 4-step | 自条件化在少步采样中的效果 |

### 5.3 迭代精炼验证

对 50 张验证图，记录 4 步推理中各步的 $x_0^{pred}$：
- 计算 $||x_0^{(k)} - x_0^{(k-1)}||_2$（校正量）
- 验证校正量递减（收敛性）
- 对比标准 RF 的各步变化量（SC-RF 应更平滑）

---

## 6. 预期贡献

1. **方法**: 首次将自条件化应用于检测 RF，实现迭代精炼式检测
2. **分析**: 从信息论（条件熵不增）和优化（残差学习）双视角提供动机分析，并坦诚讨论训练-推理分布失配及其缓解机制
3. **实践**: 零初始化保证平滑过渡，不引入辅助损失，避免梯度冲突；50% 切换提供 fallback 路径

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 增益不显著 | 中 | 高 | 图像生成领域自条件化一致性增益 +0.005~0.015，但检测 4 维 bbox 信息量低于图像像素，需实验验证 |
| 训练不稳定 | 低 | 中 | 零初始化 + 50% 概率切换，初始等价于标准 RF |
| 训练速度下降 | 中 | 低 | 50% 样本需 2x 前向，平均 1.5x；AMP 可部分补偿 |
| 训练-推理失配 | 中 | 中 | 50% 零输入训练提供 fallback；残差学习跨分布泛化；最坏退化为标准 RF |
| 4 维输入信号不足 | 中 | 中 | 可扩展为附加 cls_logits 或使用 MLP 投影；需消融验证 |

---

## 8. 实验结果与复盘

> **归档日期**: 2026-07-11
> **实验配置**: `experiments/configs/ldmdet/directions/sc_rf/sc_rf_24obj.py`
> **训练日志**: `work_dirs/sc_rf_24obj/20260709_214943/20260709_214943.log`
> **Best checkpoint**: `work_dirs/sc_rf_24obj/best_coco_bbox_mAP_epoch_82.pth`

### 8.1 训练状态

| 项目 | 值 |
|------|-----|
| 训练时间 | 2026-07-09 21:49 → 2026-07-11 01:04（约 27 小时） |
| 完成 Epoch | 112 / 150（EarlyStoppingHook 触发） |
| 早停原因 | "monitored metric did not improve in the last 30 records. best score: 0.860" |
| Early Stop 配置 | patience=30, min_delta=0.001, monitor='coco/bbox_mAP' |
| Best mAP | **0.860**（Epoch 82） |
| Last mAP | 0.856（Epoch 112） |

### 8.2 性能对比

| 实验 | 采样器 | 步数 | mAP | 与 SC-RF best 差距 |
|------|--------|------|-----|-------------------|
| A1 (baseline) | Euler | 1 步 | 0.856 | -0.004 |
| **SC-RF (best)** | **DPM-Solver++** | **4 步** | **0.860** | **0** |
| 理论目标 | - | - | ≥0.861 | +0.001 |
| A4 (直接基线) | DPM-Solver++ | 4 步 | 0.862 | +0.002 |

**核心结论**: SC-RF 构建在 A4 之上（`_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']`），使用相同采样器与步数。相比 A4 **负增益 -0.002**，未达到预期 +0.005~0.015 mAP。相比 A1（1 步 Euler）的 +0.004 增益主要来自采样器升级（Euler→DPM-Solver++），非自条件化贡献。

### 8.3 mAP 趋势分析

**阶段划分**:
- Epoch 1-9: 快速收敛（0.000 → 0.824）
- Epoch 10-43: 缓慢提升（0.819 → 0.856）
- Epoch 44-82: 平台期（0.828 → 0.860，38 epoch 仅 +0.004）
- Epoch 83-112: 完全平台期（0.849-0.859 波动，30 epoch 未超越 0.860）

**最近 20 epoch（93-112）均值约 0.853**，在 0.849-0.859 间窄幅震荡，明确进入平台期。

### 8.4 自条件化机制工作状态：✅ 完全正常

机制实现验证通过，问题不在实现层面：

| 指标 | 训练早期（Ep2） | 训练后期（Ep111-112） | 结论 |
|------|----------------|----------------------|------|
| sc_active 激活比例 | 49.9%（1957/3920） | - | ✅ 50% 概率精确生效 |
| x0_prev_norm（激活时） | ~2.45 | ~2.5-2.7 | ✅ 信号有效传递 |
| correction_mag（激活时） | **~0.01** | **~0.93-1.08** | ✅ 零初始化→逐步增长 |
| correction_mag（未激活时） | 0.003 | 0.195 | ✅ fallback 路径稳定 |

**关键观察**:
1. 50% 概率切换精确生效（49.9% 实测激活率）
2. 零初始化设计有效——初期校正幅度≈0.01，等价于标准 RF（符合 3.1 节设计）
3. 校正幅度从 ~0.01 增长到 ~1.0，证明模型确实学会利用 x0_prev 信号
4. 两条路径（激活/未激活）都正常工作

### 8.5 理论 vs 实际偏差分析

[第 7 节风险评估](#7-风险评估)中预判的风险项实际命中情况：

| 风险项（原评估） | 概率 | 实际命中 | 说明 |
|---|---|---|---|
| 增益不显著 | 中 | ✅ 命中 | 实际为负增益，比"不显著"更严重 |
| 训练不稳定 | 低 | ✅ 未命中 | 零初始化+50%切换保证了训练稳定 |
| 训练速度下降 | 中 | ✅ 命中 | 50% 样本需 2x 前向，但可接受 |
| 训练-推理失配 | 中 | ⚠️ 可能命中 | 见下方分析 |
| **4 维输入信号不足** | 中 | ✅ **命中（主因）** | 检测 bbox 仅 4 维，信息量远低于图像像素 |

**偏差原因深度分析**:

1. **4 维输入信号不足（主因）**:
   - 图像生成中自条件化的 $\hat{x}_0$ 是高维像素张量（如 3×256×256），信息量大
   - 检测任务中 $\hat{x}_0$ 仅 4 维（cxcywh），信息量极低
   - 即使模型学会利用 x0_prev（correction_mag 从 0 增长到 1.0），可利用的额外信息有限
   - 这与 [2.2 节](#22-动机分析-1条件化不增加熵)的信息论分析一致：训练时 $I(X_0; \hat{X}_0 | X_t) = 0$，推理时的跨时间步信息在 4 维空间中极为有限

2. **强基线饱和**:
   - A4（4 步 DPM-Solver++）已达 0.862，多步采样本身的迭代精炼已较充分
   - [2.6 节](#26-级联-head-与自条件化的交互)分析的 intra-step（级联 head）vs inter-step（SC-RF）互补性，在 6 个级联 head + 4 步 DPM-Solver++ 的强基线上冗余度高
   - 4 步采样已经提供了跨时间步的精炼，SC-RF 的额外跨步信息边际收益递减

3. **训练-推理分布失配的残留影响**:
   - [2.5 节](#25-训练-推理分布失配及缓解)讨论的失配问题：训练时 $\hat{X}_0^{train}=f_\theta(X_t)$ 是同时间步确定函数，推理时 $\hat{X}_0^{infer}$ 来自不同时间步
   - 虽然 50% 零输入 fallback 保证了不损害性能，但也限制了增益上限
   - 模型可能未充分学会利用推理时分布偏移的 $\hat{X}_0^{infer}$

4. **额外训练复杂度**:
   - x0_prev_proj 引入额外参数，在 150 epoch 训练预算下可能未充分优化
   - 50% 样本的 2x 前向增加了训练方差

### 8.6 实验结论

**定性判定: 机制实现正确，但性能未达预期（负增益），方向证伪。**

| 维度 | 结论 |
|------|------|
| 训练状态 | ✅ 已完成（Early Stop，112/150 epoch） |
| 自条件化机制 | ✅ 完全正常（50% 激活、零初始化过渡、校正幅度增长） |
| 性能目标达成 | ❌ 未达到目标（0.860 < 0.861） |
| 相对 A4 增益 | ❌ 负增益（-0.002） |
| 相对 A1 增益 | ⚠️ +0.004，但主要来自采样器升级，非自条件化贡献 |

**核心教训**:
1. **图像生成→检测的迁移需谨慎评估信号维度差异**: 自条件化在图像生成中的成功依赖于高维 $\hat{x}_0$ 的丰富信息，检测 4 维 bbox 的信息量不足以支撑同等增益
2. **强基线上的边际收益递减**: A4（4 步 DPM-Solver++ + 6 级联 head）已提供充分的迭代精炼，SC-RF 的跨步信息冗余
3. **机制正确 ≠ 性能提升**: 自条件化机制的所有技术指标均正常（激活率、零初始化、校正幅度增长），但仍未带来性能增益，说明问题在于任务本身的信息瓶颈而非实现

### 8.7 未探索的改进方向（仅供参考，不再实施）

因方向已证伪归档，以下改进方向仅作记录，不再实施：

1. **降低 self_conditioning_prob**: 50% 可能在强基线上引入过多训练/推理不一致，可尝试 0.2-0.3
2. **结合 1 步 Euler 采样器**: SC-RF 理论优势在于迭代精炼，在 4 步 DPM-Solver++ 上优势被稀释，可尝试在 A1（1 步 Euler）基础上验证——但 1 步无"跨时间步"信息，自条件化退化为同时间步残差，理论上无增益（见 [2.2 节](#22-动机分析-1条件化不增加熵)）
3. **扩展 x0_prev 维度**: 附加 cls_logits 或使用 MLP 投影增加信息量——但这改变了自条件化的简洁性，且收益不确定
4. **延长训练**: 平台期明确，延长训练无收益

### 8.8 对 PD-RF 的影响

原计划 [第 1 节](#1-问题动机)中"SC-RF 作为 teacher 提供更好的轨迹供 PD-RF 蒸馏"已不适用：
- SC-RF (0.860) 低于 A4 (0.862)，作为 teacher 不会优于 A4
- PD-RF 继续使用 A4 作为 teacher（见 [PD-RF 方案](./PD-RF_Progressive_Distillation.md)）

---

## 9. 归档记录

- **归档日期**: 2026-07-11
- **归档原因**: 负增益证伪（SC-RF best 0.860 < A4 baseline 0.862）
- **实验代码**: `experiments/configs/ldmdet/directions/sc_rf/sc_rf_24obj.py`（保持不动，不再修改）
- **Best checkpoint**: `work_dirs/sc_rf_24obj/best_coco_bbox_mAP_epoch_82.pth`（保留）
- **训练日志**: `work_dirs/sc_rf_24obj/20260709_214943/20260709_214943.log`（保留）
- **SwanLab**: 项目 'ldmdet-breakthrough', 实验 'sc_rf_24obj'（已停止）
- **后续方向**: PD-RF 继续使用 A4 作为 teacher，不考虑 SC-RF 作为 teacher
