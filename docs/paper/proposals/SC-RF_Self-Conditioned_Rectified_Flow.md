# SC-RF: Self-Conditioned Rectified Flow for Dense Detection

> **方向类型**: 核心方法方向（训练改进，可独立投稿）
> **可叠加方向**: PD-RF（SC-RF 作为 teacher 提供更好的轨迹供 PD-RF 蒸馏）
> **目标会议**: MICCAI 2026 / IEEE TMI
> **预期增益**: +0.005~0.015 mAP（基于图像生成领域自条件化的一致性增益）

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

## 2. 理论推导

### 2.1 设定

- **标准 RF**: 学习 $f_\theta(x_t) \approx \mathbb{E}[X_0 | X_t]$
- **SC-RF**: 学习 $f_\theta(x_t, \hat{x}_0) \approx \mathbb{E}[X_0 | X_t, \hat{X}_0]$

其中 $\hat{X}_0$ 是模型在**上一步**的预测（推理时）或无梯度前向的预测（训练时）。

### 2.2 命题 1（条件熵不增）

**声明**: $H(X_0 | X_t, \hat{X}_0) \leq H(X_0 | X_t)$

**证明**: 由条件互信息的非负性：

$$I(X_0; \hat{X}_0 | X_t) = H(X_0 | X_t) - H(X_0 | X_t, \hat{X}_0) \geq 0$$

故 $H(X_0 | X_t, \hat{X}_0) \leq H(X_0 | X_t)$。$\square$

**注**: 此处不需要 $\hat{X}_0$ 与 $X_t$ 独立。即使 $\hat{X}_0 = g(X_t)$（训练时的情形），条件互信息 $I(X_0; \hat{X}_0 | X_t) \geq 0$ 仍然成立——这是信息论基本定理，无条件独立要求。

### 2.3 命题 2（推理时的额外信息增益）

**推理时**，$\hat{X}_0^{(k)} = f_\theta(X_{t_{k-1}}, \hat{X}_0^{(k-1)})$ 来自**上一步**的中间状态 $X_{t_{k-1}}$，而非当前的 $X_{t_k}$。

**声明**: 在多步推理中，$I(X_0; \hat{X}_0^{(k)} | X_{t_k}) > 0$（严格大于零）。

**论证**:

设 $X_{t_1} \to X_{t_2}$ 为一步求解器更新（$t_1 > t_2$），则：
- $X_{t_2} = \text{solver\_step}(X_{t_1}, \hat{X}_0^{(1)})$
- $\hat{X}_0^{(1)} = f_\theta(X_{t_1}, 0)$

关键点：$X_{t_2}$ 是 $X_{t_1}$ 和 $\hat{X}_0^{(1)}$ 的函数，但**不是充分统计量**。$\hat{X}_0^{(1)}$ 包含了模型对 $X_{t_1}$ 的完整解读，而 $X_{t_2}$ 仅是这一解读的部分投射（通过求解器的一步投影）。

形式化：设 $S = \text{solver\_step}$ 为确定函数，$X_{t_2} = S(X_{t_1}, \hat{X}_0^{(1)})$。若 $S$ 不是单射（实际中 $S$ 将 $(X_{t_1}, \hat{X}_0^{(1)}) \in \mathbb{R}^8 \to X_{t_2} \in \mathbb{R}^4$，必然非单射），则存在信息损失。$\hat{X}_0^{(1)}$ 携带了被 $S$ 投影丢弃的信息，故：

$$I(X_0; \hat{X}_0^{(1)} | X_{t_2}) \geq I(X_0; \hat{X}_0^{(1)} | X_{t_1}, \hat{X}_0^{(1)}) = 0$$

在模型有预测能力且求解器有信息损失时，严格不等式成立。$\square$

### 2.4 训练时的残差学习视角

训练时 $\hat{X}_0 = f_\theta(X_t)$（无梯度），虽然 $\hat{X}_0$ 是 $X_t$ 的确定函数，但模型学习的是**残差函数**：

$$f_\theta(X_t, \hat{X}_0) = \hat{X}_0 + \Delta_\theta(X_t, \hat{X}_0)$$

其中 $\Delta_\theta$ 是校正项。这与 ResNet 的残差学习原理一致：学习 $x_0 - \hat{x}_0$ 比直接学习 $x_0$ 更容易，因为 $\hat{x}_0$ 已经是 $x_0$ 的粗略估计。

**零初始化保证**: $\hat{X}_0$ 的投影层零初始化，使得训练初期 $\Delta_\theta = 0$，SC-RF 退化为标准 RF。随着训练进行，$\Delta_\theta$ 逐渐学习有效的校正。

### 2.5 与 DPM-Solver++ 的兼容性

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
        if random.random() < 0.5:
            with torch.no_grad():
                # 无梯度前向获取 x0_pred
                cls_logits, pred_bboxes, _ = self(features, curr_bboxes, t_input)
                x0_pred_prev = self._extract_x0_pred(pred_bboxes, img_metas)
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
    x_raw = torch.randn(bs, num_proposals, 4, device=device)
    x0_pred_prev = torch.zeros_like(x_raw)  # 第一步无历史
    
    for step_idx, (t_curr, t_next) in enumerate(time_pairs):
        cls_logits, pred_bboxes, x0_raw = self._forward_at_t(
            features, x_raw, t_curr, img_metas, x0_pred_prev
        )
        x0_pred_prev = x0_raw  # 传给下一步
        # ... solver step
```

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
2. **理论**: 命题 1-2 给出条件熵不增和推理时额外信息增益的严格论证
3. **实践**: 零初始化保证平滑过渡，不引入辅助损失，避免梯度冲突

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 增益不显著 | 低 | 高 | 图像生成领域自条件化一致性增益 +0.005~0.015，检测场景预期类似 |
| 训练不稳定 | 低 | 中 | 零初始化 + 50% 概率切换，初始等价于标准 RF |
| 训练速度下降 | 中 | 低 | 50% 样本需 2x 前向，平均 1.5x；AMP 可部分补偿 |
| 多步推理无改善 | 低 | 中 | 命题 2 保证推理时有额外信息增益 |
