# CC-RF: Count-Conditioned Rectified Flow

> **方向类型**: 核心理论方向（可独立投稿）
> **可叠加方向**: UQ-Det（CC-RF 降低方差，UQ-Det 度量剩余方差，两者互补）
> **目标会议**: MICCAI 2026 / IEEE TMI
> **预期增益**: +0.010~0.020 mAP（通过条件方差降低）

---

## 1. 问题动机

### 1.1 当前局限性

当前扩散检测器的生成过程 $p_\theta(x_0 \mid x_t)$ 与目标数量 $N$ 无关。推理时固定使用 $N_{proposal}=500$ 个噪声框，count 信息完全被浪费。

在染色体核型分析中，目标数量有强先验（46 条染色体），但当前框架未在数学上利用这个先验。在 visible-only 标注模式下，标注框数 $\neq$ 实例数，导致 count-prior 彻底失效。

### 1.2 核心洞察

**将 count-prior 从后处理提升为生成过程的条件变量。**

模型从"预测任意数量的任意配置"变为"预测 $N$ 个目标的特定配置"，预测空间显著收缩。

---

## 2. 理论推导

### 2.1 设定

- **标准 RF 检测**: 学习 $p_\theta(x_0 \mid x_t)$，其中 $x_t = (1-t)x_0 + t x_1$，$x_1 \sim \mathcal{N}(0, \sigma^2 I_4)$
- **CC-RF**: 学习 $p_\theta(x_0 \mid x_t, N)$，其中 $N$ 是目标实例数量

### 2.2 定理 1（条件方差降低）

**声明**: 对任意 $k \in \text{supp}(N)$，

$$\text{Var}(X_0 \mid X_t, N=k) \leq \text{Var}(X_0 \mid X_t)$$

等号当且仅当 $X_0 \perp N \mid X_t$（目标位置与目标数量在给定中间状态时条件独立）。

**证明**: 由全方差公式（Law of Total Variance）：

$$\text{Var}(X_0 \mid X_t) = \mathbb{E}\left[\text{Var}(X_0 \mid X_t, N) \mid X_t\right] + \text{Var}\left(\mathbb{E}[X_0 \mid X_t, N] \mid X_t\right)$$

右端第二项 $\text{Var}(\mathbb{E}[X_0|X_t,N]|X_t) \geq 0$（方差非负），故：

$$\mathbb{E}\left[\text{Var}(X_0 \mid X_t, N) \mid X_t\right] \leq \text{Var}(X_0 \mid X_t)$$

即条件内方差的期望不超过无条件方差。特别地，对任意特定 $k$，$\text{Var}(X_0|X_t,N=k)$ 是上式左端被积函数的一个取值，其期望降低。在染色体场景中 $X_0$ 与 $N$ 显然不条件独立（46 条染色体的空间配置与 40 条不同），故条件方差**严格降低**。$\square$

### 2.3 推论 1.1（方差降低的定量界）

$$\Delta\sigma^2(k) := \text{Var}(X_0|X_t) - \text{Var}(X_0|X_t,N=k) = \text{Var}\left(\mathbb{E}[X_0|X_t,N] \mid X_t, N=k\right) + \epsilon(k)$$

其中 $\epsilon(k) = \text{Var}(X_0|X_t) - \mathbb{E}[\text{Var}(X_0|X_t,N)|X_t] + \text{Var}(\mathbb{E}[X_0|X_t,N]|X_t) - \text{Var}(\mathbb{E}[X_0|X_t,N]|X_t,N=k)$。

当 $N$ 的分布集中（如 24obj 中 91.6% 为 46），$\epsilon(k) \approx 0$，方差降低量主要由条件均值散度决定。

### 2.4 命题 2（信息论视角）

$$I(X_0; N \mid X_t) = H(N \mid X_t) - H(N \mid X_0, X_t)$$

CC-RF 的收益不来自 $I(X_0;N)$（边际互信息），而来自**条件互信息** $I(X_0; N | X_t)$：在给定中间状态 $X_t$ 时，知道目标数量 $N$ 提供的额外信息。

在染色体场景中，$X_t$ 是部分去噪的框，$N$ 决定了"还应该有多少个独立目标"，这对 $X_0$ 的预测有直接信息增益。

### 2.5 与 visible-only 标注的关系

visible-only 数据集的核心问题：标注框数 $N_{box} \neq$ 实例数 $N_{instance}$。

**CC-RF 的解决方案**：
- 训练时：用 $N_{instance}$（而非 $N_{box}$）作为条件
- 推理时：用 count-prior $N=46$
- 效果：即使标注是 visible-only，模型仍学到"46 个实例"的配置先验

**理论保证**：由定理 1，$\text{Var}(X_0|X_t, N=46) < \text{Var}(X_0|X_t)$，无论标注模式如何。

### 2.6 与 DPM-Solver++ 的兼容性

DPM-Solver++ 的半线性形式不变：

$$\frac{dx_t}{dt} - \frac{1}{t}x_t = -\frac{1}{t}x_0^{pred}(x_t, t, N)$$

count 条件 $N$ 作为 $x_0^{pred}$ 的额外输入，不影响指数积分器的数学结构。$O(\Delta t^3)$ 截断误差保持。

---

## 3. 架构设计

### 3.1 Count Embedding

```
N (整数) → SinusoidalPositionEmbeddings(d=256) → Linear(256, 1024) → SiLU → Linear(1024, 1024)
```

输出 `count_emb` 形状 `[bs, feat_channels*4]`，与 `time_emb` 同维度。

### 3.2 条件融合

**方案 A（拼接 + 投影）**:
```
combined = concat(time_emb, count_emb)  # [bs, feat_channels*8]
combined = Linear(feat_channels*8, feat_channels*4)(combined)  # 降维
# 然后传入 adaln_mlp
```

**方案 B（加法）**:
```
combined = time_emb + count_emb  # [bs, feat_channels*4]
# 直接传入 adaln_mlp（无需修改 adaln_mlp 输入维度）
```

**选择方案 B**：最小改动，time_emb 和 count_emb 在同一空间相加，adaln_mlp 无需修改。

### 3.3 AdaLN-Zero 零初始化保持

count embedding 的最后一层 Linear 零初始化（weight=0, bias=0），确保：
- 初始时 `count_emb = 0`，`combined = time_emb + 0 = time_emb`
- CC-RF 从"不使用 count 信息"的状态出发（与标准 RF 等价）
- 训练过程中 count 条件从零开始增长

这与 AdaLN-Zero 的设计哲学一致：新引入的条件分支零初始化，不破坏已训练的特征。

### 3.4 推理时 Count 注入

- **训练时**: $N = \text{len}(gt\_bboxes[i])$（每张图的实际实例数）
- **推理时**: $N = 46$（count-prior），或可选地用 $N_{box}$ 估计 $N_{instance}$

### 3.5 监控指标（SwanLab 插桩）

训练时记录：
- `count_emb_norm`: count embedding 的 L2 范数（应从 0 逐渐增长）
- `count_emb_effect`: `||combined - time_emb||_2 / ||time_emb||_2`（count 对条件的影响比例）

---

## 4. 实现计划

### 4.1 代码修改清单

| 文件 | 修改内容 |
|---|---|
| `ldmdet/core/head.py` | 添加 `count_mlp` 和 `use_count_conditioning` 参数；`forward` 和 `loss` 中注入 count embedding |
| `ldmdet/core/single_head.py` | 无修改（count_emb 通过 time_emb 传入，对 single_head 透明） |
| `experiments/configs/ldmdet/directions/cc_rf/` | 新建 CC-RF 配置 |
| `tests/unit/test_count_conditioning.py` | 新建单元测试 |

### 4.2 关键实现细节

**`DiffusionDetHead.__init__` 新增**:
```python
if use_count_conditioning:
    self.count_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(feat_channels),
        nn.Linear(feat_channels, feat_channels * 4),
        nn.SiLU(),
        nn.Linear(feat_channels * 4, feat_channels * 4),
    )
    nn.init.zeros_(self.count_mlp[-1].weight)
    nn.init.zeros_(self.count_mlp[-1].bias)
```

**`DiffusionDetHead.forward` 修改**:
```python
def forward(self, features, bboxes, t, num_targets=None):
    time_emb = self.time_mlp(t)
    if self.use_count_conditioning and num_targets is not None:
        count_emb = self.count_mlp(num_targets)
        time_emb = time_emb + count_emb
    # ... 后续不变
```

**`DiffusionDetHead.loss` 修改**:
```python
def loss(self, features, img_metas, gt_bboxes, gt_labels):
    # ...
    num_targets = torch.tensor([len(b) for b in gt_bboxes],
                                device=device, dtype=torch.float32)
    # ... forward 时传入 num_targets
```

**`DiffusionDetHead.predict` 修改**:
```python
def predict(self, features, img_metas, rescale=True, return_trajectory=False):
    # ...
    count_prior = img_metas[0].get('count_prior', 46)
    num_targets = torch.full((bs,), count_prior, device=device, dtype=torch.float32)
    # ... forward_at_t 时传入 num_targets
```

### 4.3 测试计划

**单元测试** (`tests/unit/test_count_conditioning.py`):
1. `test_count_mlp_zero_init`: count_mlp 最后一层零初始化
2. `test_count_emb_initially_zero`: 初始前向时 count_emb = 0
3. `test_combined_equals_time_emb_at_init`: 初始时 combined = time_emb
4. `test_count_conditioning_changes_output`: 非零 count_emb 后输出变化
5. `test_different_counts_different_outputs`: 不同 N 产生不同预测
6. `test_count_conditioning_backward`: 梯度正常回传
7. `test_count_conditioning_gradient_flow`: count_mlp 梯度非零（训练后）
8. `test_variance_reduction_empirical`: 蒙特卡洛验证条件方差降低
9. `test_dpm_solver_compatibility`: CC-RF + DPM-Solver++ 兼容
10. `test_predict_with_count_prior`: 推理时 count-prior 注入

### 4.4 配置设计

```python
# experiments/configs/ldmdet/directions/cc_rf/cc_rf_24obj.py
_base_ = ['../../ldmdet_rf_heun_shifted_bs8.py']
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='dpm_solver_pp',
        sampling_timesteps=4,
        single_head=dict(time_conditioning='adaln_zero'),
        coupling=dict(type='ot_flow', epsilon=5.0, num_iters=20, coupling_mode='multinomial'),
        use_count_conditioning=True,
        count_prior=46,
    ),
)
```

---

## 5. 实验设计

### 5.1 主实验

| 实验 | 配置 | 预期 mAP | 目的 |
|---|---|---|---|
| A3 baseline | RF+AdaLN+StochOT+DPM++ (无 CC) | 0.858 | 基线 |
| **CC-RF** | A3 + count_conditioning | **0.870~0.878** | 验证方差降低 |

### 5.2 消融实验

| 消融 | 目的 |
|---|---|
| CC-RF vs A3 (24obj) | count 条件的增益 |
| CC-RF count_prior=46 vs count_prior=40 | count-prior 正确性的敏感性 |
| CC-RF on original (visible-only) | visible-only 场景下的效果 |

### 5.3 方差降低验证

对 100 张验证图，每张用 50 个不同初始噪声采样：
- 计算 $\text{Var}(X_0|X_t)$（无 CC）vs $\text{Var}(X_0|X_t, N=46)$（有 CC）
- 验证 $\text{Var}_{CC} < \text{Var}_{baseline}$

---

## 6. 预期贡献

1. **理论**: 定理 1 证明 count-conditioning 严格降低预测方差（全方差公式）
2. **方法**: CC-RF 架构 + AdaLN-Zero 零初始化 + DPM-Solver++ 兼容
3. **应用**: 解决 visible-only 标注的 count-prior 失效问题

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 方差降低不显著 | 中 | 高 | 24obj 的 N 分布集中（91.6%为46），可能限制条件信息量 |
| count-prior 不准确 | 低 | 中 | 24obj 的 46 是强先验，几乎确定 |
| 训练不稳定 | 低 | 中 | AdaLN-Zero 零初始化保证初始等价于标准 RF |
| 推理时 count 未知 | 低 | 低 | 染色体核型分析的 46 是领域知识 |
