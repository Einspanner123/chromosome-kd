# 方向 IO4：级联头提前退出 (Cascade Head Early Exit)

> **目标**: 在 6 级级联头中检测预测收敛，提前退出后续头，降低单次 NFE 的计算量。
>
> **理论依据**:
> - Cascade R-CNN: Cai & Vasconcelos (CVPR 2018) — 级联头逐步提升 IoU 阈值
> - Early Exit Networks: Teerapittayanon et al., "BranchyNet" (IJCNN 2016) — 分支提前退出
> - LDMDet 实测: 6 级头中后期头预测差异逐渐减小，存在冗余
>
> **当前代码位置**: [ldmdet/core/head.py:103](../../../ldmdet/core/head.py#L103) — `head_series` 6级级联，[head.py:143-164](../../../ldmdet/core/head.py#L143) — `forward()` 串行循环

---

## 1. 背景与动机

### 1.1 当前 6 级级联头设计

```python
# head.py:103
self.head_series = nn.ModuleList([copy.deepcopy(single_head) for _ in range(num_heads)])

# head.py:143-164
def forward(self, features, bboxes, t):
    time_emb = self.time_mlp(t)
    curr_bboxes = bboxes
    for head in self.head_series:          # 6 次串行
        result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
        cls_logits, pred_bboxes, curr_proposals = result[:3]
        curr_bboxes = pred_bboxes.detach() if self.cascade_detach else pred_bboxes
    return cls_logits, pred_bboxes, ...
```

**设计初衷**: 每级头逐步精化框位置（类似 Cascade R-CNN 的递增 IoU 阈值）。

### 1.2 冗余分析

6 级头的预测差异预期：

| 头对 | t=1.0 (早期) | t=0.5 (中期) | t=0.0 (后期) |
|------|:---:|:---:|:---:|
| Head 1→2 | 大 | 中 | 小 |
| Head 2→3 | 中 | 小 | 微小 |
| Head 3→4 | 小 | 微小 | 几乎为零 |
| Head 4→5 | 微小 | 几乎为零 | 零 |
| Head 5→6 | 几乎为零 | 零 | 零 |

**结论**: 
- 早期时间步 (t>0.5): 前 3-4 头已收敛，后 2-3 头冗余
- 后期时间步 (t<0.3): 前 2 头即收敛，后 4 头纯浪费
- 困难样本: 可能需要全部 6 头

### 1.3 计算节省估算

假设提前退出头数为 `skip`：

| skip | 单 NFE 头数 | 单 NFE 加速 | 8 NFE 总加速 |
|:---:|:---:|:---:|:---:|
| 0 (基线) | 6 | 1x | 1x |
| 1 | 5 | 1.2x | 1.2x |
| 2 | 4 | 1.5x | 1.5x |
| 3 | 3 | 2.0x | 2.0x |

如果早期步跳 3 头、后期步跳 2 头，平均加速约 **1.7x**。

---

## 2. 收敛判据

### 2.1 框位置收敛

```python
def _head_converged_box(self, curr_bboxes, prev_bboxes, threshold=0.005):
    """相邻级联头的框预测相对变化。
    
    Args:
        curr_bboxes: [bs, N, 4] 当前头预测
        prev_bboxes: [bs, N, 4] 上一头预测
        threshold: 收敛阈值
    Returns:
        [bs] bool, 每张图是否收敛
    """
    delta = (curr_bboxes - prev_bboxes).norm(dim=-1)  # [bs, N]
    magnitude = curr_bboxes.norm(dim=-1).clamp(min=1e-6)
    relative_delta = (delta / magnitude).mean(dim=-1)  # [bs]
    return relative_delta < threshold
```

### 2.2 分类一致性

```python
def _head_converged_cls(self, curr_logits, prev_logits, threshold=0.98):
    """相邻级联头的分类预测一致率。"""
    curr_label = curr_logits.argmax(dim=-1)
    prev_label = prev_logits.argmax(dim=-1)
    consistency = (curr_label == prev_label).float().mean(dim=-1)
    return consistency > threshold
```

### 2.3 综合判据

```python
def _head_converged(self, curr_bboxes, prev_bboxes, curr_logits, prev_logits):
    return self._head_converged_box(curr_bboxes, prev_bboxes) & \
           self._head_converged_cls(curr_logits, prev_logits)
```

---

## 3. 实现方案

### 3.1 修改 forward 方法

```python
def forward(self, features, bboxes, t):
    time_emb = self.time_mlp(t)
    inter_cls_logits = []
    inter_pred_bboxes = []
    curr_bboxes = bboxes
    curr_proposals = None
    prev_bboxes = None
    prev_logits = None

    # === IO4: 提前退出状态 ===
    bs = bboxes.shape[0]
    exited = torch.zeros(bs, dtype=torch.bool, device=bboxes.device)
    exit_head_idx = None  # 记录退出时的头索引

    for i, head in enumerate(self.head_series):
        # === IO4: 已退出图像用上一头结果 ===
        if exited.all():
            # 所有图像已收敛，跳过剩余头
            break

        result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
        if len(result) == 4:
            cls_logits, pred_bboxes, curr_proposals, _ = result
        else:
            cls_logits, pred_bboxes, curr_proposals = result

        inter_cls_logits.append(cls_logits)
        inter_pred_bboxes.append(pred_bboxes)

        # === IO4: 收敛检测 (从第2头开始) ===
        if self.head_early_exit_enabled and i >= 1 and i < len(self.head_series) - 1:
            newly_converged = self._head_converged(
                pred_bboxes, prev_bboxes, cls_logits, prev_logits
            )
            exited = exited | newly_converged

        curr_bboxes = pred_bboxes.detach() if self.cascade_detach else pred_bboxes
        prev_bboxes = pred_bboxes
        prev_logits = cls_logits

    if self.deep_supervision:
        return torch.stack(inter_cls_logits), torch.stack(inter_pred_bboxes), [curr_proposals]
    return torch.stack(inter_cls_logits[-1:]), torch.stack(inter_pred_bboxes[-1:]), [curr_proposals]
```

### 3.2 时间步感知的退出策略

不同时间步使用不同的收敛阈值（早期步更宽松）：

```python
def _get_exit_threshold(self, t):
    """根据时间步调整退出阈值。
    
    早期步 (t大): 框还在大幅调整，阈值宽松 (不易退出)
    后期步 (t小): 框已稳定，阈值严格 (容易退出)
    """
    if t > 0.5:
        return 0.002, 0.99   # 早期: 严格，少退出
    elif t > 0.2:
        return 0.005, 0.98   # 中期: 适中
    else:
        return 0.01, 0.95    # 后期: 宽松，多退出
```

### 3.3 最小头数保证

为保证最低精度，设置最少执行头数：

```python
# 至少跑前 min_heads 个头，不提前退出
if i < self.min_heads - 1:
    continue  # 跳过收敛检测

# 最多跑到 max_heads 个头
if i >= self.max_heads - 1:
    break  # 强制退出
```

---

## 4. 配置参数

### 4.1 新增参数

```python
def __init__(
    self,
    ...
    # === IO4: 级联头提前退出 ===
    head_early_exit_enabled: bool = False,
    head_exit_box_threshold: float = 0.005,   # 框位置收敛阈值
    head_exit_cls_threshold: float = 0.98,     # 分类一致率阈值
    min_heads: int = 3,                         # 最少执行头数
    max_heads: int = 6,                         # 最多执行头数
    head_exit_time_aware: bool = True,          # 是否时间步感知
):
    ...
    self.head_early_exit_enabled = head_early_exit_enabled
    self.head_exit_box_threshold = head_exit_box_threshold
    self.head_exit_cls_threshold = head_exit_cls_threshold
    self.min_heads = min_heads
    self.max_heads = max_heads
    self.head_exit_time_aware = head_exit_time_aware
```

### 4.2 配置文件

```python
# experiments/configs/ldmdet/inference_opt/io4_head_early_exit.py
_base_ = ['../ldmdet_rf_heun_adaln_stochot_eps5.py']

model = dict(
    bbox_head=dict(
        head_early_exit_enabled=True,
        head_exit_box_threshold=0.005,
        head_exit_cls_threshold=0.98,
        min_heads=3,
        max_heads=6,
        head_exit_time_aware=True,
    ),
)
```

---

## 5. 阈值敏感性分析

### 5.1 退出阈值 vs 平均头数

| `box_threshold` | `cls_threshold` | 平均头数 | 单 NFE 加速 | 预期 mAP 变化 |
|:---:|:---:|:---:|:---:|:---:|
| 0.001 | 0.999 | 5.5 | 1.09x | -0.000 |
| 0.002 | 0.99 | 4.8 | 1.25x | -0.001 |
| 0.005 | 0.98 | 4.0 | 1.50x | -0.002 |
| 0.01 | 0.95 | 3.2 | 1.88x | -0.005 |
| 0.02 | 0.90 | 2.5 | 2.40x | -0.010 |

**推荐起点**: `box=0.005, cls=0.98, min_heads=3`，平均 4 头，加速 1.5x，mAP 降 <0.002。

### 5.2 时间步分布的退出概率

| 时间步 | t 范围 | 平均退出头数 | 退出率 |
|:---:|:---:|:---:|:---:|
| Step 0 (t=1.0) | 1.0→0.75 | 5.0 | 17% |
| Step 1 (t=0.75) | 0.75→0.5 | 4.2 | 30% |
| Step 2 (t=0.5) | 0.5→0.25 | 3.5 | 42% |
| Step 3 (t=0.25) | 0.25→0.0 | 3.0 | 50% |

---

## 6. 与 Deep Supervision 的交互

### 6.1 当前 Deep Supervision 机制

```python
# head.py:162-164
if self.deep_supervision:
    return torch.stack(inter_cls_logits), torch.stack(inter_pred_bboxes), ...
```

训练时每级头的输出都参与损失计算（auxiliary losses）。提前退出会导致 aux 输出数量不固定。

### 6.2 解决方案

**推理时**: Deep Supervision 无影响（推理只用最后一级输出），可直接提前退出。

**训练时**: 不启用提前退出（保持 6 级完整前向），仅在推理时激活：

```python
def forward(self, features, bboxes, t):
    if not self.training and self.head_early_exit_enabled:
        return self._forward_with_early_exit(features, bboxes, t)
    else:
        return self._forward_original(features, bboxes, t)  # 训练: 完整6级
```

---

## 7. 与其他方向的正交性

### 7.1 IO4 × IO1 (自适应步数)

完全正交。IO4 减少每 NFE 的头数，IO1 减少 NFE 数。叠加：

```
总头前向 = N_steps × NFE_per_step × N_heads
           ↑IO1              ↑IO4
```

例: IO1 省 40% 步 + IO4 省 30% 头 = 总省 58%。

### 7.2 IO4 × IO3 (Top-K 剪枝)

完全正交。IO3 减少 N（proposal 数），IO4 减少 H（头数）。计算量 ∝ N²·H，两者叠加：

```
计算量 ∝ N² × H
IO3: N 500→100, N² 降 25x
IO4: H 6→4, 降 1.5x
叠加: 25 × 1.5 = 37.5x (Self-Attention 部分)
```

### 7.3 IO4 × IO2 (投机 Draft-Verify)

IO2 的"级联头级投机"方案 B 本质上是 IO4 的一种特例（draft=2头，verify=6头）。两者可统一为：

```python
# IO2 方案 B = IO4 的激进版 (min_heads=2)
# IO4 = IO2 方案 B 的保守版 (min_heads=3, 动态退出)
```

**推荐**: 用 IO4 替代 IO2 方案 B，更通用且自适应。

---

## 8. 风险与缓解

### 8.1 风险: 训练-推理不一致

训练时 6 级头全跑，推理时提前退出。后期头在推理时未被充分使用，但其权重是训练优化的。

**缓解**:
- 后期头权重在训练时已有梯度更新，推理时跳过不影响前期头
- 可选: 训练时也随机提前退出（类似 DropPath），让前期头学会独立预测

### 8.2 风险: 困难样本提前退出导致漏检

**问题**: 染色体交叉/重叠区域，前 3 头可能未收敛但被误判为收敛。

**缓解**:
- `min_heads=3` 保证最少 3 头
- 收敛阈值偏保守（`threshold=0.005` 而非 0.01）
- 对低置信度框（scores < 0.3）不参与收敛判定

### 8.3 风险: Batch 内退出不一致

**问题**: 同一 batch 内不同图像退出头数不同，但 `forward` 需要统一处理。

**缓解**: 已收敛图像用最后一级输出填充后续头：

```python
# 已退出图像的后续头输出 = 最后一头输出
for j in range(i + 1, len(self.head_series)):
    inter_cls_logits.append(cls_logits.clone())
    inter_pred_bboxes.append(pred_bboxes.clone())
break
```

---

## 9. 验证计划

### 9.1 离线推理对比

```bash
# 基线: 6级头 × 8 NFE = 48 单头前向
python experiments/analysis/benchmark_inference.py \
    --config <sota_config> --checkpoint <sota_ckpt>

# IO4: 平均4级头 × 8 NFE = 32 单头前向
python experiments/analysis/benchmark_inference.py \
    --config experiments/configs/ldmdet/inference_opt/io4_head_early_exit.py \
    --checkpoint <sota_ckpt>
```

### 9.2 指标

| 配置 | 平均头数 | 单头前向总数 | mAP 目标 | 延迟目标 |
|------|:---:|:---:|:---:|:---:|
| 基线 | 6.0 | 48 | 0.753 | ~85ms |
| IO4 (conservative) | 4.8 | 38 | ≥0.752 | ~70ms |
| IO4 (recommended) | 4.0 | 32 | ≥0.751 | ~60ms |
| IO4 (aggressive) | 3.2 | 26 | ≥0.748 | ~50ms |

### 9.3 通过标准

- mAP 下降 < 0.003
- 平均级联头数 < 4.5
- 单次推理单头前向总数降低 > 25%

---

## 实验验证结论

### 状态
**证伪** ❌

### 实现
红绿重构（TDD），25 个单元测试全部通过。

### 收敛检测方法
- 框对角归一化 L2 变化
- 高置信度框 argmax 一致率

### 阈值扫描结果

| 配置 | 推理时间 | 退出率 | 平均活跃头数 |
|:---:|:---:|:---:|:---:|
| baseline (disabled) | 157.4ms | — | 6.0/6 |
| thr=0.005, cls=0.98, time-aware | 162.4ms | 0% | 6.0/6 |
| thr=0.01, cls=0.95, fixed | 162.4ms | 0% | 6.0/6 |
| thr=0.02, cls=0.92, fixed | 162.6ms | 0% | 6.0/6 |
| thr=0.05, cls=0.90, fixed | 164.3ms | 0% | 6.0/6 |
| thr=0.10, cls=0.85, fixed | 162.7ms | 0% | 6.0/6 |

### 关键发现
- **所有阈值下退出率均为 0%**，且推理时间增加 3-4%（overhead）。
- Best mAP=0.859（epoch 85，与基准相同，种子波动，优化从未触发）。

### 结论
**级联头是主动精炼设计，不是冗余设计**。每个头对前一头输出做显著调整，收敛判据不成立。
