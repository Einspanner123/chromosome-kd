# 方向 IO1：自适应步数提前终止 (Adaptive Step Early Termination)

> **目标**: 在采样循环中检测 x0 预测收敛，提前终止剩余步数，降低平均 NFE。
>
> **理论依据**:
> - DeepCache (CVPR 2024): 扩散采样中相邻步特征高度相似
> - DDIM (ICLR 2021): 少步采样在简单样本上已接近收敛
> - Rectified Flow 直线路径: 越接近 t=0，x_t 变化越缓慢
>
> **当前代码位置**: [ldmdet/core/head.py:218-249](../../../ldmdet/core/head.py#L218) — `predict()` 采样循环

---

## 1. 背景与动机

### 1.1 当前固定步数的问题

当前 Heun 4步采样对所有图像一视同仁：

```python
# head.py:209 — 固定 4 个时间对
time_pairs = self._sampler.build_time_pairs(device)  # [(1.0,0.75),(0.75,0.5),(0.5,0.25),(0.25,0.0)]

for step_idx, (t_curr, t_next) in enumerate(time_pairs):
    cls_logits, pred_bboxes, x0_raw = self._forward_at_t(features, x_raw, t_curr, img_metas)
    # Heun: 额外一次 model_fn 调用
    x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
```

**问题**: 染色体检测中，大多数图像是标准核型（46条框，分离度高）。这些"简单"图像在 1-2 步后 x0 预测已收敛，剩余步数是冗余计算。

### 1.2 收敛的理论依据

Rectified Flow 的直线路径 `x_t = (1-t)x_0 + t·noise` 在 t→0 时：

```
x_t ≈ x_0 + t·(noise - x_0)
‖x_t - x_0‖ = t·‖noise - x_0‖ → 0
```

因此 t 越小，x_t 越接近 x_0，模型预测的 x0_pred 越稳定。在 t<0.3 后，相邻步的 x0_pred 变化通常 <1%。

### 1.3 预期收益估算

对 24obj 验证集的图像难度分布（假设）：

| 难度 | 占比 | 当前 NFE | 优化后 NFE | 节省 |
|------|:---:|:---:|:---:|:---:|
| 简单（1步收敛） | 20% | 8 | 2 | 75% |
| 中等（2步收敛） | 50% | 8 | 4 | 50% |
| 困难（4步全跑） | 30% | 8 | 8 | 0% |
| **加权平均** | | | | **42.5%** |

---

## 2. 收敛判据设计

### 2.1 框位置收敛 (Box Position Convergence)

```python
def _is_converged_box(self, x0_curr, x0_prev, threshold=0.01):
    """框位置相对变化率。
    
    Args:
        x0_curr: [bs, N, 4] 当前步 x0 预测
        x0_prev: [bs, N, 4] 上一步 x0 预测
        threshold: 收敛阈值 (相对 L2 距离)
    Returns:
        [bs] bool, 每张图是否收敛
    """
    delta = (x0_curr - x0_prev).norm(dim=-1)  # [bs, N]
    magnitude = x0_curr.norm(dim=-1).clamp(min=1e-6)  # [bs, N]
    relative_delta = (delta / magnitude).mean(dim=-1)  # [bs]
    return relative_delta < threshold
```

### 2.2 分类置信度收敛 (Classification Convergence)

```python
def _is_converged_cls(self, cls_curr, cls_prev, threshold=0.95):
    """分类预测一致性。
    
    两次预测的 top-1 类别一致率超过 threshold 则收敛。
    """
    curr_label = cls_curr.argmax(dim=-1)  # [bs, N]
    prev_label = cls_prev.argmax(dim=-1)  # [bs, N]
    consistency = (curr_label == prev_label).float().mean(dim=-1)  # [bs]
    return consistency > threshold
```

### 2.3 综合判据

```python
def _is_converged(self, x0_curr, x0_prev, cls_curr, cls_prev):
    """位置和分类同时收敛才判定为收敛。"""
    return self._is_converged_box(x0_curr, x0_prev) & \
           self._is_converged_cls(cls_curr, cls_prev)
```

---

## 3. 实现方案

### 3.1 核心代码修改

在 [head.py](../../../ldmdet/core/head.py) 的 `predict()` 方法中添加收敛检测：

```python
@torch.no_grad()
def predict(self, features, img_metas, rescale=True, return_trajectory=False):
    device = features[0].device
    bs = len(img_metas)
    time_pairs = self._sampler.build_time_pairs(device)
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

    ensemble_results = []
    trajectory = []
    dpm_solver = self._sampler.create_dpm_solver()
    if dpm_solver is not None:
        dpm_solver.reset()

    # === IO1 新增: 收敛检测状态 ===
    x0_prev = None
    cls_prev = None
    converged = torch.zeros(bs, dtype=torch.bool, device=device)
    early_exit_threshold = getattr(self, 'early_exit_threshold', 0.01)

    for step_idx, (t_curr, t_next) in enumerate(time_pairs):
        # 已收敛的图像跳过本步（但保留 ensemble 结果）
        if converged.all():
            break

        cls_logits, pred_bboxes, x0_raw = self._forward_at_t(features, x_raw, t_curr, img_metas)

        # === IO1: 收敛检测 ===
        if x0_prev is not None:
            newly_converged = self._is_converged(
                x0_raw, x0_prev, cls_logits, cls_prev, early_exit_threshold
            )
            # 新收敛的图像：记录最终结果但不再更新 x_raw
            converged = converged | newly_converged

        if return_trajectory:
            trajectory.append((cls_logits.detach(), pred_bboxes.detach()))
        if self.use_ensemble:
            ensemble_results.append((cls_logits, pred_bboxes))
        if not ensemble_results:
            ensemble_results.append((cls_logits, pred_bboxes))

        # 已收敛图像不继续演化
        if converged.any() and not converged.all():
            # 对未收敛图像继续 Heun step，收敛图像冻结 x_raw
            x_raw_new = x_raw.clone()
            # ... 正常 step 计算 x_raw_next ...
            x_raw_new[~converged] = x_raw_next[~converged]
            x_raw = x_raw_new
        else:
            # 正常 step (见下文)
            ...

        x0_prev = x0_raw
        cls_prev = cls_logits

    results = self._sampler.post_process(ensemble_results, img_metas, rescale)
    if return_trajectory:
        return results, trajectory
    return results
```

### 3.2 配置参数

在 `DiffusionDetHead.__init__` 中添加：

```python
def __init__(
    self,
    ...
    # === IO1: 自适应步数 ===
    early_exit_enabled: bool = False,
    early_exit_threshold: float = 0.01,    # 框位置相对变化阈值
    early_exit_cls_threshold: float = 0.95, # 分类一致率阈值
    early_exit_min_steps: int = 1,          # 最少步数（不跳第0步）
):
    ...
    self.early_exit_enabled = early_exit_enabled
    self.early_exit_threshold = early_exit_threshold
    self.early_exit_cls_threshold = early_exit_cls_threshold
    self.early_exit_min_steps = early_exit_min_steps
```

### 3.3 配置文件示例

```python
# experiments/configs/ldmdet/inference_opt/io1_adaptive_steps.py
_base_ = ['../ldmdet_rf_heun_adaln_stochot_eps5.py']

model = dict(
    bbox_head=dict(
        early_exit_enabled=True,
        early_exit_threshold=0.01,
        early_exit_cls_threshold=0.95,
        early_exit_min_steps=1,
    ),
)
```

---

## 4. 阈值敏感性分析

### 4.1 阈值 vs mAP 权衡

| `threshold` | `cls_threshold` | 预期平均 NFE | 预期 mAP 变化 |
|:---:|:---:|:---:|:---:|
| 0.005 | 0.99 | 6.5 | -0.000 |
| 0.01 | 0.95 | 5.2 | -0.001 |
| 0.02 | 0.90 | 4.0 | -0.003 |
| 0.05 | 0.85 | 3.0 | -0.008 |
| 0.10 | 0.80 | 2.5 | -0.020 |

**推荐起点**: `threshold=0.01, cls_threshold=0.95`，预期 NFE 5.2（省 35%），mAP 下降 <0.001。

### 4.2 per-step 收敛概率

基于 RF 直线路径特性，预期收敛分布：

| 步数 | t 范围 | 累计收敛比例 |
|:---:|:---:|:---:|
| 1 | 1.0→0.75 | ~15% |
| 2 | 0.75→0.5 | ~45% |
| 3 | 0.5→0.25 | ~80% |
| 4 | 0.25→0.0 | 100% |

---

## 5. 风险与缓解

### 5.1 风险: Heun 二阶步的 model_fn 额外开销

Heun 每步需要 2 NFE（主步 + model_fn 验证步）。如果在第1步就收敛，仍需支付 Heun 的第2 NFE。

**缓解**: 对收敛检测改为在 **Euler 预览步** 后判断，而非 Heun 主步后：

```python
# 先用 Euler 1 NFE 预览
x0_preview = self._forward_at_t(features, x_raw, t_curr, img_metas)
if x0_prev is not None and self._is_converged(x0_preview, x0_prev, ...):
    break  # 省掉 Heun 的第2 NFE
# 未收敛才做完整 Heun
x_raw = self.rf.heun_step(x_raw, x0_preview, t_curr, t_next, model_fn)
```

### 5.2 风险: Ensemble 结果缺失

提前终止后，后续步的 ensemble 结果缺失，可能影响 NMS 后的多步投票。

**缓解**: `use_ensemble=False` 时无此问题。`use_ensemble=True` 时，对已收敛图像重复使用最后一步结果填充 ensemble：

```python
while len(ensemble_results) < len(time_pairs):
    ensemble_results.append(ensemble_results[-1])  # 重复最后结果
```

---

## 6. 验证计划

### 6.1 离线推理对比

```bash
# 基线 (Heun 4步, 8 NFE)
python experiments/analysis/benchmark_inference.py \
    --config experiments/configs/ldmdet/ldmdet_rf_heun_adaln_stochot_eps5.py \
    --checkpoint work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_*.pth

# IO1 优化
python experiments/analysis/benchmark_inference.py \
    --config experiments/configs/ldmdet/inference_opt/io1_adaptive_steps.py \
    --checkpoint work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_*.pth \
    --log-interval 50
```

### 6.2 指标

| 指标 | 基线预期 | IO1 预期 |
|------|:---:|:---:|
| mAP | 0.753 | ≥0.751 |
| 平均 NFE | 8.0 | ~5.2 |
| 平均延迟 | ~85ms | ~55ms |
| P50 延迟 | ~80ms | ~45ms |
| P95 延迟 | ~90ms | ~85ms |

### 6.3 通过标准

- mAP 下降 < 0.002
- 平均 NFE 降低 > 30%
- 无图像因提前终止导致 mAP 下降 > 0.01

---

## 实验验证结论

### 实验设计
- **训练配置**: 24obj 数据集, SOTA 基准 (RF+Heun+AdaLN+StochOT eps5), 150 epoch, EarlyStopping patience=30
- **实现方式**: 在 head.py 中添加收敛检测逻辑, 红绿重构(TDD)实现, 单元测试全部通过
- **训练恢复**: 从 epoch 69 继续训练至 EarlyStop (epoch 83)

### 实验结果

| 指标 | 数值 |
|------|------|
| Best mAP | 0.860 (epoch 83) |
| Δ vs 基准 (0.858) | +0.002 (种子波动, 优化从未触发) |
| 推理时间 | 0.195s/样本 (baseline 0.188s), 慢 3.7% |
| 退出率 | 所有阈值下均为 0% |
| 结论 | **证伪** — 理论假设完全错误 |

### 数据证据

**收敛分布分析** (50 张 val 图像, SOTA checkpoint):

| Step | t_curr | x0_Δrel mean | x0_Δrel p50 | <0.01占比 | <0.05占比 | cls一致率 |
|:----:|:------:|:------------:|:-----------:|:---------:|:---------:|:---------:|
| 1 | 0.900 | 0.418 | 0.415 | 0.0% | 0.0% | 0.337 |
| 2 | 0.750 | 0.408 | 0.413 | 0.0% | 0.0% | 0.368 |
| 3 | 0.500 | 0.424 | 0.429 | 0.0% | 0.0% | 0.371 |

- **x0 步间相对变化 mean=0.408~0.424**, 不随 t 减小而收敛
- **阈值扫描结果**: 所有阈值 (thr=0.01/0.02/0.03/0.05/0.10, cls=0.95/0.92/0.90/0.85/0.80) 退出率均为 0%
- **最宽松阈值 (0.10/0.80)**: 推理时间 157.0ms (vs baseline 149.1ms), 反而慢 5.3%
- 训练时的 SwanLab 因 SSL 错误未上传成功

### 结论与推荐

- **理论假设完全错误**: RF 的 x0_pred 在步间变化高达 40%, 不随 t→0 收敛。相邻步 Heun 演化带来显著预测变化, 收敛判据不成立
- **该方向不可行**, 不建议继续探索
