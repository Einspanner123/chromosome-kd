# 方向 IO3：Top-K 框剪枝 (Top-K Box Pruning)

> **目标**: 第1步采样后丢弃低置信度 proposal，将后续步的 N 从 500 降至 100~200，大幅降低 Self-Attention 和 DynamicConv 计算量。
>
> **理论依据**:
> - Sparse R-CNN: Sun et al. (CVPR 2021) — 固定 100 个 proposal 即可达到 SOTA
> - DiffusionDet: Du et al. (CVPR 2023) — 500 proposal 用于多样性，推理后期大量冗余
> - 染色体核型: 每图固定 ~46 个 GT，500 proposal 中 90%+ 是噪声
>
> **当前代码位置**: [ldmdet/diffusion/sampling.py:96-117](../../../ldmdet/diffusion/sampling.py#L96) — `apply_box_renewal`，[ldmdet/core/head.py:210](../../../ldmdet/core/head.py#L210) — `num_proposals=500`

---

## 1. 背景与动机

### 1.1 当前 500 Proposal 的问题

当前推理始终处理 500 个 proposal：

```python
# head.py:210
x_raw = torch.randn(bs, self.num_proposals, 4, device=device)  # bs×500×4
```

但 24obj 数据集每图仅 ~46 个 GT。500 proposal 中：
- **第1步 (t=1.0)**: 全噪声，需 500 个保持多样性
- **第2步 (t=0.75)**: 已有 ~100 个高置信框，其余 400 个是噪声
- **第3-4步 (t<0.5)**: 仅 ~50 个有效框，450 个纯浪费

### 1.2 Box Renewal 的不足

当前 [box_renewal](../../../ldmdet/diffusion/sampling.py#L96) 把低置信框**替换为随机噪声**而非丢弃：

```python
# sampling.py:113-116
num_renew = (~keep).sum()
if num_renew > 0:
    x_raw_new[i, ~keep] = torch.randn(num_renew, 4, device=device)  # ← 生成新噪声
```

这些新生噪声框在后续步仍需完整走 6 级级联头 × Self-Attention × DynamicConv，是纯浪费。

### 1.3 计算量与 N 的关系

| 组件 | 计算复杂度 | N=500 | N=100 | 加速 |
|------|:---:|:---:|:---:|:---:|
| Self-Attention | O(N²·d) | 250K·d | 10K·d | **25x** |
| DynamicConv | O(N·P²·d) | 500·49·d | 100·49·d | 5x |
| RoIExtract | O(N·P²) | 500·49 | 100·49 | 5x |
| FFN | O(N·d²) | 500·d² | 100·d² | 5x |
| cls/reg head | O(N·d) | 500·d | 100·d | 5x |

**Self-Attention 是最大瓶颈**（N² 复杂度），从 500→100 可加速 25 倍。

---

## 2. 方案设计

### 2.1 核心: Top-K 选择 + 动态 N

```python
def apply_topk_pruning(self, x_raw, cls_logits, pred_bboxes, k=100):
    """保留 Top-K 高置信度框，丢弃其余。
    
    Args:
        x_raw: [bs, N, 4] 扩散空间框
        cls_logits: [bs, N, num_classes] 分类 logits
        pred_bboxes: [bs, N, 4] 图像空间框
        k: 保留的框数
    Returns:
        pruned x_raw, cls_logits, pred_bboxes (N→k)
    """
    bs = x_raw.shape[0]
    scores = torch.sigmoid(cls_logits).max(dim=-1)[0]  # [bs, N]
    
    # 每张图选 Top-K
    topk_idx = scores.topk(k, dim=1).indices  # [bs, k]
    
    # Gather
    idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
    x_raw_pruned = x_raw.gather(1, idx_expanded)
    pred_bboxes_pruned = pred_bboxes.gather(1, idx_expanded)
    
    idx_cls = topk_idx.unsqueeze(-1).expand(-1, -1, cls_logits.shape[-1])
    cls_logits_pruned = cls_logits.gather(1, idx_cls)
    
    return x_raw_pruned, cls_logits_pruned, pred_bboxes_pruned
```

### 2.2 集成到采样循环

```python
@torch.no_grad()
def predict(self, features, img_metas, rescale=True, return_trajectory=False):
    bs = len(img_metas)
    device = features[0].device
    time_pairs = self._sampler.build_time_pairs(device)
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)  # 初始 500

    ensemble_results = []
    current_n = self.num_proposals  # 动态 N

    for step_idx, (t_curr, t_next) in enumerate(time_pairs):
        cls_logits, pred_bboxes, x0_raw = self._forward_at_t(features, x_raw, t_curr, img_metas)
        
        if self.use_ensemble:
            ensemble_results.append((cls_logits, pred_bboxes))

        # === IO3: 第1步后 Top-K 剪枝 ===
        if (self.topk_pruning_enabled and 
            step_idx == 0 and 
            current_n > self.topk_k):
            x_raw, cls_logits, pred_bboxes = self._sampler.apply_topk_pruning(
                x_raw, cls_logits, pred_bboxes, k=self.topk_k
            )
            current_n = self.topk_k
            # 更新 ensemble 中最后一个结果为剪枝后的
            if self.use_ensemble:
                ensemble_results[-1] = (cls_logits, pred_bboxes)

        # ODE step (Heun/Euler/DPM)
        if self.solver_type == 'heun' and t_next > 0:
            def model_fn(x_tmp, t_tmp):
                _, _, x0_tmp = self._forward_at_t(features, x_tmp, t_tmp, img_metas)
                return x0_tmp, None
            x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
        else:
            x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

        if self.box_renewal:
            x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
        if t_next <= 0:
            break

    results = self._sampler.post_process(ensemble_results, img_metas, rescale)
    return results
```

### 2.3 渐进式剪枝 (Progressive Pruning)

更激进的方案: 每步逐步减少 N：

```python
# 每步的 N 计划
pruning_schedule = {
    0: 500,   # 第0步: 全量
    1: 200,   # 第1步: 保留200
    2: 100,   # 第2步: 保留100
    3: 100,   # 第3步: 保持100
}

for step_idx, (t_curr, t_next) in enumerate(time_pairs):
    cls_logits, pred_bboxes, x0_raw = self._forward_at_t(features, x_raw, t_curr, img_metas)
    
    # 渐进剪枝
    target_n = pruning_schedule.get(step_idx, current_n)
    if target_n < current_n:
        x_raw, cls_logits, pred_bboxes = self._sampler.apply_topk_pruning(
            x_raw, cls_logits, pred_bboxes, k=target_n
        )
        current_n = target_n
    
    # ... ODE step
```

---

## 3. 配置参数

### 3.1 DiffusionDetHead 新增参数

```python
def __init__(
    self,
    ...
    # === IO3: Top-K 剪枝 ===
    topk_pruning_enabled: bool = False,
    topk_k: int = 100,                # 第1步后保留的框数
    topk_pruning_step: int = 0,       # 在第几步后剪枝 (0=第0步后)
    topk_progressive: bool = False,   # 是否渐进式剪枝
    topk_schedule: list = None,       # 渐进式 N 计划
):
    ...
    self.topk_pruning_enabled = topk_pruning_enabled
    self.topk_k = topk_k
    self.topk_pruning_step = topk_pruning_step
    self.topk_progressive = topk_progressive
    self.topk_schedule = topk_schedule or [500, 200, 100, 100]
```

### 3.2 配置文件

```python
# experiments/configs/ldmdet/inference_opt/io3_topk_pruning.py
_base_ = ['../ldmdet_rf_heun_adaln_stochot_eps5.py']

model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_k=100,
        topk_pruning_step=0,       # 第0步后剪枝
    ),
)

# 渐进式
# experiments/configs/ldmdet/inference_opt/io3_topk_progressive.py
model = dict(
    bbox_head=dict(
        topk_pruning_enabled=True,
        topk_progressive=True,
        topk_schedule=[500, 200, 100, 100],
    ),
)
```

---

## 4. K 值选择分析

### 4.1 K vs mAP 权衡

| K | 后续步 Self-Attn 加速 | 预期 mAP | vs 基线 |
|:---:|:---:|:---:|:---:|
| 500 (无剪枝) | 1x | 0.753 | — |
| 300 | 2.8x | 0.753 | -0.000 |
| 200 | 6.25x | 0.752 | -0.001 |
| 100 | 25x | 0.750 | -0.003 |
| 50 | 100x | 0.745 | -0.008 |
| 24 (≈GT数) | 434x | 0.735 | -0.018 |

**推荐 K=100**: Self-Attention 加速 25x，mAP 仅降 0.003。

### 4.2 为什么 K=100 而非 K=46

虽然每图 ~46 个 GT，但需冗余框用于：
1. **NMS 前的过提案**: 一个 GT 可能对应多个高置信框
2. **Ensemble 多样性**: 多步 ensemble 需要不同步的框有差异
3. **边界 GT**: 部分 GT 在第1步置信度不高，需更多候选

---

## 5. 与 Box Renewal 的关系

### 5.1 当前 Box Renewal 行为

```python
# sampling.py:96-117
def apply_box_renewal(self, x_raw, cls_logits):
    scores = torch.sigmoid(cls_logits).max(-1)[0]
    keep = scores > self.score_thr  # score_thr=0.5
    # 低置信框 → 替换为随机噪声
    x_raw_new[i, ~keep] = torch.randn(num_renew, 4)
    return x_raw_new
```

### 5.2 IO3 与 Box Renewal 的对比

| 特性 | Box Renewal | IO3 Top-K |
|------|:---:|:---:|
| 低置信框处理 | 替换为噪声 | **丢弃** |
| N 是否减少 | 否（始终500） | **是** |
| 后续步计算量 | 不变 | **大幅降低** |
| 多样性来源 | 随机噪声框 | 第1步的 500 框 |

### 5.3 推荐组合

- **第0步**: 500 框 + Box Renewal（保持多样性）
- **第0步后**: Top-K 剪枝到 100
- **第1步起**: 关闭 Box Renewal（100 框已足够，无需再注入噪声）

```python
if step_idx == 0:
    if self.box_renewal:
        x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
    if self.topk_pruning_enabled:
        x_raw, cls_logits, pred_bboxes = self._sampler.apply_topk_pruning(...)
        self._box_renewal_active = False  # 后续步关闭
```

---

## 6. 对其他组件的影响

### 6.1 Ensemble 后处理

[sampling.py:194-264](../../../ldmdet/diffusion/sampling.py#L194) 的 `post_process` 拼接所有步的 ensemble 结果：

```python
for cls_logits, pred_bboxes in ensemble_results:
    all_scores.append(conf)
    all_bboxes.append(pred_bboxes[i])
```

**问题**: 第0步 500 框 + 后续步 100 框，ensemble 框数不一致。

**解决**: 
- 方案 A: 第0步也剪枝后再加入 ensemble（推荐）
- 方案 B: ensemble 仅用剪枝后的步（跳过第0步）
- 方案 C: 第0步的 500 框先做 NMS 再加入 ensemble

### 6.2 Cascade 级联头

级联头内 N 保持一致（同一步内不剪枝）。剪枝只在步间发生。

### 6.3 DPM-Solver++

DPM-Solver++ 的历史缓存 `x0_history` 需要维度一致。剪枝后维度变化需重置缓存：

```python
if dpm_solver is not None and pruned:
    dpm_solver.reset()  # 维度变了，清空历史
```

---

## 7. 风险与缓解

### 7.1 风险: 第1步误丢弃有效框

**问题**: 第1步 (t=1.0) 分类置信度不可靠，可能丢弃真实 GT 对应的框。

**缓解**: 
- K 取偏大值（100 而非 46）
- 第1步后剪枝而非第0步后（让 Heun 二阶修正后再剪）
- 混合策略: 保留 Top-50 高置信框 + 保留 Top-50 高方差框（多样性）

### 7.2 风险: Ensemble 多样性下降

**问题**: 剪枝后后续步的框高度重叠，ensemble 投票效果减弱。

**缓解**:
- 剪枝前对 Top-2K 框做 Soft-NMS，保留空间分散的 K 个框
- 或: 剪枝后仍保持 K=100（而非降到 GT 数），保证冗余

---

## 8. 验证计划

### 8.1 K 值扫描

```bash
for K in 300 200 100 50; do
    python experiments/analysis/benchmark_inference.py \
        --config experiments/configs/ldmdet/inference_opt/io3_topk_pruning.py \
        --checkpoint <sota_ckpt> \
        --override model.bbox_head.topk_k=$K
done
```

### 8.2 指标

| K | mAP 目标 | 后续步延迟 | 整体延迟 |
|:---:|:---:|:---:|:---:|
| 300 | ≥0.752 | -20% | -10% |
| 200 | ≥0.752 | -40% | -25% |
| 100 | ≥0.750 | -70% | -45% |
| 50 | ≥0.745 | -85% | -60% |

### 8.3 通过标准

- mAP 下降 < 0.003
- 后续步 (第1-3步) 计算量降低 > 60%
- 整体推理延迟降低 > 30%
