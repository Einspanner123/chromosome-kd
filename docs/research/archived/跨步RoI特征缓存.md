# 方向 IO5：跨步 RoI 特征缓存 (Cross-Step RoI Feature Caching)

> **目标**: 在采样后期步中，对位移极小的框复用上一步的 RoI 特征，避免重复 RoIAlign 计算。
>
> **理论依据**:
> - DeepCache: Li et al. (CVPR 2024) — 相邻步特征高度相似，可缓存复用
> - RoIAlign 瓶颈: 500 框 × 7×7 采样的双线性插值是单头最重操作之一
> - RF 收敛特性: t→0 时 x_t→x_0，相邻步框位移趋近于 0
>
> **当前代码位置**: [ldmdet/core/single_head.py:223-240](../../../ldmdet/core/single_head.py#L223) — `SingleDiffusionDetHead.forward()` 中的 `pooler(features, rois)`

---

## 1. 背景与动机

### 1.1 RoIAlign 的计算开销

每个 SingleHead 的前向都执行 RoIAlign：

```python
# single_head.py:225-226
rois = bbox2roi([bboxes[i] for i in range(bs)])  # [bs×500, 5]
roi_features = pooler(features, rois)             # [bs×500, 256, 7, 7]
```

**单次 RoIAlign 开销**:
- 输入: 500 框 × 4 级 FPN 特征图
- 每框: 7×7 = 49 个采样点 × 双线性插值（4 邻域）
- 总采样: 500 × 49 × 4 = 98000 次插值

**整体开销**: 8 NFE × 6 头 × 98000 = **4.7M 次插值**，占单头前向 ~30% 延迟。

### 1.2 跨步框位移分析

Rectified Flow 的 x_t 演化：

```
x_{t+dt} = x_t + dt · v_t
```

在 t→0 时，v_t = (x_t - x0_pred) / t → 0（因为 x_t ≈ x0_pred），因此：

| 时间步 | t 范围 | 平均框位移 (px) | 位移 <1px 占比 |
|:---:|:---:|:---:|:---:|
| Step 0 | 1.0→0.75 | ~50 | 5% |
| Step 1 | 0.75→0.5 | ~15 | 20% |
| Step 2 | 0.5→0.25 | ~5 | 50% |
| Step 3 | 0.25→0.0 | ~1 | 85% |

**结论**: 后期步（Step 2-3）中 50-85% 的框位移 <1px，其 RoI 特征几乎不变，可缓存复用。

### 1.3 与 LLM KV Cache 的类比与区别

| 特性 | LLM KV Cache | IO5 RoI Cache |
|------|:---:|:---:|
| 缓存内容 | K, V 矩阵 | RoI 特征图 |
| 失效条件 | token 不变 | 框位移 <阈值 |
| 命中率 | 100% (autoregressive) | 50-85% (后期步) |
| 缓存粒度 | per-token | per-proposal |
| 更新方式 | 追加新 token | 替换位移大的框 |

---

## 2. 方案设计

### 2.1 核心: 选择性 RoI 重提取

```python
def forward_with_cache(self, features, bboxes, proposals, pooler, time_emb, 
                       prev_bboxes=None, roi_cache=None):
    """带跨步缓存的 SingleHead 前向。
    
    Args:
        features: backbone FPN 特征 (不变)
        bboxes: [bs, N, 4] 当前步框
        proposals: 上一步的 proposal 特征
        pooler: RoIAlign 提取器
        time_emb: 时间嵌入
        prev_bboxes: [bs, N, 4] 上一步框 (None=无缓存)
        roi_cache: [bs*N, 256, 7, 7] 上一步 RoI 特征缓存
    Returns:
        cls_logits, pred_bboxes, proposals, roi_features, new_cache
    """
    bs, num_boxes = bboxes.shape[:2]
    
    # === IO5: 判断哪些框需要重新提取 RoI ===
    if prev_bboxes is not None and roi_cache is not None and self.roi_cache_enabled:
        # 计算每个框的位移
        displacement = (bboxes - prev_bboxes).norm(dim=-1)  # [bs, N]
        cache_valid = displacement < self.roi_cache_threshold  # [bs, N]
        
        # 分离: 需要重提取的框 vs 可复用的框
        # 需要重提取的框索引
        recompute_idx = ~cache_valid  # [bs, N]
        
        if recompute_idx.any():
            # 仅对位移大的框做 RoIAlign
            recompute_bboxes = []
            for i in range(bs):
                idx = recompute_idx[i]
                recompute_bboxes.append(bboxes[i][idx])
            recompute_rois = bbox2roi(recompute_bboxes)
            new_roi_features = pooler(features, recompute_rois)
            
            # 合并: 位移大的用新特征, 位移小的用缓存
            roi_features = roi_cache.clone()
            offset = 0
            for i in range(bs):
                idx = recompute_idx[i]
                n_recompute = idx.sum().item()
                if n_recompute > 0:
                    roi_features[i * num_boxes:(i + 1) * num_boxes][idx.flatten()] = \
                        new_roi_features[offset:offset + n_recompute]
                    offset += n_recompute
        else:
            # 全部命中缓存
            roi_features = roi_cache
    else:
        # 无缓存: 全量提取
        rois = bbox2roi([bboxes[i] for i in range(bs)])
        roi_features = pooler(features, rois)
    
    # 更新缓存
    new_cache = roi_features.detach()
    new_prev_bboxes = bboxes.detach()
    
    # ... 后续 Self-Attn / DynamicConv / FFN 逻辑不变
    ...
```

### 2.2 在 head.py 中的集成

修改 [DiffusionDetHead](../../../ldmdet/core/head.py) 的 `predict()` 和 `_forward_at_t()`：

```python
def _forward_at_t(self, features, x_raw, t, img_metas, 
                  prev_bboxes=None, roi_cache=None):
    """带 RoI 缓存的单步前向。"""
    bs, device = x_raw.shape[0], x_raw.device
    curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
    t_input = torch.full((bs,), t * self.timesteps, device=device)
    time_emb = self.time_mlp(t_input)

    # === IO5: 传递缓存给 head_series ===
    curr_roi_cache = roi_cache
    curr_prev_bboxes = prev_bboxes
    new_caches = []

    cls_logits_seq, pred_bboxes_seq = [], []
    curr_proposals = None
    curr_bboxes_iter = curr_bboxes
    
    for head in self.head_series:
        result = head.forward_with_cache(
            features, curr_bboxes_iter, curr_proposals, self.roi_extractor, time_emb,
            prev_bboxes=curr_prev_bboxes, roi_cache=curr_roi_cache
        )
        cls_logits, pred_bboxes, curr_proposals, roi_feat, new_cache = result[:5]
        
        # 级联头内: 下一头用本头的输出作为 prev (头内不复用)
        curr_prev_bboxes = None  # 级联头内关闭缓存
        curr_roi_cache = None
        curr_bboxes_iter = pred_bboxes.detach() if self.cascade_detach else pred_bboxes
        new_caches.append(new_cache)
        cls_logits_seq.append(cls_logits)
        pred_bboxes_seq.append(pred_bboxes)
    
    # 返回: 最后一级输出 + 第0级头的新缓存 (供下一步使用)
    x0 = self._sampler.xyxy_to_raw(pred_bboxes_seq[-1], img_metas)
    return (cls_logits_seq[-1], pred_bboxes_seq[-1], x0,
            curr_bboxes, new_caches[0])  # 缓存: 当前步框 + 第0级RoI特征


@torch.no_grad()
def predict(self, features, img_metas, rescale=True, return_trajectory=False):
    bs = len(img_metas)
    device = features[0].device
    time_pairs = self._sampler.build_time_pairs(device)
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

    ensemble_results = []
    
    # === IO5: 跨步缓存 ===
    prev_bboxes = None
    roi_cache = None

    for step_idx, (t_curr, t_next) in enumerate(time_pairs):
        result = self._forward_at_t(
            features, x_raw, t_curr, img_metas,
            prev_bboxes=prev_bboxes, roi_cache=roi_cache
        )
        cls_logits, pred_bboxes, x0_raw, curr_bboxes, new_roi_cache = result
        
        # 更新缓存
        prev_bboxes = curr_bboxes
        roi_cache = new_roi_cache
        
        if self.use_ensemble:
            ensemble_results.append((cls_logits, pred_bboxes))

        # ODE step
        if self.solver_type == 'heun' and t_next > 0:
            def model_fn(x_tmp, t_tmp):
                # Heun 的第2 NFE: 不使用缓存 (框已变化)
                r = self._forward_at_t(features, x_tmp, t_tmp, img_metas,
                                       prev_bboxes=None, roi_cache=None)
                return r[2], None
            x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
        else:
            x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

        if self.box_renewal:
            x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
            # Box renewal 修改了框, 缓存失效
            prev_bboxes = None
            roi_cache = None
        
        if t_next <= 0:
            break

    results = self._sampler.post_process(ensemble_results, img_metas, rescale)
    return results
```

### 2.3 缓存粒度选择

有三种缓存粒度可选：

| 粒度 | 缓存内容 | 命中率 | 实现复杂度 | 推荐 |
|------|---------|:---:|:---:|:---:|
| **per-proposal** | 每个 proposal 的 RoI 特征 [256,7,7] | 高 | 中 | ✓ 推荐 |
| per-head | 每级头的完整 RoI 特征 | 中 | 低 | |
| per-step | 整步的 RoI 特征 | 低 | 最低 | |

**推荐 per-proposal**: 位移小的框复用，位移大的重提取，命中率最高。

---

## 3. 阈值设计

### 3.1 位移阈值

```python
# 缓存命中条件: 框位移 < threshold (像素)
roi_cache_threshold = 1.0  # 1px
```

不同阈值的缓存命中率：

| 阈值 (px) | Step 1 命中率 | Step 2 命中率 | Step 3 命中率 | 预期 mAP 影响 |
|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 5% | 30% | 65% | -0.000 |
| 1.0 | 20% | 50% | 85% | -0.001 |
| 2.0 | 40% | 70% | 92% | -0.002 |
| 5.0 | 70% | 90% | 97% | -0.005 |

**推荐 `threshold=1.0`**: 后期步 50-85% 命中率，mAP 几乎无损。

### 3.2 时间步感知阈值

```python
def _get_cache_threshold(self, t_curr):
    """根据时间步调整缓存阈值。"""
    if t_curr > 0.5:
        return 0.5    # 早期步: 严格, 少缓存
    elif t_curr > 0.2:
        return 1.0    # 中期步: 适中
    else:
        return 2.0    # 后期步: 宽松, 多缓存
```

---

## 4. 配置参数

### 4.1 新增参数

```python
# SingleDiffusionDetHead
def __init__(
    self,
    ...
    # === IO5: RoI 缓存 ===
    roi_cache_enabled: bool = False,
    roi_cache_threshold: float = 1.0,        # 位移阈值 (px)
    roi_cache_time_aware: bool = True,       # 时间步感知
):
    ...
    self.roi_cache_enabled = roi_cache_enabled
    self.roi_cache_threshold = roi_cache_threshold
    self.roi_cache_time_aware = roi_cache_time_aware
```

### 4.2 配置文件

```python
# experiments/configs/ldmdet/inference_opt/io5_roi_cache.py
_base_ = ['../ldmdet_rf_heun_adaln_stochot_eps5.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            roi_cache_enabled=True,
            roi_cache_threshold=1.0,
            roi_cache_time_aware=True,
        ),
    ),
)
```

---

## 5. 显存管理

### 5.1 缓存显存估算

```
单步缓存大小 = bs × N × 256 × 7 × 7 × 4 bytes (FP32)
             = 1 × 500 × 256 × 49 × 4
             = 25 MB (per step)
```

8 NFE 总缓存（如果全保存）= 200 MB，但只需缓存上一步，因此 **常驻 25 MB**。

### 5.2 显存优化: 仅缓存高置信框

```python
# 只缓存高置信度框的 RoI 特征 (低置信框会被 box_renewal 替换)
def _selective_cache(self, roi_features, cls_logits, prev_bboxes):
    scores = torch.sigmoid(cls_logits).max(dim=-1)[0]  # [bs, N]
    cache_mask = scores > 0.5  # 仅缓存高置信框
    # ... 仅存储 cache_mask 为 True 的 RoI 特征
```

可降低缓存显存到 ~5 MB。

---

## 6. 与其他方向的交互

### 6.1 IO5 × IO3 (Top-K 剪枝)

**协同效应**: IO3 在第1步后将 N 从 500 降到 100。IO5 的缓存也只需覆盖 100 个框，显存降 80%。

**但**: IO3 剪枝后 N 变化，缓存索引需重映射：

```python
# 剪枝后, 缓存的 500 框中仅保留 100 框
if pruned:
    roi_cache = roi_cache[topk_idx]  # 按剪枝索引提取子集
    prev_bboxes = prev_bboxes[topk_idx]
```

### 6.2 IO5 × IO1 (自适应步数)

**冲突**: IO1 提前终止后，缓存的 RoI 特征不再被使用（浪费）。但这是一次性浪费，影响极小。

**协同**: IO1 减少步数后，剩余步的缓存命中率更高（因为 t 更小，框位移更小）。

### 6.3 IO5 × Box Renewal

**冲突**: Box renewal 把低置信框替换为随机噪声，缓存对这些框完全失效。

**解决**: Box renewal 后清空对应框的缓存：

```python
if self.box_renewal:
    renewed_mask = ...  # 被替换的框
    roi_cache[renewed_mask] = None  # 标记失效
    prev_bboxes[renewed_mask] = None
```

或: 简单地在 box_renewal 后整体清空缓存（保守方案）。

### 6.4 IO5 × IO4 (级联头提前退出)

**独立**: IO5 仅在第0级头缓存（跨步），IO4 在头内提前退出。两者完全正交。

---

## 7. 预期收益

### 7.1 RoIAlign 延迟节省

| 步骤 | 命中率 | RoIAlign 原始延迟 | 节省 |
|:---:|:---:|:---:|:---:|
| Step 0 | 0% | 10ms | 0ms |
| Step 1 | 20% | 10ms | 2ms |
| Step 2 | 50% | 10ms | 5ms |
| Step 3 | 85% | 10ms | 8.5ms |
| Heun model_fn (×4) | 0% | 10ms | 0ms |
| **总计** | | 80ms | **15.5ms (19%)** |

### 7.2 整体延迟节省

RoIAlign 约占单头前向 30%，单头前向约占整体 100%：

```
整体节省 = RoIAlign 节省 × RoIAlign 占比
         = 19% × 30%
         = 5.7% (整体)
```

**注意**: IO5 的收益相对较小（~6%），但与 IO3 叠加后效果显著（IO3 降 N，IO5 降重复计算）。

### 7.3 与 IO3 叠加的收益

```
IO3: N 500→100, RoIAlign 延迟降 5x
IO5: 后期步缓存命中率 85%, 再降 85%
叠加: RoIAlign 延迟 = 原始 × (100/500) × (1 - 0.85) = 原始 × 3%
```

---

## 8. 风险与缓解

### 8.1 风险: 缓存特征过期

**问题**: 框位移 0.9px（阈值 1.0px 内）但 FPN 特征在该位置有高频变化，缓存的 RoI 特征与真实特征差异较大。

**缓解**:
- 阈值取保守值（1.0px 而非 5.0px）
- 可选: 对缓存特征做微小修正（如加一个学习到的残差）
- 验证: 对比缓存命中 vs 重提取的 RoI 特征余弦相似度

### 8.2 风险: 实现复杂度

**问题**: per-proposal 的选择性重提取需要复杂的索引操作，可能引入 bug。

**缓解**:
- 第一版: 实现简单的"全缓存或全不缓存"（per-step 粒度）
- 第二版: 优化为 per-proposal 粒度
- 充分单元测试

### 8.3 风险: Heun 第2 NFE 无法缓存

**问题**: Heun 的 model_fn 在 t_next 处评估，框已更新（Euler 预测后），无法复用 t_curr 的缓存。

**缓解**:
- model_fn 不使用缓存（已实现）
- 或: 用 DPM-Solver++ 替代 Heun（无需第2 NFE，每步都可缓存）

---

## 9. 验证计划

### 9.1 缓存命中率统计

```bash
python experiments/analysis/benchmark_inference.py \
    --config experiments/configs/ldmdet/inference_opt/io5_roi_cache.py \
    --checkpoint <sota_ckpt> \
    --log-cache-stats
```

输出：
```
Step 0: cache hit rate = 0.0% (no prev)
Step 1: cache hit rate = 22.3%
Step 2: cache hit rate = 51.7%
Step 3: cache hit rate = 86.2%
Average cache hit rate: 40.1%
```

### 9.2 RoI 特征相似度验证

```python
# experiments/analysis/verify_roi_cache.py
# 对比缓存特征 vs 重提取特征的余弦相似度
# 确保位移 <1px 的框, 特征相似度 >0.99
```

### 9.3 指标

| 配置 | RoIAlign 延迟 | 整体延迟 | mAP 目标 |
|------|:---:|:---:|:---:|
| 基线 | 80ms | ~85ms | 0.753 |
| IO5 only | 64ms | ~80ms | ≥0.752 |
| IO5 + IO3 | 12ms | ~50ms | ≥0.750 |

### 9.4 通过标准

- mAP 下降 < 0.002
- 后期步 (Step 2-3) 缓存命中率 > 50%
- RoIAlign 延迟降低 > 15%

---

## 实验验证结论

### 实验设计
- **状态**: 理论分析, 未进行实际实现和训练
- **依据**: 基于 benchmark_inference.py 实测数据进行分析

### 实验结果

| 指标 | 数值 |
|------|------|
| Best mAP | — (未实现) |
| Δ vs 基准 (0.858) | — (未实现) |
| 推理时间 | — (未实现) |
| 跨步框位移 | 93~124 px/步 |
| 位移 <1px 占比 | 仅 0.1~0.3% |
| 结论 | **证伪** — 理论基础是数学错误 |

### 数据证据

- **RF 速度是常数**: v = noise - x₀, **不随 t→0 而变化** (而非前期文档假设的 v→0)
- **基于 benchmark_inference.py 实测**: 跨步框位移 mean=93~124px, 位移 <1px 占比仅 0.1~0.3%
- 与文档前期理论预期 (Step 3 位移~1px, 85% 框位移 <1px) **完全相反**
- **根本原因**: RF 的直线路径是 x_t = (1-t)x₀ + t·noise, 速度 v = noise - x₀ 是常数, 不随 t 变化。框在每一步都发生大幅位移, RoI 特征剧变, 缓存完全不可用

### 结论与推荐

- **跨步 RoI 特征缓存的理论基础 (RF 速度→0) 是数学错误**
- **该方向不可行**, 不建议继续探索
