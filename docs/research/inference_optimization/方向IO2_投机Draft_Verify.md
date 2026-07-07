# 方向 IO2：投机 Draft-Verify (Speculative Draft-Verify)

> **目标**: 借鉴 LLM 投机解码思想，用廉价"草稿"快速预测，仅对低置信区域用完整多步采样验证。
>
> **理论依据**:
> - Speculative Decoding: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (ICML 2023)
> - DeepCache: Li et al., "DeepCache: Accelerating Diffusion Models with Deep Feature Caching" (CVPR 2024)
> - DiffusionDet: Du et al. (CVPR 2023) — 检测框 d=4 低维空间比图像生成 d≈10⁵ 更易快速收敛
>
> **当前代码位置**: [ldmdet/core/head.py:206-255](../../../ldmdet/core/head.py#L206) — `predict()`，[ldmdet/core/head.py:143-164](../../../ldmdet/core/head.py#L143) — `forward()` 级联头

---

## 1. 背景与动机

### 1.1 LLM 投机解码回顾

LLM 投机解码的核心：
1. **Draft 模型**（小、快）生成 k 个 token
2. **Verify 模型**（大、准）一次前向验证全部 k 个 token
3. 匹配的 token 直接接受，从第一个不匹配处重新采样

**关键收益**: Verify 模型一次前向处理 k 个 token，比分步生成快 k 倍。

### 1.2 LDMDet 的类比

LDMDet 中"生成"的不是 token 序列，而是 500 个 4D 框的 x0 预测。类比映射：

| LLM 投机解码 | LDMDet 类比 |
|-------------|------------|
| Draft 模型 = 小 LM | 1步 Euler 或 2级级联头 |
| Verify 模型 = 大 LM | 4步 Heun (8 NFE) 或 6级级联头 |
| Token 匹配 | 框预测一致性（位置+分类） |
| 接受/拒绝 | 高置信框接受 / 低置信框重新采样 |

### 1.3 为什么检测比 LLM 更适合投机

- **低维空间**: 框 d=4，比 token embedding d=4096 低 3 个数量级
- **数量固定**: 500 proposal，无变长序列问题
- **收敛快**: RF 直线路径在 t<0.5 后 x0 预测趋于稳定
- **空间结构**: 框有 IoU/距离度量，比 token 匹配更精细

---

## 2. 方案设计

### 2.1 方案 A: 步数级投机 (Step-Level Speculation)

**Draft**: 1步 Euler (1 NFE) 快速预测 x0_draft
**Verify**: 仅当 draft 低置信度时，跑完整 4步 Heun (8 NFE)

```python
@torch.no.no_grad()
def predict_speculative_step(self, features, img_metas, rescale=True):
    bs = len(img_metas)
    device = features[0].device

    # === Draft: 1步 Euler ===
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
    t_draft = torch.tensor([1.0], device=device)
    cls_draft, pred_draft, x0_draft = self._forward_at_t(features, x_raw, t_draft, img_metas)

    # === 置信度评估 ===
    scores = torch.sigmoid(cls_draft).max(dim=-1)[0]  # [bs, N]
    # 每张图的"难度"评估
    img_confidence = scores.mean(dim=-1)  # [bs]
    high_conf_mask = img_confidence > self.draft_accept_threshold  # [bs]

    # === 高置信图像: 直接接受 draft ===
    results = [None] * bs
    for i in range(bs):
        if high_conf_mask[i]:
            results[i] = self._sampler.post_process(
                [(cls_draft[i:i+1], pred_draft[i:i+1])], [img_metas[i]], rescale
            )[0]

    # === 低置信图像: 完整多步采样 ===
    if not high_conf_mask.all():
        low_conf_idx = ~high_conf_mask
        x_raw_hard = x_raw[low_conf_idx]
        # 跑完整 Heun 4步
        hard_results = self._full_heun_sample(
            features, x_raw_hard, [img_metas[i] for i in range(bs) if low_conf_idx[i]],
            img_metas_offset=low_conf_idx
        )
        for idx, result in enumerate(hard_results):
            orig_idx = low_conf_idx.nonzero(as_tuple=True)[0][idx]
            results[orig_idx] = result

    return results
```

### 2.2 方案 B: 级联头级投机 (Head-Level Speculation)

**Draft**: 仅跑前 2 级级联头（共 6 级）
**Verify**: 如果 draft 预测收敛（前后2头差异小），跳过后 4 头

```python
def forward_speculative_head(self, features, bboxes, t, accept_threshold=0.01):
    """级联头投机: 前2头 draft, 后4头 verify (仅低置信时执行)。"""
    time_emb = self.time_mlp(t)
    curr_bboxes = bboxes
    curr_proposals = None
    x0_after_draft = None

    # === Draft: 前 2 级头 ===
    for i, head in enumerate(self.head_series[:2]):
        result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
        cls_logits, pred_bboxes, curr_proposals = result[:3]
        curr_bboxes = pred_bboxes.detach() if self.cascade_detach else pred_bboxes

    x0_after_draft = pred_bboxes.clone()

    # === 收敛检测 ===
    # 比较第1头和第2头的预测差异
    # (需要在循环中记录第1头输出)
    # ... 见 2.3 详细实现

    # === Verify: 后 4 级头 (仅未收敛时) ===
    if not self._head_converged(pred_bboxes, prev_pred_bboxes, accept_threshold):
        for i, head in enumerate(self.head_series[2:], start=2):
            result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
            cls_logits, pred_bboxes, curr_proposals = result[:3]
            curr_bboxes = pred_bboxes.detach() if self.cascade_detach else pred_bboxes

    return cls_logits, pred_bboxes, curr_proposals
```

### 2.3 方案 C: 混合投机 (步数 + 级联头)

**最激进的方案**: 同时在步数和级联头两个维度投机。

```
Draft:  1步 Euler × 前2级头 = 2 次单头前向
Verify: 4步 Heun × 全6级头 = 48 次单头前向

加速比 (draft accepted): 48 / 2 = 24x
```

```python
@torch.no_grad()
def predict_hybrid_speculative(self, features, img_metas, rescale=True):
    bs = len(img_metas)
    device = features[0].device

    # === Draft: 1步 × 前2头 ===
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
    cls_draft, pred_draft, x0_draft = self._forward_at_t_partial(
        features, x_raw, t=1.0, img_metas, n_heads=2
    )

    # === 逐图像置信度评估 ===
    scores = torch.sigmoid(cls_draft).max(dim=-1)[0]  # [bs, N]
    # 高置信 = 简单图像
    img_difficulty = 1.0 - scores.mean(dim=-1)  # [bs], 越低越简单

    # 三级决策
    easy_mask = img_difficulty < self.easy_threshold      # 1步×2头 即可
    medium_mask = (img_difficulty >= self.easy_threshold) & \
                  (img_difficulty < self.hard_threshold)  # 需 1步×6头
    hard_mask = img_difficulty >= self.hard_threshold      # 需 4步×6头

    results = [None] * bs

    # Easy: 直接使用 draft 结果
    if easy_mask.any():
        for i in easy_mask.nonzero(as_tuple=True)[0]:
            results[i] = self._finalize(cls_draft[i:i+1], pred_draft[i:i+1], img_metas[i], rescale)

    # Medium: 补完后4级头 (1步×6头)
    if medium_mask.any():
        idx = medium_mask.nonzero(as_tuple=True)[0]
        cls_full, pred_full = self._complete_heads(features, x_raw[idx], t=1.0, img_metas, start_head=2)
        for j, i in enumerate(idx):
            results[i] = self._finalize(cls_full[j:j+1], pred_full[j:j+1], img_metas[i], rescale)

    # Hard: 完整 4步×6头
    if hard_mask.any():
        idx = hard_mask.nonzero(as_tuple=True)[0]
        hard_results = self._full_sample(features, x_raw[idx], img_metas, idx)
        for j, i in enumerate(idx):
            results[i] = hard_results[j]

    return results
```

---

## 3. 置信度评估设计

### 3.1 图像难度度量

```python
def _estimate_difficulty(self, cls_logits, pred_bboxes):
    """基于 draft 输出评估图像难度。
    
    Returns:
        [bs] difficulty score, 越高越难
    """
    scores = torch.sigmoid(cls_logits).max(dim=-1)[0]  # [bs, N]
    
    # 维度 1: 平均置信度 (越高越简单)
    avg_conf = scores.mean(dim=-1)  # [bs]
    
    # 维度 2: 高置信框占比 (越高越简单)
    high_conf_ratio = (scores > 0.5).float().mean(dim=-1)  # [bs]
    
    # 维度 3: 框重叠度 (越高越难, 染色体交叉/重叠)
    # 简化: 用框面积方差作为代理
    # ...
    
    difficulty = (1.0 - avg_conf) * 0.5 + (1.0 - high_conf_ratio) * 0.5
    return difficulty
```

### 3.2 阈值标定

通过在验证集上统计 draft 置信度与 full mAP 的关系标定：

```python
# experiments/analysis/calibrate_speculative.py
# 1. 跑 draft (1步×2头) 得到每图置信度
# 2. 跑 full (4步×6头) 得到每图 mAP
# 3. 统计: 置信度 > θ 的图像子集的 mAP 是否与全集一致
# 4. 找到最优 θ 使 (加速比 × mAP保持率) 最大化
```

---

## 4. 实现方案

### 4.1 新增方法

在 [DiffusionDetHead](../../../ldmdet/core/head.py) 中添加：

```python
class DiffusionDetHead(nn.Module):
    def __init__(self, ..., 
                 speculative_enabled: bool = False,
                 speculative_mode: str = 'step',  # 'step' | 'head' | 'hybrid'
                 draft_accept_threshold: float = 0.7,
                 easy_threshold: float = 0.3,
                 hard_threshold: float = 0.6):
        ...
        self.speculative_enabled = speculative_enabled
        self.speculative_mode = speculative_mode
        self.draft_accept_threshold = draft_accept_threshold
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold

    def _forward_at_t_partial(self, features, x_raw, t, img_metas, n_heads=None):
        """部分级联头前向 (用于 draft)。"""
        bs, device = x_raw.shape[0], x_raw.device
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=device)
        time_emb = self.time_mlp(t_input)

        curr_proposals = None
        heads = self.head_series[:n_heads] if n_heads else self.head_series
        for head in heads:
            result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
            cls_logits, pred_bboxes, curr_proposals = result[:3]
            curr_bboxes = pred_bboxes.detach() if self.cascade_detach else pred_bboxes

        x0 = self._sampler.xyxy_to_raw(pred_bboxes, img_metas)
        return cls_logits, pred_bboxes, x0

    @torch.no_grad()
    def predict(self, features, img_metas, rescale=True, return_trajectory=False):
        if self.speculative_enabled:
            if self.speculative_mode == 'step':
                return self._predict_speculative_step(features, img_metas, rescale)
            elif self.speculative_mode == 'head':
                return self._predict_speculative_head(features, img_metas, rescale)
            elif self.speculative_mode == 'hybrid':
                return self._predict_hybrid_speculative(features, img_metas, rescale)
        # 默认: 原始 predict 逻辑
        return self._predict_original(features, img_metas, rescale, return_trajectory)
```

### 4.2 配置文件

```python
# experiments/configs/ldmdet/inference_opt/io2_speculative_step.py
_base_ = ['../ldmdet_rf_heun_adaln_stochot_eps5.py']

model = dict(
    bbox_head=dict(
        speculative_enabled=True,
        speculative_mode='step',       # 步数级投机
        draft_accept_threshold=0.7,
    ),
)

# experiments/configs/ldmdet/inference_opt/io2_speculative_hybrid.py
_base_ = ['../ldmdet_rf_heun_adaln_stochot_eps5.py']

model = dict(
    bbox_head=dict(
        speculative_enabled=True,
        speculative_mode='hybrid',     # 混合投机
        easy_threshold=0.3,
        hard_threshold=0.6,
    ),
)
```

---

## 5. 预期收益

### 5.1 各方案加速比

| 方案 | Draft 开销 | Verify 开销 | 简单图(50%) | 困难图(30%) | 加权 NFE |
|------|:---:|:---:|:---:|:---:|:---:|
| 基线 (Heun 4步) | — | 8 NFE | 8 | 8 | 8.0 |
| A: 步数级 | 1 NFE | 8 NFE | 1 | 8 | 3.5 |
| B: 级联头级 | 2 单头 | 6 单头 | 2/6=0.33 NFE-equiv | 6/6=1 NFE-equiv | ~0.65 NFE-equiv |
| C: 混合 | 2 单头 | 48 单头 | 2 | 48 | 18.4 单头 (≈3.1 NFE) |

### 5.2 精度预期

| 方案 | 预期 mAP | vs 基线 (0.753) |
|------|:---:|:---:|
| A: 步数级 | 0.750 | -0.003 |
| B: 级联头级 | 0.752 | -0.001 |
| C: 混合 | 0.748 | -0.005 |

---

## 6. 风险与缓解

### 6.1 Draft 置信度偏差

**问题**: Draft 模型（1步×2头）的置信度可能系统性偏高/偏低，导致错误接受/拒绝。

**缓解**: 
- 在验证集上标定 `draft_accept_threshold`
- 使用温度缩放校准 draft 置信度
- 保守策略: 阈值偏高（倾向 verify），宁牺牲速度不牺牲精度

### 6.2 批处理不兼容

**问题**: 同一 batch 内不同图像可能走不同路径（draft vs verify），批处理效率下降。

**缓解**:
- 按 draft 置信度排序，将同难度图像分到同一 batch
- 或: batch 内统一走 verify，但对高置信图像跳过部分步数（退化为 IO1）

### 6.3 Draft 与 Verify 分布不一致

**问题**: Draft (1步 Euler) 和 Verify (4步 Heun) 的 x0 预测分布可能不同。

**缓解**:
- 使用 Consistency Training 让 1步预测与多步一致
- 或: Draft 用同样的 Heun 但只跑 1 步（省 model_fn 的第2 NFE）

---

## 7. 验证计划

### 7.1 标定实验

```bash
# 1. 收集 draft 置信度与 full mAP 的关系
python experiments/analysis/calibrate_speculative.py \
    --config <sota_config> --checkpoint <sota_ckpt> \
    --output analysis/speculative_calibration.json

# 2. 绘制 置信度-mAP 曲线，找最优阈值
python experiments/analysis/plot_calibration.py \
    --input analysis/speculative_calibration.json
```

### 7.2 端到端对比

| 方案 | mAP 目标 | NFE 目标 | 延迟目标 |
|------|:---:|:---:|:---:|
| A: 步数级 | ≥0.750 | ≤4.0 | ≤45ms |
| B: 级联头级 | ≥0.752 | ≤6.0 NFE-equiv | ≤65ms |
| C: 混合 | ≥0.748 | ≤3.5 | ≤40ms |

### 7.3 通过标准

- mAP 下降 < 0.005
- 平均 NFE 降低 > 50%
- 简单图像 (draft accepted) 占比 > 30%
