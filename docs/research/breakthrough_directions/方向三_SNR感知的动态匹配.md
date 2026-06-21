# 方向三：SNR 感知的动态匹配 (SNR-Aware Dynamic Matching)

> **目标**：将扩散时间 $t$（及其对应的信噪比 SNR）注入 SimOTA 匹配代价，使高噪声时匹配更保守、低噪声时更确定，减少错误匹配对梯度的污染。
>
> **理论依据**：
> - 信噪比与匹配不确定性：Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models" (2024)
> - 贝叶斯匹配：Probabilistic assignment in DETR-style detectors
>
> **当前代码位置**：[ldmdet/criterion/matcher.py](../../ldmdet/criterion/matcher.py), [ldmdet/criterion/costs.py](../../ldmdet/criterion/costs.py)

---

## 1. 问题分析：匹配代价与 $t$ 无关

### 1.1 当前实现

[matcher.py:38-44](../../ldmdet/criterion/matcher.py#L38-L44) 的匹配代价使用**固定权重**：

```python
self.costs = [
    FocalLossCost(weight=cost_class),    # 固定 2.0
    BBoxL1Cost(weight=cost_bbox),        # 固定 5.0
    IoUCost(iou_mode='giou', weight=cost_giou),  # 固定 2.0
]
```

[matcher.py:220](../../ldmdet/criterion/matcher.py#L220) 聚合为：

```python
cost_matrix = torch.stack(cost_list).sum(0)  # [bs, N, max_gt]
```

### 1.2 数学问题

**扩散时间 $t$ 与预测可信度的关系**：

Rectified Flow 的前向过程 $x_t = (1-t)x_0 + t x_1$，信噪比：

$$
\text{SNR}(t) = \frac{\text{Var}(x_0 \text{ 信号})}{\text{Var}(x_1 \text{ 噪声})} = \frac{(1-t)^2}{t^2}
$$

- $t \to 1$（高噪声）：$\text{SNR} \to 0$，预测 $x_0$ 几乎纯噪声，匹配代价不可靠
- $t \to 0$（低噪声）：$\text{SNR} \to \infty$，预测接近真实 $x_0$，匹配代价可信

**当前匹配的三个问题**：

1. **高噪声错误匹配**：$t \approx 1$ 时，预测框随机，但匹配器仍按固定代价分配 GT，产生大量错误正样本
2. **梯度污染**：错误匹配的 proposal 被强制学习错误的 GT，污染分类和回归梯度
3. **动态 $k$ 失真**：[matcher.py:289](../../ldmdet/criterion/matcher.py#L289) 的 `dynamic_ks` 基于 IoU，高噪声时 IoU 普遍低，导致 $k$ 偏小，GT 欠采样

---

## 2. 数学推导

### 2.1 SNR 加权代价

#### 2.1.1 基本形式

将匹配代价乘以 SNR 的单调函数：

$$
\mathcal{C}_{\text{match}}(t) = w(t) \cdot \mathcal{C}_{\text{cls}} + w(t) \cdot \mathcal{C}_{\text{bbox}} + w(t) \cdot \mathcal{C}_{\text{giou}}
$$

其中 $w(t)$ 应满足：
- $w(0) = 1$（低噪声，全权信任）
- $w(1) = 0$（高噪声，不信任）
- 单调递减

**候选 1：Logistic SNR 权重**

$$
w(t) = \frac{\text{SNR}(t)}{1 + \text{SNR}(t)} = \frac{(1-t)^2}{(1-t)^2 + t^2}
$$

性质：
- $t=0$: $w=1$
- $t=0.5$: $w=0.5$（SNR=1）
- $t=1$: $w=0$
- 对称且光滑

**候选 2：指数衰减**

$$
w(t) = \exp(-\beta \cdot t), \quad \beta > 0
$$

更激进，$\beta$ 控制衰减速度。

**候选 3：硬阈值**

$$
w(t) = \mathbb{1}[t < t_{\text{thresh}}]
$$

简单但不可微，不推荐。

#### 2.1.2 对动态 $k$ 的影响

[matcher.py:289](../../ldmdet/criterion/matcher.py#L289)：

```python
dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1, max=candidate_topk)
```

高噪声时 IoU 普遍低，`topk_ious.sum(0)` 小，$k$ 小。**这是合理的**——高噪声时应少分配正样本。但当前实现没有显式利用 $t$，而是被动受 IoU 影响。

**改进**：显式用 $w(t)$ 调整 $k$ 上界：

$$
k_{\max}(t) = \lceil w(t) \cdot k_{\max}^{\text{base}} \rceil
$$

### 2.2 贝叶斯概率匹配

#### 2.2.1 匹配的后验概率

更严格地，匹配应基于后验概率 $p(\text{assign} \mid x_t, t)$，而非点估计代价。

将匹配建模为 softmax：

$$
p(i \to j \mid x_t, t) = \frac{\exp(-\mathcal{C}_{ij} / \tau(t))}{\sum_{j'} \exp(-\mathcal{C}_{ij'} / \tau(t))}
$$

其中 $\tau(t)$ 为温度：
- $\tau \to 0$：硬分配（argmax）
- $\tau \to \infty$：均匀分配

#### 2.2.2 温度与 SNR 的关系

高噪声时匹配不确定，应提高温度（更模糊）；低噪声时应降低温度（更确定）：

$$
\tau(t) \propto \frac{1}{\text{SNR}(t)} = \frac{t^2}{(1-t)^2}
$$

或更平缓：

$$
\tau(t) = \tau_0 \cdot (1 + \gamma / \text{SNR}(t))
$$

#### 2.2.3 与 SimOTA 的关系

SimOTA 的动态 $k$ 本质上是**确定性的 top-$k$ 分配**。贝叶斯视角下，可改为**概率采样**：

$$
\text{assign}(i) \sim \text{Categorical}(p(i \to \cdot \mid x_t, t))
$$

但这会引入随机性，可能损害训练稳定性。**折中**：保留 SimOTA 的确定性分配，但用 $\tau(t)$ 调整代价的"锐度"。

### 2.3 损失加权的一致性

匹配代价加权后，**损失也应同步加权**，否则匹配和损失不一致：

$$
\mathcal{L}_{\text{total}} = \sum_t w(t) \cdot \mathcal{L}(t)
$$

当前 [criterion.py](../../ldmdet/criterion/criterion.py) 的深度监督损失未按 $t$ 加权。改进后应：

```python
loss = w(t) * (loss_cls + loss_bbox + loss_giou)
```

---

## 3. 实现方案

### 3.1 SNR 权重计算

新建 [ldmdet/criterion/snr_weight.py](../../ldmdet/criterion/snr_weight.py)：

```python
"""SNR 感知权重计算"""

import torch
from torch import Tensor


def logistic_snr_weight(t: Tensor) -> Tensor:
    """Logistic SNR 权重: w(t) = SNR/(1+SNR) = (1-t)^2 / ((1-t)^2 + t^2)

    性质: w(0)=1, w(0.5)=0.5, w(1)=0, 单调递减

    Args:
        t: [bs] 或标量, 扩散时间 ∈ [0, 1]

    Returns:
        w: 同形状, ∈ [0, 1]
    """
    return (1 - t) ** 2 / ((1 - t) ** 2 + t ** 2 + 1e-8)


def exponential_snr_weight(t: Tensor, beta: float = 3.0) -> Tensor:
    """指数衰减权重: w(t) = exp(-β·t)

    Args:
        t: [bs] 或标量
        beta: 衰减速率 (越大越激进)

    Returns:
        w: 同形状, ∈ (0, 1]
    """
    return torch.exp(-beta * t)


def get_snr_weight(t: Tensor, mode: str = 'logistic', **kwargs) -> Tensor:
    """统一接口"""
    if mode == 'logistic':
        return logistic_snr_weight(t)
    elif mode == 'exponential':
        return exponential_snr_weight(t, kwargs.get('beta', 3.0))
    elif mode == 'none':
        return torch.ones_like(t)
    else:
        raise ValueError(f"Unknown SNR weight mode: {mode}")
```

### 3.2 SNR 感知匹配器

修改 [matcher.py](../../ldmdet/criterion/matcher.py)，新增 `SNRAwareMatcher`：

```python
"""SNR 感知动态匹配器"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from ldmdet.criterion.costs import BBoxL1Cost, FocalLossCost, IoUCost
from ldmdet.criterion.matcher import DiffusionDetMatcher
from ldmdet.criterion.snr_weight import get_snr_weight
from ldmdet.data.structures import InstanceData, ModelOutput


class SNRAwareMatcher(DiffusionDetMatcher):
    """SNR 感知 SimOTA 匹配器.

    在标准 SimOTA 基础上, 按扩散时间 t (经 SNR 变换) 加权匹配代价:
    - 高噪声 (t→1): 代价权重低, 匹配更保守, dynamic_k 上界小
    - 低噪声 (t→0): 代价权重高, 匹配更确定

    Args:
        snr_mode: 'logistic' | 'exponential' | 'none'
        snr_beta: exponential 模式的衰减率
        k_max_scale: dynamic_k 上界的 t 感知缩放 (False=不缩放)
        cost_threshold: 代价超过此值时不分配 (高噪声保护)
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        center_radius: float = 2.5,
        candidate_topk: int = 5,
        match_costs: List = None,
        snr_mode: str = 'logistic',
        snr_beta: float = 3.0,
        k_max_scale: bool = True,
        cost_threshold: float = 50.0,
    ):
        super().__init__(
            cost_class=cost_class,
            cost_bbox=cost_bbox,
            cost_giou=cost_giou,
            center_radius=center_radius,
            candidate_topk=candidate_topk,
            match_costs=match_costs,
        )
        self.snr_mode = snr_mode
        self.snr_beta = snr_beta
        self.k_max_scale = k_max_scale
        self.cost_threshold = cost_threshold

    @torch.no_grad()
    def forward(
        self,
        outputs: ModelOutput,
        targets: List[InstanceData],
        t: Optional[Tensor] = None,
    ) -> List[Tuple[Tensor, Tensor]]:
        """SNR 感知匹配.

        Args:
            outputs: 模型输出
            targets: GT 列表
            t: [bs] 扩散时间, 若为 None 则退化为标准匹配
        """
        if t is None:
            return super().forward(outputs, targets)

        # 计算 SNR 权重
        snr_w = get_snr_weight(t, mode=self.snr_mode, beta=self.snr_beta)  # [bs]

        pred_logits = outputs.pred_logits
        pred_bboxes = outputs.pred_boxes
        bs = pred_logits.size(0)
        N = pred_logits.size(1)

        max_gt = max(t.bboxes.size(0) for t in targets)
        if max_gt == 0:
            return [(torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                     torch.zeros(N, dtype=torch.long, device=pred_bboxes.device))] * bs

        # 构建 GT padding (复用父类逻辑)
        gt_bboxes_padded = pred_bboxes.new_zeros(bs, max_gt, 4)
        gt_labels_padded = pred_logits.new_full((bs, max_gt), 0, dtype=torch.long)
        gt_num = []
        for i, tgt in enumerate(targets):
            n = tgt.bboxes.size(0)
            gt_num.append(n)
            if n > 0:
                gt_bboxes_padded[i, :n] = tgt.bboxes
                gt_labels_padded[i, :n] = tgt.labels

        # 批量代价
        cost_matrix, pairwise_ious = self._batched_cost(
            pred_logits, pred_bboxes, gt_labels_padded, gt_bboxes_padded, gt_num, max_gt
        )

        # SNR 加权代价: [bs] → [bs, 1, 1]
        snr_w_view = snr_w.view(bs, 1, 1)
        cost_matrix = cost_matrix * snr_w_view

        # 高噪声保护: 代价超过阈值的 proposal 不参与匹配
        # (注意: 这是软保护, 通过抬高代价实现)
        high_noise_mask = snr_w < 0.1  # w < 0.1 视为高噪声
        if high_noise_mask.any():
            cost_matrix[high_noise_mask] += self.cost_threshold

        # 逐图动态 k 匹配
        results = []
        for i in range(bs):
            n = gt_num[i]
            if n == 0:
                results.append((
                    torch.zeros(N, dtype=torch.bool, device=pred_bboxes.device),
                    torch.zeros(N, dtype=torch.long, device=pred_bboxes.device),
                ))
            else:
                k_max = self.candidate_topk
                if self.k_max_scale:
                    k_max = max(int(snr_w[i].item() * self.candidate_topk), 1)
                results.append(
                    self._dynamic_k_matching_t(
                        cost_matrix[i, :, :n],
                        pairwise_ious[i, :, :n],
                        n,
                        k_max,
                    )
                )
        return results

    @torch.no_grad()
    def _dynamic_k_matching_t(
        self, cost, pairwise_ious, num_gt, k_max
    ) -> Tuple[Tensor, Tensor]:
        """t 感知的 dynamic k 匹配, k 上界由 SNR 调整"""
        matching_matrix = torch.zeros_like(cost)
        pairwise_ious = torch.nan_to_num(pairwise_ious, nan=0.0, posinf=1.0, neginf=0.0)
        candidate_topk = min(k_max, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1, max=candidate_topk)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[pos_idx, gt_idx] = 1.0

        # 多 GT 竞争同一 proposal: 选代价最低的
        anchor_matching_gt = matching_matrix.sum(1) > 1
        if anchor_matching_gt.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[anchor_matching_gt], dim=1)
            matching_matrix[anchor_matching_gt] *= 0
            matching_matrix[anchor_matching_gt, cost_argmin] = 1.0

        assigned_gt = matching_matrix.argmax(dim=1)
        assigned_mask = matching_matrix.sum(dim=1) > 0
        return assigned_mask, assigned_gt
```

### 3.3 损失加权同步

修改 [criterion.py](../../ldmdet/criterion/criterion.py)，在深度监督循环中按 $t$ 加权：

```python
# 深度监督: 每层 head 的损失按 t 加权
for i, (cls_logits, pred_bboxes) in enumerate(zip(inter_cls_logits, inter_pred_bboxes)):
    # ... 计算 loss_cls, loss_bbox, loss_giou ...

    # SNR 加权
    if self.snr_weighted_loss:
        from ldmdet.criterion.snr_weight import get_snr_weight
        snr_w = get_snr_weight(t, mode=self.snr_mode)  # [bs]
        # 广播到 [bs, 1] 用于加权每图损失
        weight = snr_w.mean()  # 标量, 因为同 batch 同 t
        loss_cls = loss_cls * weight
        loss_bbox = loss_bbox * weight
        loss_giou = loss_giou * weight

    # ... 累加到总损失 ...
```

### 3.4 配置示例

```python
# experiments/configs/ldmdet_snr_matching.py
model = dict(
    bbox_head=dict(
        criterion=dict(
            type='DiffusionDetCriterion',
            matcher=dict(
                type='SNRAwareMatcher',
                snr_mode='logistic',
                k_max_scale=True,
                cost_threshold=50.0,
            ),
            snr_weighted_loss=True,
        ),
    ),
)
```

---

## 4. 实验设计

### 4.1 主实验

| 实验 ID | 匹配器 | $w(t)$ | 损失加权 | 假设 |
|---------|--------|--------|---------|------|
| E3.0 | 标准 SimOTA | — | ✗ | 基线 |
| E3.1 | SNR 感知 | logistic | ✗ | 仅匹配改善 |
| E3.2 | SNR 感知 | logistic | ✓ | 匹配+损失一致 |
| E3.3 | SNR 感知 | exponential ($\beta=3$) | ✓ | 更激进衰减 |
| E3.4 | SNR 感知 | exponential ($\beta=6$) | ✓ | 极激进 |

### 4.2 匹配稳定性评估

对同一图，在不同 $t$ 下前向，计算匹配结果的 IoU 方差：

$$
\text{Stability}(x) = \text{Var}_t[\text{IoU}(\text{match}(x, t), \text{match}(x, t_{\text{ref}}))]
$$

预期：SNR 感知匹配在 $t \to 1$ 时方差更小（因为高噪声时匹配保守，接近空匹配）。

### 4.3 按训练阶段评估

| 训练阶段 | $t$ 分布 | 预期收益 |
|---------|---------|---------|
| 早期 | 均匀 $t \sim U[0,1]$ | 高噪声匹配改善显著 |
| 中期 | 偏低 $t$ | 收益减小 |
| 后期 | 极低 $t$ | 几乎无差异 |

### 4.4 消融

- $w(t)$ 形式：logistic vs exponential
- $k_{\max}$ 缩放：开/关
- 损失加权：开/关（验证匹配与损失不一致的危害）
- `cost_threshold`：$\{10, 50, 100, \infty\}$

---

## 5. 预期收益与风险

### 收益

1. **高噪声梯度去噪**：减少 $t \approx 1$ 时的错误匹配，降低梯度噪声
2. **训练稳定性**：早期训练（高噪声比例高）更稳定
3. **匹配质量**：低噪声时匹配更确定，提升最终检测精度
4. **实现成本低**：仅修改 matcher 和 criterion，不改模型结构

### 风险

1. **过度抑制**：$w(t)$ 过小可能导致高噪声时几乎无正样本，训练信号不足
2. **$t$ 分布偏移**：若训练时 $t$ 采样偏向低噪声（如 RF 的 shifted schedule），SNR 加权效果减弱
3. **与 box_renewal 的交互**：box_renewal 在每步重置 proposal，可能削弱 $t$ 感知的效果

### 缓解

- 设 $w(t)$ 下界（如 $\max(w(t), 0.1)$），避免完全抑制
- `cost_threshold` 设较大值（50），仅抑制极端高噪声
- 与方向四（非线性轨迹）正交，可联合实验

---

## 6. 验收标准

- [ ] `snr_weight` 单元测试：$w(0)=1, w(1)=0$，单调递减
- [ ] `SNRAwareMatcher` 单元测试：$t$=None 退化为标准匹配，$t=1$ 时正样本数显著减少
- [ ] E3.0 vs E3.2：训练早期 loss 曲线更平滑，最终 mAP 提升 0.5-1%
- [ ] 匹配稳定性：高 $t$ 时匹配 IoU 方差降低 > 30%
