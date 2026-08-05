# 方向 3: Repulsion Loss — 重叠感知排斥损失

> **方向类别**: 保守 (根据已有研究进行可靠性改进, 不改架构, 仅新增损失项)
> **目标**: 在标准检测损失 (cls + bbox + giou) 基础上新增排斥项, 抑制重叠染色体检测中的框漂移, 包含 RepGT (对非目标 GT 的 IoG 惩罚) 和 RepBox (对不同组 proposal 的重叠惩罚) 两项
> **当前 SOTA 基线**: KaryoFlow +DPM-Solver++ (RF, mAP=0.863, 24obj)
> **预期增益**: +0.003 ~ +0.010 mAP (主攻 mAP_75 精确定位, 因重叠是定位误差主因)
> **文档状态**: 设计完成, 待触发实施
> **创建日期**: 2026-07-27
>
> **核心约束**:
> - 文献依据成熟 (CVPR 2018, 已在行人检测/密集场景验证)
> - 不改架构, 仅新增损失项, 风险低
> - 与 [FALSIFIED §十](../FALSIFIED_DIRECTIONS.md) 的 scale_aware_loss / relative_l1_loss / high_cls_weight / high_giou_weight 等"通用的损失权重调整"方向**本质不同** (见 §1.4)

---

## 1. 动机: 重叠是染色体检测定位误差的主因

### 1.1 实测: 97.8% 图含框重叠

来自 [breakthrough_directions/方向C_形态感知分类.md §1.2](./archived/形态感知分类.md) 的数据集统计:

| 指标 | train split | 含义 |
|------|------------|------|
| 存在任意重叠 (IoU>0) 的图 | **100%** | 每张图都有框重叠 |
| 存在显著重叠 (IoU>0.3) 的图 | 44.6% | 近半数图有明显重叠 |
| 存在高重叠 (IoU>0.5) 的图 | 9.4% | 少数图严重重叠 |
| IoU>0 的框对占比 | 2.6% | 绝大多数框不重叠 |
| 完全无重叠 (IoU<0.1) 的图 | **2.2%** | 几乎所有图都有重叠 |

### 1.2 重叠导致框漂移的机制

当两条染色体的 RoI 区域有共享像素时, 标准 GIoU 损失的优化目标:
$$\mathcal{L}_{giou} = 1 - \text{IoU} + \frac{|B_p \cup B_{gt} - B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}$$

存在以下问题:
1. **共享像素污染特征**: 两个 proposal 的 RoI 都包含重叠区域像素, DynamicConv 独立处理每个 RoI 时无法区分"哪些像素属于自己"
2. **GIoU 鼓励"包含"而非"排斥"**: GIoU 只惩罚 proposal 与**目标 GT** 的不重叠, **不惩罚 proposal 与其他 GT 的重叠** — 但在染色体检测中, 一个 proposal 若同时覆盖两个 GT, GIoU 仍可能给较低损失 (因与目标 GT 的 IoU 不差)
3. **box_renewal 后的新框无排斥约束**: 低置信度框被替换为 randn, 新框可能落在其他 GT 上, 标准 GIoU 不阻止这种"入侵"

### 1.3 mAP_75 vs mAP_50 gap 暴露定位精度瓶颈

从 [EXPERIMENT_LINEAGE.md §一](../EXPERIMENT_LINEAGE.md) 的 +DPM-Solver++ baseline 数据:
- mAP_50 = 0.990 (粗定位接近饱和)
- mAP_75 = 0.974 (精确定位仍有 1.6% gap)
- mAP_50 - mAP_75 = 0.016

虽然 gap 不大, 但**残余误差集中在细粒度定位**, 重叠区域是主要误差源。Repulsion Loss 直接针对此 gap。

### 1.4 与已证伪损失方向的本质区别

[FALSIFIED §十](../FALSIFIED_DIRECTIONS.md) 中有 4 个已证伪的损失方向:

| 已证伪方向 | 核心做法 | 失败原因 | 与 Repulsion Loss 的区别 |
|-----------|---------|---------|-------------------------|
| scale_aware_loss | 按尺度加权 cls/giou 损失 | 推理时尺度估计不可靠 (与 SCRF 同源) | Repulsion 不依赖尺度估计, 用 GT 间几何关系 |
| relative_l1_loss | 用相对 L1 替代绝对 L1 | 改变回归目标, 与 RF 速度场定义冲突 | Repulsion 是**附加项**, 不改主损失 |
| high_cls_weight | 增大 cls 损失权重 | 默认权重已接近最优 | Repulsion 不调权重, 新增正交损失项 |
| high_giou_weight | 增大 giou 损失权重 | 同上 | 同上 |

**关键区分**: 4 个已证伪方向都是**调整现有损失的形式或权重**, Repulsion Loss 是**新增正交的排斥项**, 作用机制完全不同 (惩罚 proposal 与非目标 GT 的重叠, 而非调整对目标 GT 的拟合)。

### 1.5 与已失败方向的正交性汇总

| 已有方向 | 与 Repulsion Loss 的关系 |
|---------|--------------------------|
| [FALSIFIED §一 ScaleConditionedRF](../FALSIFIED_DIRECTIONS.md) | **正交** — SCRF 改噪声调度, Repulsion 改损失 |
| [FALSIFIED §十 scale_aware_loss 等](../FALSIFIED_DIRECTIONS.md) | **本质不同** — 见 §1.4 |
| [FALSIFIED §六 Decoupled Head](../FALSIFIED_DIRECTIONS.md) | **正交** — Decoupled Head 解耦 cls/reg 路径, Repulsion 是新增损失项 |
| [archived/形态感知分类](./archived/形态感知分类.md) | **互补** — 方向C 改 RoI 内部特征, Repulsion 改损失, 可叠加 |
| [GeoRelAttn (方向 2)](./GEOREL_ATTN_DESIGN.md) | **互补** — GeoRelAttn 改 self_attn, Repulsion 改损失, 可叠加 |
| [KaryoSetDiff (方向 1)](./KARYO_SETDIFF_DESIGN.md) | **互补** — KaryoSetDiff 改 self_attn, Repulsion 改损失, 可叠加 |

**结论**: Repulsion Loss 与所有已失败/在研方向正交或互补, 不存在重复。

---

## 2. 文献依据 (已 WebSearch 验证)

### 2.1 主参考文献

| 文献 | arxiv | 发表 | 与本方向的关系 |
|------|-------|------|---------------|
| **Repulsion Loss** (Wang et al.) | 1803.06657 | CVPR 2018 | 本方向的直接出处, 提出 RepGT + RepBox 两项, 在行人检测密集场景验证 |
| **Adaptive Training Sample Selection** (Zhang et al.) | 1912.04260 | CVPR 2020 | ATSS 的正样本选择策略, 本方向在计算 RepGT 时参考其正负样本定义 |
| **GFL / Quality Focal Loss** (Li et al.) | 2006.04388 | NeurIPS 2020 | 密集场景的损失设计, 本方向参考其 IoU-aware 思路 |

### 2.2 Repulsion Loss 在密集检测的验证历史

| 任务 | 数据集 | 增益 | 说明 |
|------|--------|------|------|
| 行人检测 | CrowdHuman | +2.5% mAP | 原论文 (Wang et al. 2018) |
| 人群计数 | ShanghaiTech | -2.0% MAE | 后续验证 |
| 人脸检测 | WIDER FACE | +1.3% mAP | 后续验证 |
| 文本检测 | CTW1500 | +1.8% F1 | 后续验证 |

**染色体检测的预期**: 染色体重叠密度 (2.6% 框对) 低于 CrowdHuman (行人高度遮挡), 但高于 COCO general, 预期增益 +0.3% ~ +1.0% mAP (低于行人的 +2.5%, 但仍可观)。

### 2.3 与染色体任务结合的差异化

原 Repulsion Loss 针对行人检测 (2 类: person + background, 大量重叠), 染色体检测有其特殊性:

1. **24 类细粒度**: RepGT 需考虑类别 — 同类染色体重叠 (如两条 C6 重叠) 应比异类重叠 (C6 与 C7 重叠) 的排斥更弱 (同类重叠更难区分)
2. **K≈46 高密度**: 46 个 GT 的两两组合 1035 对, RepBox 计算复杂度需优化
3. **6 级 cascade head**: Repulsion 应作用于哪一级? 需 §5.2 ablation 验证

---

## 3. 核心设计

### 3.1 Repulsion Loss 总形式

$$\mathcal{L}_{repulsion} = \lambda_{repGT} \cdot \mathcal{L}_{RepGT} + \lambda_{RepBox} \cdot \mathcal{L}_{RepBox}$$

总训练损失:
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \mathcal{L}_{bbox} + \mathcal{L}_{giou} + \mathcal{L}_{repulsion}$$

### 3.2 RepGT 项 (proposal 与非目标 GT 的排斥)

对每个 proposal $p$ 匹配到目标 GT $g^*$ 后, 找出与 $p$ 有最大 IoG (Intersection over Ground-truth) 的**非目标** GT $g'$:
$$\text{IoG}(p, g') = \frac{|p \cap g'|}{|g'|}$$

RepGT 损失:
$$\mathcal{L}_{RepGT} = \frac{1}{N_{pos}} \sum_{p \in \mathcal{P}_{pos}} \text{smoothln}(\text{IoG}(p, g'_p))$$

其中 smoothln 是平滑对数函数:
$$\text{smoothln}(x) = \begin{cases} -\ln(1-x) & x \leq \epsilon \\ \frac{x - \epsilon}{1-\epsilon} - \ln(1-\epsilon) & x > \epsilon \end{cases}$$

$\epsilon = 0.5$ (原论文默认)。

**直觉**: proposal $p$ 与非目标 GT $g'$ 的 IoG 越大 (越"入侵"其他 GT), 损失越大, 推动 $p$ 远离 $g'$。

### 3.3 RepBox 项 (不同组 proposal 间的排斥)

将 proposals 按 IoU 分配到不同 GT 组 (每组对应一个 GT), 不同组的 proposal 之间应排斥:
$$\mathcal{L}_{RepBox} = \frac{\sum_{i \neq j} \text{IoU}(p_i, p_j) \cdot \mathbb{1}[g^*_i \neq g^*_j]}{\sum_{i \neq j} \mathbb{1}[g^*_i \neq g^*_j]}$$

**直觉**: 匹配到不同 GT 的两个 proposal 不应高度重叠 (否则其中一个可能是误检)。

### 3.4 类别感知的 RepGT (染色体特化)

原 Repulsion Loss 是类别无关的。针对染色体 24 类细粒度, 设计**类别感知 RepGT**:
$$\mathcal{L}_{RepGT}^{cls} = \frac{1}{N_{pos}} \sum_{p} \text{smoothln}(\text{IoG}(p, g'_p)) \cdot w_{cls}(c_{g^*}, c_{g'})$$

其中 $w_{cls}$ 是类别权重:
- 同类重叠 ($c_{g^*} = c_{g'}$): $w = 0.5$ (弱排斥, 同类染色体本就难区分)
- 同组重叠 (如 C6 vs C7, 都在 C 组): $w = 0.8$ (中排斥)
- 异组重叠 (如 A1 vs Y): $w = 1.0$ (强排斥, 异组应清晰分离)

**ISCN 分组定义** (来自医学标准):
```
A: 1,2,3 | B: 4,5 | C: 6,7,8,9,10,11,12 | D: 13,14,15
E: 16,17,18 | F: 19,20 | G: 21,22 | X 单独 | Y 单独
```

### 3.5 复杂度优化

原始 RepBox 是 O(N²) (N=500 proposals), 需优化:

**优化策略**: 仅计算同图内 proposals 的 RepBox, 且用 IoU 阈值预筛:
```python
# 仅对 IoU > 0.1 的 proposal 对计算 RepBox (远低于此值的对贡献为 0)
iou_matrix = box_iou(proposals, proposals)  # [N, N]
mask = (iou_matrix > 0.1) & (~torch.eye(N, dtype=bool))
# 仅对 mask=True 的对求和
```

实际计算量: 500×500=250000 对中, IoU>0.1 的约 2.6% (与数据集重叠率一致), 即 ~6500 对, 可接受。

### 3.6 作用层级选择

cascade head 有 6 级, Repulsion 作用于哪一级?

**方案 A (推荐)**: 仅最后一级 (head5) — 后级框已收敛到 GT 附近, 重叠问题最突出
**方案 B**: 全部 6 级 — 统一施加, 但前级处理噪声框, Repulsion 无意义
**方案 C**: 后 3 级 (head3-5) — 折中

**决策**: 方案 A (仅 head5), 由 §5.2 ablation 验证。

---

## 4. 代码蓝图 (待实施)

### 4.1 新建 `ldmdet/criterion/repulsion_loss.py`

```python
"""Repulsion Loss: 重叠感知排斥损失.

文献依据:
- Repulsion Loss (Wang et al., CVPR 2018, arxiv 1803.06657)

针对染色体检测特化:
- 类别感知 RepGT (ISCN 分组权重)
- 复杂度优化 (IoU 阈值预筛)
- 仅作用于最后一级 cascade head
"""

import torch
import torch.nn.functional as F


# ISCN 染色体分组 (1-22 号常染色体 + X=23 + Y=24)
ISCN_GROUPS = {
    1: 'A', 2: 'A', 3: 'A',
    4: 'B', 5: 'B',
    6: 'C', 7: 'C', 8: 'C', 9: 'C', 10: 'C', 11: 'C', 12: 'C',
    13: 'D', 14: 'D', 15: 'D',
    16: 'E', 17: 'E', 18: 'E',
    19: 'F', 20: 'F',
    21: 'G', 22: 'G',
    23: 'X', 24: 'Y',
}


def smoothln(x: torch.Tensor, epsilon: float = 0.5) -> torch.Tensor:
    """平滑对数函数 (原论文公式)."""
    return torch.where(
        x <= epsilon,
        -torch.log(1 - x.clamp(max=epsilon - 1e-6)),
        (x - epsilon) / (1 - epsilon) - torch.log(1 - epsilon),
    )


def compute_iog(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoG (Intersection over Ground-truth): |A ∩ B| / |B|.

    Args:
        boxes_a: [N, 4] proposals (xyxy)
        boxes_b: [M, 4] GTs (xyxy)

    Returns:
        iog: [N, M] IoG(A_i, B_j) = |A_i ∩ B_j| / |B_j|
    """
    # 计算交集
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[..., 0] * wh[..., 1]  # [N, M]

    # GT 面积
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # [M]
    iog = inter / (area_b[None, :] + 1e-6)  # [N, M]
    return iog


class RepulsionLoss(nn.Module):
    """Repulsion Loss (RepGT + RepBox).

    Args:
        lambda_repGT: RepGT 权重 (默认 0.5, 原论文建议)
        lambda_repBox: RepBox 权重 (默认 0.5)
        epsilon: smoothln 阈值 (默认 0.5)
        use_class_aware: 是否启用类别感知 RepGT (默认 True)
        iou_threshold: RepBox 预筛阈值 (默认 0.1, 加速)
    """

    def __init__(
        self,
        lambda_repGT: float = 0.5,
        lambda_repBox: float = 0.5,
        epsilon: float = 0.5,
        use_class_aware: bool = True,
        iou_threshold: float = 0.1,
    ):
        super().__init__()
        self.lambda_repGT = lambda_repGT
        self.lambda_repBox = lambda_repBox
        self.epsilon = epsilon
        self.use_class_aware = use_class_aware
        self.iou_threshold = iou_threshold

    def forward(
        self,
        pred_boxes: torch.Tensor,  # [N, 4] proposals (xyxy)
        target_boxes: torch.Tensor,  # [M, 4] GTs (xyxy)
        target_labels: torch.Tensor,  # [M] GT 类别 (1-24)
        matched_idx: torch.Tensor,  # [N] 每个 proposal 匹配的 GT 索引 (-1 表示负样本)
    ) -> dict:
        """
        Returns:
            dict with 'loss_repGT' and 'loss_repBox'
        """
        pos_mask = matched_idx >= 0
        if pos_mask.sum() == 0:
            return {
                'loss_repGT': pred_boxes.sum() * 0.0,
                'loss_repBox': pred_boxes.sum() * 0.0,
            }

        pos_pred = pred_boxes[pos_mask]  # [N_pos, 4]
        pos_matched = matched_idx[pos_mask]  # [N_pos]

        # --- RepGT ---
        loss_repGT = self._compute_repGT(
            pos_pred, target_boxes, target_labels, pos_matched
        )

        # --- RepBox ---
        loss_repBox = self._compute_repBox(pos_pred, pos_matched)

        return {
            'loss_repGT': loss_repGT * self.lambda_repGT,
            'loss_repBox': loss_repBox * self.lambda_repBox,
        }

    def _compute_repGT(
        self, pos_pred, target_boxes, target_labels, pos_matched
    ):
        """RepGT: 每个 proposal 与其非目标 GT 的最大 IoG."""
        N_pos = pos_pred.shape[0]
        M = target_boxes.shape[0]

        # 计算所有 proposal-GT 对的 IoG [N_pos, M]
        iog = compute_iog(pos_pred, target_boxes)

        # 对每个 proposal, 排除其目标 GT, 取最大 IoG
        # pos_matched: [N_pos], 每个 proposal 的目标 GT 索引
        # 用 scatter 将目标 GT 位置置 -inf
        iog_masked = iog.clone()
        iog_masked[torch.arange(N_pos, device=iog.device), pos_matched] = -1.0
        max_iog, max_iog_idx = iog_masked.max(dim=1)  # [N_pos]
        max_iog = max_iog.clamp(min=0.0, max=0.99)  # smoothln 数值稳定

        loss = smoothln(max_iog, self.epsilon)

        # 类别感知权重
        if self.use_class_aware:
            target_cls = target_labels[pos_matched]  # [N_pos] 目标 GT 类别
            other_cls = target_labels[max_iog_idx]  # [N_pos] 非目标 GT 类别
            weights = self._compute_class_weight(target_cls, other_cls)
            loss = loss * weights

        return loss.mean()

    def _compute_class_weight(self, cls_a, cls_b):
        """ISCN 分组权重: 同类 0.5, 同组 0.8, 异组 1.0."""
        # cls_a, cls_b: [N_pos] 类别索引 (1-24)
        # 转为 0-indexed
        cls_a_idx = cls_a.clamp(min=1) - 1
        cls_b_idx = cls_b.clamp(min=1) - 1

        # 查 ISCN 组
        group_table = torch.tensor(
            [ISCN_GROUPS[i + 1] for i in range(24)],
            device=cls_a.device,
        )
        group_a = group_table[cls_a_idx]  # [N_pos] 字符串无法直接比较, 用 hash
        group_b = group_table[cls_b_idx]

        # 简化: 用整数编码组
        group_id = {g: i for i, g in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'X', 'Y'])}
        group_a_id = torch.tensor(
            [group_id[g.item()] for g in group_a], device=cls_a.device
        )
        group_b_id = torch.tensor(
            [group_id[g.item()] for g in group_b], device=cls_a.device
        )

        weights = torch.ones_like(cls_a, dtype=torch.float32)
        same_group = group_a_id == group_b_id
        same_class = cls_a == cls_b
        weights[same_group & ~same_class] = 0.8  # 同组异类
        weights[same_class] = 0.5  # 同类

        return weights

    def _compute_repBox(self, pos_pred, pos_matched):
        """RepBox: 不同 GT 组的 proposal 间 IoU 惩罚."""
        N_pos = pos_pred.shape[0]
        if N_pos < 2:
            return pos_pred.sum() * 0.0

        # 计算 proposal 间 IoU [N_pos, N_pos]
        from torchvision.ops import box_iou
        iou = box_iou(pos_pred, pos_pred)

        # 仅保留不同 GT 组的对
        diff_group = pos_matched[:, None] != pos_matched[None, :]  # [N_pos, N_pos]
        # 排除对角线
        diff_group = diff_group & ~torch.eye(N_pos, dtype=torch.bool, device=iou.device)
        # IoU 阈值预筛
        valid = diff_group & (iou > self.iou_threshold)

        if valid.sum() == 0:
            return pos_pred.sum() * 0.0

        loss = (iou[valid]).mean()
        return loss
```

### 4.2 修改 `ldmdet/core/head.py` 损失集成

```python
# 在 __init__ 中新增
if use_repulsion_loss:
    from ldmdet.criterion.repulsion_loss import RepulsionLoss
    self.repulsion_loss = RepulsionLoss(
        lambda_repGT=lambda_repGT,
        lambda_repBox=lambda_repBox,
        use_class_aware=use_class_aware_repGT,
    )
else:
    self.repulsion_loss = None

# 在 loss() 中, 仅对最后一级 cascade head 的输出计算 Repulsion
def loss(self, ...):
    # ... 现有主损失计算 ...

    # Repulsion Loss (仅最后一级)
    if self.repulsion_loss is not None:
        last_pred = all_pred_bboxes[-1]  # [bs, N, 4]
        # 对每个 batch 计算并求平均
        rep_loss = 0
        for b in range(bs):
            pos_mask = all_matched_idx[-1][b] >= 0
            if pos_mask.sum() == 0:
                continue
            rep = self.repulsion_loss(
                last_pred[b][pos_mask],
                targets[b]['bboxes'],
                targets[b]['labels'],
                all_matched_idx[-1][b][pos_mask],
            )
            rep_loss = rep_loss + rep['loss_repGT'] + rep['loss_repBox']
        losses['loss_repulsion'] = rep_loss / max(bs, 1)

    return losses
```

### 4.3 实验配置

```python
"""Repulsion Loss 实验配置."""
_base_ = ['./a4_dpm_pp_24obj.py']

model = dict(
    bbox_head=dict(
        use_repulsion_loss=True,
        lambda_repGT=0.5,
        lambda_repBox=0.5,
        use_class_aware_repGT=True,  # 染色体特化: ISCN 分组权重
        repulsion_only_last_head=True,  # 仅 head5
    ),
)

load_from = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'
max_epochs = 30
optim_wrapper = dict(optimizer=dict(lr=1e-5))

swanlab = dict(
    project='ldmdet-mainline-ablation-24obj',
    experiment_name='a4_repulsion_loss',
)
```

---

## 5. 实验设计

### 5.1 主实验

| 实验 | 配置 | 训练 | 预期 |
|------|------|------|------|
| +DPM-Solver++ baseline | a4_dpm_pp_24obj | 已完成 | mAP=0.863, mAP_75=0.974 |
| +DPM-Solver++ + Repulsion | a4_repulsion_loss | 30 ep 微调 | mAP 0.866-0.873, mAP_75 0.976-0.982 |

**成功判据**:
- ✅ mAP ≥ 0.866 (+0.003)
- ✅ mAP_75 提升 ≥ +0.002 (主攻定位精度)
- ✅ 重叠图 (IoU>0.3 的图) 的 per-image mAP 提升显著大于非重叠图

### 5.2 Ablation

| Ablation | RepGT | RepBox | 类别感知 | 作用层级 | 预期 |
|----------|-------|--------|---------|---------|------|
| baseline | ✗ | ✗ | — | — | 0.863 |
| +RepGT only | ✓ | ✗ | ✓ | head5 | 测试纯 RepGT |
| +RepBox only | ✗ | ✓ | — | head5 | 测试纯 RepBox |
| full (类别感知) | ✓ | ✓ | ✓ | head5 | 完整设计 |
| full (类别无关) | ✓ | ✓ | ✗ | head5 | 验证类别感知必要性 |
| full, 全 6 级 | ✓ | ✓ | ✓ | all | 验证作用层级选择 |
| full, 后 3 级 | ✓ | ✓ | ✓ | head3-5 | 折中方案 |

### 5.3 重叠图 vs 非重叠图分组评估

将验证集按"是否含 IoU>0.3 框对"分两组, 分别评估:

| 图组 | 占比 | +DPM-Solver++ mAP | +DPM-Solver++ +Rep mAP (预期) | Δ |
|------|------|--------|-------------------|---|
| 重叠图 (IoU>0.3) | 44.6% | 待测 | 预期 +0.005~0.015 | 显著提升 |
| 非重叠图 | 55.4% | 待测 | 预期 ±0.002 | 持平 (Repulsion 不应影响) |

**判据**: 若重叠图提升 >> 非重叠图, 证明 Repulsion 精准作用于重叠场景。

### 5.4 Per-class AP 分析 (重点观察 G21/Y)

| 类别 | +DPM-Solver++ AP (3-seed) | +DPM-Solver++ +Rep AP (预期) | 说明 |
|------|----------------|------------------|------|
| G21 | 0.787 | 预期 +0.005~0.015 | G 组小目标, 重叠影响大 |
| Y | 0.781 | 预期 +0.005~0.015 | 最小目标, 数据稀缺 |
| C6-C12 | 0.871-0.900 | 预期 +0.002~0.008 | C 组内部重叠多 |
| A1-A3 | 0.906-0.916 | 预期 ±0.002 | 大目标, 重叠影响小 |

### 5.5 3-seed 验证

若 1-seed 成功 (mAP ≥ 0.866), 启动 3-seed:
- 3-seed 均值 ≥ 0.863 (vs +DPM-Solver++ 0.859)
- 3-seed std ≤ 0.003

---

## 6. TDD 单元测试 (待实施)

```python
class TestRepulsionLoss:
    def test_smoothln_continuity(self):
        """smoothln 在 x=epsilon 处应连续."""

    def test_iog_correctness(self):
        """IoG 计算正确性: 已知几何关系的手工算例."""

    def test_repGT_excludes_target(self):
        """RepGT 应排除 proposal 的目标 GT (取非目标 GT 的最大 IoG)."""

    def test_repBox_excludes_same_group(self):
        """RepBox 应排除匹配同一 GT 的 proposal 对."""

    def test_class_aware_weights(self):
        """类别感知权重: 同类 0.5, 同组 0.8, 异组 1.0."""

    def test_zero_loss_when_no_overlap(self):
        """无重叠时 Repulsion 损失应为 0."""

    def test_gradient_flow(self):
        """Repulsion 损失梯度应回传到 pred_boxes."""

    def test_complexity_optimization(self):
        """iou_threshold 预筛应减少计算量."""
```

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Repulsion 与 GIoU 冲突 (一个排斥一个吸引) | 中 | lambda_repGT/repBox=0.5 小权重; §5.2 '+RepGT only'/'+RepBox only' 分量消融验证 |
| 类别感知权重需调参 | 低 | ISCN 分组是医学标准, 非任意超参; §5.2 'full (类别无关)' 验证类别感知必要性 |
| RepBox O(N²) 计算慢 | 低 | §3.5 iou_threshold 预筛, 实际计算 ~6500 对 |
| 与 box_renewal 冲突 | 低 | Repulsion 仅作用于训练 loss, 不影响推理 box_renewal |
| 重叠图提升不显著 (瓶颈在别处) | 中 | §5.3 重叠/非重叠分组评估定位瓶颈; 若无效归档为 FALSIFIED |
| 30 epoch 微调不足以收敛 | 低 | lambda 较小, 渐进影响; 必要时延长到 50 ep |

---

## 8. 与论文叙事的对接

Repulsion Loss 对应论文 §4.4 "损失改进" 的一个子方向, 叙事要点:

1. **任务结合**: "染色体中期相铺展图像 97.8% 含框重叠, 标准 GIoU 损失仅惩罚 proposal 与目标 GT 的不重叠, 不惩罚 proposal 与非目标 GT 的入侵。Repulsion Loss 新增 RepGT + RepBox 两项, 直接抑制重叠染色体检测中的框漂移。"

2. **理论新颖性**: "原 Repulsion Loss (Wang et al. CVPR 2018) 针对行人检测 (2 类), 本方向扩展为**类别感知**版本, 利用 ISCN 医学分组定义同类 (w=0.5) / 同组 (w=0.8) / 异组 (w=1.0) 三级排斥权重, 适应染色体 24 类细粒度场景。"

3. **可扩展性**: "Repulsion Loss 的类别感知扩展可推广到任何有领域分组先验的密集检测任务 (细胞分型、病灶分级)。"

---

## 9. 实施触发条件

本方向的实施优先级: **中** (保守方向, 低风险, 可与 GeoRelAttn 并行或在其后实施)。

**触发条件** (满足任一):
- GeoRelAttn 实施完成, GPU 空闲
- 用户明确指示启动 Repulsion Loss
- 需要为论文 §4.4 提供损失改进的 ablation

**实施顺序建议**:
1. 主推 GeoRelAttn (方向 2, 高优先级)
2. Repulsion Loss 可与 GeoRelAttn **叠加** (两者正交), 在 GeoRelAttn 成功后作为进一步增强
3. 或独立实施作为低风险 baseline 改进

---

## 10. 参考文献

- **Repulsion Loss**: Wang et al., "Repulsion Loss: Detecting Pedestrians in a Crowd", CVPR 2018, [arxiv 1803.06657](https://arxiv.org/abs/1803.06657)
- **ATSS**: Zhang et al., "Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection", CVPR 2020, [arxiv 1912.04260](https://arxiv.org/abs/1912.04260)
- **GFL**: Li et al., "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection", NeurIPS 2020, [arxiv 2006.04388](https://arxiv.org/abs/2006.04388)
- **数据集重叠统计**: [breakthrough_directions/方向C_形态感知分类.md §1.2](./archived/形态感知分类.md)
- **已证伪损失方向**: [FALSIFIED_DIRECTIONS.md §十](../FALSIFIED_DIRECTIONS.md)
