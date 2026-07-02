# 方向 J：确定性 Cascade 精化 (彻底脱离生成范式)

> **目标**: 移除扩散时间步 t, 回归确定性迭代精化, 用端到端梯度贯通的多阶段 cascade 攻克 mAP_90 瓶颈。
>
> **理论依据**:
> - Cascade R-CNN: Cai & Vasconcelos, "Cascade R-CNN" (CVPR 2018)
> - Sparse R-CNN: Sun et al. (CVPR 2021) — 迭代精化原型
> - DINO: Zhang et al. (ICLR 2023) — DETR 迭代精化
>
> **当前代码位置**:
> - [ldmdet/core/head.py:213](../../ldmdet/core/head.py#L213) — `curr_bboxes = pred_bboxes.detach()` (梯度截断)
> - [ldmdet/core/head.py:196-217](../../ldmdet/core/head.py#L196-L217) — head 串行 forward

---

## 1. 背景与动机

### 1.1 当前架构的"伪迭代"问题

当前 6 个 head 串行, 但**梯度被 detach 截断**:

```python
# head.py:213
curr_bboxes = pred_bboxes.detach()   # ← 梯度截断!
```

这意味着:
- head_2 不会让 head_1 学得更好
- 等价于 6 次独立预测 + ensemble, **不是真正的迭代精化**
- 训练目标只看单步 x_0_pred, 无多步协同

### 1.2 扩散时间步的冗余

染色体检测的特点:
- 46 条框, 数量固定
- 框分布简单 (GMM 即可拟合, 见方向 F1)
- 不需要"生成多样性"

**扩散时间步 t 的作用**: 在生成任务中控制噪声水平, 但在 4 维框检测中:
- t 引入了"加噪-去噪"的自找麻烦
- 训练-推理 t 分布不一致 (exposure bias)
- 多步采样累积误差 → mAP_90 = 0.483

### 1.3 Cascade 精化的核心思想

放弃 t, 用**确定性多阶段精化**:

```
Stage 1: 粗框提议 (从 anchor 或学习查询)
    ↓  RoIAlign + 检测头
Stage 2: 中等精化 (IoU 阈值 0.5)
    ↓  梯度贯通, 端到端训练
Stage 3: 高 IoU 精化 (IoU 阈值 0.75) ← 专攻 mAP_90
    ↓
输出
```

**关键区别**:
- **无 t**: 每步是确定性回归, 不是 t 条件化去噪
- **端到端**: head 间不 detach, 梯度贯通所有阶段
- **IoU 递增**: 每阶段用更高 IoU 阈值, 类似 Cascade R-CNN

---

## 2. 子方向说明

### J1: 端到端迭代精化 (核心改动)

**问题**: 当前 head 间 detach, 无梯度贯通。

**方案**: 移除 detach, 端到端训练多阶段 head。

**实现**:
```python
class CascadeHead(nn.Module):
    def __init__(self, num_stages=3, iou_thresholds=[0.5, 0.6, 0.7]):
        super().__init__()
        self.stages = nn.ModuleList([
            SingleDiffusionDetHead(...) for _ in range(num_stages)
        ])
        self.iou_thresholds = iou_thresholds

    def forward(self, features, proposals, gt_bboxes=None):
        curr_bboxes = proposals  # 初始提议 (anchor 或学习查询)
        all_results = []

        for stage_idx, stage in enumerate(self.stages):
            cls_logits, pred_bboxes, _ = stage(features, curr_bboxes, ...)
            all_results.append((cls_logits, pred_bboxes))

            if self.training:
                # J1: 关键 — 不 detach, 梯度贯通
                curr_bboxes = pred_bboxes  # ← 无 .detach()!
                # 用更高 IoU 阈值重新分配正样本 (Cascade 风格)
                curr_bboxes = self._resample_positive(
                    curr_bboxes, gt_bboxes, self.iou_thresholds[stage_idx]
                )
            else:
                curr_bboxes = pred_bboxes

        return all_results  # 多阶段输出, 深度监督
```

**复杂度**: 低 — 改动集中在 head.py forward, 不动 backbone。

**预期收益**: mAP_90 +5-10% (梯度贯通 + IoU 递增)。

### J2: 学习查询初始化 (替代随机噪声)

**问题**: 当前从 randn 噪声开始, 需要扩散过程"找框"。

**方案**: 用可学习查询 (类 DETR) 作为初始提议, 无需扩散。

**实现**:
```python
class LearnableProposals(nn.Module):
    def __init__(self, num_proposals=100, feat_channels=256):
        super().__init__()
        # 可学习初始框 (cxcywh 归一化)
        self.init_bboxes = nn.Parameter(torch.randn(num_proposals, 4))
        nn.init.uniform_(self.init_bboxes, -1, 1)  # 合理初始化

        # 可学习初始特征 (类 Sparse R-CNN)
        self.init_features = nn.Parameter(torch.randn(num_proposals, feat_channels))

    def forward(self, batch_size):
        bboxes = self.init_bboxes.unsqueeze(0).expand(batch_size, -1, -1)
        features = self.init_features.unsqueeze(0).expand(batch_size, -1, -1)
        return bboxes, features
```

**复杂度**: 低 — 新增可学习参数, 但移除整个扩散过程。

**预期收益**: 收敛更快, mAP +1-2% (无噪声干扰)。

### J3: 多尺度 Cascade (专攻小目标)

**问题**: 不同尺度目标需要不同精化策略。

**方案**: 每阶段用不同 FPN 层特征, 小目标用 P2, 大目标用 P5。

**实现**:
```python
class MultiScaleCascade(nn.Module):
    def forward(self, features, proposals):
        # Stage 1: 用 P5 (语义) 粗定位
        s1 = self.stage1(features[-1], proposals)
        # Stage 2: 用 P3 (平衡) 中精化
        s2 = self.stage2(features[1], s1)
        # Stage 3: 用 P2 (细节) 高精化
        s3 = self.stage3(features[0], s2)
        return [s1, s2, s3]
```

**复杂度**: 中 — 需调整 RoIExtractor 的 featmap_strides。

**预期收益**: mAPs (小目标) +3-5%。

---

## 3. 与当前架构的对比

### 3.1 关键差异

| 维度 | 当前 (DiffusionDet 风格) | 方向 J (Cascade) |
|------|------------------------|-----------------|
| **时间步 t** | 有, 随机训练多步推理 | **无**, 确定性 |
| **head 间梯度** | detach (截断) | **贯通** |
| **初始提议** | randn 噪声 | **可学习查询** |
| **采样** | 多步 ODE (4 步) | **单次前向** |
| **IoU 策略** | 全阶段相同 | **递增** (0.5→0.6→0.7) |
| **训练目标** | 单步 x_0_pred | **多阶段深度监督** |
| **exposure bias** | 有 (训练-推理 t 不一致) | **无** (确定性) |

### 3.2 保留的部分

- **backbone + neck**: 完全保留 (ResNet + FPN/LAMFPN)
- **RoIExtractor**: 保留 (可用方向 A 的 Deformable)
- **检测头结构**: 保留 SingleDiffusionDetHead, 只移除 time_conditioning

---

## 4. 实现细节

### 4.1 移除时间步条件化

```python
class CascadeSingleHead(nn.Module):
    """无时间步的检测头 (基于 SingleDiffusionDetHead 改造)"""
    def __init__(self, ...):
        super().__init__()
        # 保留: self_attn, ffn, cls_head, reg_head
        # 移除: time_mlp, time_conditioning
        # J2: 增加可学习 proposal feature 交互
        self.proposal_interaction = nn.MultiheadAttention(...)

    def forward(self, features, bboxes, proposal_features):
        roi_features = self.roi_extractor(features, bbox2roi(bboxes))
        # 与 proposal_features 交互 (类 Sparse R-CNN)
        updated = self.proposal_interaction(proposal_features, roi_features, roi_features)
        # 精化
        cls_logits = self.cls_head(updated)
        pred_bboxes = self.apply_deltas(self.reg_head(updated), bboxes)
        return cls_logits, pred_bboxes, updated
```

### 4.2 IoU 递增正样本重分配

```python
def _resample_positive(self, pred_bboxes, gt_bboxes, iou_thresh):
    """每阶段用更高 IoU 阈值重新分配正样本 (Cascade 风格)"""
    ious = bbox_iou(pred_bboxes, gt_bboxes)
    # 只保留 IoU > thresh 的作为正样本
    positive_masks = ious > iou_thresh
    # 重新分配...
    return reassigned_bboxes
```

### 4.3 深度监督训练

```python
def loss(self, all_stage_results, gt_bboxes, gt_labels):
    total_loss = {}
    for stage_idx, (cls_logits, pred_bboxes) in enumerate(all_stage_results):
        # 每阶段独立计算 loss, 用递增 IoU 阈值
        loss = self.criterion(
            cls_logits, pred_bboxes, gt_bboxes, gt_labels,
            iou_thresh=self.iou_thresholds[stage_idx]
        )
        # 后阶段权重更高 (专攻高 IoU)
        weight = 0.5 + 0.5 * stage_idx / len(all_stage_results)
        for k, v in loss.items():
            total_loss[f's{stage_idx}_{k}'] = weight * v
    return total_loss
```

---

## 5. 文件组织

```
ldmdet/
├── core/
│   ├── cascade_head.py        # 新增: CascadeHead (J1)
│   ├── learnable_proposals.py # 新增: 可学习提议 (J2)
│   └── multi_scale_cascade.py # 新增: 多尺度 cascade (J3)
├── tests/
│   └── test_direction_j.py
└── experiments/configs/ldmdet/
    ├── direction_j_cascade.py
    └── direction_j_multiscale.py
```

---

## 6. 测试计划

### 6.1 单元测试

```python
class TestCascadeHead:
    def test_no_detach_between_stages(self):
        """阶段间梯度应贯通 (无 detach)"""
        out = model(features, proposals)
        loss = out[-1][1].sum()
        loss.backward()
        # Stage 1 的参数应有梯度 (证明梯度从 Stage 3 回传)
        assert model.stages[0].cls_head[-1].weight.grad is not None

    def test_iou_threshold_increasing(self):
        """IoU 阈值应递增"""
        assert model.iou_thresholds[0] < model.iou_thresholds[1]
        assert model.iou_thresholds[1] < model.iou_thresholds[2]

    def test_single_forward_pass(self):
        """推理应单次前向 (无多步采样)"""
        results = model(features, proposals)
        assert len(results) == 3  # 3 阶段

class TestLearnableProposals:
    def test_proposals_are_learnable(self):
        """初始提议应为可学习参数"""
        assert isinstance(model.init_bboxes, nn.Parameter)

    def test_no_random_noise(self):
        """无 randn 调用 (确定性)"""
        # mock torch.randn 验证不调用
```

---

## 7. 实验计划

### 7.1 主实验

| 实验 | 配置 | 预期 |
|------|------|------|
| J1-cascade | 3 阶段端到端 | mAP_90 +5-10% |
| J2-learnable | J1 + 可学习提议 | mAP +1-2% |
| J3-multiscale | J1 + 多尺度 | mAPs +3-5% |
| J-all | J1+J2+J3 | mAP_90 +10-15% |

### 7.2 消融

| 消融 | 验证 |
|------|------|
| detach vs 不 detach | J1 核心贡献 |
| 阶段数 2/3/4 | 最优阶段数 |
| IoU 阈值 0.5/0.6/0.7 vs 0.4/0.5/0.6 | IoU 策略影响 |
| 可学习 vs randn 提议 | J2 贡献 |

---

## 8. 风险与缓解

### 8.1 端到端训练不稳定

**风险**: 多阶段梯度贯通易梯度爆炸/消失。

**缓解**:
- 残差连接: `curr_bboxes = curr_bboxes + delta` (小步精化)
- LayerNorm + 梯度裁剪
- 分阶段 warmup: 先训 Stage 1, 再解冻 Stage 2/3

### 8.2 失去框数量灵活性

**风险**: DDPM 支持动态 N (推理时改变框数), J2 固定。

**缓解**: 染色体 46 条固定, 无损失。若需灵活, 用 padding + NMS。

### 8.3 训练显存增加

**风险**: 端到端 3 阶段显存 ×3。

**缓解**:
- 梯度检查点 (`use_checkpoint=True`)
- 阶段间 checkpoint (前阶段冻结)

---

## 9. 成功指标

| 指标 | baseline (RF 4步) | 目标 (J-all) |
|------|------------------|-------------|
| mAP | 0.858 | +0.02~0.03 |
| mAP_90 | 0.483 | +0.10~0.15 |
| 推理时间 | 1.0x | 2-3x 加速 (单次前向) |
| 训练显存 | 1.0x | 1.5x (梯度贯通) |

---

## 10. 实现路线图

1. **Phase 1 (3 周)**: J1 — 端到端 cascade, 移除 detach, 验证梯度贯通
2. **Phase 2 (2 周)**: J2 — 可学习提议, 替代 randn
3. **Phase 3 (3 周)**: J3 — 多尺度 cascade (若需提升小目标)

---

## 11. 参考文献

- Cascade R-CNN: https://arxiv.org/abs/1712.00726
- Sparse R-CNN: https://arxiv.org/abs/2011.12450
- DINO: https://arxiv.org/abs/2203.03605
- DiffusionDet: https://arxiv.org/abs/2211.09788
