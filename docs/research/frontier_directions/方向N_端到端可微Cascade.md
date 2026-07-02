# 方向 N：端到端可微 Cascade Head (Differentiable Cascade Head)

> **目标**: 去掉 head 间的 detach, 让梯度贯通所有 stage, 使前一个 head 的框回归直接优化后一个 head 的 RoI 采样质量, 攻克 RoI 特征质量不足导致的高 IoU 瓶颈 (mAP_90 = 0.48)。
>
> **与方向 J 的关系**: 方向 J 是概念性方向 (移除 t, 回归确定性精化); 方向 N 是**可立即实施的具体架构方案**, 保留扩散框架但解决 detach 瓶颈, 是通向方向 J 的第一步。
>
> **理论依据**:
> - Cascade R-CNN: Cai & Vasconcelos, CVPR 2018 — 多 stage 精化
> - Sparse R-CNN: Sun et al., CVPR 2021 — 迭代精化 + 可学习查询
> - RoIAlign 可微性: torchvision RoIAlign 对框坐标支持梯度回传 (双线性插值)
>
> **当前代码位置**:
> - [ldmdet/core/head.py:213](../../ldmdet/core/head.py#L213) — `curr_bboxes = pred_bboxes.detach()` (梯度截断)
> - [ldmdet/core/single_head.py:181-182](../../ldmdet/core/single_head.py#L181-L182) — RoI 采样位置 = 噪声框
> - [ldmdet/core/roi_extractor.py:73-123](../../ldmdet/core/roi_extractor.py#L73-L123) — RoIAlign 固定网格采样

---

## 1. RoI 特征质量不足的根因分析

诊断显示 mAP_90 = 0.4834 (远低于 mAP_50 = 0.9428), 高 IoU 精度差。根因有 4 层, 从代码可清晰追溯:

### 根因 1: RoI 采样位置是"噪声框", 不是"真实目标位置"

[single_head.py:181-182](../../ldmdet/core/single_head.py#L181-L182):
```python
rois = bbox2roi([bboxes[i] for i in range(bs)])   # bboxes = x_t (噪声框)
roi_features = pooler(features, rois)              # 在噪声框位置采样
```

`bboxes` 来自 `x_t` (扩散中间态)。**关键问题**:
- 训练时 t 随机, 高 t 时 x_t ≈ 纯噪声 → RoIAlign 在**图像随机位置**采样 → 特征与真实染色体无关
- 推理时初始 x_raw 是 randn → 同样采样到背景或错位位置
- backbone 特征是**空间对齐**的 (CNN 平移不变性), 但在错位框上提取的特征本质是噪声

**这是 4 维框扩散最致命的问题**: 图像扩散中 x_t 是整图加噪, 特征仍在原位置; 框扩散中 x_t 是错位框, **采样位置错了**。

### 根因 2: RoIAlign 固定网格采样, 不感知染色体形态

[roi_extractor.py:40-46](../../ldmdet/core/roi_extractor.py#L40-L46):
```python
RoIAlign(output_size=7, sampling_ratio=2, ...)   # 固定 7×7 网格, 每格 2 点采样
```

7×7 网格对染色体这种**细长结构** (长宽比 1:4~1:10) 严重欠采样:
- 染色体长臂方向只有 7 个采样点, 着丝粒、臂长比例等关键形态信息丢失
- 网格是**规则**的, 不知道染色体怎么摆、弯不弯
- 方向 A2 (DeformableRoIAlign) 尝试缓解, 但偏移量也是从噪声框特征学的, 治标不治本

### 根因 3: proposals 初始化是 RoI 特征的全局平均池化

[single_head.py:184-185](../../ldmdet/core/single_head.py#L184-L185):
```python
if proposals is None:
    proposals = roi_features.flatten(2).mean(-1)   # 全局平均池化 7×7 → 1
```

第一个 head 的 proposals 是 RoI 特征的**全局平均池化**, 这把 7×7 空间信息压成 1 维, 丢失了:
- 染色体的空间布局 (着丝粒在中间、臂向两端延伸)
- 哪个区域是染色体主体 vs 背景

后续 head 的 proposals 是前一个 head 的输出, 但**初始信息已丢失**。

### 根因 4: head 间 detach 切断梯度, 无法端到端优化采样位置

[head.py:213](../../ldmdet/core/head.py#L213):
```python
curr_bboxes = pred_bboxes.detach()   # ← 梯度截断
```

6 个 head 串行, 但每个 head 的 RoI 采样位置是**前一个 head 的 detach 输出**:
- head_1 预测的框 → detach → head_2 在这框上采样 → head_2 的梯度**无法回传**让 head_1 学得更好
- 等价于 6 次独立预测, 不是真正的迭代精化
- 训练时 head_1 不知道自己预测的框会让 head_2 采样得更好还是更差

### 因果链总结

```
x_t 是噪声框 (扩散本质)
    ↓
RoIAlign 在噪声位置采样 (根因1)
    ↓
7×7 固定网格欠采样染色体形态 (根因2)
    ↓
全局平均池化丢失空间布局 (根因3)
    ↓
head 间 detach 无法优化采样位置 (根因4)
    ↓
RoI 特征质量差 → x_0_pred 精度上限低 → mAP_90 = 0.48
```

**关键洞察**: 根因 1-3 是"采样质量"问题, 根因 4 是"采样位置无法优化"问题。本方向 (N) 直接攻击根因 4, 间接缓解根因 1 (端到端后 head_1 会主动学"让 head_2 采样得更好"的框)。

---

## 2. 架构设计

### 2.1 核心思想

去掉 detach, 让梯度贯通所有 head, 使前一个 head 的框回归**直接优化**后一个 head 的 RoI 采样质量。

| 当前架构 | 端到端架构 |
|---------|----------|
| `curr_bboxes = pred_bboxes.detach()` | `curr_bboxes = pred_bboxes` (不 detach) |
| 6 head 独立预测 + ensemble | 6 head 级联精化, 只监督最后一个 |
| 训练: 每个 head 单独监督 (deep_supervision) | 训练: 所有 head 共同优化最终框 |
| RoI 采样位置不可导 | RoI 采样位置可导 (通过 pred_bboxes) |

### 2.2 完整架构

```python
class DifferentiableCascadeHead(nn.Module):
    """端到端级联 head: 梯度贯穿所有 stage, RoI 采样位置可导.

    关键改进:
    1. 不 detach: curr_bboxes = pred_bboxes (梯度回传)
    2. RoIAlign 用可微采样 (支持 pred_bboxes 的梯度)
    3. 每个 stage 的 RoI 特征质量被后续 stage 的损失优化
    4. 只监督最后一个 stage 的输出 (避免中间 stage 梯度冲突)
    """

    def forward(self, features, bboxes, t):
        """
        Args:
            features: FPN 特征 [P2, P3, P4, P5]
            bboxes: (bs, N, 4) 初始框 (x_t, 噪声态)
            t: (bs,) 扩散时间
        Returns:
            cls_logits: (bs, N, num_classes) 最后一个 stage
            pred_bboxes: (bs, N, 4) 最后一个 stage
        """
        time_emb = self.time_mlp(t)
        curr_bboxes = bboxes          # ← 不 detach
        curr_proposals = None
        last_cls, last_pred = None, None

        for i, head in enumerate(self.head_series):
            cls_logits, pred_bboxes, curr_proposals = head(
                features, curr_bboxes, curr_proposals,
                self.roi_extractor, time_emb
            )
            # 关键: 不 detach, 梯度贯穿
            curr_bboxes = pred_bboxes   # ← 端到端
            last_cls, last_pred = cls_logits, pred_bboxes

        return last_cls, last_pred, curr_proposals
```

### 2.3 训练目标: 只监督最后 stage

```python
def loss(self, features, img_metas, gt_bboxes, gt_labels):
    # ... 扩散耦合, 生成 x_t ...

    # 只前向一次, 取最后 stage 输出
    cls_logits, pred_bboxes, _ = self(features, curr_bboxes, t_input)

    # 只对最后 stage 计算损失 (避免中间 stage 梯度冲突)
    losses = self.criterion(
        {'pred_logits': cls_logits, 'pred_boxes': pred_bboxes},
        targets, t=t
    )
    return losses
    # 梯度自动回传到所有 head, 因为没有 detach
```

---

## 3. 关键技术挑战与解决方案

### 3.1 挑战 1: RoIAlign 是否可微?

**答案: 部分可微**。RoIAlign 内部用双线性插值, 对**输入特征**可微, 对**输入框坐标**也可微 (通过双线性插值的坐标梯度)。

```python
# torchvision RoIAlign 已支持坐标梯度
rois.requires_grad_(True)
roi_features = RoIAlign(features, rois)  # rois 的梯度可回传
```

所以去掉 detach 后, pred_bboxes 的梯度能通过 RoIAlign 回传到前一个 head。这是整个方案成立的技术基础。

**验证方法**: 单元测试, 验证 `roi_extractor(features, rois)` 对 `rois` 有梯度。

### 3.2 挑战 2: 6 stage 端到端的显存爆炸

**解决方案: 梯度检查点**。保留每个 head 的前向, 但反向时重算:

```python
from torch.utils.checkpoint import checkpoint

for i, head in enumerate(self.head_series):
    # 用 checkpoint 包装, 只存中间激活的输入输出
    cls_logits, pred_bboxes, curr_proposals = checkpoint(
        head, features, curr_bboxes, curr_proposals,
        self.roi_extractor, time_emb, use_reentrant=False
    )
    curr_bboxes = pred_bboxes  # 不 detach
```

显存从 O(6 × stage_activation) 降到 O(1 × stage_activation + 6 × input_output)。

### 3.3 挑战 3: 训练时高 t 的噪声框导致 RoI 采样崩溃

**问题**: 高 t 时 x_t ≈ randn, pred_bboxes 可能是负数或超大值, RoIAlign 会采样到图像外。

**解决方案: 框 clamp + 渐进式训练**

```python
def forward(self, features, bboxes, t):
    curr_bboxes = bboxes
    for i, head in enumerate(self.head_series):
        cls_logits, pred_bboxes, _ = head(
            features, curr_bboxes, None, self.roi_extractor, time_emb
        )
        # 关键: clamp 到图像范围, 避免 RoI 采样崩溃
        pred_bboxes = self._clamp_bboxes(pred_bboxes, img_metas)
        curr_bboxes = pred_bboxes  # 不 detach
    return cls_logits, curr_bboxes

def _clamp_bboxes(self, bboxes, img_metas):
    """clamp 到图像范围 [0, H] × [0, W]"""
    h, w = img_metas[0]['img_shape'][:2]
    bboxes[..., 0::2] = bboxes[..., 0::2].clamp(0, w)
    bboxes[..., 1::2] = bboxes[..., 1::2].clamp(0, h)
    return bboxes
```

### 3.4 挑战 4: 梯度消失/爆炸 (6 stage 链式)

**解决方案 A: 残差预测 + 数值稳定**

当前 reg head 已是残差模式 (`apply_deltas` = bboxes + delta), 梯度天然通过。但 6 stage 链式可能梯度爆炸。

**数值稳定 trick: 混合 detach (渐进式)**

```python
class CascadeHead(nn.Module):
    def forward(self, features, bboxes, ...):
        curr_bboxes = bboxes
        for head in self.head_series:
            cls_logits, pred_bboxes, _ = head(...)
            # 大部分梯度, 小部分 detach 稳定
            alpha = 0.9  # 可学习或固定
            curr_bboxes = alpha * pred_bboxes + (1 - alpha) * pred_bboxes.detach()
        return cls_logits, curr_bboxes
```

**解决方案 B: 仅最后 stage 不 detach (最低风险)**

```python
for i, head in enumerate(self.head_series):
    result = head(features, curr_bboxes, ...)
    if i < len(self.head_series) - 1:
        curr_bboxes = pred_bboxes.detach()  # 中间 head 仍 detach
    else:
        curr_bboxes = pred_bboxes  # 最后一个不 detach
```

只让最后一个 head 的梯度回传到前一个 head 的框预测, 风险最低, 验证用。

### 3.5 挑战 5: deep_supervision 的兼容性

当前 deep_supervision=True 时, 每个 head 都有监督。端到端后应改为:

```python
# 方案 A: 只监督最后 stage (推荐, 避免梯度冲突)
deep_supervision = False

# 方案 B: 保留中间监督, 但权重递减 (折中)
loss_weights = [0.1, 0.1, 0.2, 0.3, 0.5, 1.0]  # 前几个 head 权重低
```

推荐方案 A, 因为端到端的核心是"所有 head 协同优化最终框", 中间监督会引入冲突目标。

---

## 4. 分阶段实施计划

### Phase 1: 最小验证 (低风险)

**目标**: 验证"去掉 detach 后梯度能否回传, mAP 是否有提升"。

**改动**: 只改 [head.py:213](../../ldmdet/core/head.py#L213) 一行:
```python
# 原: curr_bboxes = pred_bboxes.detach()
curr_bboxes = pred_bboxes  # 不 detach
```

**配置**:
```python
model = dict(
    bbox_head=dict(
        deep_supervision=False,  # 只监督最后 stage
    ),
)
```

**验证指标**: mAP, mAP_75, mAP_90, 显存占用, 训练稳定性。

**风险**: 中等 (可能梯度爆炸) → 加梯度裁剪 `clip_grad_norm=1.0`。

### Phase 2: 梯度检查点 (显存优化)

**目标**: 解决 6 stage 端到端的显存问题。

**改动**: 用 `checkpoint` 包装每个 head:
```python
from torch.utils.checkpoint import checkpoint

for head in self.head_series:
    cls_logits, pred_bboxes, curr_proposals = checkpoint(
        head, features, curr_bboxes, curr_proposals,
        self.roi_extractor, time_emb, use_reentrant=False
    )
    curr_bboxes = pred_bboxes  # 不 detach
```

**验证**: 显存从 X GB 降到 Y GB, 训练时间增加 < 30%。

### Phase 3: 框 clamp + 数值稳定

**目标**: 解决高 t 噪声框导致的采样崩溃。

**改动**: 加入 `_clamp_bboxes` 和混合 detach。

### Phase 4: 完整端到端 (如果 1-3 收益明显)

**目标**: 全部去 detach + 梯度检查点 + 残差预测。

---

## 5. 预期收益与风险

### 5.1 收益预期

| 维度 | 预期 | 原因 |
|------|------|------|
| **mAP_90** | +0.05~0.10 | RoI 采样位置被优化, 高 IoU 精度提升 |
| **mAP_75** | +0.01~0.03 | 整体精度提升 |
| **训练稳定性** | 风险中 | 6 stage 链式可能梯度爆炸 |
| **显存** | 增加 2-3x | 需梯度检查点缓解 |
| **训练时间** | +20% | 反向重算 |

### 5.2 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 梯度爆炸 | 中 | 梯度裁剪 + 混合 detach (alpha=0.9) |
| 显存 OOM | 高 | 梯度检查点 + 降低 batch_size |
| RoI 采样崩溃 (高 t) | 中 | 框 clamp 到图像范围 |
| 收益不明显 | 低 | Phase 1 先验证单 head, 再决定是否全量 |

### 5.3 与 D' 证伪的关系

D' 方向证伪了"reg head 系统性尺度收缩"假设, 但**没有证伪"单步 x_0_pred 精度上限"假设**。方向 N 直接攻击"采样位置不可导"根因, 与 D' 证伪不冲突, 反而强化了"瓶颈在 RoI 质量而非 reg head"的判断。

---

## 6. 测试计划

### 6.1 单元测试

```python
class TestDifferentiableCascade:
    def test_roi_grad_flow(self):
        """RoIAlign 对框坐标应有梯度"""
        features = [torch.randn(1, 256, 32, 32, requires_grad=True)]
        rois = torch.tensor([[0, 10.0, 10.0, 50.0, 80.0]], requires_grad=True)
        roi_feat = roi_extractor(features, rois)
        roi_feat.sum().backward()
        assert rois.grad is not None  # 框坐标有梯度

    def test_no_detach_grad_flow(self):
        """去掉 detach 后, head_1 的参数应有梯度"""
        model = build_cascade_head(num_heads=2, detach=False)
        losses = model.loss(features, gt_bboxes, gt_labels)
        losses['total'].backward()
        assert any(p.grad is not None for p in model.head_series[0].parameters())

    def test_detach_no_grad(self):
        """detach 时, head_1 的参数应无梯度 (基线对照)"""
        model = build_cascade_head(num_heads=2, detach=True)
        losses = model.loss(features, gt_bboxes, gt_labels)
        losses['total'].backward()
        assert all(p.grad is None for p in model.head_series[0].parameters())

    def test_clamp_bboxes(self):
        """框 clamp 应限制到图像范围"""
        bboxes = torch.tensor([[-100.0, -100.0, 1000.0, 1000.0]])
        clamped = model._clamp_bboxes(bboxes, img_metas)
        assert clamped.min() >= 0
        assert clamped.max() <= 256

    def test_gradient_checkpoint_memory(self):
        """梯度检查点应降低显存"""
        # 测试 with/without checkpoint 的显存占用
```

### 6.2 集成测试

```python
def test_phase1_integration():
    """Phase 1: 去 detach + deep_supervision=False"""
    cfg = Config.fromfile('direction_n_cascade_e2e.py')
    model = MODELS.build(cfg.model)
    losses = model.loss(features, gt_bboxes, gt_labels)
    total = sum(losses.values())
    total.backward()
    # 验证所有 head 有梯度
    for i, head in enumerate(model.bbox_head.head_series):
        assert any(p.grad is not None for p in head.parameters()), f'head_{i} 无梯度'
```

---

## 7. 配置文件

### 7.1 Phase 1 (最小验证)

```python
# experiments/configs/ldmdet/direction_n_cascade_e2e.py
"""方向 N Phase 1: 端到端可微 Cascade (最小验证)

仅修改:
  - head 间不 detach (梯度贯通)
  - deep_supervision=False (只监督最后 stage)
  - 梯度裁剪 clip_grad_norm=1.0 (稳定训练)

预期: mAP_90 +0.03~0.05, 验证"采样位置可导"的收益
"""

_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        deep_supervision=False,    # 只监督最后 stage
        cascade_detach=False,       # 新增: head 间不 detach
    ),
)

# 梯度裁剪
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
```

### 7.2 Phase 2-4 (完整版)

```python
# experiments/configs/ldmdet/direction_n_cascade_full.py
_base_ = ['./direction_n_cascade_e2e.py']

model = dict(
    bbox_head=dict(
        use_gradient_checkpoint=True,   # 梯度检查点
        clamp_bboxes=True,              # 框 clamp 防采样崩溃
        detach_alpha=0.9,              # 混合 detach (0=全detach, 1=全梯度)
    ),
)
```

---

## 8. 实施路线图

```
Phase 1: 单行改动验证 (1 天)
  ├─ 改 head.py:213 (去 detach)
  ├─ deep_supervision=False
  ├─ 加梯度裁剪
  └─ 训练验证 → 决定是否继续

         ↓ (如果有收益)

Phase 2: 梯度检查点 (2 天)
  ├─ checkpoint 包装每个 head
  ├─ 显存测试
  └─ 训练验证

         ↓

Phase 3: 数值稳定 (1 天)
  ├─ 框 clamp
  ├─ 混合 detach (alpha=0.9)
  └─ 训练验证

         ↓ (如果 Phase 1 无收益)

Phase 4: 诊断
  ├─ 检查梯度范数 (是否消失/爆炸)
  ├─ 检查 RoI 采样位置分布
  └─ 决定是否回退到方向 J (确定性 cascade)
```

---

## 9. 与其他方向的关系

### 9.1 互补关系

| 方向 | 关系 | 说明 |
|------|------|------|
| 方向 D (BoxRefine) | **互补** | D 攻"单步精度上限", N 攻"采样位置可导", 可叠加 |
| 方向 G (LAMFPN) | **互补** | G 提升 backbone 特征质量, N 提升 RoI 采样质量 |
| 方向 J (确定性 cascade) | **演进** | N 是 J 的第一步 (保留扩散 t), J 是 N 的最终形态 (移除 t) |
| 方向 H (Flow Matching) | **正交** | H 改扩散过程, N 改 head 架构, 可组合 |

### 9.2 优先级

**推荐顺序**: N (Phase 1) → G → D → H

理由:
1. N 的 Phase 1 改动最小 (一行代码), 风险可控, 收益可能最大 (直接攻 mAP_90)
2. 如果 N 收益明显, 说明 detach 是主要瓶颈 → 走方向 J (彻底脱离扩散)
3. 如果 N 无收益, 说明 detach 不是瓶颈 → 集中精力做 G/D (特征质量)

### 9.3 与 D' 证伪的关系

D' 证伪了"reg head 系统性尺度收缩"假设, 但**没有证伪"单步 x_0_pred 精度上限"假设**。

方向 N 直接攻击"采样位置不可导"根因, 与 D' 证伪不冲突, 反而强化了"瓶颈在 RoI 质量而非 reg head"的判断。

---

## 10. 参考文献

- Cascade R-CNN: Cai & Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection", CVPR 2018
- Sparse R-CNN: Sun et al., "Sparse R-CNN: End-to-End Object Detection with Learnable Proposals", CVPR 2021
- DINO: Zhang et al., "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection", ICLR 2023
- RoIAlign: He et al., "Mask R-CNN", ICCV 2017 (RoIAlign 可微性)
- Deformable RoI: Dai et al., "Deformable Convolutional Networks", ICCV 2017

---

## 11. 附录: 当前架构 vs 端到端架构对比

```
当前架构 (detach):
  x_t → head1 → pred1.detach() → head2 → pred2.detach() → ... → head6 → pred6
  损失: L(pred1) + L(pred2) + ... + L(pred6)  (各自独立, 互不影响)

  head1 学: "我预测的框要接近 GT" (不知道下游采样)
  head2 学: "我预测的框要接近 GT" (不知道下游采样)
  ...

端到端架构 (no detach):
  x_t → head1 → pred1 → head2 → pred2 → ... → head6 → pred6
  损失: L(pred6)  (梯度回传到所有 head)

  head1 学: "我预测的框要让 head2 采样得更好"
  head2 学: "我预测的框要让 head3 采样得更好"
  ...
  head6 学: "我预测的框要最接近 GT"

  → head1 会主动学"预测对 head2 有利的框", 而非"直接预测 GT"
  → 这是真正的迭代精化, 不是 6 次独立预测
```
