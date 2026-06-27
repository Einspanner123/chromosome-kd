# 方向 B：解耦分类与定位分支 (Decoupled Classification and Localization)

> **目标**：突破分类与定位损失竞争导致的性能瓶颈。诊断显示 IoU=0.5 时 Cls 误差 43.4%，IoU=0.9 时 Loc 误差 30.3%，两者在不同 IoU 阈值下交替主导，暗示共享主干导致梯度竞争。
>
> **理论依据**：
> - Decoupled Head: Wu et al., "Rethinking Classification and Localization for Object Detection" (2020)
> - Focal Loss 原始论文中分类/回归独立设计的思想
> - YOLOX 解耦头设计
>
> **当前代码位置**：[ldmdet/core/single_head.py](../../ldmdet/core/single_head.py) — `self_attn + inst_interact + FFN` 共享，末端才分 `cls_head` 和 `reg_head`

---

## 1. 问题分析

### 1.1 当前实现

[single_head.py:55-79](../../ldmdet/core/single_head.py#L55-L79) 的 SingleDiffusionDetHead:

```python
# 共享主干
self.self_attn = nn.MultiheadAttention(...)       # 共享
self.inst_interact = DynamicConv(...)             # 共享 (RoI 特征提取)
self.linear1/linear2 = FFN                        # 共享
self.norm1/norm2/norm3 = LayerNorm                # 共享
# 末端才分支
self.cls_head = ...  # 分类头
self.reg_head = ...  # 回归头
```

分类和定位损失反向传播时，梯度都流经共享的 self_attn + FFN，可能导致:
- 分类困难类 (G/F 组) 的梯度方向与定位精化需求冲突
- 训练不稳定性 (2a 实验中 grad_norm 首值 164-19344，爆炸现象普遍)

### 1.2 诊断证据

| IoU 阈值 | Cls 误差 | Loc 误差 | 主导误差 |
|---------|---------|---------|---------|
| 0.5 | 43.4% | 10.5% | **分类** |
| 0.75 | 24.7% | 18.5% | 分类略多 |
| 0.9 | 13.2% | 48.3% | **定位** |

分类和定位在不同 IoU 下交替主导，共享主干难以同时优化两者。

---

## 2. 子方向说明

### B1: 双分支 head (特征解耦)

**方案**：将 `self_attn + FFN` 拆为 cls 和 reg 各有独立分支，仅共享 `inst_interact` (RoI 特征提取，因为特征提取本身无梯度方向偏好)。

```python
# 改造后
self.shared_inst_interact = DynamicConv(...)     # 共享 (RoI 特征)
self.cls_self_attn = nn.MultiheadAttention(...)  # 分类专用
self.cls_ffn = FFN(...)                          # 分类专用
self.reg_self_attn = nn.MultiheadAttention(...)  # 回归专用
self.reg_ffn = FFN(...)                          # 回归专用
self.cls_head = ...
self.reg_head = ...
```

**收益**：分类和定位梯度独立流动，不再竞争。

### B2: 分类头 LayerNorm 稳定化

**方案**：在 cls_head 前加额外 LayerNorm，缓解分类 logit 的数值不稳定 (2a 实验中分类损失首值高达 38-142)。

```python
self.cls_norm = nn.LayerNorm(feat_channels)
# forward:
cls_feat = self.cls_norm(feat)
cls_logits = self.cls_head(cls_feat)
```

**收益**：减少分类损失爆炸，稳定训练。

### B3: 分阶段损失权重调度

**方案**：训练早期 (epoch < 30) 加大 cls 权重 (困难分类优先)，后期加大 reg 权重 (定位精化)。

```python
# 在 criterion 中
if epoch < 30:
    cls_w, reg_w = 3.0, 4.0  # 分类优先
else:
    cls_w, reg_w = 2.0, 6.0  # 定位优先
```

**收益**：根据训练阶段动态调整，避免早期定位损失淹没分类信号。

---

## 3. 推荐组合配置 (B1 + B2 + B3)

**子方向互相抵消评估**：
- B1 特征解耦 → 分类/定位梯度独立
- B2 分类稳定 → 减少分类损失爆炸
- B3 分阶段调度 → 早期分类优先，后期定位优先

三者**方向一致** (都旨在减少分类/定位竞争)，B2 强化 B1 的稳定性，B3 在 B1 解耦基础上进一步优化时机。**不会互相抵消**，可组合。

**组合配置**：

| 配置项 | baseline | B1+B2+B3 组合 |
|--------|----------|---------------|
| single_head 结构 | 共享主干 | 双分支 (B1) |
| cls_head 前置 | 无 | LayerNorm (B2) |
| loss_cls 权重 | 固定 2.0 | epoch<30: 3.0, ≥30: 2.0 (B3) |
| loss_bbox 权重 | 固定 5.0 | epoch<30: 4.0, ≥30: 6.0 (B3) |

**预期**：
- mAP50 +0.01 ~ +0.02 (分类改善)
- mAP75 +0.01 ~ +0.02 (定位改善)
- 训练稳定性提升 (grad_norm 首值从 1000+ 降至 500 以下)
- 若 mAP 整体 +0.01 ~ +0.03 且训练更稳定，方向 B 确认可行

---

## 4. 实现步骤

1. 新建 [ldmdet/core/decoupled_single_head.py](../../ldmdet/core/) — 双分支 head
2. 修改 [experiments/configs/baselines/](../../experiments/configs/baselines/) 新建 `decoupled_head.py`
3. 在 criterion 中实现分阶段权重 (通过 hook 或 train_loop)
4. 配置验证后训练

---

## 5. 风险与可行性

| 风险 | 评估 |
|------|------|
| 参数量增加 (双分支) | 增加 ~20% head 参数，backbone 不变，可接受 |
| B3 分阶段调度实现 | 需修改 criterion 或加 hook，复杂度中 |
| 双分支可能减弱特征共享的正向作用 | 仅共享 inst_interact，保留必要的特征复用 |

**结论**：方向 B 可行，B1+B2+B3 组合实验可一次性验证。与方向 A (特征增强) 互补 — A 改进特征输入，B 改进 loss/分支结构，两者可后续组合。
