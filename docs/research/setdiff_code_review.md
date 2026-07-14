# SetDiff 代码审查问题记录

> 审查来源: 2026-07-13 ~ 07-14 代码审查 (subagent × 2 + 第三方 Reviewer)
> 状态: P0 待修复, P1/P2 待排期

---

## P0: 必须修复（当前 mAP=0 的直接原因）

### P0-1: 无背景类机制 — 未匹配 slot 完全无监督

**来源**: Agent 1 (CRITICAL) + Agent 2 (角度4 🔴) + Reviewer

**文件**: `setdiff/criterion/set_loss.py:163-180`

**问题**: 未匹配 slot（label=-1）被完全排除在分类损失之外:
```python
valid_mask = matched_labels >= 0
flat_logits = pred_logits[valid_mask]  # 仅取匹配的 slot
```
推理时 N=300 个 slot 全部输出 ~0.5 置信度 → 大量 FP → mAP=0。

**修复**: cls_head 输出 `num_classes+1`，未匹配 slot 分配 background 类 ID。

### P0-2: Self-Attention 噪声注入 — 未匹配 slot 污染活跃状态

**来源**: Reviewer (独有)

**文件**: `setdiff/core/set_encoder.py` (Transformer Decoder 的 self-attention 机制)

**问题**: 即使修复了 P0-1 的背景类 loss，253 个未匹配 slot 的输入仍是纯噪声（N(0,1)），它们在 decoder self-attention 中与匹配 slot 交互，通过加权平均将随机噪声注入活跃状态的表示。

与 DETR 的核心差异: DETR 的匈牙利匹配基于网络预测（动态），所有 query 都有意义。SetDiff 中未匹配 slot 永远只是噪声（无梯度改善它们）。

**修复**: 需要 attention mask 或架构调整分离未匹配 slot 的注意力路径。

---

## P1: 高优先级（影响收敛质量）

### P1-1: 噪声↔GT 匹配尺度不匹配

**来源**: Agent 1 (CRITICAL) + Agent 2 (角度2 ⚠️)

**文件**: `setdiff/matching/hungarian.py:69-74`

**问题**: HungarianMatcher 的匹配代价仅使用 L2 距离，噪声 x₁ ~ N(0,1) 与 GT x₀ ∈ [0,1] 尺度不匹配，L2 距离被噪声幅值主导 → 匹配本质随机。

### P1-2: 推理时噪声聚类角点

**来源**: Reviewer (独有)

**文件**: `setdiff/core/rectified_flow.py` (Euler 采样的初始噪声抽样)

**问题**: 推理时 N(0,1) 采样可能所有 300 个点落在图像同一角落（cx,cy ∈ [-3,-2]），4 步 Euler 步进后预测框集中在左上角 → 整图检测失败。

**可能的缓解**: 对初始噪声做空间采样约束（拒绝采样/混合策略）。

---

## P2: 可优化（样本效率/收敛速度）

### P2-1: 无 auxiliary losses

**来源**: Agent 1 (WARNING)

**文件**: `setdiff/core/set_encoder.py:130-135`

**问题**: `nn.TransformerDecoder(num_layers=6)` 仅最后一层输出用于损失。DETR 每层 decoder 后附加预测头并计算辅助损失。

### P2-2: 匹配稳定性（随机排列永不衰减）

**来源**: Agent 2 (角度5 ⚠️)

**问题**: SetDiff 的匹配始终基于随机噪声（与网络能力无关），即使网络学会完美检测，匹配仍然随机。对比 DETR 匹配基于网络输出→随训练稳定→正反馈。

### P2-3: 时间步 t 未覆盖 [0,1] 边界

**来源**: Agent 1 (WARNING)

**文件**: `setdiff/models/set_head.py:122-123`

**问题**: `torch.rand(B)` 返回 [0,1)，t=1.0 在训练中从未被采样。但推理从 t=1.0 开始去噪。

### P2-4: 损失权重未调优

**来源**: Agent 1 (WARNING)

**文件**: `setdiff/criterion/set_loss.py:106-110`

**问题**: `weight_dict = {'loss_cls': 1.0, 'loss_box': 1.0, 'loss_diff': 1.0}`。DETR 标准为 cls=1, bbox=5, giou=2。

---

## P3: 低优先级（未来优化）

### P3-1: 无 attention mask 策略

**来源**: Reviewer (独有)

**问题**: 从固定 N=46（染色体）推广到变长 COCO 时，padding slot 的 attention behavior 需要定义。

### P3-2: 死代码 snr_scale

**来源**: Agent 1 (INFO)

**文件**: `setdiff/diffusion/set_rf.py:24-25`

**问题**: `snr_scale` 参数被接收但未被 `RectifiedFlow` 的 `q_sample/step/heun_step` 使用。

---

## 修复状态追踪

| 问题 | 严重度 | 修复人 | 预计日期 | 状态 |
|------|--------|--------|---------|------|
| P0-1: 无背景类 | 🔴 P0 | - | - | 待修复 |
| P0-2: SA 噪声注入 | 🔴 P0 | - | - | 待修复 |
| P1-1: 匹配尺度 | 🟡 P1 | - | - | 待排期 |
| P1-2: 噪声聚类 | 🟡 P1 | - | - | 待排期 |
| P2-1~4: 优化项 | 🟢 P2 | - | - | 待排期 |
| P3-1~2: 未来项 | ⚪ P3 | - | - | 待排期 |
