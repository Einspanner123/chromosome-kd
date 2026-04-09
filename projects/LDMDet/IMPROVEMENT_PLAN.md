# LDMDet 改进方案

## 当前最佳结果

| 实验 | Best mAP | mAP@50 | mAP@75 | vs 基线 | 核心改进 |
|---|---|---|---|---|---|
| `ldmdet_baseline` | 0.725 | 0.921 | 0.812 | — | DiffusionDet 基线（DDPM, scale-shift, 1步） |
| `ldmdet_rf` | 0.733 | 0.939 | 0.828 | +0.8% | RF 替代 DDPM |
| `ldmdet_rf_shifted_schdule` | 0.747 | 0.937 | 0.837 | +2.2% | RF + shifted schedule |
| `ldmdet_flowdet_adaln` | 0.751 | 0.943 | 0.843 | +2.6% | RF + AdaLN-Zero + shifted + Heun（最佳） |
| `ldmdet_flowdet_sinkhorn` | 0.720 | 0.928 | 0.809 | -0.5% | RF + Sinkhorn OT（下降） |
| `ldmdet_flowdet_adaln_ot` | 0.735 | 0.941 | 0.832 | +1.0% | RF + AdaLN-Zero + OT |

---

## 问题1：仅限染色体数据集，缺少 COCO 等标准 Benchmark

### 解决方案

1. 基于 `ldmdet_baseline.py` 创建 COCO 版配置，修改 `num_classes=80`，`data_root` 指向 COCO 路径
2. 训练设置对齐 DiffusionDet 原论文：ResNet-50 + FPN，1x schedule（12 epochs），batch_size=16
3. 必须跑的 3 组实验：
   - `ldmdet_baseline`（DiffusionDet 基线）→ 对齐原论文 45.5 mAP
   - `ldmdet_flowdet_adaln`（最佳模型）→ 证明 RF+AdaLN 在 COCO 上也有效
   - `ldmdet_rf_shifted_schdule`（中间消融）→ 验证 shifted schedule 的通用性

---

## 问题2：OT 耦合反而降低性能

### 诊断

1. 可视化 OT 分配矩阵热力图，观察分配是否过于集中
2. 对比 OT vs SimOTA 的匹配统计（每个 GT 平均匹配 proposal 数量、前景/背景比例）
3. 检查 OT 的代价矩阵设计是否适合检测场景

### 修复方案

**方案 A：放松 OT 约束（推荐）**
- 修改质量约束 b，允许每个 GT 接收更多 proposal
- `proposals_per_gt = max(3 * N // max(K, 1), 1)`

**方案 B：OT + SimOTA 混合匹配（最推荐）**
- 训练时：用 SimOTA 做标签分配（保证检测质量）
- OT 仅用于 (noise, GT) 配对（减小传输距离）
- OT 不替代 SimOTA，而是替代随机噪声采样

**方案 C：调整 Sinkhorn 超参**
- epsilon ∈ {0.5, 1.0, 2.0}（当前 0.1 可能太小）
- dustbin_cost ∈ {3.0, 5.0, 7.0}（当前 10.0 可能太大）

**方案 D：OT 仅用于噪声初始化**
- 不用 OT 做标签分配，而是用 OT 指导噪声框的初始化位置

---

## 问题3：模块组合无叠加效应

### 诊断

`full`（0.740）< `adaln`（0.751），各模块组合时存在梯度冲突或优化困难。

### 修复方案

**方案 A：分阶段训练**
- Stage 1: 只训练 RF + shifted schedule（收敛到 0.747）
- Stage 2: 冻结 backbone/neck，只微调 detection head 加 AdaLN-Zero

**方案 B：梯度冲突分析 + 损失权重调优**
- 分析不同 loss 之间的梯度方向
- 调整 velocity_loss_weight ∈ {0.1, 0.5, 1.0, 2.0}

**方案 C：模块逐一叠加的消融实验**
- 补齐缺失的消融点：RF+shifted+Heun、RF+shifted+Heun+AdaLN+structured_noise 等

---

## 问题4：无推理速度数据

### 解决方案

编写 benchmark 脚本，测量不同采样步数的 FPS 和 mAP：
- 对比 DiffusionDet baseline vs LDMDet 各变体
- 测量 1步/2步/4步/8步/10步 的 FPS
- 同时记录每个 steps 的 mAP

---

## 问题5：Consistency/Reflow 无训练结果

### Consistency Distillation 实验

1. 用最佳模型 ldmdet_flowdet_adaln 作为 teacher
2. 创建 consistency 蒸馏配置，student 只用 1 步
3. 目标：1步推理 mAP ≥ 0.73

### Reflow 实验

1. 用最佳模型生成 (noise, detection) 配对
2. 用配对数据重新训练
3. 目标：1步推理 mAP ≥ 0.74

---

## 问题6：提升幅度偏小（+2.6%）

### 方案 A：更强的 Backbone
- Swin-Transformer-L + FPN，预期 +3-4% mAP

### 方案 B：训练策略优化
- 增加训练 epoch：150 → 300
- 更大 batch size
- 使用 EMA
- 数据增强（当前 NoAug！）

### 方案 C：检测头改进
- 增加 proposal 数量 500 → 1000
- 堆叠 2-3 层检测头
- 分类头层数 1 → 3

### 方案 D：理论创新提升贡献度
- 量化测量 ODE 轨迹的曲率
- OT 耦合的理论分析
- 提出新的评估指标（采样效率、轨迹直度）

---

## 优先级排序

| 优先级 | 问题 | 方案 | 预期收益 | 工作量 |
|---|---|---|---|---|
| P0 | 推理速度数据 | benchmark 脚本 | 论文核心卖点 | 1天 |
| P0 | 数据增强 + 训练策略 | 方案6B | +2-3% mAP | 2-3天 |
| P1 | COCO 实验 | 方案1 | 通用性证明 | 3-5天 |
| P1 | OT 诊断+修复 | 方案2B（混合匹配） | 修复负面效果 | 2-3天 |
| P2 | Consistency/Reflow 实验 | 方案5 | 单步推理卖点 | 5-7天 |
| P2 | 消融补齐 | 方案3C | 完整性 | 3-5天 |
| P3 | 更强 Backbone | 方案6A | +3-4% mAP | 3-5天 |
| P3 | 理论分析 | 方案6D | 提升论文深度 | 1-2周 |
