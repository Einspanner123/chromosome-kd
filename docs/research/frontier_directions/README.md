# 前沿范式创新方向 (Frontier Paradigm Innovations)

> **背景**：经过方向 A-G 的实验与瓶颈分析，确认 DDPM 范式在染色体检测任务上已达结构性瓶颈。
> 本目录收录**脱离 DDPM 范式**的前沿创新方向，按时间维度和风险等级分层。
>
> **决策依据**：方向 A-F 实验结果是判断"是否该脱离 DDPM"的实证依据。
> 若方向 D (BoxRefine) 收益明显 → 走 Consistency Model / Cascade 精化；
> 若方向 G (LAMFPN) 收益明显 → 先优化特征再考虑换范式;
> 若所有方向边际收益 < 0.5% → 果断换 Flow Matching / Consistency Model。

---

## 方向索引

### 已证伪方向 (2026-07-12, 24obj 数据集)

| 方向 | 实验结果 | 基线 (a1) | Delta | 失败原因 |
|------|---------|-----------|-------|---------|
| ~~**N: 端到端可微 Cascade**~~ | mAP=0.684 (epoch 41) | 0.856 | **-0.172** | 去 detach 后训练严重不稳定, mAP 震荡 0.35~0.68 |
| ~~**H: Flow Matching 速度预测**~~ | mAP=0.823 (epoch 95) | 0.856 | **-0.033** | 级联架构与速度预测不兼容, Head 1-5 输入≈x_0 丢失 x_noise 信息, 速度损失收敛到 Var(x_noise)=4 注入梯度噪声 |

> **SwanLab**: https://swanlab.cn/@einspanner/ldmdet-frontier-directions
> - Direction N: runs/tb983lhy
> - Direction H: runs/sh5750nr

### 短期 (1-3 个月)：直接演进，低风险

| 方向 | 核心改动 | 预期收益 | 风险 |
|------|---------|---------|------|
| **I: Consistency Model 检测** | 自一致性约束, 单步直接出框 | mAP_90 +5-10% | 中 (4 维框 EMA 稳定性存疑) |

### 中长期 (6-12 个月)：彻底脱离生成范式

| 方向 | 核心改动 | 预期收益 | 风险 |
|------|---------|---------|------|
| **J: 确定性 Cascade 精化** | 移除时间步, 端到端迭代回归 | mAP_90 +10-15% | 中高 |

### 长期研究 (12+ 个月)：领域特化创新

| 方向 | 核心改动 | 预期收益 | 风险 |
|------|---------|---------|------|
| **K: 核型结构化生成** | GNN 建模框间关系, 结构化输出 | 论文级创新 | 高 |
| **L: 形态先验强注入** | 自监督形态编码器 + metric learning | 分类误差 -20% | 中高 |
| **M: Score-based 检测** | 学习 score ∇log p(x), Langevin 采样 | 理论探索 (4 维框无优势) | 高 (非优先) |

> **方向间关系**: 详见 [方向间关系与纠正说明.md](方向间关系与纠正说明.md)。核心关系: N 是 J 的前置验证; H ⊥ N (正交可组合); I 消除 exposure bias (H 只减少); M 降级为理论探索。

---

## 决策树 (已更新 2026-07-12)

```
方向 N (已证伪: mAP=0.684, -0.172) ─── detach 不是可优化方向
        │
        ▼
方向 H (已证伪: mAP=0.823, -0.033) ─── 级联架构与速度预测不兼容
        │
        ▼
方向 I: Consistency Model ──── 下一个验证方向
        │
        ├─ 收益 > 1%? ──── 是 ──→ 走 J (确定性 Cascade)
        │
        否
        │
        ▼
方向 J: 确定性 Cascade 精化 (移除时间步)
        │
        ▼
若追求论文创新 → K / L
(M 仅理论探索, 非优先)
```

**关键结论**: H 和 N 均已证伪。detach 截断和速度损失都不是性能瓶颈。
下一步应验证方向 I (Consistency Model) 或方向 J (确定性 Cascade)。

## 文献依据

- **FlowDet** (arxiv 2512.16771, 2025-12): 首个 CFM 检测, COCO +3.6% AP
- **Consistency Models** (Song et al., 2023): 单步生成理论框架
- **DiffusionDet** (ICCV 2023): 我们当前的基线范式
- **Discriminative Flow Matching** (arxiv 2603.13928, 2026-03): 局部 flow predictor
