# PD-RF: Direct Knowledge Distillation for Efficient Rectified Flow Detection

> **方向类型**: 效率方向（推理加速，可独立投稿或与 SC-RF 叠加）
> **可叠加方向**: SC-RF（SC-RF 训练的模型作为 PD-RF 的 teacher，提供更高质量的轨迹）
> **目标会议**: MICCAI 2026 / IEEE TMI
> **预期效果**: 4-step (55ms) → 1-step (14ms) 推理加速，mAP ≥ A1 baseline + 0.005（≥0.861）
>
> **命名说明**: "PD-RF" 中的 "PD" 原指 Progressive Distillation (Salimans et al., 2022)，
> 但本方案实际采用**直接 4→1 蒸馏**（direct distillation），而非 Salimans 的渐进式
> 级联（N→N/2→...→1）。为保持与既有文档的引用一致性，保留 "PD-RF" 缩写，但读者应
> 注意此处为直接蒸馏。详见 1.2 节。

---

## 1. 问题动机

### 1.1 当前局限性

A4 配置使用 DPM-Solver++ 4步采样，单张推理 ~55ms。在临床核型分析中，一个病例包含数十张显微图像，累计推理时间影响工作流效率。

当前 1步 Euler 采样（A1 baseline）的 mAP 为 0.856，比 4步 DPM-Solver++（0.862）低 0.006。这个差距源于 Euler 1步的截断误差。

### 1.2 核心洞察

**RF 轨迹近似直线，多步求解器的精炼过程可被单步前向逼近。**

RF 的核心性质是路径直化（path straightening）：训练良好的 RF 的 $x_t \to x_0$ 映射接近线性。4步 DPM-Solver++ 的迭代精炼本质上是在这条近直线轨迹上做数值积分，而一条直线可以用单步精确表示。

**知识蒸馏**利用这一点：训练一个少步学生模型去匹配多步教师模型的输出。

本方案采用**直接蒸馏**（direct distillation）：从 4步教师直接蒸馏到 1步学生，跳过中间步数。这与 Salimans et al. 2022 的**渐进蒸馏**（progressive distillation，N→N/2→...→1 级联）不同：渐进蒸馏通过多级中间模型逐步压缩步数，每级蒸馏到上一级的一半步数；直接蒸馏则一步到位。直接蒸馏实现更简单（无需训练中间模型），但对 1步学生的表达能力要求更高，因 4→1 的压缩比大于渐进式每一级的 2:1 压缩。

### 1.3 与 Reflow 的区别

| 方面 | Reflow | PD-RF |
|---|---|---|
| 机制 | 用模型预测生成新轨迹，重新训练 | 用教师输出作为监督，蒸馏到学生 |
| 训练阶段 | 两阶段（生成轨迹 + 重新训练） | 单阶段（联合 GT + 蒸馏损失） |
| 梯度结构 | 速度损失（MSE on $v$）vs 检测损失（SimOTA匹配+Focal+L1+GIoU），目标函数形式不同 | 蒸馏损失（MSE on $x_0^{raw}$）vs 检测损失（SimOTA匹配+Focal+L1+GIoU），形式仍不同但同优化 $x_0$ |
| 已知结果 | Epoch 1 后退化（梯度冲突明确） | 预期较稳定（见 2.3 动机分析，但梯度对齐性不保证） |

**关键区别**: Reflow 的速度损失优化 $v \to (x_1 - x_0^{reflow})$，而 $x_0^{reflow}$ 是模型自身的预测（非 GT），与检测损失目标不一致，且速度损失与检测损失的梯度结构显著不同，产生明确冲突。PD-RF 的蒸馏损失和检测损失都涉及 $x_0$ 预测质量，但**梯度结构并不对齐**（蒸馏损失是 raw 空间 MSE，检测损失含 SimOTA 匹配 + Focal + L1 + GIoU，见 2.3 节分析）。PD-RF 的稳定性来自蒸馏作为**正则化信号**（将教师的多步精炼知识压缩到学生单步前向），而非梯度对齐。

---

## 2. 理论分析

> **定位声明**: 本节为动机分析（motivation analysis），非严格定理。知识蒸馏是模型压缩领域成熟技术（Hinton et al., 2015; Salimans et al., 2022），本方案将其迁移到检测 RF 的推理加速。以下分析提供理论动机，并坦诚讨论梯度结构差异等局限性，不声称为新颖理论贡献。

### 2.1 设定

- **教师模型** $\theta_T$: A4 配置，4步 DPM-Solver++，$x_0^T = \text{Solve}(\theta_T, x_1, N=4)$
- **学生模型** $\theta_S$: 1步前向，$x_0^S = f_{\theta_S}(x_1, t=0)$
- **共享初始噪声**: $x_1 \sim \mathcal{N}(0, \sigma^2 I)$，教师和学生使用相同的 $x_1$
- **共享耦合**: 同一 $x_1$ 对应同一 GT 分配，故 proposal $i$ 的输出可直接对比

### 2.2 动机分析 1：RF 轨迹直化度

**性质**: 对于训练良好的 RF，轨迹 $x_t = (1-t)x_0 + t x_1$ 的实际数值积分轨迹与线性插值的偏差 $O(\epsilon)$，其中 $\epsilon$ 是 RF 训练残差。

**论证**: RF 的训练目标是最小化 $\mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$。当训练收敛时，$v_\theta \approx x_1 - x_0$（常数速度），轨迹为直线。直线的任意步长数值积分都是精确的（单步即可）。

实际中 RF 训练残差 $\epsilon > 0$（因检测任务的复杂性），但 DPM-Solver++ 的 4步积分已能很好地处理这个残差。蒸馏的目标是让学生在 1步内逼近这个 4步结果。

**重要限定**: 
1. 此性质仅说明"RF 轨迹近似直线使得单步逼近**理论可行**"，不保证 1步学生能完全恢复 4步教师的质量。实际差距取决于 $\epsilon$ 的大小和学生模型的表达能力。
2. **box_renewal 的影响**: 推理阶段 `apply_box_renewal` 会中途替换低置信度 proposal 为随机噪声，这打破了"沿同一轨迹积分"的假设——被 renew 的 proposal 实际上跳到了新轨迹上。但 box_renewal 主要影响低置信度 proposal（背景框），高质量 proposal 的轨迹仍近似直线。蒸馏时教师关闭 box_renewal（见 3.1 节），保证 proposal 对应；学生 1步推理无 box_renewal，直接从 $x_1$ 预测 $x_0$。$\square$

### 2.3 动机分析 2：蒸馏损失与检测损失的梯度结构差异

> **修订说明**: 本节原为"命题 4（蒸馏损失与检测损失的对齐性）"，声称两者梯度方向对齐
> ($\nabla \mathcal{L}_{det} \cdot \nabla \mathcal{L}_{distill} > 0$)。该命题**不成立**，
> 原因是真实检测损失的梯度结构与 MSE 蒸馏损失显著不同。本节重写为诚实的动机分析，
> 明确梯度结构差异，并将蒸馏定位为**正则化信号**而非梯度对齐的优化。

**检测损失 $\mathcal{L}_{det}$ 的真实结构**:

检测损失包含分类与回归两部分：

$$\mathcal{L}_{det} = \mathcal{L}_{cls}(\text{Focal}) + \lambda_1 \mathcal{L}_{L1}(\text{box}) + \lambda_2 \mathcal{L}_{GIoU}(\text{box})$$

其中关键的**SimOTA 动态 Top-K 匹配**步骤（`DiffusionDetMatcher`，非匈牙利二部图匹配）：
1. 对每个样本，计算 $P$ 个 proposal 与 $G$ 个 GT 的代价矩阵（FocalLossCost + BBoxL1Cost + IoUCost）
2. 基于 IoU 动态确定每个 GT 的 $k$ 个候选 proposal（`dynamic_k`），再做去歧义分配
3. 匹配结果是**离散分配**（combinatorial, 非可微），每个 GT 分配约 $k$ 个 proposal

$\nabla_{\theta_S} \mathcal{L}_{det}$ 的结构因损失组件而异：
- **分类损失 $\mathcal{L}_{cls}$（稠密）**: 所有 $P$ 个 proposal 均参与分类（匹配 proposal 分配 GT 类别，未匹配 proposal 分配背景类），梯度稠密作用于全部 proposal
- **回归损失 $\mathcal{L}_{L1} + \mathcal{L}_{GIoU}$（稀疏）**: 仅匹配的 proposal（约 $G \cdot k$ 个）接收 box 梯度，未匹配 proposal 无 box 梯度（`fg_masks` 掩码）
- **非线性梯度**: Focal Loss 梯度 $\propto (1-p)^\gamma$ 对置信度敏感；GIoU 梯度依赖框的相对位置
- **离散匹配依赖**: 梯度通过匹配分配传播，而分配本身不可微（虽梯度可经 matched pairs 回传）

**蒸馏损失 $\mathcal{L}_{distill}$ 的结构**:

$$\mathcal{L}_{distill} = \frac{1}{B \cdot P \cdot 4} \sum_{b,p} \|x_{0,b,p}^{S,raw} - \text{sg}(x_{0,b,p}^{T,raw})\|^2$$

其中 $\text{sg}$ 是 stop-gradient（教师输出 detach）。其梯度结构为：
- **稠密激活**: 所有 $P$ 个 proposal 均接收梯度（无论是否匹配 GT）
- **线性梯度**: $\nabla_{x_0^S} \mathcal{L}_{distill} \propto (x_0^S - x_0^T)$，均匀的回归梯度
- **无匹配依赖**: 不涉及 SimOTA 匹配，梯度直接作用于所有 proposal
- **仅回归空间**: 蒸馏仅在 box raw 空间（4维），不涉及分类 logits（见 2.6 节讨论）

**梯度对齐性分析**:

两个损失的梯度**结构显著不同**，不能声称 $\nabla \mathcal{L}_{det} \cdot \nabla \mathcal{L}_{distill} > 0$：
1. **box 梯度激活范围不同**: 检测 box 损失仅作用于匹配 proposal（稀疏），蒸馏损失作用于全部 proposal（稠密）
2. **梯度形式不同**: 检测 box 损失含 L1 + GIoU（非线性、依赖框相对位置）；蒸馏损失是纯 MSE（线性梯度）
3. **梯度方向不保证对齐**: 当教师的预测 $x_0^T$ 与 GT 方向不一致时，蒸馏损失拉向 $x_0^T$ 而检测损失拉向 GT，梯度方向可能相反

**正确的理论定位：蒸馏作为 box 回归的正则化信号**

PD-RF 的稳定性不依赖梯度对齐，而依赖以下机制：

1. **教师作为 box 软监督**: $x_0^T$ 是教师 4步精炼的高质量 box 预测。蒸馏损失提供了稠密的、全 proposal 的 box 监督信号，**补充了检测 box 损失仅在匹配 proposal 上的稀疏监督**。注意：分类损失本身已是稠密的（所有 proposal 参与分类），故蒸馏主要补充的是 box 回归的稀疏性。

2. **多步知识压缩**: 教师的 4步 DPM-Solver++ 包含了迭代精炼的隐式知识（求解器修正）。蒸馏将这些知识压缩到学生的单步前向中，学生无需显式多步即可近似教师的多步行为。

3. **正则化效应**: 蒸馏损失约束学生的 box 预测空间，防止 1步前向过拟合到训练分布。这与知识蒸馏在分类任务中的正则化作用一致。

**与 Reflow 的对比（修正版）**:

Reflow 的速度损失 $\mathcal{L}_{v} = \|v_\theta - (x_1 - x_0^{reflow})\|^2$ 与检测损失的冲突是**明确的**：$x_0^{reflow}$ 是模型自身预测（非 GT），优化目标与检测损失不一致，且速度空间与坐标空间不同。

PD-RF 的蒸馏损失虽与检测损失梯度结构不同，但**优化目标一致**（都希望 $x_0^S$ 接近 $x_0^{GT}$，而 $x_0^T \approx x_0^{GT}$）。梯度冲突**可能发生但概率较低**：仅当教师预测方向与 GT 方向显著偏离时。实际中教师已收敛（mAP=0.862），偏离较小。但需注意这是 batch 平均意义的论述，逐 proposal 可能有差异——故通过 `pd_gradient_alignment` 诊断指标实证监控。

**残留风险**: 若 $\lambda$ 过大，蒸馏损失可能主导优化，使学生过度模仿教师的具体预测而非学习泛化特征。需通过 $\lambda$ 消融实验确定最优值（见 5.2 节）。$\square$

### 2.6 蒸馏范围讨论：仅 box vs 含分类

当前方案仅在 box raw 空间（4维）蒸馏，不涉及分类 logits。这是设计选择而非疏漏：

- **box 蒸馏的充分性**: 1步与4步的主要差距在 box 回归精度（Euler 1步截断误差影响 box 位置），分类质量主要取决于特征提取（backbone+neck）而非采样步数
- **分类已稠密监督**: 检测损失中分类损失（Focal）已是稠密的（所有 proposal 参与），无需蒸馏补充
- **可选扩展**: 若实验发现学生分类质量不足，可增加分类 logits 的 KL 蒸馏项：$\mathcal{L}_{cls\_distill} = \text{KL}(\text{softmax}(z^S/T) \| \text{softmax}(z^T/T))$，其中 $T$ 是温度。此为消融实验的可选项，不作为基础方案。

### 2.4 组合损失

总损失为检测损失与蒸馏损失的加权和：

$$\mathcal{L} = \mathcal{L}_{det}(\theta_S, x_1, x_0^{GT}) + \lambda \cdot \mathcal{L}_{distill}(\theta_S, \theta_T, x_1)$$

其中 $\lambda$ 控制蒸馏强度。$\lambda = 0$ 退化为标准训练，$\lambda \to \infty$ 退化为纯蒸馏。

**推荐**: $\lambda = 1.0$（等权）作为起点。由于蒸馏损失和检测损失的梯度结构不同（见 2.3），等权不保证最优，需通过消融实验（5.2 节）确定最优 $\lambda$。若训练不稳定（loss 震荡），可降低 $\lambda$；若学生收敛过慢（蒸馏信号不足），可提高 $\lambda$。

### 2.5 蒸馏空间选择

蒸馏在 **raw 坐标空间**（归一化 cxcywh × snr_scale）进行，而非像素空间：
- 教师和学生的 $x_0^{pred}$ 都在 raw 空间（通过 `_sampler.xyxy_to_raw` 转换）
- raw 空间是 RF 的自然工作空间，尺度一致
- 避免了图像空间的尺度差异

$$\mathcal{L}_{distill} = \frac{1}{B \cdot P \cdot 4} \sum_{b,p} \|x_{0,b,p}^{S,raw} - x_{0,b,p}^{T,raw}\|^2$$

其中 $B$ 是 batch size，$P$ 是 proposal 数。

---

## 3. 架构设计

### 3.1 训练流程

> **实现缺口修正说明**（基于 subagent 审查）:
> 1. **梯度流**: 原伪代码学生前向在 `torch.no_grad()` 下，蒸馏损失梯度无法回传 → 修复: 学生前向有梯度
> 2. **噪声共享**: 原伪代码教师和学生各自生成随机噪声，proposal 对应关系断裂 → 修复: 显式共享 x_raw 和耦合
> 3. **教师 x0 提取**: 原伪代码从 `predict` 的 post-NMS 结果提取，形状不兼容 → 修复: 复用 `_forward_at_t` 获取 raw x0
> 4. **box_renewal**: 教师推理中 box_renewal 替换 proposal，破坏对应 → 修复: 蒸馏时教师关闭 box_renewal

```python
def loss_with_distillation(self, features, img_metas, gt_bboxes, gt_labels):
    device = features[0].device
    bs = len(img_metas)

    # === 1. 共享噪声与耦合（核心：教师和学生必须使用同一 x_raw 和同一 GT 分配）===
    # 生成共享初始噪声 x_raw (教师和学生共用)
    x_raw_shared = torch.randn(bs, self.num_proposals, 4, device=device)
    # 构建共享耦合（同一 x_raw 对应同一 GT 分配）
    targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
    # ... 使用 x_raw_shared 做耦合，得到共享的 matched_gt_indices ...

    # === 2. 教师前向（4步 DPM-Solver++，无梯度 — 教师冻结）===
    # 关键: 教师使用 _forward_at_t 多步循环获取 raw x0, 不用 predict (post-NMS)
    # 关键: 教师推理关闭 box_renewal 和 ensemble, 保持 proposal 对应
    with torch.no_grad():
        teacher_x0_raw = self._teacher_multistep_x0(
            self.teacher_model, features, img_metas, x_raw_shared
        )  # [bs, num_proposals, 4] raw 空间

    # === 3. 学生前向（1步，标准训练流程 — 有梯度，计算检测损失）===
    # 学生使用与教师相同的 x_raw_shared 和耦合
    student_losses = self.loss_with_shared_noise(
        features, img_metas, gt_bboxes, gt_labels, x_raw_shared, targets
    )

    # === 4. 学生 1步 x0_pred（必须有梯度 — 蒸馏损失需反传到学生参数）===
    #    注意: 不在 no_grad 下！
    student_x0_raw = self._student_single_step_x0(
        features, img_metas, x_raw_shared
    )  # [bs, num_proposals, 4] raw 空间, 有梯度

    # === 5. 蒸馏损失: 学生(有梯度) vs 教师(detach) ===
    distill_loss = F.mse_loss(student_x0_raw, teacher_x0_raw.detach())

    # === 6. 组合损失 ===
    student_losses['distill_loss'] = distill_loss * self.distill_lambda
    return student_losses

def _teacher_multistep_x0(self, teacher, features, img_metas, x_raw):
    """教师多步推理获取 raw x0 (不经过 NMS, 关闭 box_renewal)."""
    time_pairs = teacher._sampler.build_time_pairs(features[0].device)
    x = x_raw.clone()
    # 蒸馏模式: 关闭 box_renewal 和 ensemble, 保持 proposal 对应
    for step_idx, (t_curr, t_next) in enumerate(time_pairs):
        _, _, x0 = teacher._forward_at_t(features, x, t_curr, img_metas)
        # DPM-Solver++ step (教师专用)
        dpm_solver = teacher._sampler.create_dpm_solver()
        if dpm_solver is not None:
            dpm_solver.reset()  # 每步重置? 需按教师配置
        if dpm_solver is not None:
            x = dpm_solver.step(x, x0, t_curr, step_idx)
        else:
            x = teacher.rf.step(x, x0, t_curr, t_next)
        # 不调用 apply_box_renewal!
    return x0  # 返回最后一步的 raw x0
```

**关键实现要点**:
- **噪声共享（必须）**: `x_raw_shared` 是教师和学生的共享输入。若不共享，proposal $i$ 在教师输出和学生输出中对应不同的随机种子，`||x_0^S[i] - x_0^T[i]||²` 无意义
- **教师 x0 提取（必须）**: 不能从 `predict` 的 post-NMS 结果提取（可变数量、已去重）。必须复用 `_forward_at_t` 的返回值（固定 `[bs, P, 4]` raw 张量）
- **box_renewal 关闭（必须）**: 教师推理时关闭 `box_renewal`，否则低置信度 proposal 被替换为随机噪声，破坏与学生的 proposal 对应
- **学生前向不能在 `no_grad` 下**: 蒸馏损失需对 $\theta_S$ 求梯度，$x_0^S$ 必须保留计算图
- **教师输出 detach**: 显式 `.detach()` 确保 stop-gradient
- **耦合共享**: 教师和学生的 GT 匹配索引必须一致（同一 `matched_gt_indices`），否则 box 对比无意义

### 3.2 推理流程

学生模型推理与标准 RF 完全相同，但仅用 1步 Euler：

```python
# 学生推理: 1步 Euler
solver_type = 'euler'
sampling_timesteps = 1
```

### 3.3 教师模型加载与学生初始化

**教师模型**从 A4 的最佳 checkpoint 加载，冻结参数：

```python
teacher_model = build_model(teacher_cfg)
teacher_model.load_state_dict(load_checkpoint(teacher_ckpt))
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
```

**学生模型初始化策略**: 学生从 A4 checkpoint 初始化（而非从头训练），原因：
1. 学生与教师架构相同（仅 solver_type 和 sampling_timesteps 不同），可直接加载权重
2. 从已收敛的 A4 初始化加速蒸馏收敛，学生只需学习"1步逼近4步"的调整
3. 避免从头训练的分类/回归基础能力重建，聚焦蒸馏目标

```python
# 学生初始化: 加载 A4 权重, 仅 solver/sampling 参数不同
student_model = build_model(student_cfg)  # solver_type='euler', sampling_timesteps=1
student_model.load_state_dict(load_checkpoint(teacher_ckpt))  # 从 A4 初始化
```

### 3.4 监控指标（SwanLab 插桩）

训练时记录：
- `pd_distill_loss`: 蒸馏损失值
- `pd_student_teacher_gap`: $\|x_0^S - x_0^T\|_2$（学生-教师预测差距，应逐渐减小）
- `pd_gradient_alignment`: $\cos(\nabla \mathcal{L}_{det}, \nabla \mathcal{L}_{distill})$（梯度余弦相似度，**诊断指标**：用于实证检验 2.3 节的梯度结构差异分析。不预设 > 0，可能为负——若持续负值则说明梯度冲突，需降低 $\lambda$。**计算成本**: 需对 $\mathcal{L}_{det}$ 和 $\mathcal{L}_{distill}$ 分别反向传播取梯度，显存与时间约翻倍，建议每 100 步采样一次而非每步计算）
- `pd_det_loss_ratio`: $\mathcal{L}_{det} / (\mathcal{L}_{det} + \lambda \mathcal{L}_{distill})$
- `pd_teacher_quality`: $\|x_0^T - x_0^{GT}\|_2$（教师预测与 GT 差距，监控教师是否在该 batch 上可靠）

---

## 4. 实现计划

### 4.1 代码修改清单

| 文件 | 修改内容 |
|---|---|
| `ldmdet/core/head.py` | 新增 `loss_with_distillation` 方法；`__init__` 新增 `teacher_model` 和 `distill_lambda` 参数 |
| `experiments/runners/train_distill.py` | 新建蒸馏训练脚本（或扩展现有 train.py） |
| `experiments/configs/ldmdet/directions/pd_rf/` | 新建 PD-RF 配置 |
| `tests/unit/test_distillation.py` | 新建单元测试 |

### 4.2 关键实现细节

**`DiffusionDetHead.__init__` 新增**:
```python
self.distill_lambda = distill_lambda  # 默认 1.0
self.teacher_model = None  # 外部注入
```

**`DiffusionDetHead.loss_with_distillation`**（含噪声共享与教师x0提取修复）:
```python
def loss_with_distillation(self, features, img_metas, gt_bboxes, gt_labels):
    if self.teacher_model is None or self.distill_lambda == 0:
        return self.loss(features, img_metas, gt_bboxes, gt_labels)

    device = features[0].device
    bs = len(img_metas)

    # 1. 共享噪声与耦合（教师和学生使用同一 x_raw 和 GT 分配）
    x_raw_shared = torch.randn(bs, self.num_proposals, 4, device=device)
    targets = self._normalize_targets(gt_bboxes, gt_labels, img_metas, bs)
    # 使用 x_raw_shared 构建共享耦合（matched_gt_indices 对教师/学生一致）

    # 2. 教师多步推理获取 raw x0（无梯度, 关闭 box_renewal, 不经 NMS）
    with torch.no_grad():
        teacher_x0 = self._teacher_multistep_x0(
            self.teacher_model, features, img_metas, x_raw_shared
        )  # [bs, P, 4]

    # 3. 学生前向（有梯度, 使用共享 x_raw, 计算检测损失）
    losses = self.loss_with_shared_noise(
        features, img_metas, gt_bboxes, gt_labels, x_raw_shared, targets
    )

    # 4. 学生 1步 x0_pred（有梯度, 不在 no_grad 下！）
    student_x0 = self._student_single_step_x0(features, img_metas, x_raw_shared)

    # 5. 蒸馏损失: 学生(有梯度) vs 教师(detach)
    distill_loss = F.mse_loss(student_x0, teacher_x0.detach())
    losses['distill_loss'] = distill_loss * self.distill_lambda
    return losses
```

### 4.3 测试计划

**单元测试** (`tests/unit/test_distillation.py`):

1. `test_teacher_model_frozen`: 教师参数不更新
2. `test_distill_loss_computation`: 蒸馏损失正确计算
3. `test_distill_loss_gradient_flow`: 蒸馏损失梯度流向学生（**核心测试**：验证无 no_grad bug，学生参数 `.grad` 非空）
4. `test_teacher_output_detached`: 教师输出 detach，不参与学生计算图
5. `test_shared_noise_proposal_correspondence`: **关键测试**：教师和学生使用同一 `x_raw_shared`，验证 proposal $i$ 的教师输出和学生输出对应同一初始噪声（非各自独立生成）
6. `test_teacher_x0_raw_shape`: 教师返回的 x0 是 `[bs, P, 4]` raw 张量（非 post-NMS 的可变长度结果）
7. `test_teacher_box_renewal_disabled`: 蒸馏时教师推理不调用 `apply_box_renewal`，proposal 数量保持 $P$
8. `test_lambda_zero_disables_distillation`: $\lambda=0$ 时无蒸馏
9. `test_student_1step_inference`: 学生 1步推理正常
10. `test_gradient_alignment_diagnostic`: 检测损失与蒸馏损失梯度余弦相似度（**诊断测试**：不预设符号，验证可计算且记录值）

**实验级验证**（非单元测试，需多 epoch 训练）:
- 蒸馏损失训练中递减
- 学生质量逐渐接近教师

### 4.4 配置设计

```python
# experiments/configs/ldmdet/directions/pd_rf/pd_rf_24obj.py
# 基于 A4 DPM-Solver++ SOTA 配置 (24obj 数据集), 学生用 1步 Euler
_base_ = ['../mainline_ablation_24obj/a4_dpm_pp_24obj.py']
model = dict(
    bbox_head=dict(
        solver_type='euler',          # 学生用 Euler (覆盖 base 的 dpm_solver_pp)
        sampling_timesteps=1,         # 1步推理 (覆盖 base 的 4步)
        use_distillation=True,
        distill_lambda=1.0,
    ),
)
# 教师配置与 checkpoint (A4 最佳模型)
teacher_config = 'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py'
teacher_checkpoint = 'work_dirs/a4_dpm_pp_24obj/best.pth'
```

---

## 5. 实验设计

### 5.1 主实验

| 实验 | 配置 | 步数 | 推理时间 | 预期 mAP | 目的 |
|---|---|---|---|---|---|
| A4 teacher (24obj) | DPM-Solver++ | 4 | ~55ms | 0.862 | 教师基线（24obj 数据集 SOTA） |
| A1 baseline (24obj) | Euler | 1 | ~14ms | 0.856 | 无蒸馏 1步基线 |
| **PD-RF** (24obj) | Euler + distillation | 1 | ~14ms | **0.858~0.862** | 蒸馏 1步（目标: ≥ A1 baseline + 0.005，即 ≥0.861） |

**注**: 所有实验使用 24obj 完整实例标注数据集。目标设定为 ≥ A1 baseline + 0.005（而非 ≥0.95×teacher），因后者已被 A1 baseline（0.856）满足，无区分度。

### 5.2 消融实验

| 消融 | 目的 |
|---|---|
| PD-RF $\lambda$=0.0/0.5/1.0/2.0/5.0 | 最优蒸馏权重 |
| PD-RF 1步 vs 2步 | 步数-质量权衡 |
| PD-RF from A4 vs from SC-RF | 教师质量对蒸馏的影响 |

### 5.3 效率评估

| 模型 | 步数 | 单图推理 | 50图/病例 | 加速比 |
|---|---|---|---|---|
| A4 | 4 | 55ms | 2.75s | 1.0× |
| PD-RF | 1 | 14ms | 0.70s | **3.9×** |

---

## 6. 预期贡献

1. **方法**: 将直接知识蒸馏应用于检测 RF，实现 4→1 步推理加速（3.9×），mAP ≥ A1 baseline + 0.005
2. **分析**: 从 RF 轨迹直化度（动机分析 1）和梯度结构差异（动机分析 2）双视角提供动机分析，并坦诚讨论：(a) 蒸馏损失与检测损失梯度结构的不同（MSE vs SimOTA 匹配+Focal+L1+GIoU）；(b) 蒸馏定位为 box 正则化信号而非梯度对齐优化；(c) 噪声共享、教师x0提取、box_renewal 等实现关键点
3. **实践**: 共享噪声 + 教师冻结 + 学生有梯度前向 + stop-gradient + box_renewal关闭 的正确实现；10个单元测试覆盖关键行为（含噪声共享、梯度流、教师x0形状等核心测试）；$\lambda$ 消融实验确定最优蒸馏强度

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 1步质量差距大 | 中 | 高 | 可退至 2步蒸馏，仍比 4步快 2× |
| 蒸馏训练不稳定/梯度冲突 | 中 | 中 | 蒸馏作为 box 正则化信号（见 2.3 动机分析 2），梯度结构差异已明确分析；通过 `pd_gradient_alignment` 监控梯度余弦相似度，若持续负值则降低 $\lambda$ |
| 教师过拟合 | 低 | 低 | A4 使用 save_best 机制，取验证最优 checkpoint |
| 训练显存增加 | 中 | 中 | 教师无梯度，可用 inference_mode；学生前向共享特征；梯度对齐诊断指标需额外反向传播，建议每 N 步采样一次 |
| no_grad 实现错误 | 低 | 高 | 单元测试 `test_distill_loss_gradient_flow` 验证学生参数 `.grad` 非空；`test_teacher_output_detached` 验证教师输出 detach |
| 噪声未共享/proposal 对应断裂 | 中 | 高 | 单元测试 `test_shared_noise_proposal_correspondence` 验证共享 `x_raw`；伪代码显式生成 `x_raw_shared` 并传入教师和学生 |
| 教师 x0 提取错误 (post-NMS) | 中 | 高 | 单元测试 `test_teacher_x0_raw_shape` 验证返回 `[bs, P, 4]`；使用 `_forward_at_t` 而非 `predict` |
| box_renewal 破坏对应 | 中 | 中 | 单元测试 `test_teacher_box_renewal_disabled`；教师蒸馏推理显式关闭 box_renewal |
| 仅蒸馏 box 不蒸馏分类 | 低 | 低 | 分类损失已稠密（见 2.6 节）；若实验不足可增加 KL 蒸馏作为消融 |
