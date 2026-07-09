# PD-RF: Progressive Distillation for Efficient Rectified Flow Detection

> **方向类型**: 效率方向（推理加速，可独立投稿或与 SC-RF 叠加）
> **可叠加方向**: SC-RF（SC-RF 训练的模型作为 PD-RF 的 teacher，提供更高质量的轨迹）
> **目标会议**: MICCAI 2026 / IEEE TMI
> **预期效果**: 4-step (55ms) → 1-step (14ms) 推理加速，mAP 保持 ≥0.95×teacher

---

## 1. 问题动机

### 1.1 当前局限性

A4 配置使用 DPM-Solver++ 4步采样，单张推理 ~55ms。在临床核型分析中，一个病例包含数十张显微图像，累计推理时间影响工作流效率。

当前 1步 Euler 采样（A1 baseline）的 mAP 为 0.856，比 4步 DPM-Solver++（0.862）低 0.006。这个差距源于 Euler 1步的截断误差。

### 1.2 核心洞察

**RF 轨迹近似直线，多步求解器的精炼过程可被单步前向逼近。**

RF 的核心性质是路径直化（path straightening）：训练良好的 RF 的 $x_t \to x_0$ 映射接近线性。4步 DPM-Solver++ 的迭代精炼本质上是在这条近直线轨迹上做数值积分，而一条直线可以用单步精确表示。

**渐进蒸馏**（Progressive Distillation, Salimans et al. 2022）利用这一点：训练一个少步学生模型去匹配多步教师模型的输出，逐步将 $N$ 步压缩到 $N/2$ 步，最终达到 1步。

### 1.3 与 Reflow 的区别

| 方面 | Reflow | PD-RF |
|---|---|---|
| 机制 | 用模型预测生成新轨迹，重新训练 | 用教师输出作为监督，蒸馏到学生 |
| 训练阶段 | 两阶段（生成轨迹 + 重新训练） | 单阶段（联合 GT + 蒸馏损失） |
| 梯度冲突 | 检测损失 vs 速度损失冲突 | 蒸馏目标与检测目标对齐（同为 $x_0$ 预测） |
| 已知结果 | Epoch 1 后退化 | 预期稳定（蒸馏目标与 GT 对齐） |

**关键区别**: Reflow 的速度损失与检测损失优化不同目标（路径直化 vs 检测精度），产生梯度冲突。PD-RF 的蒸馏损失和检测损失都优化 $x_0$ 预测质量，目标对齐，不会冲突。

---

## 2. 理论推导

### 2.1 设定

- **教师模型** $\theta_T$: A4 配置，4步 DPM-Solver++，$x_0^T = \text{Solve}(\theta_T, x_1, N=4)$
- **学生模型** $\theta_S$: 1步前向，$x_0^S = f_{\theta_S}(x_1, t=0)$
- **共享初始噪声**: $x_1 \sim \mathcal{N}(0, \sigma^2 I)$，教师和学生使用相同的 $x_1$
- **共享耦合**: 同一 $x_1$ 对应同一 GT 分配，故 proposal $i$ 的输出可直接对比

### 2.2 命题 3（RF 轨迹直化度）

**声明**: 对于训练良好的 RF，轨迹 $x_t = (1-t)x_0 + t x_1$ 的实际数值积分轨迹与线性插值的偏差 $O(\epsilon)$，其中 $\epsilon$ 是 RF 训练残差。

**论证**: RF 的训练目标是最小化 $\mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$。当训练收敛时，$v_\theta \approx x_1 - x_0$（常数速度），轨迹为直线。直线的任意步长数值积分都是精确的（单步即可）。

实际中 RF 训练残差 $\epsilon > 0$（因检测任务的复杂性），但 DPM-Solver++ 的 4步积分已能很好地处理这个残差。蒸馏的目标是让学生在 1步内逼近这个 4步结果。$\square$

### 2.3 命题 4（蒸馏损失与检测损失的对齐性）

**声明**: 蒸馏损失 $\mathcal{L}_{distill} = \|x_0^S - x_0^T\|^2$ 与检测损失 $\mathcal{L}_{det}$ 的梯度方向对齐，不会产生梯度冲突。

**论证**:
- 教师模型 $\theta_T$ 已训练至收敛，$x_0^T$ 是高质量预测
- 检测损失 $\mathcal{L}_{det}$ 优化 $x_0^S \to x_0^{GT}$（真实标注）
- 蒸馏损失 $\mathcal{L}_{distill}$ 优化 $x_0^S \to x_0^T$
- 由于 $x_0^T \approx x_0^{GT}$（教师已收敛），两个损失的目标近似相同
- 故梯度方向对齐：$\nabla_{\theta_S} \mathcal{L}_{det} \cdot \nabla_{\theta_S} \mathcal{L}_{distill} > 0$

这与 Reflow 不同：Reflow 的速度损失优化 $v \to (x_1 - x_0^{reflow})$，而 $x_0^{reflow}$ 是模型自身的预测（非 GT），与检测损失目标不一致。$\square$

### 2.4 组合损失

总损失为检测损失与蒸馏损失的加权和：

$$\mathcal{L} = \mathcal{L}_{det}(\theta_S, x_1, x_0^{GT}) + \lambda \cdot \mathcal{L}_{distill}(\theta_S, \theta_T, x_1)$$

其中 $\lambda$ 控制蒸馏强度。$\lambda = 0$ 退化为标准训练，$\lambda \to \infty$ 退化为纯蒸馏。

**推荐**: $\lambda = 1.0$（等权），因为两个损失目标对齐。

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

```python
def loss_with_distillation(self, features, img_metas, gt_bboxes, gt_labels, teacher_model):
    # 1. 学生前向（1步，标准训练流程）
    student_losses = self.loss(features, img_metas, gt_bboxes, gt_labels)
    
    # 2. 教师前向（4步 DPM-Solver++，无梯度）
    with torch.no_grad():
        teacher_results = teacher_model.predict(features, img_metas)
        # 提取教师的 x0_pred（在 raw 空间）
    
    # 3. 学生 1步前向获取 x0_pred（无梯度，用于蒸馏）
    with torch.no_grad():
        student_x0_pred = self._single_step_x0_pred(features, img_metas)
    
    # 4. 蒸馏损失
    distill_loss = F.mse_loss(student_x0_pred, teacher_x0_pred)
    
    # 5. 组合损失
    total_losses = student_losses
    total_losses['distill_loss'] = distill_loss * self.distill_lambda
    return total_losses
```

### 3.2 推理流程

学生模型推理与标准 RF 完全相同，但仅用 1步 Euler：

```python
# 学生推理: 1步 Euler
solver_type = 'euler'
sampling_timesteps = 1
```

### 3.3 教师模型加载

教师模型从 A4 的最佳 checkpoint 加载，冻结参数：

```python
teacher_model = build_model(teacher_cfg)
teacher_model.load_state_dict(load_checkpoint(teacher_ckpt))
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
```

### 3.4 监控指标（SwanLab 插桩）

训练时记录：
- `pd_distill_loss`: 蒸馏损失值
- `pd_student_teacher_gap`: $\|x_0^S - x_0^T\|_2$（学生-教师预测差距，应逐渐减小）
- `pd_gradient_alignment`: $\cos(\nabla \mathcal{L}_{det}, \nabla \mathcal{L}_{distill})$（梯度余弦相似度，应 > 0）
- `pd_det_loss_ratio`: $\mathcal{L}_{det} / (\mathcal{L}_{det} + \lambda \mathcal{L}_{distill})$

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

**`DiffusionDetHead.loss_with_distillation`**:
```python
def loss_with_distillation(self, features, img_metas, gt_bboxes, gt_labels):
    # 标准检测损失
    losses = self.loss(features, img_metas, gt_bboxes, gt_labels)
    
    if self.teacher_model is None or self.distill_lambda == 0:
        return losses
    
    # 教师 4步推理获取 x0_pred
    with torch.no_grad():
        teacher_pred = self.teacher_model.predict(features, img_metas)
    
    # 学生 1步推理获取 x0_pred
    with torch.no_grad():
        student_x0 = self._single_step_predict_raw(features, img_metas)
    
    # 蒸馏损失
    distill_loss = self._compute_distill_loss(student_x0, teacher_x0)
    losses['distill_loss'] = distill_loss * self.distill_lambda
    return losses
```

### 4.3 测试计划

**单元测试** (`tests/unit/test_distillation.py`):

1. `test_teacher_model_frozen`: 教师参数不更新
2. `test_distill_loss_computation`: 蒸馏损失正确计算
3. `test_distill_loss_gradient_flow`: 蒸馏损失梯度流向学生
4. `test_lambda_zero_disables_distillation`: $\lambda=0$ 时无蒸馏
5. `test_student_1step_inference`: 学生 1步推理正常
6. `test_gradient_alignment_positive`: 检测损失与蒸馏损失梯度对齐
7. `test_teacher_student_same_noise`: 教师和学生使用相同初始噪声
8. `test_distill_loss_decreases`: 训练过程中蒸馏损失递减
9. `test_student_quality_approaches_teacher`: 学生质量逐渐接近教师

### 4.4 配置设计

```python
# experiments/configs/ldmdet/directions/pd_rf/pd_rf_24obj.py
_base_ = ['../../ldmdet_rf_heun_shifted_bs8.py']
model = dict(
    bbox_head=dict(
        diffusion_type='rectified_flow',
        solver_type='euler',       # 学生用 Euler
        sampling_timesteps=1,      # 1步推理
        single_head=dict(time_conditioning='adaln_zero'),
        coupling=dict(type='ot_flow', epsilon=5.0, num_iters=20, coupling_mode='multinomial'),
        use_distillation=True,
        distill_lambda=1.0,
    ),
)
teacher_config = 'experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py'
teacher_checkpoint = 'work_dirs/a4_dpm_pp_24obj/best.pth'
```

---

## 5. 实验设计

### 5.1 主实验

| 实验 | 配置 | 步数 | 推理时间 | 预期 mAP | 目的 |
|---|---|---|---|---|---|
| A4 teacher | DPM-Solver++ | 4 | ~55ms | 0.862 | 教师基线 |
| A1 baseline | Euler | 1 | ~14ms | 0.856 | 无蒸馏 1步基线 |
| **PD-RF** | Euler + distillation | 1 | ~14ms | **0.858~0.861** | 蒸馏 1步 |

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

1. **方法**: 首次将渐进蒸馏应用于检测 RF，实现 1步高效推理
2. **理论**: 命题 3-4 论证 RF 轨迹直化性与蒸馏可行性，蒸馏-检测梯度对齐性
3. **应用**: 临床核型分析推理加速 3.9×，mAP 保持 ≥0.95×teacher

---

## 7. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 1步质量差距大 | 中 | 高 | 可退至 2步蒸馏，仍比 4步快 2× |
| 蒸馏训练不稳定 | 低 | 中 | 蒸馏目标与检测目标对齐（命题 4），梯度不冲突 |
| 教师过拟合 | 低 | 低 | A4 使用 save_best 机制，取验证最优 checkpoint |
| 训练显存增加 | 中 | 低 | 教师无梯度，可用 inference_mode；学生前向共享特征 |
