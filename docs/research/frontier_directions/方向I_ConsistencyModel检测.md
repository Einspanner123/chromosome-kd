# 方向 I：Consistency Model 检测 (单步直接生成框)

> **目标**：用 Consistency Model (CM) 实现单步直接出框, 消除多步采样累积误差, 突破 mAP_90 = 0.483 的精度瓶颈。
>
> **理论依据**:
> - Consistency Models: Song et al., "Consistency Models" (ICML 2023)
> - Improved Consistency Training: Song & Dhariwal (ICML 2024)
> - Latent Consistency Models: Luo et al. (2023)
>
> **当前代码位置**:
> - [ldmdet/core/head.py](../../ldmdet/core/head.py) — `predict` 多步采样
> - [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py)

---

## 1. 背景与动机

### 1.1 多步采样的根本问题

当前 RF + Heun 采样需要 4 步, 每步误差累积:

```
x_1 (噪声) → x_{0.75} → x_{0.5} → x_{0.25} → x_0 (GT)
     ↓           ↓           ↓           ↓
   误差_1      误差_2      误差_3      误差_4
```

**累积效应**: 每步 x_0_pred 的误差通过 `get_velocity` 放大, 导致 mAP_90 仅 0.483。

**根因**: 训练时模型只看单个 t 的预测, 从未学过"多步轨迹的一致性"。

### 1.2 Consistency Model 的核心思想

CM 训练一个函数 **f_θ(x_t, t)** 满足**自一致性约束**:

```
f_θ(x_t, t) = f_θ(x_{t'}, t')    对同一 ODE 轨迹上的任意 t, t'
```

即: **同一条轨迹上的所有点, 映射到同一个输出**。

推理时**单步直接出框**:
```
x_0 = f_θ(x_noise, t=1)   # 一步到位
```

### 1.3 对染色体任务的优势

1. **4 维框信息量低**: 单步足够表达, 不需要多步生成
2. **消除累积误差**: 单步无累积, 直接攻 mAP_90
3. **训练目标对齐**: 一致性约束天然消除 exposure bias
4. **推理速度 4-8x**: 单步前向 vs 4 步采样

---

## 2. 子方向说明

### I1: Consistency Training (CT) — 从零训练

**问题**: 直接训练一致性模型, 不需要教师。

**方案**: 用一致性约束作为训练目标。

**实现**:
```python
class ConsistencyHead(nn.Module):
    """一致性检测头: f_θ(x_t, t) → x_0"""
    def __init__(self, ...):
        super().__init__()
        self.single_head = SingleDiffusionDetHead(...)
        # EMA 目标网络 (用于一致性约束)
        self.target_head = copy.deepcopy(self.single_head)
        for p in self.target_head.parameters():
            p.requires_grad_(False)

    def forward(self, features, x_t, t):
        # 学生: f_θ(x_t, t)
        x0_student = self.single_head(features, x_t, t)
        # 目标: f_θ'(x_{t+Δt}, t+Δt) — EMA 网络
        with torch.no_grad():
            x0_target = self.target_head(features, x_{t+Δt}, t+Δt)
        return x0_student, x0_target

# Loss
def consistency_loss(x0_student, x0_target, x0_gt):
    # 一致性: 学生和目标输出应相同
    loss_consist = F.mse_loss(x0_student, x0_target.detach())
    # 监督: 学生应接近 GT (可选, 防止退化)
    loss_supervise = F.l1_loss(x0_student, x0_gt)
    return loss_consist + λ * loss_supervise
```

**复杂度**: 中 — 需 EMA 网络和调度策略, 但不动 backbone。

**预期收益**: mAP_90 +5-10%, 推理 4x 加速。

### I2: Consistency Distillation (CD) — 从教师蒸馏

**问题**: CT 从零训练可能不稳定, 难以收敛。

**方案**: 用当前 RF 多步模型作为教师, 蒸馏单步学生。

**实现**:
```python
def consistency_distill_loss(student, teacher, x_t, t, features):
    # 教师多步: x_t → x_{t+Δt} (一步 ODE)
    with torch.no_grad():
        x0_teacher = teacher(features, x_t, t)  # 教师预测
        x_next = rf.step(x_t, x0_teacher, t, t + Δt)  # ODE 一步

    # 学生: f_θ(x_t, t) 和 f_θ(x_{next}, t+Δt) 应一致
    x0_student_t = student(features, x_t, t)
    x0_student_next = student(features, x_next, t + Δt)
    return F.mse_loss(x0_student_t, x0_student_next.detach())
```

**复杂度**: 中高 — 需先训练教师, 但 CD 比 CT 更稳定。

**预期收益**: 与 I1 相当, 但训练更稳定。

### I3: Latent Consistency (潜空间一致性)

**问题**: 直接在 4 维框空间做一致性, 表达能力受限。

**方案**: 在 RoI 特征潜空间做一致性, 框回归只是解码。

**实现**:
```python
# 一致性在特征空间, 不在框空间
def latent_consistency(features, x_t, t):
    roi_feat_t = roi_extractor(features, x_t)
    roi_feat_next = roi_extractor(features, x_next)

    z_t = encoder(roi_feat_t)       # 潜表示
    z_next = encoder(roi_feat_next)

    # 潜空间一致性
    loss = F.mse_loss(z_t, z_next.detach())

    # 框解码
    x0_pred = decoder(z_t)
    return loss + F.l1_loss(x0_pred, x0_gt)
```

**复杂度**: 高 — 需设计潜空间编码器。

**预期收益**: 表达能力更强, 但收益边际递减。

---

## 3. 训练策略细节

### 3.1 时间步调度

CM 训练的关键是**时间步调度**:

```python
class Schedule:
    def __init__(self, total_steps=10000):
        self.total_steps = total_steps

    def get_t(self, current_step):
        # 初期: t 大 (粗粒度一致性)
        # 后期: t 小 (细粒度一致性)
        progress = current_step / self.total_steps
        t_max = 1.0 - 0.5 * progress  # 从 1.0 降到 0.5
        t_min = 0.5 - 0.4 * progress  # 从 0.5 降到 0.1
        return random.uniform(t_min, t_max)

    def get_dt(self, current_step):
        # Δt 随训练减小 (初期大步, 后期小步)
        progress = current_step / self.total_steps
        return 0.1 * (1 - 0.5 * progress)
```

### 3.2 EMA 衰减调度

```python
def ema_decay_schedule(step, total_steps):
    # 初期 EMA 衰减快 (目标网络快速更新)
    # 后期 EMA 衰减慢 (目标网络稳定)
    return 0.9 + 0.09 * (step / total_steps)  # 0.9 → 0.99
```

### 3.3 学生-目标网络权重更新

```python
def update_target_ema(student, target, decay):
    with torch.no_grad():
        for s, t in zip(student.parameters(), target.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)
```

---

## 4. 与当前架构的集成

### 4.1 文件改动

```
ldmdet/
├── core/
│   ├── consistency_head.py    # 新增: ConsistencyHead (I1/I2)
│   └── head.py                # 修改: predict 支持单步模式
├── diffusion/
│   └── consistency.py         # 新增: 一致性约束 + 调度
├── criterion/
│   └── consistency_loss.py    # 新增: 一致性损失
└── tests/
    └── test_direction_i.py
```

### 4.2 配置示例

```python
_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        head_type='ConsistencyHead',  # I1
        consistency_mode='ct',        # 'ct' (I1) 或 'cd' (I2)
        ema_decay=0.99,
        dt_schedule='progressive',    # Δt 调度
        sampling_steps=1,             # 单步推理
        # I2 蒸馏
        teacher_ckpt='work_dirs/rf_heun_adaln/last_checkpoint',
    ),
)
```

### 4.3 与现有方向兼容性

- **方向 H (CFM)**: 可组合, CM 可蒸馏 CFM 教师
- **方向 D (BoxRefine)**: 兼容, 可在 CM 输出后精化
- **方向 G (LAMFPN)**: 完全兼容

---

## 5. 测试计划

### 5.1 单元测试

```python
class TestConsistencyHead:
    def test_self_consistency(self):
        """同轨迹两点输出应一致"""
        x_t, t = ..., 0.5
        x_next, t_next = ode_step(x_t, t), 0.5 + dt
        out_t = model(x_t, t)
        out_next = model(x_next, t_next)
        assert torch.allclose(out_t, out_next, atol=1e-3)

    def test_single_step_inference(self):
        """单步推理: f_θ(x_noise, 1) → x_0"""
        x_noise = torch.randn(N, 4)
        x0 = model(x_noise, t=1.0)
        assert x0.shape == (N, 4)

    def test_ema_update(self):
        """EMA 更新应使目标网络趋近学生"""
        update_target_ema(student, target, decay=0.9)
        assert torch.allclose(target.param, 0.9 * old + 0.1 * student)

class TestSchedule:
    def test_t_decrease_over_training(self):
        """t 应随训练递减"""
        t_early = schedule.get_t(step=100)
        t_late = schedule.get_t(step=9000)
        assert t_early > t_late
```

---

## 6. 实验计划

### 6.1 主实验

| 实验 | 模式 | 教师 | 预期 |
|------|------|------|------|
| I1-CT | 从零训练 | 无 | mAP_90 +5-8% |
| I2-CD | 蒸馏 | RF 4步 | mAP_90 +5-10% |
| I3-latent | 潜空间 | RF 4步 | mAP_90 +8-12% |

### 6.2 消融

| 消融 | 验证 |
|------|------|
| EMA decay 0.9/0.95/0.99 | 目标网络稳定性 |
| Δt 调度 fixed vs progressive | 调度策略影响 |
| 监督损失权重 λ | CT 中监督的作用 |
| 1步 vs 2步推理 | CM 步数-质量 |

### 6.3 训练命令

```bash
# I1 CT 从零训练
python experiments/runners/train.py experiments/configs/ldmdet/direction_i_consistency_ct.py \
    --work-dir work_dirs/direction_i_ct --seed 42

# I2 CD 蒸馏 (需先有教师)
python experiments/runners/train.py experiments/configs/ldmdet/direction_i_consistency_cd.py \
    --work-dir work_dirs/direction_i_cd --seed 42
```

---

## 7. 风险与缓解

### 7.1 训练不稳定

**风险**: CT 从零训练易发散, 一致性约束难收敛。

**缓解**:
- 优先用 I2 (CD) 蒸馏, 教师提供稳定监督
- warmup: 前 10 epoch 用纯监督 (无一致性), 逐步加一致性损失
- 梯度裁剪 + 学习率 warmup

### 7.2 单步精度上限

**风险**: 单步可能无法达到多步精度。

**缓解**:
- 保留 2 步模式作为 fallback
- 叠加方向 D (BoxRefine) 后精化
- I3 (latent) 提升表达力

### 7.3 EMA 网络显存

**风险**: 目标网络增加 ~50% 显存。

**缓解**: 目标网络用 FP16, 或共享 backbone (只 EMA head)。

---

## 8. 成功指标

| 指标 | baseline (RF 4步) | 目标 (CM 1步) |
|------|------------------|---------------|
| mAP | 0.858 | +0.01~0.02 |
| mAP_90 | 0.483 | +0.05~0.10 |
| 推理时间 | 1.0x | 4x 加速 |
| 训练耗时 | 1.0x | 1.5x (含 EMA) |

---

## 9. 实现路线图

1. **Phase 1 (3 周)**: I2-CD — 蒸馏, 用当前 RF 作教师
2. **Phase 2 (2 周)**: I1-CT — 从零训练, 对比 CD
3. **Phase 3 (3 周)**: I3-latent — 潜空间一致性 (若 I1/I2 收益明显)

---

## 10. 参考文献

- Consistency Models: https://arxiv.org/abs/2303.01469
- Improved Consistency Training: https://arxiv.org/abs/2310.14189
- Latent Consistency Models: https://arxiv.org/abs/2310.04378
