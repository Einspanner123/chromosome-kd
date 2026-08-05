# 方向 H：Flow Matching 检测 (Conditional Flow Matching for Detection)

> **状态：已证伪 (2026-07-12, 最终验证)**
>
> **实验结果**：24obj 数据集, use_cfm=True, predict_velocity=True, velocity_loss_weight=1.0,
> best mAP=**0.823** (epoch 95) vs 基线 0.856 → **Delta = -0.033** (性能显著下降)
>
> **SwanLab**：https://swanlab.cn/@einspanner/ldmdet-frontier-directions/runs/sh5750nr
>
> **失败原因**：
> 1. **级联架构与速度预测根本性不兼容**：Head 1-5 的输入 ≈ x_0 (去噪后的框), 丢失 x_noise 信息, 无法计算速度目标 v = x_noise - x_start
> 2. **速度损失收敛到 Var(x_noise) = 4**：因 Head 1-5 输入已不含噪声, 速度 MSE 收敛到噪声方差, 注入梯度噪声而非有效监督
> 3. **FlowDet 论文实际预测端点 x̂₁ 而非速度**：用标准检测损失 (L1/GIoU) 而非速度 MSE, 我们误读了论文的参数化方式
> 4. **基线的 x_0 预测 (via delta 回归) 本质是相对速度预测**：通过 apply_deltas 将回归头输出转换为框坐标, 是级联架构的正确参数化, 无需显式速度预测
>
> **实验配置**：[h_cfm_velocity_24obj.py](../../experiments/configs/ldmdet/directions/frontier_directions/h_cfm_velocity_24obj.py) (已删除)
>
> **代码清理 (2026-07-12)**：CFM 相关代码已从 head.py 中完全删除, 包括 use_cfm、velocity_loss_weight、predict_velocity 等参数及 velocity→x0→xyxy 转换逻辑。single_head.py 的 predict_velocity 参数和 criterion.py 的 use_cfm_weighting 参数保留但默认 False (死代码, 不影响功能)。相关测试文件 (test_cfm_fix.py, test_direction_h_cfm.py) 已删除。
>
> ---
>
> **目标**：用 Conditional Flow Matching (CFM) 代替当前 Rectified Flow (RF) 的"扩散思维", 实现更直更短的传输路径, 支持单步采样。
>
> **理论依据**：
> - FlowDet: Baty et al., "FlowDet: Unifying Object Detection and Generative Transport Flows" (arxiv 2512.16771, 2025-12)
> - Conditional Flow Matching: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
> - Rectified Flow: Liu et al., "Flow Straight and Fast" (ICLR 2023) — 我们当前所用
>
> **当前代码位置**：
> - [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py)
> - [ldmdet/core/single_head.py](../../ldmdet/core/single_head.py) — `_predict` 方法
> - [ldmdet/core/head.py](../../ldmdet/core/head.py) — `loss` 方法

---

## 1. 背景与动机

### 1.1 当前 RF 的"扩散思维"问题

我们当前的 Rectified Flow 实际上是**披着 RF 外衣的扩散思维**:

```python
# 当前 single_head.py:199-206 — 模型预测 x_0
def _predict(self, fc_feature, bboxes, bs, num_boxes):
    class_logits = self.cls_head(fc_feature)
    pred_bboxes = self._predict_bboxes(fc_feature, bboxes)  # ← x_0_pred
    return class_logits, pred_bboxes, ...
```

```python
# 当前 rectified_flow.py:42-45 — 采样时从 x_0_pred 反推速度
def get_velocity(self, x_t, x_0_pred, t):
    return (x_t - x_0_pred) / torch.clamp(t, min=1e-5)
```

**问题**:
1. **训练目标错配**: 训练时模型直接预测 x_0, 但损失是 L1(x_0_pred, x_0_gt), 没有显式约束速度场
2. **推理多步累积误差**: 每步 x_0_pred 误差通过 `get_velocity` 放大, 累积到下一步 x_t
3. **暴露偏差 (Exposure Bias)**: 训练时 t 随机, 模型从未见过自己的多步采样轨迹

### 1.2 CFM 的核心改进

CFM 直接学习传输向量场 **v_θ(x_t, t)**, 而非 x_0:

```
当前 RF:  训练 L1(x_0_pred, x_0_gt),  推理 v = (x_t - x_0_pred) / t
CFM:      训练 MSE(v_θ, x_1 - x_0),    推理 v_θ 直接用
```

**优势**:
- 训练目标 = 推理用目标, **消除 exposure bias**
- 速度场是线性的 (RF 直线路径), 单步 Euler 即可近似: x_0 ≈ x_1 - v_θ(x_1, 1)
- 路径比 DDPM 弯曲随机路径更直更短, 同样步数质量更高

---

## 2. 子方向说明

### H1: 速度场直接预测 (核心改动)

**问题**: 当前模型预测 x_0, 间接反推速度, 训练-推理目标不一致。

**方案**: 修改 `_predict` 输出速度 v, 修改 loss 直接监督速度。

**实现**:
```python
# 修改 single_head.py
def _predict(self, fc_feature, bboxes, bs, num_boxes):
    class_logits = self.cls_head(fc_feature)
    # H1: 预测速度 v 而非 x_0
    velocity = self._predict_bboxes(fc_feature, bboxes)  # 现在输出 v
    return class_logits, velocity, ...

# 修改 head.py loss
def loss(self, features, img_metas, gt_bboxes, gt_labels):
    ...
    all_cls_logits, all_velocity, all_curr_proposals = self(features, curr_bboxes, t_input)
    # H1: 速度监督
    # velocity_target = x_noise - x_start  (RF 已计算, rectified_flow.py:39)
    loss_v = F.mse_loss(all_velocity, velocity_target)
    # 可选: 同时保留 x_0 监督作为辅助
    x0_pred = x_t - t * all_velocity  # 从速度反推 x_0
    loss_bbox = L1(x0_pred, x_start)
```

**复杂度**: 低 — 代码改动集中在 `_predict` 和 loss, 不动 backbone/neck。

**预期收益**: mAP +1-2%, 推理 2 步即可 (当前 4 步)。

### H2: Optimal Transport 配对 (OT-CFM)

**问题**: 当前 GHSS 耦合策略在训练时随机配对 (noise, gt), 路径非最优。

**方案**: 用 mini-batch OT 找最优配对, 使路径更直。

**实现**:
```python
# 训练前对每个 batch 做 OT
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

def ot_couple(noise, gt):
    cost = torch.cdist(noise, gt)  # (N, M)
    geom = pointcloud.PointCloud(cost)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    plan = out.matrix  # 软分配矩阵
    # 取 argmax 作为硬配对
    return plan.argmax(dim=1)
```

**复杂度**: 中 — OT 求解每 batch ~10ms, 训练耗时 +5%。

**预期收益**: 路径更直, 单步采样质量提升, mAP +0.5-1%。

### H3: 单步采样蒸馏

**问题**: 即使 CFM, 多步采样仍有累积误差。

**方案**: 训练多步 CFM 教师模型, 蒸馏出单步学生模型。

**实现**:
```python
# 教师模型 (多步, 高质量)
teacher_pred = multi_step_cfm_sample(x_noise, steps=4)

# 学生模型 (单步)
student_pred = single_step_cfm(x_noise, t=1)
loss_distill = F.mse_loss(student_pred, teacher_pred.detach())
loss_total = loss_distill + α * loss_supervised
```

**复杂度**: 高 — 需先训练教师, 再蒸馏。

**预期收益**: 推理速度 4x, mAP 持平或略降 (但单步精度上限高于当前 RF 单步)。

---

## 3. 与当前架构的集成

### 3.1 文件改动清单

```
ldmdet/
├── diffusion/
│   ├── rectified_flow.py     # 修改: 新增 CFM 类 (或重构 RF)
│   └── flow_matching.py      # 新增: CFM 实现
├── core/
│   ├── single_head.py        # 修改: _predict 输出 v (H1)
│   └── head.py               # 修改: loss 监督 v (H1)
├── criterion/
│   └── velocity_loss.py      # 新增: 速度 MSE 损失
└── tests/
    └── test_direction_h.py   # 新增: 单元测试

experiments/configs/ldmdet/
└── direction_h_flow_matching.py  # 新增: 配置
```

### 3.2 配置示例

```python
_base_ = ['./rf_heun_adaln.py']

model = dict(
    bbox_head=dict(
        diffusion_type='cfm',            # H1: 切换到 CFM
        velocity_prediction=True,        # H1: 预测速度而非 x_0
        coupling_type='ot',              # H2: OT 配对
        sampling_steps=2,                # H1: 单步或 2 步采样
    ),
)
```

### 3.3 与现有方向的兼容性

- **方向 A-G**: 完全兼容 (CFM 只改扩散过程, 不动 backbone/neck/head 结构)
- **方向 D (BoxRefine)**: 兼容, 可叠加在 CFM 采样之后
- **方向 F (StructuredPrior)**: 兼容, CFM 的初始分布可任意

**推荐组合**: H1 (CFM 核心) + D (BoxRefine) + G (LAMFPN) — 分别改进扩散、精化、特征。

---

## 4. 测试计划

### 4.1 单元测试

```python
class TestFlowMatching:
    def test_velocity_target_correct(self):
        """速度目标应为 x_1 - x_0"""
    def test_single_step_approximation(self):
        """单步采样: x_0 ≈ x_1 - v_θ(x_1, 1)"""
    def test_training_inference_consistency(self):
        """训练和推理用同一个 v_θ, 无 exposure bias"""

class TestOTCoupling:
    def test_ot_pairs_minimize_cost(self):
        """OT 配对应最小化总传输成本"""
    def test_ot_preserves_count(self):
        """配对后样本数不变"""
```

### 4.2 集成测试

```python
def test_direction_h_integration():
    """端到端: CFM 训练 + 单步采样"""
    cfg = Config.fromfile('experiments/configs/ldmdet/direction_h_flow_matching.py')
    model = MODELS.build(cfg.model)
    # 验证训练 loss 含 velocity MSE
    losses = model.loss(...)
    assert 'loss_velocity' in losses
    # 验证单步采样
    model.eval()
    results = model.predict(...)
    assert len(results) > 0
```

---

## 5. 实验计划

### 5.1 主实验

| 实验 | 配置 | 预期 |
|------|------|------|
| H1-baseline | RF 改 CFM, 2 步采样 | mAP +1-2% |
| H1-1step | CFM 单步采样 | mAP 持平, 推理 4x |
| H2-OT | H1 + OT 配对 | mAP +0.5-1% |
| H3-distill | H1 蒸馏单步 | 单步 mAP 接近多步 |

### 5.2 消融

| 消融 | 验证 |
|------|------|
| 预测 x_0 vs 预测 v | H1 核心改动贡献 |
| 采样步数 1/2/4/8 | CFM 步数-质量曲线 |
| OT vs GHSS vs random | H2 OT 贡献 |

### 5.3 训练命令

```bash
# H1 CFM
python experiments/runners/train.py experiments/configs/ldmdet/direction_h_flow_matching.py \
    --work-dir work_dirs/direction_h_flow_matching --seed 42 --gpu-id 0
```

---

## 6. 风险与缓解

### 6.1 速度场学习不稳定

**风险**: 速度 v 的尺度比 x_0 大 (v = x_1 - x_0 ∈ [-2, 2] vs x_0 ∈ [-1, 1]), 可能梯度爆炸。

**缓解**:
- 速度 MSE 损失加权重衰减
- 用 SNR 加权 (已有 [ldmdet/criterion/snr_weight.py](../../ldmdet/criterion/snr_weight.py))
- warmup: 前 5 epoch 混合 x_0 监督和 v 监督

### 6.2 单步采样精度不足

**风险**: 单步 Euler 近似误差大。

**缓解**:
- H1 先用 2 步, H3 再蒸馏到单步
- 保留 BoxRefine (方向 D) 作为后精化

---

## 7. 成功指标

| 指标 | baseline (RF 4步) | 目标 (CFM 2步) | 目标 (CFM 1步蒸馏) |
|------|------------------|---------------|-------------------|
| mAP | 0.858 | +0.015 | +0.010 |
| mAP_90 | 0.483 | +0.02 | +0.015 |
| 推理时间 | 1.0x | 2x 加速 | 4x 加速 |
| 训练耗时 | 1.0x | 1.0x | 1.5x (含蒸馏) |

---

## 8. 实现路线图

1. **Phase 1 (2 周)**: H1 — 修改 `_predict` 和 loss, 跑通 CFM 训练
2. **Phase 2 (1 周)**: H2 — 加 OT 配对, 对比 GHSS
3. **Phase 3 (2 周)**: H3 — 蒸馏单步模型

---

## 9. 参考文献

- FlowDet: https://arxiv.org/abs/2512.16771
- Conditional Flow Matching: https://arxiv.org/abs/2302.00482
- Rectified Flow: https://arxiv.org/abs/2209.03003
- OT-CFM: https://arxiv.org/abs/2302.00482
