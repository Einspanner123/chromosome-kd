# ReFlow (Standard MSE) 与 Head Distillation 实现方案

> **创建日期**: 2026-07-23
> **最近修订**: 2026-07-23 (v2 — 整合 subagent 评价 + 插桩设计)
> **状态**: 方案设计完成, 待实施
> **关联**: [TODO_DIRECTIONS.md §八 ReFlow](../TODO_DIRECTIONS.md), [TODO_DIRECTIONS.md §九 Head Distillation](../TODO_DIRECTIONS.md)
> **可行性调研**: 基于代码深读 (rectified_flow.py / head.py / coupling / criterion) + 历史教训 (已证伪 velocity-loss ReFlow, N_cascade e2e, PD-RF)
> **subagent 评价结论**:
> - ReFlow: 原方案 NO-GO (criterion target 设计有根本缺陷), v2 改为混合 target (cls=GT, box=x_0^pred)
> - Head Distillation: CONDITIONAL GO, v2 整合 6 项修正

---

## 一、ReFlow (Standard MSE 版) 实现方案 (v2)

### 1.1 核心机制

用已训练 A4 模型 (mAP=0.863) 对训练集推理, 生成新 coupling `(x_0^pred, x_1^noise)` 替代原始 `(x_0^GT, x_1^noise)`, 再用**标准检测损失** (无 velocity loss) 训练 2-RF, 拉直轨迹。

- 触发条件: η_str 实测 3.39~8.35 (远超阈值 0.1), 轨迹显著非直线
- 与已证伪版本区别: 不加 velocity loss → 消除梯度冲突 (旧版 cos=−0.104)

### 1.2 criterion target 设计 (v2 关键修正)

> ⚠ **v1 方案被 subagent 否决**: v1 "target=GT, x_t 端点改为 x_0^pred" 在 RF 理论上无法实现拉直。若轨迹端点为 x_0^pred 而 criterion target 仍为 GT, 模型学习的是 "从基于 x_0^pred 的轨迹映射回 GT", 轨迹未被拉直, 退化为带噪声标签的普通训练。

**v2 混合 target 设计** (兼顾 RF 拉直 + 检测任务结构):

```
_build_training_targets (reflow 模式):
  gt_diffusion = (norm_gt * 2 - 1) * snr_scale       # GT 扩散空间 (用于 SimOTA 匹配)
  noise  = 预存 x_1^noise (从 coupling 文件加载, 固定)
  x_start = 预存 x_0^pred (A4 推理结果, 轨迹端点)
  matched_idx = 预存 OT 匹配 (沿用 A4 推理时的分配)
  x_noisy, x_noise = rf.q_sample(x_start, noise, t)  # x_t = (1-t)·x_0^pred + t·noise

loss (混合 target):
  cls_target = GT              # 分类: SimOTA 用 GT 分配正负样本 (x_0^pred cls 不可靠)
  box_target = x_0^pred        # 回归: 正样本的 box target 改为 x_0^pred (拉直目标)
  L = Focal(cls_pred, cls_target=GT) + L1+GIoU(box_pred, box_target=x_0^pred)
```

**理论依据**:
- **分类用 GT**: SimOTA 需要真实类别标签分配正负样本, x_0^pred 的分类分数含误检, 不能作为 cls target
- **回归用 x_0^pred**: RF 拉直的核心是让模型从 x_t 回归到轨迹端点 x_0^pred。box 回归 target 改为 x_0^pred 后, 模型学习 (x_0^pred, x_1^noise) 间的直线映射, 实现 2-Rectification
- **正样本筛选仍用 GT**: SimOTA 的 dynamic-k + IoU 匹配仍基于 GT (哪些 proposal 对应哪个 GT), 仅 box 回归 target 替换

### 1.3 EMA Teacher 防漂移 (v2 新增)

> subagent 建议: 引入 EMA teacher 缓解 circular dependency (模型用自预测训练的 confirmation bias)

```
EMA teacher θ_teacher = α·θ_teacher + (1-α)·θ_student   (α=0.999)
每 N epoch 用 EMA teacher 重新生成 coupling (而非固定用 A4)
```

- **Phase 1 (ep 0-10)**: 用 A4 原始 coupling (固定)
- **Phase 2 (ep 10+)**: 每 5 epoch 用 EMA teacher 重新生成 coupling, 监控 `x0_drift_from_a4`
- **早停信号**: 若 `self_pred_amplification` (EMA teacher 预测 vs A4 预测的偏差) 单调增长, 触发早停

### 1.4 Per-dim Reflow (v2 新增, 可选)

> subagent 建议: 仅对 cx/cy 做 reflow (h 维度曲率小, w 维度差距小, 见方向 A 诊断)

- **维度选择依据**: 方向 A Phase 1 诊断显示 h 维度曲率显著小于 cx/cy (3-5×), w 维度差距较小 (1.5-2×)
- **实现**: box_target 的 cx/cy 用 x_0^pred, w/h 仍用 GT
- **风险**: 半拉直半未拉直可能引入新的轨迹不一致, 作为 v2 可选项 (config 开关 `reflow_dims='cxcy'`)

### 1.5 Phase 1: 生成新 Coupling

**新建脚本** `experiments/runners/generate_reflow_couplings.py`:
```python
# 对每张训练图:
#   1. 固定 seed 生成 x_raw_noise = randn(1, 500, 4)
#   2. A4 推理 (DPM-Solver++ 4步), 关闭 box_renewal (保持 proposal 对应)
#   3. 保存 (x_raw_noise, x0_pred_raw, image_id, gt_bboxes, matched_idx)
# 注意: matched_idx 需在推理时用 OT 重新计算 (与训练时 _couple_single_image 一致)
# 输出: work_dirs/reflow_couplings/train_couplings.pt (K 组轮换保留 stochastic 性)
```

- **box_renewal 必须关闭** (否则替换低置信 proposal, 破坏对应关系, 参照 PD-RF 教训)
- **K 组轮换**: 生成 K=5 组 couplings (不同 seed), 训练时按 epoch 轮换, 保留 OT 多样性
- **推理时间**: ~120ms/图 × 2000 图 × 5 组 ≈ 20 分钟
- **v2 简化**: v1 的 matched_idx 冗余存储已删除 — 训练时从 coupling 文件直接加载 (x_start, noise), matched_idx 在线重算 (基于 GT, 与 criterion target 一致)

### 1.6 Phase 2: 代码改动

| 文件 | 改动 |
|------|------|
| `ldmdet/diffusion/rectified_flow.py` | `RectifiedFlow.__init__` 新增 `use_reflow_coupling: bool=False`, `reflow_dims: str='all'` |
| `ldmdet/core/head.py` | `__init__` 新增 `use_reflow_coupling`, `reflow_coupling_path`, `ema_teacher`; 预加载 coupling 到内存 (~16MB) |
| `ldmdet/core/head.py` | `_build_training_targets`: reflow 模式从预存 coupling 加载 x_start/noise (替代在线 OT) |
| `ldmdet/core/head.py` | `_couple_single_image`: reflow 模式跳过 OT, 直接返回预存 (x_start, matched_idx 在线重算) |
| `ldmdet/criterion/criterion.py` | `deep_supervision`: 新增 `box_target_mode='gt'|'x0_pred'`; reflow 模式下正样本 box target 替换为 x_0^pred |
| `ldmdet/core/head.py` | EMA teacher hook: 每 step 更新 θ_teacher, 每 5ep 触发 coupling 重生成 |
| `experiments/configs/.../reflow_standard_24obj.py` | 新建: 基于 a4, `use_reflow_coupling=True`, `reflow_dims='cxcy'`, lr=1e-5, ≤50ep |

### 1.7 风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| **Circular Dependency** (模型用自预测训练, confirmation bias) | **高** | EMA teacher (§1.3); 限制 ≤50ep; ep5/10/20 检查 mAP, 下降立即停; 监控 `self_pred_amplification` |
| **混合 target 破坏 SimOTA 一致性** (cls=GT, box=x_0^pred) | **中** | 仅正样本 box target 替换; 负样本不受影响; 冒烟测试 10ep 观察 loss 收敛 |
| box_renewal 训推不一致 (生成关, 训练开) | 中 | Phase 1 生成时也尝试开 box_renewal 版本对照; 或训练时也关 box_renewal |
| OT stochastic 性丢失 | 中 | K=5 组轮换 |
| 强基线饱和 (η_str 拉直≠mAP 提升) | 中 | 监控 η_str 变化 + mAP, 双指标判断 |
| PD-RF 前车之鉴 (自预测监督崩塌) | 中 | 冒烟测试 (1-seed 30ep), ep5 内崩塌则停止 |
| per-dim reflow 轨迹不一致 (若启用) | 中 | 默认 `reflow_dims='all'`, per-dim 作为消融对照 |

### 1.8 成功判据

1. mAP ≥ A4 baseline (0.863)
2. η_str 显著下降 (8.35 → < 3.0)
3. 1-2 步推理 mAP 接近 4 步 (减少 NFE)
4. **方向价值判据** (不仅看 mAP): `self_pred_amplification` 未单调增长 + `per_class_improved_count ≥ 12/24`

---

## 二、Head Distillation 实现方案 (v2)

### 2.1 核心机制

Teacher (H=6, A4 冻结) 监督 Student (H=3) 的中间特征, headwise 蒸馏:
```
L_total = L_det(student) + λ · L_distill
L_distill = (1/3) Σ_k MSE(student_fc_feature_k, teacher_fc_feature_{map(k)}.detach())
```

- Student 3 head 用 Teacher 对应 head 初始化 (非随机)
- S1 已证 H=3 S=8 持平 baseline → H=3 可行, N_cascade e2e 失败因随机初始化

### 2.2 v2 六项修正 (subagent 评价)

| # | v1 方案 | v2 修正 | 理由 |
|---|---------|---------|------|
| 1 | 未冻结 backbone | **冻结 Student backbone** (load A4, requires_grad=False) | 蒸馏聚焦 head, 减少 param 量, 避免 backbone 漂移破坏 Teacher 特征对齐 |
| 2 | 蒸馏 pred_bboxes (最终 box 输出) | **蒸馏 fc_feature** (box head 前的中间特征) | 特征比坐标更丰富; 避免 bbox 坐标空间 mismatch; 梯度更稳定 |
| 3 | Student head 0/1/2 ← Teacher head 1/3/5 | **映射 {0→0, 1→2/3, 2→5}** | Student head 0 (输入端) ↔ Teacher head 0 (输入对齐); Student head 2 (主输出) ↔ Teacher head 5 (main 对齐) |
| 4 | λ=0.1 起步 | **λ=0.05 起步** | 更保守, 降低初期梯度冲突风险 (PD-RF 教训) |
| 5 | 未指定 coupling_mode | **coupling_mode='argmax'** (确定性) | Student/Teacher 必须看到相同 matched_idx, argmax 保证 proposal 对齐 (multinomial 会破坏) |
| 6 | §2.5 deep_supervision "等权重" | **修正: distill 仅作用于 main head, aux head 不蒸馏** | 避免蒸馏 + aux loss 叠加导致 main head 梯度过大; deep_supervision aux 权重可降至 0.5 |

### 2.3 数据流改动

**当前 (H=6 训练)**:
```
forward: for i in 6 heads: x = head_i(x); collect inter_cls/inter_bbox
loss: main(head_6) + aux(head_1..5), each SimOTA+Focal+L1+GIoU
```

**Head Distillation (H=3 训练, v2)**:
```
# 共享噪声 (Student/Teacher 同一 x_raw, t, matched_idx, coupling_mode='argmax')
with no_grad:
    teacher_inter_cls, teacher_inter_bbox, teacher_fc_features = teacher.forward(features, x_shared, t_shared)  # [6,...]
student_inter_cls, student_inter_bbox, student_fc_features = self.forward(features, x_shared, t_shared)  # [3,...]
L_det = criterion(student_outputs, GT)  # main(head_3) + aux(head_1..2, weight=0.5)
# headwise feature 蒸馏 (映射: student_k → teacher_map[k])
L_distill = (1/3) Σ_k MSE(student_fc_feat[k], teacher_fc_feat[map(k)].detach())
L_total = L_det + λ · L_distill   # λ=0.05 起步
```

### 2.4 方案 A (Headwise Feature Matching) 实现

**Teacher 中间输出已现成**: `forward()` 已收集 6 head 的 `inter_cls_logits`/`inter_pred_bboxes`。v2 额外收集 `fc_feature` (box head 前的 256-d 特征, 现有代码在 `_forward_head` 内部, 需暴露)。

**Head 映射** (v2):
```python
# Student (H=3) → Teacher (H=6)
distill_head_map = {0: 0, 1: 2, 2: 5}  # 0-indexed
# Student head 0 ↔ Teacher head 0 (输入对齐, 都是第一个 head)
# Student head 1 ↔ Teacher head 2 (中间, 进度对齐 ~1/2 处)
# Student head 2 ↔ Teacher head 5 (main 对齐, 都是最后一个 head = 主输出)
```

**Student 配置** (v2):
```python
_base_ = ['./a4_dpm_pp_24obj.py']
model = dict(
    bbox_head=dict(
        num_heads=3,
        use_distillation=True,
        distill_lambda=0.05,           # v2: 0.05 起步 (v1 为 0.3)
        distill_match_mode='feature',  # v2: feature 蒸馏 (v1 为 'bbox')
        distill_head_map={0: 0, 1: 2, 2: 5},  # v2: 输入/main 对齐
        coupling_mode='argmax',        # v2: 确定性 coupling
        freeze_backbone=True,          # v2: 冻结 backbone
        deep_supervision_aux_weight=0.5,  # v2: aux 权重降低
    ),
)
teacher_config = 'a4_dpm_pp_24obj.py'
teacher_checkpoint = 'work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth'
```

**Student head 初始化** (v2): 加载 A4 checkpoint, Student head 0/1/2 ← Teacher head 0/2/5 (与 distill_head_map 一致)。

### 2.5 代码改动

| 文件 | 改动 |
|------|------|
| `ldmdet/core/head.py` | `__init__` 新增 `teacher_model`, `distill_lambda`, `distill_match_mode`, `distill_head_map`, `freeze_backbone`, `deep_supervision_aux_weight`; 新增 `loss_with_distillation` 方法 |
| `ldmdet/core/head.py` | `_forward_head`: 暴露 `fc_feature` (box head 前特征) 供蒸馏使用 |
| `ldmdet/core/head.py` | `freeze_backbone=True` 时: backbone + neck `requires_grad_(False)`, 不进 optimizer |
| `experiments/mmdet_bridge/detector.py` | `LDMDetDetector.__init__` 新增 `teacher_config/teacher_ckpt`; 构建 teacher 并 `__setattr__` 注入 (避免参数进 optimizer); `loss()` 分支调用蒸馏版 |
| `ldmdet/criterion/losses.py` | 新增 `FeatureDistillLoss` (MSE on fc_feature, 可选 normalize) |
| `experiments/configs/.../h3_distill_24obj.py` | 新建: num_heads=3, use_distillation=True (v2 全部参数) |
| `tests/unit/test_head_distill.py` | 单元测试 (teacher 冻结/梯度流/噪声共享/backbone 冻结/feature 对齐) |

**cascade_detach 天然规避**: headwise 蒸馏每个 head 独立有蒸馏损失, 不依赖跨 head 梯度流, 不受 `cascade_detach=True` 影响 (PD-RF 时间维度蒸馏的痛点)。

### 2.6 deep_supervision 处理 (v2 修正)

> **v1 矛盾**: v1 §2.5 说 "Student 保持 deep_supervision 不变 (1 main + 2 aux, 等权重)", 但 deep_supervision 的 aux loss 与蒸馏 loss 叠加在 main head 上会导致 main head 梯度过大。

**v2 方案**:
- Student deep_supervision: main(head_3) + aux(head_1, head_2), **aux 权重降至 0.5** (v1 为 1.0)
- 蒸馏 loss **仅作用于 main head (head_2, 0-indexed)**: `L_distill = MSE(student_fc_feat[2], teacher_fc_feat[5])` + 可选中间 head
- aux head (head_0, head_1) 不加蒸馏 loss, 仅靠检测 loss + head 初始化传递 Teacher 知识

### 2.7 风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| 蒸馏梯度与检测梯度冲突 (PD-RF 教训) | **中** | 监控 `distill/grad_alignment` (P0); λ 从 0.05 保守起步; 必要时降 λ |
| Teacher 预测非完美监督 | 中 | Teacher=A4 (mAP=0.863), 软特征仍优于随机初始化 |
| λ 调参困难 | 低 | 并行 [0.05, 0.1, 0.2]; 监控 `loss_ratio` 自动调参 |
| Student capacity 不足 | 低 | 不减每 head 参数量, 只减数量 |
| backbone 冻结导致特征不适应 H=3 | 低 | S1 已证 H=3 S=8 持平, backbone 特征足够 |
| feature 蒸馏维度 mismatch | 低 | fc_feature 均为 256-d (Student/Teacher 同架构), 无需投影层 |

### 2.8 成功判据

1. mAP ≥ 0.84 (允许比 A4 0.863 小幅下降)
2. 比 S1 直接训练 H=3 有显著提升
3. NFE 24 → 12 (2× 加速)
4. **方向价值判据**: `grad_alignment > 0` (无冲突) + `per_class_improved_count ≥ 10/24`

---

## 三、优先级与实施顺序

| 方向 | 可行性 | 内在风险 | 建议 |
|------|--------|---------|------|
| Head Distillation | **高** (Teacher 输出现成, S1 证 H=3 可行, headwise 规避 detach, v2 6 项修正后风险可控) | 中 (梯度冲突可监控) | **优先实施** |
| ReFlow | 中偏高 (η_str 强支撑, 消除梯度冲突, v2 混合 target 修复理论缺陷) | **高** (circular dependency, EMA teacher 增加复杂度) | 其次, 冒烟测试先行 |

**联合收益**: ReFlow (4→2步) + Head Distillation (6→3 head) = 24 NFE → 6 NFE (4× 加速)

---

## 四、插桩监控指标设计 (v2 详细版)

> **设计原则** (方向价值判断标准 2026-07-23): 不仅凭 mAP 判定方向成败, 通过插桩指标综合评估训练健康度、梯度动态、per-class 改善分布、收敛行为。
>
> **优先级**: P0 = 必须实现 (决定方向去留); P1 = 重要 (辅助诊断); P2 = 可选 (锦上添花)

### 4.1 ReFlow 专用指标

#### P0 (必须)

| 指标 | 含义 | 实现 | 判读 |
|------|------|------|------|
| `reflow/x0_drift_from_a4` | EMA teacher 预测 vs A4 原始预测的 L2 距离 | 每 5ep 用 EMA teacher 推理训练集子集 (200 图), 对比 A4 coupling 文件 | 单调增长 = confirmation bias 加剧, 触发早停 |
| `reflow/self_pred_amplification` | EMA teacher 预测与 A4 预测偏差的放大率 | `‖x0_ema - x0_a4‖ / ‖x0_a4 - GT‖` | >1 = 误差放大 (circular dependency), 立即停 |
| `reflow/mAP_per_epoch_slope` | 最近 5 epoch mAP 线性回归斜率 | 从 SwanLab/coco/bbox_mAP 取最近 5 点最小二乘 | 斜率 < -0.001 = 退化趋势 |
| `reflow/per_class_improved_count` | vs A4 baseline per-class AP 改善的类别数 | 每 5ep eval per-class AP, 对比 A4 baseline | < 8/24 改善 = 方向无价值 |
| `reflow/per_class_degraded_count` | vs A4 baseline per-class AP 退化 (>0.005) 的类别数 | 同上 | > 8/24 退化 = 方向有害 |

#### P1 (重要)

| 指标 | 含义 | 实现 | 判读 |
|------|------|------|------|
| `reflow/eta_str_step{0,1,2}` | reflow 前后 η_str 对比 | 推理时 record_inference_eta_str (已有 probe) | 下降 = 拉直成功 (目标 < 3.0) |
| `reflow/coupling_diversity` | K 组 coupling 间方差 | 训练开始时计算 K 组 x_0^pred 的方差 | 接近 0 = stochastic 性丢失 |
| `reflow/train_loss_gap` | reflow loss vs A4 同期 loss | 从 SwanLab 取 A4 同 epoch loss 对比 | 偏离 > 50% = 分布偏移 |
| `reflow/box_target_gap` | x_0^pred 与 GT 的 box 差距 (正样本) | 训练时 L1(x_0^pred, GT) on positive samples | 跟踪混合 target 的合理性 |

#### P2 (可选)

| 指标 | 含义 |
|------|------|
| `reflow/per_dim_eta_str` | 各维度 (cx/cy/w/h) 的 η_str, 验证 per-dim reflow 效果 |
| `reflow/coupling_regen_count` | EMA teacher 触发 coupling 重新生成的次数 |

### 4.2 Head Distillation 专用指标

#### P0 (必须)

| 指标 | 含义 | 实现 | 判读 |
|------|------|------|------|
| `distill/grad_alignment_global` | 蒸馏梯度与检测梯度的余弦相似度 (全局) | `torch.autograd.grad` 两次反传 (见 §4.4) | < 0 = 冲突 (PD-RF 教训 cos=−0.104); 持续 < 0 触发降 λ |
| `distill/grad_alignment_head{k}` | 逐 head 的梯度对齐 | 同上, 分 head 统计 | 定位冲突来源 head |
| `distill/head{k}_feat_gap` | Student head_k vs Teacher head_{map(k)} 的 fc_feature MSE | 训练时前向计算 | 下降 = Student 逼近 Teacher |
| `distill/loss_ratio` | L_distill / L_det 比值 | 训练时记录 | 监控蒸馏 loss 占比; λ 调参依据 |
| `distill/per_class_improved_count` | vs A4 baseline per-class AP 改善的类别数 | 每 5ep eval per-class AP | < 10/24 = 方向价值不足 |

#### P1 (重要)

| 指标 | 含义 | 实现 | 判读 |
|------|------|------|------|
| `distill/teacher_feat_norm` | Teacher 中间 head fc_feature 范数 | 前向时记录 | 监督信号强度, 归零 = Teacher 退化 |
| `distill/student_feat_norm` | Student fc_feature 范数 | 同上 | 与 Teacher 对比, 监控 Student 学习进度 |
| `distill/main_head_mAP_gap` | Student main head mAP vs Teacher main head mAP | 推理时 eval 单 head | 收敛监控 |
| `distill/backbone_grad_norm` | backbone 梯度范数 (应为 0 若冻结) | optimizer step 后检查 | 确认冻结生效 |

#### P2 (可选)

| 指标 | 含义 |
|------|------|
| `distill/feat_cosine_sim` | Student/Teacher fc_feature 余弦相似度 (方向对齐) |
| `distill/lambda_adapt_log` | λ 自适应调整日志 (若实现自动调参) |

### 4.3 共用指标 (已有 probe + 新增)

| 指标 | 含义 | 优先级 | 来源 |
|------|------|--------|------|
| `train/loss/*`, `train/t` | 训练 loss 分解 + 时间步 | 已有 | probe |
| `cascade/head{i}/*` | 逐 head 中间统计 | 已有 | probe |
| `coupling/*` | OT coupling 统计 | 已有 | probe |
| `rf/*` | RF 采样统计 | 已有 | probe |
| `common/mAP_std_last10` | 最近 10 epoch mAP 标准差 | P0 | 新增, 从 eval 结果计算 |
| `common/mAP_collapse_flag` | mAP 突降 > 0.05 触发 | P0 | 新增, eval 后检查 |
| `common/grad_norm` | 全局梯度范数 | P1 | 已有 (clip_grad) |
| `common/per_class_ap_full` | 24 类完整 per-class AP | P1 | eval 时记录 |

### 4.4 梯度对齐监控实现 (关键技术)

> ⚠ **不能用 hook** (PD-RF 教训: hook 捕获的梯度被 clip_grad 修改, 无法反映真实蒸馏梯度方向)
>
> **方案**: `torch.autograd.grad` 两次独立反传 (retain_graph=True)

```python
def compute_grad_alignment(student_outputs, teacher_features, GT, distill_loss_fn, det_loss_fn, lambda_distill):
    """计算蒸馏梯度与检测梯度的余弦相似度"""
    # 第一次反传: 仅检测 loss
    det_loss = det_loss_fn(student_outputs, GT)
    det_grads = torch.autograd.grad(det_loss, student_params, retain_graph=True)
    det_grad_flat = torch.cat([g.flatten() for g in det_grads])

    # 第二次反传: 仅蒸馏 loss
    distill_loss = distill_loss_fn(student_features, teacher_features)
    distill_grads = torch.autograd.grad(distill_loss, student_params, retain_graph=False)
    distill_grad_flat = torch.cat([g.flatten() for g in distill_grads])

    # 余弦相似度
    cos_sim = F.cosine_similarity(
        det_grad_flat.unsqueeze(0), distill_grad_flat.unsqueeze(0)
    ).item()
    return cos_sim, det_loss.item(), distill_loss.item()

# 调用频率: 仅 collect_step (每 50 step 一次), 避免训练减速
# student_params = [p for p in model.parameters() if p.requires_grad]
```

**实现注意**:
- `retain_graph=True` 保留计算图供第二次反传, 之后再 `retain_graph=False` 释放
- 仅在 collect_step 触发 (probe 已有机制), 避免每 step 反传两次拖慢训练
- `student_params` 排除 Teacher 参数 (Teacher no_grad) 和冻结的 backbone 参数
- 逐 head 版本: 对每个 head 的 fc_feature 单独计算蒸馏梯度, 与检测梯度对比

---

## 五、实施计划

### Phase 1: Head Distillation (优先)

| 步骤 | 内容 | 预计时间 |
|------|------|---------|
| 1.1 | 代码: head.py 暴露 fc_feature + distill 参数 + backbone 冻结 | 2h |
| 1.2 | 代码: detector.py teacher 注入 + loss 分支 | 1h |
| 1.3 | 代码: FeatureDistillLoss + grad_alignment 插桩 | 2h |
| 1.4 | 单元测试: teacher 冻结/梯度流/噪声共享/backbone 冻结 | 1h |
| 1.5 | 冒烟测试: 5ep 训练, 验证 grad_alignment > 0 + loss 收敛 | 2h (GPU) |
| 1.6 | 正式训练: 30ep + λ 网格 [0.05, 0.1, 0.2] | ~24h (GPU) |
| 1.7 | 评估 + 文档更新 | 2h |

### Phase 2: ReFlow (其次)

| 步骤 | 内容 | 预计时间 |
|------|------|---------|
| 2.1 | 脚本: generate_reflow_couplings.py (A4 推理 + K=5 组) | 3h |
| 2.2 | 代码: head.py reflow 模式 + EMA teacher + 混合 target | 4h |
| 2.3 | 代码: criterion.py box_target_mode 插桩 | 2h |
| 2.4 | 代码: self_pred_amplification + x0_drift 插桩 | 2h |
| 2.5 | 单元测试: coupling 加载/混合 target/EMA 更新 | 1h |
| 2.6 | 冒烟测试: 10ep, 验证 self_pred_amplification 不增长 | 3h (GPU) |
| 2.7 | 正式训练: 50ep + per-dim 消融 | ~40h (GPU) |
| 2.8 | 评估 + 文档更新 | 2h |

---

## 六、subagent 评价历史记录

### 6.1 ReFlow 评价 (v1 → v2)

- **v1 问题**: criterion target=GT, x_t 端点=x_0^pred → RF 理论上无法拉直, 退化为带噪声标签训练
- **v2 修正**: 混合 target (cls=GT, box=x_0^pred) + EMA teacher + per-dim 可选
- **残留风险**: 混合 target 的 SimOTA 一致性需冒烟测试验证

### 6.2 Head Distillation 评价 (v1 → v2)

- **v1 问题**: (1) backbone 未冻结 (2) 蒸馏 pred_bboxes 而非 feature (3) head 映射输入不对齐 (4) λ=0.3 过大 (5) coupling 不确定 (6) deep_supervision 矛盾
- **v2 修正**: 6 项全部修正 (见 §2.2)
- **残留风险**: 梯度冲突 (P0 grad_alignment 监控)

### 6.3 插桩设计评价

- **梯度对齐**: 必须用 `torch.autograd.grad` 两次反传, 不能用 hook (hook 捕获 clip 后梯度)
- **P0 指标**: 覆盖 confirmation bias (ReFlow) + 梯度冲突 (Distill) + per-class 改善 (通用) + mAP 崩塌 (通用)
- **调用频率**: 仅 collect_step, 避免训练减速
