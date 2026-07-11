# 方向 U：CCBR — Cross-Cascade Box Renewal (跨级联框更新)

> **状态：方案设计阶段 (未实验)**
>
> **基线**：a3_full_sota, mAP=0.858 (RF + Heun 4步 + AdaLN-Zero + StochasticOT ε=5 + box_renewal, 24obj amodal)
>
> **核心思想**：增强现有 box_renewal 机制, 在 6 级级联 Head 之间建立跨级信息共享。当前 box_renewal 仅在扩散时间步之间生效且仅使用最后一级 Head 的置信度; CCBR 在级联 Head 之间引入 box_renewal, 并利用后面 Head (更精化) 的置信度信息指导前面 Head 的框更新策略。关键约束: **保持 `cascade_detach=True`**, 仅在推理时共享信息, 不引入训练不稳定。
>
> **关键区分**：CCBR **不去掉 detach** (区别于已证伪的方向 N 端到端可微 Cascade, mAP=0.684)。CCBR 是**纯推理期的信息共享增强**, 不改变训练动态, 详见 §4。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [理论依据](#2-理论依据)
3. [详细方案设计](#3-详细方案设计)
4. [深入可行性分析 — 与端到端可微 Cascade 的本质区别](#4-深入可行性分析--与端到端可微-cascade-的本质区别)
5. [风险评估](#5-风险评估)
6. [预期收益分析](#6-预期收益分析)
7. [实现路线图](#7-实现路线图)
8. [与已证伪方向的系统对比](#8-与已证伪方向的系统对比)
9. [参考文献](#9-参考文献)

---

## 1. 背景与动机

### 1.1 box_renewal: 扩散检测最强的单一特有机制

阅读 [sampling.py:96-117](../../ldmdet/diffusion/sampling.py#L96-L117) 可知, box_renewal 是扩散检测区别于传统检测器的核心机制:

```python
# sampling.py:96-117 — apply_box_renewal
def apply_box_renewal(self, x_raw, cls_logits):
    """框更新：低置信度框替换为随机噪声"""
    scores = torch.sigmoid(cls_logits).max(-1)[0]
    x_raw_new = x_raw.clone()
    for i in range(bs):
        keep = scores[i] > self.score_thr       # 高置信度: 保留 (利用)
        if keep.sum() < self.min_keep:           # 兜底: 至少保留 min_keep 个
            _, topk_idx = scores[i].topk(min(self.min_keep, ...))
            keep[topk_idx] = True
        num_renew = (~keep).sum()
        if num_renew > 0:
            x_raw_new[i, ~keep] = torch.randn(num_renew, 4, device=device)  # 低置信度: 替换为噪声 (探索)
    return x_raw_new
```

**机制本质**:
- **利用 (Exploit)**: 高置信度框 (`scores > score_thr`) 保留, 继续在当前轨迹上精化
- **探索 (Explore)**: 低置信度框替换为随机噪声, 给模型"重新开始"的机会
- **兜底**: `min_keep` 保证至少保留 N 个框, 防止全部被替换

**实证贡献**: 基于瓶颈消融实验 (chromo 数据集, SOTA=0.746):
- `box_renewal=True` (基线): mAP = 0.746
- `box_renewal=False` (关闭): mAP = 0.730
- **Δ = +0.016 mAP** — 最强的单一扩散特有机制贡献

box_renewal 的 +0.016 增益甚至超过了 1→8 步采样递增的总和 (+0.015, 但需 8× 计算量), 是性价比最高的扩散特有能力。

### 1.2 当前 box_renewal 的两大局限

阅读 [head.py:317-395](../../ldmdet/core/head.py#L317-L395) 的 `predict` 方法和 [head.py:174-229](../../ldmdet/core/head.py#L174-L229) 的 `forward` 方法, 可识别 box_renewal 当前存在两大结构性局限:

#### 局限一: 仅在扩散时间步之间生效, 未在级联 Head 之间生效

```python
# head.py:332-385 — predict 方法 (简化)
for step_idx, (t_curr, t_next) in enumerate(time_pairs):
    # ① 一次 _forward_at_t 调用 = 完整 6 级级联前向
    cls_logits, pred_bboxes, x0_raw, velocity = self._forward_at_t(features, x_raw, t_curr, img_metas)
    ...
    # ② 求解器步进 (Euler/Heun/DPM)
    x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn, ...)
    # ③ box_renewal — 仅在此处! 用最后一级 Head 的 cls_logits
    if self.box_renewal:
        x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
```

```python
# head.py:182-217 — forward 方法 (级联核心)
for head in self.head_series:          # 6 级串行
    result = head(features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb)
    cls_logits, pred_bboxes, curr_proposals = result
    inter_cls_logits.append(cls_logits)
    inter_pred_bboxes.append(pred_bboxes)
    ...
    curr_bboxes = next_bboxes.detach() if self.cascade_detach else next_bboxes
    # ↑ 级联 Head 之间: 无 box_renewal! 直接传递框
```

**问题**: 6 级 Head 串行精化, 但 Head 之间的框传递**没有探索机制**。若 Head 1 在某个 proposal 上预测错误 (低置信度), 这个错误框直接传给 Head 2 的 RoIAlign 采样位置, Head 2 在错误位置采样 → 特征质量差 → 继续预测错误 → 错误传播到 Head 6。

box_renewal 的"探索"能力被限制在了时间步维度, 未渗透到级联维度。

#### 局限二: 仅使用最后一级 Head 的置信度, 丢弃了中间 Head 的信息

```python
# head.py:333 — _forward_at_t 返回最后一级 Head 的 cls_logits
cls_logits, pred_bboxes, x0_raw, velocity = self._forward_at_t(features, x_raw, t_curr, img_metas)

# head.py:481-485 — _forward_at_t 只取最后一级
cls_logits_seq, pred_output_seq, _ = self(features, curr_bboxes, t_input)
cls_logits_last = cls_logits_seq[-1]      # ← 只用最后一级!
pred_output_last = pred_output_seq[-1]
```

**问题**: 6 级 Head 逐级精化, 每级都有独立的 cls_logits。当前 box_renewal 仅使用 `cls_logits_last` (Head 6 的置信度), 丢弃了 Head 1-5 的置信度信息。

但后面 Head 的置信度更准 (在更精化的框上预测), 这是**信息浪费**: 前面 Head 的置信度本可用于指导更精细的 renewal 策略。

### 1.3 级联单向信息流的局限

当前 6 级级联是**严格单向**的:

```
Head 1 → Head 2 → Head 3 → Head 4 → Head 5 → Head 6
 (粗)                                              (精)
```

信息只能从前向后流动 (前一级的输出作为后一级的输入), 后面 Head 的精化结果**无法反馈**到前面 Head 的决策中。

**局限场景**: Head 6 对某个 proposal 给出高置信度 (说明这个框是真实目标), 但 Head 1 对同一 proposal 给出低置信度 (早期框还很粗糙)。当前架构下, Head 1 无法"知道" Head 6 的判断, 可能在 box_renewal 时错误地替换了这个框。

虽然级联内的 box_renewal 当前不存在, 但即使引入, 若仅用本级 Head 的置信度做 renewal 判断, 前面 Head (粗精化) 的低置信度会导致过多框被替换, 破坏后面 Head 已建立精化轨迹的连续性。

### 1.4 CCBR 的核心洞察

CCBR 的核心: **让后面 Head 的精化结果指导前面 Head 的 box_renewal 策略**, 同时**保持梯度截断**, 仅在推理时共享信息。

```
当前架构 (单向, 无级联 renewal):
  时间步 T:  Head1 → Head2 → Head3 → Head4 → Head5 → Head6 → [box_renewal 用 Head6 置信度]
                                                                  ↓
                                                             下一时间步 T-1

CCBR 架构 (跨级 renewal + 后向置信度传播):
  时间步 T:  Head1 →[renewal_1]→ Head2 →[renewal_2]→ ... → Head6
                  ↑                              ↑
                  └──── 后向置信度反馈 ──────────┘
                  (Head6 的精化置信度指导 Head1 的 renewal 阈值)
                                                              ↓
                                                         下一时间步 T-1
```

关键设计原则:
1. **级联间引入 box_renewal**: 在 Head k → Head k+1 之间加入探索机制, 避免错误框传播
2. **后向置信度传播**: Head 6 (最精化) 的置信度反馈指导前面 Head 的 renewal 策略
3. **保持 `cascade_detach=True`**: 不改变训练动态, 仅推理时增强
4. **复用已验证机制**: 增强而非替换 box_renewal (已验证 +0.016 mAP)

---

## 2. 理论依据

### 2.1 扩散采样中的探索-利用理论

box_renewal 本质是扩散采样中的 **ε-greedy 策略** 在连续空间上的推广:

- **利用 (Exploit)**: 高置信度框保留, 在当前轨迹上继续精化 — 对应 ε-greedy 的 `1-ε` 概率选最优动作
- **探索 (Explore)**: 低置信度框替换为随机噪声, 从新的初始条件重新开始 — 对应 ε-greedy 的 `ε` 概率随机探索

**理论框架**: 扩散采样可视为在框空间 $\mathcal{X} \subset \mathbb{R}^{4}$ 上的马尔可夫决策过程 (MDP):
- **状态** $s_t = (x_t, t)$: 当前框 $x_t$ 和扩散时间 $t$
- **动作** $a_t$: 框更新策略 (保留 or 替换为噪声)
- **奖励** $r$: 检测置信度 (cls_logits 的 sigmoid)
- **策略** $\pi(a | s)$: box_renewal 的阈值策略

**贝尔曼最优性**: 最优策略应在探索与利用之间平衡:
$$V^*(s_t) = \max_\pi \mathbb{E}\left[\sum_{k=0}^{T-t} \gamma^k r_{t+k} \mid s_t, \pi\right]$$

当前 box_renewal 用固定阈值 `score_thr` 做二分决策, 是最简单的**固定策略**。CCBR 的理论改进:

1. **状态增强**: 级联 Head 间的 renewal 引入更细粒度的决策点 (从 T 个决策点变为 T×6 个)
2. **信息增强**: 后向置信度传播让前面 Head 的决策基于后面 Head 的精化结果, 接近**部分可观测 MDP (POMDP)** 的信念状态更新
3. **策略自适应**: 不同 Head 用不同 renewal 阈值, 接近**状态相关策略** $\pi(a | s, k)$

### 2.2 置信度传播理论

级联 Head 的置信度具有**单调精化性**: 后面 Head 在更精化的框上预测, 置信度更可靠。

**形式化**: 设 Head $k$ 的置信度为 $s_k = \sigma(\text{cls\_logits}_k)$, 则:

$$\mathbb{E}[\text{IoU}(\hat{x}_k, x^*) | s_k > \tau] \leq \mathbb{E}[\text{IoU}(\hat{x}_{k+1}, x^*) | s_{k+1} > \tau]$$

即: 在相同阈值 $\tau$ 下, 后面 Head 的预测 IoU 更高。这是 Cascade R-CNN (Cai & Vasconcelos, CVPR 2018) 的核心理论依据 — 级联结构使后面 Head 的正样本质量递增。

**CCBR 的置信度传播**: 利用后面 Head 的高置信度信息, 指导前面 Head 的 renewal:
- 若 Head 6 对某 proposal 高置信 → 这个 proposal 是真实目标 → Head 1-5 不应替换它 (即使前面 Head 置信度低)
- 若 Head 6 对某 proposal 低置信 → 这个 proposal 可能是假阳 → Head 1-5 可以更积极地 renewal

这等价于**后验概率更新**:
$$p(\text{keep} | s_1, \dots, s_6) \propto p(s_6 | \text{keep}) \cdot p(\text{keep} | s_1, \dots, s_5)$$

后面 Head 的置信度 $s_6$ 作为"后验观测", 修正前面 Head 的"先验"判断。

### 2.3 多级决策的后悔上界

引入级联间 renewal 后, 决策点从 $T$ 增加到 $T \times 6$。根据多臂老虎机理论:

**后悔上界** (对 K 臂老虎机, T 步):
$$R_T \leq O(\sqrt{KT \log T})$$

当前: $K=2$ (keep/renew), $T$ = 采样步数 (4 步 Heun → 4 个决策点)
CCBR: $K=2$, $T \times 6$ = 24 个决策点

**关键洞察**: 后悔上界随 $\sqrt{T}$ 增长, 但收益也随 $T$ 增长 (更多决策点 = 更多纠正机会)。在 $T \times 6$ 个决策点下, 若每个决策点的纠正概率为 $p$, 则总纠正次数 $\approx p \cdot T \cdot 6$, 远超当前 $p \cdot T$。

**风险控制**: 决策点增多也意味着错误决策增多。CCBR 通过**后向置信度传播**降低错误决策概率: 后面 Head 的置信度指导使前面 Head 的 renewal 更保守 (不误杀高置信度框)。

### 2.4 与 DiffusionDet 原始 box_renewal 的理论关系

DiffusionDet (Chi et al., ICCV 2023) 的 box_renewal 设计用于时间步维度, 理论依据是:
- 扩散采样的不同时间步, 模型对同一 proposal 的预测可能不同
- 低置信度框在下一时间步"重新开始", 给模型更多采样机会

CCBR 将这一理论**推广到级联维度**:
- 级联 Head 的不同级, 对同一 proposal 的预测也不同 (逐级精化)
- 低置信度框在下一级 Head "局部重新开始", 在级联内部提供更多精化机会
- 但级联内的"重新开始"不能完全重置 (否则破坏级联的逐级精化), CCBR 采用**软 renewal** (见 §3.3)

---

## 3. 详细方案设计

### 3.1 整体架构

CCBR 在现有架构上增加两个组件:

1. **级联间 box_renewal (Inter-Head Renewal)**: 在 Head k → Head k+1 之间加入 box_renewal
2. **后向置信度传播 (Backward Confidence Propagation)**: Head 6 的置信度反馈指导前面 Head 的 renewal 阈值

```
时间步 T 的单次级联前向 (CCBR):

  x_raw (T 时刻噪声框)
    ↓
  ┌─ Head 1 ─→ cls_1, pred_1
  │    ↓ [inter_renewal_1: 用融合置信度 c_1 调整框]
  │    ↓
  ├─ Head 2 ─→ cls_2, pred_2
  │    ↓ [inter_renewal_2: 用融合置信度 c_2 调整框]
  │    ↓
  ├─ ... (Head 3-5 同理)
  │    ↓
  └─ Head 6 ─→ cls_6, pred_6, x0_raw
       ↓
  [后向置信度传播: c_6 = σ(cls_6) 反向指导 c_1...c_5 的阈值]
       ↓
  求解器步进 (Euler/Heun) → x_raw (T-1 时刻)
       ↓
  [时间步间 box_renewal (现有, 保留): 用 cls_6]
       ↓
  下一时间步
```

### 3.2 组件一: 级联间 box_renewal (Inter-Head Renewal)

#### 3.2.1 设计原则

级联间 renewal 与时间步间 renewal 有本质区别:

| 维度 | 时间步间 renewal (现有) | 级联间 renewal (CCBR 新增) |
|------|------------------------|---------------------------|
| 作用域 | 扩散空间 ($x_{\text{raw}}$, cxcywh 归一化) | 图像空间 ($xyxy$, RoIAlign 输入) |
| 替换内容 | 替换为随机噪声 (重新开始扩散轨迹) | 替换为**带扰动的本级预测** (软重启, 不完全重置) |
| 决策依据 | 最后一级 Head 的 cls_logits | 融合置信度 (本级 + 后向传播) |
| 频率 | 每时间步 1 次 (4 步 → 4 次) | 每级 Head 1 次 (6 级 → 5 次) |
| 风险 | 低 (下一时间步重新预测) | 中 (破坏级联逐级精化的连续性) |

**关键设计: 软 renewal (Soft Renewal)**

级联间不能像时间步间那样直接替换为纯随机噪声, 否则:
- Head k 的精化结果被完全丢弃
- Head k+1 从纯噪声框开始, 等于跳过前 k 级
- 破坏级联的逐级精化设计

CCBR 采用**软 renewal**: 对低置信度框, 不替换为纯噪声, 而是融合本级预测与一个扰动项:

$$x_{k+1}^{(i)} = \begin{cases} \hat{x}_k^{(i)} & \text{if } c_k^{(i)} > \tau_k \quad \text{(保留)} \\ \alpha \cdot \hat{x}_k^{(i)} + (1-\alpha) \cdot (\hat{x}_k^{(i)} + \sigma_{\text{renew}} \cdot \varepsilon) & \text{if } c_k^{(i)} \leq \tau_k \quad \text{(软 renewal)} \end{cases}$$

其中:
- $\hat{x}_k^{(i)}$ = Head k 对第 $i$ 个 proposal 的预测 (xyxy)
- $c_k^{(i)}$ = 融合置信度 (见 §3.3)
- $\tau_k$ = Head k 的 renewal 阈值 (由后向置信度传播调整)
- $\alpha \in [0.5, 1.0]$ = 软 renewal 保留率 (Phase 1 用 $\alpha=0.7$)
- $\sigma_{\text{renew}}$ = 扰动幅度 (Phase 1 用框对角线长度的 10%)
- $\varepsilon \sim \mathcal{N}(0, I_4)$

**直觉**: 软 renewal 不是"从零开始", 而是"在当前位置附近抖动", 给 Head k+1 一个略微不同的采样位置, 避免在错误位置上反复精化。

#### 3.2.2 实现伪代码

```python
def apply_inter_head_renewal(
    self,
    pred_bboxes,      # [bs, N, 4] xyxy, Head k 的预测
    fused_conf,       # [bs, N] 融合置信度 (见 §3.3)
    threshold,        # [bs] 或标量, Head k 的 renewal 阈值
    alpha=0.7,        # 软 renewal 保留率
    sigma_scale=0.1,  # 扰动幅度 = 框对角线 × sigma_scale
):
    """级联间软 box_renewal"""
    bs, N, _ = pred_bboxes.shape
    device = pred_bboxes.device

    # 计算每个框的对角线长度 (作为扰动尺度的参考)
    diag = torch.sqrt(
        (pred_bboxes[..., 2] - pred_bboxes[..., 0])**2
        + (pred_bboxes[..., 3] - pred_bboxes[..., 1])**2
    )  # [bs, N]

    # 决策: 保留 or 软 renewal
    keep = fused_conf > threshold  # [bs, N]
    # 兜底: 至少保留 min_keep 个
    for i in range(bs):
        if keep[i].sum() < self.min_keep:
            _, topk_idx = fused_conf[i].topk(min(self.min_keep, N))
            keep[i, topk_idx] = True

    # 软 renewal: alpha * pred + (1-alpha) * (pred + sigma * noise)
    noise = torch.randn(bs, N, 4, device=device)
    sigma = sigma_scale * diag.unsqueeze(-1)  # [bs, N, 1]
    perturbed = pred_bboxes + sigma * noise
    renewed = alpha * pred_bboxes + (1 - alpha) * perturbed

    # 应用: 保留的用原预测, 低置信度的用软 renewal
    result = torch.where(keep.unsqueeze(-1), pred_bboxes, renewed)
    return result
```

### 3.3 组件二: 后向置信度传播 (Backward Confidence Propagation)

#### 3.3.1 设计原则

后面 Head 的置信度更可靠, 应指导前面 Head 的 renewal 决策。但级联是单向前向的, 如何在单向架构中实现"后向"信息传播?

**CCBR 的方案: 两遍前向 (Two-Pass Forward)**

1. **第一遍 (前向)**: 正常运行 Head 1→6, 收集每级的 cls_logits
2. **后向传播**: 用 Head 6 的置信度计算"融合置信度", 传给前面 Head
3. **第二遍 (前向 + 级联间 renewal)**: 用融合置信度指导级联间 renewal

但这会**增加一倍计算量** (两遍前向)。为控制开销, CCBR 采用**近似方案**:

#### 3.3.2 近似方案: 延迟置信度传播

**关键观察**: 推理时有多个时间步 (Heun 4 步)。**上一时间步 Head 6 的置信度, 可用于当前时间步 Head 1-5 的 renewal 指导**。

```
时间步 T:
  Head 1 → Head 2 → ... → Head 6 → cls_6(T)
                                    ↓ [缓存 cls_6(T)]
  求解器步进 → x_raw(T-1)

时间步 T-1:
  [inter_renewal_1 用 cls_6(T) 指导] → Head 1 → Head 2 → ... → Head 6 → cls_6(T-1)
                                              ↓ [缓存 cls_6(T-1)]
  求解器步进 → x_raw(T-2)
  ...
```

**优势**:
- 无需两遍前向, 计算开销仅增加级联间 renewal 本身
- 利用时间步间的时序信息: 上一时间步的精化结果指导当前时间步的级联决策
- 第一个时间步无历史信息, 用本级置信度兜底

#### 3.3.3 融合置信度算法

对 Head $k$ 在时间步 $T$, 定义融合置信度 $c_k^{(T)}$:

$$c_k^{(T)} = \lambda \cdot \sigma(\text{cls\_logits}_k^{(T)}) + (1-\lambda) \cdot \sigma(\text{cls\_logits}_6^{(T+1)})$$

其中:
- $\sigma(\text{cls\_logits}_k^{(T)})$ = 当前时间步本级 Head $k$ 的置信度
- $\sigma(\text{cls\_logits}_6^{(T+1)})$ = **上一时间步** Head 6 的置信度 (缓存)
- $\lambda \in [0, 1]$ = 融合权重 (Phase 1 用 $\lambda=0.5$)

**特殊情况**: 第一个时间步 ($T = T_{\max}$) 无历史信息:
$$c_k^{(T_{\max})} = \sigma(\text{cls\_logits}_k^{(T_{\max})}) \quad \text{(仅用本级置信度)}$$

#### 3.3.4 自适应阈值调整

后向置信度不仅融合到置信度本身, 还用于**调整 renewal 阈值**:

$$\tau_k = \tau_{\text{base}} \cdot \left(1 + \beta \cdot (\bar{c}_6^{(T+1)} - \bar{c}_k^{(T)})\right)$$

其中:
- $\tau_{\text{base}}$ = 基础阈值 (当前 `score_thr=0.05`)
- $\bar{c}_6^{(T+1)}$ = 上一时间步 Head 6 的平均置信度
- $\bar{c}_k^{(T)}$ = 当前时间步 Head $k$ 的平均置信度
- $\beta$ = 阈值调整强度 (Phase 1 用 $\beta=0.5$)

**直觉**:
- 若 Head 6 的平均置信度高于 Head $k$ → 后面 Head 更"自信" → 提高 $\tau_k$, 前面 Head 更积极 renewal (让后面 Head 重新精化)
- 若 Head 6 的平均置信度低于 Head $k$ → 后面 Head 不自信 → 降低 $\tau_k$, 前面 Head 保守保留

#### 3.3.5 实现伪代码

```python
@torch.no_grad()
def predict_with_ccbr(self, features, img_metas, rescale=True):
    """CCBR 推理: 级联间 renewal + 后向置信度传播"""
    bs = len(img_metas)
    time_pairs = self._sampler.build_time_pairs(features[0].device)
    x_raw = torch.randn(bs, self.num_proposals, 4, device=features[0].device)

    # CCBR: 缓存上一时间步 Head 6 的置信度
    prev_head6_scores = None  # [bs, N] or None (第一步)

    ensemble_results = []
    for step_idx, (t_curr, t_next) in enumerate(time_pairs):
        # === CCBR: 带级联间 renewal 的前向 ===
        cls_logits_last, pred_bboxes_last, x0_raw, inter_cls = (
            self._forward_at_t_ccbr(
                features, x_raw, t_curr, img_metas, prev_head6_scores
            )
        )
        # 缓存当前时间步 Head 6 的置信度, 供下一时间步使用
        prev_head6_scores = torch.sigmoid(cls_logits_last).max(-1)[0]  # [bs, N]

        if self.use_ensemble:
            ensemble_results.append((cls_logits_last, pred_bboxes_last))

        # 求解器步进 (不变)
        if self.solver_type == 'heun' and t_next > 0:
            ...
            x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, model_fn)
        else:
            x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

        # 时间步间 box_renewal (现有, 保留)
        if self.box_renewal:
            x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits_last)

        if t_next <= 0:
            break

    return self._sampler.post_process(ensemble_results, img_metas, rescale)


def _forward_at_t_ccbr(self, features, x_raw, t, img_metas, prev_head6_scores):
    """CCBR 前向: 6 级级联 + 级联间 renewal"""
    bs = x_raw.shape[0]
    curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
    t_input = torch.full((bs,), t * self.timesteps, device=x_raw.device)
    time_emb = self.time_mlp(t_input)

    inter_cls_logits = []
    curr_proposals = None

    for k, head in enumerate(self.head_series):
        cls_logits, pred_bboxes, curr_proposals = head(
            features, curr_bboxes, curr_proposals, self.roi_extractor, time_emb
        )
        inter_cls_logits.append(cls_logits)

        # === CCBR: 级联间 renewal (除最后一级外) ===
        if k < len(self.head_series) - 1:
            curr_scores = torch.sigmoid(cls_logits).max(-1)[0]  # [bs, N]

            # 融合置信度: 当前级 + 上一时间步 Head 6
            if prev_head6_scores is not None:
                fused_conf = (
                    self.ccbr_lambda * curr_scores
                    + (1 - self.ccbr_lambda) * prev_head6_scores
                )
                # 自适应阈值
                mean_diff = (
                    prev_head6_scores.mean() - curr_scores.mean()
                )
                threshold = self.score_thr * (
                    1 + self.ccbr_beta * mean_diff.item()
                )
                threshold = max(threshold, 0.01)  # 下限
            else:
                fused_conf = curr_scores
                threshold = self.score_thr

            # 软 renewal
            curr_bboxes = self._sampler.apply_inter_head_renewal(
                pred_bboxes, fused_conf, threshold,
                alpha=self.ccbr_alpha, sigma_scale=self.ccbr_sigma,
            )
        else:
            # 最后一级: 正常传递 (cascade_detach 仍生效)
            pass

        # 保持 cascade_detach (CCBR 核心: 不改训练动态)
        curr_bboxes = (
            curr_bboxes.detach()
            if self.cascade_detach
            else curr_bboxes
        )

    cls_logits_last = inter_cls_logits[-1]
    pred_bboxes_last = pred_bboxes  # 最后一级的输出
    x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)

    return cls_logits_last, pred_bboxes_last, x0, inter_cls_logits
```

### 3.4 配置参数

CCBR 新增配置参数 (全部有合理默认值, 可零配置启动):

```python
# experiments/configs/ldmdet/directions/u_ccbr_24obj.py
_base_ = ['../mainline_ablation_24obj/a3_full_sota_24obj.py']

model = dict(
    bbox_head=dict(
        # CCBR 总开关
        use_ccbr=True,

        # 级联间 renewal 参数
        ccbr_alpha=0.7,           # 软 renewal 保留率
        ccbr_sigma=0.1,           # 扰动幅度 (框对角线比例)

        # 后向置信度传播参数
        ccbr_lambda=0.5,          # 融合权重: 当前级 vs 上一时间步 Head6
        ccbr_beta=0.5,            # 自适应阈值调整强度

        # 保持现有机制不变
        cascade_detach=True,      # ← 关键: 保持 detach!
        box_renewal=True,         # ← 保留时间步间 renewal
        deep_supervision=True,    # ← 保留深度监督
    ),
)
```

### 3.5 与现有代码的集成方案

CCBR 是**纯推理期优化**, 改动集中在 `DiffusionSampler` 和 `predict` 方法, **不改变训练逻辑**:

| 文件 | 改动 | 说明 |
|------|------|------|
| `ldmdet/diffusion/sampling.py` | 新增 `apply_inter_head_renewal` 方法 | §3.2.2 软 renewal 实现 |
| `ldmdet/core/head.py` | 新增 `_forward_at_t_ccbr` 方法 | §3.3.5 CCBR 前向逻辑 |
| `ldmdet/core/head.py` | 修改 `predict` 方法, 增加缓存逻辑 | 缓存 `prev_head6_scores` |
| `ldmdet/core/head.py` | `__init__` 新增 CCBR 参数 | `use_ccbr`, `ccbr_alpha` 等 |
| 配置文件 | 新增 `u_ccbr_24obj.py` | 启用 CCBR |

**关键: 训练路径完全不变**。`loss` 方法、`forward` 方法 (训练时)、`criterion` 全部保持原样。CCBR 只在 `predict` (推理) 路径上生效。

---

## 4. 深入可行性分析 — 与端到端可微 Cascade 的本质区别

### 4.1 方向 N (端到端可微 Cascade) 失败复盘

阅读 [方向N文档](方向N_端到端可微Cascade.md) 可知, 方向 N 的核心改动是:

```python
# 方向 N: 去掉 detach
curr_bboxes = pred_bboxes        # ← 不 detach, 梯度贯通
deep_supervision = False         # ← 只监督最后 stage
```

**实验结果**: mAP = **0.684** (vs 基线 0.856, **Δ = -0.172**), EarlyStoppingHook 在 epoch 71 触发。

**失败根因** (方向 N 文档 §失败原因):
1. 去掉 detach 后 6 stage 链式梯度导致训练严重不稳定 — mAP 在 0.35~0.68 间剧烈震荡
2. cascade 结构对梯度截断有强依赖, 去 detach 后各 head 的梯度冲突导致性能崩塌
3. 与 Direction B (DecoupledHead, mAP=0.702) 一致 — cascade 架构改动风险极高

**核心教训**: 6 级级联 Head 的链式梯度回传会导致**训练动态崩溃**。这不是"调参能解决"的问题, 而是架构性的梯度冲突。

### 4.2 CCBR 与方向 N 的本质区别

| 维度 | 方向 N (已证伪) | CCBR (本方向) |
|------|----------------|---------------|
| **是否改训练** | ✅ 改 (去 detach + 去 deep_supervision) | ❌ **不改** (训练路径完全不变) |
| **梯度流** | 6 级链式梯度回传 | **梯度仍截断** (`cascade_detach=True`) |
| **信息共享方式** | 训练时梯度贯通 | **推理时前向信息共享** (无梯度) |
| **训练稳定性** | 崩溃 (mAP 0.35~0.68 震荡) | **不变** (训练动态与基线完全相同) |
| **deep_supervision** | 关闭 (只监督最后 stage) | **保留** (6 级独立监督) |
| **失败模式** | 训练崩溃, 不可恢复 | 最坏退化为基线 (关闭 `use_ccbr` 即可) |
| **可回退性** | 差 (需重训) | **好** (推理期开关, 无需重训) |

### 4.3 为什么 CCBR 保持 detach 是安全的

这是 CCBR 可行性的核心论据。从三个层面分析:

#### 4.3.1 训练动态不变性

**方向 N 的问题**: 去掉 detach 后, Head 6 的损失梯度通过 `pred_bboxes` 链式回传到 Head 1。这导致:
- Head 1 的梯度同时来自自身损失 (deep_supervision) 和 Head 6 的损失 (链式回传)
- 6 级链式相乘, 梯度尺度指数增长或消失
- 各 Head 的梯度方向冲突 (Head 1 想"预测 GT", Head 6 想"让 Head 1 预测对 Head 2 有利的框")

**CCBR 的安全性**: `cascade_detach=True` 保持, 训练时:
- Head k 的梯度仅来自本级损失 (deep_supervision)
- Head k 的参数更新不依赖 Head k+1 的损失
- 训练动态与 a3_full_sota **完全一致**

```python
# CCBR 的训练路径 (与基线完全相同)
for head in self.head_series:
    cls_logits, pred_bboxes, curr_proposals = head(...)
    curr_bboxes = next_bboxes.detach()  # ← detach 保留!
    # loss 对每级独立计算 (deep_supervision)
```

```python
# CCBR 的推理路径 (新增级联间 renewal)
for k, head in enumerate(self.head_series):
    cls_logits, pred_bboxes, curr_proposals = head(...)
    if k < 5 and use_ccbr:
        curr_bboxes = apply_inter_head_renewal(pred_bboxes, fused_conf, ...)
    else:
        curr_bboxes = pred_bboxes
    curr_bboxes = curr_bboxes.detach()  # 推理时本就 no_grad, detach 无影响
```

**关键**: 推理时 `@torch.no_grad()` 已生效, `detach()` 本就无实际作用。CCBR 在推理时对 `curr_bboxes` 做 renewal 操作, 不涉及任何梯度回传。

#### 4.3.2 训练-推理一致性分析

**潜在风险**: CCBR 在推理时改变了级联间的框传递方式 (加入 renewal), 但训练时级联间是直接传递。这是否引入"训练-推理不一致" (类似 exposure bias)?

**分析**: 这种不一致是**良性的**, 原因:

1. **box_renewal 本身就有训练-推理不一致**:
   - 训练时: 无 box_renewal (训练用 GT 耦合的 x_t, 不需要 renewal)
   - 推理时: 有 box_renewal (时间步间)
   - 但 box_renewal 已验证 +0.016 mAP, 说明这种不一致是**有益的**

2. **级联间 renewal 的逻辑与时间步间 renewal 一致**:
   - 都是对低置信度框做"探索"
   - 都是用模型自己的置信度做判断
   - 软 renewal 比纯噪声替换更温和, 不一致性更弱

3. **模型训练时已学会处理"不同质量的框"**:
   - 训练时 t 随机, 高 t 时 x_t ≈ 噪声 (低质量框), 低 t 时 x_t ≈ GT (高质量框)
   - 模型已在各种框质量上训练, 推理时的 renewal 只是增加框的多样性, 不超出训练分布

**对比方向 N 的 exposure bias**: 方向 N 去 detach 后, 训练时 Head 1 的梯度依赖 Head 6 的损失, 但推理时 Head 1 的前向不依赖 Head 6 — 这才是真正的训练-推理不一致。CCBR 不存在此问题。

#### 4.3.3 数值稳定性

**方向 N 的数值问题**:
- 6 级链式梯度: $\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial \hat{x}_6} \cdot \prod_{k=1}^{5} \frac{\partial \hat{x}_{k+1}}{\partial \hat{x}_k} \cdot \frac{\partial \hat{x}_1}{\partial \theta_1}$
- 雅可比矩阵连乘, 数值不稳定

**CCBR 的数值稳定性**:
- 推理时 `@torch.no_grad()`, 无梯度计算
- 软 renewal 是简单的仿射变换: $\alpha \cdot x + (1-\alpha) \cdot (x + \sigma \varepsilon)$
- 无矩阵连乘, 无指数运算, 数值绝对稳定

### 4.4 CCBR 的下界保证

**定理 (CCBR 下界)**: 设基线 (无 CCBR) 的 mAP 为 $M_0$, 则 CCBR 的 mAP $M_{\text{CCBR}}$ 满足:

$$M_{\text{CCBR}} \geq M_0 - \epsilon_{\text{renewal}}$$

其中 $\epsilon_{\text{renewal}}$ 是软 renewal 引入的扰动误差, 且 $\epsilon_{\text{renewal}} \to 0$ 当 $\alpha \to 1$ (软 renewal 退化为保留)。

**证明思路**:
- 当 $\alpha = 1$ 时, 软 renewal 退化为完全保留 (`renewed = pred_bboxes`), CCBR 等价于基线
- 当 $\alpha < 1$ 时, 低置信度框被轻微扰动, 但 Head k+1 仍在扰动框附近采样 (非完全随机)
- 若扰动使结果变差, 可通过 `use_ccbr=False` 关闭, 回退到基线

**实践含义**: CCBR 是**非负收益方向** (最坏退化为基线), 与方向 N 的**可能负收益** (训练崩溃, -0.172) 形成鲜明对比。

### 4.5 计算开销分析

#### 4.5.1 单次前向开销

CCBR 在每次级联前向中增加 5 次 `apply_inter_head_renewal` (Head 1-5 各一次):

| 操作 | 复杂度 | 单次耗时 (bs=2, N=500) |
|------|--------|----------------------|
| 置信度计算 (sigmoid + max) | $O(bs \cdot N)$ | ~0.05 ms |
| 融合置信度 (加权平均) | $O(bs \cdot N)$ | ~0.02 ms |
| 软 renewal (where + randn) | $O(bs \cdot N \cdot 4)$ | ~0.1 ms |
| **单次级联间 renewal** | $O(bs \cdot N)$ | **~0.17 ms** |
| **5 次级联间 renewal** | $O(5 \cdot bs \cdot N)$ | **~0.85 ms** |

#### 4.5.2 总推理开销

基线推理: Heun 4 步 × 2 NFE = 8 次模型前向, 每次前向含 6 级 Head。

| 组件 | 基线 | CCBR | 增量 |
|------|------|------|------|
| 模型前向 (6 级 Head) | 8 × ~50 ms = 400 ms | 8 × ~50 ms = 400 ms | 0% |
| 级联间 renewal | 0 | 8 × 0.85 ms = 6.8 ms | +1.7% |
| 时间步间 renewal | 8 × 0.1 ms = 0.8 ms | 8 × 0.1 ms = 0.8 ms | 0% |
| **总推理时间** | ~401 ms | ~408 ms | **+1.7%** |

**结论**: CCBR 的计算开销 **< 2%**, 远低于 PCSE 的 +80%~250% 或增加采样步数的 +100%~700%。

#### 4.5.3 显存开销

CCBR 新增的中间张量:
- `prev_head6_scores`: [bs, N] = 2 × 500 × 4B = 4 KB
- 融合置信度、噪声等: 每级 ~[bs, N, 4] = 16 KB, 5 级 = 80 KB
- **总显存增量**: < 100 KB (可忽略)

---

## 5. 风险评估

### 5.1 风险一: 软 renewal 扰动破坏级联精化连续性

**风险描述**: 级联 Head 的设计意图是逐级精化 (Head 1 粗 → Head 6 精)。在级联间引入 renewal (即使软 renewal) 可能破坏这种连续性: Head k 的精化结果被扰动后, Head k+1 在扰动位置采样, 可能比在原始位置采样更差。

**量化风险**: 若软 renewal 的扰动幅度 $\sigma_{\text{renew}}$ 过大, Head k+1 的 RoIAlign 采样位置偏离真实目标, 特征质量下降。最坏情况: 扰动使框移出目标区域, Head k+1 采样到背景。

**缓解方案**:
1. **保守初始参数**: $\alpha=0.7$ (70% 保留原始预测), $\sigma_{\text{renew}}=0.1 \times \text{diag}$ (扰动幅度为框对角线的 10%)。这意味着扰动后的框仅偏离原始预测 ~3% (对角线长度), 仍在目标区域内
2. **仅对低置信度框 renewal**: 高置信度框完全保留 (`keep = fused_conf > threshold`), 仅低置信度框被扰动。高置信度框的精化轨迹不受影响
3. **Phase 1 消融**: 在验证集上对比 $\alpha \in \{0.5, 0.7, 0.9, 1.0\}$, 找到最优保留率。$\alpha=1.0$ 等价于关闭 CCBR, 作为对照
4. **框 clamp**: 扰动后的框 clamp 到图像范围, 防止采样到图像外

### 5.2 风险二: 后向置信度传播的时序滞后

**风险描述**: CCBR 的后向置信度传播用**上一时间步** Head 6 的置信度指导**当前时间步** Head 1-5 的 renewal。这存在时序滞后: 上一时间步的置信度可能不反映当前时间步的状态。

**量化风险**: 若相邻时间步的框状态变化大 (高 t 时框快速移动), 上一时间步的 Head 6 置信度与当前时间步的 Head 1 置信度相关性弱, 融合可能引入噪声。

**缓解方案**:
1. **第一个时间步用本级置信度兜底**: `prev_head6_scores is None` 时, 融合退化为仅用本级置信度, 等价于无后向传播
2. **融合权重可调**: $\lambda=0.5$ (等权) 是保守初值。若时序滞后严重, 可增大 $\lambda$ (更依赖本级置信度), $\lambda \to 1.0$ 时退化为无后向传播
3. **时间步间相关性验证**: Phase 0 先测量相邻时间步 Head 6 置信度的 Pearson 相关系数。若 $r > 0.7$ (高度相关), 时序滞后可接受; 若 $r < 0.3$, 降低 $\lambda$ 或改用两遍前向方案
4. **进阶方案 (Phase 3)**: 若时序滞后是瓶颈, 可实现两遍前向 (第一遍收集 Head 6 置信度, 第二遍用精确的 Head 6 置信度指导), 开销翻倍但无滞后

### 5.3 风险三: 自适应阈值的不稳定性

**风险描述**: CCBR 的自适应阈值 $\tau_k = \tau_{\text{base}} \cdot (1 + \beta \cdot (\bar{c}_6 - \bar{c}_k))$ 依赖平均置信度差。若某些图像的置信度分布异常 (如全低置信度, $\bar{c}_6 \approx \bar{c}_k \approx 0$), 阈值可能不稳定。

**量化风险**: 极端情况下, 若 $\bar{c}_6 \ll \bar{c}_k$ (Head 6 比 Head 1 不自信, 反常情况), 阈值可能降为负数或零, 导致所有框都被 renewal。

**缓解方案**:
1. **阈值下限 clamp**: `threshold = max(threshold, 0.01)`, 防止阈值过低
2. **阈值上限 clamp**: `threshold = min(threshold, 0.5)`, 防止阈值过高导致全部保留
3. **Phase 1 用固定阈值**: 先不启用自适应阈值 (`ccbr_beta=0`), 仅用固定 `score_thr` 做级联间 renewal。验证基础机制有效后再启用自适应
4. **退化安全**: 若自适应阈值异常, CCBBR 退化为固定阈值的级联间 renewal, 仍优于无 renewal 的基线

### 5.4 风险四 (附加): 收益不显著

**风险描述**: CCBR 的收益可能不如预期。若级联间 renewal 的探索收益被级联精化的利用收益抵消 (Head k+1 本就能纠正 Head k 的错误), CCBR 的净增益可能 < +0.003 mAP, 在噪声范围内。

**量化风险**: box_renewal 在时间步间贡献 +0.016 mAP, 但级联间的情境不同 — 级联本身就是精化机制, renewal 的"探索"可能与级联的"利用"目标冲突。

**缓解方案**:
1. **Phase 0 诊断**: 先测量当前基线中, Head 1 低置信度但 Head 6 高置信度的 proposal 比例。若 > 10%, CCBR 有明确纠正空间; 若 < 2%, 收益有限
2. **分层验证**: Phase 1 仅启用级联间 renewal (无后向传播), Phase 2 再加后向传播。分层评估各组件贡献
3. **最坏情况**: 若 CCBBR 无增益, 关闭 `use_ccbr` 即可回退, 无沉没成本 (训练无需重做)

---

## 6. 预期收益分析

### 6.1 基于 box_renewal +0.016 的定量外推

**已知数据**:
- box_renewal (时间步间): +0.016 mAP (no_box_renewal=0.730 vs baseline=0.746)
- box_renewal 在 4 个时间步决策点上贡献 +0.016 → 单决策点贡献 ~+0.004

**CCBR 的决策点分析**:

| 决策点类型 | 数量 | 单点贡献 (估) | 总贡献 (估) |
|-----------|------|--------------|------------|
| 时间步间 renewal (现有) | 4 | +0.004 | +0.016 (已验证) |
| 级联间 renewal (CCBR 新增) | 5 × 4 = 20 | +0.001 ~ +0.002 | +0.020 ~ +0.040 |
| 后向置信度传播 | 4 (每时间步 1 次) | +0.0005 ~ +0.001 | +0.002 ~ +0.004 |

**注**: 级联间 renewal 的单点贡献低于时间步间, 原因:
1. 级联间用软 renewal (扰动较小), 探索强度低于时间步间的纯噪声替换
2. 级联间 renewal 在同一时间步内, 框状态变化较小, 纠正空间有限
3. 级联本身已有精化能力, renewal 的边际收益递减

**收益递减修正**: 考虑收益递减 (类似 1→8 步采样的递减规律), 实际总贡献打 0.5 折扣:

| 增益来源 | 机制 | 预估 Δ mAP | 依据 |
|---------|------|-----------|------|
| 级联间 renewal | 在级联内增加探索点 | +0.005 ~ +0.012 | 20 个决策点 × 0.001 ~ 0.002, 打 0.5 折扣 |
| 后向置信度传播 | 更准的 renewal 决策 | +0.001 ~ +0.003 | 减少误 renewal, 提升决策质量 |
| 软 renewal 保连续性 | 避免破坏级联精化 | -0.001 ~ 0 | 软 renewal 比纯噪声温和, 但仍有扰动风险 |
| **合计** | | **+0.005 ~ +0.015** | 取中位数 **+0.010** |

**保守估计**: +0.003 mAP (0.858 → 0.861)
**中位估计**: +0.008 mAP (0.858 → 0.866)
**乐观估计**: +0.015 mAP (0.858 → 0.873)

### 6.2 与现有组件的增益对比

| 组件 | 增益 (Δ mAP) | 计算开销 | 风险 | 状态 |
|------|-------------|---------|------|------|
| box_renewal (时间步间) | +0.016 | 1× (无额外) | 已验证 | ✅ 基线 |
| 1→8 步采样 | +0.015 | 8× | 已验证 (递减) | ✅ 可选 |
| AdaLN-Zero | +0.011 | 1× | 已验证 | ✅ 基线 |
| StochasticOT ε=5 | +0.002 | 1× | 已验证 | ✅ 基线 |
| **CCBR (本方向)** | **+0.005 ~ +0.015** | **+1.7%** | 中 | 🔬 待验证 |
| PCSE (K=5) | +0.006 ~ +0.016 | +80%~250% | 中 | 🔬 待验证 |
| DPM-Solver++ 8步 | +0.002 | +12.5% | 已验证 | ✅ 可选 |
| CFM (已证伪) | -0.002 | 1× | 已失败 | ⛔ |
| 端到端可微 Cascade (已证伪) | -0.172 | 1× | 已崩溃 | ⛔ |

**关键观察**: CCBR 的性价比 (增益/开销) 是所有未验证方向中最高的:
- 开销仅 +1.7% (远低于 PCSE 的 +80%~250%)
- 增益预估与 PCSE 相当 (+0.010 vs +0.010)
- 不改变训练动态 (风险低于任何改训练的方向)

### 6.3 增益上限分析

CCBR 的增益上限受限于:

1. **级联精化能力**: 若 Head 6 已能纠正 Head 1 的所有错误, 级联间 renewal 无额外收益。上限: Head 1-5 中低置信度但最终被 Head 6 纠正的 proposal 比例
2. **软 renewal 的探索强度**: $\alpha=0.7$ 的软 renewal 比纯噪声替换温和, 探索强度有限。上限: $\alpha \to 0$ 时退化为纯噪声, 但破坏连续性
3. **时序滞后的信息质量**: 上一时间步的置信度与当前时间步的相关性。上限: 相关系数 $r$

**实际上限**: 预计 +0.015 mAP (乐观), 不超过 +0.020 (理论上限)。

---

## 7. 实现路线图

### Phase 0: 前置诊断 (2 天)

**目标**: 量化级联间的"错杀"比例, 验证 CCBR 有明确纠正空间。

**任务**:
1. 修改 `_forward_at_t` (临时), 收集验证集 100 张图上每级 Head 的 cls_logits
2. 统计: Head 1 低置信度 (`< score_thr`) 但 Head 6 高置信度 (`> 0.5`) 的 proposal 比例
3. 统计: 相邻时间步 Head 6 置信度的 Pearson 相关系数
4. 统计: Head 1-5 中低置信度框在 Head 6 的最终 IoU 分布

**决策点**:
- 若"错杀"比例 < 2% → CCBBR 增益有限, 重新评估优先级
- 若"错杀"比例 > 10% → CCBBR 有明确空间, 推进 Phase 1
- 若时序相关系数 $r < 0.3$ → 后向传播用两遍前向方案

### Phase 1: 核心实现 — 级联间 renewal (1 周)

**目标**: 实现级联间软 renewal (无后向传播), 验证基础机制收益。

**代码改动清单**:
1. `ldmdet/diffusion/sampling.py`:
   - 新增 `apply_inter_head_renewal(pred_bboxes, fused_conf, threshold, alpha, sigma_scale)` 方法
   - 参照 `apply_box_renewal` (L96-117) 的结构, 但用软 renewal 逻辑
2. `ldmdet/core/head.py`:
   - `__init__`: 新增 `use_ccbr`, `ccbr_alpha`, `ccbr_sigma` 参数
   - 新增 `_forward_at_t_ccbr` 方法 (基于 `_forward_at_t` L470-498 改造)
   - `predict` (L317-395): 增加 `use_ccbr` 分支, 调用 `_forward_at_t_ccbr`
3. `experiments/configs/ldmdet/directions/u_ccbr_24obj.py`:
   - 新增配置, `use_ccbr=True`, `ccbr_beta=0` (Phase 1 不启用自适应阈值)

**验证指标**: mAP, AP50, AP75, 与基线对比。

**预期**: +0.003 ~ +0.008 mAP (仅级联间 renewal, 无后向传播)

### Phase 2: 后向置信度传播 (1 周)

**目标**: 在 Phase 1 基础上启用后向置信度传播, 验证信息共享收益。

**代码改动清单**:
1. `ldmdet/core/head.py`:
   - `__init__`: 新增 `ccbr_lambda`, `ccbr_beta` 参数
   - `_forward_at_t_ccbr`: 增加 `prev_head6_scores` 参数, 实现融合置信度
   - `predict`: 增加 `prev_head6_scores` 缓存逻辑
2. `experiments/configs/ldmdet/directions/u_ccbr_v2_24obj.py`:
   - `ccbr_lambda=0.5`, `ccbr_beta=0.5` (启用后向传播)

**验证指标**: mAP, 对比 Phase 1, 量化后向传播的边际贡献。

**预期**: 在 Phase 1 基础上再 +0.001 ~ +0.003 mAP

### Phase 3: 调参与优化 (1 周)

**目标**: 网格搜索 CCBR 参数, 找到最优配置。

**任务**:
1. 网格搜索 $\alpha \in \{0.5, 0.6, 0.7, 0.8, 0.9\}$ (软 renewal 保留率)
2. 网格搜索 $\lambda \in \{0.3, 0.5, 0.7\}$ (融合权重)
3. 网格搜索 $\sigma_{\text{renew}} \in \{0.05, 0.1, 0.2\}$ (扰动幅度)
4. 在测试集上最终评估

**预期产出**: 最优参数组合, 测试集最终 mAP。

### Phase 4 (可选): 两遍前向方案

**目标**: 若 Phase 2 的时序滞后是瓶颈, 实现精确的后向置信度传播。

**任务**:
1. 第一遍前向: 正常运行 6 级级联, 收集 Head 6 的 cls_logits
2. 第二遍前向: 用 Head 6 的精确置信度指导级联间 renewal
3. 开销翻倍, 但无时序滞后

**决策点**: 仅当 Phase 2 增益 < +0.003 且诊断显示时序滞后严重时推进。

---

## 8. 与已证伪方向的系统对比

### 8.1 与方向 N (端到端可微 Cascade) 的对比 — 核心

这是 CCBR 可行性分析的关键。方向 N 是最接近 CCBR 的已证伪方向, 必须深入对比。

#### 8.1.1 表面相似性

| 特征 | 方向 N | CCBR |
|------|--------|------|
| 改动位置 | 级联 Head 间 | 级联 Head 间 |
| 目标 | 提升 RoI 采样质量 | 提升框更新质量 |
| 涉及 box_renewal | 否 | 是 |

表面看, 两者都改动级联 Head 间的信息流, 风险似乎相似。但实际上, 两者的**机制本质完全不同**。

#### 8.1.2 本质区别: 梯度流 vs 前向信息

**方向 N 的机制 (训练时梯度贯通)**:
```python
# 训练时
curr_bboxes = pred_bboxes  # 不 detach
# 梯度路径: L(Head 6 输出) → ∂/∂pred_5 → ∂/∂pred_4 → ... → ∂/∂θ_1
# 6 级链式梯度回传, 训练动态改变
```

**CCBR 的机制 (推理时前向信息共享)**:
```python
# 推理时 (@torch.no_grad())
curr_bboxes = apply_inter_head_renewal(pred_bboxes, fused_conf, ...)
# 无梯度! 仅前向信息处理
# 训练时: curr_bboxes = pred_bboxes.detach() (不变)
```

**关键区别**:
- 方向 N 改变**训练时的梯度流** → 6 级链式梯度 → 训练崩溃
- CCBR 改变**推理时的前向信息流** → 无梯度 → 训练不变

#### 8.1.3 失败模式对比

| 失败模式 | 方向 N | CCBR |
|---------|--------|------|
| 训练崩溃 | ✅ 发生 (mAP 0.35~0.68 震荡) | ❌ 不可能 (训练不变) |
| 梯度爆炸/消失 | ✅ 发生 (6 级链式) | ❌ 不可能 (无梯度) |
| 梯度冲突 | ✅ 发生 (各 Head 目标冲突) | ❌ 不可能 (无梯度) |
| Exposure bias | ✅ 加剧 (训练-推理不一致) | ❌ 良性 (同 box_renewal) |
| 收益不显著 | — | ⚠️ 可能 (但可回退) |
| 软 renewal 破坏连续性 | — | ⚠️ 可能 (但可调 α) |

**方向 N 的失败是灾难性的** (mAP -0.172, 不可恢复); **CCBR 的最坏情况是退化为基线** (mAP ±0, 可回退)。

#### 8.1.4 为什么方向 N 失败但 CCBR 不会

**根因分析**: 方向 N 失败的根因是**训练动态崩溃**, 不是"级联间信息共享"本身有问题。

类比:
- 方向 N = "让 6 个 Head 通过梯度协同优化" → 梯度冲突 → 崩溃
- CCBR = "让 6 个 Head 通过推理时信息共享协同决策" → 无梯度冲突 → 安全

这就像:
- "让 6 个人通过共同承担一个 KPI 协作" (方向 N) → 互相推诿, KPI 崩溃
- "让 6 个人在执行时共享信息" (CCBR) → 各自 KPI 不变, 但执行更协调

**结论**: 方向 N 证伪了"训练时梯度贯通级联", 但**没有证伪"推理时信息共享级联"**。CCBR 是后者, 与方向 N 的失败机制正交。

### 8.2 与方向 H (CFM) 的对比

| 维度 | CFM (方向 H, 已证伪) | CCBR (本方向) |
|------|---------------------|---------------|
| 是否改训练 | ✅ 改 (loss + head 输出) | ❌ 不改 (纯推理) |
| 是否引入新信息 | ❌ 不引入 (仅对齐目标) | ✅ 引入 (后向置信度传播) |
| 失败模式 | 无增益 (-0.002) | 退化为基线 (无负增益) |
| 可回退性 | 差 (需重训) | 好 (推理开关) |

### 8.3 与方向 R (PCSE) 的对比

| 维度 | PCSE (方向 R) | CCBR (本方向) |
|------|--------------|---------------|
| 作用阶段 | 推理 (后处理) | 推理 (采样过程内) |
| 计算开销 | +80%~250% (K=5) | +1.7% |
| 信息来源 | 核型先验 (外部知识) | 级联 Head 间 (模型内部) |
| 是否可组合 | ✅ 可与 CCBR 组合 | ✅ 可与 PCSE 组合 |

**协同可能**: CCBR 提升单假设质量, PCSE 从 K 个更好的假设中选最优。两者可叠加: CCBR 提升基线 → PCSE 在更高基线上选择 → 增益叠加。

### 8.4 与方向 J (确定性 Cascade) 的对比

| 维度 | 方向 J (确定性 Cascade) | CCBR (本方向) |
|------|----------------------|---------------|
| 是否保留扩散 | ❌ 移除 t | ✅ 保留 RF |
| 是否改训练 | ✅ 改 (去 detach + 移除 t) | ❌ 不改 |
| 实施周期 | 3-8 周 | 1-2 周 |
| 风险 | 中高 (架构大改) | 低 (推理增强) |

**关系**: CCBR 是方向 J 的"安全前置验证"。若 CCBR 的级联间信息共享有收益, 说明级联 Head 间的信息流动确实不足, 为方向 J 提供依据; 若无收益, 说明级联间的信息流动不是瓶颈, 方向 J 的优先级降低。

### 8.5 与现有 box_renewal 的关系

CCBR **不替代**现有 box_renewal, 而是**增强**它:

```
现有:  时间步间 box_renewal (用 Head 6 置信度)
CCBR:  时间步间 box_renewal (保留) + 级联间 box_renewal (新增, 用融合置信度)
```

**关键**: CCBR 的级联间 renewal 是 box_renewal 在级联维度的推广, 两者机制一致 (探索-利用), 但作用域不同。box_renewal 已验证 +0.016 mAP, CCBR 在此基础上增加级联维度的探索, 是对已验证机制的增强, 风险可控。

---

## 9. 参考文献

### 9.1 扩散检测与 box_renewal
- DiffusionDet: Chi et al. "DiffusionDet: Diffusion Model for Object Detection." ICCV, 2023. — box_renewal 原始设计
- Rectified Flow: Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR, 2023. — 当前所用 RF 采样
- DPM-Solver++: Lu et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models." NeurIPS, 2022. — 求解器理论

### 9.2 级联检测与置信度传播
- Cascade R-CNN: Cai & Vasconcelos. "Cascade R-CNN: Delving into High Quality Object Detection." CVPR, 2018. — 级联 Head 逐级精化理论, 置信度单调性
- Sparse R-CNN: Sun et al. "Sparse R-CNN: End-to-End Object Detection with Learnable Proposals." CVPR, 2021. — 迭代精化 + proposal 交互
- DINO: Zhang et al. "DINO: DETR with Improved Denoising Anchor Boxes for End-to-End Object Detection." ICLR, 2023. — 去噪锚框的迭代精化

### 9.3 探索-利用与多臂老虎机
- Sutton & Barto. "Reinforcement Learning: An Introduction." 2nd Ed., MIT Press, 2018. — ε-greedy 策略, 探索-利用权衡
- Auer et al. "Finite-time Analysis of the Multiarmed Bandit Problem." Machine Learning, 2002. — 后悔上界理论
- Kaelbling et al. "Planning and Acting in Partially Observable Stochastic Domains." Artificial Intelligence, 1998. — POMDP 信念状态更新

### 9.4 集成与信息融合
- Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS, 2017. — 深度集成, 多次采样选择
- Gal & Ghahramani. "Dropout as a Bayesian Approximation." ICML, 2016. — MC Dropout, 随机性利用

### 9.5 项目内相关方向
- [方向 N: 端到端可微 Cascade (已证伪)](方向N_端到端可微Cascade.md) — 去 detach 失败教训, CCBR 的核心对比
- [方向 R: PCSE 配对一致性随机集成评分](方向R_PCSE_配对一致性随机集成评分.md) — 推理期优化, 可与 CCBR 组合
- [方向 J: 确定性 Cascade 精化](方向J_确定性Cascade精化.md) — CCBR 的长期演进方向
- [方向 H: Flow Matching 检测 (已证伪)](方向H_FlowMatching检测.md) — CFM 失败教训
- [方向 S: BPCVF 框对一致性速度流](方向S_BPCVF_框对一致性速度流.md) — 速度场结构化, 与 CCBR 正交

---

## 附录 A: CCBR 完整实现伪代码

```python
class DiffusionSampler:
    """扩展 DiffusionSampler, 增加 apply_inter_head_renewal"""

    def apply_inter_head_renewal(
        self,
        pred_bboxes: Tensor,      # [bs, N, 4] xyxy
        fused_conf: Tensor,       # [bs, N] 融合置信度
        threshold: float,         # renewal 阈值
        alpha: float = 0.7,       # 软 renewal 保留率
        sigma_scale: float = 0.1, # 扰动幅度 (框对角线比例)
    ) -> Tensor:
        """级联间软 box_renewal

        与 apply_box_renewal 的区别:
        1. 作用在 xyxy 空间 (级联间), 非 raw 空间 (时间步间)
        2. 软 renewal (alpha * pred + (1-alpha) * perturbed), 非纯噪声
        3. 用融合置信度 (本级 + 后向传播), 非仅最后一级
        """
        bs, N, _ = pred_bboxes.shape
        device = pred_bboxes.device

        # 框对角线长度作为扰动尺度
        diag = torch.sqrt(
            (pred_bboxes[..., 2] - pred_bboxes[..., 0]).clamp(min=1e-6) ** 2
            + (pred_bboxes[..., 3] - pred_bboxes[..., 1]).clamp(min=1e-6) ** 2
        )  # [bs, N]

        # 决策: 保留 or 软 renewal
        keep = fused_conf > threshold  # [bs, N]
        for i in range(bs):
            if keep[i].sum() < self.min_keep:
                _, topk_idx = fused_conf[i].topk(
                    min(self.min_keep, N)
                )
                keep[i, topk_idx] = True

        # 软 renewal
        noise = torch.randn(bs, N, 4, device=device)
        sigma = sigma_scale * diag.unsqueeze(-1)  # [bs, N, 1]
        perturbed = pred_bboxes + sigma * noise
        renewed = alpha * pred_bboxes + (1 - alpha) * perturbed

        # 框 clamp 到图像范围 (需 img_metas, 此处省略)
        # renewed = self._clamp_bboxes(renewed, img_metas)

        result = torch.where(keep.unsqueeze(-1), pred_bboxes, renewed)
        return result


class DiffusionDetHead(nn.Module):
    """扩展 DiffusionDetHead, 增加 CCBR 支持"""

    def __init__(self, ..., use_ccbr=False, ccbr_alpha=0.7,
                 ccbr_sigma=0.1, ccbr_lambda=0.5, ccbr_beta=0.5):
        super().__init__(...)
        self.use_ccbr = use_ccbr
        self.ccbr_alpha = ccbr_alpha
        self.ccbr_sigma = ccbr_sigma
        self.ccbr_lambda = ccbr_lambda
        self.ccbr_beta = ccbr_beta

    @torch.no_grad()
    def predict(self, features, img_metas, rescale=True):
        """CCBR 增强的推理流程"""
        if not self.use_ccbr:
            return self._predict_original(features, img_metas, rescale)

        device = features[0].device
        bs = len(img_metas)
        time_pairs = self._sampler.build_time_pairs(device)
        x_raw = torch.randn(bs, self.num_proposals, 4, device=device)

        # CCBR: 缓存上一时间步 Head 6 的置信度
        prev_head6_scores = None

        ensemble_results = []
        for step_idx, (t_curr, t_next) in enumerate(time_pairs):
            # CCBR 前向
            cls_logits, pred_bboxes, x0_raw, _ = self._forward_at_t_ccbr(
                features, x_raw, t_curr, img_metas, prev_head6_scores
            )

            # 缓存 Head 6 置信度
            prev_head6_scores = torch.sigmoid(cls_logits).max(-1)[0]

            if self.use_ensemble:
                ensemble_results.append((cls_logits, pred_bboxes))

            # 求解器步进 (不变)
            if self.solver_type == 'heun' and t_next > 0:
                def model_fn(x_tmp, t_tmp):
                    return self._forward_at_t_ccbr(
                        features, x_tmp, t_tmp, img_metas,
                        prev_head6_scores
                    )[:3]
                x_raw = self.rf.heun_step(
                    x_raw, x0_raw, t_curr, t_next, model_fn
                )
            else:
                x_raw = self.rf.step(x_raw, x0_raw, t_curr, t_next)

            # 时间步间 box_renewal (现有, 保留)
            if self.box_renewal:
                x_raw = self._sampler.apply_box_renewal(x_raw, cls_logits)
            if t_next <= 0:
                break

        return self._sampler.post_process(
            ensemble_results, img_metas, rescale
        )

    def _forward_at_t_ccbr(self, features, x_raw, t, img_metas,
                           prev_head6_scores=None):
        """CCBR 前向: 6 级级联 + 级联间 renewal"""
        bs = x_raw.shape[0]
        curr_bboxes = self._sampler.raw_to_xyxy(x_raw, img_metas)
        t_input = torch.full((bs,), t * self.timesteps, device=x_raw.device)
        time_emb = self.time_mlp(t_input)

        inter_cls_logits = []
        curr_proposals = None

        for k, head in enumerate(self.head_series):
            result = head(
                features, curr_bboxes, curr_proposals,
                self.roi_extractor, time_emb
            )
            if len(result) == 4:
                cls_logits, pred_bboxes, curr_proposals, _ = result
            else:
                cls_logits, pred_bboxes, curr_proposals = result
            inter_cls_logits.append(cls_logits)

            # CCBR: 级联间 renewal (除最后一级)
            if k < len(self.head_series) - 1:
                curr_scores = torch.sigmoid(cls_logits).max(-1)[0]

                if prev_head6_scores is not None:
                    # 融合置信度
                    fused_conf = (
                        self.ccbr_lambda * curr_scores
                        + (1 - self.ccbr_lambda) * prev_head6_scores
                    )
                    # 自适应阈值
                    mean_diff = (
                        prev_head6_scores.mean() - curr_scores.mean()
                    ).item()
                    threshold = self._sampler.score_thr * (
                        1 + self.ccbr_beta * mean_diff
                    )
                    threshold = max(min(threshold, 0.5), 0.01)
                else:
                    fused_conf = curr_scores
                    threshold = self._sampler.score_thr

                curr_bboxes = self._sampler.apply_inter_head_renewal(
                    pred_bboxes, fused_conf, threshold,
                    alpha=self.ccbr_alpha,
                    sigma_scale=self.ccbr_sigma,
                )
            else:
                curr_bboxes = pred_bboxes

            # 保持 cascade_detach (CCBR 核心: 不改训练动态)
            if self.cascade_detach:
                curr_bboxes = curr_bboxes.detach()

        cls_logits_last = inter_cls_logits[-1]
        pred_bboxes_last = pred_bboxes
        x0 = self._sampler.xyxy_to_raw(pred_bboxes_last, img_metas)

        return cls_logits_last, pred_bboxes_last, x0, inter_cls_logits
```

## 附录 B: Phase 0 诊断脚本伪代码

```python
def diagnose_ccbr_potential(model, dataloader, num_images=100):
    """诊断 CCBR 的潜在收益空间"""
    model.eval()
    stats = {
        'head1_low_head6_high': 0,  # 错杀: Head1 低置信但 Head6 高置信
        'total_proposals': 0,
        'head6_scores': [],
        'head1_scores': [],
    }

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= num_images:
                break
            features = batch['features']
            img_metas = batch['img_metas']
            x_raw = torch.randn(1, model.num_proposals, 4)

            # 收集所有 Head 的 cls_logits
            curr_bboxes = model._sampler.raw_to_xyxy(x_raw, img_metas)
            t_input = torch.full((1,), 0.5 * model.timesteps)
            time_emb = model.time_mlp(t_input)

            all_cls = []
            for head in model.head_series:
                cls, pred, _ = head(features, curr_bboxes, None,
                                    model.roi_extractor, time_emb)
                all_cls.append(cls)
                curr_bboxes = pred.detach()

            head1_scores = torch.sigmoid(all_cls[0]).max(-1)[0]
            head6_scores = torch.sigmoid(all_cls[-1]).max(-1)[0]

            # 统计错杀比例
            head1_low = head1_scores < 0.05   # Head1 低置信
            head6_high = head6_scores > 0.5   # Head6 高置信
            wrong_kill = (head1_low & head6_high).sum().item()

            stats['head1_low_head6_high'] += wrong_kill
            stats['total_proposals'] += head1_scores.numel()
            stats['head6_scores'].append(head6_scores.cpu())
            stats['head1_scores'].append(head1_scores.cpu())

    wrong_kill_ratio = stats['head1_low_head6_high'] / stats['total_proposals']
    print(f"错杀比例 (Head1 低 & Head6 高): {wrong_kill_ratio:.2%}")
    print(f"  → {'CCBR 有空间' if wrong_kill_ratio > 0.05 else 'CCBR 空间有限'}")

    # 相邻时间步 Head6 置信度相关性
    # (需多次采样, 此处省略)
    return stats
```
