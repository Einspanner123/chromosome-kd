# 方向 T：CASS (Class-Aware Stochastic Sampling, 类别感知随机采样)

> **核心命题**: 不同染色体类别因尺度/形态/分类难度差异显著, 应当使用**不同的扩散采样策略**(t 分布、步数分配、自适应推理), 而非当前的"全局统一 shifted schedule"。
>
> **基线**: a3_full_sota, mAP=0.858 (RF + Heun 4步 + AdaLN-Zero + StochasticOT ε=5, 24obj amnal)
>
> **代码位置**:
> - [ldmdet/diffusion/rectified_flow.py](../../../ldmdet/diffusion/rectified_flow.py) — RF 前向扩散与采样器
> - [ldmdet/core/head.py](../../../ldmdet/core/head.py) — `_sample_t` (L415-423), t 采样核心
> - [ldmdet/diffusion/sampling.py](../../../ldmdet/diffusion/sampling.py) — `build_time_pairs` (L59-85), 推理时间网格
> - [experiments/configs/ldmdet/ldmdet_rf_heun_shifted_bs2.py](../../../experiments/configs/ldmdet/ldmdet_rf_heun_shifted_bs2.py) — shifted 配置 (s=3.0)

---

## 1. 背景与动机

### 1.1 染色体尺度的天然层级

人类染色体按 Denver 分类分为 7 组, 尺度跨度约 5:1:

| 组 | 类别 | 物理尺度 | 形态特征 | 检测难度 |
|----|------|----------|----------|----------|
| A (1-3) | A1, A2, A3 | 7-8 μm | 最大, metacentric | 易 |
| B (4-5) | B4, B5 | 5-6 μm | 大, 含 short arm | 中 |
| C (6-12, X) | C6-C12, X | 4-5 μm | 中等, **组内高度相似** | **最难分类** |
| D (13-15) | D13, D14, D15 | 3-4 μm | 中等, acrocentric | 中 |
| E (16-18) | E16, E17, E18 | 3 μm | 较小 | 中 |
| F (19-20) | F19, F20 | ~2 μm | 最小 metacentric | 难 |
| G (21-22, Y) | G21, G22, Y | ~1.5 μm | 最小 acrocentric | **最难检测** |

### 1.2 Per-class AP 差异的实证

基于 SOTA checkpoint (chromo 数据集, mAP=0.748) 的 per-class AP 分析:

| 组 | 代表类 | AP 范围 | 组均值 | 与全局均值差 |
|----|--------|---------|--------|-------------|
| A | A1(0.785), A2(0.809), A3(0.805) | 0.785-0.809 | **0.800** | +0.052 |
| B | B4(0.813), B5(0.777) | 0.777-0.813 | **0.795** | +0.047 |
| C | C6-C12, X | 0.750-0.793 | **0.775** | +0.027 |
| D | D13(0.712), D14(0.736), D15(0.725) | 0.712-0.736 | **0.724** | -0.024 |
| E | E16(0.731), E17(0.723), E18(0.761) | 0.723-0.761 | **0.738** | -0.010 |
| F | F19(0.729), F20(0.717) | 0.717-0.729 | **0.723** | -0.025 |
| G | G21(0.691), G22(0.617) | 0.617-0.691 | **0.654** | -0.094 |
| Y | Y(0.615) | 0.615 | **0.615** | -0.133 |

**关键观察**:

1. **AP 极差 0.194**: 最优类 A2(0.809) 与最差类 Y(0.615) 的 AP 差距高达 19.4 个百分点。
2. **尺度-AP 相关性非线性**: A→G 的尺度递减对应 AP 递减, 但 D 组(中等尺度) AP=0.724 低于 E 组(更小尺度) AP=0.738 — **尺度不是唯一因素**。
3. **C 组的"分类瓶颈"**: C 组尺度中等但 7 类高度相似, C9=0.750 是 C 组最低, 体现的是**分类难度**而非尺度难度。
4. **Y/G22 的双重困境**: 既是尺度最小, 又是形态相似(acrocentric), AP 最低。

### 1.3 现有 shifted schedule 的"一刀切"问题

当前 SOTA 使用全局 shifted schedule (`rf_shift=3.0`):

```python
# head.py L415-423
def _sample_t(self, bs, device):
    t = torch.rand((bs,), device=device)       # ← per-image, 非 per-proposal!
    if self.rf_schedule == 'shifted':
        t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
    return t
```

```python
# sampling.py L78-80 (推理时间网格)
elif self.rf_schedule == 'shifted':
    s = self.rf_shift
    times = s * times / (1 + (s - 1) * times)   # ← 同一 s=3.0 用于所有类别
```

**问题**: 无论 A1(最大) 还是 G22(最小), 训练时 t 采样分布完全相同, 推理时采样步数也完全相同。这忽略了:
- 小目标需要更多高 t (噪声端) 训练以学会从强噪声中恢复
- 大目标信噪比高, 少步即可收敛, 过多步数浪费计算
- C 组相似类别需要更多低 t (数据端) 精细化训练以区分细节

---

## 2. 理论依据

### 2.1 RF 中的 SNR 与目标尺度的关系

在 Rectified Flow 中, 前向扩散为:

$$x_t = (1-t) x_0 + t \cdot \varepsilon, \quad t \in [0, 1]$$

其中 $x_0$ 是数据(GT 框, 已归一化到 $[-\text{snr\_scale}, \text{snr\_scale}]$), $\varepsilon \sim \mathcal{N}(0, I)$。

定义信噪比:

$$\text{SNR}(t) = \frac{(1-t)^2 \|x_0\|^2}{t^2 \|\varepsilon\|^2} = \frac{(1-t)^2}{t^2} \cdot \frac{\|x_0\|^2}{\sigma_\varepsilon^2}$$

由于 $x_0$ 被归一化到 $[-2, 2]$ (snr_scale=2.0), $\|x_0\| \approx \text{snr\_scale} = 2$, $\sigma_\varepsilon = 1$:

$$\text{SNR}(t) \approx 4 \cdot \frac{(1-t)^2}{t^2}$$

**关键洞察**: 在 raw 空间中, SNR 与类别无关(所有框归一化到同一范围)。但**检测难度是尺度相关的**:

- 原始图像空间中, 一个 G 组框(~1.5μm)和 A 组框(~8μm)的尺度比约 5:1
- 同样的 raw 空间定位误差 $\delta$, 在图像空间对应 $\delta \cdot \text{img\_size} / \text{snr\_scale}$ 像素
- 对于 G 组框(约 30 像素), 2 像素误差 = 6.7% 相对误差; 对 A 组框(约 150 像素), 2 像素误差 = 1.3% 相对误差
- **小目标需要更高的有效 SNR 才能达到相同的相对定位精度**

定义"有效检测 SNR" (effective detection SNR):

$$\text{SNR}_{\text{det}}(t, c) = \text{SNR}(t) \cdot \left(\frac{r_c}{r_{\text{ref}}}\right)^2$$

其中 $r_c$ 是类别 $c$ 的特征尺度, $r_{\text{ref}}$ 是参考尺度。小目标 $r_c$ 小, $\text{SNR}_{\text{det}}$ 低, 需要更低 $t$(更高 SNR)才能可靠检测。

### 2.2 Shifted Schedule 的信息论分析

当前 shifted schedule 将均匀分布 $t \sim U[0,1]$ 变换为:

$$t' = \frac{s \cdot t}{1 + (s-1) \cdot t}, \quad s = 3.0$$

逆变换: $t = \frac{t'}{s - (s-1) \cdot t'}$, Jacobian: $\frac{dt}{dt'} = \frac{s}{(s - (s-1) \cdot t')^2}$

变换后 PDF:

$$p(t') = \frac{s}{(s - (s-1) \cdot t')^2}, \quad t' \in [0, 1]$$

| $t'$ | $p(t')$ (s=3) | 相对均匀分布 |
|-------|---------------|-------------|
| 0.0 | 0.333 | 0.33× |
| 0.5 | 0.750 | 0.75× |
| 1.0 | 3.000 | 3.00× |

**shifted schedule 的效果**: $s > 1$ 使 $t$ 分布偏向高值(噪声端), 模型在训练时更多面对高噪声状态。这是**全局有益**的, 因为高噪声状态更难学习, 需要更多训练样本。

**但问题是**: 这个偏移量 $s=3.0$ 对所有类别一视同仁。从 $\text{SNR}_{\text{det}}$ 的角度看:
- A 组: $\text{SNR}_{\text{det}}$ 本身就高, $s=3.0$ 的高 $t$ 训练是"锦上添花"
- G 组: $\text{SNR}_{\text{det}}$ 低, $s=3.0$ 可能"不够", 需要 $s=5.0$ 甚至更高
- C 组: $\text{SNR}_{\text{det}}$ 中等, 但分类难度高, 需要更多低 $t$ (数据端)训练

### 2.3 多尺度扩散理论

在图像生成领域, 多尺度扩散已有成熟理论:

1. **PAB (Pyramid Attention Broadcast)**: 不同分辨率层使用不同步数, 高频细节需要更多步
2. **DeepCache**: 浅层(U-Net early layers)计算频率低于深层
3. **SDXL 的多分辨率训练**: 不同分辨率用不同噪声调度

在检测领域, DiffusionDet 已验证不同尺度目标有不同最优步数:
- APs (小目标) 从 1→8 步提升 +1.6% (0.520→0.525, DPM-Solver++ 实验)
- APl (大目标) 从 1→8 步反而下降 -0.9% (0.642→0.633)
- **这直接证明: 大小目标的最优步数不同**

CASS 将这一观察从**尺度维度**推广到**类别维度**, 不仅考虑尺度, 还考虑形态/分类难度。

### 2.4 类别感知 t 采样的最优性

设类别 $c$ 的最优 t 分布为 $p_c^*(t)$, 满足:

$$p_c^*(t) = \arg\min_{p} \mathbb{E}_{t \sim p} \left[ \mathcal{L}_{\text{det}}(t, c) + \lambda \mathcal{L}_{\text{vel}}(t, c) \right]$$

其中 $\mathcal{L}_{\text{det}}(t, c)$ 是类别 $c$ 在时间 $t$ 的检测损失, $\mathcal{L}_{\text{vel}}(t, c)$ 是速度损失。

**启发式推导**:
- 小目标在高 $t$ 时 $\text{SNR}_{\text{det}}$ 极低, 检测损失大, **梯度信号强**, 应多采样
- 大目标在高 $t$ 时仍可检测, 但低 $t$ 的精细化对定位更重要
- C 组在中等 $t$ 时需学会区分类间细节, 中 $t$ 区域应加密

因此 $p_c^*(t)$ 应满足:
- G/Y 组: $p_c(t)$ 偏向高 $t$ (类似 $s_c > 3.0$)
- A/B 组: $p_c(t)$ 更均匀甚至偏向低 $t$ (类似 $s_c < 3.0$)
- C/D/E/F 组: $p_c(t)$ 适中 (类似 $s_c \approx 3.0$)

---

## 3. 详细方案设计

### 3.1 类别感知 t 采样分布 (训练)

#### 3.1.1 设计: 分组 shifted schedule

将 24 类按 Denver 分组, 每组使用不同的 shift 参数 $s_g$:

```python
# 分组 shift 参数 (启发式初值, 需实验调优)
GROUP_SHIFTS = {
    'A': 2.0,   # 大目标, SNR 高, 少高-t 训练
    'B': 2.5,   # 大目标
    'C': 3.0,   # 中等, 分类难, 保持当前
    'D': 3.0,   # 中等
    'E': 3.5,   # 较小
    'F': 4.0,   # 小目标, 需更多高-t 训练
    'G': 5.0,   # 最小, 最多高-t 训练
    'Y': 5.0,   # 最小 + 形态难
}
```

#### 3.1.2 Per-proposal t 采样

当前架构中, t 是 per-image 采样 (`t = torch.rand((bs,))`)。CASS 需要 per-proposal 采样, 因为同一图像内有不同类别的 GT:

```python
def _sample_t_cass(self, bs, num_proposals, matched_labels, device):
    """类别感知 per-proposal t 采样
    
    Args:
        bs: batch size
        num_proposals: proposal 数量 (500)
        matched_labels: [bs, num_proposals] 每个 proposal 匹配的 GT 类别
        device: torch device
    
    Returns:
        t: [bs, num_proposals] per-proposal 时间步
    """
    t_base = torch.rand((bs, num_proposals), device=device)
    
    # 根据 matched_labels 查表得到每个 proposal 的 shift
    # label 0-23 → group shift
    shifts = torch.zeros_like(t_base)
    for label_id, shift in enumerate(self.label_to_shift):
        shifts[matched_labels == label_id] = shift
    
    # 应用 per-proposal shifted schedule
    t = shifts * t_base / (1 + (shifts - 1) * t_base)
    return t
```

#### 3.1.3 q_sample 的兼容性

现有 `q_sample` 已支持 per-sample t:

```python
# rectified_flow.py L37-38
t_view = t.view(-1, *([1] * (x_start.dim() - 1)))  # 支持 (N, 1) 广播
x_t = (1.0 - t_view) * x_start + t_view * x_noise
```

当 t 从 (bs, 1) 变为 (bs, num_proposals, 1) 时, 广播机制天然兼容, **无需修改 q_sample**。

#### 3.1.4 关键设计决策: 保持线性路径

**CASS 不改变 RF 的前向过程**, 仍然是 $x_t = (1-t)x_0 + t \cdot \varepsilon$。改变的只是 **t 的采样分布**:

| 方面 | ScaleConditionedRF (已证伪) | CASS (本方案) |
|------|---------------------------|--------------|
| 前向过程 | $x_t = (1-t_{\text{eff}})x_0 + t_{\text{eff}} \varepsilon$, $t_{\text{eff}} = t^{1/\kappa(s)}$ | $x_t = (1-t)x_0 + t \varepsilon$ (标准) |
| 时间条件 | 模型看到的 t 与实际噪声水平**不一致** | 模型看到的 t 与实际噪声水平**一致** |
| 路径形状 | 非线性 (尺度依赖) | 线性 (尺度无关) |
| 推理 | 需要知道尺度计算 $t_{\text{eff}}$ | 标准 t 网格, 无需额外信息 |

**这是 CASS 区别于 ScaleConditionedRF 的核心理论优势**: CASS 只改变"在哪些 t 值上训练更多", 不改变"t 值对应的噪声水平"。模型的时间条件始终是正确的。

### 3.2 类别感知步数分配 (推理)

#### 3.2.1 分组步数策略

推理时, 不同尺度类别使用不同采样步数:

```python
# 分组步数策略
GROUP_STEPS = {
    'A': 2,   # 大目标, 2 步足够
    'B': 2,
    'C': 4,   # 中等, 分类需精细
    'D': 3,
    'E': 4,
    'F': 6,   # 小目标, 需更多步
    'G': 8,   # 最小, 最多步
    'Y': 8,
}
```

#### 3.2.2 实现: 分组时间网格

当前 `build_time_pairs` 生成统一时间网格。CASS 需要为不同类别生成不同网格:

```python
def build_class_aware_time_pairs(self, device, num_classes=24):
    """构建类别感知时间网格
    
    Returns:
        time_pairs_per_class: List[List[Tuple[float, float]]], 
                              per-class 时间对列表
    """
    time_pairs_per_class = []
    for c in range(num_classes):
        group = self.label_to_group[c]
        steps = GROUP_STEPS[group]
        shift = GROUP_SHIFTS[group]
        
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        times = shift * times / (1 + (shift - 1) * times)
        
        pairs = [(times[i].item(), times[i+1].item()) 
                 for i in range(len(times) - 1)]
        time_pairs_per_class.append(pairs)
    return time_pairs_per_class
```

#### 3.2.3 挑战: 推理时的类别未知性

推理时我们不知道每个 proposal 的真实类别(正是要预测的)。解决方案见 3.3。

### 3.3 自适应推理策略

#### 3.3.1 两阶段自适应推理

```
阶段 1: 快速粗预测 (2步, 全局 shifted schedule)
    ↓ 预测每个 proposal 的类别
阶段 2: 按预测类别分组, 对小目标/难类 proposal 增加采样步数
    ↓ 
最终输出
```

**具体实现**:

```python
@torch.no_grad()
def predict_adaptive(self, features, img_metas, rescale=True):
    device = features[0].device
    bs = len(img_metas)
    
    # === 阶段 1: 快速粗预测 (2步) ===
    x_raw = torch.randn(bs, self.num_proposals, 4, device=device)
    coarse_time_pairs = self._build_coarse_time_pairs(device)  # 2步
    
    for t_curr, t_next in coarse_time_pairs:
        cls_logits, pred_bboxes, x0_raw, _ = self._forward_at_t(
            features, x_raw, t_curr, img_metas
        )
        x_raw = self.rf.heun_step(x_raw, x0_raw, t_curr, t_next, ...)
    
    # 获取粗预测类别
    coarse_scores = torch.sigmoid(cls_logits)  # [bs, N, num_classes]
    coarse_labels = coarse_scores.argmax(-1)   # [bs, N]
    coarse_confs = coarse_scores.max(-1)[0]    # [bs, N]
    
    # === 阶段 2: 按预测类别分组精化 ===
    # 对高置信度的小目标/难类 proposal 增加步数
    refine_mask = self._needs_refine(coarse_labels, coarse_confs)
    
    if refine_mask.any():
        x_raw_refine = x_raw.clone()
        refine_labels = coarse_labels[refine_mask]
        
        # 按组分别精化 (不同组用不同步数)
        for group in ['F', 'G', 'Y', 'C']:  # 需精化的组
            group_mask = self._get_group_mask(refine_labels, group)
            if group_mask.sum() == 0:
                continue
            
            steps = GROUP_STEPS[group]
            shift = GROUP_SHIFTS[group]
            time_pairs = self._build_group_time_pairs(steps, shift, device)
            
            # 从粗预测点继续精化 (而非从噪声重新开始)
            for t_curr, t_next in time_pairs:
                # 只对 group_mask 内的 proposal 做前向
                x_raw_refine[mask] = self._refine_step(
                    features, x_raw_refine[mask], t_curr, t_next, img_metas
                )
        
        # 合并: 用精化结果替换需要精化的 proposal
        x_raw[refine_mask] = x_raw_refine[refine_mask]
    
    # 最终后处理
    results = self._sampler.post_process(
        [(cls_logits, pred_bboxes)], img_metas, rescale
    )
    return results
```

#### 3.3.2 计算量分析

| 策略 | 总 NFE (500 proposals) | 说明 |
|------|----------------------|------|
| 当前 (Heun 4步) | 8 | 全局 4 步 × 2 (Heun) |
| CASS 粗预测 | 4 | 2 步 × 2 |
| CASS 精化 (仅 ~30% proposals) | 2-6 | 按 group 分配, 加权平均 ~4 |
| **CASS 总计** | **~8** | 与当前相当, 但小目标获得更多步 |

**关键**: CASS 的总计算量与当前持平, 但通过**非均匀分配**让难类获得更多计算资源。

---

## 4. 深入可行性分析

### 4.1 与现有 shifted schedule 的关系

#### 4.1.1 数学等价性分析

当前 shifted schedule 对所有类别使用 $s = 3.0$:

$$t' = \frac{3t}{1 + 2t}$$

CASS 对类别组 $g$ 使用 $s_g$:

$$t'_g = \frac{s_g \cdot t}{1 + (s_g - 1) \cdot t}$$

**当 $s_g = s$ 时, CASS 退化为当前 shifted schedule**。因此 CASS 是 shifted schedule 的**严格推广**, 不引入新的数学结构, 仅解除"全局统一 $s$"的约束。

#### 4.1.2 与 ScaleConditionedRF 的本质区别

ScaleConditionedRF (E4.1, 已证伪, mAP=0.743) 使用:

$$t_{\text{eff}} = t^{1/\kappa(s)}, \quad \kappa(s) = 1 + \lambda \cdot \frac{s_{\max} - s}{s_{\max}}$$

其中 $s = \sqrt{\text{area}(x_0)}$ 是框尺度。

**关键区别**:

| 维度 | ScaleConditionedRF | CASS |
|------|-------------------|------|
| 变换对象 | $t \to t_{\text{eff}}$ (改变**实际噪声水平**) | $t \to t'$ (改变**采样分布**, 不改变噪声水平) |
| 前向过程 | $x_t = (1-t_{\text{eff}})x_0 + t_{\text{eff}} \varepsilon$ (尺度依赖) | $x_t = (1-t')x_0 + t' \varepsilon$ (标准) |
| 时间条件 | $\text{time\_emb}(t)$ 对应 $t_{\text{eff}}$ 的噪声, **不一致** | $\text{time\_emb}(t')$ 对应 $t'$ 的噪声, **一致** |
| 推理 | 需要知道 $s$ 计算 $t_{\text{eff}}$, 但推理时 $s$ 未知 | 标准 $t$ 网格, 无需额外信息 |
| 失败原因 | 时间条件与实际噪声不匹配, 模型学到错误的 (t, 噪声) 映射 | 不适用 (CASS 不改变映射) |

**数学证明 CASS 的时间一致性**:

在 CASS 中, 前向过程为 $x_{t'} = (1-t')x_0 + t' \varepsilon$, 模型接收 $\text{time\_emb}(t')$。

模型学到的映射: $f_\theta(x_{t'}, t') \to x_0$。

由于 $t'$ 与 $x_{t'}$ 的噪声水平严格对应 ($(1-t')$ 的信号比例, $t'$ 的噪声比例), 模型的时间条件始终正确。

而在 ScaleConditionedRF 中, 前向过程为 $x_t = (1-t_{\text{eff}})x_0 + t_{\text{eff}} \varepsilon$, 但模型接收 $\text{time\_emb}(t)$ (名义时间, 非有效时间)。

模型学到的映射: $f_\theta(x_t, t) \to x_0$, 但 $x_t$ 的实际噪声水平由 $t_{\text{eff}}$ 决定, 与 $t$ 不一致。

当 $\kappa \neq 1$ 时, $t_{\text{eff}} \neq t$, 模型面对:
- 相同 $t$, 不同噪声水平 (不同尺度的框)
- 相同噪声水平, 不同 $t$ (不同尺度的框)

这破坏了扩散模型的核心假设: **时间条件唯一确定噪声水平**。

#### 4.1.3 CASS 的信息论优势

从信息论角度, CASS 改变的是训练数据的**分布**, 而非**映射函数**:

$$\min_\theta \mathbb{E}_{t \sim p(t)} \mathbb{E}_{(x_0, \varepsilon)} \left[ \|f_\theta(x_t, t) - x_0\|^2 \right]$$

- ScaleConditionedRF 改变 $x_t$ 的构造 (改变 $t \to t_{\text{eff}}$), 使模型学习不同的映射
- CASS 改变 $p(t)$ (采样分布), 模型仍学习标准映射, 但在**特定 $t$ 区域获得更多训练样本**

CASS 的效果类似于**重要性采样**: 在难学的时间区域多采样, 提高模型在这些区域的精度, 但不改变模型学到的函数族。

### 4.2 训练-推理一致性

#### 4.2.1 训练 t 分布 vs 推理 t 网格

| 阶段 | t 来源 | 分布 |
|------|--------|------|
| 训练 | `_sample_t` | 连续, per-proposal, 类别感知 shifted |
| 推理 | `build_time_pairs` | 离散, per-group, 类别感知 shifted |

**一致性分析**:

1. **分布形状一致**: 训练和推理都使用 shifted schedule, 形状参数 $s_g$ 可对齐
2. **离散化误差**: 推理用离散网格逼近连续分布, 这是标准扩散的固有误差, 非新问题
3. **per-proposal vs per-group**: 训练是 per-proposal (每个 proposal 有自己的 t), 推理是 per-group (同组 proposal 共享时间网格)。这是**可接受的近似**, 因为同组类别尺度相近, 最优步数也相近

#### 4.2.2 与标准扩散的训练-推理差异对比

标准扩散本身就有训练-推理差异:
- 训练: $t \sim U[0, T]$ (连续)
- 推理: $t \in \{T, T-\Delta, ..., 0\}$ (离散网格)

CASS 不引入新的差异类型, 仅仅是让这个差异变成**类别条件化的**。

#### 4.2.3 自适应推理的两阶段一致性

两阶段推理中, 阶段 1 的粗预测使用 2 步, 阶段 2 的精化使用额外 2-6 步。

**潜在问题**: 阶段 1 的 2 步预测可能不准确, 导致错误的类别分组, 进而错误的步数分配。

**缓解**: 
- 阶段 1 只用于区分"大目标 vs 小目标", 这是低精度任务, 2 步足够
- 不依赖细粒度类别区分 (如 C6 vs C7), 只需区分组 (A vs G)

### 4.3 工程复杂度

#### 4.3.1 Per-proposal time_emb 的内存开销

当前 time_emb: `(bs, feat_channels)` = `(2, 256)` = 512 floats = 2KB

CASS time_emb: `(bs, num_proposals, feat_channels)` = `(2, 500, 256)` = 256K floats = 1MB

**内存增量**: +1MB per head, 6 heads = +6MB。相比 RoI 特征 (bs × 500 × 256 × 7×7 ≈ 12GB), **可忽略**。

#### 4.3.2 代码改动评估

| 文件 | 改动 | 复杂度 | 风险 |
|------|------|--------|------|
| `head.py` `_sample_t` | per-image → per-proposal, 添加类别查表 | 低 | 低 |
| `head.py` `_build_training_targets` | 传递 per-proposal t 到 q_sample | 低 | 低 |
| `head.py` `forward` | time_emb 从 (bs, C) → (bs, N, C) | 中 | 中 |
| `single_head.py` `_forward_adaln_zero` | 移除 `repeat_interleave`, 改用 reshape | 中 | 中 |
| `single_head.py` `_forward_scale_shift` | 同上 | 中 | 中 |
| `sampling.py` `build_time_pairs` | 添加 per-group 时间网格 | 低 | 低 |
| `head.py` `predict` | 添加两阶段自适应推理 | 高 | 高 |
| 新增配置文件 | 分组参数 | 低 | 低 |

**关键改动**: `single_head.py` 的 `_forward_adaln_zero` (L321-346):

```python
# 当前 (per-image time_emb):
adaln_params = self.adaln_mlp(time_emb)           # (bs, C')
adaln_params = torch.repeat_interleave(adaln_params, num_boxes, dim=0)  # (bs*N, C')

# CASS (per-proposal time_emb):
adaln_params = self.adaln_mlp(time_emb)           # (bs, N, C')
adaln_params = adaln_params.reshape(bs * num_boxes, -1)  # (bs*N, C')
```

这是一个**等价替换**: 当所有 proposal 的 time_emb 相同时, 两种写法结果一致; 当 time_emb 不同时, 只有 per-proposal 写法正确。

#### 4.3.3 与现有组件的兼容性

| 组件 | 兼容性 | 说明 |
|------|--------|------|
| StochasticOT ε=5 | ✅ 兼容 | OT 耦合在 q_sample 之前, 不受 t 采样影响 |
| AdaLN-Zero | ✅ 兼容 | 见 4.3.2, 改动等价 |
| Heun/DPM-Solver++ | ✅ 兼容 | 推理 solver 不变, 只是时间网格不同 |
| box_renewal | ✅ 兼容 | 不依赖 t |
| Cascade 6 heads | ✅ 兼容 | 每 head 独立处理, time_emb 透传 |
| AMP (bfloat16) | ✅ 兼容 | per-proposal 不影响精度策略 |

---

## 5. 风险评估

### 风险 1: Per-proposal t 导致训练不稳定

**描述**: 当前所有 proposal 共享同一 t, 梯度方向一致。Per-proposal t 后, 不同 proposal 有不同 t, 梯度方向可能冲突, 导致训练不稳定。

**严重性**: 中

**依据**: ScaleConditionedRF 的训练波动达 ±4.5% (E4.3), 部分原因可能是 per-proposal 有效时间不同。虽然 CASS 不改变前向过程, 但 per-proposal t 仍可能导致 head 内不同 proposal 的损失梯度方向不一致。

**缓解方案**:
1. **渐进式引入**: 先用 per-group t (同组 proposal 共享 t), 验证稳定性后再 per-proposal
2. **梯度裁剪**: 当前已有 `clip_grad=dict(max_norm=1.0)`, 可收紧到 0.5
3. **PCGrad**: 如不稳定, 启用 PCGrad 处理不同 proposal 间的梯度冲突
4. **预热**: 前 10 epoch 使用 per-image t (当前方式), 之后切换到 per-proposal

### 风险 2: 自适应推理的粗预测错误传播

**描述**: 两阶段推理中, 阶段 1 的 2 步粗预测可能将大目标误判为小目标 (或反之), 导致错误的步数分配, 反而降低精度。

**严重性**: 中

**依据**: 2 步采样时 mAP 约 0.740 (vs 4 步 0.753), 下降 1.3%。在 24obj 数据集上, 2 步可能不足以可靠区分 A 组和 G 组。

**缓解方案**:
1. **保守分组**: 阶段 1 只区分"明确大"(置信度 > 0.9)和"明确小", 中间区域用默认步数
2. **回退机制**: 如果阶段 1 置信度低于阈值, 使用全局 4 步
3. **尺度先验**: 除了类别, 还可以用预测框的面积作为辅助判断 (小面积 → 多步)
4. **集成**: 阶段 1 用多次 2 步采样取平均, 提高粗预测可靠性

### 风险 3: 分组参数 ($s_g$, 步数) 的调优空间过大

**描述**: 7 个组 × 2 个参数 (shift, steps) = 14 个超参数, 网格搜索空间巨大, 难以充分调优。

**严重性**: 中

**依据**: ScaleConditionedRF 仅 2 个参数 (λ, s_max) 都未找到最优值, 14 个参数的搜索难度更高。

**缓解方案**:
1. **参数化约束**: 用连续函数约束 $s_g = f(\text{scale}_g)$, 将 7 个 $s_g$ 压缩为 2 个参数 (斜率, 截距)
2. **迁移已知结果**: 从 DPM-Solver++ 实验的 APs/APl 步数敏感性数据校准初始值
3. **渐进式调优**: 先固定步数, 只调 shift; 再固定 shift, 调步数
4. **贝叶斯优化**: 用 BO 替代网格搜索, 14 参数下 BO 效率远高于网格

### 风险 4: 与 StochasticOT 的交互效应未知

**描述**: StochasticOT ε=5 的耦合策略可能与 per-proposal t 产生未知的交互效应。OT 耦合在 t 采样之后执行, 不同的 t 可能影响 OT 的最优配对。

**严重性**: 低-中

**依据**: 当前 OT 耦合在 `_build_training_targets` 中执行, 此时 t 已确定。Per-proposal t 改变了每个 proposal 的噪声水平, 可能影响 OT 的 cost matrix (如果 cost 依赖 t)。

**缓解方案**:
1. **验证 OT cost 对 t 的依赖**: 检查 `OTFlowCoupling.couple` 是否使用 t, 如果不依赖则无影响
2. **消融实验**: 先在 Random coupling 上验证 CASS, 再叠加 StochasticOT
3. **如果 OT 依赖 t**: 在 OT cost 中考虑 t 的影响, 或固定 t 做 OT

---

## 6. 预期收益分析

### 6.1 基于 per-class AP 差异的定量估计

当前 per-class AP 极差 0.194 (A2=0.809 vs Y=0.615)。假设 CASS 能将低 AP 类别提升到"组均值"水平:

| 组 | 当前 AP | 目标 AP (组内均衡) | 提升 | 类别数 | 加权贡献 |
|----|---------|-------------------|------|--------|---------|
| G | 0.617-0.691 | 0.680 | +0.039 | 2 | +0.0033 |
| Y | 0.615 | 0.650 | +0.035 | 1 | +0.0015 |
| F | 0.717-0.729 | 0.750 | +0.027 | 2 | +0.0023 |
| D | 0.712-0.736 | 0.760 | +0.036 | 3 | +0.0045 |
| C (低) | 0.750-0.767 | 0.780 | +0.022 | 4 | +0.0037 |
| A/B (高) | 0.777-0.813 | 保持 | 0 | 5 | 0 |
| **总计** | | | | **24** | **+0.015** |

**保守估计**: mAP +0.010~0.015 (从 0.858 提升到 0.868~0.873)

**乐观估计**: 如果自适应推理额外提升小目标定位精度, mAP +0.020~0.025

### 6.2 分组收益分析

| 收益来源 | 机制 | 预期 mAP 增量 |
|----------|------|-------------|
| 类别感知 t 采样 (训练) | 小目标获得更多高-t 训练, 提升去噪能力 | +0.005~0.008 |
| 类别感知步数 (推理) | 小目标获得更多采样步, 提升定位精度 | +0.003~0.005 |
| 自适应推理 | 计算资源从大目标转移到小目标 | +0.002~0.003 |
| **合计** | | **+0.010~0.016** |

### 6.3 风险调整后的预期

考虑到 ScaleConditionedRF 失败的先例, 需要打折:

| 场景 | 概率 | mAP 增量 |
|------|------|---------|
| 成功 (超越 SOTA) | 40% | +0.010~0.015 |
| 持平 (±0.003) | 35% | ~0 |
| 失败 (低于 SOTA) | 25% | -0.005~0.010 |

**期望值**: +0.003~0.005 mAP

### 6.4 与其他方向的收益对比

| 方向 | 预期 mAP 增量 | 状态 | 风险 |
|------|-------------|------|------|
| DPM-Solver++ 8步 (已验证) | +0.002 | ✅ 已确认 | 低 |
| CASS (本方案) | +0.003~0.005 | 待验证 | 中 |
| KCEC (已证伪) | -0.009 | ❌ 失败 | — |
| ScaleConditionedRF (已证伪) | -0.010 | ❌ 失败 | — |
| CFM 速度预测 (已证伪) | -0.035 | ❌ 失败 | — |

---

## 7. 实现路线图

### Phase 1: 验证 per-proposal t 的可行性 (1-2 天)

**目标**: 在不改变采样分布的前提下, 验证 per-proposal t 不会破坏训练。

**改动**:
- `head.py`: `_sample_t` 改为 per-proposal, 但 shift 参数统一 (s=3.0)
- `single_head.py`: `_forward_adaln_zero` 和 `_forward_scale_shift` 移除 `repeat_interleave`
- 验证: 训练 10 epoch, mAP 不低于当前同期

**验收标准**: 10 epoch mAP ≥ 0.7 (当前同期约 0.72)

### Phase 2: 类别感知 t 采样 (3-5 天)

**目标**: 实现分组 shifted schedule, 验证训练收益。

**改动**:
- `head.py`: `_sample_t_cass` 实现, 添加 `label_to_shift` 查表
- `head.py`: `_build_training_targets` 传递 per-proposal t 和 matched_labels
- 配置: 新增 `cass_group_shifts` 参数
- 训练: 150 epoch, seed=42, 与 a3_full_sota 对齐

**消融实验**:
| 实验 | shift 配置 | 预期 |
|------|-----------|------|
| E-T0 | 全局 s=3.0 (per-proposal, 退化为 Phase 1) | 基线 |
| E-T1 | 分组 shift (A=2, B=2.5, C=3, D=3, E=3.5, F=4, G=5, Y=5) | +0.005~0.010 |
| E-T2 | 仅小目标 shift (A-G 保持 3.0, G/Y=5.0) | +0.003~0.005 |
| E-T3 | 仅大目标 shift (A/B=2.0, 其余保持 3.0) | 验证大目标是否受益 |

**验收标准**: E-T1 mAP ≥ 0.860 (超 SOTA +0.002)

### Phase 3: 类别感知步数 (3-5 天)

**目标**: 实现推理时分组步数, 验证推理收益。

**改动**:
- `sampling.py`: `build_class_aware_time_pairs` 实现
- `head.py`: `predict` 支持分组时间网格
- 评估: 在 Phase 2 最佳 checkpoint 上, 离线测试不同步数策略

**消融实验**:
| 实验 | 步数配置 | 预期 NFE | 预期 mAP |
|------|---------|---------|---------|
| E-T4 | 全局 4 步 (当前 SOTA) | 8 | 0.858 |
| E-T5 | 分组步数 (A/B=2, C/D=3, E/F=4, G/Y=6) | ~6 | +0.002~0.004 |
| E-T6 | 分组步数 (A/B=2, C/D=4, E/F=6, G/Y=8) | ~8 | +0.003~0.005 |

**验收标准**: E-T5 或 E-T6 mAP ≥ 0.862, 且 NFE ≤ 8

### Phase 4: 自适应推理 (5-7 天)

**目标**: 实现两阶段自适应推理, 验证端到端收益。

**改动**:
- `head.py`: `predict_adaptive` 实现
- `sampling.py`: 添加 `build_coarse_time_pairs` 和 `build_group_time_pairs`
- 评估: 端到端 mAP 和推理时间

**验收标准**: mAP ≥ 0.865, 推理时间 ≤ 当前 1.2×

### 代码改动清单

| 文件 | 改动类型 | 行数估计 | Phase |
|------|---------|---------|-------|
| `ldmdet/core/head.py` | `_sample_t` per-proposal | +30 | 1 |
| `ldmdet/core/head.py` | `_sample_t_cass` 类别感知 | +40 | 2 |
| `ldmdet/core/head.py` | `forward` time_emb 扩展 | +10 | 1 |
| `ldmdet/core/head.py` | `predict_adaptive` | +80 | 4 |
| `ldmdet/core/single_head.py` | `_forward_adaln_zero` | +5/-3 | 1 |
| `ldmdet/core/single_head.py` | `_forward_scale_shift` | +5/-3 | 1 |
| `ldmdet/diffusion/sampling.py` | `build_class_aware_time_pairs` | +30 | 3 |
| `experiments/configs/ldmdet/directions/cass/` | 新配置文件 | +100 | 2-4 |
| **总计** | | **~300** | |

---

## 8. 与已证伪方向的对比

### 8.1 vs ScaleConditionedRF (E4.1, mAP=0.743, -0.010 vs SOTA)

**ScaleConditionedRF 的失败原因**:

1. **时间条件不一致**: $t_{\text{eff}} = t^{1/\kappa(s)}$ 改变了实际噪声水平, 但模型条件化在名义 $t$ 上, 学到错误的 (t, 噪声) 映射
2. **推理时尺度未知**: 推理时没有 GT 框的尺度, 无法计算 $\kappa(s)$, 被迫用近似值
3. **路径非线性**: 不同尺度的框走不同的 ODE 轨迹, RF 的"直线路径"优势丧失
4. **与随机耦合冲突**: 非线性路径 + 随机耦合 = 路径优化更困难

**CASS 如何避免这些问题**:

| 问题 | ScaleConditionedRF | CASS |
|------|-------------------|------|
| 时间条件不一致 | $t_{\text{eff}} \neq t$, 条件错误 | $t' = t'$, 条件正确 (仅改变采样分布) |
| 推理时尺度未知 | 需要知道 $s$ 计算 $t_{\text{eff}}$ | 标准 $t$ 网格, 无需 $s$ |
| 路径非线性 | $t^{1/\kappa}$ 使路径弯曲 | 线性路径 $x_t = (1-t)x_0 + t\varepsilon$ 保持 |
| 与耦合冲突 | 非线性路径干扰 OT 配对 | 线性路径, OT 配对不受影响 |

**核心区别**: ScaleConditionedRF 改变**扩散过程本身**(前向 ODE), CASS 只改变**训练数据的采样分布**(哪些 $t$ 值被更频繁地采样)。前者破坏了 RF 的数学结构, 后者只是重要性采样。

### 8.2 vs CFM 速度预测 (mAP=0.823, -0.035 vs SOTA)

**CFM 的失败原因**:

1. **优化目标改变**: 从预测 $x_0$ 变为预测速度 $v = \varepsilon - x_0$, 改变了损失景观
2. **级联 head 的误差累积**: 每级 head 预测速度, 速度误差通过 $x_0 = x_t - t \cdot v$ 传播
3. **与 AdaLN-Zero 不兼容**: AdaLN 的零初始化策略对 $x_0$ 预测优化, 对速度预测效果差

**CASS 的区别**:
- CASS **不改变预测目标**(仍然预测 $x_0$)
- CASS **不改变损失函数**(仍然 L1 + GIoU + Focal)
- CASS **不改变级联结构**(仍然 6 级 head + detach)

CASS 只改变 t 采样分布和推理步数, 是**最小侵入式**的改动。

### 8.3 vs KCEC (mAP=0.744, -0.009 vs SOTA)

**KCEC 的失败原因**:

1. **信息截断**: OT 耦合层的信息(nucleus ploidy)无法传播到 Transformer 特征层
2. **特征层无同源概念**: 模型学不到"这对配对存在是因为 ploidy 约束"
3. **梯度路径断裂**: 耦合策略不在梯度流中

**CASS 的区别**:
- CASS 的信息(t 采样分布)**直接进入梯度流**: 每个 proposal 的 t 影响其 $x_t$, 进而影响损失和梯度
- CASS **不需要模型理解"为什么这个 t 更大"**: 模型只需学会在给定 t 下预测 $x_0$, 更频繁的高 t 训练自然提升高噪声下的精度
- CASS 是**隐式条件化**: 通过数据分布间接引导模型, 而非显式注入先验

---

## 9. 参考文献

1. **Rectified Flow**: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023)
   - RF 直线路径理论, $x_t = (1-t)x_0 + t\varepsilon$

2. **Shifted Schedule**: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3, ICML 2024)
   - $t' = s \cdot t / (1 + (s-1) \cdot t)$, 用于 SD3 的高分辨率生成

3. **DiffusionDet**: Zhang et al., "DiffusionDet: Diffusion Model for Object Detection" (ICCV 2023)
   - 扩散检测框架, 噪声→框的演化

4. **DPM-Solver++**: Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models" (NeurIPS 2022)
   - RF-DPM-Solver++ 推导, 处理 $t \to 0$ 奇点

5. **ScaleConditionedRF (已证伪)**: 项目内部, E4.1 实验
   - $t_{\text{eff}} = t^{1/\kappa(s)}$, mAP=0.743 (-0.010 vs SOTA)
   - 失败原因: 时间条件不一致

6. **SDXL Multi-Resolution**: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis" (ICLR 2024)
   - 多分辨率训练中的尺度感知噪声调度

7. **Importance Sampling for Diffusion**: Kang et al., "GigaGAN: Scaling up GANs for Text-to-Image Synthesis" (ICLR 2024)
   - 重要性采样在扩散训练中的应用

8. **Cascade R-CNN**: Cai & Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection" (CVPR 2018)
   - 多阶段 IoU 递增的思想, 类比 CASS 的步数递增

9. **Denver Classification**: ISCN (International System for Human Cytogenomic Nomenclature)
   - 染色体 Denver 分组标准, A-G 组的尺度/形态差异

10. **Per-class AP Analysis**: 项目内部, `experiments/analysis/per_class_ap.py`
    - SOTA checkpoint 的 per-class AP 数据, 量化类别间 AP 差异

---

## 10. 总结

CASS (Class-Aware Stochastic Sampling) 的核心贡献在于**将扩散采样策略从"全局统一"推广到"类别感知"**, 通过三个层面的改进:

1. **训练端**: per-proposal 类别感知 t 采样分布, 让小目标/难类获得更多高噪声训练
2. **推理端**: per-group 类别感知步数分配, 让小目标获得更多采样步
3. **系统级**: 两阶段自适应推理, 让计算资源从大目标转移到小目标

**与已证伪方向的本质区别**: CASS 不改变 RF 的前向过程(保持线性路径 $x_t = (1-t)x_0 + t\varepsilon$), 不改变预测目标(仍然预测 $x_0$), 不改变损失函数。它只改变"在哪些 t 值上训练更多"和"推理时分配多少步数", 是**最小侵入式**的改动, 避免了 ScaleConditionedRF 的时间条件不一致问题和 CFM 的优化目标改变问题。

**预期收益**: mAP +0.010~0.015 (保守), 风险调整后期望 +0.003~0.005。

**最大风险**: Per-proposal t 导致训练不稳定, 但可通过渐进式引入和梯度裁剪缓解。
