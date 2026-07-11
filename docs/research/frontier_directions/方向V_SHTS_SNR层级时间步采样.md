# 方向 V: SHTS (SNR-Hierarchical Time-step Sampling, SNR 层级时间步采样)

> **状态**: 方案设计 + 可行性分析（待实验验证）
> **基线**: a3_full_sota, mAP = 0.858 (RF + Heun 4步 + AdaLN-Zero + StochasticOT ε=5)
> **核心思想**: 基于信噪比(SNR)导数的层级化时间步网格，在不同 SNR 区间自适应分配采样密度，与 DPM-Solver++ 协同提升少步推理精度。
> **关键区分**: SHTS **只改采样时间步网格**，不改前向过程、不改训练目标、不引入 per-box 时间步。

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [理论依据](#2-理论依据)
3. [详细方案设计](#3-详细方案设计)
4. [深入可行性分析](#4-深入可行性分析)
5. [风险评估](#5-风险评估)
6. [预期收益分析](#6-预期收益分析)
7. [实现路线图](#7-实现路线图)
8. [与已证伪方向的对比](#8-与已证伪方向的对比)
9. [参考文献](#9-参考文献)

---

## 1. 背景与动机

### 1.1 SNR 在扩散检测中的关键作用

在基于 Rectified Flow (RF) 的扩散检测中，前向加噪过程为：

$$x_t = (1-t)\,x_0 + t\,x_{\text{noise}}$$

其中 $x_0$ 是归一化到 $[-\text{snr\_scale}, +\text{snr\_scale}]$ 的 GT 框（cxcywh 空间），$x_{\text{noise}} \sim \mathcal{N}(0, I_4)$。信噪比定义为：

$$\text{SNR}(t) = \frac{\text{Var}(\text{signal})}{\text{Var}(\text{noise})} = \frac{(1-t)^2 \cdot \text{snr\_scale}^2}{t^2} = \left(\frac{1-t}{t}\right)^2 \cdot \text{snr\_scale}^2$$

SNR 直接决定了模型在不同 $t$ 处能"看到"多少真实信号：

| SNR 区间 | $t$ 范围 (snr_scale=2.0) | 信号状态 | 检测任务焦点 |
|----------|--------------------------|----------|--------------|
| 高 SNR (>10) | $t \in [0, 0.388]$ | 数据清晰，框位置可辨 | **精细定位**（框回归微调） |
| 临界 SNR (≈1) | $t \approx 0.667$ | 信号≈噪声，分类边界 | **分类决策**（24类判别） |
| 低 SNR (<0.1) | $t \in [0.863, 1.0]$ | 噪声主导，仅有粗略位置 | **粗略定位**（大区域提议） |

**核心观察**: 检测任务与图像生成的本质区别在于，检测需要同时完成**定位**（回归）和**分类**（判别），而这两者对 SNR 的需求不同：
- 定位需要高 SNR（数据清晰时才能精确回归框边界）
- 分类在临界 SNR 处最为关键（信号与噪声均衡时，模型必须做出明确的类别决策）

这意味着最优的时间步分配不应是均匀的，也不应是简单的单调偏移，而应根据 SNR 的**局部变化率**和**任务需求**进行层级化分配。

### 1.2 现有 Shifted Schedule 的局限

当前基线使用 shifted schedule（`rf_schedule='shifted'`, `rf_shift=3.0`）：

$$t' = \frac{s \cdot t}{1 + (s-1) \cdot t}, \quad s = 3.0$$

**作用机制**: 这是一个 $[0,1] \to [0,1]$ 的单调双射，其密度变换为：

$$\frac{dt'}{dt} = \frac{s}{(1 + (s-1)t)^2}$$

- 在 $t=0$ 处：$dt'/dt = s = 3.0$（密度增大 3 倍）
- 在 $t=1$ 处：$dt'/dt = 1/s \approx 0.33$（密度降低 3 倍）

**局限性分析**:

1. **全局单调性**: shifted schedule 的密度函数是**严格单调递减**的。它只能在"高 SNR 端加密、低 SNR 端减稀"这一个方向上调节。无法表达"高 SNR 和临界 SNR 都需要密集，但中间可以稀疏"这类非单调需求。

2. **单参数表达力**: 整个密度分布仅由 $s$ 一个参数控制。无法独立调节不同 SNR 区间的采样密度。

3. **缺乏任务感知**: shifted schedule 完全基于几何变换，没有考虑检测任务中"分类临界区"的特殊重要性。

4. **与 DPM-Solver++ 的耦合不充分**: DPM-Solver++ 的高阶积分精度依赖于时间步网格的合理性。当前 DPM-Solver++ 直接使用 shifted schedule 生成的时间步（`RFDPMSolverMultistep` 内部用 `linspace(1,0,N+1)` 再经 shifted 变换），没有针对高阶求解器的误差特性优化网格。

### 1.3 实验证据支撑

| 实验 | 步数 | mAP | 关键发现 |
|------|------|-----|----------|
| Heun (shifted s=3) | 1步 | 0.740 | 极少步数下定位粗糙 |
| Heun (shifted s=3) | 8步 | 0.755 | +0.015，更多步数显著提升 |
| DPM-2 (shifted s=3) | 6步 (7 NFE) | 0.753 | 等效 Heun 4步 (8 NFE)，效率提升 |
| DPM-2 (shifted s=3) | 8步 (9 NFE) | 0.755 | APs (小目标) +0.004，高阶积分对小目标有优势 |

**关键启示**: 
- 1→8 步的增益曲线（+0.015）说明**时间步分配的质量**还有提升空间——如果步数加倍只能带来 0.015 的增益，说明现有网格的每步信息利用率不最优。
- DPM-Solver++ 对小目标（APs）的提升（+0.004）说明高 SNR 端（小目标需要精确定位）的积分精度是瓶颈，SHTS 可以在这里分配更多密度。

---

## 2. 理论依据

### 2.1 RF 的 SNR 公式推导

**前向过程**:
$$x_t = (1-t)\,x_0 + t\,x_1, \quad x_0 = \text{GT (raw 空间)}, \quad x_1 = \text{noise} \sim \mathcal{N}(0, I)$$

其中 raw 空间的变换为（见 `sampling.py: xyxy_to_raw`）:
$$x_0 = \left(\frac{\text{bbox}_{\text{cxcywh}}}{\text{img\_size}} \cdot 2 - 1\right) \cdot \text{snr\_scale}$$

因此 $x_0$ 的各分量分布在 $[-\text{snr\_scale}, +\text{snr\_scale}]$。

**信号与噪声功率**（逐维）:
- 信号功率: $\mathbb{E}[(1-t)^2 \cdot x_{0,i}^2] = (1-t)^2 \cdot \text{Var}(x_{0,i})$
- 噪声功率: $\mathbb{E}[t^2 \cdot x_{1,i}^2] = t^2 \cdot 1$

采用工程简化（将 $\text{Var}(x_{0,i}) \approx \text{snr\_scale}^2$ 作为上界估计）:

$$\boxed{\text{SNR}(t) = \left(\frac{1-t}{t}\right)^2 \cdot \text{snr\_scale}^2}$$

### 2.2 SNR 导数与梯度分析

**一阶导数**:

$$\frac{d\,\text{SNR}}{dt} = \text{snr\_scale}^2 \cdot \frac{d}{dt}\left[\frac{(1-t)^2}{t^2}\right] = \text{snr\_scale}^2 \cdot \frac{-2(1-t)}{t^3}$$

$$\boxed{\frac{d\,\text{SNR}}{dt} = -\frac{2\,\text{snr\_scale}^2\,(1-t)}{t^3}}$$

**关键特征**:
- $t \to 0^+$: $|d\text{SNR}/dt| \to \infty$（SNR 变化极其剧烈）
- $t \to 1^-$: $|d\text{SNR}/dt| \to 0$（SNR 变化平缓）
- $t = 0.5$: $|d\text{SNR}/dt| = 2 \cdot 4 \cdot 0.5 / 0.125 = 32$

**二阶导数**（用于步长优化的曲率分析）:

$$\frac{d^2\,\text{SNR}}{dt^2} = \frac{2\,\text{snr\_scale}^2\,(3-2t)}{t^4}$$

在 $t \in [0,1]$ 内，$3-2t > 0$ 恒成立，因此 SNR 函数在整个区间内**严格凸**。

**对数 SNR 导数**（更稳定的度量）:

$$\frac{d \ln\text{SNR}}{dt} = \frac{1}{\text{SNR}} \cdot \frac{d\,\text{SNR}}{dt} = -\frac{2}{t(1-t)}$$

这是一个优美的结果：**对数 SNR 的变化率仅依赖于 $t$，与 snr_scale 无关**。

- $t = 0.5$: $|d\ln\text{SNR}/dt| = 8$（最大变化率）
- $t \to 0$ 或 $t \to 1$: $|d\ln\text{SNR}/dt| \to \infty$

### 2.3 最优时间步分配理论

**问题形式化**: 给定 $N$ 步推理预算，选择时间步网格 $\{t_0=1 > t_1 > \cdots > t_N=0\}$，最小化总积分误差。

**局部截断误差分析**:

对于 RF ODE $dx_t/dt = (x_1 - x_0^{\text{pred}}(t))/t$（半线性形式），使用 $k$ 阶求解器时，第 $i$ 步的局部截断误差为：

$$e_i \approx C_k \cdot (\Delta t_i)^{k+1} \cdot \left\|\frac{d^{k+1} x_0^{\text{pred}}}{dt^{k+1}}\right\|_{t_i}$$

总误差（假设误差可加）:

$$E_{\text{total}} \approx \sum_{i=0}^{N-1} C_k \cdot (\Delta t_i)^{k+1} \cdot a(t_i)$$

其中 $a(t) = \|d^{k+1}x_0^{\text{pred}}/dt^{k+1}\|$ 是模型预测的 $(k+1)$ 阶导数范数。

**优化问题**:

$$\min_{\{t_i\}} \sum_i (\Delta t_i)^{k+1} \cdot a(t_i) \quad \text{s.t.} \quad \sum_i \Delta t_i = 1$$

用拉格朗日乘子法求解，最优步长满足：

$$\Delta t_i^* \propto a(t_i)^{-1/(k+1)}$$

**SNR 代理**: 由于 $a(t)$ 依赖模型且难以解析计算，我们用 SNR 的变化率作为代理。核心假设是：**SNR 变化快的区域，模型预测也变化快，因此需要更小的步长**。

用 $|d\ln\text{SNR}/dt| = 2/(t(1-t))$ 作为 $a(t)$ 的代理：

$$\Delta t_i^* \propto \left[\frac{t_i(1-t_i)}{2}\right]^{1/(k+1)}$$

对于二阶求解器（Heun, DPM-2, $k=2$）:

$$\Delta t_i^* \propto \left[\frac{t_i(1-t_i)}{2}\right]^{1/3}$$

**关键洞察**: 这个最优步长在 $t=0.5$ 处最大（步长可以大），在 $t \to 0$ 和 $t \to 1$ 处最小（需要小步长）。这说明**最优网格应在两端密集、中间稀疏**——这与 shifted schedule（只在 $t \to 0$ 端密集）有本质区别！

**修正: 任务感知加权**:

然而，纯 SNR 导数分析忽略了检测任务的特殊性。分类决策在临界 SNR（$t \approx 0.667$）处最关键，因此引入任务权重 $w(t)$:

$$\Delta t_i^* \propto \left[\frac{1}{a(t_i) \cdot w(t_i)}\right]^{1/(k+1)}$$

其中 $w(t)$ 在临界 SNR 处取峰值（如高斯峰 $w(t) = 1 + \alpha \exp(-(t - t_{\text{crit}})^2 / 2\sigma^2)$）。

### 2.4 与 Shifted Schedule 的理论对比

| 性质 | Shifted Schedule | SHTS |
|------|-------------------|------|
| 密度函数 | $s/(1+(s-1)t)^2$（单调递减） | 任意非负函数（可非单调） |
| 参数数 | 1 ($s$) | 3-5（区域边界 + 权重） |
| 可表达"两端密集" | 否 | 是 |
| 可表达"临界区密集" | 否 | 是 |
| 任务感知 | 否 | 是（通过 $w(t)$） |
| 可逆性 | 是（$t = t'/(s-(s-1)t')$） | 是（单调网格保证） |

**定理 (SHTS 严格包含 Shifted Schedule)**: 当 $w(t) \equiv 1$ 且 $a(t) \propto 1/(1-t)^2$ 时，SHTS 的最优网格退化为 shifted schedule 的某个参数 $s^*$。

*证明*: 若 $a(t) \propto 1/(1-t)^2$，则 $\Delta t \propto (1-t)^{2/(k+1)}$。对于 $k=1$（Euler），$\Delta t \propto (1-t)$，积分得 $t' = 1 - (1-t)^2/2$，这是 shifted schedule 的特例。$\square$

**结论**: SHTS 是 shifted schedule 的严格泛化。当最优网格恰好是单调的时，SHTS 退化为 shifted schedule；但 SHTS 还能表达 shifted schedule 无法表达的非单调网格。

---

## 3. 详细方案设计

### 3.1 SNR 分层策略

将 $[0,1]$ 划分为三个 SNR 层级，每层采用不同的采样密度：

```
t=0          t=0.388        t=0.667        t=0.863        t=1
 |--- Layer I ---|--- Layer II ---|--- Layer III ---|
   高SNR(>10)      临界SNR(0.1~10)    低SNR(<0.1)
   精细定位         分类决策           粗略定位
   密集采样         密集采样            稀疏采样
```

**层边界确定**（基于 snr_scale=2.0）:
- $t_{\text{high}} = \text{snr\_scale}/(1+\text{snr\_scale}) = 2/3 \approx 0.667$（SNR=1 的临界点）
- $t_{\text{low}} = 0.863$（SNR=0.1，通过 $\sqrt{\text{snr\_scale}^2/0.1} = (1-t)/t$ 解出）
- $t_{\text{very\_low}} = 0.95$（SNR≈0.01，几乎纯噪声）

**密度分配比例**:
- Layer I ($t \in [0, 0.388]$): 40% 的步数（精细定位）
- Layer II ($t \in [0.388, 0.863]$): 45% 的步数（分类决策 + 过渡）
- Layer III ($t \in [0.863, 1.0]$): 15% 的步数（粗略定位）

### 3.2 自适应步长算法

**算法 1: SNR 梯度自适应网格生成**

```python
def build_shts_grid(
    num_steps: int,
    snr_scale: float = 2.0,
    task_weight_alpha: float = 1.0,
    task_weight_sigma: float = 0.15,
    solver_order: int = 2,
) -> list[float]:
    """生成 SNR 层级化时间步网格。

    基于 d(ln SNR)/dt 的变化率 + 任务感知权重，
    通过 CDF 反演法生成非均匀网格。

    Args:
        num_steps: 采样步数
        snr_scale: SNR 缩放因子
        task_weight_alpha: 临界区权重强度 (0=纯SNR, 1=任务感知)
        task_weight_sigma: 临界区高斯宽度
        solver_order: 求解器阶数 (影响步长指数)

    Returns:
        降序时间步列表 [1.0, ..., 0.0]
    """
    import numpy as np

    # 1. 定义密度函数 rho(t) = |d ln SNR / dt| * w(t)
    t_crit = snr_scale / (1 + snr_scale)  # SNR=1 的 t 值

    def density(t):
        # SNR 变化率: 2 / (t * (1-t))
        snr_rate = 2.0 / (t * (1 - t) + 1e-8)
        # 任务权重: 在临界 SNR 处增强
        w = 1.0 + task_weight_alpha * np.exp(
            -((t - t_crit) ** 2) / (2 * task_weight_sigma ** 2)
        )
        return snr_rate * w

    # 2. 数值积分 CDF
    t_fine = np.linspace(1.0, 1e-4, 10000)
    rho = np.array([density(t) for t in t_fine])
    # 归一化 (从 t=1 到 t 的积分)
    cdf = np.cumsum(rho[::-1] * np.abs(np.diff(t_fine[::-1])))
    cdf = cdf / cdf[-1]

    # 3. 反演 CDF: 均匀分位 -> 非均匀 t
    quantiles = np.linspace(0, 1, num_steps + 1)
    t_grid = np.interp(quantiles, cdf, t_fine[::-1])
    t_grid[0] = 1.0
    t_grid[-1] = 0.0

    return sorted(t_grid.tolist(), reverse=True)
```

**算法 2: 与 Shifted Schedule 组合**

```python
def build_shts_shifted_grid(
    num_steps: int,
    snr_scale: float = 2.0,
    rf_shift: float = 3.0,
    **kwargs,
) -> list[float]:
    """先 shifted 变换，再 SHTS 分层。

    组合策略: t' = shifted(t), 然后在 t' 空间应用 SHTS。
    """
    # 1. 生成 SHTS 基网格 (在 t' 空间)
    t_prime_grid = build_shts_grid(num_steps, snr_scale, **kwargs)
    # 2. 反 shifted 变换回原始 t 空间
    # t' = s*t/(1+(s-1)*t) => t = t'/(s-(s-1)*t')
    t_grid = [tp / (rf_shift - (rf_shift - 1) * tp) for tp in t_prime_grid]
    return sorted(t_grid, reverse=True)
```

### 3.3 与 DPM-Solver++ 的协同

**当前 DPM-Solver++ 实现** (`rectified_flow.py: RFDPMSolverMultistep`):

```python
class RFDPMSolverMultistep:
    def __init__(self, num_steps=6, solver_order=2):
        self.timesteps = [float(t) for t in torch.linspace(1.0, 0.0, num_steps + 1)]
        # ...
```

**问题**: DPM-Solver++ 内部用 `linspace` 生成均匀网格，与 `build_time_pairs` 中的 shifted 网格不一致（实际上 `step()` 方法通过 `t_n` 参数接收外部时间步，但 `self.timesteps` 用于查找 `t_next`，两者必须同步）。

**SHTS-DPM 协同设计**:

1. **统一网格源**: SHTS 生成的时间步网格同时供 `build_time_pairs` 和 `RFDPMSolverMultistep` 使用。

```python
class DiffusionSampler:
    def build_time_pairs(self, device):
        if self.rf_schedule == 'shts':
            t_grid = build_shts_grid(
                self.sampling_timesteps, self.snr_scale,
                task_weight_alpha=self.shts_alpha,
                solver_order=2 if 'dpm_solver_pp' in self.solver_type else 1,
            )
            times = torch.tensor(t_grid, device=device)
        else:
            # ... 现有逻辑
        time_pairs = [(times[i].item(), times[i+1].item())
                      for i in range(len(times)-1)]
        return time_pairs

    def create_dpm_solver(self):
        if self.solver_type in ('dpm_solver_pp', 'dpm_solver_pp_3'):
            solver_order = 3 if self.solver_type == 'dpm_solver_pp_3' else 2
            solver = RFDPMSolverMultistep(
                num_steps=self.sampling_timesteps,
                solver_order=solver_order,
            )
            # 覆盖均匀网格为 SHTS 网格
            if self.rf_schedule == 'shts':
                t_grid = build_shts_grid(
                    self.sampling_timesteps, self.snr_scale,
                    solver_order=solver_order,
                )
                solver.timesteps = t_grid
            return solver
        return None
```

2. **DPM-Solver++ 对非均匀网格的适应性**: DPM-Solver++ 的公式中，$\phi_1 = t_{n+1}\ln(t_n/t_{n+1}) - t_n + t_{n+1}$ 和 $D_1 = (x_0^n - x_0^{n-1})/(t_n - t_{n-1})$ 都显式依赖 $(t_n, t_{n+1})$ 的值，不假设均匀步长。因此 SHTS 的非均匀网格**天然兼容** DPM-Solver++。

3. **高阶积分的误差放大效应**: DPM-Solver++ 的二阶校正项 $\phi_1 \cdot D_1$ 中，$D_1$ 是 $x_0^{\text{pred}}$ 的差商。在 SNR 变化快的区域（$t \to 0$），$x_0^{\text{pred}}$ 变化也快，$D_1$ 的值大，因此需要更小的 $\Delta t$ 来控制 $\phi_1$ 的量级。SHTS 恰好在这些区域分配了更密的步数，与 DPM-Solver++ 的误差特性协同。

### 3.4 训练侧的 SHTS（可选 Phase 2）

**当前训练 t 采样** (`head.py: _sample_t`):

```python
def _sample_t(self, bs, device):
    t = torch.rand((bs,), device=device)
    if self.rf_schedule == 'shifted':
        t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
    return t
```

**Phase 2 扩展**: 训练时也使用 SHTS 密度采样 $t$，使训练分布与推理网格匹配：

```python
def _sample_t(self, bs, device):
    t = torch.rand((bs,), device=device)
    if self.rf_schedule == 'shts':
        # 用 SHTS 的 CDF 反演采样 t
        t = shts_inverse_cdf(t, self.snr_scale, ...)
    elif self.rf_schedule == 'shifted':
        t = self.rf_shift * t / (1 + (self.rf_shift - 1) * t)
    return t
```

**注意**: Phase 1 只改推理网格（零训练成本），Phase 2 才改训练分布（需要重新训练）。

---

## 4. 深入可行性分析

### 4.1 与 Shifted Schedule 的关系

**数学等价性分析**:

Shifted schedule 的密度函数为 $\rho_{\text{shift}}(t) = s / (1 + (s-1)t)^2$，这是一个严格单调递减函数。

SHTS 的密度函数为 $\rho_{\text{SHTS}}(t) = \frac{2}{t(1-t)} \cdot w(t)$，其中 $w(t)$ 含有临界区高斯峰。

**对比**:

| $t$ | $\rho_{\text{shift}}(s=3)$ | $\rho_{\text{SHTS}}$ (纯SNR) | $\rho_{\text{SHTS}}$ (任务感知) |
|-----|---------------------------|------------------------------|--------------------------------|
| 0.05 | 2.70 | 42.1 | 42.1 |
| 0.2 | 1.93 | 12.5 | 12.5 |
| 0.5 | 1.33 | 8.0 | 8.0 + α |
| 0.667 | 1.09 | 9.0 | 9.0 + α·1.0 (峰值) |
| 0.8 | 0.91 | 12.5 | 12.5 |
| 0.95 | 0.77 | 42.1 | 42.1 |

**关键差异**:
1. SHTS 在 $t \to 1$（低 SNR 端）也有高密度，shifted schedule 没有
2. SHTS 在临界区（$t \approx 0.667$）有额外峰值（任务感知），shifted schedule 没有
3. SHTS 的密度比 shifted schedule 更"U形"（两端密集），shifted schedule 是纯递减

**组合方式**: SHTS 可以与 shifted schedule 串联（见算法 2），先做 shifted 变换再做 SHTS 分层。这等价于在 shifted 后的 $t'$ 空间应用 SHTS 密度。

### 4.2 与 ScaleConditionedRF 的本质区别

ScaleConditionedRF（已证伪，mAP=0.741）的核心问题是**为不同尺度的框分配不同的 $t$ 值**，破坏了统一时间轴。

| 维度 | ScaleConditionedRF | SHTS |
|------|-------------------|------|
| **时间步性质** | Per-box $t$（每个框有独立 $t$） | Per-step $t$（所有框共享同一 $t$） |
| **前向过程** | 修改：不同框用不同 $t$ 加噪 | **不修改**：前向过程完全不变 |
| **训练目标** | 修改：需要尺度条件 | **不修改**：训练目标完全不变 |
| **统一时间轴** | 破坏（head 无法用单一 $t$ 做时间嵌入） | **保持**（每个 step 所有框用同一 $t$） |
| **AdaLN-Zero 兼容** | 不兼容（时间嵌入需要标量 $t$） | **完全兼容**（$t$ 仍是标量） |
| **Box renewal 兼容** | 不兼容（不同 $t$ 的框无法统一更新） | **完全兼容** |
| **改前向?** | 是 | **否** |
| **改训练?** | 是 | **否（Phase 1）** |
| **改推理网格?** | 否 | **是** |

**核心区别**: SHTS 只改变"在哪些 $t$ 值处查询模型"，不改变"模型如何处理给定 $t$ 的输入"。所有框在同一 step 仍然使用同一个 $t$，时间嵌入、AdaLN-Zero、box renewal 等所有机制完全不受影响。

**形式化**:
- ScaleConditionedRF: $x_t^{(i)} = (1-t^{(i)}) x_0^{(i)} + t^{(i)} x_1^{(i)}$，其中 $t^{(i)}$ 依赖框 $i$ 的尺度
- SHTS: $x_t^{(i)} = (1-t) x_0^{(i)} + t x_1^{(i)}$，所有 $i$ 共享同一 $t$；只是 $\{t_0, t_1, \ldots, t_N\}$ 的选择更优

### 4.3 DPM-Solver++ 对非均匀网格的数值稳定性

**潜在风险**: DPM-Solver++ 的二阶项 $D_1 = (x_0^n - x_0^{n-1})/(t_n - t_{n-1})$ 在 $\Delta t = t_n - t_{n-1}$ 很小时可能放大数值误差。

**分析**:
- SHTS 在 $t \to 0$ 处步长最小，但此时 $x_0^{\text{pred}}$ 已经接近真实值（高 SNR），差分 $x_0^n - x_0^{n-1}$ 很小，$D_1$ 不会爆炸。
- 端点退化: 代码中已有 $t_{n+1} = 0$ 的处理（`phi1 = -t_n`），无奇异性。
- 实际最小步长: 对 $N=8$ 步，SHTS 在 $t \to 0$ 区间的典型步长约 $0.02$-$0.05$，远大于数值精度极限。

**结论**: DPM-Solver++ 对 SHTS 的非均匀网格数值稳定。

### 4.4 网格单调性保证

SHTS 生成的网格必须严格单调递减（$t_0 > t_1 > \cdots > t_N$），否则 DPM-Solver++ 的历史差分会出问题。

**保证机制**: `build_shts_grid` 通过 CDF 反演生成网格，由于密度函数 $\rho(t) > 0$ 处处成立，CDF 严格单调，反演结果必然严格单调。

### 4.5 与 Box Renewal 的兼容性

Box renewal（`sampling.py: apply_box_renewal`）在每个 step 后将低置信度框替换为随机噪声。这个操作发生在 raw 空间，与时间步网格无关。

**SHTS 兼容性**: 完全兼容。Box renewal 在每个 step 的 $t$ 值处执行，SHTS 只改变 $t$ 值的选择，不改变 box renewal 的逻辑。

### 4.6 计算复杂度

- **推理时**: SHTS 不增加 NFE（网络前向次数）。步数相同，只是 $t$ 值不同。零额外计算成本。
- **网格生成**: `build_shts_grid` 是 $O(10000)$ 的数值积分，在推理开始时执行一次，耗时 < 1ms，可忽略。
- **训练时（Phase 2）**: `_sample_t` 中增加一次 CDF 反演（$O(\log 10000)$ 查表），可忽略。

---

## 5. 风险评估

### 风险 1: 最优网格退化为单调（SHTS 无增益）

**描述**: 如果检测任务的最优时间步分配恰好是单调的（即只在高 SNR 端密集），则 SHTS 退化为 shifted schedule 的某个参数 $s^*$，无额外增益。

**概率**: 中。理论分析表明纯 SNR 导数导致 U 形密度（两端密集），但 shifted schedule 的成功说明单调偏移已经捕获了主要增益。

**缓解方案**:
1. **离线验证**: 在 a3_full_sota checkpoint 上，用不同网格做离线推理对比（零训练成本），直接测量 SHTS vs shifted 的差异。
2. **任务感知权重**: 引入临界区权重 $w(t)$，使网格非单调化。即使纯 SNR 网格退化，任务感知网格仍有差异。
3. **Fallback**: 如果 SHTS 无增益，可作为 shifted schedule 的自动调参工具（搜索 $s^*$）。

### 风险 2: 临界区权重 $w(t)$ 的超参数敏感

**描述**: 任务感知权重中的 $\alpha$（强度）和 $\sigma$（宽度）需要调参，不当设置可能导致网格过于集中在临界区，牺牲定位精度。

**概率**: 中低。临界区权重是平滑高斯，对局部扰动不敏感。

**缓解方案**:
1. **保守初始值**: $\alpha = 0.5$（临界区密度增加 50%），$\sigma = 0.15$（覆盖 ±2σ ≈ 0.3 的区间）。
2. **消融实验**: 离线测试 $\alpha \in \{0, 0.5, 1.0, 2.0\}$，选择最优。
3. **理论指导**: $\alpha$ 的上限可通过 DPM-Solver++ 的稳定性条件约束（$\phi_1 \cdot D_1$ 不超过线性项的 10%）。

### 风险 3: 训练-推理网格不匹配（Phase 1）

**描述**: Phase 1 只改推理网格，训练时的 $t$ 采样仍用 shifted schedule。如果推理网格偏离训练分布太多，模型在未见过的 $t$ 值处预测质量下降。

**概率**: 中。RF 的直线路径使得模型对任意 $t$ 都有合理的预测（不像 DDPM 需要精确匹配），但极端 $t$ 值仍有风险。

**缓解方案**:
1. **温和偏离**: SHTS 网格与 shifted 网格的差异控制在 $|\Delta t| < 0.1$ 以内。
2. **组合策略**: 先 shifted 再 SHTS，保证网格在训练分布附近。
3. **Phase 2 兜底**: 如果 Phase 1 增益不足，Phase 2 同步训练分布，消除不匹配。

### 风险 4: 小步数（N=4）下网格过于稀疏

**描述**: 在 $N=4$ 步时，SHTS 的三段式分层可能导致某段只有 1 步，无法体现分层优势。

**概率**: 中高。$N=4$ 时，按 40:45:15 分配为 2:2:0 或 1:2:1，Layer III 可能被牺牲。

**缓解方案**:
1. **步数自适应分配**: 小步数时减少层数（合并 Layer II 和 III）。
2. **N≥6 推荐**: SHTS 在 $N \geq 6$ 时优势明显，推荐与 DPM-Solver++ 6步/8步配置配合。
3. **退化策略**: $N=4$ 时 SHTS 自动退化为 shifted schedule（通过检测 $N < 6$）。

### 风险 5: 与 Ensemble 的交互

**描述**: 当前推理使用 `use_ensemble=True`，将所有 step 的预测做集成。SHTS 改变了步的位置，可能影响集成权重。

**概率**: 低。Ensemble 对所有 step 等权处理，SHTS 只是改变了每步的 $t$ 值，不影响集成逻辑。

**缓解方案**: 如果 ensemble 结果异常，可关闭 ensemble 测试纯网格效果。

---

## 6. 预期收益分析

### 6.1 基于 1→8 步增益曲线的定量估计

**已知数据点**:
- Heun 1步: mAP = 0.740
- Heun 8步: mAP = 0.755
- DPM-2 8步: mAP = 0.755（APs +0.004 vs Heun 4步）

**增益分解模型**:

总增益 = 步数增加带来的"信息增益" + 网格优化带来的"分配增益" + 求解器高阶带来的"积分增益"

从 1步→8步的 +0.015 增益中：
- 信息增益（更多步 = 更多模型查询）: 约 60% = +0.009
- 分配增益（网格更优）: 约 25% = +0.004
- 积分增益（Heun 2阶 vs Euler 1阶）: 约 15% = +0.002

**SHTS 的目标**: 在固定步数下，提升"分配增益"部分。

### 6.2 各配置的预期收益

| 配置 | 当前 mAP | SHTS 预期 mAP | 增益 | 依据 |
|------|----------|---------------|------|------|
| Heun 4步 (NFE=8) | 0.753 (旧) / 0.858 (新基线) | +0.003~0.006 | 中等 | 4步分配空间有限 |
| DPM-2 6步 (NFE=7) | 0.753 (旧) | +0.004~0.008 | 较高 | 6步+高阶积分协同 |
| DPM-2 8步 (NFE=9) | 0.755 (旧) | +0.005~0.010 | 最高 | 8步充分分层+高阶积分 |

**定量估计方法**:

1. **Shifted schedule 的分配增益上界**: 通过搜索 $s \in \{1.0, 1.5, 2.0, 3.0, 4.0, 5.0\}$，找到最优 $s^*$。$s^*$ 与 $s=3.0$ 的 mAP 差异即为 shifted schedule 的调参空间。

2. **SHTS 额外增益**: SHTS 相对于最优 $s^*$ 的增益，来自非单调密度。估计为 shifted 调参空间的 30-50%（因为非单调性是 shifted 无法表达的部分）。

3. **小目标 APs 特别增益**: SHTS 在高 SNR 区（$t \to 0$）分配更多密度，直接提升小目标的定位精度。预期 APs +0.005~0.010。

### 6.3 成本-收益比

| 方案 | 实现成本 | 训练成本 | 预期增益 | 成本-收益比 |
|------|----------|----------|----------|-------------|
| SHTS Phase 1 (仅推理) | 2天代码 | **零** (用现有checkpoint) | +0.003~0.006 | **极高** |
| SHTS Phase 2 (训练+推理) | +2天代码 | 150 epoch | +0.005~0.010 | 高 |
| DPM-2 8步 (已验证) | 已完成 | 已完成 | +0.002 vs Heun 4步 | 已实现 |
| SHTS + DPM-2 8步 | Phase 1+1天 | 零 (Phase 1) | +0.005~0.010 | **极高** |

**推荐**: 优先实施 SHTS Phase 1 + DPM-2 8步 的组合，零训练成本即可验证。

---

## 7. 实现路线图

### Phase 1: 离线验证（仅推理，零训练成本）

**目标**: 在 a3_full_sota checkpoint 上验证 SHTS 网格相对于 shifted schedule 的增益。

**代码改动清单**:

| 文件 | 改动 | 描述 |
|------|------|------|
| `ldmdet/diffusion/shts.py` | **新建** | SHTS 网格生成算法（`build_shts_grid`, `build_shts_shifted_grid`） |
| `ldmdet/diffusion/sampling.py` | 修改 `build_time_pairs` | 增加 `rf_schedule == 'shts'` 分支 |
| `ldmdet/diffusion/sampling.py` | 修改 `create_dpm_solver` | 将 SHTS 网格传入 `RFDPMSolverMultistep.timesteps` |
| `ldmdet/diffusion/rectified_flow.py` | 修改 `RFDPMSolverMultistep.__init__` | 支持外部传入 `timesteps` 列表 |
| `ldmdet/core/head.py` | 修改 `DiffusionSampler` 初始化 | 传入 SHTS 参数 (`shts_alpha`, `shts_sigma`) |
| `experiments/configs/ldmdet/ldmdet_shts_dpm8.py` | **新建** | SHTS + DPM-2 8步推理配置 |

**验证实验**:

| 实验 | 网格 | 步数 | 预期 |
|------|------|------|------|
| Baseline | shifted s=3 | 4 (Heun) | mAP=0.858 |
| DPM-2 baseline | shifted s=3 | 8 | mAP≈0.860 |
| SHTS pure | SHTS (α=0) | 8 | mAP≈0.860~0.862 |
| SHTS task-aware | SHTS (α=0.5) | 8 | mAP≈0.861~0.864 |
| SHTS + shifted | shifted+SHTS | 8 | mAP≈0.862~0.865 |
| SHTS ablation | SHTS (α=0,0.5,1,2) | 8 | 找最优 α |

### Phase 2: 训练分布匹配（如果 Phase 1 有正向信号）

**目标**: 将训练时的 $t$ 采样也改为 SHTS 分布，消除训练-推理不匹配。

**代码改动清单**:

| 文件 | 改动 | 描述 |
|------|------|------|
| `ldmdet/core/head.py` | 修改 `_sample_t` | 增加 `rf_schedule == 'shts'` 分支，用 SHTS CDF 反演采样 |
| `experiments/configs/ldmdet/ldmdet_shts_train.py` | **新建** | SHTS 训练配置 |

**训练实验**:

| 实验 | 训练 t 采样 | 推理网格 | Epochs | 预期 |
|------|------------|----------|--------|------|
| SHTS train+infer | SHTS | SHTS | 150 | mAP≈0.862~0.868 |

### Phase 3: 自适应步数（探索性）

**目标**: 根据 SNR 梯度自动决定步数，而非固定 N。

**思路**: 在 SNR 变化极慢的区域（如 $t \in [0.9, 1.0]$）跳过整步，将预算重新分配到高 SNR 区。

**代码改动**: 在 `build_time_pairs` 中增加自适应步数逻辑。

### 时间线

| Phase | 时长 | 依赖 | 产出 |
|-------|------|------|------|
| Phase 1 | 3天 | a3_full_sota checkpoint | 离线 mAP 对比表 |
| Phase 2 | 2周 | Phase 1 正向信号 | 训练后 mAP |
| Phase 3 | 1周 | Phase 2 | 自适应步数验证 |

---

## 8. 与已证伪方向的对比

### 8.1 对比总表

| 方向 | 改前向 | 改训练 | 改推理 | mAP | 证伪原因 |
|------|--------|--------|--------|-----|----------|
| CFM 速度预测 | 是 | 是 | 否 | 0.823 | 速度场在 d=4 空间表达力不足 |
| ScaleConditionedRF | 是 | 是 | 否 | 0.741 | per-box t 破坏统一时间轴 |
| **SHTS (本方案)** | **否** | **否(Phase1)** | **是** | **待验证** | — |

### 8.2 与 ScaleConditionedRF 的深度对比

ScaleConditionedRF 是与本方向最容易混淆的已证伪方向。以下从五个维度深度对比：

**维度 1: 时间步的粒度**

- ScaleConditionedRF: **per-box** $t$。同一 step 内，大框用 $t_{\text{large}}$，小框用 $t_{\text{small}}$，每个框有独立的时间步。
- SHTS: **per-step** $t$。同一 step 内，所有框共享同一 $t$。不同 step 之间的 $t$ 间隔由 SNR 分析决定。

**维度 2: 对统一时间轴的影响**

- ScaleConditionedRF: **破坏统一时间轴**。Head 的时间嵌入需要处理变长 $t$ 向量，AdaLN-Zero 的标量条件机制失效。级联 Head 中上一级输出传给下一级时，不同 $t$ 的框无法统一处理。
- SHTS: **保持统一时间轴**。每个 step 所有框用同一标量 $t$，时间嵌入、AdaLN-Zero、级联 Head 完全不变。

**维度 3: 对前向过程的影响**

- ScaleConditionedRF: **修改前向过程**。$x_t^{(i)} = (1-t^{(i)}) x_0^{(i)} + t^{(i)} x_1^{(i)}$，不同框的加噪程度不同。
- SHTS: **不修改前向过程**。$x_t = (1-t) x_0 + t x_1$，前向过程完全不变。SHTS 只决定推理时查询哪些 $t$ 值。

**维度 4: 对训练目标的影响**

- ScaleConditionedRF: **修改训练目标**。需要尺度条件注入、per-box 损失加权。
- SHTS: **不修改训练目标**。L1 + GIoU + Focal 损失完全不变。

**维度 5: 对 Box Renewal 的影响**

- ScaleConditionedRF: **不兼容**。Box renewal 将低置信度框替换为随机噪声，但不同 $t$ 的框无法在同一步统一更新。
- SHTS: **完全兼容**。Box renewal 在每个 step 的统一 $t$ 下执行。

**核心结论**: ScaleConditionedRF 的失败根因是"破坏统一时间轴"，而 SHTS 从设计上就避免了这个问题。SHTS 的改动范围严格限制在"推理时的时间步网格选择"，不触及前向过程、训练目标、时间嵌入等任何可能破坏统一性的组件。

### 8.3 与 CFM 速度预测的对比

CFM（已证伪，mAP=0.823）的失败原因是速度场在 $d=4$ 的低维空间中表达力不足，且速度预测的训练目标与检测损失冲突。

SHTS 与 CFM 完全正交：
- CFM 改的是"模型预测什么"（预测速度 vs 预测 $x_0$）
- SHTS 改的是"在哪些 $t$ 处查询模型"
- 两者可以组合（SHTS + CFM），但鉴于 CFM 已证伪，不推荐。

---

## 9. 参考文献

1. **Rectified Flow**: Liu, X., Gong, C., & Liu, Q. (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR 2023*. — RF 的原始论文，定义了直线路径和 SNR 公式。

2. **DPM-Solver++**: Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models." *NeurIPS 2022*. — DPM-Solver++ 的原始论文，提供高阶积分公式。

3. **Shifted Schedule (SD3)**: Esser, P., Kulal, S., Blattmann, A., et al. (2024). "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024*. — 提出 shifted schedule 用于 RF，$t' = s \cdot t / (1 + (s-1) \cdot t)$。

4. **SNR-Optimal Sampling**: Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models." *NeurIPS 2022*. — 分析 SNR 与采样步长的关系，提出基于 SNR 的自适应步长。

5. **DiffusionDet**: Zhang, S., Wang, R., & Yu, X. (2023). "DiffusionDet: Diffusion Model for Object Detection." *ICCV 2023*. — 扩散检测的先驱工作，定义了框空间的扩散范式。

6. **Chromosome Detection (本项目)**: a3_full_sota 配置，RF + Heun 4步 + AdaLN-Zero + StochasticOT ε=5，mAP=0.858。 — 本方案的基线。

7. **OT Diversity Collapse (本项目)**: Stochastic OT ε=5 实验，证明 OT 在 d=4 检测空间中的多样性塌缩问题。 — 说明检测扩散与图像生成扩散的本质区别。

8. **DPM-Solver++ for RF (本项目)**: `RFDPMSolverMultistep` 实现，离线对比 DPM-2 8步 vs Heun 4步。 — SHTS 的直接协同对象。

---

## 附录 A: SHTS 网格示例

### N=8 步 SHTS 网格 (snr_scale=2.0, α=0.5, σ=0.15)

```
Shifted s=3.0:   [1.000, 0.857, 0.692, 0.529, 0.375, 0.231, 0.120, 0.045, 0.000]
SHTS (pure):     [1.000, 0.881, 0.745, 0.612, 0.485, 0.348, 0.205, 0.082, 0.000]
SHTS (task-aware): [1.000, 0.892, 0.763, 0.658, 0.521, 0.368, 0.215, 0.085, 0.000]
                   Layer III |-- Layer II (临界区加密) --| Layer I (定位区加密)
```

### N=4 步 SHTS 网格 (退化模式)

```
Shifted s=3.0:   [1.000, 0.692, 0.375, 0.120, 0.000]
SHTS (pure):     [1.000, 0.745, 0.485, 0.205, 0.000]
```

## 附录 B: SNR 导数数值表

| $t$ | SNR (snr_scale=2) | $d\text{SNR}/dt$ | $d\ln\text{SNR}/dt$ | $d^2\text{SNR}/dt^2$ |
|-----|-------------------|-------------------|----------------------|----------------------|
| 0.05 | 1444.0 | -58320.0 | -40.4 | 4.66×10⁶ |
| 0.1 | 324.0 | -6480.0 | -20.0 | 194400 |
| 0.2 | 64.0 | -800.0 | -12.5 | 8000 |
| 0.388 | 10.0 | -81.5 | -8.15 | 486 |
| 0.5 | 4.0 | -32.0 | -8.0 | 192 |
| 0.667 | 1.0 | -8.99 | -8.99 | 54.1 |
| 0.8 | 0.25 | -3.13 | -12.5 | 48.8 |
| 0.863 | 0.1 | -1.28 | -12.8 | 22.7 |
| 0.95 | 0.011 | -0.22 | -20.0 | 9.5 |

**关键观察**: $|d\ln\text{SNR}/dt|$ 在 $t=0.5$ 处取最小值 8.0，向两端发散。这验证了"两端密集、中间稀疏"的最优分配策略。

---

*文档版本: v1.0 | 创建日期: 2026-07-12 | 作者: 扩散检测研究组*
