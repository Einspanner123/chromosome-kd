`······`································`········`·······`·······························`·······`·························································································`···························# DPM-Solver++ 向 Rectified Flow 的移植推导

> **目标**：将 DPM-Solver++ / v3 的高阶多项式积分框架适配到 Rectified Flow ODE 上，实现显著减少推理 NFE。
>
> **日期**：2026-05-27
> **状态**：推导阶段，尚未实现
> **版本**：v2.0 — 整合 $t$ 空间直接插值与显式闭式公式

______________________________________________________________________

## 1. 问题设定与符号约定

### 1.1 Rectified Flow ODE

LDMDet 使用 Rectified Flow。前向过程：

$$x_t = (1-t) x_0 + t x_1, \quad x_1 \sim \mathcal{N}(0, I), \quad t \in [0, 1]$$

概率流 ODE（反向采样，$t: 1 \to 0$）：

$$\frac{dx_t}{dt} = v_\theta(x_t, t) \tag{1.1}$$

模型输出 $x_0^{\text{pred}} = f_\theta(x_t, t)$。LDMDet 直接输出 $x_0$ 预测（非 $v$ 或 $\epsilon$）。速度场：

$$v_\theta(x_t, t) = \frac{x_t - x_0^{\text{pred}}}{t} \tag{1.2}$$

**推理任务**：从 $x_1 \sim \mathcal{N}(0, I)$ 积到 $t=0$ 得 $x_0$。

### 1.2 当前求解器

| 方法 | 阶数 | NFE/步 | 总 NFE (4 步) |
|:---|:---|:---|:---|
| Euler | 1 | 1 | 4 |
| Heun | 2 | 2 | 8 |

**目标**：同等 NFE 下提升精度阶数，或更少 NFE 达到同等精度。

______________________________________________________________________

## 2. 半线性形式与坐标系选择

### 2.1 RF ODE 的半线性重构

将 (1.2) 代入 (1.1)：

$$\frac{dx_t}{dt} - \frac{1}{t} x_t = -\frac{1}{t} x_0^{\text{pred}} \tag{2.1}$$

标准一阶线性 ODE 形式 $dx/dt + P(t)x = Q(t)$，$P=-1/t$，$Q=-x_0^{\text{pred}}/t$。

### 2.2 精确积分解

积分因子 $\mu(t) = \exp(-\int P dt) = 1/t$。两边同乘：

$$\frac{d}{dt}\left(\frac{x_t}{t}\right) = -\frac{x_0^{\text{pred}}}{t^2}$$

从 $t_n$ 到 $t_{n+1}$ 定积分（$t_n > t_{n+1}$，反向）：

$$\frac{x_{t_{n+1}}}{t_{n+1}} - \frac{x_{t_n}}{t_n} = \int_{t_{n+1}}^{t_n} \frac{x_0^{\text{pred}}}{t^2} dt$$

**精确积分形式（核心出发点）**：

$$x_{t_{n+1}} = \frac{t_{n+1}}{t_n} x_{t_n} + t_{n+1} \int_{t_{n+1}}^{t_n} \frac{x_0^{\text{pred}}(x_\tau, \tau)}{\tau^2} d\tau \tag{2.2}$$

### 2.3 坐标系 A：$\lambda = \ln(t)$（DPM-Solver 正统）

令 $\lambda = \ln t$（$t = e^\lambda$），$dt/t = d\lambda$。代入 (2.2)：

$$x_{n+1} = e^{h} x_n + e^{\lambda_{n+1}} \int_{\lambda_{n+1}}^{\lambda_n} e^{-\lambda} x_0^{\text{pred}}(\lambda) d\lambda$$

其中 $h = \lambda_{n+1} - \lambda_n = \ln(t_{n+1}/t_n)$，$h < 0$（反向）。

对 $x_0^{\text{pred}}(\lambda)$ 做 Lagrange 插值，以 $e^{-\lambda}$ 为权精确积分，得 DPM-Solver++ 系数。

⚠️ $t=0$ 时 $\lambda = -\infty$，需设 $t_{\text{end}} = 10^{-3}$ 截断 + 线性投影 $x_0 \approx x_0^{\text{pred}}(x_{t_{\text{end}}})$。

### 2.4 坐标系 B：$t$ 空间直接插值（推荐，专为 RF 优化）

不在 $\lambda$ 空间转换，直接在 $t$ 空间对 $x_0^{\text{pred}}(t)$ 做多项式拟合代入 (2.2)。

**优势**：
1. $t_{n+1}=0$ 时利用 $\lim_{x\to0} x\ln x = 0$ 自然退化——**无奇点，无需截断**
2. 完全兼容 `linspace(1, 0, N+1)`——无需重采样
3. 半线性项 $\frac{t_{n+1}}{t_n}x_{t_n}$ 被精确求解，仅 $x_0^{\text{pred}}$ 的积分为近似

______________________________________________________________________

## 3. $t$ 空间显式公式：1 至 3 阶

记号约定（$n$ 为当前步，$n+1$ 为目标步）：
- $x_n = x_{t_n}$，$\hat{x}_n = x_0^{\text{pred}}(x_{t_n}, t_n)$
- $t_n > t_{n+1}$（反向采样）

### 3.1 一阶公式（等价 Euler）

将 $\hat{x}(\tau) \equiv \hat{x}_n$（常数）代入 (2.2)：

$$\begin{aligned}
x_{n+1} &= \frac{t_{n+1}}{t_n} x_n + t_{n+1} \hat{x}_n \int_{t_{n+1}}^{t_n} \frac{d\tau}{\tau^2} \\[4pt]
        &= \frac{t_{n+1}}{t_n} x_n + \left(1 - \frac{t_{n+1}}{t_n}\right) \hat{x}_n
\end{aligned} \tag{3.1}$$

**NFE = 1/步**。此即 RF 半线性框架下的等效 Euler。

### 3.2 二阶多步公式（RF-DPM2-Multistep）

已知两个历史点 $(t_{n-1}, \hat{x}_{n-1})$ 和 $(t_n, \hat{x}_n)$，在 $\tau$ 上做线性插值：

$$\hat{x}(\tau) = \hat{x}_n + \frac{\hat{x}_n - \hat{x}_{n-1}}{t_n - t_{n-1}} (\tau - t_n)$$

代入 (2.2) 精确积分：

$$x_{n+1} = \frac{t_{n+1}}{t_n} x_n + \left(1 - \frac{t_{n+1}}{t_n}\right) \hat{x}_n + \phi_1 \cdot D_1 \tag{3.2}$$

其中：
$$D_1 = \frac{\hat{x}_n - \hat{x}_{n-1}}{t_n - t_{n-1}}$$
$$\phi_1 = t_{n+1} \ln\left(\frac{t_n}{t_{n+1}}\right) - t_n + t_{n+1} \qquad (t_{n+1} > 0)$$

**🔥 终点退化**：当 $t_{n+1} = 0$，利用 $\lim_{x\to0} x\ln x = 0$：

$$\phi_1 \big|_{t_{n+1}=0} = -t_n$$

代入 (3.2) 得：

$$x_0 = \frac{t_n \hat{x}_{n-1} - t_{n-1} \hat{x}_n}{t_n - t_{n-1}} \tag{3.2z}$$

**完全无奇点，无需额外 NFE。**

**NFE 预算**：predictor 用历史信息（0 NFE），当前步原有 1 NFE。起步用 1 阶。

### 3.3 三阶多步公式（RF-DPM3-Multistep）

已知三点 $(t_{n-2}, \hat{x}_{n-2}), (t_{n-1}, \hat{x}_{n-1}), (t_n, \hat{x}_n)$，构造二次插值。

用牛顿差商简化表达：

$$D_1 = \frac{\hat{x}_n - \hat{x}_{n-1}}{t_n - t_{n-1}}$$
$$D_2 = \frac{1}{t_n - t_{n-2}} \left[ D_1 - \frac{\hat{x}_{n-1} - \hat{x}_{n-2}}{t_{n-1} - t_{n-2}} \right]$$

二次插值多项式：

$$\hat{x}(\tau) = \hat{x}_n + D_1(\tau - t_n) + D_2(\tau - t_n)(\tau - t_{n-1})$$

代入 (2.2) 精确积分：

$$x_{n+1} = \frac{t_{n+1}}{t_n} x_n + \left(1 - \frac{t_{n+1}}{t_n}\right) \hat{x}_n + \phi_1 D_1 + \phi_2 D_2 \tag{3.3}$$

其中 $\phi_1$ 同上，$\phi_2$（$t_{n+1} > 0$）：

$$\phi_2 = t_{n+1}\left[t_n \ln\left(\frac{t_n}{t_{n+1}}\right) + t_{n+1} - t_n\right] - \frac{t_n^2 - t_{n+1}^2}{2} + t_n(t_n - t_{n+1})$$

当 $t_{n+1} = 0$ 时：

$$\phi_2 \big|_{t_{n+1}=0} = \frac{t_n^2}{2}$$

**NFE = 1/步**（启动步除外）。全局 3 阶。

______________________________________________________________________

## 4. DPM-Solver v3 EMS 机制（可选）

DPM-Solver v3 引入**经验模型统计（EMS）**：神经网络在不同 $t$ 的预测方差不同，纯数学高阶插值可能"过拟合"到高方差段。

可选的调和系数 $\alpha(t)$：

```python
alpha = 1.0 if 0.3 < t_n < 0.7 else 0.8
prev_sample = linear_term + alpha * correction_term
```

**建议初版不加 EMS**：先用纯数学公式验证，EMS 是后期调优。

______________________________________________________________________

## 5. 与现有代码的接入点

### 5.1 修改清单

| 文件 | 内容 |
|:---|:---|
| `rectified_flow.py` | 新增 `RFDPMSolverMultistep` 类（带历史缓存） |
| `diffusiondet_head.py` | `predict()` 新增 `solver_type='dpm_solver_pp'` 分支 |

### 5.2 LDMDet 适配关键

- **模型直接输出 $x_0^{\text{pred}}$**，跳过 $\hat{X}_0 = X_t - t v_\theta$ 的反推步骤
- **时间方向**：$t: 1 \to 0$，与本文一致
- **建议 $N=6$ 步起步**（NFE=7-8），充分释放多步法优势
  - 2 阶：NFE = 7（起步 1 + 后续 5×1 + 终点 1）
  - 3 阶：NFE = 8（起步 2 + 后续 4×1 + 终点 1）

### 5.3 接口伪代码

```python
class RFDPMSolverMultistep:
    def __init__(self, num_steps=6, solver_order=2):
        self.timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
        self.order = solver_order
        self.x0_history = []
        self.t_history = []

    def step(self, x, x0_pred, t_n, step_idx):
        t_next = self.timesteps[step_idx + 1]
        self.x0_history.append(x0_pred); self.t_history.append(t_n)
        if len(self.x0_history) > self.order:
            self.x0_history.pop(0); self.t_history.pop(0)

        linear = (t_next/t_n)*x + (1 - t_next/t_n)*x0_pred
        if len(self.x0_history) == 1:
            return linear  # 1阶
        elif len(self.x0_history) >= 2:
            tp, xp = self.t_history[-2], self.x0_history[-2]
            phi1 = t_next*math.log(t_n/t_next)-t_n+t_next if t_next>1e-7 else -t_n
            return linear + (phi1/(t_n-tp))*(x0_pred - xp)  # 2阶
        # ... 3阶分支
```

______________________________________________________________________

## 6. NFE 效率与预期

| 配置 | 步数 | NFE | 阶数 | vs Heun(4步/8NFE) |
|:---|:---|:---|:---|:---|
| Heun（当前） | 4 | 8 | 2 | 基准 0.753 |
| DPM-2 | 4 | 5 | 2 | 同阶, -37% NFE |
| DPM-2 | 6 | 7 | 2 | 同阶, 更密步长 |
| **DPM-3** | **6** | **8** | **3** | **高1阶, 同等NFE** |
| DPM-3 | 8 | 10 | 3 | 天花板更高 |

**关键风险**：
1. 多步法起步：前 1-2 步低阶，精度损失小
2. Shifted schedule：仅影响训练 $t$ 分布，推理 ODE 不变

______________________________________________________________________

## 7. 实现路线图

- **Phase 1**：`RFDPMSolverMultistep(order=2)`，6 步/7 NFE，对打 Heun (8 NFE)
- **Phase 2**：`order=3`，同等 NFE 下追求 mAP 提升
- **Phase 3**（可选）：自适应步长 + EMS 调优

______________________________________________________________________

> **下一阶段**：Phase 1 代码实现
