# DDIM：加速采样的奥义（从 DDPM 到非马尔可夫链）

DDPM 虽然效果好，但采样需要 $T=1000$ 步，速度太慢。
DDIM (Denoising Diffusion Implicit Models) 的核心贡献在于：**证明了即使扩散过程不是马尔可夫链，也能训练出同样的模型，并且可以跳步采样（例如只用 50 步）。**

本笔记将基于 [DDPM推导.md](file:///home/linkst/workplace/chromo/chromosome-kd/DDPM推导.md) 的基础，推导 DDIM 的加速原理。

---

## 1. 核心思想：寻找新的后验分布

在 DDPM 中，我们依赖马尔可夫链 $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ 来定义前向过程。
DDIM 的作者 Song 等人提出：**只要保证边缘分布 $q(\mathbf{x}_t|\mathbf{x}_0)$ 不变（还是那个高斯），中间的转移过程 $q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)$ 其实可以随便改！**

只要边缘分布不变，我们训练好的 DDPM 模型（预测 $\boldsymbol{\epsilon}_\theta$）就依然适用，不需要重新训练。

---

## 2. 非马尔可夫链的后验构造

我们要构造一个新的后验分布 $q_\sigma(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$，使得它满足以下条件：
1.  **一致性**：从 $\mathbf{x}_0$ 推导出的边缘分布 $q(\mathbf{x}_t|\mathbf{x}_0)$ 必须和 DDPM 一样。
2.  **可控性**：引入一个参数 $\sigma_t$，控制采样的随机程度。

DDIM 给出的后验公式如下（为了直观，直接写成生成形式）：

$$ \mathbf{x}_{t-1} = \underbrace{\sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{预测的 } \mathbf{x}_0} }_{\text{指向 } \mathbf{x}_0 \text{ 的方向}} + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t)}_{\text{指向噪声的方向}} + \underbrace{\sigma_t \boldsymbol{\epsilon}_t}_{\text{随机噪声}} $$

这个公式看似复杂，其实含义非常清晰：
*   **第一项**：利用模型预测出的 $\mathbf{x}_0$（去噪后的干净图像），按 $\sqrt{\bar{\alpha}_{t-1}}$ 的比例重建。
*   **第二项**：利用模型预测出的噪声 $\boldsymbol{\epsilon}_\theta$，按一定比例加回去，保持分布方差正确。
*   **第三项**：显式的随机噪声注入。

---

## 3. 两种特殊情况

### 3.1 $\sigma_t = \tilde{\beta}_t$ (DDPM)
当 $\sigma_t$ 取 DDPM 中的真实后验方差 $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ 时：
上式完全退化为 DDPM 的采样公式。
也就是说，**DDPM 只是 DDIM 框架下 $\sigma_t$ 取特定值的一个特例。**

### 3.2 $\sigma_t = 0$ (DDIM)
这是 DDIM 最激动人心的地方。
当我们将随机噪声项 $\sigma_t$ 设为 0 时，采样过程变成了**确定性过程（Deterministic）**！
$$ \mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta $$

**意义**：
1.  **确定性映射**：给定初始噪声 $\mathbf{x}_T$，生成的图像 $\mathbf{x}_0$ 是唯一确定的。这使得我们可以对 Latent Space 进行插值（Interpolation）。
2.  **一致性**：这等价于常微分方程（ODE）求解器。

---

## 4. 为什么可以加速（跳步采样）？

在 DDPM 中，我们必须一步步走 ($t \to t-1$)，因为推导依赖 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$。
但在 DDIM 的公式中，**我们直接定义了从任意状态 $\mathbf{x}_t$ 和预测的 $\mathbf{x}_0$ 如何生成 $\mathbf{x}_{t-1}$**。

由于 $\mathbf{x}_0$ 是模型预测出来的，我们可以把它看作一个“锚点”。
我们可以定义任意一个子序列 $\tau = [\tau_1, \tau_2, \dots, \tau_S]$，例如 $[1000, 900, 800, \dots, 0]$。
直接套用 DDIM 公式，从 $\mathbf{x}_{\tau_i}$ 跳到 $\mathbf{x}_{\tau_{i-1}}$：

$$ \mathbf{x}_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \left( \frac{\mathbf{x}_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}}\boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_{\tau_i}}} \right) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \boldsymbol{\epsilon}_\theta + \sigma_{\tau_i} \boldsymbol{\epsilon} $$

因为是确定性（或近乎确定性）的过程，跳步带来的误差比随机游走（DDPM）要小得多。
通常 DDIM 只需要 50 步甚至 20 步就能生成高质量图像，比 DDPM 快 20-50 倍。

---

## 5. 总结与对比

| 特性 | DDPM | DDIM |
| :--- | :--- | :--- |
| **采样过程** | 随机 (Stochastic) | 确定性 (Deterministic) ($\sigma=0$) |
| **马尔可夫性** | 是 | 否 |
| **采样步数** | 必须 $T$ 步 (1000) | 可任意跳步 (10-100) |
| **生成质量** | 多样性好，细节丰富 | 结构更清晰，收敛更快 |
| **模型训练** | 预测 $\boldsymbol{\epsilon}$ | **完全相同** (无需重训) |
| **插值能力** | 差 (随机性干扰) | 强 (Latent 线性插值) |

---

## 6. 代码实现提示 (PyTorch)

在实现 DDIM 时，只需修改采样循环：

```python
# 假设我们有一个时间步序列 time_pairs = [(1000, 950), (950, 900), ...]
for t, t_prev in time_pairs:
    # 1. 预测噪声
    epsilon = model(x_t, t)
    
    # 2. 预测 x_0
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_prev = alphas_cumprod[t_prev]
    
    pred_x0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    
    # 3. 计算指向 x_t 的方向 (DDIM 公式)
    # sigma = 0 for deterministic DDIM
    sigma = 0 
    dir_xt = sqrt(1 - alpha_bar_prev - sigma**2) * epsilon
    
    # 4. 更新 x_{t-1}
    x_prev = sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * randn()
    
    x_t = x_prev
```
