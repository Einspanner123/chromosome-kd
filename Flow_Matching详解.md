# Flow Matching 详解：生成模型的新范式

## 1. 核心直觉：从“弯曲”到“直线”

如果你已经理解了 [DDPM](DDPM推导.md)，你会发现扩散模型本质上是在学习如何逆转一个缓慢的退化过程。但在数学上，扩散模型的样本演化路径通常是**弯曲的（Curved）**，这意味着在采样时（ODE 求解），步长不能太大，否则会偏离路径。

**Flow Matching (FM)** 的核心思想非常直接：

> 既然我们要把噪声 $x_1$ 变成数据 $x_0$，为什么不让它走**直线**呢？

### 直观对比

- **Diffusion (DDPM/LDM):** 路径受高斯噪声影响，是随机且弯曲的轨迹。
- **Flow Matching:** 学习一个**恒定的速度场 (Velocity Field)**，让粒子在 $t \\in \[0, 1\]$ 时间内，以最平滑的直线从分布 $p_0$ 运动到 $p_1$。

______________________________________________________________________

## 2. 数学基础：概率流与速度场

### 2.1 概率路径 (Probability Path)

假设我们有一个随时间演化的概率分布 $p_t(x)$。

- $p_0(x)$ 是真实数据分布。
- $p_1(x)$ 是标准高斯噪声分布 $\\mathcal{N}(0, I)$。
  *注：在 FM 文档中，通常习惯将 $t=0$ 设为数据，$t=1$ 设为噪声，这与 DDPM 的 $T \\to 0$ 相对应。*

### 2.2 速度场 (Velocity Field)

如果我们能找到一个向量场 $v_t(x)$（即在 $t$ 时刻，位置 $x$ 处的速度），那么样本的运动遵循常微分方程 (ODE)：
$$\\frac{dx_t}{dt} = v_t(x_t)$$

**Flow Matching 的目标：** 训练一个神经网络 $v\_\\theta(x_t, t)$ 来拟合这个真实的速度场。

______________________________________________________________________

## 3. 核心公式推导：条件流匹配 (Conditional Flow Matching)

直接匹配全量分布的速度场非常困难，FM 引入了和 Diffusion 类似的技巧：**条件化**。

### 3.1 条件概率路径

对于给定的样本 $x_0$，我们定义它到噪声 $x_1$ 的直线路径为：
$$x_t = \\psi_t(x_0, x_1) = (1-t)x_0 + t x_1$$

### 3.2 条件速度场

对上面的 $x_t$ 关于 $t$ 求导，我们直接得到了该样本的**速度**：
$$v_t(x_t | x_0, x_1) = \\frac{d x_t}{dt} = x_1 - x_0$$
这个速度非常简单：它就是一个指向目标（噪声或数据）的恒定向量！

### 3.3 损失函数 (Loss Function)

训练时，我们不再预测噪声 $\\epsilon$，而是预测这个速度向量：
$$L\_{FM} = \\mathbb{E}_{t \\sim \\mathcal{U}\[0,1\], x_0 \\sim p_{data}, x_1 \\sim \\mathcal{N}(0,I)} | v\_\\theta(x_t, t) - (x_1 - x_0) |^2$$
其中 $x_t = (1-t)x_0 + t x_1$。

______________________________________________________________________

## 4. 为什么 Flow Matching 更好？

1. **采样速度极快：** 因为学习到的路径是近乎直线的，ODE 求解器可以用极大的步长（甚至 1 步）达到很高的精度。
2. **训练更稳定：** 相比于扩散模型在 $t \\to 0$ 附近的梯度爆炸或不稳定性，FM 的速度场在整个 $t \\in \[0,1\]$ 范围内通常更平滑。
3. **更广的泛化：** FM 不局限于高斯噪声，它可以实现任意两个分布之间的传输（Rectified Flow）。

______________________________________________________________________

## 5. 与 Diffusion 的联系

如果你把 FM 的路径写成：
$$x_t = \\alpha_t x_0 + \\sigma_t x_1$$
你会发现：

- 当 $\\alpha_t, \\sigma_t$ 是特定曲线（如 $1-t$ 和 $t$）时，它是 **Flow Matching**。
- 当 $\\alpha_t, \\sigma_t$ 遵循扩散系数（如 $\\sqrt{\\bar{\\alpha}\_t}$ 和 $\\sqrt{1-\\bar{\\alpha}\_t}$）时，它退化回 **Diffusion ODE**。

**结论：** Diffusion 是 Flow Matching 在特定概率路径下的特例，而 Flow Matching 选择了最简单的直线路径。

______________________________________________________________________

## 6. 工业界应用

目前顶级模型已经全面转向 Flow Matching 或其变体（Rectified Flow）：

- **Stable Diffusion 3 (SD3):** 使用 Rectified Flow。
- **Flux.1:** 目前开源界最强的文生图模型，核心就是 Flow Matching + Transformer (DiT)。
- **Sana:** 高效率高分辨率模型。

______________________________________________________________________

## 7. 代码实现简意 (PyTorch)

```python
import torch

class FlowMatching:
    def __init__(self, model):
        self.model = model

    def get_loss(self, x0):
        # 1. 采样时间 t
        t = torch.rand(x0.shape[0], 1, 1, 1, device=x0.device)

        # 2. 采样噪声 x1
        x1 = torch.randn_like(x0)

        # 3. 计算直线路径上的点 xt
        # xt = (1-t)x0 + t*x1
        xt = (1 - t) * x0 + t * x1

        # 4. 目标速度正是 (x1 - x0)
        target_velocity = x1 - x0

        # 5. 模型预测速度
        predicted_velocity = self.model(xt, t.squeeze())

        # 6. 计算 MSE Loss
        return torch.mean((predicted_velocity - target_velocity) ** 2)

    @torch.no_grad()
    def sample(self, shape, steps=20):
        # 从噪声开始 (t=1)
        x = torch.randn(shape)
        dt = 1.0 / steps

        # ODE 求解：x_{t-dt} = x_t - v * dt
        # 注意：采样是从 t=1 (噪声) 走向 t=0 (数据)
        for i in range(steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((shape[0],), t)

            v = self.model(x, t_tensor)
            x = x - v * dt  # 逆向积分

        return x
```

> **思考题：** 在采样代码中，为什么是用 `x - v * dt`？
> **答案：** 因为我们定义 $t=1$ 是噪声，$t=0$ 是数据。从 $1$ 走到 $0$ 是逆时间方向，所以 $\\Delta x = v \\cdot \\Delta t$，而 $\\Delta t = -dt$。
