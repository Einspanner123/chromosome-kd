## Diffusion ODE 理论详解：从随机轨迹到确定性流

## 1. 为什么需要 ODE 视角？

在 [DDPM推导.md](./DDPM推导.md) 中，扩散过程被定义为离散的马尔可夫链。每一步去噪都带有随机性（注入了 $\\sigma_t \\mathbf{z}$）。

**ODE 视角的意义**：

1. **确定性映射**：将“概率分布的演变”转化为“粒子运动的轨迹”。给定一个初始噪声 $\\mathbf{x}\_T$，通过 ODE 求解，它将唯一、确定地对应到一张图像 $\\mathbf{x}\_0$。
2. **采样加速**：既然是 ODE，我们就可以利用成熟的数值积分器（如 Euler, Runge-Kutta, Heun）来求解，从而用极少的步数完成生成。
3. **可逆性**：ODE 是可逆的。你可以把图片“精确”地编码回噪声，再“精确”地还原回来。

______________________________________________________________________

## 2. 核心概念：概率流 ODE (Probability Flow ODE)

Song Yang 等人在 2021 年证明，任何扩散过程（SDE）都对应一个等价的 **概率流 ODE**，它们在每一个时间点 $t$ 的边缘分布 $p_t(\\mathbf{x})$ 完全一致。

### 2.1 连续扩散方程 (SDE)

前向加噪过程可以看作是一个随机微分方程：
$$d\\mathbf{x} = \\mathbf{f}(\\mathbf{x}, t)dt + g(t)d\\mathbf{w}$$
其中 $\\mathbf{f}$ 是漂移项（让数据向中心收缩），$g(t)$ 是扩散系数（注入噪声）。

### 2.2 概率流 ODE 公式

对应的确定性轨迹方程为：
$$\\frac{d\\mathbf{x}}{dt} = \\mathbf{f}(\\mathbf{x}, t) - \\frac{1}{2} g(t)^2 \\nabla\_\\mathbf{x} \\log p_t(\\mathbf{x})$$

- **$\\nabla\_\\mathbf{x} \\log p_t(\\mathbf{x})$**：这就是著名的 **Score Function（分数函数）**。
- **直观理解**：ODE 的运动方向由两部分组成：一部分是向中心收缩的力，另一部分是沿着概率密度梯度上升的力。

______________________________________________________________________

## 3. Score Function 与神经网络

你会发现，ODE 公式里唯一的未知数是 $\\nabla\_\\mathbf{x} \\log p_t(\\mathbf{x})$。

#### 3.1 分数与噪声的等价性证明

为什么预测噪声就是在估计分数？我们看前向加噪公式：
$$\\mathbf{x}\_t = \\sqrt{\\bar{\\alpha}\_t}\\mathbf{x}\_0 + \\sqrt{1 - \\bar{\\alpha}\_t}\\boldsymbol{\\epsilon}, \\quad \\boldsymbol{\\epsilon} \\sim \\mathcal{N}(0, \\mathbf{I})$$

对于给定的 $\\mathbf{x}\_0$，$\\mathbf{x}\_t$ 的条件分布为 $\\mathcal{N}(\\sqrt{\\bar{\\alpha}\_t}\\mathbf{x}\_0, (1 - \\bar{\\alpha}\_t)\\mathbf{I})$。其对数概率密度为：
$$\\log p(\\mathbf{x}\_t|\\mathbf{x}\_0) = -\\frac{1}{2}\\log(2\\pi(1-\\bar{\\alpha}\_t)) - \\frac{|\\mathbf{x}\_t - \\sqrt{\\bar{\\alpha}\_t}\\mathbf{x}\_0|^2}{2(1 - \\bar{\\alpha}\_t)}$$

求梯度（分数）：
$$\\nabla\_{\\mathbf{x}\_t} \\log p(\\mathbf{x}\_t|\\mathbf{x}\_0) = - \\frac{\\mathbf{x}\_t - \\sqrt{\\bar{\\alpha}\_t}\\mathbf{x}\_0}{1 - \\bar{\\alpha}\_t}$$

由于 $\\mathbf{x}\_t - \\sqrt{\\bar{\\alpha}\_t}\\mathbf{x}\_0 = \\sqrt{1 - \\bar{\\alpha}_t}\\boldsymbol{\\epsilon}$，代入得：
$$\\nabla_{\\mathbf{x}\_t} \\log p(\\mathbf{x}\_t|\\mathbf{x}\_0) = - \\frac{\\sqrt{1 - \\bar{\\alpha}\_t}\\boldsymbol{\\epsilon}}{1 - \\bar{\\alpha}\_t} = - \\frac{\\boldsymbol{\\epsilon}}{\\sqrt{1 - \\bar{\\alpha}\_t}}$$

**核心结论**：模型预测的噪声 $\\boldsymbol{\\epsilon}_\\theta$ 实际上就是 **负的标准化分数**。
$$\\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}\_t, t) \\approx - \\sqrt{1 - \\bar{\\alpha}_t} \\nabla_{\\mathbf{x}\_t} \\log p_t(\\mathbf{x}\_t)$$

______________________________________________________________________

## 4. 数值求解器（Numerical Solvers）

既然扩散过程变成了 $\\frac{d\\mathbf{x}}{dt} = \\mathbf{V}(\\mathbf{x}, t)$，我们可以使用任何数值积分方法来求解：

### 4.1 Euler Method (一阶)

最简单的方法，假设步长内速度不变：
$$\\mathbf{x}\_{t-h} = \\mathbf{x}\_t - h \\cdot \\mathbf{V}(\\mathbf{x}\_t, t)$$
*对应 DDIM 的最基础形式，步数少时误差较大。*

### 4.2 Heun's Method (二阶)

先预估一步，再用终点的速度修正起点的速度：

1. 预估：$\\tilde{\\mathbf{x}}\_{t-h} = \\mathbf{x}\_t - h \\cdot \\mathbf{V}(\\mathbf{x}\_t, t)$
2. 修正：$\\mathbf{x}\_{t-h} = \\mathbf{x}\_t - \\frac{h}{2} \[\\mathbf{V}(\\mathbf{x}_t, t) + \\mathbf{V}(\\tilde{\\mathbf{x}}_{t-h}, t-h)\]$
   *对应工业界常用的采样器（如 Heun, DPM-Solver 2），能显著提升画质。*

______________________________________________________________________

## 5. ODE Inversion：确定性编辑的基石

由于 ODE 是确定性的，我们可以反向运行求解器（从 $t=0$ 到 $t=T$）：

1. 输入一张真实的图片 $\\mathbf{x}\_0$。
2. 通过 ODE 向前积分，得到它对应的“原始噪声” $\\mathbf{x}\_T$。
3. **应用**：如果你修改了这个 $\\mathbf{x}\_T$（比如加一点偏移），再解回 $\\mathbf{x}\_0$，你就能实现**保持身份不变的图像编辑**。

______________________________________________________________________

## 6. 代码实现：极简 Euler 采样器

```python
@torch.no_grad()
def ode_sampler(model, x_T, steps=50):
    x = x_T
    dt = 1.0 / steps

    for i in range(steps):
        t = 1.0 - i * dt  # 从 1 降到 0

        # 1. 预测噪声
        eps = model(x, t)

        # 2. 计算速度向量 (Probability Flow ODE 项)
        # 这里简化了系数，实际需根据具体 SDE 类型计算
        v = compute_ode_velocity(eps, x, t)

        # 3. Euler 更新
        x = x - v * dt

    return x
```

______________________________________________________________________

## 7. 总结

| 特性         | DDPM (离散/随机)     | Diffusion ODE (连续/确定)             |
| :----------- | :------------------- | :------------------------------------ |
| **数学工具** | 马尔可夫链、贝叶斯   | 微分方程、向量场                      |
| **生成轨迹** | 随机、抖动           | 平滑、确定                            |
| **采样步数** | 多 (1000+)           | 少 (20-50)                            |
| **核心目标** | 预测噪声 $\\epsilon$ | 预测分数 $\\nabla \\log p$ 或速度 $v$ |

______________________________________________________________________

*下一章预告：我们将推导如何从 SDE 导出概率流 ODE，并手写一个简单的 Euler 求解器。*
