# Flow Matching vs DDPM/Score SDE：数学原理的本质区别

## 〇、一句话概括

|  | DDPM / Score SDE | Flow Matching |
| --- | --- | --- |
| **核心思想** | 逐步添加噪声 → 学习**逆转噪声过程**（去噪） | 直接构造**噪声到数据的传输路径** → 学习沿路径的**速度场** |
| **数学对象** | 分数函数 $\nabla_x \log p_t(x)$（概率密度的梯度） | 速度场 $v_t(x)$（向量场 / 流） |

---

## 一、DDPM / Score SDE 的数学框架

### 1.1 前向过程（加噪）

定义一个连续时间的**随机微分方程（SDE）**：

$$dx = f(x,t)\,dt + g(t)\,dW_t$$

其中 $W_t$ 是布朗运动。典型选择（VP-SDE）：

$$dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW_t$$

- 从 $t=0$（数据 $x_0 \sim p_{\text{data}}$）到 $t=T$（近似纯噪声 $x_T \sim \mathcal{N}(0, I)$）
- 这是一个**随机扩散过程**，转移核为高斯：

$$q(x_t | x_0) = \mathcal{N}(x_t; \alpha_t x_0,\, \sigma_t^2 I)$$

### 1.2 逆向过程（生成）

Anderson (1982) 定理告诉我们，逆时间 SDE 为：

$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right] dt + g(t)\,d\bar{W}_t$$

> **关键**：需要知道**分数函数** $\nabla_x \log p_t(x)$
> 

### 1.3 训练目标

用神经网络 $s_\theta(x_t, t)$ 来近似分数函数：

$$\mathcal{L}*{\text{Score}} = \mathbb{E}*{t, x_0, \epsilon}\left[\left\| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0) \right\|^2\right]$$

由于 $q(x_t|x_0) = \mathcal{N}(\alpha_t x_0, \sigma_t^2 I)$，等价于**预测噪声**：

$$\boxed{\mathcal{L}*{\text{DDPM}} = \mathbb{E}*{t, x_0, \epsilon}\left[\left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2\right]}$$

### 1.4 采样

从 $x_T \sim \mathcal{N}(0,I)$ 出发，**逐步**求解逆向 SDE（随机）或对应的 **Probability Flow ODE**（确定性）：

$$dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right] dt$$

> 采样轨迹通常是**弯曲的**，需要较多步数。
> 

---

## 二、Flow Matching 的数学框架

### 2.1 核心出发点：构造 ODE 流

**直接**定义一个从 $t=0$（噪声）到 $t=1$（数据）的**常微分方程**：

$$\frac{dx_t}{dt} = v_t(x_t)$$

其中 $v_t$ 是**时变速度场**。

> ⚠️ 注意：**没有随机项 $dW_t$**，整个过程是确定性的 ODE，不是 SDE。
> 

### 2.2 条件流匹配（Conditional Flow Matching）

**问题**：我们无法直接获得边际速度场 $v_t(x)$（因为它依赖于未知的 $p_t(x)$）。

**解法**：对每个数据点 $x_1$ 分别定义一条**条件路径**：

$$\psi_t(x_0 | x_1): \quad x_t = (1-t)\,x_0 + t\,x_1$$

其中 $x_0 \sim \mathcal{N}(0, I)$（噪声），$x_1 \sim p_{\text{data}}$。

沿这条路径的**条件速度场**极其简单：

$$\boxed{u_t(x_t | x_1) = x_1 - x_0}$$

> 这就是一条**直线**！速度恒定，方向从噪声指向数据。
> 

### 2.3 训练目标

$$\boxed{\mathcal{L}*{\text{FM}} = \mathbb{E}*{t \sim U[0,1],\, x_0 \sim \mathcal{N}(0,I),\, x_1 \sim p_{\text{data}}}\left[\left\| v_\theta(x_t, t) - (x_1 - x_0) \right\|^2\right]}$$

其中 $x_t = (1-t)x_0 + tx_1$。

> **Lipman et al. (2023)** 证明：最小化条件损失等价于最小化（不可直接计算的）边际损失。
> 

### 2.4 采样

从 $x_0 \sim \mathcal{N}(0,I)$ 出发，求解 ODE：

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t)\,\Delta t$$

用 Euler 方法或高阶 ODE 求解器即可。

---

## 三、核心数学区别对比

```
DDPM / Score SDE                          Flow Matching
═══════════════════                       ═══════════════════

数据 x₀ ──加噪──→ 噪声 x_T              噪声 x₀ ──直线──→ 数据 x₁
         (SDE 前向)                              (ODE 流)
噪声 x_T ──去噪──→ 数据 x₀
         (逆向 SDE/ODE)

           ↓                                      ↓

学习：score ∇log p_t(x)               学习：velocity v_t(x)
      或等价的 ε(噪声)                       即 (x₁ - x₀)

           ↓                                      ↓

轨迹：弯曲 (curved)                    轨迹：尽可能直 (straight)
步数：多 (20-1000 步)                  步数：少 (10-50 步 典型)
```

### 对比表

| 维度 | DDPM / Score SDE | Flow Matching (Rectified Flow) |
| --- | --- | --- |
| **数学对象** | 随机微分方程 (SDE) | 常微分方程 (ODE) |
| **学习目标** | 分数 $\nabla_x \log p_t(x)$（对数密度梯度） | 速度场 $v_t(x)$（传输方向） |
| **回归目标** | 预测噪声 $\epsilon$ | 预测位移 $x_1 - x_0$ |
| **噪声调度** | 精心设计 $\beta(t)$ 或 $\alpha_t, \sigma_t$ | 简单线性插值 $x_t = (1-t)x_0 + tx_1$ |
| **前向过程** | 物理启发的扩散过程 | 几何启发的最优传输路径 |
| **传输路径** | 弯曲、间接 | 直线（rectified） |
| **训练随机性** | SDE 过程本身含随机性 | ODE 确定性，仅采样对有随机性 |
| **采样效率** | 弯曲路径 → 需要更多离散化步数 | 直线路径 → 更少步数即可精确积分 |
| **理论来源** | 非平衡热力学 / 朗之万动力学 | 连续归一化流 (CNF) / 最优传输 |
| **调度敏感度** | 对噪声调度非常敏感 | 路径构造天然简洁，超参数少 |

---

## 四、为什么直线路径更优？— 几何直觉

```
DDPM Probability Flow ODE 的轨迹:         Flow Matching 的轨迹:

  噪声                                      噪声
   ·                                          ·
    \\                                          \\
     \\                                          \\
      \\___                                       \\
          \\___                                    \\
              \\                                    \\
               ·                                    ·
             数据                                  数据

  弯曲 → 需要小步长才能精确跟踪             直线 → Euler 1步也有不错精度
```

**数学解释**：

ODE 数值积分的误差与**轨迹曲率**成正比。设真实轨迹为 $\phi_t$：

- **Euler 方法截断误差** $\propto \|\ddot{\phi}_t\| \cdot (\Delta t)^2$
- 直线轨迹：$\ddot{\phi}_t = 0$ → 截断误差最小化
- 弯曲轨迹：$\ddot{\phi}_t \neq 0$ → 需要更小的 $\Delta t$（更多步数）

---

## 五、更深层的联系与统一

### 5.1 DDPM 的 Probability Flow ODE 其实也是一个流

DDPM 的 Probability Flow ODE：

$$dx = \underbrace{\left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right]}_{\text{有效速度场 } \tilde{v}_t(x)} dt$$

这本质上也定义了一个速度场。**但**这个速度场是**由扩散前向过程导出的**，路径形状由 $\beta(t)$ 决定，通常不是直线。

### 5.2 统一视角

$$\text{Score Matching} \xleftarrow{\text{重参数化}} \text{速度场} \xrightarrow{\text{最优传输}} \text{Flow Matching}$$

实际上，DDPM 的 $\epsilon$-prediction 和 FM 的 $v$-prediction 可以通过简单线性变换互转：

$$v_t = \alpha_t \epsilon - \sigma_t x_0 \quad \Longleftrightarrow \quad \epsilon = \frac{\alpha_t v_t + \sigma_t x_t}{\alpha_t^2 + \sigma_t^2}$$

**真正的区别不在参数化方式，而在于**：

1. **路径的选择**（弯曲 vs 直线）
2. **噪声调度的角色**（核心 vs 边缘化）
3. **理论出发点**（逆向 SDE vs 构造性 ODE）

---

## 六、Rectified Flow 的 Reflow 操作

**Rectified Flow**（Liu et al.）进一步提出了一个独特的**迭代拉直**操作：

```
第1轮：训练 FM → 得到 v₁(x,t)
       用 v₁ 生成配对 (x₀, x₁) → 轨迹可能还有弯曲

第2轮：用新配对 (x₀, x₁) 重新训练 FM → 得到 v₂(x,t)
       轨迹更直了

第k轮：重复 → 轨迹越来越接近直线
       → 最终几乎1步就能生成
```

这是 Flow Matching 独有的——在 DDPM 框架下没有对应操作。

---

## 七、总结

> **DDPM/Score SDE** 问的是："我已经有一个物理扩散过程，如何**逆转**它？"
→ 学习对数密度梯度（score），逆转随机扩散
> 
> 
> **Flow Matching** 问的是："从噪声到数据，**最直接的路怎么走？**"
> → 学习速度场，沿（接近）直线传输概率质量
> 

**本质区别**：**路径的选择自由度**。DDPM 的路径被前向扩散过程锁定（弯曲），而 Flow Matching 将路径设计视为可优化的自由度，选择了最优传输意义下的直线路径，从而获得了更高的采样效率和更简洁的数学形式。