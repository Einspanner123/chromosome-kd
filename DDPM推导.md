# DDPM 从基础概率到公式推导（一步一步）

本笔记面向学过高等数学和基础概率论的读者（本科生水平），手把手从最简单的概率工具出发，推导 DDPM（Denoising Diffusion Probabilistic Models）的核心公式与训练目标。

我们尽量避免高深测度论，使用**向量正态分布**与**贝叶斯公式**的基本性质即可读懂。

- **目标**：明确每个变量和参数的含义；从前向扩散到逆向去噪的完整推导；最终得到可实现的训练与采样过程。
- **结构**：记号与参数 $\to$ 概率工具箱 $\to$ 正向扩散 $\to$ 后验与逆向过程 $\to$ 训练目标（ELBO） $\to$ 算法伪代码

---

## 1. 记号与参数

| 符号 | 含义 | 备注 |
| :--- | :--- | :--- |
| $\mathbf{x}_0$ | 原始数据（例如图像） | 维度为 $\mathbb{R}^d$，通常归一化到 $[-1, 1]$ |
| $\mathbf{x}_t$ | 第 $t$ 步的随机变量（含噪图像） | $t \in \{1, 2, \dots, T\}$。它是前向第 $t$ 步的输出，也是逆向第 $t$ 步的输入 |
| $T$ | 扩散总步数 | 经典设置 $T=1000$ |
| $\beta_t$ | 第 $t$ 步注入的噪声方差 | 满足 $\beta_t = 1 - \alpha_t$ |
| $\alpha_t$ | $1 - \beta_t$ | 第 $t$ 步保留的信号比例 |
| $\bar{\alpha}_t$ | $\prod_{s=1}^{t} \alpha_s$ | 从 $0$ 到 $t$ 累积保留的信号比例 |
| $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | 标准正态噪声 | 这里的 $\mathbf{I}$ 是单位矩阵。因为图像是多维向量，方差演变为协方差矩阵；$\mathbf{I}$ 表示各维度独立且方差均为 1 |
| $q(\cdot)$ | 真实数据驱动的“前向过程” | **加噪过程**。代表基于真实数据的演化路径，是固定的、不可学习的“真相” |
| $p_\theta(\cdot)$ | 模型参数 $\theta$ 定义的“逆向过程” | **去噪过程**。代表神经网络试图模拟的生成路径，是需要学习的“模型” |
| $\theta$ | 神经网络的所有可学习参数 | 包含 U-Net 中所有的卷积权重、偏置等。训练的目标就是寻找最优的 $\theta$ |

**补充概念：什么是单位方差（Unit Variance）？**
在多维向量语境下，指随机变量的协方差矩阵为单位矩阵 $\mathbf{I}$。这意味着数据每个维度的方差均为 1 且相互独立。在 DDPM 中，通过维持单位方差，可以保证数据在加噪/去噪过程中数值范围始终稳定，不会发生数值爆炸或坍缩。

**符号小贴士**：
- $\sigma$：标准差（标量）
- $\sigma^2$：方差（标量）
- $\boldsymbol{\Sigma}$ (大写 Sigma)：协方差矩阵（矩阵）
- $\propto$：正比于。推导中常用来忽略与自变量无关的归一化常数，只保留核心项。
- **关于下标 $t$**：在逆向分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 中，下标 $t$ 代表“当前正在执行第 $t$ 步去噪动作”。虽然结果是 $t-1$，但参数（如 $\tilde{\boldsymbol{\mu}}_t$）通常用当前步 $t$ 索引。

---

## 2. 概率工具箱（用到就够）

1.  **正态分布的重参数化（Reparameterization Trick）**：
    若 $z \sim \mathcal{N}(\mu, \sigma^2)$, 我们可以写成 $z = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, 1)$。
    - **为什么能写成线性方程？**：正态分布具有“仿射变换不变性”。对标准正态分布 $\epsilon$ 进行缩放（乘 $\sigma$）和位移（加 $\mu$），结果依然服从正态分布，且均值和方差正好对应。
    - **为什么 $\epsilon$ 是标准正态？**：因为它充当了“噪声源”，是一个不含任何参数（均值0，方差1）的纯随机变量。
    - **核心作用**：将“随机性”与“参数”剥离。
        - **为什么直接采样不可导？**：梯度需要确定的解析映射。直接采样 $z \sim \mathcal{N}(\mu, \sigma^2)$ 类似于“掷色子”，输出值与参数之间没有确定的函数关系，导致梯度无法在计算图中回传。
        - **重参数化的妙处**：将 $z$ 写成 $z = \mu + \sigma \epsilon$ 后，$z$ 变成了关于 $\mu$ 和 $\sigma$ 的**确定性可微函数**。此时随机性被封装在外部变量 $\epsilon$ 中，梯度可以通过 $\mu$（偏导为 1）和 $\sigma$（偏导为 $\epsilon$）正常传导，从而实现对网络参数的训练。

2.  **独立高斯的线性叠加仍为高斯**：
    若 $X \sim \mathcal{N}(\mu_X, \sigma_X^2), Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$ 且独立，则 $aX + bY \sim \mathcal{N}(a\mu_X + b\mu_Y, a^2\sigma_X^2 + b^2\sigma_Y^2)$。
    - **方差运算法则**：
        1. $\text{Var}(kX) = k^2 \text{Var}(X)$
        2. $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ （仅当 $X, Y$ 独立时）
    - **协方差矩阵（Covariance Matrix）**：
        对于多维随机向量 $\mathbf{X}$，其方差表现为矩阵形式：$\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$。
        - 对角线元素是各维度的方差；非对角线元素是维度间的协方差（相关性）。
        - 当 $\boldsymbol{\Sigma} = \mathbf{I}$ 时，意味着各维度独立且方差均为 1。
    - **注意**：方差的系数是平方关系。这就是为什么前向过程系数使用 $\sqrt{\alpha_t}$ 和 $\sqrt{1-\alpha_t}$，因为它们的平方和正好为 1。

3.  **贝叶斯公式与条件概率符号**：
    标准形式：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$。
    - **关于符号的详细解释**：
        - `;`（分号）：用于区分**随机变量**与**分布参数**。例如在 $q(\mathbf{x}_t; \mu, \sigma^2)$ 中，分号前的 $\mathbf{x}_t$ 是自变量，分号后是描述它的均值和方差。
        - `,`（逗号）：
            1. **参数分隔**：在分布参数列表中隔开不同参数，如 $\mathcal{N}(\text{mean}, \text{variance})$。
            2. **联合条件**：在条件概率 `|` 右侧表示“且”，如 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$ 表示同时给定 $\mathbf{x}_t$ 和 $\mathbf{x}_0$。
    - **演变**：在 DDPM 中，我们常处理 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$。这里 `|` 后面的 $\mathbf{x}_t, \mathbf{x}_0$ 都是“已知条件”。可以理解为：在给定当前图 $\mathbf{x}_t$ 和最初始图 $\mathbf{x}_0$ 的情况下，推测上一步图 $\mathbf{x}_{t-1}$ 的分布。

4.  **KL 散度（Kullback-Leibler Divergence）**：
    衡量两个概率分布 $P$ 和 $Q$ 的差异。
    - **符号解释**：$D_{KL}(P || Q)$ 中的 `||` 是数学中的标准分割符，读作“$P$ 相对于 $Q$ 的散度”。注意 KL 散度是不对称的，即 $D_{KL}(P || Q) \neq D_{KL}(Q || P)$。
    - **为什么等价于 MSE？（推导过程）**：
        假设有两个高斯分布，它们的方差相同均为 $\sigma^2\mathbf{I}$，均值分别为 $\mu_1$ 和 $\mu_2$。
        KL 散度的定义为：
        $$D_{KL}(p_1 || p_2) = \int p_1(x) \log \frac{p_1(x)}{p_2(x)} dx = \mathbb{E}_{x \sim p_1} [\log p_1(x) - \log p_2(x)]$$
        代入 $d$ 维高斯分布的概率密度函数 $p(x) = \frac{1}{(2\pi\sigma^2)^{d/2}} \exp(-\frac{\|x-\mu\|^2}{2\sigma^2})$：
         1. $\log p_1(x) = -\frac{d}{2}\log(2\pi\sigma^2) - \frac{\|x-\mu_1\|^2}{2\sigma^2}$
         2. $\log p_2(x) = -\frac{d}{2}\log(2\pi\sigma^2) - \frac{\|x-\mu_2\|^2}{2\sigma^2}$
         相减（常数项抵消）得：
         $$\log p_1(x) - \log p_2(x) = \frac{1}{2\sigma^2} (\|x-\mu_2\|^2 - \|x-\mu_1\|^2)$$
         对 $x \sim p_1$ 求期望，关键在于展开 $\|x-\mu_2\|^2$。我们利用恒等式 $x-\mu_2 = (x-\mu_1) + (\mu_1-\mu_2)$：
         $$\|x-\mu_2\|^2 = \|(x-\mu_1) + (\mu_1-\mu_2)\|^2 = \|x-\mu_1\|^2 + \|\mu_1-\mu_2\|^2 + 2(x-\mu_1)^T(\mu_1-\mu_2)$$
         将此式代入 $D_{KL}$ 公式中：
         $$D_{KL} = \frac{1}{2\sigma^2} \mathbb{E}_{x \sim p_1} [ \underbrace{\|x-\mu_1\|^2 + \|\mu_1-\mu_2\|^2 + 2(x-\mu_1)^T(\mu_1-\mu_2)}_{\|x-\mu_2\|^2} - \|x-\mu_1\|^2 ]$$
         $$D_{KL} = \frac{1}{2\sigma^2} \mathbb{E}_{x \sim p_1} [ \|\mu_1-\mu_2\|^2 + 2(x-\mu_1)^T(\mu_1-\mu_2) ]$$
         由于 $\mathbb{E}_{x \sim p_1}[x] = \mu_1$，所以交叉项的期望 $\mathbb{E}[2(x-\mu_1)^T(\mu_1-\mu_2)] = 0$。而 $\|\mu_1-\mu_2\|^2$ 是常数。
         最终剩：
         $$D_{KL}(p_1 || p_2) = \frac{1}{2\sigma^2} \|\mu_1 - \mu_2\|^2$$
88.    - **结论**：在方差固定的情况下，最小化 KL 散度完全等价于最小化均值之间的欧式距离平方，即**均方误差 (Mean Squared Error, MSE)**。这就是为什么 DDPM 的损失函数最后变成了一个简单的 $L_2$ 损失。

---

## 3. 正向扩散（Forward Process）

### 3.1 单步转移
我们将每一步的加噪定义为条件高斯分布：
$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_{t-1}, (1 - \alpha_t)\mathbf{I})$$
利用重参数化技巧，可以写成：
$$\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**设计初衷：为什么系数是 $\sqrt{\alpha_t}$？**
1. **方差保持（Variance Preservation）**：假设 $\mathbf{x}_{t-1}$ 具有单位方差，则 $\text{Var}(\mathbf{x}_t) = (\sqrt{\alpha_t})^2 \text{Var}(\mathbf{x}_{t-1}) + (\sqrt{1-\alpha_t})^2 \text{Var}(\boldsymbol{\epsilon}) = \alpha_t + (1-\alpha_t) = 1$。这保证了在加噪过程中数据不会发生数值爆炸或坍缩。
2. **信号衰减**：$\sqrt{\alpha_t} < 1$ 像是一个“遗忘系数”，每一步都在遗忘一部分旧信息，同时通过注入等量方差的噪声来维持整体能量守恒。

### 3.2 任意步转移（一步到位）
我们不需要迭代采样，可以直接从 $\mathbf{x}_0$ 得到 $\mathbf{x}_t$。

**推导过程：**
1. 根据单步公式：$\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}$
2. 展开一步：$\mathbf{x}_t = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}) + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1}$
   $= \sqrt{\alpha_t \alpha_{t-1}}\mathbf{x}_{t-2} + (\sqrt{\alpha_t(1-\alpha_{t-1})}\boldsymbol{\epsilon}_{t-2} + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_{t-1})$
3. **利用高斯叠加性质**：两个独立高斯分布 $a\boldsymbol{\epsilon}_1 + b\boldsymbol{\epsilon}_2$ 的结果仍为高斯分布，其方差为 $a^2 + b^2$。
   - 这里新噪声的方差为：$\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t \alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t \alpha_{t-1}$。
   - 所以：$\mathbf{x}_t = \sqrt{\alpha_t \alpha_{t-1}}\mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}}\bar{\boldsymbol{\epsilon}}$
4. 递归 $t$ 次，定义 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，最终得到：
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

写成分布形式：
$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

**直观理解**：随着 $t \to T$，$\bar{\alpha}_t \to 0$，均值趋向 0，方差趋向 1，$\mathbf{x}_T$ 变成了纯高斯噪声。

---

## 4. 逆向过程与后验分布

我们的目标是训练一个模型 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 来逆转扩散过程。

### 4.1 真实的后验分布 $q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)$
虽然直接求 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 很困难（因为需要积分掉 $\mathbf{x}_0$），但如果我们**已知 $\mathbf{x}_0$**，后验分布是可以算出来的！

**推导过程（条件贝叶斯）：**
1. **基础公式**：$P(A|B, C) = \frac{P(B|A, C)P(A|C)}{P(B|C)}$（即在已知背景 $C$ 的情况下，对 $A, B$ 使用贝叶斯定理）。
2. **变量代入**：令 $A = \mathbf{x}_{t-1}, B = \mathbf{x}_t, C = \mathbf{x}_0$。
3. **得到公式**：
$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}$$
4. **化简**：由于扩散过程是马尔可夫链，已知 $\mathbf{x}_{t-1}$ 后 $\mathbf{x}_t$ 与 $\mathbf{x}_0$ 独立，即 $q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t | \mathbf{x}_{t-1})$。
    - **深度解析**：为什么只简化了这一项？因为只有在“已知中介 $\mathbf{x}_{t-1}$”时，前后两代才独立。而在 $q(\mathbf{x}_{t-1}|\mathbf{x}_0)$ 和 $q(\mathbf{x}_t|\mathbf{x}_0)$ 中，由于中介状态未知，初始状态 $\mathbf{x}_0$ 依然是推导当前状态的唯一依据，必须保留。
    - **递归视角**：由于 $\mathbf{x}_t$ 是由 $\mathbf{x}_{t-1}$ 直接递推得到的，$\mathbf{x}_0$ 对 $\mathbf{x}_t$ 的所有影响都已经完整封装在了 $\mathbf{x}_{t-1}$ 的数值中。因此，已知 $\mathbf{x}_{t-1}$ 后，$\mathbf{x}_0$ 变成了冗余信息。
    - **注意**：$q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t | \mathbf{x}_{t-1})$ 是**严格等式**。因为噪声 $\boldsymbol{\epsilon}_{t-1}$ 是独立生成的，$\mathbf{x}_0$ 不会提供任何关于 $\mathbf{x}_t$ 的额外信息。保留 $\mathbf{x}_0$ 只会增加计算复杂度，而不会提高预测准确度。

带入正态分布公式并配方（Completing the Square），可以证明这也是一个高斯分布：
$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$$

**详细推导过程：**
为什么只关注指数项？因为高斯分布的均值和方差完全由其指数部分的二次型决定。
- **分母去哪了？** 根据贝叶斯公式，分母 $q(\mathbf{x}_t|\mathbf{x}_0)$ 转化到指数上后是减法项。由于它不含变量 $\mathbf{x}_{t-1}$，对配方（求均值和方差）没有任何贡献，可以视作常数丢弃。
- **核心逻辑**：利用 $f(x) \propto \exp(-\frac{(x-\mu)^2}{2\sigma^2})$，我们只关注指数项中关于 $\mathbf{x}_{t-1}$ 的二次项和一次项。
1. **分子项 1** ($q(\mathbf{x}_t|\mathbf{x}_{t-1})$): $\exp\left( -\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{2(1-\alpha_t)} \right)$ —— **注意：单步转移用 $\alpha_t$**
2. **分子项 2** ($q(\mathbf{x}_{t-1}|\mathbf{x}_0)$): $\exp\left( -\frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{2(1-\bar{\alpha}_{t-1})} \right)$ —— **注意：从 0 开始的累积转移用 $\bar{\alpha}_{t-1}$**
3. **合并指数项**（忽略与 $\mathbf{x}_{t-1}$ 无关的常数）：
   $$\text{Exp} \propto -\frac{1}{2} \left[ \frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} \right]$$
4. **展开并提取 $\mathbf{x}_{t-1}$ 的系数**：
   - $\mathbf{x}_{t-1}^2$ 的系数 $A = \frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha}_{t-1}} = \frac{\alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1-\bar{\alpha}_{t-1})} = \frac{1-\bar{\alpha}_t}{\beta_t(1-\bar{\alpha}_{t-1})}$
   - $\mathbf{x}_{t-1}$ 的一次项系数 $B = \frac{\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}\mathbf{x}_0$
5. **根据高斯分布性质 $\mu = B/A$ 和 $\sigma^2 = 1/A$**：
    - **原理**：标准高斯指数项为 $-\frac{1}{2\sigma^2}(x^2 - 2\mu x + \dots)$。将其与配方结果 $-\frac{1}{2}(Ax^2 - 2Bx + \dots)$ 对比，立得 $\sigma^2 = 1/A, \mu = B/A$。
    - 得到后验方差 $\tilde{\beta}_t = 1/A = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$
    - 得到后验均值 $\tilde{\boldsymbol{\mu}}_t = B/A = \dots$ （化简后得到下方公式）

**关键公式（后验均值）：**
$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t$$
- **直观理解**：后验均值是原始图像 $\mathbf{x}_0$ 和当前含噪图像 $\mathbf{x}_t$ 的**加权平均**。它告诉我们，为了回到上一步，我们需要在“保留现状”和“回归原图”之间做一个权衡。
**关键公式（后验方差）：**
$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

### 4.2 模型的参数化
既然真实的后验是高斯，我们的模型 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 也应该建模为高斯：
$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$$
我们希望 $\boldsymbol{\mu}_\theta$ 尽可能接近 $\tilde{\boldsymbol{\mu}}_t$。

**如何设计网络输出？**
观察 $\tilde{\boldsymbol{\mu}}_t$ 的公式，它依赖于 $\mathbf{x}_t$ 和 **未知的 $\mathbf{x}_0$**。在推理生成时，我们无法直接获得 $\mathbf{x}_0$。
1. **反解 $\mathbf{x}_0$**：根据 3.2 节的前向公式，我们可以用 $\mathbf{x}_t$ 和噪声 $\boldsymbol{\epsilon}$ 反解出 $\mathbf{x}_0$：
   $$\mathbf{x}_0 = \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$
2. **代入均值公式**：将这个 $\mathbf{x}_0$ 代入 $\tilde{\boldsymbol{\mu}}_t$ 的公式，经过化简得到：
   $$\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$

**深刻洞察**：
这一步代入将“预测图像”的任务巧妙转化为了“预测噪声”的任务。
- 模型不再需要直接去猜上一时刻的图像长什么样。
- 模型只需要观察 $\mathbf{x}_t$，并预测出**前向传播中注入的那堆噪声 $\boldsymbol{\epsilon}$**。
- 一旦噪声预测准确，通过数学公式即可完美回退到上一状态。这正是为什么 DDPM 的 U-Net 预测目标通常是噪声的原因。

这告诉我们：**要预测均值 $\tilde{\boldsymbol{\mu}}_t$，本质上就是要预测噪声 $\boldsymbol{\epsilon}$。**
所以，我们设计一个神经网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 来预测添加到图像上的噪声。
于是模型的均值定义为：
$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

**为什么要预测均值而不是方差？**
1. **主导地位**：均值 $\boldsymbol{\mu}$ 决定了去噪的方向和图像的结构；方差 $\sigma^2$ 仅决定了生成过程中的随机噪声强度。
2. **工程简化**：DDPM 发现将方差固定为常数（如 $\sigma_t^2 = \beta_t$）已能获得极佳效果。
3. **训练稳定性**：最小化均值之间的 KL 散度等价于 MSE 损失，这让训练变得异常简单且稳定。

---

## 5. 训练目标（Loss Function）

**概念补丁：**
- **极大似然估计 (Maximum Likelihood Estimation, MLE)**：通过调整模型参数，使得观测数据在模型下出现的概率（似然）达到最大的方法。
- **变分推断 (Variational Inference, VI)**：当直接计算复杂的后验分布或似然概率涉及高维积分（数学上不可积或计算量巨大）时，通过引入一个简单的参考分布来寻找原目标近似解的一种统计学方法。
- **变分下界 (Variational Lower Bound, VLB)** / **证据下界 (Evidence Lower Bound, ELBO)**：这是同一个概念。它为对数似然提供了一个可以计算的下限。在优化过程中，我们通过最大化这个下界来间接实现极大似然估计。

**直观类比：**
- **极大似然估计 (MLE)**：像是在一堆陶器碎片中寻找最可能的原始模具。
- **变分推断 (VI)**：因为原始地图太复杂画不出来，所以先画一张简单的草图，并不断修改让它接近真迹。
- **变分下界 (VLB)**：因为无法直接衡量草图和真迹的差距，所以找了一个“及格线”，只要超过这个线，草图就一定在变好。

**数学严谨性补丁：**
1. **MLE 实例**：对于样本 $\{x_i\} \sim \mathcal{N}(\mu, \sigma^2)$，其对数似然为 $\ell(\mu) = \sum \log p(x_i|\mu)$。最大化 $\ell(\mu)$ 对应于寻找使数据出现概率最大的 $\mu^*$。**注意**：在 DDPM 中，$p$ 代表我们要训练的模型，而 $q$ 是预设的、不可变的加噪过程，因此我们只对 $p$ 进行极大似然估计。
2. **VLB 的数学起源**：利用 Jensen 不等式 $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$。
   - **推导逻辑**：
     1. 首先利用边缘化（全概率公式）：$p(\mathbf{x}_0) = \int p(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}$。
     2. 引入辅助分布 $q$（分子分母同乘）：$p(\mathbf{x}_0) = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} d\mathbf{x}_{1:T}$。
     3. 取对数并应用 Jensen 不等式：
   $$\log p(\mathbf{x}_0) = \log \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} d\mathbf{x}_{1:T} \geq \mathbb{E}_q \left[ \log \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] = \text{VLB}$$
   - **符号说明**：$0:T$ 表示序列 $\{\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_T\}$。$p(\mathbf{x}_{0:T})$ 是整条扩散路径的联合分布。引入 $q$ 是为了将无法直接计算的高维积分转化为可以通过采样估算的期望。
3. **KL 散度的关系**：$\log p(\mathbf{x}_0) = \text{VLB} + D_{KL}(q \parallel p)$。由于 KL 散度始终 $\geq 0$，因此优化 VLB 就是在不断逼近真实的对数似然。
4. **积分归一化（Normalization）**：为什么 $\int q(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T} = 1$？
   - **概率公理**：任何概率密度函数在其定义域上的积分（或求和）必须为 1。
   - **直观理解**：$q(\mathbf{x}_{1:T}|\mathbf{x}_0)$ 是关于路径 $\mathbf{x}_{1:T}$ 的条件概率分布。它涵盖了所有可能的中间状态。既然是“概率”，所有可能性加起来必然是 100%（即 1）。
   - **推导技巧**：在 5.1 节的第 1 步中，我们将 $\log p(\mathbf{x}_0)$ 乘以 $\int q \dots d\mathbf{x}_{1:T}$。因为 $\log p(\mathbf{x}_0)$ 不含积分变量 $\mathbf{x}_{1:T}$，它相当于一个常数。这种“无中生有”的操作是为了在 $\log$ 外部构造出关于 $q$ 的期望 $\mathbb{E}_q$。
5. **期望与积分的关系**：
   - **定义**：对于连续随机变量 $X \sim p(x)$，函数 $f(X)$ 的期望定义为 $\mathbb{E}_{X \sim p}[f(X)] = \int f(x) p(x) dx$。
   - **区别**：$\int p(x) dx$ 是在求概率总和（恒等于 1）；而 $\int f(x) p(x) dx$ 是在求 $f(x)$ 的加权平均值。
   - **在 DDPM 中的意义**：公式推导中频繁切换 $\mathbb{E}_q[\dots]$ 和 $\int q(\dots) \dots d\mathbf{x}$。写成期望符号 $\mathbb{E}$ 的核心目的有两个：
     1. **数学简化**：隐藏复杂的积分符号，让公式更清晰。
     2. **采样指导**：期望符号告诉我们，这个积分可以通过从 $q$ 中采样 $\mathbf{x}_t$ 并取平均值来近似计算。这就是为什么训练算法第一步是“采样 $\mathbf{x}_0$”和“采样噪声 $\epsilon$”。
6. **蒙特卡洛（Monte Carlo, MC）方法**：
    - **核心思想**：用“统计平均”近似“解析积分”。
    - **数学保障**：**大数定律（Law of Large Numbers）**。当采样量 $N \to \infty$ 时，采样平均值 $\frac{1}{N}\sum f(x_i)$ 几乎处处收敛于期望值 $\mathbb{E}[f(X)]$。
    - **维度的诅咒**：在高维空间（如图像空间），传统的数值积分（如网格法）计算量随维度指数爆炸；而蒙特卡洛方法的误差收敛速度为 $O(1/\sqrt{N})$，与维度无关。
    - **DDPM 的极致简化**：在训练 Loss $L_{simple}$ 中，我们要对 $t \sim \text{Uniform}$ and $\epsilon \sim \mathcal{N}$ 求期望。实际代码中，每个 step 我们只采样**一个**随机的 $t$ 和**一个**随机的 $\epsilon$。
      - *疑问*：只用一个样本点，梯度不准怎么办？
      - *答案*：虽然单次采样的梯度具有很大的随机性（噪声），但由于神经网络训练是数万次迭代的累积过程，这些采样噪声在时间轴上会互相抵消。这本质上是在利用 **随机梯度下降（SGD）** 的特性来自动完成高维积分的累加。
 7. **全概率公式（Law of Total Probability）**：
     - **概念**：如果一个事件 $A$ 的发生依赖于一组互斥且完备的中间状态 $B_i$，则 $A$ 的总概率等于在各状态下发生概率的加权和：$P(A) = \sum P(A|B_i)P(B_i)$。
     - **从求和到积分**：当中间状态 $B$ 是连续变量（如像素值）时，无限细分的求和就演变成了积分：$p(a) = \int p(a|b)p(b) db$。
     - **边缘化（Marginalization）**：这个术语源于统计学。想象一个关于 X 和 Y 的二维概率分布表，如果你把某一行（或某一列）的所有概率加起来，结果通常写在表格的**边缘（Margin）**。这一过程本质上是“消除”掉你不关心的变量（通过积分/求和），从而得到目标变量的单一分布。
     - **在 DDPM 中的应用**：数据 $\mathbf{x}_0$ 的概率 $p(\mathbf{x}_0)$ 可以看作是所有可能的扩散/去噪路径 $\mathbf{x}_{1:T}$ 的积分总和：
       $$p(\mathbf{x}_0) = \underbrace{\int \dots \int}_{T \text{ 次积分}} p(\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_T) d\mathbf{x}_1 \dots d\mathbf{x}_T = \int p(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}$$
       通过对中间路径 $\mathbf{x}_{1:T}$ 进行边缘化，我们得到了初始状态的边缘分布。这本质上是在穷举所有可能的演化路径并求和。
 8. **KL 散度（Kullback-Leibler Divergence）**：
   - **直观比喻（翻译成本）**：想象你要把一本德文书（真实分布 $q$）翻译成中文。如果你用一套完美的翻译标准（模型 $p$ 贴合 $q$），那么额外的信息损耗就是 0。但如果你的翻译标准很烂，读者在阅读时就需要不断查字典、脑补缺失的信息，这些“额外的阅读成本”就是 KL 散度。
   - **数学定义**：$D_{KL}(q \parallel p) = \mathbb{E}_{x \sim q} [\log q(x) - \log p(x)]$。它衡量的是：当你观察到数据 $x$ 时，真实概率 $q$ 与模型概率 $p$ 之间对数差值的平均值。
   - **为什么能衡量‘距离’**：
     1. **非负性**：$D_{KL}(q \parallel p) \geq 0$，且仅在 $p=q$ 时为 0。这保证了它能作为一个有效的“差距指标”。
     2. **非对称性**：注意 $D_{KL}(q \parallel p) \neq D_{KL}(p \parallel q)$。在 DDPM 中，我们始终以真实加噪轨迹 $q$ 为基准去训练模型 $p$。
   - **在 DDPM 中的角色**：
     我们的目标是最小化 $D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))$。
     由于这两个分布都是高斯分布，根据高斯分布的性质，这个 KL 散度最终会简化为两个分布**均值之间的均方误差（MSE）**。这就是为什么 DDPM 的 Loss 函数长得像 MSE 的原因。

我们希望模型学到的分布 $p_\theta(\mathbf{x}_0)$ 最大化真实数据的似然。

### 5.1 变分下界（VLB）的严谨数学推导

我们从对数似然 $\log p(\mathbf{x}_0)$ 的原始定义出发，通过一系列数学等价变换进行拆解。

#### 第一步：似然的期望化（构造 $q$ 分布）
根据概率的**边缘化定义**，$\mathbf{x}_0$ 的概率是所有可能扩散路径 $\mathbf{x}_{1:T}$ 的积分总和。我们利用 $\int q(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T} = 1$ 这个性质：

$$\begin{aligned}
\log p(\mathbf{x}_0) &= \log p(\mathbf{x}_0) \cdot 1 \\
&= \log p(\mathbf{x}_0) \cdot \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) d\mathbf{x}_{1:T} \\
&= \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \log p(\mathbf{x}_0) d\mathbf{x}_{1:T} \quad \text{(因为 $\log p(\mathbf{x}_0)$ 与积分变量无关)}
\end{aligned}$$

#### 第二步：引入联合分布（利用贝叶斯法则）
根据条件概率公式 $p(A|B) = p(A,B)/p(B)$，可得 $p(\mathbf{x}_0) = \frac{p(\mathbf{x}_{0:T})}{p(\mathbf{x}_{1:T}|\mathbf{x}_0)}$。代入上式：

$$\log p(\mathbf{x}_0) = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \log \left( \frac{p(\mathbf{x}_{0:T})}{p(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right) d\mathbf{x}_{1:T}$$

#### 第三步：引入辅助分布 $q$（配凑法）
为了构造出 $q$ 与 $p$ 的对比关系，我们在 $\log$ 分式中同时乘上 $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$：

$$\log p(\mathbf{x}_0) = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \log \left( \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \cdot \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right) d\mathbf{x}_{1:T}$$

#### 第四步：公式拆解（似然分解恒等式）
利用 $\log(A \cdot B) = \log A + \log B$ 和积分的线性性质：

$$\begin{aligned}
\log p(\mathbf{x}_0) &= \underbrace{\int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \log \left( \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right) d\mathbf{x}_{1:T}}_{\text{VLB (Variational Lower Bound)}} + \underbrace{\int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \log \left( \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right) d\mathbf{x}_{1:T}}_{\text{Gap (KL Divergence)}} \\
&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] + D_{KL}(q(\mathbf{x}_{1:T}|\mathbf{x}_0) \parallel p(\mathbf{x}_{1:T}|\mathbf{x}_0))
\end{aligned}$$

- **VLB（变分下界）**：这是我们实际能够通过采样优化的目标。
- **Gap（误差项）**：这是真实后验 $p(\mathbf{x}_{1:T}|\mathbf{x}_0)$ 与我们定义的加噪后验 $q$ 之间的 KL 散度。

**结论**：由于 KL 散度始终 $\geq 0$，因此 $\log p(\mathbf{x}_0) \geq \text{VLB}$。最大化 VLB 的过程，不仅是在逼近似然函数的最大值，同时也是在强迫模型去噪过程 $p$ 的轨迹尽可能贴合真实加噪轨迹 $q$。

- **Gap 项**：衡量了“人为定义的加噪路径 $q$”与“模型学到的去噪路径 $p$”之间的差距。
- **下界性质**：由于 $D_{KL} \geq 0$，所以 $\log p(\mathbf{x}_0) \geq \text{VLB}$。
- **训练逻辑**：由于直接最大化 $\log p(\mathbf{x}_0)$ 涉及高维积分，我们转而最大化 $\text{VLB}$。这不仅能间接提升似然，还能让模型的逆向路径 $p$ 尽可能贴合真实的加噪后验 $q$。

最终损失函数定义为负的 VLB：

$$L_{VLB} = \mathbb{E}_q \left[ \underbrace{D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) || p(\mathbf{x}_T))}_{L_T: 常数项} + \sum_{t=2}^T \underbrace{D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) || p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}: 去噪匹配项} - \underbrace{\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0: 重建项} \right]$$

核心关注 $L_{t-1}$：它是两个高斯分布的 KL 散度。
回顾第 2 节的工具箱：**最小化两个高斯的 KL 散度 $\Leftrightarrow$ 最小化它们均值的 MSE。**
真实均值 $\tilde{\boldsymbol{\mu}}_t$ 包含真实噪声 $\boldsymbol{\epsilon}$，模型均值 $\boldsymbol{\mu}_\theta$ 包含预测噪声 $\boldsymbol{\epsilon}_\theta$。
代入比较，Loss 简化为：
$$L_{simple}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\underbrace{\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}}_{\mathbf{x}_t}, t) \|^2 \right]$$

**人话总结**：训练过程就是 **随机选一张图，随机加点噪，然后让网络去猜加了多少噪。** 猜得越准，去噪能力越强。

---

## 6. 算法流程

### 6.1 训练算法（Training）
1. 重复进行以下步骤直到收敛：
2. 从数据集采样 $\mathbf{x}_0$。
3. 随机采样时间步 $t \sim \text{Uniform}(\{1, \dots, T\})$。
4. 随机采样噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
5. 构造输入 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$。
6. 计算梯度下降，最小化 $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$。

### 6.2 采样算法（Sampling / Inference）
1. 从标准正态分布采样 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。
2. 从 $t = T, T-1, \dots, 1$ 循环：
3. 采样 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（如果是最后一步 $t=1$，则 $\mathbf{z}=\mathbf{0}$）。
4. 计算去噪后的均值：
   $$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$
   *(注：这里的系数 $\frac{1-\alpha_t}{\dots}$ 即前文的 $\frac{\beta_t}{\dots}$)*
5. 循环结束，得到生成的 $\mathbf{x}_0$。

---

## 7. 深度思考：为什么是高斯分布？

### 7.1 数据分布的真实面貌
- **现实数据是非高斯的**：无论是图像、语音还是你的染色体数据，其分布 $p(\mathbf{x}_0)$ 都是极其复杂的、多峰的（Multi-modal）。
- **流形假设（Manifold Hypothesis）**：高维数据通常分布在低维的流形上。扩散模型的作用是通过加噪，将这些孤立的、极细的流形“弥散”到整个空间，让网络更容易学习到数据分布的梯度。

### 7.2 为什么加噪过程选高斯？
1. **数学可解析性**：正如我们在第 2 节和第 4 节看到的，高斯分布在卷积、乘法、配方上都有闭式解。如果换成其他分布（如 Cauchy 分布），公式推导会直接卡死。
2. **中心极限定理 (Central Limit Theorem, CLT)**：该定理保证了无论初始分布 $p(\mathbf{x}_0)$ 多么奇怪，经过大量独立高斯噪声的叠加，$p(\mathbf{x}_T)$ 最终一定会变成平滑的标准正态分布。这给了生成过程一个统一的、确定的起点。
3. **最大熵原理**：在给定方差的情况下，高斯分布是信息熵最大的分布。这意味着它对原始数据结构的破坏最彻底，也最“公平”。

### 7.3 总结
扩散模型并不是在假设你的数据是正态分布，而是在利用正态分布作为**桥梁**。它通过预测噪声，学习如何从一个无序的正态分布中，“逆熵”推导出你数据集中那个复杂的、有序的真实分布。

---

## 8. 常见问题（Q&A）

**Q: 为什么要预测噪声 $\epsilon$，而不是直接预测 $x_0$？**
A: 虽然理论上都可以，但实验发现在图像生成任务中，预测噪声的效果更好，训练更稳定。这有点像 ResNet，预测残差（变化量）往往比预测绝对值更容易。

**Q: $\sigma_t$ 在采样时怎么取？**
A: 论文中给出了两种方案，效果差不多：
1. $\sigma_t^2 = \beta_t$ （假设后验方差很大，适用于 $x_0$ 未知性较强时）
2. $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ （真实的后验方差）
通常简单起见取前者。

**Q: DDPM 为什么慢？**
A: 因为采样需要 $T=1000$ 步串行计算，每一步都要过一遍网络。后续的 DDIM、Latent Diffusion (Stable Diffusion) 都是为了解决这个问题。
