# 方向 M：Score-based 检测 (Score-Based Generative Detection)

> **目标**: 用 Score-based Generative Model (SGM) 学习框分布的 score function ∇log p(x), 推理用 Langevin dynamics 采样, 提供比 DDPM 更通用的概率框架, 支持任意先验分布 (含 StructuredPrior GMM)。
>
> **理论依据**:
> - Score Matching: Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (NeurIPS 2019)
> - NCSN: Song & Ermon, "Improved Techniques for Training Score-Based Generative Models" (NeurIPS 2020)
> - SDE: Song et al., "Score-Based Generative Modeling through SDEs" (ICLR 2021)
> - Annealed Langevin Dynamics
>
> **当前代码位置**:
> - [ldmdet/diffusion/rectified_flow.py](../../ldmdet/diffusion/rectified_flow.py) — 当前 RF 实现
> - [ldmdet/diffusion/structured_prior.py](../../ldmdet/diffusion/structured_prior.py) — GMM 先验 (方向 F1)

---

## 1. 背景与动机

### 1.1 DDPM 的局限: 强假设高斯先验

DDPM/RF 假设:
- 前向: `x_t = √α_t · x_0 + √(1-α_t) · ε`, ε ~ N(0, I)
- 反向: 学习高斯转移 p_θ(x_{t-1} | x_t)

**问题**: 标准高斯先验与染色体框分布不匹配 (方向 F1 已用 GMM 缓解, 但 DDPM 框架仍假设高斯转移)。

### 1.2 Score-based 的优势

SGM 直接学习 **score function** s_θ(x, t) ≈ ∇log p_t(x), 不假设转移分布形式:

| 特性 | DDPM | Score-based |
|------|------|-------------|
| 学习目标 | 噪声 ε 或 x_0 | score ∇log p(x) |
| 转移假设 | 高斯 | **无假设** |
| 先验分布 | 标准高斯 | **任意** (含 GMM) |
| 采样 | DDIM/Heun | Langevin/PC |
| 理论框架 | 概率流 ODE | SDE (更通用) |

### 1.3 对染色体任务的意义

1. **匹配真实先验**: 方向 F1 的 GMM 可直接作为 SGM 的初始分布
2. **多模态建模**: 46 条框是多模态分布, SGM 天然支持
3. **理论统一**: DDPM/RF 是 SGM 的特例, 换 SGM 不丢已有成果

---

## 2. 子方向说明

### M1: Denoising Score Matching (DSM) 训练

**问题**: 直接估计 ∇log p(x) 困难 (需 score matching)。

**方案**: Denoising Score Matching — 加噪声后预测 score, 等价于 DDPM 但理论更通用。

**实现**:
```python
class ScoreHead(nn.Module):
    """Score 预测头 (替代当前的 x_0 预测)"""
    def __init__(self, feat_channels=256):
        super().__init__()
        # 复用 SingleDiffusionDetHead 结构
        self.feature_extractor = SingleDiffusionDetHead(...)
        # 输出 4 维 score (而非 x_0)
        self.score_head = nn.Linear(feat_channels, 4)

    def forward(self, features, x_t, t):
        roi_feat = self.feature_extractor(features, x_t, t)
        score = self.score_head(roi_feat)  # s_θ(x_t, t) ≈ ∇log p_t(x_t)
        return score

# Loss: Denoising Score Matching
def dsm_loss(score_pred, x_t, x_0, noise, t, sigma_t):
    """
    DSM 目标: s_θ(x_t, t) ≈ -ε / σ_t
    其中 x_t = x_0 + σ_t * ε, ε ~ N(0, I)
    """
    score_target = -noise / sigma_t  # 解析 score
    return F.mse_loss(score_pred, score_target)
```

**复杂度**: 低 — 改输出维度和 loss, 不动 backbone。

**预期收益**: 理论更通用, 实际与 DDPM 相当 (因 DDPM 是 SGM 特例)。

### M2: Noise Conditioned Score Network (NCSN)

**问题**: 单一噪声尺度的 score 估计在低密度区域不准。

**方案**: 多噪声尺度条件化, 学习多尺度 score。

**实现**:
```python
class NCSN(nn.Module):
    """Noise Conditioned Score Network"""
    def __init__(self, num_noise_scales=10):
        super().__init__()
        self.sigmas = torch.linspace(0.01, 1.0, num_noise_scales)
        self.score_net = ScoreHead(...)

    def forward(self, features, x, sigma_idx):
        # 条件化噪声尺度
        t = self.sigmas[sigma_idx]
        return self.score_net(features, x, t)
```

**复杂度**: 中 — 需调噪声调度。

**预期收益**: 多尺度建模, 低密度区域 (稀有框) 质量提升。

### M3: Annealed Langevin Dynamics 采样

**问题**: 标准 Langevin 在多模态分布易困局部最优。

**方案**: 退火 Langevin — 从大噪声到小噪声逐步采样。

**实现**:
```python
def annealed_langevin_sample(score_net, x_init, sigmas, num_steps=100, eps=2e-5):
    """退火 Langevin 动力学采样"""
    x = x_init
    for sigma_idx, sigma in enumerate(sigmas):
        alpha = eps * sigma ** 2 / sigmas[-1] ** 2  # 步长随噪声调整
        for _ in range(num_steps):
            score = score_net(x, sigma_idx)
            x = x + alpha * score + np.sqrt(2 * alpha) * torch.randn_like(x)
    return x
```

**复杂度**: 中 — 采样步数多, 但每步轻量。

**预期收益**: 多模态分布采样质量提升, 支持任意先验。

### M4: Predictor-Corrector (PC) 采样器

**问题**: 纯 Langevin 采样慢, 纯 ODE 精度不足。

**方案**: PC 采样 — ODE predictor + SDE corrector。

**实现**:
```python
def pc_sample(score_net, x_init, time_steps):
    """Predictor-Corrector 采样"""
    x = x_init
    for t, t_next in zip(time_steps[:-1], time_steps[1:]):
        # Predictor: ODE 一步
        score = score_net(x, t)
        x = x + (t - t_next) * score  # Euler

        # Corrector: Langevin 几步
        for _ in range(num_corrector_steps):
            score = score_net(x, t_next)
            x = x + eps * score + np.sqrt(2 * eps) * torch.randn_like(x)
    return x
```

**复杂度**: 中 — PC 组合, 质量与速度平衡。

**预期收益**: 比 DDIM 质量 +10%, 比 Heun +5%。

---

## 3. 与 StructuredPrior (方向 F1) 的协同

### 3.1 SGM 天然支持 GMM 先验

```python
# DDPM: 假设 x_1 ~ N(0, I)
# SGM: x_T ~ 任意分布, 如 GMM

def sample_initial_from_gmm(gmm_prior, n):
    """从 GMM 采样初始框 (方向 F1)"""
    return gmm_prior.sample(n)

# SGM 的 score 在 GMM 上有解析形式
def gmm_score(x, gmm):
    """GMM 的解析 score: ∇log p(x) = Σ w_k ∇log N(x; μ_k, σ_k) / Σ w_k N(...)"""
    ...
```

**关键优势**: DDPM 反向过程假设高斯转移, 无法直接用 GMM 先验; SGM 的 Langevin 采样可从任意分布开始。

---

## 4. 文件组织

```
ldmdet/
├── diffusion/
│   ├── score_matching.py       # M1: DSM 训练
│   ├── ncsn.py                 # M2: NCSN 多尺度
│   ├── langevin_sampler.py     # M3: 退火 Langevin
│   └── pc_sampler.py           # M4: PC 采样器
├── core/
│   └── score_head.py           # Score 预测头
├── tests/
│   └── test_direction_m.py
└── experiments/configs/ldmdet/
    └── direction_m_score_based.py
```

---

## 5. 测试计划

```python
class TestScoreMatching:
    def test_score_target_correct(self):
        """DSM 目标应为 -ε / σ"""
    def test_score_approximates_gradient(self):
        """学到的 score 应近似真实 ∇log p(x)"""

class TestLangevinSampler:
    def test_convergence(self):
        """Langevin 应收敛到高密度区域"""
    def test_multimodal_support(self):
        """应能覆盖多模态分布"""

class TestPCSampler:
    def test_quality_vs_heun(self):
        """PC 质量应 >= Heun"""
    def test_arbitrary_prior(self):
        """应支持 GMM 等任意先验"""
```

---

## 6. 实验计划

### 6.1 主实验

| 实验 | 配置 | 预期 |
|------|------|------|
| M1-dsm | DSM 训练 | 与 DDPM 持平 (基线) |
| M2-ncsn | 多尺度 score | 稀有框 +1% |
| M3-langevin | 退火 Langevin | 多模态覆盖 +1-2% |
| M4-pc | PC 采样 | 质量 +5% vs Heun |
| M-gmm | M4 + GMM 先验 | 与方向 F1 协同 |

### 6.2 消融

| 消融 | 验证 |
|------|------|
| DDPM vs SGM | 理论框架差异 |
| 高斯 vs GMM 先验 | 先验影响 |
| Langevin 步数 | 收敛性 |
| PC vs 纯 Predictor | Corrector 贡献 |

---

## 7. 风险与缓解

### 7.1 采样慢

**风险**: Langevin 需多步 (100+), 比 DDIM 慢。

**缓解**:
- PC 采样 (Predictor 少步 + Corrector 几步)
- 蒸馏到少步 (类似方向 H3)

### 7.2 Score 估计在低密度区不准

**风险**: 框空间低密度区域 (稀有类别) score 难学。

**缓解**:
- NCSN 多噪声尺度 (M2)
- 方向 E 类别平衡采样 (补充稀有类数据)

### 7.3 理论门槛高

**风险**: SGM 理论复杂, 调参难。

**缓解**:
- 先做 M1 (DSM, 与 DDPM 等价验证)
- 再逐步加 M2-M4
- 参考官方实现

---

## 8. 成功指标

| 指标 | baseline (RF 4步) | 目标 (SGM PC) |
|------|------------------|---------------|
| mAP | 0.858 | +0.01~0.02 |
| 稀有类 AP (Y) | 0.901 | +0.02 |
| 先验灵活性 | 仅高斯 | 任意 (GMM 等) |
| 采样步数 | 4 | 10-20 (可调) |

---

## 9. 实现路线图

1. **Phase 1 (3 周)**: M1 — DSM 训练, 验证与 DDPM 等价
2. **Phase 2 (3 周)**: M3+M4 — Langevin + PC 采样
3. **Phase 3 (2 周)**: M2 — NCSN 多尺度 (若收益明显)
4. **Phase 4 (2 周)**: M-gmm — 与方向 F1 GMM 协同

---

## 10. 参考文献

- Score Matching: https://arxiv.org/abs/1907.05600
- NCSN: https://arxiv.org/abs/2006.09011
- SDE: https://arxiv.org/abs/2011.13456
- PC Sampler: https://arxiv.org/abs/2011.13456
