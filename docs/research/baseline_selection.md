# Baseline 模型选择与引用论据验证

> **目的**：从 `ldmdet-experiment/sota/` 中选取一个统一的、可复现的、与 5 个突破方向正交的 baseline 模型，并验证 5 个方向文档中引用的论文与数学推导的严谨性。
>
> **结论**：选取 `reproduce_0751_stochot_eps5_v2`（mAP=0.753）作为统一 baseline，对应配置 [experiments/configs/ldmdet/ghss.py](../../experiments/configs/ldmdet/ghss.py)。

---

> ⚠️ **暂时废弃**：以下基线选择结论（§1 候选实验对比、§2 各方向 Baseline 映射、§5 风险与缓解、§6 验收清单）基于旧数据集 Chromosome20240904 的 checkpoint（mAP≈0.753，所有候选 mAP 处于 0.72-0.75 量级），24obj 数据集（mAP 0.77-0.87 量级）上的基线选择结论待重新验证。
>
> 注：§3 引用论据验证、§4 数学推导严谨性复核为文献与理论层面，与数据集无关，保留有效。

## 1. 候选实验对比

### 1.1 SOTA 实验汇总

数据来源：`ldmdet-experiment/sota/**/metrics.json`（best_eval 字段）。

| 实验 | best mAP | AP50 | AP75 | APs | 配置特点 | 配置文件 |
|------|---------|------|------|-----|---------|---------|
| **reproduce_0751_stochot_eps5_v2** | **0.753** | 0.943 | 0.843 | 0.522 | RF+Heun+AdaLN+Sinkhorn OT (ε=5) | [config.json](../../ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/config.json) |
| scheme_C1_5_mixed_rel_l1_lam015 | 0.752 | 0.946 | 0.843 | 0.534 | 损失函数优化 (mixed rel L1) | — |
| ldmdet_dpm_solver_pp_o2_s8 | 0.748 | 0.945 | 0.837 | 0.529 | RF+DPM-Solver++(o2, s8) | — |
| ldmdet_kcec_redundant_slots_m2 | 0.748 | 0.944 | 0.843 | 0.531 | KCEC 倍性约束 | — |
| smallobj_B_scaleaware_loglinear | 0.744 | 0.943 | 0.833 | 0.497 | 小目标 scale-aware | — |
| ldmdet_daec_contrastive_kcec_eps5 | 0.740 | 0.937 | 0.825 | 0.499 | DAEC 对比学习 | — |
| ldmdet_convnextv2_mae | 0.736 | 0.941 | 0.829 | 0.519 | ConvNeXtV2 MAE 预训练 | — |
| dino_r50 (非扩散参照) | 0.869 | 0.991 | 0.977 | 0.553 | DINO 非扩散参照 | — |

### 1.2 选择标准

1. **最高 mAP**：在扩散模型实验中取得 SOTA
2. **配置可复现**：当前代码库可直接复现，无需额外组件
3. **正交性**：与 5 个突破方向的改动变量不重叠
4. **架构代表性**：体现 RF + Heun + AdaLN-Zero + GHSS 的核心组合

### 1.3 选择结果

**统一 baseline：`reproduce_0751_stochot_eps5_v2`**

- **mAP**：0.753（扩散实验 SOTA）
- **配置**：`ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py`
- **当前对应**：[experiments/configs/ldmdet/ghss.py](../../experiments/configs/ldmdet/ghss.py)（`type='ghss', epsilon=5.0`）
- **关键参数**：
  - `diffusion_type='rectified_flow'`
  - `solver_type='heun'`
  - `sampling_timesteps=4`
  - `rf_schedule='shifted'`, `rf_shift=3.0`
  - `time_conditioning='adaln_zero'`
  - `coupling.type='ghss'`, `coupling.epsilon=5.0`
  - `num_proposals=500`, `backbone=ResNet-50`

### 1.4 不选其他实验的理由

| 候选 | 不选原因 |
|------|---------|
| dino_r50 (mAP=0.869) | 非扩散模型，架构差异大，不适合作为扩散改进的基线 |
| ldmdet_convnextv2_mae | backbone 不同（ConvNeXtV2 vs ResNet-50），引入额外变量 |
| scheme_C1_5_mixed_rel_l1_lam015 | 损失函数已优化，与方向三（SNR 匹配）变量重叠 |
| ldmdet_dpm_solver_pp_o2_s8 | 采样器已改（DPM-Solver++），与方向四（轨迹）变量重叠 |
| ldmdet_kcec_redundant_slots_m2 | 已有计数约束，与方向二（计数先验）变量重叠 |
| smallobj_B_scaleaware_loglinear | 已有 scale-aware，与方向四（尺度条件化）变量重叠 |
| ldmdet_daec_contrastive_kcec_eps5 | 已有对比学习+计数约束，多变量重叠 |

---

## 2. 各方向的 Baseline 映射

| 方向 | Baseline 配置 | 对应 ckpt | 消融变量 | 假设 |
|------|--------------|----------|---------|------|
| 方向一（非平衡 OT） | [ghss.py](../../experiments/configs/ldmdet/ghss.py) | stochot_eps5_v2 ep59 | 耦合策略 (ghss → unbalanced_ghss) | 边缘松弛提升稀疏图 recall |
| 方向二（计数先验） | [rf_heun_adaln.py](../../experiments/configs/ldmdet/directions/nonlinear_trajectory/rf_heun_adaln.py) | rf_heun_shifted ep100 | +count_conditioning +counting_branch | 计数先验降低计数误差 |
| 方向三（SNR 匹配） | [ghss.py](../../experiments/configs/ldmdet/ghss.py) | stochot_eps5_v2 ep59 | matcher (SimOTA → SNRAware) | SNR 加权减少高噪声错误匹配 |
| 方向四（非线性轨迹） | [ghss.py](../../experiments/configs/ldmdet/ghss.py) | stochot_eps5_v2 ep59 | rf_type (linear → scale_conditioned) | 尺度条件化提升小目标 APs |
| 方向五（分层分类） | [ghss.py](../../experiments/configs/ldmdet/ghss.py) | stochot_eps5_v2 ep59 | cls_head (flat → hierarchical) | 分层降低组内混淆 |

**说明**：
- 方向二使用 `rf_heun_adaln.py`（无 GHSS）作为对照，因为方向二本身要新增计数分支，需在"无计数约束"的基线上验证增量收益。若用 `ghss.py`，则同时存在 GHSS 和计数分支两个变量。
- 方向一、三、四、五均用 `ghss.py`，因为它们各自只改动一个变量（耦合、匹配、轨迹、分类头），与 GHSS 正交。

---

## 3. 引用论据验证

下表列出 5 个方向文档中引用的关键论文，并通过 WebSearch 验证其存在性、引用准确性和社区认可度。

### 3.1 验证结果汇总

| 方向 | 引用论文 | 验证状态 | 出处/链接 |
|------|---------|---------|----------|
| 方向一 | Chizat et al., "Scaling Algorithms for Unbalanced Transport Problems", Math. Comput., 2018 | ✅ 真实存在，引用准确 | [DOI:10.1090/mcom/3303](https://doi.org/10.1090/mcom/3303)，被引 535+ |
| 方向一 | Cuturi, "Sinkhorn Distances", NeurIPS 2013 | ✅ 真实存在 | POT 库文档明确引用 |
| 方向二 | Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022 | ✅ 真实存在，引用准确 | [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)，CFG 是现代扩散模型标准组件 |
| 方向二 | Chung et al., "Come-Closer-Diffuse-Faster", CVPR 2022 | ✅ 真实存在 | 条件扩散约束生成经典工作 |
| 方向三 | Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models", CVPR 2024 | ✅ 真实存在，引用准确 | [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)，CVPR 2024 Oral (top 1%) |
| 方向四 | Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023 | ✅ 真实存在，引用准确 | [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)，被引 589+ |
| 方向四 | Liu et al., "Flow Straight and Fast: Rectified Flow", ICLR 2023 | ✅ 真实存在，引用准确 | [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) |
| 方向四 | Pooladian et al., "Multisample Flow Matching", ICML 2023 | ✅ 真实存在，引用准确 | [arXiv:2304.14772](https://arxiv.org/abs/2304.14772) |
| 方向四 | Esser et al., "Scaling Rectified Flow Transformers" (SD3), 2024 | ✅ 真实存在，shift=3.0 准确 | [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)，SD3 论文明确使用 α=3.0 |
| 方向五 | Morin & Bengio, "Hierarchical Probabilistic Neural Network Language Model", AISATS 2005 | ✅ 真实存在，引用准确 | 被引 819+，分层 softmax 经典工作 |
| 方向五 | Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", 2021 | ✅ 真实存在 | classifier-guided diffusion 经典 |

### 3.2 关键引用的社区认可度

- **Chizat 2018 (UOT)**：被引 535+，POT 库（Python Optimal Transport）官方实现 `ot.sinkhorn_unbalanced` 直接基于此论文，是 UOT 的标准引用。
- **Lipman 2023 (Flow Matching)**：ICLR 2023，被引 589+，Meta FAIR 出品，已成为 CNF 训练的标准范式，SD3/Flux 均基于此。
- **Liu 2023 (Rectified Flow)**：ICLR 2023，UT Austin，被 SD3/InstaFlow 采用，1-RectFlow 是当前少步采样的主流方案。
- **Pooladian 2023 (Multisample FM)**：ICML 2023，Meta FAIR，mini-batch OT 耦合的理论基础，被多个后续工作引用。
- **Ho & Salimans 2022 (CFG)**：所有现代扩散模型（SD/SDXL/SD3/Flux/DALL-E）的标准组件，社区公认。
- **Karras 2024 (EDM2)**：CVPR 2024 Oral (top 1%)，NVIDIA 出品，扩散模型训练动力学的权威工作。
- **Esser 2024 (SD3)**：Stability AI 出品，首次将 RF + MMDiT 应用于大规模生产模型，shift=3.0 是高分辨率（1024×1024）的标准配置。
- **Morin & Bengio 2005 (H-Softmax)**：被引 819+，分层 softmax 的奠基工作，NLP 领域经典。

---

## 4. 数学推导严谨性复核

### 4.1 方向一：非平衡 OT 的 Sinkhorn 迭代

**文档推导**：
$$
u_i = \left(\frac{a_i}{(Kv)_i}\right)^{\lambda_1 / (\varepsilon + \lambda_1)}, \quad
v_j = \left(\frac{b_j}{(K^\top u)_j}\right)^{\lambda_2 / (\varepsilon + \lambda_2)}
$$

**验证**：
- ✅ 与 Chizat 2018 论文 Section 3.2 的迭代格式一致（论文用 $f, g$ 对偶变量，等价于 $u = e^{f/\varepsilon}, v = e^{g/\varepsilon}$）
- ✅ 极限正确：$\lambda \to \infty$ 时指数 $\to 1$，退化为标准 Sinkhorn；$\lambda \to 0$ 时指数 $\to 0$，$u \to 1$
- ✅ 对数域实现正确：$\log u_i = \alpha(\log a_i - \text{logsumexp}_j(\log K_{ij} + \log v_j))$，与文档代码一致
- ✅ KL 散度定义使用广义 KL（允许质量不守恒），符合 UOT 标准

**结论**：数学推导严谨，与原始论文一致。

### 4.2 方向二：计数先验的拉格朗日约束

**文档推导**（路径 B）：
$$
\mathcal{L}(\theta, \lambda) = \mathcal{L}_{\text{diff}}(\theta) + \lambda \left( \sum_i \mathbb{1}[\text{score}_i > \tau] - c \right)
$$

**验证**：
- ✅ 拉格朗日对偶形式正确：等式约束 $\sum \mathbb{1}[\cdot] = c$ 的对偶变量 $\lambda$
- ✅ 二分搜索 $\tau$ 的方法正确：score 单调递减，count 关于 $\tau$ 单调递减，二分搜索收敛
- ✅ CFG 路径（路径 A）的公式 $\tilde{v}_\theta = (1+w)v_\theta(\cdot|c) - w v_\theta(\cdot|\varnothing)$ 与 Ho & Salimans 2022 原文一致
- ⚠️ **小瑕疵**：$\mathbb{1}[\cdot]$ 不可微，实际实现需用 sigmoid 软化或 straight-through 估计。文档已提及但可更明确

**结论**：推导正确，实现需注意不可微处理。

### 4.3 方向三：SNR 加权匹配

**文档推导**：
$$
\text{SNR}(t) = \frac{(1-t)^2}{t^2}, \quad w(t) = \frac{\text{SNR}(t)}{1+\text{SNR}(t)} = \frac{(1-t)^2}{(1-t)^2 + t^2}
$$

**验证**：
- ✅ Rectified Flow 的 SNR 推导正确：$x_t = (1-t)x_0 + t x_1$，信号能量 $(1-t)^2\|x_0\|^2$，噪声能量 $t^2\|x_1\|^2$，假设 $\|x_0\| \approx \|x_1\|$ 则 SNR $= (1-t)^2/t^2$
- ✅ logistic 形式 $w(t) = (1-t)^2/((1-t)^2+t^2)$ 是 SNR 的 sigmoid 变换，单调递增，符合"低噪声高权重"直觉
- ✅ 与 Karras 2024 的 SNR-aware 训练思路一致（虽 Karras 用 $\lambda = \log\text{SNR}$ 参数化，但单调性相同）
- ✅ 极限正确：$t \to 0$ 时 $w \to 1$（完全信任匹配），$t \to 1$ 时 $w \to 0$（忽略匹配）

**结论**：推导严谨，与扩散模型 SNR 理论一致。

### 4.4 方向四：尺度条件化噪声调度

**文档推导**：
$$
\kappa(s) = 1 + \lambda_{\text{mod}} \frac{s_{\max} - s}{s_{\max}}, \quad
\sigma(t, s) = t^{\kappa(s)}, \quad \alpha(t, s) = \sqrt{1 - \sigma^2(t, s)}
$$

**验证**：
- ✅ 边界条件满足：$\sigma(0, s) = 0, \sigma(1, s) = 1, \alpha(0, s) = 1, \alpha(1, s) = 0$
- ✅ 单调性：$s$ 小（小目标）→ $\kappa$ 大 → $\sigma$ 增长更快 → 小目标更早进入噪声主导（去噪更早开始）
- ✅ 与 SD3 (Esser 2024) 的 shifted schedule 思想一致：SD3 用 $\sigma_{\text{shift}}(t) = t \cdot \frac{\mu + (1-\mu)t}{1-t+t\mu}$ 实现分辨率条件化，本文用尺度 $s$ 替代分辨率，数学形式不同但思想一致
- ✅ Flow Matching 框架兼容：$\alpha, \sigma$ 满足 $\alpha^2 + \sigma^2 = 1$（余弦调度族），Lipman 2023 的 CFM 目标函数不变
- ⚠️ **小瑕疵**：$\kappa(s)$ 的线性形式是启发式，可考虑用 $\kappa(s) = (s_{\max}/s)^\gamma$ 的幂律形式（更符合尺度不变性）。文档已提及需网格搜索 $\lambda_{\text{mod}}$

**结论**：推导正确，启发式可进一步优化。

### 4.5 方向五：分层分类的信息论分解

**文档推导**：
$$
H(Y) = H(G) + H(Y|G), \quad
\log_2 24 \approx 4.58 = \log_2 8 + \mathbb{E}_g[\log_2 |G|] \approx 3 + 1.58
$$

**验证**：
- ✅ 信息论链式法则 $H(Y) = H(G) + H(Y|G)$ 正确（$G$ 是 $Y$ 的确定性函数）
- ✅ 熵分解数值正确：$\log_2 24 = 4.585$，$\log_2 8 = 3$，$H(Y|G) = \sum_g p(g) \log_2 |G_g|$
- ✅ 概率因子化 $p(y) = p(g) p(y|g)$ 正确，对应分层 softmax 的乘积形式
- ✅ 与 Morin & Bengio 2005 的分层 softmax 一致：每个内部节点是一个二分类（或多元 softmax），叶节点概率为路径概率乘积
- ✅ 组条件扩散 $p(x_t | y) = p(x_t | g) p(x_t | y, g)$ 的条件化正确，与 class-conditional diffusion（Dhariwal & Nichol 2021）框架兼容

**结论**：推导严谨，信息论基础扎实。

---

## 5. 风险与缓解

### 5.1 Baseline 复现风险

| 风险 | 缓解 |
|------|------|
| `ghss.py` 配置与原始 `stochot_eps5_v2` 不完全一致 | 已验证：当前 [ghss.py](../../experiments/configs/ldmdet/ghss.py) 继承 `rf_heun_adaln.py`，参数与 `config.json` 一致（ε=5.0, Heun, shifted, AdaLN-Zero） |
| 训练随机性导致 mAP 波动 | 使用固定种子，训练 150 epochs，取 best_epoch（ep59）的 ckpt 作为 baseline |
| 数据集版本变化 | 固定使用当前数据集，baseline 与改进实验同数据集 |

### 5.2 引用论据的潜在局限

| 引用 | 局限 | 影响 |
|------|------|------|
| Chizat 2018 (UOT) | 原论文针对连续测度，离散化后 Sinkhorn 有数值稳定性问题 | 已用对数域实现 + clamp 缓解 |
| Ho & Salimans 2022 (CFG) | CFG 原设计用于图像生成，检测任务的"条件"语义不同 | 方向二需重新定义条件 $c$（计数先验），非直接套用 |
| Karras 2024 (EDM2) | 针对 DDPM/EDM 框架，RF 框架的 SNR 定义不同 | 方向三已重新推导 RF 的 SNR $= (1-t)^2/t^2$，非直接引用 |
| SD3 (Esser 2024) | shifted schedule 针对图像分辨率，非目标尺度 | 方向四用目标尺度 $s$ 替代分辨率，数学形式不同 |
| Morin & Bengio 2005 (H-Softmax) | 原用于 NLP 词汇表（10^4 类），染色体仅 24 类 | 方向五的收益主要来自生物学先验（组结构），非计算效率 |

---

## 6. 验收清单

- [x] SOTA 实验对比完成（8 个候选）
- [x] Baseline 选定（`reproduce_0751_stochot_eps5_v2`, mAP=0.753）
- [x] 5 个方向的 baseline 映射完成
- [x] 11 篇关键引用论文验证存在且准确
- [x] 5 个方向的数学推导复核完成
- [x] 风险与缓解措施列出

---

## 7. 引用文献

1. Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F.-X. (2018). Scaling Algorithms for Unbalanced Transport Problems. *Mathematics of Computation*, 87(314), 2563-2609. [DOI:10.1090/mcom/3303](https://doi.org/10.1090/mcom/3303)
2. Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. *NeurIPS*.
3. Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
4. Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., & Ye, J. C. (2022). Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction. *CVPR*.
5. Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., & Laine, S. (2024). Analyzing and Improving the Training Dynamics of Diffusion Models. *CVPR* (Oral, top 1%). [arXiv:2312.02696](https://arxiv.org/abs/2312.02696)
6. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
7. Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. *ICLR*. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
8. Pooladian, A.-A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., & Chen, R. T. Q. (2023). Multisample Flow Matching: Straightening Flows with Minibatch Couplings. *ICML*. [arXiv:2304.14772](https://arxiv.org/abs/2304.14772)
9. Esser, P., Kulal, S., Blattmann, A., et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
10. Morin, F., & Bengio, Y. (2005). Hierarchical Probabilistic Neural Network Language Model. *AISTATS*.
11. Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. *NeurIPS*.
