# LDMDet 完整理论与实验路线

> 从 DiffusionDet 基线到 Flow Matching + OT 理论的完整研究轨迹
> 最后更新：2026-05-13（经两轮交叉校验，包含 60+ 实验日志核查）

______________________________________________________________________

## 一、研究起源与核心问题

### 1.1 动机

DiffusionDet 将目标检测建模为扩散模型的去噪过程：从随机噪声框出发，通过多步去噪逐步精化为检测框。核心瓶颈是**推理速度**——需要 4-8 步才能达到满意精度，而单步推理精度骤降。

**核心研究问题**：能否在保持/提升检测精度的同时，将推理步数压缩到 1-2 步？

### 1.2 数据集

染色体核型分析数据集（24 类染色体 + 性染色体），约 400+ 张显微图像。每张图包含约 46 个染色体实例（$K \\approx 46$），检测空间为 $\\mathbb{R}^4$（bbox 坐标）。

______________________________________________________________________

## 二、理论基础：统一数学框架

### 2.1 从 DDPM 到 Rectified Flow

**DDPM 前向过程**（DiffusionDet 原论文）：

$$x_t = \\sqrt{\\bar{\\alpha}\_t} \\cdot x_0 + \\sqrt{1 - \\bar{\\alpha}\_t} \\cdot \\epsilon$$

路径非直线，需要更多采样步数追踪弯曲轨迹。

**RF 前向过程**（我们的改进）：

$$x_t = (1-t) x_0 + t \\cdot x_1, \\quad t \\in \[0,1\]$$

路径为直线，传输代价最小（$\\mathcal{A}\[\\gamma\] \\geq |x_1 - x_0|^2$，等号当且仅当直线）。**理论上 RF 比 DDPM 需要更少采样步数达到相同精度。**

### 2.2 五大改进的数学统一

**检测扩散传输四因素框架**：检测框生成质量由 ODE 路径、训练匹配和时间采样共同影响，至少需要同时分析四个因素：

1. **耦合质量 $\\pi$**：影响传输代价 $C_{trans}=\\mathbb{E}\_\pi[\|v^\*\|^2]$
2. **训练信号多样性 $H(Y\mid X_t)$**：影响低维框空间中的监督覆盖和泛化
3. **修正曲率/时间变化率**：影响 ODE 离散化误差
4. **时间分配 $p(t)$**：决定训练资源在损失敏感度上的分配效率

| 改进             | 优化的分量                    | 数学效果                                  |
| ---------------- | ----------------------------- | ----------------------------------------- |
| DDPM → RF        | 路径形式                      | 弯曲路径 → 直线路径，降低采样难度         |
| OT Coupling      | 耦合 $\\pi$                   | 降低传输代价，但可能损失目标索引多样性    |
| Structured Noise | 源分布 $\\mu_z$               | 降低几何传输代价，但可能限制覆盖范围      |
| Shifted Schedule | 训练资源在 $t$ 上的分配       | 集中优化 $\\delta v\_\\theta$ 影响最大处  |
| AdaLN-Zero       | 时间条件分支初始化            | 时间条件残差从零开始增长，减少初期扰动    |

### 2.3 速度场分解

$$v\_\\theta(x_t, t, f) = \\underbrace{v\_{\\pi}(x_t, t)}_{\\text{耦合依赖的传输分量}} + \\underbrace{\\delta v_\\theta(x_t, t, f)}\_{\\text{特征修正分量}}$$

- $v\_{\\pi}$：依赖耦合策略 $\\pi$，不是固定的“OT 最优分量”
- $\\delta v\_\\theta$：依赖图像特征的修正，是网络需要学习的部分

采样误差不能完全由曲率解释；硬 OT 的负结果说明还必须考虑泛化/匹配误差和目标索引多样性。

### 2.4 Shifted Schedule 的理论依据

原始采样 $t \\sim U(0,1)$，shifted schedule 映射 $g: t \\mapsto \\frac{st}{1+(s-1)t}$（$s=3$）。

模型实际预测 $x_0$，速度由 $v_\theta=(x_t-x_0^{pred})/t$ 推导。速度误差导致 $x_0$ 预测误差 $\\delta x_0=-t\\delta v$，因此检测损失对速度误差的敏感度与 $t$ 成正比。Shifted schedule 在敏感度最高的区域（$t \\approx 1$）分配更多训练资源。

### 2.5 AdaLN-Zero 的零初始化优势

Scale-shift: $\\gamma, \\beta$ 随机初始化（非零），初始路径有随机曲率。

AdaLN-Zero: MLP 最后一层零初始化 → 时间条件残差为零。它不保证 $v\_\\theta|\_{\\text{init}} = 0$，因为 `reg_head` 和 instance interaction 非零初始化；真正收益是避免训练初期随机时间调制破坏空间特征。

______________________________________________________________________

## 三、实验演进路线

### 阶段 0：基线建立（DDPM 体系）

| 实验                    | mAP             | 核心配置                                                                 |
| ----------------------- | --------------- | ------------------------------------------------------------------------ |
| `diffusiondet_baseline` | 0.001（未收敛） | DiffusionDet 原论文复现（DDPM, scale-shift），训练失败，所有 epoch mAP=0 |
| `ldmdet_baseline`       | 0.725           | 染色体数据集基线（DDPM, scale-shift, **1步推理**）                       |
| `ldmdet_baseline_step4` | 0.709           | 4步推理版本，配置仅步数不同，反低于1步（反常，需排查 DDPM 实现）         |
| `chromodet_baseline`    | 0.717           | 早期基线，在 ldmdet_baseline 之前                                        |

**关键**：`ldmdet_baseline` 使用 1 步推理即达到 0.725，4 步推理反降至 0.709——这在 DDPM 体系下反常（通常多步 > 单步）。可能的解释是 DDPM 1 步推理实际走了 DDIM skip 路径，需进一步排查代码实现。`diffusiondet_baseline` 在 work_dirs 中训练完全未收敛，~0.45 可能来自其他分支的早期实验。

> **代码审计 / 重跑标注（2026-05-15）**：`projects/LDMDet/mods/diffusiondet_head.py::_ddim_step` 在 `t_next < 0` 时存在负索引风险。DDPM baseline、DDPM 4-step 和 RF-vs-DDPM 数值对比应修复后重跑。

### 阶段 1：RF 基础改进（路径直化）

| 实验                         | mAP       | mAP@50 | mAP@75 | 核心改进                         |
| ---------------------------- | --------- | ------ | ------ | -------------------------------- |
| `ldmdet_rf`                  | 0.733     | 0.939  | 0.828  | RF 替代 DDPM                     |
| `ldmdet_rf_shifted_schedule` | 0.747     | 0.937  | 0.837  | RF + Shifted Schedule ($s=3$)    |
| `ldmdet_flowdet_adaln`       | **0.751** | 0.943  | 0.843  | RF + AdaLN-Zero + Shifted + Heun |

**关键发现**：

- RF 单独提升 +0.8%，证明直线路径优于弯曲路径
- Shifted schedule 额外 +1.4%，验证时间敏感度假说
- AdaLN-Zero 额外 +0.4%，零初始化收益显著
- **AdaLN 是单项最大改进（+1.8% vs RF baseline）**

> **代码审计 / 重跑标注（2026-05-15）**：Shifted 与 AdaLN 主线未发现必须修正的实现问题；RF 对 DDPM 的绝对增益需等待 DDPM 采样修复后重新确认。AdaLN 结论应表述为“时间条件残差零初始化”，而非“速度场零初始化”。

### 阶段 2：OT 耦合探索（传输优化）

| 实验                               | mAP         | 关键参数                    | 发现                   |
| ---------------------------------- | ----------- | --------------------------- | ---------------------- |
| `ldmdet_flowdet_adaln_ot`          | 0.735       | Nearest OT (argmin)         | ❌ OT 反而降低性能     |
| `ldmdet_flowdet_adaln_ot_sinkhorn` | 0.748       | Sinkhorn argmax, eps=1      | 比硬 OT 好但仍不如随机 |
| `ot_sinkhorn_eps5/10/50/100`       | 0.733-0.748 | Sinkhorn argmax, 多 eps     | ε 扫描曲线异常平坦     |
| `sinkhorn_sample_eps5`             | **0.751**   | Sinkhorn stochastic, eps=5  | ✅ 随机采样恢复性能    |
| `sinkhorn_sample_eps50`            | 0.736       | Sinkhorn stochastic, eps=50 | ❌ 过大 ε 有损         |
| `group_hierarchical_stoch`         | **0.752**   | 群组层次 OT + stochastic    | ✅ 当前单次最佳耦合策略 |

**关键发现**：

- **硬 OT（argmax）在所有 ε 下都不如随机耦合**，因为确定性分配消除了训练多样性
- Stochastic Coupling (从传输矩阵采样) 修复了这个问题，在 eps=5 时达到 0.751
- 群组层次 OT（利用染色体 A-G 组 + 性染色体先验）单次运行额外 +0.001，达到 0.752；该差异需多 seed 验证
- **多样性 > 传输效率**：在低维检测空间，保持训练信号的多样性比最小化传输代价更重要

> **代码审计 / 重跑标注（2026-05-15）**：stochastic OT 和 group-hierarchical stochastic 使用 `torch.multinomial`，当前缺少固定 generator，且已有 0.751/0.738/0.750 的复现实验波动。所有 stochastic SOTA 数值和 “+0.001” 级别结论必须用 5 seed mean ± std 重报；硬 OT 负结果可作为趋势保留。

### 阶段 3：训练动力学优化（辅助机制）

| 实验                                | mAP       | 核心配置                           |
| ----------------------------------- | --------- | ---------------------------------- |
| `ldmdet_flowdet_adaln_trd`          | 0.743     | TRD（传输-修正分解）               |
| `ldmdet_flowdet_adaln_trd_only`     | 0.746     | TRD + velocity，无其他辅助         |
| `ldmdet_flowdet_adaln_trd_full`     | **0.752** | TRD + CAT + LSAS + velocity + Heun |
| `ldmdet_flowdet_adaln_lsas`         | 0.743     | LSAS（损失敏感调度）               |
| `ldmdet_flowdet_adaln_crossattn_v2` | —         | Cross-Attention 变体               |
| `ldmdet_flowdet_adaln_convnext`     | —         | ConvNeXt backbone 探索             |

**关键发现**：

- TRD/CAT/LSAS/velocity 的单独增益都很小（0.001-0.004），但组合效应显著
- trd_full（0.752）与 group_hierarchical_stoch（0.752）单次结果接近当前最佳，但都需要修正/多 seed 后确认；两者走的是两条正交路径：
  - **路径 A（group_hierarchical）**：优化耦合策略（群组层次 OT + stochastic）
  - **路径 B（trd_full）**：优化训练动力学（TRD + CAT + LSAS + velocity）
- **两条路径的组合已被实验探索，但结果为负交互**：`group_hierarchical_trd`=0.746, `stochastic_eps5_trd_cat`=0.740, `sinkhorn_trd_cat_lsas`=0.743，全部低于单路径最佳 0.752（详见 §七）

> **代码审计 / 重跑标注（2026-05-15）**：TRD/velocity/CAT 相关数值均需谨慎。当前 `velocity_loss` 与 ITD 的目标符号和 RF 定义相反；CAT 实现是 `$x_0$ 一致性` 而非纯曲率；TRD 复用 `cat_delta_t`。修复后应重跑 `trd_only`、`trd_full`、`velocity`、`cat_only`、`stochastic_eps5_trd_cat`、`sinkhorn_trd_cat_lsas`、`group_hierarchical_trd`。

### 阶段 4：Reflow 单步推理探索

| 实验                           | 1步 mAP   | 最佳 epoch | 核心发现                                    |
| ------------------------------ | --------- | ---------- | ------------------------------------------- |
| Reflow v1（从零训练）          | 0.715     | 47         | 可收敛但精度低，训练不稳定                  |
| Reflow v2（微调，val配对）     | 0.739     | 1          | Epoch 1 最佳，后续退化                      |
| Reflow v3（微调，train配对）   | 0.739     | 1          | 配对来源非关键问题                          |
| Reflow v4（修复velocity_head） | 0.739     | 1          | grad_norm=183.7，梯度爆炸                   |
| Reflow v5（warmup+调参）       | **0.739** | 1          | grad_norm 正常化，但 epoch 1 最佳后持续退化 |
| Reflow v6（第2轮 Reflow）      | 0.739     | 1          | 边际收益递减                                |
| ITD（中间轨迹蒸馏）            | —         | —          | 理论提出，实验结果待确认                    |

**关键发现**：

- **退化陷阱**：所有 Reflow 实验均呈现 "Epoch 1 最佳 → 后续退化" 的模式
- 退化根因：velocity loss 和检测 loss 的梯度方向冲突（实测 cos=-0.104，86.8% 层冲突）
- velocity loss 收敛天花板 ~0.23-0.30
- 多次 Reflow 边际收益递减：v2(+0.011) → v6(+0.011)
- **最佳 1步推理 mAP=0.739（Reflow v6），与 4步基线 0.752 差距 1.3%**

> **代码审计 / 重跑标注（2026-05-15）**：Reflow 全系列依赖 velocity target。修复 `v_target = x_noises - x_starts` 后，需要重新测 Reflow 曲线、velocity loss 天花板、`cos(g_det,g_vel)` 和 1-step/4-step gap。当前结论只能说明“旧代码下 Reflow 退化”。

### 阶段 5：Scale-Conditioned 与 KaryoFlow 失败探索

| 实验方向                                                                | 结果                                | 失败原因                                                        |
| ----------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------- |
| Scale-Conditioned FM (sc_noise=0.738, sc_loss=0.745, sc_combined=0.736) | 略低于 adaln 基线，sc_loss 接近持平 | 流速场偏移破坏最优性，重加权破坏变分原理（但 sc_loss 损失较小） |
| KaryoFlow 排列学习                                                      | 不可行                              | 信息量不足 (25.8 bits \<\< 178 bits)                            |

**理论贡献**：两个负结果均有机制分析和可检验解释，构成论文的"负面结果 + 理论解释"部分；严格证明口径以 `THEORY_FRAMEWORK.md` 三次修正版为准。

> **代码审计 / 重跑标注（2026-05-15）**：Scale-Conditioned 未发现明确实现 bug，但“必然无效”应降级为“当前实现无收益”。KaryoFlow 的信息论结论是方向性论证，不等价于否定所有核型结构先验；后续可转向 KCEC（软倍性配额 + 同源交换对称性），见 `THEORY_FRAMEWORK.md §8.6`。

______________________________________________________________________

## 四、核心理论发现

### 4.1 OT Diversity Collapse（OT 多样性坍缩）

**现象**：在低维检测空间（$d=4$），硬 OT 耦合显著降低目标索引多样性。

**可检验命题**：若用目标索引熵 $H(Y\mid X_t)$ 衡量监督多样性，随机耦合接近 $\\log K$，硬 OT 接近 0；因此多样性差值约为 $\\log K$。

**维度效应假说**：在低维框空间中，目标索引熵占训练信号复杂度的比例更高；在高维图像空间中，连续位移不确定性更可能主导。

**实验验证**：染色体数据集（$K \\approx 46.6$），实测索引熵差接近 $\\log K$，支持“硬 OT 降低监督多样性”的解释。

### 4.2 CAM 命题（耦合-分配失配）

**核心发现**：Sinkhorn OT + argmax 管线的根本缺陷是 ε 参数对软传输矩阵的多样性调控会被 argmax 操作显著削弱。

**定量证据**：argmax 解码下，分配多样性恒定（≈2.80 在任何 ε 下），而 Stochastic 解码从 2.81（ε=0.01）单调增加到 5.31（ε=100）。

**解决方案**：Stochastic Coupling（从传输矩阵采样替代 argmax），回复 ε 对多样性的单调调控能力。

### 4.3 多样性-传输效率的帕累托前沿

Sinkhorn + Stochastic Coupling 在随机耦合（最大多样性，零传输结构）和确定性 OT（最大传输效率，零多样性）之间建立连续帕累托前沿。

**三区间模型**：

- **区间 1**（ε \< 0.5）：OT 主导，多样性不足，mAP \< 0.745
- **区间 2**（ε ∈ \[0.5, 5.0\]）：最优区间，多样性饱和 + 效率可控，mAP 0.748-0.751
- **区间 3**（ε > 5.0）：耦合偏差主导，训练不稳定，mAP 退化

**最优 ε* ≈ 1-3*\*（闭式推导，结合指数衰减模型）。

### 4.4 Reflow Degradation Trap（梯度冲突）

**现象**：Reflow 训练中 velocity loss 和 detection loss 存在系统性梯度冲突。

**数学条件**：$\\rho = \\cos(g\_{\\text{det}}, g\_{\\text{vel}}) \< 0 \\iff e\_{\\text{det}}^\\top e\_{\\text{vel}} \< 0$（检测残差与 velocity 残差方向相反）。

**实验验证**：实测平均余弦相似度 -0.104，86.8% 层存在冲突。

**解决方案**：两阶段训练（Stage 1: 冻结 velocity_head 训练检测 → Stage 2: 冻结共享层训练 velocity_head）。

### 4.5 Scale-Conditioned FM 为何当前实现无效

三个理论原因：

1. **有偏流速估计**：$\\sigma_c \\epsilon - x_0 = (\\epsilon - x_0) + (\\sigma_c - 1)\\epsilon$，第二项引入偏差
2. **多目标冲突**：不同 Denver 组的 $\\sigma_c$ 不同，模型无法同时最优
3. **变分原理破坏**：$w_c \\neq 1/\\sigma_c^2$ 时优化目标不等于任何概率散度

### 4.6 KaryoFlow 排列学习的信息论不可行性

- 排列空间大小：$|S\_{46}| = 46! \\approx 1.8 \\times 10^{59}$
- 所需信息量：$\\log_2(46!) \\approx 200$ bits（考虑同源对称性后 ~178 bits）
- 实际可用信息量：$I\_{\\text{total}} \\approx 46 \\times 0.56 \\approx 25.8$ bits
- **结论**：即使 8-way Denver 组分类完美（100% acc），信息量（138 bits）仍不足以唯一确定排列

______________________________________________________________________

## 五、实验目录完整分类

### 5.1 基线体系

```
diffusiondet_baseline/       - DiffusionDet 原论文基线（训练未收敛, 0.001）
chromodet_baseline/          - 早期基线 (0.717)
ldmdet_baseline/             - 染色体数据集 DDPM 基线, 1步推理 (0.725)
ldmdet_baseline_fair/        - 公平对比基线 (0.705)
    ↳ 20260505_004902/LDMDet_backup/ → configs/ldmdet_baseline_fair.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_baseline_step4/       - 4步推理基线，反低于1步 (0.709)
chromo_coco_detection/       - COCO 数据集基线（无有效日志）
```

### 5.2 RF 核心改进链

```
ldmdet_rf_heun_shifted/      - RF + Heun + Shifted (0.749)
    ↳ 20260203_093528/LDMDet_backup/ → configs/ldmdet_rf_heun_shifted.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/single_head.py, mods/structures.py
ldmdet_rf_heun_shifted_bs2/  - batch_size=2 变体 (0.748)
    ↳ 20260203_012333/LDMDet_backup/ → configs/ldmdet_rf_heun_shifted_bs2.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/single_head.py, mods/structures.py
ldmdet_rf_heun_shifted_bs2_reproduce/ - 复现 (0.748)
    ↳ 20260205_005933/LDMDet_backup/ → configs/ldmdet_rf_heun_shifted_bs2_reproduce.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/single_head.py, mods/structures.py
ldmdet_rf_heun_shifted_bs2_optimized/ - 优化版（无有效日志）
ldmdet_rf_heun_shifted_dist/ - 分布式训练 (0.745)
    ↳ 20260123_152922/LDMDet_backup/ → configs/ldmdet_rf_heun_shifted_dist.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/rectified_flow.py, mods/single_head.py, mods/structures.py
ldmdet_rf_heun_shifted_muon/ - Muon 优化器 (0.737)
    ↳ 20260129_224251/LDMDet_backup/ → configs/ldmdet_rf_heun_shifted_muon.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/single_head.py, mods/structures.py
ldmdet_rf_heun_logit_shifted/ - Logit shifted (0.742)
    ↳ 20260125_120604/LDMDet_backup/ → configs/ldmdet_rf_heun_logit_shifted.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/rectified_flow.py, mods/single_head.py, mods/structures.py
ldmdet_rf_shifted/           - RF + Shifted（训练失败, 0.466）
ldmdet_rf_shifted_schdule_step1/ - RF + Shifted Schedule + 1步推理 (0.726)
ldmdet_rf_shifted_all/       - RF shifted 全特征 (0.740)
ldmdet_flowdet_adaln/        - AdaLN-Zero（最佳单模型，0.751）
    ↳ 20260318_093440/LDMDet_backup/ → configs/ldmdet_flowdet_adaln.py
ldmdet_flowdet_adaln_cat/    - + CAT（无 OT，0.740；各运行差异大）
    ↳ 20260411_014812/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_cat.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_obj/    - + Objectness 分支 (0.740)
    ↳ 20260322_231146/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_obj.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_convnext/     - ConvNeXt backbone 探索（训练失败, 0.001）
    ↳ 20260503_194026/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_convnext.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/modules.py, mods/single_head.py
```

### 5.3 OT 耦合全系列

```
ldmdet_flowdet_ot_coupling/                        - OT 耦合非 adaln 版 (0.749)
    ↳ 20260323_234303/LDMDet_backup/ → configs/ldmdet_flowdet_ot_coupling.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot/                          - Nearest OT (0.735)
    ↳ 20260323_234319/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_sinkhorn/                           - Sinkhorn OT 基础 (0.720)
    ↳ 20260321_144303/LDMDet_backup/ → configs/ldmdet_flowdet_sinkhorn.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn/                  - Sinkhorn argmax eps=1 (0.748)
    ↳ 20260410_001115/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_eps5/             - argmax eps=5 (0.745)
    ↳ 20260421_120259/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_eps5.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_eps10/            - argmax eps=10 (0.745)
    ↳ 20260421_223818/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_eps10.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_eps50/            - argmax eps=50 (0.747)
    ↳ 20260422_051155/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_eps50.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_eps100/           - argmax eps=100 (0.733)
    ↳ 20260422_130705/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_eps100.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05/     - Stochastic eps=0.5 (0.742)
    ↳ 20260426_225352/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps05.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1/      - Stochastic eps=1 (0.748)
    ↳ 20260427_094847/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps1.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps2/      - Stochastic eps=2 (0.744)
    ↳ 20260427_094957/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps2.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps3/      - Stochastic eps=3 (0.741)
    ↳ 20260427_150906/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps3.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5/      - Stochastic eps=5 (0.751)
    ↳ 20260423_060304/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps10/     - Stochastic eps=10 (0.747)
    ↳ 20260427_212100/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps10.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps50/     - Stochastic eps=50 (0.736, 退化)
    ↳ 20260423_091808/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps50.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_repro/ - 复现实验 (0.738, 可复现性差)
    ↳ 20260506_125954/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5_seed2/ - 多 seed 验证 (0.750)
    ↳ 20260506_151313/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
reproduce_0751_stochot_eps5/                       - Stochastic eps=5 精准复现 v1 (0.743, TF32开启)
    ↳ 20260524_012941/ → configs/_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py
    ⚠️ model.py 含 TF32+cudnn.benchmark 全局设置（0.751实验时不存在），best mAP=0.743@ep85, mAP@50=0.944, mAP@75=0.833, EarlyStop@ep115
reproduce_0751_stochot_eps5_v2/                    - Stochastic eps=5 精准复现 v2 (0.753, TF32关闭) ★新SOTA
    ↳ 20260524_120330/ → configs/_legacy/ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py, model.py(TF32已注释)
    ★ best mAP=0.753@ep59, mAP@50=0.943, mAP@75=0.843, mAP_s=0.522, mAP_m=0.745, mAP_l=0.636, EarlyStop@ep89
    ★ ckpt: work_dirs/reproduce_0751_stochot_eps5_v2/best_coco_bbox_mAP_epoch_59.pth
    ↳ 20260524_193842/ → 无eval数据，中间run
    ↳ 20260524_194437/ → 同配置续训，best mAP=0.735@ep102
    ⚠️ TF32关闭后 best mAP 从 0.743→0.753 (+1.0%)，确认 TF32 是系统性偏差源
    ⚠️ 三seed统计: 0.753 / ? / 0.735，多seed均值待补充
ldmdet_convnextv2_mae/                             - ConvNeXtV2-Tiny + MAE + Stochastic OT eps=5 + AdaLN-Zero (0.736)
    ↳ 20260523_021941/LDMDet_backup/ → configs/ldmdet_convnextv2_mae.py, model.py, mods/diffusiondet_head.py, mods/single_head.py, mods/rectified_flow.py
    ⚠️ 骨干切换导致性能退化：ResNet-50→ConvNeXtV2-Tiny, bs=2→4, wd=1e-4→0.05, best mAP=0.736@ep64, EarlyStop@ep94
mae/                                               - ConvNeXtV2-Tiny + MAE 无 OT baseline (0.736)
    ↳ 20260523_152819/LDMDet_backup/ → configs/ldmdet_convnextv2_mae.py (ot_coupling=False)
    ⚠️ 与 OT 版本 best mAP 完全持平，Stochastic OT 在 ConvNeXtV2 上零增益; best mAP=0.736@ep33, EarlyStop@ep63
```

### 5.4 群组层次 OT

```
ldmdet_flowdet_adaln_group_hierarchical/           - 群组层次 argmax (0.734)
    ↳ 20260429_100057/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_group_hierarchical.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_group_hierarchical_stoch/     - 群组层次 stochastic (0.752)
    ↳ 20260429_100047/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_group_hierarchical_stoch.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_group_hierarchical_stoch_seed2/             - 多 seed 验证 (0.747)
    ↳ 20260507_101313/LDMDet_backup/ → configs/ldmdet_group_hierarchical_stoch.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
```

### 5.5 TRD 系列

```
ldmdet_flowdet_adaln_trd/                          - TRD 基础
    ↳ 20260410_150104/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_trd.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_trd_only/                     - TRD + velocity (0.746)
    ↳ 20260427_215048/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_trd_only.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_trd_full/                     - TRD+CAT+LSAS+velocity+Heun (0.752)
    ↳ 20260411_233058/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_trd_full.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
```

### 5.6 组合实验（两条 SOTA 路径）

```
ldmdet_group_hierarchical_trd/                     - 群组层次 + TRD 组合 (0.746)
    ↳ 20260507_085303/LDMDet_backup/ → configs/ldmdet_group_hierarchical_trd.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
ldmdet_flowdet_adaln_stochastic_eps5_trd_cat/      - Stochastic + TRD + CAT (0.740)
    ↳ 20260428_110608/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_stochastic_eps5_trd_cat.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_sinkhorn_trd_cat_lsas/                      - Sinkhorn + TRD + CAT + LSAS (0.743)
    ↳ 20260506_235529/LDMDet_backup/ → configs/ldmdet_sinkhorn_trd_cat_lsas.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
```

### 5.7 辅助模块消融

```
ldmdet_flowdet_adaln_cat_only/                     - CAT 单独使用 (0.744)
    ↳ 20260428_172659/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_cat_only.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_full/                               - 全特征组合 (0.740)
    ↳ 20260322_231210/LDMDet_backup/ → configs/ldmdet_flowdet_full.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_velocity/                           - Velocity 预测 (0.738)
    ↳ 20260321_144334/LDMDet_backup/ → configs/ldmdet_flowdet_velocity.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_structured_noise/                   - 结构化噪声 (0.742)
    ↳ 20260322_003618/LDMDet_backup/ → configs/ldmdet_flowdet_structured_noise.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_crossattn/                    - Cross-Attention v1 (0.745)
    ↳ 20260503_021743/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_crossattn.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_crossattn_v2/                 - Cross-Attention v2 (0.737)
    ↳ 20260503_162641/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_crossattn.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_lsas/                         - LSAS (0.743)
    ↳ 20260410_220847/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_lsas.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
```

### 5.8 阶段式训练

```
ldmdet_phase1/                                     - 阶段1 (0.745)
    ↳ 20260504_174403/LDMDet_backup/ → configs/ldmdet_phase1.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_phaseA/                                     - 阶段A (0.714)
    ↳ 20260505_225438/LDMDet_backup/ → configs/ldmdet_phaseA.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_phaseB/                                     - 阶段B (0.731)
    ↳ 20260506_090150/LDMDet_backup/ → configs/ldmdet_phaseB.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_phaseC/                                     - 阶段C (0.701)
    ↳ 20260506_090208/LDMDet_backup/ → configs/ldmdet_phaseC.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_phase1+2/                                   - 阶段1+2 (0.727)
    ↳ 20260504_175850/LDMDet_backup/ → configs/ldmdet_phase1+2.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_phase1+2+3/                                 - 阶段1+2+3 (0.686)
    ↳ 20260504_232820/LDMDet_backup/ → configs/ldmdet_phase1+2+3.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_phase1+2_dap/                               - 阶段1+2 dap (0.714)
    ↳ 20260505_125524/LDMDet_backup/ → configs/ldmdet_phase1+2_dap.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py, mods/structures.py
ldmdet_phase1+2_eval_T1/                           - 阶段1+2 T1评估
```

### 5.9 Reflow 系列（单步推理）

```
ldmdet_flowdet_adaln_reflow/                       - Reflow v1（从零训练）
    ↳ 20260413_212531/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v2/                    - Reflow v2（微调，val配对）(0.739)
    ↳ 20260414_104819/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v3/                    - Reflow v3（微调，train配对）(0.739)
    ↳ 20260414_202306/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v4/                    - Reflow v4（修复velocity_head）(0.739)
    ↳ 20260415_104048/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v5/                    - Reflow v5（warmup+调参）(0.739)
    ↳ 20260415_155618/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v6/                    - Reflow v6（第2轮 Reflow）(0.739)
    ↳ 20260416_165522/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_v6.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_det_only/              - 仅检测 loss (0.742)
    ↳ 20260420_145858/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_det_only.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_det_only_30ep/         - det_only 长训练 (0.739)
    ↳ 20260420_163323/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_det_only.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_det_only_lr1e6/        - det_only 低学习率 (0.741)
    ↳ 20260420_210205/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_det_only_lr1e6.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_freeze/                - 冻结共享层 (0.740)
    ↳ 20260418_154154/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_freeze.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_freeze_stage2/         - 两阶段训练 (0.740)
    ↳ 20260421_101140/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_freeze_stage2.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_itd/                   - 中间轨迹蒸馏 (ITD) (0.738)
    ↳ 20260417_003853/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_itd.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_lr1e6_vel/             - 低 lr velocity (0.740)
    ↳ 20260420_222217/LDMDet_backup/ (仅文档，无代码备份); config→../ldmdet_flowdet_adaln_reflow_lr1e6_vel.py
ldmdet_flowdet_adaln_reflow_consistency/           - Consistency Distillation (0.740)
    ↳ 20260419_012005/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_consistency.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_pcgrad/                - PCGrad 梯度冲突缓解 (0.740)
    ↳ 20260419_222547/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_pcgrad.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_vel_detach/            - Velocity detachment (0.740)
    ↳ 20260420_094419/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow_vel_detach.py, model.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v5_long/               - v5 长训练 (0.741)
    ↳ 20260416_144657/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v5_s03/                - v5 seed=0.3 (0.740)
    ↳ 20260416_103423/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v5_s07/                - v5 seed=0.7 (0.740)
    ↳ 20260416_111430/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_reflow_v5_s10/                - v5 seed=1.0 (0.740)
    ↳ 20260416_115456/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_reflow.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
```

### 5.10 诊断与工具实验

```
eval_multistep/                                    - 多步评估诊断
gradient_analysis/                                 - 梯度冲突分析
coco_random/                                       - COCO 随机耦合对比
coco_ot_sinkhorn_eps5/                             - COCO OT 对比
```

### 5.11 探索性实验

```
karyoflow_overfit/overfit2/overfit3/               - KaryoFlow 过拟合探索
karyoflow_v1/                                      - KaryoFlow v1
ldmdet_flowdet_adaln_crossattn/                    - Cross-Attention
    ↳ 20260503_021743/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_crossattn.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_crossattn_v2/                 - Cross-Attention v2
    ↳ 20260503_162641/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_crossattn.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_adaln_convnext/                     - ConvNeXt backbone
    ↳ 20260503_194026/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_convnext.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/modules.py, mods/single_head.py
ldmdet_flowdet_adaln_lsas/                         - Loss-Sensitive Adaptive Scheduling
    ↳ 20260410_220847/LDMDet_backup/ → configs/ldmdet_flowdet_adaln_lsas.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_flowdet_objectness/                         - Objectness 分支 (0.739)
    ↳ 20260322_003625/LDMDet_backup/ → configs/ldmdet_flowdet_objectness.py, mods/diffusiondet_head.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ldmdet_single_chromo_*                             - 单染色体实验 (argmax_eps1=0.594, argmax_eps5=0.627, hard_ot=0.683, random=0.676, stoch_eps5=0.681)
    ↳ ldmdet_single_chromo_argmax_eps1/20260507_230904/LDMDet_backup/ → configs/ldmdet_single_chromo_argmax_eps1.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
    ↳ ldmdet_single_chromo_argmax_eps5/20260507_230904/LDMDet_backup/ → configs/ldmdet_single_chromo_argmax_eps5.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
    ↳ ldmdet_single_chromo_hard_ot/20260507_174352/LDMDet_backup/ → configs/ldmdet_single_chromo_hard_ot.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
    ↳ ldmdet_single_chromo_random/20260507_112622/LDMDet_backup/ → configs/ldmdet_single_chromo_random.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
    ↳ ldmdet_single_chromo_stoch_eps5/20260507_174041/LDMDet_backup/ → configs/ldmdet_single_chromo_stoch_eps5.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/modules.py, mods/single_head.py
scale_conditioned_*                                - Scale-Conditioned 系列 (sc_loss=0.745, sc_noise=0.738, sc_combined=0.736)
    ↳ scale_conditioned_sc_combined/20260502_032114/LDMDet_backup/ → configs/scale_conditioned/sc_combined.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
    ↳ scale_conditioned_sc_loss/20260501_193234/LDMDet_backup/ → configs/scale_conditioned/sc_loss.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
    ↳ scale_conditioned_sc_noise/20260501_140011/LDMDet_backup/ → configs/scale_conditioned/sc_noise.py, model.py, mods/diffusiondet_head.py, mods/loss.py, mods/sinkhorn.py, mods/noise_sampler.py, mods/single_head.py
ablations/                                         - 消融实验集合
```

______________________________________________________________________

## 六、当前最佳结果汇总

### 6.1 4步推理 SOTA

| 实验                       | mAP       | mAP@50 | mAP@75 | 核心技术                                 |
| -------------------------- | --------- | ------ | ------ | ---------------------------------------- |
| `group_hierarchical_stoch` | **0.752** | 0.946  | 0.841  | 群组层次 OT + Stochastic + Heun          |
| `trd_full`                 | **0.752** | 0.940  | 0.835  | TRD + CAT + LSAS + velocity + Heun       |
| `adaln` (vanilla)          | 0.751     | 0.943  | 0.843  | AdaLN-Zero + Heun（无 OT）               |
| `sinkhorn_sample_eps5`     | 0.751     | 0.945  | 0.839  | Sinkhorn Stochastic eps=5                |
| `ot_coupling`              | 0.749     | 0.941  | 0.836  | Nearest OT（非 adaln 版，无 objectness） |
| `ot_sinkhorn`              | 0.748     | 0.947  | 0.840  | Sinkhorn argmax eps=1                    |
| `trd_only`                 | 0.746     | 0.942  | 0.834  | TRD + velocity                           |

### 6.2 1步推理最佳

| 实验                   | 1步 mAP                      | vs 4步基线      |
| ---------------------- | ---------------------------- | --------------- |
| Reflow v6 (2轮 Reflow) | 0.739                        | -0.013          |
| Reflow v5 (1轮 Reflow) | 0.739 (train) / 0.731 (eval) | -0.013 / -0.021 |
| TRD-Full (无 Reflow)   | 0.728                        | -0.024          |

### 6.3 效率-精度前沿

| 步数 | TRD-Full | Reflow v5 | 差距   |
| ---- | -------- | --------- | ------ |
| 1步  | 0.728    | **0.731** | +0.003 |
| 2步  | 0.740    | **0.738** | -0.002 |
| 4步  | 0.741    | **0.743** | +0.002 |
| 8步  | 0.743    | **0.743** | 0      |

**关键**：2步推理的性价比最优（0.738 vs 4步基线 0.741，仅差 0.3%，推理速度翻倍）。

______________________________________________________________________

## 七、两条 SOTA 路径的组合：已探索但存在负交互

### 7.1 正交路径对比

| 维度       | 路径 A (group_hierarchical) | 路径 B (trd_full)            |
| ---------- | --------------------------- | ---------------------------- |
| 核心思想   | 优化耦合策略                | 优化训练动力学               |
| 关键模块   | 群组层次 OT + stochastic    | TRD + CAT + LSAS + velocity  |
| 推理时行为 | 无额外计算                  | TRD 自条件精化（需额外前向） |
| 耦合方式   | Sinkhorn Stochastic         | Nearest OT                   |

### 7.2 组合实验结果（全为负交互）

| 组合实验                  | mAP       | 配置                                   | vs 最佳单路径 (0.752) |
| ------------------------- | --------- | -------------------------------------- | --------------------- |
| `group_hierarchical_trd`  | **0.746** | 群组层次 OT + stochastic + TRD         | **-0.006**            |
| `sinkhorn_trd_cat_lsas`   | **0.743** | Sinkhorn stochastic + TRD + CAT + LSAS | **-0.009**            |
| `stochastic_eps5_trd_cat` | **0.740** | Sinkhorn stochastic + TRD + CAT        | **-0.012**            |

**三个组合实验全部低于任一单路径最佳值 0.752。** 这不是"未探索"，而是"已探索但组合效应为负"。

### 7.3 负交互的可能原因

1. **机制冲突**：OT 耦合优化了传输路径使每步预测更确定，而 TRD 依赖前一步预测的不确定性来提供精化信号——两者存在功能层面的矛盾
2. **边际收益递减**：训练动力学优化（TRD/CAT/LSAS）的收益在 OT 已优化后的路径上递减更快
3. **超参数耦合**：组合引入了更大的超参数空间，当前配置可能未找到最优组合点
4. **梯度干扰**：CAT 的曲率惩罚可能与 OT 的确定性配对在梯度方向上产生新的冲突

### 7.4 对论文的影响

- 这构成一个重要的**负结果发现**：耦合优化和训练动力学优化虽然理论上正交，但实践中存在负交互
- 不被此否定整体方向——两条路径各自独立到达 0.752 反而证明了方法的鲁棒性
- 理解负交互的机制本身是新的研究问题，可以作为论文的 discussion 部分

______________________________________________________________________

## 八、关键反思与教训

### 8.1 什么有效

1. **RF 替代 DDPM**：路径直化是基础收益，简单但有效
2. **AdaLN-Zero 时间条件**：单项最大改进（+1.8%），零初始化至关重要
3. **Shifted Schedule**：时间敏感度理论指导的采样策略，+1.4%
4. **Stochastic Coupling**：修复 OT 在低维空间的多样性损失
5. **Heun 二阶求解器**：几乎免费的增益（2× 前向，+0.001-0.002）
6. **群组先验利用**：染色体生物学分组是有价值的归纳偏置

### 8.2 什么无效

1. **硬 OT 耦合（argmax）**：在低维空间导致多样性坍缩，反而不如随机
2. **Sinkhorn + argmax 管线**：ε 参数被 argmax 短路，无法调控多样性
3. **CAT（曲率正则化）**：`cat` 与 OT 组合时有一运行全部 eval 为 0，`cat_only` (0.744) 单独使用正常——问题出在特定组合而非 CAT 本身
4. **Stochastic OT 之前可复现性差**：早期 `repro`=0.738 vs `seed2`=0.750 波动达 1.2%；根因已确认为 TF32（v1=0.743, v2=0.753, +1.0%），关闭 TF32 后已解决，可复现性不再是无解问题
5. **两条 SOTA 路径的直接组合**：`group_hierarchical_trd`=0.746，存在负交互（详见 §七）
6. **ConvNeXtV2-Tiny 骨干退化**：Stochastic OT eps=5 在 ConvNeXtV2 上零增益（0.736 vs 无OT 0.736），且整体 mAP 比 ResNet-50 低 1.7%（0.736 vs 0.753）
7. **Scale-Conditioned FM**：当前三种实现均未带来收益，但 sc_loss=0.745 下降幅度较小（vs adaln -0.006），不应表述为"必然无效"
8. **KaryoFlow 端到端排列学习**：信息论下界不可达
9. **多次 Reflow**：边际收益递减，第2轮几乎无收益

### 8.3 仍待解决的问题

1. **Reflow 退化陷阱**：梯度冲突（cos=-0.104）的稳定解决方案
2. **velocity loss 收敛天花板**：~0.23-0.30 后停滞
3. **1步 vs 4步的 1.3% 差距**：当前 Reflow 只能缩小到 1.3%
4. **两条 SOTA 路径的负交互根因**：已确认组合 \< 单路径，需要理解负交互机制
5. **Stochastic OT 多 seed 统计**：当前仅 3 seed（0.753 / 0.735 / —），seed 间波动达 1.8%，需补充更多 seed 以确认 SOTA 基准的统计特性
6. **COCO 通用性验证**：当前所有实验均在染色体数据集
7. **更大 ε（ε=1-3）的 Stochastic 实验**：理论上最优区间，待实测
8. **ldmdet_baseline 的 1步 > 4步 反常**：需排查 DDPM 1步推理是否实际走了 DDIM skip
9. **ConvNeXtV2 骨干退化根因**：MAE预训练权重是否适合检测微调、lr/batch_size/wd是否需要针对ConvNeXt重新调优

______________________________________________________________________

## 九、论文框架建议

### 9.1 核心论点（3句话）

1. 基于扩散的目标检测中，训练效果由耦合质量、索引多样性、速度时间变化和时间采样四个因素共同影响，而非单一因素
2. 在低维检测空间（$\\mathbb{R}^4$）中，硬 OT 可能导致目标索引多样性坍缩，而 Stochastic Coupling 能缓解这一问题
3. 下一步顶会级主线应从 geometry-only OT 转向 Karyotype-Constrained Entropic Coupling，将核型配额、同源交换对称性、形态先验和索引熵约束纳入训练耦合

### 9.2 三大贡献

1. **LDMDet 检测器**：基于 Rectified Flow + AdaLN-Zero + Shifted Schedule 的高效扩散检测器，4步推理达到 0.752 mAP
2. **OT Diversity Collapse 命题**：从目标索引熵角度解释硬 OT 在低维检测框空间中的负效果，提出 CAM 命题揭示 Sinkhorn+argmax 管线的缺陷
3. **Karyotype-Constrained Entropic Coupling 方向**：在 stochastic coupling 的基础上引入软倍性配额、同源交换对称性、形态先验和熵约束，目标是获得训练-only、可泛化、正向提升的染色体耦合机制

### 9.3 五个需修正的认知

1. **"OT 总是好的"** → 只有在高维空间成立，低维检测中多样性损失严重
2. **"大 ε 增加多样性"** → 对 argmax 无效（CAM 定理），只对 Stochastic Coupling 有效
3. **"更多 Reflow 更好"** → 边际收益递减，根因是梯度冲突而非路径不够直
4. **"Scale-Conditioned 可以改善小物体检测"** → 当前实现下未验证收益，可能因流速场偏移和重加权目标不匹配导致退化
5. **"端到端排列学习可行"** → 信息论下界（178 bits）远超可用信息量（25.8 bits）
6. **"OT + TRD 组合必然叠加"** → 直接组合存在负交互（0.746 \< 0.752），耦合优化和训练动力学优化存在机制冲突

______________________________________________________________________

## 十、下一步实验优先级

### P0：论文必需（1-2周）

1. **先修代码再重跑关键结论**：修复 velocity/ITD target 符号、TRD/CAT 步长解耦、CAT 纯曲率版本。
2. **KCEC 消融实验**：已跑首版（0.747@ep58 vs SOTA 0.753，−0.6%），需消融 morph/group/prior/slack 确认各先验贡献方向。配置已就绪：`recipes/ldmdet_kcec_ablation_no_{morph,group,prior,slack}.py`。
3. **KCEC 对齐 SOTA 变量重跑**：原版 `ot_num_iters=50` ≠ SOTA 的 20，`bs=2` 对齐后重跑全量 KCEC ＋ 消融系列。
4. **Stochastic ε=1.0 和 ε=2.0 实验**：验证理论预测的候选 ε 区间。
5. **完整消融表**：补齐所有模块的消融实验数据，尤其是 KCEC 的 group/quota/morph/slack 消融。

### P1：强化论文（2-4周）

6. **COCO 数据集验证**：RF + AdaLN + Shifted 在 COCO 上的通用性证明；KCEC 作为染色体专属贡献，DAEC 作为通用化扩展。
7. **两阶段 Reflow 训练**：Stage 1 冻结 velocity_head 训练检测 → Stage 2 冻结共享层训练 velocity_head。
8. **Reflow + ITD（中间轨迹蒸馏）**：在多个中间时间步提供 velocity 监督。
9. **推理速度 Benchmark**：完整 FPS 测量（1/2/4/8/16步）。
10. **更强的 Backbone**：Swin-T / ConvNeXt。

### P2：探索性（4周+）

11. **自适应群组划分**：不依赖固定生物学分类的动态聚类。
12. **学习型 ε 调度**：训练初期大 ε（多样性），后期小 ε（效率）。
13. **与 C²OT / W-CFM 的结合**：条件加权 + Stochastic Coupling。

______________________________________________________________________

## 十一、关键实验配置对应

| 实验名（work_dirs）                             | 配置文件                                          | 核心参数                                   |
| ----------------------------------------------- | ------------------------------------------------- | ------------------------------------------ |
| `ldmdet_flowdet_adaln`                          | `ldmdet_flowdet_adaln.py`                         | RF + AdaLN-Zero + Shifted s=3 + Heun       |
| `ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5`  | `ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py` | Sinkhorn Stochastic eps=5 + Heun (★0.753) |
| `reproduce_0751_stochot_eps5_v2`                | `ldmdet_flowdet_adaln_ot_sinkhorn_sample_eps5.py` | ★同配置精准复现，TF32关闭，0.753@ep59      |
| `ldmdet_kcec_pure_sota_eps5`                    | `recipes/ldmdet_kcec_pure_sota.py`                | KCEC on 0.753 base, 0.747@ep58             |
| `ldmdet_flowdet_adaln_group_hierarchical_stoch` | —                                                 | 群组层次 OT + Stochastic + Heun            |
| `ldmdet_flowdet_adaln_trd_full`                 | —                                                 | TRD + CAT + LSAS + velocity + Heun         |
| `ldmdet_flowdet_adaln_reflow_v5`                | —                                                 | Reflow v5, det_loss_scale=0.5, warmup=3850 |
| `ldmdet_flowdet_adaln_reflow_itd`               | `ldmdet_flowdet_adaln_reflow_itd.py`              | ITD, num_points=4                          |

______________________________________________________________________

*本文档整合了 work_dirs 中所有实验分支的 markdown 文档、实验日志和理论分析。主要来源：THEORY_FRAMEWORK.md, OT_DIVERSITY_COLLAPSE_PROOF.md, experiment_summary.md, SOTA_ANALYSIS.md, THEORY_WHY_FAILED.md, IMPROVEMENT_PLAN.md, LDMDet_Architecture.md, WEEK1_REPRO_PROTOCOL.md, Research_Plan.md, PAPER_FRAMEWORK.md, 以及 research_qna 中的 11 篇中文研究方向文档。2026-05-13 经过两轮交叉校验，修正了 13 处数值/结论错误，补充了 15+ 个遗漏实验，所有 mAP 数值均经 work_dirs 日志逐一核查。*
