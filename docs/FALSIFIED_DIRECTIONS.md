# 证伪方向记录 (Falsified Directions)

> 📋 **命名约定**: 本文档使用描述性名称 (Dataset 1 / Dataset 2 / RF+Heun / +AdaLN-Zero / +Stoch. Coupling / +DPM-Solver++ / Top-K / Adaptive Step / Draft-Verify / Head Early-Exit / RoI Feature Cache / Cascade Head Count / ScaleConditionedRF / FBM variants / Deformable P1 Head / Decoupled Head / Morphology-Contrastive Head / Box Refine Net / Class-Balanced Sampling / Structured Prior Head)。内部实验代号 (24obj / chromo / A0-A4 / IO1-IO5 / E4.3 / E6.x / Direction A-F / N_cascade / StochOT) 仅保留在文件路径、配置名和 SwanLab run_id 中以兼容工程实现。

> 本文档梳理 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 研究过程中所有已证伪/失败的研究方向。
> 这些负面结果对应论文 Appendix B "被证伪的方向", 记录内部研究决策, 证明最终设计选择合理性。
> 每条记录附 SwanLab run_id 与可靠数据源地址, 与 [docs/EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md) 主路线文档格式一致。
>
> 📌 **数据集说明**: 除特别标注 Dataset 2 的实验外, 多数结论基于旧数据集 Chromosome20240904 (Dataset 1, mAP≈0.72-0.75), baseline = `rf_heun_adaln` = 0.746 ± 0.001 (3 seeds)。Dataset 2 (mAP 量级 0.77-0.87) 实验单独标注, 不可与 Dataset 1 对比。
>
> SwanLab 用户名: `einspanner` (登录态见 `/home/linkst/.swanlab/.netrc`, api_key 已配置)
> URL 拼接示例: `https://swanlab.cn/@einspanner/<project>/runs/<run_id>`
>
> 更新时间: 2026-07-28

## 〇、Baseline 对照与证伪判据

| 数据集 | Baseline | mAP | SwanLab | 用途 |
|--------|----------|-----|---------|------|
| Dataset 1 (Chromosome20240904) | rf_heun_adaln (3 seeds) | **0.746 ± 0.001** | ldmdet-ablation/rf_heun_adaln_seed* | 多数证伪实验对照 |
| Dataset 2 (24 Chromosomes Object) | +Stoch. Coupling + DPM-Solver++ (3 seeds) | **0.859 ± 0.003** | ldmdet-mainline-ablation-24obj | Inference-Time Optimization directions / Cascade Head Count / Flow Matching 对照 |

证伪判据:
- 显著退化 (Δ ≤ −0.005, 超出种子方差 ±0.001-0.003): ⛔ 证伪
- 持平 (|Δ| < 0.002): 🟠 边际, 不采用
- 失败/崩溃 (无有效 mAP): ⛔ 失败

---

## 一、ScaleConditionedRF (尺度调制噪声调度, 证伪, 最详细)

### 核心设想

ScaleConditionedRF (SCRF) 提出尺度调制噪声调度 κ(s), 使小目标 (w·h 小, 信噪比低) 在更早的 t 去噪:
- 调度变换: `t_eff = t^(1/κ(s))`, 其中 `κ(s) = 1 + λ(s_max − s)/s_max`
- 参数: λ=0.5, s_max=0.15, s = sqrt(w·h) (归一化后)
- 训练时: 从 GT 计算 s (精确); 推理时: 从 x0_pred 计算 s (近似)
- 期望: 小目标 κ→1.5, t_eff→t^0.67, 在更早的 t 完成主要去噪, 缓解小目标检测困难

### 证伪过程

ScaleConditionedRF (OLD) (usnvd63f, 0.752) 早期声称 SCRF 有效, 但核查备份 head.py 发现 ScaleConditionedRF **未真正集成到 head.py**, 0.752 全部来自 OTFlowCoupling + 种子方差。
经 TDD 红绿重构 (48 单元测试全部通过), head.py 4 处真正集成 SCRF 后重训, 结果 0.741 < 0.746 baseline (−0.005), **证伪**。

#### 实验证明目的

验证 ScaleConditionedRF 真正集成到 head.py 后是否能超越 baseline 0.746。

- ScaleConditionedRF (OLD) (ScaleConditionedRF + OTFlowCoupling argmax eps=1.0, SCRF **未集成**)
  -- config: experiments/configs/ldmdet/directions/nonlinear_trajectory/nonlinear_trajectory.py
  -- work_dir: work_dirs/nonlinear_trajectory/
  -- SwanLab: ldmdet-ablation, run_id=usnvd63f, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/usnvd63f
  -- best mAP=0.7520 @ epoch 94
  -- ⚠ ScaleConditionedRF 当时未集成到 head.py, 0.752 主要来自 OTFlowCoupling + 种子方差

- ScaleConditionedRF eps=2.0 (OLD) (SCRF **未集成**, eps 调优)
  -- config: experiments/configs/ldmdet/directions/nonlinear_trajectory/nonlinear_trajectory_e43_eps2.py
  -- work_dir: work_dirs/nonlinear_trajectory_e43_eps2/
  -- SwanLab: ldmdet-ablation, run_id=cdtmijl0, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/cdtmijl0
  -- best mAP=0.7520 @ epoch 100
  -- ⚠⚠ 关键修正: 配置 dump 虽含 scale_conditioned_rf 字段, 但当时 head.py 未集成, ScaleConditionedRF 完全未生效! 0.752 实际来自 OTFlowCoupling(eps=2.0, argmax) + Heun + 种子方差

- ScaleConditionedRF eps=2.0 (NEW) (SCRF **真正集成**, TDD 红绿重构后) ⛔ 证伪
  -- config: experiments/configs/ldmdet/directions/nonlinear_trajectory/nonlinear_trajectory_e43_eps2.py (同上)
  -- work_dir: work_dirs/nonlinear_trajectory_e43_eps2_real/
  -- SwanLab: ldmdet-ablation, run_id=hkn0fc7w, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/hkn0fc7w
  -- best mAP=**0.741** @ epoch 69 (训练至 ep93 平台化)
  -- 进展: ep1=0.000 → ep10=0.564 → ep55=0.737 → ep69=**0.741** → ep93 平台
  -- ✓ TDD 红绿重构后, head.py 4 处真正集成 ScaleConditionedRF:
     1. _forward_diffusion (前向加噪, scales 参数)
     2. _build_training_targets (从 GT 计算 scales 经 matched_idx 映射)
     3. predict Euler/Heun 路径 (推理时从 x0_pred 计算 scales)
     4. _compute_inference_scales (raw→normalized cxcywh→sqrt(w*h))
  -- 48 单元测试全部通过 (test_nonlinear_trajectory.py)
  -- commit: e757b856 feat(scale-conditioned-rf): integrate ...
  -- ⛔⛔ **关键结论: ScaleConditionedRF 真正启用后性能下降 (0.741 < 0.746 baseline, Δ=−0.005)**
  -- Δ vs OLD = −0.011 (0.741 < 0.752, OLD 的 SCRF 未生效)

- 3-seed 复现 (ScaleConditionedRF eps=2.0 配置, 验证可复现性, SCRF **未集成**)
  -- config: experiments/configs/ldmdet/directions/nonlinear_trajectory/nonlinear_trajectory.py (seeds 1, 2, 3)
  -- seed 1
     -- work_dir: work_dirs/nonlinear_trajectory_seed1/
     -- SwanLab: nonlinear-3seed-repro, name=nonlinear_e43_seed1, run_id=k7nnzvuq, URL: https://swanlab.cn/@einspanner/nonlinear-3seed-repro/runs/k7nnzvuq
     -- best mAP=0.7460 @ epoch 72
  -- seed 2
     -- work_dir: work_dirs/nonlinear_trajectory_seed2/
     -- SwanLab: nonlinear-3seed-repro, name=nonlinear_e43_seed2, run_id=clpof6nn, URL: https://swanlab.cn/@einspanner/nonlinear-3seed-repro/runs/clpof6nn
     -- best mAP=0.7490 @ epoch 110
  -- seed 3 ⛔ 失败 (误启动, 仅 2 epoch 即被杀)
     -- work_dir: work_dirs/nonlinear_trajectory_seed3/ (无效)
     -- SwanLab: nonlinear-3seed-repro, run_id=kaz1tog7, mAP=0.0000
  -- 初步均值 (seed1+seed2): 0.7475 ± 0.0015, 落在 ±0.018 容差内
  -- 证实 0.752 的高方差, 均值 0.7475 与 baseline 0.746 持平 → 0.752 不是 SCRF 的贡献

### 理论缺陷分析

ScaleConditionedRF 真正启用后性能下降, 根本原因有四:

#### 缺陷 1: 训练-推理尺度不一致 (最致命)

```
训练: s = sqrt(w_gt · h_gt)          ← GT 真实尺度, 精确
推理: s = sqrt(w_pred · h_pred)      ← x0_pred 预测尺度, 早期 t≈1 时近乎随机
```

- 采样初期 t≈1, 模型输入几乎是纯噪声, x0_pred 完全不可靠
- 从不可靠的 x0_pred 计算的 κ(s) 和 t_eff 也是错误的
- **错误的 t_eff 导致采样轨迹偏离训练时学到的分布**, 误差逐步累积
- 这是结构性缺陷: 推理时无法获得 GT 尺度, 任何 proxy 都不可靠

#### 缺陷 2: 破坏 RF 的统一时间轴

Rectified Flow 理论要求所有样本在同一 t 下共享同一速度场 v(x_t, t):

```
标准 RF:  所有 box 在 t=0.5 时, x_t = (1-0.5)x_0 + 0.5·noise   ← 统一
SCRF:     大目标 t_eff=0.5, 小目标 t_eff=0.35                   ← 分裂!
          模型在 t=0.5 时同时看到"半噪声"和"三分之一噪声"的混合输入
```

模型无法在单个 t_input 下正确处理不同 t_eff 的样本, 速度场定义被破坏。

#### 缺陷 3: 小目标的数值不稳定

```python
v = (x_t - x0_pred) / t_eff   # t_eff 小时, 误差被放大
```

- 小目标 κ→1.5, t_eff→t^0.67, t_eff 在 t 小时趋近 0
- 除以小 t_eff 放大 x0_pred 的预测误差
- Heun 二阶进一步放大 (两次除法)

#### 缺陷 4: 推理时尺度反馈循环

```
x0_pred (噪声) → 算 s → 算 κ → 算 t_eff → step → 新 x → 新 x0_pred (仍噪声) → ...
```

每步推理都依赖上一步的噪声预测来决定本步的积分路径, 误差在 4 步采样中滚雪球。

### 历史归因修正

0.753 和 0.752 的改进 **全部来自 OT coupling 策略**, ScaleConditionedRF 从未生效:

| 实验 | mAP | Coupling | ε | bs | SCRF 配置 | SCRF 实际生效 | 真实改进来源 |
|------|-----|----------|---|----|-----------|--------------|-------------|
| reproduce_0751_stochot_eps5_v2 | **0.753** | sinkhorn_stochastic | 5.0 | 2 | 无 | — | Stochastic Coupling ε=5 |
| scheme_C1_5_mixed_rel_l1_lam015 | **0.752** | sinkhorn_stochastic | 5.0 | 2 | 无 | — | + mixed_relative_l1 损失 |
| nonlinear_trajectory ScaleConditionedRF (usnvd63f) | **0.752** | ot_flow (argmax) | 1.0 | 4 | 有 | ❌ **未集成** | ot_flow coupling |
| nonlinear_trajectory_e43_eps2 OLD (cdtmijl0) | **0.752** | ot_flow (argmax) | 2.0 | 4 | 有 | ❌ **未集成** | ot_flow coupling ε=2.0 |
| **nonlinear_trajectory_e43_eps2 NEW** (hkn0fc7w) | **0.741** | ot_flow (argmax) | 2.0 | 4 | 有 | ✅ **真正集成** | SCRF 导致 −0.005 |

历史归因修正时间线:

| 时间 | 旧认知 | 新认知 | 证据 |
|------|--------|--------|------|
| 2026-06-24 | ScaleConditionedRF (0.752) 归功于 ScaleConditionedRF + OT | 0.752 全部来自 OTFlowCoupling | 备份 head.py 确认 SCRF 未集成, 仅用于诊断回调 |
| 2026-06-27 | ScaleConditionedRF eps2 (0.752) 进一步验证 SCRF 有效 | 同上, SCRF 仍未集成 | 备份 head.py 确认 |
| 2026-07-02 | — | SCRF 真正集成后 0.741 < 0.746, **证伪** | TDD 集成 + 真正训练实验 |

### 教训

1. **配置字段存在 ≠ 代码生效**: dumped config 中含 `scale_conditioned_rf` 字段, 但 head.py 未集成, ScaleConditionedRF 完全未生效。后续所有"声称有效"的方向必须经代码级核查 (备份 head.py / commit diff)。
2. **TDD 红绿重构的价值**: 48 单元测试强制覆盖 head.py 4 处集成点, 避免"以为集成实际未集成"的盲点。
3. **OT coupling 是唯一有效改进**: 但提升幅度有限 (+0.005~0.007), 架构天花板约 0.75。
4. **代码保留**: SCRF 代码和测试保留在仓库中 (commit e757b856), 作为证伪记录供后续研究参考。

---

## 二、FBM 生成式迁移 (FBM variants, 证伪)

### 核心设想

用 ChromoGen 生成模型 (UNet) 的特征通过 Feature Bridge Module (FBM) 增强 LDMDet 检测器, 期望生成模型在染色体数据上学到的形态先验能提升检测精度。

### 证伪证据

4 个配置均未超越 baseline 0.746, 简单 frozen (FBM Frozen) 优于复杂增强 (FBM Enhanced/FBM Cross-Attention)。

#### 实验证明目的

验证不同 FBM 架构 (frozen / enhanced / crossattn) 是否能超越 baseline 0.746。

- FBM Frozen (FBM alpha 可学习 + UNet 全冻结)
  -- config: experiments/configs/ldmdet/gen_transfer_phase1_e6_2_frozen.py
  -- work_dir: work_dirs/gen_transfer_phase1_e6_2_frozen/
  -- SwanLab: ldmdet-ablation, run_id=qyzudrgb, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/qyzudrgb
  -- best mAP=0.7370 @ epoch 66 (Δ=−0.009)
  -- 最简单的 frozen 配置反而最优, 但仍低于 baseline

- FBM Enhanced (per-channel gate + GroupNorm + UNet 部分解冻)
  -- config: experiments/configs/ldmdet/gen_transfer_phase1_e6_3_enhanced.py
  -- work_dir: work_dirs/gen_transfer_phase1_e6_3_enhanced/
  -- SwanLab: ldmdet-ablation, run_id=m2fyzmf9, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/m2fyzmf9
  -- best mAP=0.7030 @ epoch 19 (Δ=−0.043)
  -- UNet 部分解冻后性能反而大幅下降

- FBM Frozen-Enhanced (同 FBM Enhanced 但 UNet 全冻结, 隔离 FBM 架构效果)
  -- config: experiments/configs/ldmdet/gen_transfer_phase1_e6_3b_frozen_enhanced.py
  -- work_dir: work_dirs/gen_transfer_phase1_e6_3b_frozen_enhanced/
  -- SwanLab: ldmdet-ablation, run_id=bbe2yrcg, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/bbe2yrcg
  -- best mAP=0.6960 @ epoch 16 (Δ=−0.050)
  -- 隔离实验确认 enhanced FBM 架构本身就有害

- FBM Cross-Attention (Cross-Attention FBM + UNet 部分解冻 + zero-init gamma)
  -- config: experiments/configs/ldmdet/gen_transfer_phase1_e6_4_crossattn.py
  -- work_dir: work_dirs/gen_transfer_phase1_e6_4_crossattn/
  -- SwanLab: ldmdet-ablation, run_id=x8j5l7mw, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/x8j5l7mw
  -- best mAP=0.7330 @ epoch 35 (停止 @ epoch 48, Δ=−0.013)
  -- ⚠ gamma 零初始化导致梯度信号微弱, cross-attention 路径未激活, 退化为 simple gate

### 失败原因分析

1. **FBM Cross-Attention gamma 零初始化致 attention 路径未激活**: zero-init gamma 使 cross-attention 输出在初始阶段被完全抑制, 梯度信号微弱, 路径无法被训练, 实际退化为 simple gate (与 FBM Frozen 等价)。后续改进建议: 非零 gamma 初始化 (如 0.1) 或移除 gamma。
2. **FBM Enhanced UNet 部分解冻破坏生成模型先验**: UNet 在检测损失下微调, 偏离了原始生成模型学到的形态先验, 注入的特征反而引入噪声。
3. **FBM 架构复杂度反比于性能**: FBM Frozen (最简单) > FBM Cross-Attention > FBM Enhanced > FBM Frozen-Enhanced, 表明 FBM 的价值不在于架构复杂度, 而在于生成模型特征本身与检测任务的兼容性不足。
4. **生成模型与检测器的特征空间不匹配**: ChromoGen UNet 学到的特征服务于像素级生成, 而 LDMDet 需要的是 bbox 级结构化特征, 两者特征空间不兼容。

---

## 三、Flow Matching Detection (CFM 替代 RF, 证伪)

### 核心设想

用 Conditional Flow Matching (CFM) 替代 Rectified Flow (RF) 作为训练范式, 期望 CFM 的 mini-batch OT 耦合能进一步提升精度 (参考 FlowDet)。

### 证伪证据

#### 实验证明目的

验证 CFM 替代 RF 是否能超越 Dataset 2 baseline +Stoch. Coupling (0.859)。

- Flow Matching Detection (CFM 训练)
  -- 数据集: Dataset 2
  -- SwanLab: ldmdet-frontier-directions
  -- mAP=0.823 (Δ=−0.033 vs +Stoch. Coupling baseline 0.859, Dataset 2)
  -- 对应论文 Appendix B "Flow Matching Detection" 方向

### 失败原因分析

1. **CFM 未能提供 RF 的直线轨迹优势**: RF 的核心优势是直线 ODE 路径使少步推理截断误差低, CFM 的训练目标不强制直线轨迹。
2. **mini-batch OT 在低维检测空间中诱发 OT Diversity Collapse**: 论文 §3.3 形式化分析 (命题 1-2) 显示 d=4, K≈46 时 ΔH/H≈0.69, OT 耦合多样性坍缩至零, 损害训练。CFM 默认使用 mini-batch OT, 触发该病理。
3. **修正了 FlowDet 的结论**: FlowDet 报告"高阶 solver 表现更差"但未分析原因, 本实验通过 solver×step 解耦 (论文 §4.5.1) 证明在匹配步数下高阶 solver 略优, FlowDet 的结论源于 CFM 训练范式本身的缺陷而非 solver。

---

## 四、Cascade Head Count E2E (重训 cascade head 数量, 证伪)

### 核心设想

重训架构, 把 cascade head 数量从 6 改为其他值 (减小 H), 期望减少计算量同时保持精度。基于 S1 理论分析 (theory_analysis_RF_DPM.md §2), cascade head 与 solver step 构成算子分裂, H 与 S 应可交换。

### 证伪证据

#### 实验证明目的

验证减小 cascade head 数量 H (从 6 减为更小值) 是否能在保持精度的情况下减少计算量。

- Cascade Head Count E2E (减小 H, 端到端重训)
  -- 数据集: Dataset 2
  -- SwanLab: ldmdet-frontier-directions
  -- mAP=0.684 (Δ=−0.172 vs +Stoch. Coupling baseline 0.859, Dataset 2, ⛔ 证伪, 性能坍塌)
  -- 对应论文 Appendix B "Cascade Head Count E2E" 方向

### 失败原因分析

1. **减小 H 破坏横向收敛性**: S1 命题 S1.1 (theory_analysis_RF_DPM.md §2.4) 指出, cascade head 序列在固定 t 上是压缩映射时存在不动点 x_t*, H 充分大时才近似收敛。减小 H 使横向精化不充分, x_t 偏离不动点。
2. **S 未相应增加**: S1 命题 S1.3 指出 H 与 S 的可交换性边界——减小 H 需增大 S 补偿, 但 H×S 不是不变量。因 A_t (solver) 是二阶而 B_{t,k} (cascade head) 是一阶精化, H 减半需 S 增加多于两倍。本实验未增加 S, 横向和纵向都不足。
3. **DPM-Solver++ 4 NFE 框架失效**: S1 推论 S1.2 指出, DPM-Solver++ 把每个 time step 内的 H 个 cascade head 视为单次"复合 v_θ 评估" (B_t* ∘ A_t) 的前提是 H=6 足够大。减小 H 使该假设失效, DPM-Solver++ 框架不再适用。
4. **理论预测与实验一致**: S1 理论分析 (theory_analysis_RF_DPM.md §2.5) 明确预测 "减小 H 破坏横向收敛性, 而 S 未相应增加", 本实验证实该预测, mAP 坍塌 −0.172。

---

## 五、Inference-Time Optimization directions 推理时优化 (证伪)

### 核心设想

在 +Stoch. Coupling checkpoint (DPM-Solver++ 4 步 + Top-K 剪枝) 上不重训, 仅推理时优化以减少 NFE / 加速推理。5 个方向 (Adaptive Step, Draft-Verify, Head Early-Exit, RoI Feature Cache; Top-K 未实验) 均证伪。

### 证伪证据

#### 实验证明目的

验证各类推理时优化 (adaptive step / draft-verify / head early-exit / RoI cache) 是否能减少 NFE 同时保持精度。

所有实验均在 +Stoch. Coupling checkpoint 上推理时评估, 无重训。

- Adaptive Step (根据 x0_pred 收敛提前终止)
  -- config: experiments/configs/ldmdet/directions/inference_opt/io1.py
  -- 数据集: Dataset 2
  -- 证伪: x0 相对 Δ 最小 0.166, 无法提前终止
  -- 失败原因: 早期步骤 x0_pred 不稳定 (与 ScaleConditionedRF 缺陷 1 同源), 收敛判据不可靠

- Draft-Verify (早期步骤 draft, 后期 verify)
  -- config: experiments/configs/ldmdet/directions/inference_opt/io2.py
  -- 数据集: Dataset 2
  -- 证伪: 早期步骤 cls 一致率仅 57%, 无法区分 draft 与 verify
  -- 失败原因: 早期 t≈1 时模型输入近乎纯噪声, 类别预测不可靠, draft-verify 无法建立信任阈值

- Head Early-Exit (cascade head 提前退出)
  -- config: experiments/configs/ldmdet/directions/inference_opt/io4.py
  -- 数据集: Dataset 2
  -- 证伪: 所有阈值下退出率 0%
  -- 失败原因: 6 个 cascade head 顺序精化, 每个 head 都对最终精度有贡献 (S1 命题 S1.1 的横向收敛性), 无法提前退出

- RoI Feature Cache (缓存 RoI 特征避免重复计算)
  -- config: experiments/configs/ldmdet/directions/inference_opt/io5.py
  -- 数据集: Dataset 2
  -- 证伪: box 位移 93-124 px/步, 缓存失效
  -- 失败原因: 每步 box renewal + solver 推进使 proposal 位置大幅变化 (93-124 px/步), RoIAlign 区域完全不重叠, 缓存命中率接近 0

### 失败原因分析

1. **推理时优化的根本限制**: 所有推理时优化方向都依赖"早期步骤信息足以决策", 但 RF 在 t≈1 时模型输入近乎纯噪声, 任何基于 x0_pred 的决策都不可靠。
2. **与 R1 直线度指标的区分**: R1 (theory_analysis_RF_DPM.md §1.6) 仅观测 η_str 不做决策, 避开了 Adaptive Step 的失败模式; Adaptive Step 依赖 x0_pred 做决策, 触发失败。
3. **cascade head 不可精简**: S1 理论分析 (theory_analysis_RF_DPM.md §2.5) 预测 H 与 S 不可简单交换, Head Early-Exit 实验证实每个 head 都必要。
4. **box renewal 与 cache 互斥**: box renewal 是 LDMDet 的核心机制 (低置信度 proposal 重置为噪声), 但它使 box 位置每步大幅变化, 与任何基于位置稳定性的缓存策略 (RoI Feature Cache) 互斥。

---

## 六、Decoupled Head (⚠ 数据更正: 证伪存疑)

### 核心设想

解耦分类与回归路径, 减少多任务干扰。期望分类 head 和回归 head 独立优化, 各自达到更优性能。

### 证伪证据

#### 实验证明目的

验证 DecoupledHead 是否能通过减少多任务干扰超越 baseline 0.746。

- Decoupled Head (替代共享特征路径)
  -- config: experiments/configs/ldmdet/direction_b_decoupled_head.py
  -- work_dir: work_dirs/direction_exps/direction_b_decoupled_head/20260629_152911/
  -- SwanLab: ldmdet-ablation, run_id=2ckzmso4c94fojr8jobej, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/2ckzmso4c94fojr8jobej
  -- best mAP=0.7490 @ epoch 74 (Δ=+0.003 vs baseline 0.746, 在 noise 范围内; 原记录 0.7020 @ ep25 为中间值, 已更正)
  -- 训练在 epoch 87 中断 (iter 150/385), 86 个 epoch 已完成评估, best 在 epoch 74
  -- ⚠ 证伪结论存疑: 真实 best 略超 baseline, 非"显著退化"; 单 seed + 训练中断, 需 3-seed 复核 (2026-07-26 subagent 三重证据确认: scalars.json + best checkpoint 文件名 + 训练日志)

⚠ **数据更正注记 (2026-07-26)**: 以下失败原因分析基于旧错误数据 (best=0.702, Δ=−0.044)。subagent 三重证据核实确认真实 best=0.749 @ ep74 (Δ=+0.003, 略超 baseline)。这些失败原因分析可能不再成立, 保留作为历史记录, 待 3-seed 复核后重新评估。

### 失败原因分析

1. **解耦导致特征共享信息丢失**: 分类和回归在共享特征路径下互相约束, 共同学习"这是什么 + 在哪里"的联合表示。解耦后两路径独立优化, 失去联合约束, 特征表示退化。
2. **小数据集下多任务正则化更重要**: Dataset 1 仅 1540 张训练图, 共享路径的多任务损失提供天然正则化, 解耦后模型过拟合风险增加。
3. **与 Cascade R-CNN 设计哲学冲突**: Cascade R-CNN 的 cascade head 共享特征路径正是利用多任务联合学习, Decoupled Head 的解耦设计与该哲学冲突。

---

## 七、Morphology-Contrastive / Structured Prior / Deformable P1 / Class-Balanced (未启动废弃)

### 核心设想

四个方向在 Architecture Decoupling series 设计阶段被规划, 但因 Decoupled Head 数据更正 (证伪存疑, 真实 best 0.749 略超 baseline) + Box Refine Net 持平 baseline, 整个 Architecture Decoupling series 停止推进, 这四个方向未启动即废弃。

### 废弃状态

#### 实验证明目的

(无, 未启动)

- Morphology-Contrastive Head
  -- 核心设想: 形态学先验 + 对比学习增强 C 组染色体判别
  -- 状态: ⛔ 已废弃 (未启动), 无实验数据
  -- 废弃原因: Decoupled Head 证伪后 Architecture Decoupling series 停止推进

- Structured Prior Head
  -- 核心设想: 结构化先验 + 100 proposals (减少 proposal 数量)
  -- 状态: ⛔ 已废弃 (未启动), 无实验数据
  -- 废弃原因: 同上; 且后续 bottleneck 消融 (proposals_100) 证实 100 proposals 会导致 SIGKILL (见 §十)

- Deformable P1 Head
  -- 核心设想: P1 阶段 + Deformable Conv 增强几何建模
  -- 状态: ⛔ 已废弃 (未启动), 无实验数据
  -- 废弃原因: 同上

- Class-Balanced Sampling
  -- 核心设想: 类别平衡采样缓解 Y 染色体数据稀缺
  -- 状态: ⛔ 已废弃 (未启动), 无实验数据
  -- 废弃原因: 同上; 且后续 bottleneck 消融 (class_balanced_sampling) 证实该方向会导致 SIGKILL (见 §十)

### 失败原因分析

无 (未启动)。废弃决策基于:
1. Decoupled Head 数据更正 (原 Δ=−0.044 基于错误数据, 真实 best 0.749 @ ep74, Δ=+0.003 在 noise 内), 证伪存疑 (单 seed + 训练中断, 待 3-seed 复核); 但 Box Refine Net 持平 + Cascade Head Count E2E 严重退化仍表明架构解耦方向边际收益不足
2. Box Refine Net 持平 baseline (Δ=+0.001), 表明架构增强方向边际收益不足
3. 整个 Architecture Decoupling series 让位于非线性轨迹实验 (ScaleConditionedRF, FBM variants, E7) 和后续 Dataset 2 主路线消融 (DDPM baseline → RF+Heun → +AdaLN-Zero → +Stoch. Coupling → +DPM-Solver++)

---

## 八、h_velocity_loss (失败)

### 核心设想

velocity loss 变体, 探索 RF 训练目标的替代参数化 (与 R3 理论分析 theory_analysis_RF_DPM.md §4 相关, 但 R3 是 x0 vs v-prediction 的等价性分析, h_velocity_loss 是实验性变体)。

### 失败状态

#### 实验证明目的

验证 h_velocity_loss 变体是否能提升 Dataset 2 baseline。

- h_velocity_loss (velocity loss 变体)
  -- 数据集: Dataset 2
  -- SwanLab: ldmdet-frontier-directions
  -- 状态: ⛔ CRASHED (训练崩溃), 无有效 mAP
  -- 无 best checkpoint

### 失败原因分析

1. **训练崩溃**: 具体崩溃原因未记录 (日志已清理), 但 v-prediction 在 t→0 时的 1/t² 梯度放大 (R3 命题 R3.2, theory_analysis_RF_DPM.md §4.3) 可能是数值不稳定的根源。
2. **与 R3 理论一致**: R3 分析指出, 在 shifted schedule (s=3.0) 下 v-prediction 在 t<0.5 时 1/t²>4 加剧方差, 低维 d=4 设置下该放大效应更显著。h_velocity_loss 的崩溃与该理论预测一致。
3. **不推荐重试**: R3 理论分析 (theory_analysis_RF_DPM.md §5.3) 明确指出 "x0 vs v-prediction 重训: 不推荐 (重训成本高, 且理论分析已预防审稿人质疑)"。

---

## 九、FBM SimpleGate (失败)

### 核心设想

FBM simple gate 源预训练, 期望在 Dataset 2 源预训练阶段验证 simple gate FBM 架构的可行性, 为后续 few-shot 跨数据集微调提供基础。

### 失败状态

#### 实验证明目的

验证 FBM SimpleGate 在 Dataset 2 源预训练阶段是否能达到与 LDMDet SOTA 相当的精度。

- FBM SimpleGate (源预训练)
  -- 数据集: Dataset 2 (源预训练)
  -- config: experiments/configs/few_shot/source_pretrain/ldmdet_fbm_simplgate_24obj.py
  -- work_dir: work_dirs/few_shot/source_pretrain_ldmdet_fbm_simplgate_24obj/20260701_203410/
  -- SwanLab: few-shot-benchmark, run_id=hyuiam5m, URL: https://swanlab.cn/@einspanner/few-shot-benchmark/runs/hyuiam5m
  -- best mAP=0.6770 @ epoch 5 (epoch 5 即停止, ⛔ 已停止)
  -- 对比: LDMDet SOTA 源预训练 best @ epoch 26 (mAP 未公开但远高于 0.677)

### 失败原因分析

1. **训练早期即停滞**: epoch 5 达到 best mAP=0.677 后停止, 表明模型在早期就达到性能上限, 后续训练无收益。
2. **simple gate 表达能力不足**: simple gate (alpha 标量加权) 无法有效融合 ChromoGen UNet 特征与 LDMDet 检测特征, 注入的生成模型特征反而引入噪声。
3. **与 FBM Frozen 一致**: FBM Frozen (FBM alpha 可学习 + UNet 全冻结) 在 Dataset 1 上 mAP=0.737 (Δ=−0.009), 同样是 simple gate 架构, 同样未超越 baseline, 证实 simple gate FBM 架构本身的有效性不足。
4. **CrossAttn 变体仍需观察**: FBM CrossAttn 源预训练 (run_id=9hj8pe4a) 在 epoch 10 达到 0.810, 持续上升中, 但尚未完成, 是 FBM 方向的最后希望。

---

## 十、Bottleneck 失败系列

### 核心设想

基于 SOTA 模型 (rf_heun_adaln + Sinkhorn Stochastic Coupling, 0.746) 的瓶颈分析, 针对分类损失、回归损失、采样策略等瓶颈点设计消融, 期望找到能超越 0.746 的配置。

### 证伪证据

7 个配置均未超越 baseline 0.746, 其中 2 个训练失败 (SIGKILL)。

#### 实验证明目的

验证各类瓶颈消融 (损失权重 / 采样策略 / proposal 数量) 是否能超越 baseline 0.746。

- scale_aware_loss (尺度感知损失)
  -- config: experiments/configs/bottleneck/scale_aware_loss.py
  -- work_dir: work_dirs/bottleneck/ablation/scale_aware_loss/20260624_112630/
  -- SwanLab: ldmdet-ablation, run_id=o8elke3rcmln1i47fznr5, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/o8elke3rcmln1i47fznr5
  -- best mAP=0.742 (Δ=−0.004)

- relative_l1_loss (相对 L1 损失)
  -- config: experiments/configs/bottleneck/relative_l1_loss.py
  -- work_dir: work_dirs/bottleneck/ablation/relative_l1_loss/20260626_093627/
  -- SwanLab: ldmdet-ablation, run_id=cmxgctni1n1d9g9go0wcf, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/cmxgctni1n1d9g9go0wcf
  -- best mAP=0.740 (Δ=−0.006)

- high_cls_weight (高分类损失权重)
  -- config: experiments/configs/bottleneck/high_cls_weight.py
  -- work_dir: work_dirs/bottleneck/ablation/high_cls_weight/20260628_180423/
  -- SwanLab: ldmdet-ablation, run_id=m9dnqrl43khkl3jcbjico, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/m9dnqrl43khkl3jcbjico
  -- best mAP=0.739 (Δ=−0.007)

- high_giou_weight (高 GIoU 损失权重)
  -- config: experiments/configs/bottleneck/high_giou_weight.py
  -- work_dir: work_dirs/bottleneck/ablation/high_giou_weight/20260625_230646/
  -- SwanLab: ldmdet-ablation, run_id=b3w4gwly842j700yqg9og, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/b3w4gwly842j700yqg9og
  -- best mAP=0.737 (Δ=−0.009)

- no_box_renewal (消融 box renewal 机制)
  -- config: experiments/configs/bottleneck/no_box_renewal.py
  -- work_dir: work_dirs/bottleneck/ablation/no_box_renewal/20260623_211757/
  -- SwanLab: ldmdet-ablation, run_id=52o1g5pq09eddplyzbl9q, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/52o1g5pq09eddplyzbl9q
  -- best mAP=0.730 (Δ=−0.016, 消融)
  -- ⚠ **结论修正 (2026-07-30)**: box_renewal 是原版 DiffusionDet 的**推理时**机制 (projects/DiffusionDet/diffusiondet/head.py), **不在 loss() 路径** (ldmdet/core/head.py:628 loss() vs :1186 predict())。训练时 box_renewal=True/False **不影响模型权重** (梯度完全由 loss() 决定), 只影响验证评估时 predict() 的 proposal 回收。
  -- **-0.016 退化根因: early stopping 选择偏差**, 非 model 能力退化。box_renewal=False 时验证评估无 proposal 回收 → val mAP 波动更大 → early stopping 选择了次优 checkpoint (best@ep37, 仅训练 40 epoch)。反证: 推理切换实验 (Dataset 1 A4 K=500) 表明同一 baseline checkpoint 在 renewal OFF 下 mAP=0.743 (vs renewal ON 0.744, Δ=−0.001), 若 no_box_renewal 选择了同等质量 checkpoint, renewal OFF 应达 ~0.743 而非 0.730。
  -- **可完全移除 (推荐配置 K≥200)**: 推理时关闭 renewal 已 3-seed 验证安全 (ΔmAP=−0.0003), DPM++ D1 校正自然连续 (无 renewal 噪声干扰), 不需要 D3 化解路径 A 的 per-proposal D1 掩码。

- class_balanced_sampling (类别平衡采样) ⛔ FAILED
  -- config: experiments/configs/bottleneck/class_balanced_sampling.py
  -- work_dir: work_dirs/bottleneck/ablation/class_balanced_sampling/20260629_013833/ (无 scalars.json)
  -- SwanLab: ldmdet-ablation, run_id=vsq3xd3bw51iss1n1at15, URL: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/vsq3xd3bw51iss1n1at15
  -- 状态: SIGKILL (内存溢出或超时), 无有效 mAP
  -- 对应 Class-Balanced Sampling 的实验性验证, 证实该方向不可行

- proposals_100 (100 proposals 替代 500) ⛔ FAILED
  -- config: experiments/configs/bottleneck/proposals_100.py
  -- work_dir: work_dirs/bottleneck/ablation/proposals_100/20260624_021155/ (无 scalars.json)
  -- SwanLab: ldmdet-ablation (无 run_id)
  -- 状态: SIGKILL, 无有效 mAP
  -- 对应 Structured Prior Head 的实验性验证, 证实该方向不可行

### 失败原因分析

1. **损失权重调整方向错误**: high_cls_weight (0.739) 和 high_giou_weight (0.737) 均低于 baseline, 表明默认权重 (λ_cls=2.0, λ_giou=2.0) 已接近最优, 增大权重反而过拟合。
2. **尺度感知损失设计不足**: scale_aware_loss (0.742) 试图引入尺度先验, 但与 ScaleConditionedRF (§一) 类似, 推理时尺度估计不可靠, 损失设计未解决该问题。
3. **box renewal 可完全移除 (K≥200)**: box_renewal 是原版 DiffusionDet 的**推理时**机制, 不在 loss() 路径, 训练时 True/False 不影响模型权重。no_box_renewal "训练消融" (0.730, Δ=−0.016) 的退化根因是 **early stopping 选择偏差** (验证评估无 renewal → val mAP 波动 → 选择次优 checkpoint), 非 model 能力退化。推理时切换实验 (EXPERIMENT_LINEAGE §六 方案 B, DPM-Solver++, 3-seed) 表明: **推理时关闭 renewal 在 K≥200 下不损失精度** (ΔmAP=−0.0003), DPM++ D1 校正自然连续, 不需要 D3 化解路径 A; 仅 K=100 (非推荐配置) 下有 −0.031±0.012 退化 (proposal 稀缺时回收机制价值凸显)。
4. **类别平衡采样内存溢出**: class_balanced_sampling 触发 SIGKILL, 可能是加权采样器在小批量 (bs=2) 下内存占用过高。Class-Balanced Sampling 的设想虽合理, 但实现层面不可行。
5. **100 proposals 容量不足**: proposals_100 触发 SIGKILL, 可能是 proposal 数量减少后某些维度不匹配。即使能训练, Dataset 2 每张图约 46 个目标, 100 proposals 覆盖 46 + 重叠冗余时容量紧张 (与 Top-K K=100 掉点 −0.010 同源)。
6. **架构天花板约 0.75**: 7 个消融均未超越 0.746, 证实 Dataset 1 上架构天花板约 0.75, 需换数据集 (Dataset 2) 或换范式 (RF + DPM-Solver++) 才能突破。

---

## 十一、早期失败/调试实验

### 核心设想

早期探索阶段的简化/调试实验, 用于验证训练流程或测试新架构, 多数因简化配置 (无 aug) 或架构不匹配而失败。

### 失败证据

#### 实验证明目的

(各类早期验证, 多为调试性质)

- multi_seed/rf_heun_adaln (无 aug, 简化非标准)
  -- SwanLab: chromosome-kd (同名冲突, 已废弃)
  -- mAP=0.712 (简化非标准, 不可作为 baseline)
  -- 失败原因: 简化 aug (无 multi-scale/crop), 不是 DiffusionDet 默认 aug, 与 0.746 不可对比

- multi_seed/rf_heun_adaln_bs4 (无 aug, batch_size=4)
  -- SwanLab: 无 (本地实验)
  -- mAP=0.702
  -- 失败原因: 同上, 简化 aug + bs=4 调整未带来收益

- multi_seed/hard_ot (无 aug, Hard OT coupling)
  -- SwanLab: chromosome-kd, run_id=hard_ot_seed789
  -- mAP=0.705
  -- 失败原因: 简化 aug + Hard OT 触发 OT Diversity Collapse (论文 §3.3 命题 1-2)

- debug_train_final (调试实验)
  -- work_dir: work_dirs/debug_train_final/
  -- mAP=0.525
  -- 失败原因: 调试用, 非完整训练

- stability/warm_restart_* (warm restart 稳定性实验)
  -- SwanLab: chromosome-kd, bs8-warm-restart-*
  -- mAP=0.714-0.728
  -- 失败原因: warm restart 未带来稳定收益, 性能低于 0.746 baseline

- scheme_a_dinov2_s (DINOv2-S 主干, 全失败)
  -- work_dir: work_dirs/scheme_a_dinov2_s/
  -- SwanLab: chromosome-kd, dinov2_s-rf-heun-adaln-stochot
  -- mAP=0.502-0.676
  -- 失败原因: DINOv2-S 主干与 LDMDet 架构不匹配, 预训练特征与检测任务空间不兼容 (与 FBM §二 同源)

- scheme_E_bifpn (BiFPN 颈部)
  -- work_dir: work_dirs/scheme_E_bifpn/
  -- SwanLab: chromosome-kd, arch_E_bifpn
  -- mAP=0.744 (Δ=−0.002, 持平 baseline)
  -- 失败原因: BiFPN 未带来显著提升, 且增加计算量, 未采用

- cspnext_l_rf_heun_adaln_stochot (CSPNeXt-L 主干)
  -- work_dir: work_dirs/cspnext_l_rf_heun_adaln_stochot/
  -- SwanLab: chromosome-kd, cspnext-l-rf-heun-adaln-stochot
  -- mAP=0.730 (Δ=−0.016)
  -- 失败原因: CSPNeXt-L 主干在 Dataset 1 上表现不如 ResNet-50, 可能是预训练权重与染色体数据不兼容

### 失败原因分析

1. **简化 aug 不可作为 baseline**: multi_seed/* 系列 (0.702-0.712) 使用简化 aug (无 multi-scale/crop), 与 DiffusionDet 默认 aug 不一致, 不可作为 baseline。统计 73 个实验显示 56/73 使用 DiffusionDet 默认 aug, multi_seed_aug/* (0.746) 才是标准 baseline。
2. **DINOv2 主干不兼容**: scheme_a_dinov2_s 全失败 (0.502-0.676), DINOv2 预训练特征服务于图像级任务, 与 bbox 级检测任务特征空间不兼容, 与 FBM §二 失败原因同源。
3. **BiFPN 边际**: scheme_E_bifpn (0.744) 持平 baseline, BiFPN 的多尺度融合在染色体数据上未带来收益, 且增加计算量。
4. **CSPNeXt-L 不适配小数据**: cspnext_l (0.730) 低于 baseline, CSPNeXt-L 主干参数量更大, 在 Dataset 1 (1540 张) 上过拟合。
5. **warm restart 无效**: stability/warm_restart_* (0.714-0.728) 未带来稳定收益, RF 训练的 CosineAnnealing 已足够, warm restart 反而打断收敛。
6. **调试实验无参考价值**: debug_train_final (0.525) 等调试实验仅用于验证训练流程, 无完整训练, 性能无意义。

---

## 十三、Head Distillation 失败配置 (配置Bug导致失败, 方法本身有效)

> **失败性质**: ⛔ **训练配置问题** (非理论问题, 非方法局限)
> **方法有效性**: ✅ 修复配置验证有效 (mAP=0.860 持平 A4 0.863, → [LINEAGE §十五](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md))
> **归档日期**: 2026-07-27

### 实验配置

| 参数 | 失败配置 | 修复配置 (成功) |
|------|-----------|---------------|
| `freeze_backbone` | **True** (Bug根因) | False |
| backbone 初始化 | ImageNet 预训练 (未加载 A4) | **从 A4 checkpoint 加载** |
| lr | 5e-5 | 1e-5 |
| max_epoch | 150 | 50 |
| distill_lambda | 0.05 | 0.05 |
| distill_head_map | {0:0, 1:2, 2:5} | 同上 |
| teacher_checkpoint | A4 best ep117 | 同上 |

### 失败根因 (配置Bug)

`freeze_backbone=True` 导致 Student backbone 参数冻结, 停在 ImageNet 预训练权重 (未加载 A4 染色体训练权重)。因果链:

```
freeze_backbone=True
  → Student backbone 停在 ImageNet 特征分布
  → Teacher head 权重来自 A4 (在染色体特征上学习), 期望接收 A4 风格特征
  → Teacher forward 接收 Student 的 ImageNet 特征 → OOD 输入 → Teacher fc_feature 无意义
  → 蒸馏目标无效 → loss_distill 停滞在 0.033 不下降
  → mAP 停滞在 0.717 (Δ=-0.146 vs A4 0.863)
```

### 训练数据证据

| 指标 | 失败配置 (ep109 best) | 修复配置 (ep10 best) | 说明 |
|------|----------------|-------------------|------|
| best mAP | 0.717 | 0.860 | 修复配置 持平 A4 0.863 |
| ep1 mAP | 0.000 | 0.798 | 修复配置 的 A4 backbone 加载成功 |
| loss_distill (最终) | 0.0326 | 0.0250 | 修复配置 低 23.3%, 蒸馏目标有效 |
| loss_distill 趋势 | ep80 后停滞 | 持续下降 50% | 失败配置蒸馏目标不可学 |
| per-class AP | 全部劣化 (Δ -0.056~-0.289) | 与 A4 对齐 (Δ -0.012~+0.004) | 失败配置全面崩坏 |

### 方法学判定

- **失败配置是配置Bug, 不是理论问题**: 代码实现正确 (Teacher 构造/冻结/MSE 蒸馏/head 映射/梯度流全部正确, Subagent 3 代码审查确认)
- **方法本身有效**: 修复配置 mAP=0.860 持平 A4 0.863 (Δ=-0.003 在 3-seed noise ±0.003 内)
- **理论目标达成**: NFE 24→12 加速 2x + 精度持平
- **未超越 A4**: 修复配置 仅持平, 无增益 (但"持平"可能已是蒸馏最佳结果, 因 backbone 从 A4 加载本身就是知识继承)

### 复活因素 (已实现)

修复配置 已是修复后的结果, 无需重新训练。若要验证"蒸馏 > 直接训练"需补 H=3 from-scratch baseline 对照。

### 教训

1. **freeze_backbone 与 teacher_checkpoint 一致性检查**: 冻结 backbone 时必须确保 backbone 权重与 Teacher 期望的特征分布一致
2. **蒸馏目标有效性验证**: loss_distill 不下降是蒸馏目标无效的直接信号, 应在 ep10 内检查
3. **修复配置 对照价值**: 通过修复配置的对照实验, 清晰隔离了"配置Bug" vs "方法局限"

---

## 十四、ReFlow Standard MSE (方法本质失败, 重试确认)

> **失败性质**: ⛔ **方法本质失败** (重试确认: 配置 Bug 已修复, 但方法固有风险依然存在, 30 epoch 零改善)
> **方法学判定**: 方法本质问题 (cls/box 不一致 + Circular dependency + mAP_75 退化, 配置修复后仍失败)
> **归档日期**: 2026-07-27 (v1 失败归档) / 2026-07-28 (重试确认归档)

### 实验配置

| 参数 | 当前run (失败) | 重试配置 (计划) |
|------|---------------|----------------|
| `load_from` | **None** (Bug根因) | **A4 best ep117** |
| lr | 1e-5 | **5e-5** (对齐 baseline) |
| max_epoch | 50 | **150** (对齐 baseline) |
| box_target_mode | 'x0_pred' | 同上 |
| use_reflow_coupling | True | 同上 |
| reflow_coupling_path | train_couplings.pt | 同上 |

### 当前run失败原因

#### 主因: 3个配置handicaps (训练配置问题)

1. **缺失 `load_from`**: 模型从零训练 (应从 A4 checkpoint 初始化)
2. **lr=1e-5**: 比 baseline (5e-5) 低 5x
3. **max_epoch=50**: 仅为 baseline (150) 的 1/3

→ 严重欠训练 → best mAP=0.646@ep42 (非用户记忆中的 0.542, 0.542 是 ep50 末值已回退)

#### 次因: 4个方法固有风险 (理论问题)

| 风险 | 描述 | 性质 |
|------|------|------|
| **cls/box 不一致** | OT 重算 matched_idx (noise↔GT) 但 box_target=x0_pred (A4 预测) → cls 说"GT_j"但 box 说"A4 预测的另一个框" | 方法设计问题 |
| **Circular dependency** | 模型用 A4 自预测训练, 强化 A4 的系统误差 | 方法设计问题 |
| **box_renewal 训推不一致** | 生成关、eval 开 → proposal 分布偏移 | 方法设计问题 |
| **mAP_75 崩塌** | 模型学习预测 A4 的 x0_pred, 而 A4 对噪声 proposal 的预测本身模糊 → 模型学到模糊定位 | 方法本质问题 |

### 训练数据证据

| 指标 | 当前run (ep42 best) | A4 baseline | 说明 |
|------|-------------------|-------------|------|
| best mAP | 0.646 | 0.863 | Δ=-0.217 |
| ep50 mAP | 0.542 (已回退) | — | best 后持续下降 |
| mAP_50 @ ep42 | 0.959 | 0.988 | 粗定位接近 |
| mAP_75 @ ep42 | 0.733 | 0.974 | **精确定位崩塌** |
| mAP_75 @ ep50 | 0.543 | — | 定位精度持续退化 |
| mAP_50-mAP_75 gap | 0.226→0.392 | 0.014 | gap 扩大 73% |
| loss 趋势 | ep42 后饱和 | — | lr 已衰减至 0 |

### 方法学判定 (重试确认: 方法本质失败)

- **代码实现正确**: coupling 加载/q_sample 公式/空间转换/per-proposal 对齐全部正确 (Subagent 3 代码审查确认)
- **v1 失败不可挽救**: best 0.646 已回退, lr 已衰减至 0, mAP_75 崩塌
- **重试确认方法本质失败**: 配置 Bug 已修复 (load_from=A4 + lr=5e-5 + 150ep), 但 30 epoch 零改善, mAP_75 仍崩塌
- **重试判据命中**: mAP_75 在 ep2=0.687, ep19=0.608 两次跌破 0.70; best 始终在 ep1 (=A4 checkpoint 本身)

### 重试实验结果 (2026-07-27 16:03 → 22:01, EarlyStopping @ ep31)

> workstation `100.99.131.26`: work_dirs/reflow_standard_24obj/
> SwanLab run_id: bj8bmny5 (本地 scalars.json 完整, project 'ldmdet-reflow' 上传异常 404)
> 配置: load_from=A4 best ep117 + lr=5e-5 + max_epoch=150 + warmup 5ep + cosine

#### 重试配置 vs v1 失败配置

| 参数 | v1 (失败) | 重试 (确认失败) |
|------|-----------|----------------|
| `load_from` | **None** (Bug根因) | **A4 best ep117** ✅ 修复 |
| lr | 1e-5 | **5e-5** ✅ 修复 |
| max_epoch | 50 | **150** ✅ 修复 |
| EarlyStopping | 未配置 | patience=30 (ep31 触发) |

#### 逐 epoch mAP 关键数据

| Epoch | mAP | mAP_50 | mAP_75 | 备注 |
|-------|-----|--------|--------|------|
| **1** | **0.862** | 0.989 | **0.970** | **best** = A4 checkpoint 本身, 非 reflow 贡献 |
| 2 | 0.618 | 0.974 | 0.687 | ⚠ mAP 崩塌 -0.244, mAP_75 跌破 0.70 |
| 11 | 0.812 | 0.986 | 0.946 | 重试阶段最高 mAP (仍 < A4 0.863) |
| 19 | 0.586 | 0.937 | 0.608 | ⚠ mAP 最低, mAP_75 再次跌破 0.70 |
| 31 | 0.794 | 0.983 | 0.917 | EarlyStopping 触发 (30 ep 零改善) |

#### 重试统计

| 指标 | v1 (失败) | 重试 (确认失败) | A4 baseline |
|------|-----------|----------------|-------------|
| best mAP | 0.646 @ ep42 | **0.862 @ ep1** (= A4 本身) | 0.863 |
| 重试阶段最高 mAP (ep2-31) | — | 0.812 @ ep11 (Δ=-0.051) | — |
| 重试阶段最低 mAP (ep2-31) | — | 0.586 @ ep19 (Δ=-0.277) | — |
| mAP_75 跌破 0.70 次数 | 持续 | **2 次** (ep2, ep19) | 0 次 |
| mAP_75 跌破 0.80 次数 | 持续 | **8 次** | 0 次 |
| 训练停止原因 | lr 衰减至 0 | **EarlyStopping** (patience=30) | — |
| 配置 Bug | 3 个 (load_from/lr/epoch) | **全部修复** ✅ | — |

### 重试确认: 方法本质失败根因

配置 Bug 修复后, 4 个方法固有风险依然全部命中 (与 v1 相同):

1. **cls/box 不一致** (方法设计问题): OT 重算 matched_idx (noise↔GT) 但 box_target=x0_pred (A4 预测) → cls 说"GT_j"但 box 说"A4 预测的另一个框"
2. **Circular dependency** (方法设计问题): 模型用 A4 自预测训练, 强化 A4 的系统误差 → 30 epoch 零改善 (best 始终 ep1)
3. **box_renewal 训推不一致** (方法设计问题): 生成关、eval 开 → proposal 分布偏移
4. **mAP_75 退化** (方法本质问题): 模型学习预测 A4 的 x0_pred, 而 A4 对噪声 proposal 的预测本身模糊 → 模型学到模糊定位 (mAP_75 两次跌破 0.70)

**关键证据**: best=0.862 @ ep1 是 A4 checkpoint 加载后的初始状态 (load_from=A4), reflow coupling 训练 30 个 epoch **零改善** — 证明 reflow 不仅没有拉直轨迹提升性能, 反而持续损害 A4 已学到的表示 (ep2 起 mAP 崩塌至 0.618).

### 数据修正

- 用户记忆中的 "best mAP=0.542 @ ep50" 是错误的, 实际 best mAP=0.646 @ ep42 (0.542 是 ep50 末值, 已从峰值回退 0.104)
- "已证伪 ReFlow velocity loss 版 mAP=0.739" 标签错误, 0.739 来自 nonlinear_trajectory_seed2 (既无 velocity loss 也无 reflow coupling), 真正的 h_velocity_loss best=0.856@ep61

### 教训

1. **微调实验必须设置 load_from**: 从零训练 + 微调参数 (低 lr + 少 epoch) 是致命组合
2. **mAP_75 是定位精度的关键指标**: mAP_50 持平但 mAP_75 崩塌说明方法损害精细定位
3. **用户标准 "配置问题不能否定方法失败"**: 需从方法固有风险角度判定, 不能仅归咎于配置
4. **重试隔离了"配置 Bug" vs "方法本质问题"** (2026-07-28 确认): 配置 Bug 修复后 (load_from+A4+lr=5e-5+150ep), 方法本质问题依然存在 — best 0.862@ep1 = A4 本身, 30 epoch 零改善, mAP_75 仍崩塌 (ep2=0.687, ep19=0.608). EarlyStopping @ ep31 自动终止. **ReFlow 2-Rectification (Standard MSE) 方向正式判定方法本质失败**
5. **best@ep1 = checkpoint 本身是微调失败的强信号**: 若 best 始终在 ep1 且后续零改善, 说明训练目标 (reflow coupling x0_pred) 不仅无益反而有害, 应立即检查 cls/box 一致性

---

## 十五、VLR (Velocity Lipschitz Regularization) — 致命理论错误 (理论评审淘汰, 未实验)

### 核心设想

VLR 提出: 在训练损失中增加速度场雅可比 Frobenius 范数正则项 $\mathcal{L}_{\text{VLR}} = \lambda \cdot \mathbb{E}_t[\|J_{v_\theta}\|_F^2]$, 期望驱动 $v_\theta$ 趋向 $x_t$ 的常数函数 (即 $J_{v_\theta} \to 0$), 与 RF 的直线前向协同。理论根: 引理 2.1 (真实速度 $v^* = \varepsilon - x_0$ 是常数, 故 $J_{v^*} = 0$)、引理 2.2 (网络速度 $v_\theta = (f_\theta - x_t)/t$, 故 $J_{v_\theta} = (J_{f_\theta} - I)/t$)、Bartlett et al. (2017) 谱裕度界、Hutchinson 迹估计。

### 评审淘汰过程

**状态**: ⛔ 理论评审淘汰 (未真实实验, 仅经 R1 A↔B 评审)
**R1 评分**: 3.0/10 (Strong Reject, Reviewer B)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_VLR_FINAL.md` + `VLR_REVIEW_R1.md`)

### 理论缺陷分析

#### 缺陷 1 (致命): 正则化目标与理想 RF 直接矛盾

- **错误描述**: VLR 最小化 $\|J_{v_\theta}\|_F^2$, 驱动 $J_{v_\theta} \to 0$。但理想 RF 下 ($f_\theta \equiv x_0$, $J_{f_\theta} = 0$):
  $$J_{v_\theta} = \frac{J_{f_\theta} - I}{t} = \frac{0 - I}{t} = -\frac{I}{t} \neq 0$$
  $$\|J_{v_\theta}\|_F^2 = \left\|-\frac{I}{t}\right\|_F^2 = \frac{4}{t^2} \neq 0$$
- **错误依据**: 引理 2.1 证明的是真实速度 $J_{v^*} = 0$ ($v^* = \varepsilon - x_0$ 不依赖 $x_t$), 但 VLR 正则化的是网络速度 $J_{v_\theta}$ ($v_\theta = (f_\theta - x_t)/t$ **显式依赖 $x_t$**)。两者是不同对象。
- **为何不可修复**: 这是正则化目标本身的错误。理想 RF 的 $J_{v_\theta} = -I/t \neq 0$, VLR 强制 $J_{v_\theta} \to 0$ 等于偏离理想 RF。无法通过调参或截断解决。

#### 缺陷 2 (致命): 正则化方向与理想 RF 完全相反

- **错误描述**: 由 $J_{v_\theta} = (J_{f_\theta} - I)/t$, VLR 最小化 $\|J_{v_\theta}\|_F^2 = \|J_{f_\theta} - I\|_F^2 / t^2$ 驱动 $J_{f_\theta} \to I$。但理想 RF 要求 $J_{f_\theta} = \partial x_0 / \partial x_t = 0$ (因为 $x_0$ 不依赖 $x_t$)。
- **错误依据**: 定理 2.2 的推导。对比 ISLR (正则化 $\|J_{f_\theta}\|_F^2$ 驱动 $J_{f_\theta} \to 0$, 与理想 RF 一致) — VLR 与 ISLR 仅差一个 $-I/t$ 项, 但正则化方向完全相反。
- **为何不可修复**: VLR 驱动 $J_{f_\theta} \to I$ 意味着 $f_\theta$ 趋向 $x_t$ 的恒等函数加偏移 ($f_\theta = x_t + t \cdot g(t)$), 而非理想 RF 的 $f_\theta = x_0$ (常数)。方向错误无法修复。

#### 缺陷 3 (严重): §3.4 自相矛盾

- **错误描述**: 文档 §3.4 写道 "$\hat{x}_0 = x_0$ 与 $x_t$ 无关, 但数值上 $\partial \hat{x}_0 / \partial x_t \to I$"。
- **错误依据**: 若 $\hat{x}_0$ 与 $x_t$ 无关 (常数函数), 则导数在**所有** $x_t$ 处都为 0, 包括 $x_t \to x_0$ 时。"$\partial \hat{x}_0 / \partial x_t \to I$" 的说法混淆了 "$\hat{x}_0$ 趋近 $x_t$" (函数值) 和 "$\partial \hat{x}_0 / \partial x_t$ 趋近 $I$" (导数)。
- **为何不可修复**: 这是 VLR 论证链条的关键环节, 用于将 $J_{f_\theta} \to I$ 包装为 "理想 RF 的数值表现"。矛盾暴露后, VLR 的理论包装失效。

#### 缺陷 4 (严重): $1/t^2$ 因子与 h_velocity_loss (§八) 崩溃风险高度重叠

- **错误描述**: 定理 2.3 表明 VLR 损失含 $1/t^2$ 因子: $\mathcal{L}_{\text{VLR}} = \lambda \cdot \mathbb{E}_t[t^{-2} \|J_{f_\theta} - I\|_F^2]$。在 $t \to 0$ 时 $1/t^2 \to \infty$。
- **错误依据**: h_velocity_loss (FALSIFIED §八) 因 v-prediction 的 $1/t^2$ 梯度放大崩溃。VLR 的 $1/t^2$ 来源 (从 $v_\theta = (f_\theta - x_t)/t$ 的 $1/t$ 因子平方) 虽与 h_velocity_loss 不同, 但数值效果相似。文档使用 $t_{\text{eps}} = 0.1$ 截断, 但 $t \in [0.1, 0.25]$ 范围内 $1/t^2$ 仍有 16-100× 权重。
- **为何不可修复**: $1/t^2$ 是 $v_\theta = (f_\theta - x_t)/t$ 定义的固有结果, 无法消除 (除非不正则化 $v_\theta$, 但那就不是 VLR)。

### 与已证伪方向的关系

- **与 h_velocity_loss (§八) 高度重叠**: 两者都涉及 $1/t^2$ 数值不稳定风险。h_velocity_loss 因 v-prediction 的 $1/t^2$ 崩溃 (CRASHED), VLR 的 $1/t^2$ 来自速度场雅可比的 $1/t$ 因子平方, 机制不同但数值效果相似。
- **与 ScaleConditionedRF (§一) 概念相关**: VLR 使用 $t_{\text{eps}} = 0.1$ 截断 $t < 0.1$ 的样本, 导致训练分布与推理分布不一致, 与 SCRF 的 "训练-推理不一致" 失败模式 (§一 缺陷 1) 概念相似。

### 教训

1. **正则化目标必须与理想解一致**: VLR 的核心错误是将真实速度 $J_{v^*} = 0$ 错误迁移到网络速度 $J_{v_\theta}$, 但 $v_\theta$ 显式依赖 $x_t$, 理想值非零。设计正则化时必须验证 "理想解处的正则值" 是否为零。
2. **$1/t^2$ 在 RF 中是已知风险**: h_velocity_loss 已证伪, VLR 重蹈覆辙, 表明任何含 $1/t^2$ 因子的正则化都需严格论证数值稳定性。

---

## 十六、DSCR (Dahlquist-Stability Cascade Regularization) — 范畴错误 (理论评审淘汰, 未实验)

### 核心设想

DSCR 提出: 将 6 级 cascade head 视为离散动力系统 $x^{(k)} = g_k(x^{(k-1)})$, 计算复合 Jacobian $\mathcal{J}_{\text{cas}} = J_K \cdot J_{K-1} \cdots J_1$, 用 power iteration 估计谱范数 $\sigma_{\max}(\mathcal{J}_{\text{cas}})$, 惩罚超出稳定阈值 $1-\gamma = 0.9$ 的部分。理论根: Dahlquist 稳定性理论 (1956, 1963)、根条件、修正方程/后向误差分析 (Hairer & Lubich 2006)、刚性 ODE 理论。

### 评审淘汰过程

**状态**: ⛔ 理论评审淘汰 (未真实实验, 仅经 R1 A↔B 评审)
**R1 评分**: 4.0/10 (Reject, Reviewer B)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_DSCR.md` + `DSCR_REVIEW_R1.md`)

### 理论缺陷分析

#### 缺陷 1 (致命): Dahlquist 稳定性理论的范畴错误

- **错误描述**: Dahlquist 稳定性理论是关于**数值时间步进方法** (Euler, Runge-Kutta, 多步法) 求解 ODE 时的稳定性, 要求放大矩阵特征值 $\leq 1$。但 KaryoFlow 的 6 级 cascade head **不是时间步进方法**: cascade head 在固定时间 $t$ 内操作 (不推进时间), 每级 head 是学习的映射 $g_k$ (非 ODE 离散化), cascade head 之间无 "步长" $h$ 概念。
- **错误依据**: Dahlquist (1956, 1963) 的研究对象是 $y_{n+1} = \Phi(y_n, h)$ 应用于测试方程 $y' = \lambda y$ 时的放大因子。DPM-Solver++ 的 4 步时间步进才是 Dahlquist 意义上的数值方法, cascade head 是步内的学习精化。
- **为何不可修复**: 这是范畴错误 — 将神经网络的层间传播视为 ODE 数值方法。除非完全放弃 Dahlquist 理论包装, 改用深度学习 Jacobian 谱分析 (Pennington 2018, Xiao 2018), 但那已不是 DSCR。

#### 缺陷 2 (严重): 修正方程分析的假设不成立

- **错误描述**: 文档引用 Hairer & Lubich (2006) 修正方程理论 (定理 2.6), 要求 $g_k(x) = x + \epsilon f(x) + O(\epsilon^2)$, 其中 $\epsilon = 1/K$ 是 "单级步长"。但 $\epsilon = 1/6 \approx 0.167$ 不是小参数 (修正方程理论需 $\epsilon \ll 1$), 且 cascade head 不保证 near-identity 或辛映射 (symplectic)。
- **错误依据**: Hairer & Lubich Theorem 1 要求 $g_k$ 是辛映射或接近恒等的映射。学习的 cascade head 可以输出与输入完全不同的框, 不满足此性质。每级精化幅度 0.05-0.15 (归一化坐标) 在 cxcywh 空间可能更大。
- **为何不可修复**: $\epsilon = 1/6$ 是架构固定值 (H=6 cascade head), 无法减小; head 的学习性质不保证 near-identity。

#### 缺陷 3 (严重): 计算开销严重低估

- **错误描述**: 文档声称 "约 5% 训练开销", 实际估算:
  - 每级 head 的 Jacobian $J_k \in \mathbb{R}^{4 \times 4}$ 需要 4 次 VJP (每次 `torch.autograd.grad` with `create_graph=True`)
  - 6 级 cascade: $6 \times 4 = 24$ 次 VJP, 每次涉及完整反向传播
  - Power iteration 3-5 步 + 二阶自动微分 (create_graph=True, 内存约 3-4×)
  - Per-proposal: 若 $M=16$, 则 16 × 72 = 1152 次等效反传
- **错误依据**: 对比基线训练 (1 次前向 + 1 次反传), 开销可能 >500%, 远非 "5%"。
- **为何不可修复**: 复合 Jacobian $\mathcal{J}_{\text{cas}} = J_K \cdots J_1$ 的计算成本是结构性的, 无法通过优化消除。

#### 缺陷 4 (严重): cascade_detach=True 阻止跨级 Jacobian 计算

- **错误描述**: head.py 中 `cascade_detach=True` (line 48, 570-577), 每级 head 的输入被 detach: `curr_bboxes = pred_bboxes.detach()`。这意味着第 $k$ 级 head 的输入 $x^{(k-1)}$ 不保留对 $x^{(k-2)}$ 的梯度, 无法构建跨级 Jacobian 链。
- **错误依据**: DSCR 需要在 `cascade_detach=False` 模式下训练, 或单独构建非 detach 的前向通路。前者改变训练动态 (可能影响精度), 后者增加实现复杂度。
- **为何不可修复**: `cascade_detach=True` 是 KaryoFlow 的核心设计 (S1 理论分析要求), 改为 False 会破坏横向收敛性。

### 与已证伪方向的关系

- **与 Cascade Head Count E2E (§四) 高度相关**: E2E 减小 H 导致崩溃 (mAP=0.684, Δ=−0.172), DSCR 约束 $\|\mathcal{J}_{\text{cas}}\|_2 \leq 0.9$ 可能过度约束 cascade 精化能力, 类似减小 H 的效果。
- **与 Head Early-Exit (§五) 相关**: Head Early-Exit 证明每个 head 都必要 (退出率 0%), DSCR 试图约束 cascade 复合行为, 可能与 "每个 head 必要" 的结论冲突。

### 教训

1. **理论包装必须匹配实际机制**: Dahlquist 稳定性理论包装宏大但根基不牢 — cascade head 不是 ODE 数值方法。后续方向必须验证理论前提与实际机制的匹配性。
2. **计算开销估算必须实测**: "5%" 的声称与实际 >500% 相差 100 倍。后续方向的开销估算必须基于代码级分析, 不能依赖文档声称。

---

## 十七、BDS-RF (Bounded-Differences Stability-Regularized RF) — 适用对象错误 (理论评审淘汰, 未实验)

### 核心设想

BDS-RF 提出: 对预测框 $\hat{x}_0$ 施加高斯扰动 $\delta$, 计算扰动前后检测损失的差 $\Delta\ell = \ell(\hat{x}_0 + \delta) - \ell(\hat{x}_0)$, 惩罚 $(\Delta\ell)^2$。期望通过约束预测扰动稳定性提升泛化。理论根: McDiarmid 不等式 (1989)、Efron-Stein 不等式 (1981)、经验 Bernstein 不等式、算法稳定性理论 (Bousquet & Elisseeff 2002)。

### 评审淘汰过程

**状态**: ⛔ 理论评审淘汰 (未真实实验, 仅经 R1 A↔B 评审)
**R1 评分**: 5.0/10 (Reject, Reviewer B)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_BDS_RF.md` + `BDS_RF_REVIEW_R1.md`)

### 理论缺陷分析

#### 缺陷 1 (致命): McDiarmid 不等式的适用对象错误

- **错误描述**: McDiarmid 有界差不等式是关于**独立随机变量的函数**的浓度不等式: $\Pr[f(X_1, \ldots, X_n) - \mathbb{E}f \leq -t] \leq \exp(-2t^2 / \sum c_i^2)$, 要求 $X_1, \ldots, X_n$ **独立**。BDS-RF 将其应用于预测框 $\hat{x}_0$ 的扰动, 但 $\hat{x}_0$ 是网络参数 $\theta$ 和输入 $x_t$ 的确定性函数, 不是随机变量。
- **错误依据**: McDiarmid (1989) 中的 "变量" 是训练样本 $X_i$ (独立), 不是预测 $\hat{x}_0^{(i)}$ (非独立, 由网络参数决定)。对 $\hat{x}_0$ 施加扰动 $\delta$ 是数据增强/一致性正则化, 与 McDiarmid 浓度不等式无关。
- **为何不可修复**: 这是概念性错误 — 将 "预测扰动稳定性" (prediction stability) 与 "输入扰动浓度" (input concentration) 混淆。除非完全替换为 Lipschitz 连续性/局部平滑性理论, 但那已不是 BDS-RF 的理论包装。

#### 缺陷 2 (致命): Efron-Stein 不等式同样适用对象错误

- **错误描述**: Efron-Stein 不等式 (1981) 界定**独立随机变量函数的方差**: $\text{Var}(f(X_1, \ldots, X_n)) \leq \frac{1}{2} \sum_i \mathbb{E}[(f - f^{(i)})^2]$, 要求 $X_i$ 独立, "替换" 是替换输入变量。BDS-RF 对 $\hat{x}_0$ (网络输出) 施加扰动, 不是对训练样本做替换。
- **错误依据**: Efron-Stein 的 $f^{(i)}$ 是将 $X_i$ 替换为独立副本后的函数值, 不是对输出加噪声。
- **为何不可修复**: 与缺陷 1 同源, 适用对象错误。

#### 缺陷 3 (严重): 算法稳定性概念混淆

- **错误描述**: 文档引用 Bousquet & Elisseeff (2002) 算法稳定性, 但算法稳定性是指**训练集改变一个样本时, 学习算法输出的函数变化有界**: $|f_S - f_{S^{(i)}}| \leq \beta$。这是关于**训练集扰动**的稳定性, 不是关于**预测扰动**的稳定性。
- **错误依据**: Bousquet & Elisseeff (2002) 的 $\beta$-稳定性是 leave-one-out 意义下的稳定性, 与 BDS-RF 的预测扰动正则化完全不同。BDS-RF 的实际机制更接近一致性正则化 (FixMatch, Π-model, Mean Teacher), 但这些方法的理论基础是 Lipschitz 连续性, 不是浓度不等式。
- **为何不可修复**: 理论包装与实际机制严重不匹配。若改用 Lipschitz/平滑性理论, BDS-RF 沦为普通的一致性正则化, 无独立理论贡献。

#### 缺陷 4 (中等): 与 box_renewal 的交互未讨论

- **错误描述**: KaryoFlow 的 box_renewal 机制使 proposal 位置每步大幅变化 (93-124 px/步, 见 §五 RoI Feature Cache)。BDS-RF 的扰动 $\delta$ 在归一化坐标 [0,1] 下, 相对于 box_renewal 的位移可能微不足道。
- **错误依据**: FALSIFIED §五 RoI Feature Cache 证实 box 位移 93-124 px/步, 缓存命中率接近 0。若 $\delta$ 远小于 box_renewal 位移, BDS-RF 的正则化效果可能被淹没。
- **为何不可修复**: box_renewal 是训练核心机制 (no_box_renewal 训练消融 Δ=−0.016, §十; 推理时 K≥200 可安全关闭但 K=100 仍有 −0.016 退化, 见 LINEAGE §六); BDS-RF 的扰动尺度无法与 box_renewal 的位移竞争。

### 与已证伪方向的关系

- **与 scale_aware_loss (§十) 概念相关**: 两者都涉及预测/框的稳定性, scale_aware_loss 失败因推理尺度不可靠 (0.742, Δ=−0.004)。BDS-RF 同样在预测框上施加扰动, 面临 "扰动后的框可能不在合理范围内" 的类似问题。
- **与 RoI Feature Cache (§五) 相关**: box_renewal 93-124 px/步位移与 BDS-RF 扰动尺度的交互问题, 与 RoI Feature Cache 因 box 位移过大而失效 (缓存命中率 0%) 同源。

### 教训

1. **浓度不等式的适用对象必须严格区分**: McDiarmid/Efron-Stein 是关于独立随机变量的浓度, 不是确定性预测的扰动。后续方向若使用浓度不等式, 必须验证随机性来源。
2. **理论包装与实际机制必须匹配**: BDS-RF 本质是一致性正则化, 却包装为浓度不等式和稳定性理论, 包装与机制严重不匹配是淘汰主因。

---

## 十八、PDR (Proposal Dependency Regularization) — 三大理论支柱均有根本问题 (理论评审淘汰, 未实验)

### 核心设想

PDR 提出: 在训练损失中增加 HSIC (Hilbert-Schmidt Independence Criterion) 正则项, 惩罚重叠 proposal 对之间的统计依赖。期望降低 proposal 间依赖 → 提升有效样本量 (ESS) → 收紧 PAC-Bayes 界。理论根: HSIC (Gretton 2005)、Janson 浓度不等式 (2001)、PAC-Bayes 泛化界 (McAllester 1999)。核心机制: 重叠 proposal 共享 RoI 特征 → 预测依赖 → ESS 下降 → 泛化受损。

> **注**: 用户曾特别关注此方向 (打开 PDR 设计文档), 但理论评审确认三大支柱均有根本问题。

### 评审淘汰过程

**状态**: ⛔ 理论评审淘汰 (未真实实验, 仅经 R1 A↔B 评审)
**R1 评分**: 5.0/10 (Reject, Reviewer B)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_PDR.md` + `PDR_REVIEW_R1.md`)

### 理论缺陷分析

#### 缺陷 1 (致命): PAC-Bayes 界的适用对象错误

- **错误描述**: PAC-Bayes 界 $\mathbb{E}_Q[R(\theta)] \leq \mathbb{E}_Q[\hat{R}_S(\theta)] + \sqrt{(\text{KL}(Q \| P) + \ln(2\sqrt{n}/\delta))/(2n)}$ 中的 $n$ 是**训练图像数** (i.i.d. 样本), 不是 proposal 数。PDR 论证链 "降低 proposal 间 HSIC → $n_{\text{eff}} \uparrow$ → 界收紧" 混淆了两者。
- **错误依据**: Proposals 是模型输出 (给定图像和噪声后的确定性预测), 不是 i.i.d. 训练样本。PAC-Bayes 界的 i.i.d. 假设是关于训练图像 $z_i = (x_i, y_i)$ 的独立性, 不是关于 proposal 预测的独立性。降低 proposal 间 HSIC **不影响** PAC-Bayes 界中的 $n$。
- **为何不可修复**: 将 proposal 间依赖等同于训练样本间依赖, 是对 PAC-Bayes 框架的根本误解。除非重新推导 proposal 级别的泛化界 (需全新理论), 但当前 PAC-Bayes 框架不适用。

#### 缺陷 2 (致命): HSIC 在确定性变量上无统计意义

- **错误描述**: HSIC 度量两个随机变量 $X, Y$ 的**分布**依赖性, 需从联合分布 $P_{XY}$ 中采样多个 i.i.d. 样本。PDR 在单个 batch 内对单张图像的 $K \approx 46$ 个 proposals 计算 HSIC, 但 $Z_k = [\hat{x}_0^{(k)}; \text{softmax}(p^{(k)})]$ 在给定图像和模型参数后是**确定性的**。
- **错误依据**: HSIC 的经验估计 $\text{HSIC}_b(Z_i, Z_j) = m^{-2} \text{tr}(\tilde{K} \tilde{L})$ 中 $m$ 应是从分布中采样的样本数, 不是 proposal 数。对确定性变量计算 HSIC 实际是核矩阵的迹, 非统计依赖性的估计。
- **为何不可修复**: 除非明确随机性来源 (如噪声 proposal $x_1$ 的随机性), 并证明在不同 $x_1$ 采样下 HSIC 估计的收敛性, 但文档未提供此推导。

#### 缺陷 3 (严重): Janson 不等式的适用对象错误

- **错误描述**: Janson 不等式 (2001, Theorem 2.1) 是关于**依赖的 Bernoulli 随机变量之和**的浓度, 其中 $\Delta$ 度量**事件依赖** (event dependence, 如 $X_i$ 和 $X_j$ 共享底层随机变量)。PDR 将其应用于 "proposals 作为依赖样本", 但 proposals 的 "依赖" 是**特征依赖** (feature dependence, 核矩阵非对角元非零)。
- **错误依据**: 推论 1.5 中 $\rho \approx 0.3$ 导致 $n_{\text{eff}} \approx 6.3$ 的计算, 将 HSIC 与 Pearson 相关系数 $\rho$ 混淆, 且将 Janson 的 $\Delta$ 与 $\rho$ 直接挂钩, 缺乏严格推导。Janson 的 $\Delta$ 是关于 Bernoulli 事件的协方差, 不是核矩阵的迹。
- **为何不可修复**: 事件依赖与特征依赖是不同概念, 无法通过简单类比连接。

#### 缺陷 4 (严重): 重叠 proposal 的预测依赖是检测的正确行为

- **错误描述**: PDR 惩罚重叠 proposal 对的依赖, 但在检测任务中, 重叠 proposal 的预测相关性是**正确行为** — 如果 proposal A 和 B 都与同一 GT 匹配, 它们的预测**应该**相似 (都指向 GT)。
- **错误依据**: PDR 惩罚 $Z_A, Z_B$ 的依赖, 可能迫使模型对重叠 proposal 做出**不同**预测, 这与检测目标矛盾 — 我们希望重叠 proposal 预测一致, 然后 NMS 去重。PDR 可能降低 proposal 预测一致性 → NMS 难以去重 → 更多假阳性。
- **为何不可修复**: 这是 PDR 设计理念的根本矛盾。除非设计仅惩罚 "冗余依赖" (非 GT 驱动的依赖) 的机制 (如条件 HSIC), 但文档未提供。

### 与已证伪方向的关系

- **与 Class-Balanced Sampling (§七/§十) 概念相关**: PDR 的类感知边权 $w_{ij} = 1/(\pi_{c_i} + \pi_{c_j})$ 使罕见类 (Y 染色体) 边权 ≈ 43.1, 常染色体 ≈ 11.5, 本质上是类别频率重加权, 与 CBS (SIGKILL) 和 SeesawLoss (0.744, 证伪) 的目标一致。
- **与 CFM / OT Diversity Collapse (§三) 概念相关**: OT Diversity Collapse 是 proposal 耦合多样性的问题, PDR 试图降低 proposal 依赖, 但 OT DC 是关于噪声-GT 耦合, PDR 是关于 proposal 输出依赖。

### 教训

1. **泛化界中的 $n$ 必须严格区分**: PAC-Bayes 界的 $n$ 是训练图像数 (i.i.d.), 不是 proposal 数。后续方向若使用泛化界, 必须验证 $n$ 的语义。
2. **统计量在确定性变量上无意义**: HSIC 度量分布依赖性, 对确定性变量 (给定输入后的网络输出) 无统计意义。必须明确随机性来源。
3. **正则化对象不能是检测的正确行为**: 重叠 proposal 预测一致是检测的正确行为, 惩罚它会适得其反。

---

## 十九、LDCR (Large Deviation Cascade Regularization) — 渐近性不满足 (理论评审淘汰, 未实验)

### 核心设想

LDCR 提出: 用大偏差理论 (Cramér/Gärtner-Ellis) 分析 6 级 cascade head 误差序列 $\{e_k\}$ 的尾部概率, 用 LogSumExp 形式的经验 CGF (累积量生成函数) 估计速率函数, 惩罚尾部概率上界。期望通过指数级收紧尾部概率界提升稀有类 (Y, G21 染色体) AP。理论根: Cramér 定理 (1938)、Gärtner-Ellis 定理 (1977/1984)、Chernoff 界 (1952)、Markov 链 LDP (Donsker-Varadhan)。

### 评审淘汰过程

**状态**: ⛔ 理论评审淘汰 (未真实实验, 仅经 R1 A↔B 评审)
**R1 评分**: 5.6/10 (Reject, Reviewer B)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_LDCR.md` + `LDCR_REVIEW_R1.md`)

### 理论缺陷分析

#### 缺陷 1 (致命): K=6 不满足大偏差理论渐近性

- **错误描述**: 大偏差理论 (Cramér, Gärtner-Ellis) 是 $K \to \infty$ 的渐近理论, 要求样本数趋于无穷才能建立 LDP (大偏差原理)。KaryoFlow 的 $K=6$ (cascade head 级数) 远非渐近区域。
- **错误依据**: Cramér 定理和 Gärtner-Ellis 定理都要求 $K \to \infty$。$K=6$ 时, $P(\bar{e}_K > \varepsilon)$ 的指数衰减率 $I(\varepsilon)$ 与 CGF 的 Legendre 变换的关系不严格成立。Chernoff 界仍成立 (对任意 $K$), 但 "LDP" 与 "速率函数" 概念不适用。此外, Gärtner-Ellis 定理对 Markov 链的扩展 (Donsker-Varadhan) 要求 Markov 链同质 (转移核不随 $k$ 变化), 而 KaryoFlow 的 6 级 cascade head 参数不同 (每级 head 有独立权重), 转移核非同质。
- **为何不可修复**: $K=6$ 是架构固定值, 无法增大至渐近区域。Chernoff 界虽有效但非 "大偏差理论", 理论包装失效。

#### 缺陷 2 (严重): cascade_detach=True 与 Markov 链 LDP 论证矛盾

- **错误描述**: 文档定义 1.1 称 "由于 cascade_detach=True, 反向传播时各级独立, 但前向传播构成 Markov 链"。LDCR 对所有级求和 $(1/K) \sum \Lambda_k$, 但 cascade_detach=True 切断了反向传播的 Markov 依赖性。
- **错误依据**: LDCR 实际是对 $K$ 个独立误差 (反向传播视角) 的 CGF 求平均, 非对 Markov 链误差的 LDP。前向 Markov 依赖性对训练梯度的影响为零 (因反向传播独立)。论断 1 称 "误差沿链传播", 但 cascade_detach=True 切断了这种反向传播影响。
- **为何不可修复**: cascade_detach=True 是 KaryoFlow 的核心设计 (S1 理论分析要求), 改为 False 会破坏横向收敛性。LDCR 的 Markov 链 LDP 论证与实际实现矛盾。

#### 缺陷 3 (严重): "d=4 CGF 维度红利" 论断误导

- **错误描述**: 论断 4 称 "误差 $e_k = \|\hat{x}_0^{(k)} - x_0\|_2 \in \mathbb{R}_{\geq 0}$ 是标量 (bbox 4 维空间的 L2 范数), CGF 是一维的; 对图像生成 ($d = 3 \times 256^2 = 196608$), 误差范数分布尾部极重, CGF 估计不可行"。
- **错误依据**: 误差范数**在任何维度都是标量** — 图像生成的误差 $e = \|\hat{x} - x\|_2$ 也是标量, 其 CGF 也是一维的。"CGF 维度红利" 不存在。$d=4$ 的真正优势在于 $N_{\text{pos}}$ (前景 proposal 数) 在检测中固定 (~500), 而图像生成的 "样本" 是像素 (数量巨大), 但这与 CGF 维度无关。
- **为何不可修复**: 论断 4 是 LDCR 对 $d=4$ 任务匹配的主要论证, 错误后需重新定位 $d=4$ 的优势。

#### 缺陷 4 (中等): 稀有类受益缺乏严格证明

- **错误描述**: 论断 5 称 "稀有类 (Y, G21) 误差分布右尾更重, LDCR 的 LogSumExp 对大误差样本赋予指数级权重, 稀有类不成比例受益"。
- **错误依据**: "稀有类误差右尾更重" 是假设非证明 — 稀有类样本少不直接导致误差大。LogSumExp 是误差加权非类别加权, 若多数类也有大误差样本 (如重叠区域), LDCR 会同样加权它们, 稀有类不一定 "不成比例受益"。
- **为何不可修复**: 需补充稀有类 vs 多数类误差分布的实验诊断, 但未实验无法验证。

#### 缺陷 5 (中等): LogSumExp 梯度爆炸风险

- **错误描述**: LogSumExp($\lambda e_k$) 的梯度 $\partial/\partial e_k = \exp(\lambda e_k) / \sum \exp(\lambda e_j)$, 当 $\lambda$ 大且 $e_k$ 有大异常值时, 梯度权重集中在最大误差样本, 训练被 outlier 主导。
- **错误依据**: $\lambda$ 是超参, $\lambda$ 小则 LDCR 退化为 L1 (仅控制均值), $\lambda$ 大则梯度爆炸。文档未给出 $\lambda$ 的选择准则。
- **为何不可修复**: $\lambda$ 选择对 LDCR 有效性关键, 但无理论指导, 需经验调参。

### 与已证伪方向的关系

- **与 SeesawLoss / Class-Balanced Sampling (§七/§十) 潜在重叠**: LDCR 的 "稀有类受益" 论断与 CBS 的频率补偿机制有相似目标。若稀有类误差系统性大于多数类, LDCR 与 SeesawLoss (0.744, 证伪) 的效果可能趋同。
- **与 high_cls/high_giou_weight (§十) 潜在重叠**: LDCR 增加新正则项, 总损失权重调整, 与损失权重调整有相似性。

### 教训

1. **渐近理论在有限 $K$ 下不严格成立**: 大偏差理论是 $K \to \infty$ 的渐近理论, $K=6$ 远非渐近区域。后续方向若使用渐近理论, 必须验证 $K$ 是否足够大。
2. **前向 Markov 依赖 + 反向 detach = 半耦合**: cascade_detach=True 切断了反向传播的 Markov 依赖性, 使 Markov 链 LDP 论证失效。必须区分前向依赖与反向梯度流。

---

## 二十、PCR-Matcher (Perturbation-Conditioned Robust Matcher) — 条件数误用 (理论评审淘汰, 未实验)

### 核心设想

PCR-Matcher 提出: 用 Renegar 条件数分析 SimOTA 硬匹配的不稳定性 (条件数发散 → 匹配对代价扰动敏感), 用熵正则 OT (非平衡 Sinkhorn) 替代 SimOTA 的硬 argmin, 软化匹配决策。期望通过有界化条件数 ($1/\varepsilon$ 界) 提升密集重叠场景的匹配稳定性, 稀有类不成比例受益。理论根: Renegar 条件数 (1994, 1995)、Sinkhorn 算法 (Cuturi 2013)、非平衡 OT (Chizat et al. 2018)、Franklin-Lorenz 收敛性 (1989)。

### 评审淘汰过程

**状态**: ⛔ 理论评审淘汰 (未真实实验, 仅经 R1 A↔B 评审)
**R1 评分**: 5.7/10 (Reject, Reviewer B)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_PCR_MATCHER.md` + `PCR_MATCHER_REVIEW_R1.md`)

### 理论缺陷分析

#### 缺陷 1 (致命): 引理 1.3 误用 Renegar 条件数

- **错误描述**: 引理 1.3 定义 "SimOTA LP 的 Renegar 条件数为 $\kappa_R(C) = \|C\|_F / \Delta_C^{\min}$", 其中 $\Delta_C^{\min} = \min_{i, j_1 \neq j_2} |C_{ij_1} - C_{ij_2}|$ 是同一 proposal 对不同 GT 的最小代价差。证明称 "约束矩阵 A 结构固定良态, 灵敏度完全由代价向量 c 决定"。
- **错误依据**: Renegar (1994, 1995) 定义 LP $\min\{c^T x : Ax \leq b\}$ 的条件数为 $\kappa_R(A, b, c) = \|A\| \cdot \|b\| / \sigma_{\min}(A^*, \bar{c})$, 主要依赖**约束矩阵 A** 和 b, 不是代价向量 c。"灵敏度完全由 c 决定" 是对 Renegar 条件数的误解。$\kappa_R(C) = \|C\|_F / \Delta_C^{\min}$ 更接近 "代价间距灵敏度" (启发式), 非严格 Renegar 条件数。
- **为何不可修复**: 引理 1.3 是 PCR-Matcher 理论核心 (硬匹配条件数发散 → 需要熵正则化)。若 $\kappa_R(C)$ 不是真正的 Renegar 条件数, "条件数发散" 论证失效。除非使用正确的 Renegar 定义 (涉及 A, b, c), 但 SimOTA 的约束矩阵 A 结构使条件数分析复杂化。

#### 缺陷 2 (严重): 引理 1.5 Tikhonov 对偶论证不严格

- **错误描述**: 引理 1.5 声称 "熵正则 OT 等价于在传输多面体上对线性目标施加 Tikhonov 正则化, 其 Hessian 为 $\text{diag}(1/P^*_{ij})$ (正定)"。
- **错误依据**: 标准 Tikhonov 正则化 $\min_x \|Ax-b\|^2 + \eta\|x\|^2$ 的 Hessian 是 $A^T A + \eta I$ (**常数**), 不是自适应的 $\text{diag}(1/P^*_{ij})$ (依赖解 $P^*_\varepsilon$)。"自适应 Tikhonov" 是定性类比, 非严格等价。两者的光滑性机制不同: Tikhonov 通过 Hessian 正定化, 熵通过指数核 $\exp(-C/\varepsilon)$。
- **为何不可修复**: 引理 1.5 是定理 1.6 (条件数有界化) 的理论基础。可直接从 Sinkhorn 解闭式 $P^*_{ij} = u_i \exp(-C_{ij}/\varepsilon) v_j$ 证明光滑性, 无需 Tikhonov 类比, 但文档选择了不严格的论证路径。

#### 缺陷 3 (严重): 定理 1.6 的 1/ε 界引用不准确

- **错误描述**: 定理 1.6 声称 "熵正则 OT 解 $P^*_\varepsilon(C)$ 对代价矩阵 $C$ 的灵敏度为 $\|dP^*_\varepsilon / dC\| \leq 1/\varepsilon$", 证明引用 Franklin & Lorenz (1989) Theorem 3.1。
- **错误依据**: Franklin & Lorenz (1989) Theorem 3.1 是关于 **Sinkhorn 迭代的收敛性**, 不是关于解对 $C$ 的灵敏度。该定理证明 Sinkhorn 迭代线性收敛到 $P^*$, 不直接给出 $dP^*/dC$ 的界。证明中 "$d(\log u_i)/dC_{kl}$ 有界 (因 Sinkhorn 迭代压缩性)" 的界未给出具体值, 实际可能依赖 $\varepsilon$ 和 $N$。
- **为何不可修复**: $1/\varepsilon$ 界是 PCR-Matcher 的核心理论保证。需严格推导 $dP^*/dC$ 界 (不依赖 Franklin-Lorenz Theorem 3.1), 或引用正确文献 (如 Peyré-Cututi 2019)。

#### 缺陷 4 (中等): 与 AMR (Assignment Margin Regularization) 重叠

- **错误描述**: AMR (hinge 正则保留 SimOTA) 和 PCR-Matcher (Sinkhorn 替换 SimOTA) 都针对 SimOTA 不稳定性, 稀有类受益论证相似 ("稀有类间隔/条件数更差 → 正则化边际收益更大")。
- **错误依据**: AMR 用 McDiarmid/Bousquet-Elisseeff (集中性+稳定性), PCR-Matcher 用 Renegar/Sinkhorn (条件数+熵正则), 理论根不同但目标相同。两者应选择其一, 非同时采纳。
- **为何不可修复**: 两者机制不同 (AMR 正则 cost 矩阵, PCR-Matcher 软化匹配), 但目标重叠。若 AMR 已被推荐, PCR-Matcher 的独立价值有限。

### 与已证伪方向的关系

- **与 Class-Balanced Sampling / SeesawLoss (§七/§十) 概念相关**: PCR-Matcher 的 "稀有类受益" 论断与 CBS 的频率补偿有相似目标, "类无关均匀熵正则" 与 SeesawLoss 的频率加权都试图帮助稀有类。
- **与已证伪方向无实质重合**: PCR-Matcher 改 matcher, 不改 RF/损失/head/cascade, 与 §一~§十一 已证伪方向无实质重叠。

### 教训

1. **条件数定义必须严格**: Renegar 条件数是关于约束矩阵 A, 不是代价向量 c。$\|C\|_F / \Delta_C^{\min}$ 是启发式度量, 非严格 Renegar 条件数。后续方向若使用条件数, 必须验证定义的正确性。
2. **文献引用必须精确**: Franklin & Lorenz Theorem 3.1 是关于 Sinkhorn 迭代收敛性, 不是解对 C 的灵敏度。引用前必须核实定理内容。

---

## 二十一、Traj-SAM (Trajectory-Integrated Sharpness-Aware Minimization) — 训练开销+概念重叠 (理论评审淘汰, 未实验)

### 核心设想

Traj-SAM 提出: 将 SAM (Sharpness-Aware Minimization) 扩展到 RF 轨迹积分形式, 用独立 $t'$ 采样收紧 Jensen 间隙 ($\Delta_J^{\text{traj}} \leq (1/\sqrt{B}) \text{Std}_t[\nabla L(\theta, t)]$)。Focal-SAM 扩展: 类条件扰动半径 $\rho_c = \rho_0 \cdot (\bar{n}/n_c)^{1/2}$, 对稀有类施加更大扰动。期望通过锐度最小化提升小数据集 ($n=2274$) 泛化, PAC-Bayes KL 项 $\sim 1/\sqrt{n}$ 主导。理论根: SAM (Foret et al. 2021)、PAC-Bayes (McAllester 1999)、Bartlett-Foster-Telgarsky 有效秩 (2020)。

### 评审淘汰过程

**状态**: ⛔ 理论评审淘汰 (未真实实验, 仅经 R1 A↔B 评审; 仅标准 SAM 部分可采纳, 但训练开销不可接受)
**R1 评分**: 6.2/10 (Conditional Recommendation, 仅标准 SAM 部分, Reviewer B)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_TRAJ_SAM.md` + `TRAJ_SAM_REVIEW_R1.md`)

### 理论缺陷分析

#### 缺陷 1 (严重): 定理 2 Cauchy-Schwarz 应用错误

- **错误描述**: 定理 2 第 3 步 (期望) 称 "$\mathbb{E}_{t,t'}[g(t)^T g(t') / \|g(t)\|] \leq \|g_{\text{avg}}\|^2$ (Cauchy-Schwarz)", 声称 Traj-SAM 估计的方差严格小于标准 SAM, Jensen 间隙收紧 $\sqrt{B}$ 倍。
- **错误依据**: 正确推导: $\mathbb{E}_{t,t'}[g(t)^T g(t') / \|g(t)\|] = \mathbb{E}_t[g_{\text{avg}}^T g(t) / \|g(t)\|] \leq \mathbb{E}_t[\|g_{\text{avg}}\|] = \|g_{\text{avg}}\|$ (由 Cauchy-Schwarz $g_{\text{avg}}^T g(t) / \|g(t)\| \leq \|g_{\text{avg}}\|$)。原文写 $\leq \|g_{\text{avg}}\|^2$, **多了一个平方**, 是错误的。正确界是 $\leq \|g_{\text{avg}}\|$, 不是 $\leq \|g_{\text{avg}}\|^2$。
- **为何不可修复**: 定理 2 是 Traj-SAM 的核心理论创新 (独立 $t'$ 采样收紧 Jensen 间隙)。Cauchy-Schwarz 应用错误使 "收紧 $\sqrt{B}$ 倍" 结论不可靠。需重新推导方差分解, 但可能无法得到原结论。

#### 缺陷 2 (严重): Focal-SAM 类条件 $\rho_c$ 与 SAM 全局扰动哲学冲突

- **错误描述**: Focal-SAM 定义 $\rho_c = \rho_0 \cdot (\bar{n}/n_c)^{1/2}$, 对 Y 染色体 $\rho_Y \approx 1.93 \rho_0$, 对常染色体 $\rho_{\text{auto}} \approx 0.98 \rho_0$。但 SAM 的核心是 "在参数空间 $\rho$-球内寻找最大损失", 扰动是**全局的** (所有参数同一 $\rho$)。
- **错误依据**: 参数 $\theta$ 是共享的 (同一网络处理所有类), 如何在单次 forward-backward 中对不同类施加不同扰动? Focal-SAM 需在 batch 中按类别分组, 每组用不同扰动半径, 但 SAM 的双 forward-backward 是对整个 batch 的, 不是按类分组的。定理 4 证明引用 NTK (Neural Tangent Kernel) 理论, 但 NTK 在无限宽网络下成立, KaryoFlow 的 cascade head 是有限宽的, NTK 近似可能不适用。
- **为何不可修复**: Focal-SAM 的工程实现复杂度高, 且理论基础 (NTK) 在有限网络不严格。除非简化为 "稀有类样本权重 × SAM 扰动", 但那已非原始 Focal-SAM 设计。

#### 缺陷 3 (严重): 训练时间翻倍

- **错误描述**: SAM 的双 forward-backward 使训练时间 $\sim 2\times$。KaryoFlow 的 A4 训练 150 epoch, 单 seed 约 3-5 天 (A6000)。Traj-SAM 需 6-10 天单 seed, 3-seed 需 18-30 天。
- **错误依据**: SAM 的 $\epsilon^* = \rho \nabla L(\theta; B) / \|\nabla L(\theta; B)\|$ 需要第一次 forward-backward 计算梯度, 然后扰动参数 $\theta + \epsilon^*$, 再做第二次 forward-backward 评估扰动后损失并反传。双倍计算量是 SAM 的固有开销。
- **为何不可修复**: 除非使用 subset SAM (如 25% 样本) 或仅在训练后期启用 SAM, 但前者降低 SAM 有效性, 后者增加调度复杂度。

#### 缺陷 4 (中等): 与 TFR/VCR 概念重叠

- **错误描述**: Traj-SAM 通过锐度最小化正则化轨迹平坦性, 与 TFR (Trajectory Flatness Regularization) 和 VCR (Velocity Consistency Regularization) 概念重叠 — 都正则化轨迹的平坦性。
- **错误依据**: TFR 正则化 $\|\partial \hat{x}_0 / \partial t\|^2$, VCR 正则化 $v_\theta$ 时间一致性, Traj-SAM 正则化损失景观锐度 (间接使轨迹平坦)。三者目标相似 (平坦轨迹), 但机制不同 (TFR/VCR 直接正则轨迹, Traj-SAM 通过优化器间接正则)。
- **为何不可修复**: 若 TFR/VCR 已被推荐, Traj-SAM 的边际收益有限。SAM 本身任务无关 (通用优化器改进), 非 RF 检测专属创新。

#### 缺陷 5 (中等): 推论 3.1 (d=4 Hessian 可控) 论证错误

- **错误描述**: 推论 3.1 声称 "Hessian 项 $(\rho^2/2p)\|\nabla^2 L\|_F^2$ 中, $\nabla^2 L \in \mathbb{R}^{p \times p}$ 的有效秩由数据维度决定, 对 RF 的 x0-prediction, 有效秩 $\leq K \times 4 = 184 \ll p \sim 10^6$"。
- **错误依据**: Hessian $\nabla^2 L(\theta)$ 是损失对**参数** $\theta \in \mathbb{R}^p$ 的 Hessian, 是 $p \times p$ 矩阵, 其有效秩由参数空间结构决定, 不是数据维度 $d=4$。Bartlett-Foster-Telgarsky (2020) 的有效秩界是关于神经网络的 Rademacher 复杂度, 不是 Hessian 的 Frobenius 范数。将数据维度 $d=4$ 直接映射到 Hessian 有效秩是错误的。
- **为何不可修复**: Hessian 有效秩通常远大于数据维度, 受网络参数量、深度、激活函数等影响。"d=4 使 Hessian 项可控" 不成立。

### 与已证伪方向的关系

- **与 h_velocity_loss (§八/§十四) 区分清晰**: Traj-SAM 改优化器, h_velocity_loss 改损失, 机制不同。
- **与 stability/warm_restart (§十一) 概念相关**: warm_restart (0.714-0.728, 证伪) 试图改善训练稳定性, Traj-SAM 通过锐度最小化改善稳定性, 目标相似但机制不同。
- **与 SeesawLoss / CBS (§七/§十) 潜在重叠 (Focal-SAM 部分)**: Focal-SAM 的 $\rho_c \sim 1/\sqrt{n_c}$ 直接依赖类频率, 与 SeesawLoss (0.744, 证伪) 的频率驱动机制相似。若 Focal-SAM 与 SeesawLoss 效果趋同, 可能也失败。

### 教训

1. **Cauchy-Schwarz 应用必须验证平方**: Traj-SAM 定理 2 的 $\leq \|g_{\text{avg}}\|^2$ 应为 $\leq \|g_{\text{avg}}\|$, 多了一个平方使核心结论失效。后续方向使用 Cauchy-Schwarz 必须逐项验证量纲。
2. **训练时间翻倍是 SAM 的固有开销**: 双 forward-backward 无法避免。后续方向若使用 SAM, 必须评估训练时间可行性。
3. **数据维度 ≠ Hessian 有效秩**: Hessian 是参数空间的 $p \times p$ 矩阵, 有效秩由参数空间结构决定, 不是数据维度 $d=4$。

---

## 二十二、冗余淘汰方向 (MEC-RF / EXER-RF / MDC-RF) — 与已推荐方向高度冗余 (理论评审淘汰, 未实验)

> **失败性质**: ⛔ **冗余淘汰** (非理论错误, 而是与已推荐方向 BEAR/TFR 高度冗余, 经评审建议合并或淘汰)
> **方法学判定**: 部分方向有核心数学错误 (MEC-RF Theorem 2.3, MDC-RF Theorem 2.2), 但主要淘汰原因是冗余性
> **归档日期**: 2026-07-28

这三个方向并非纯理论错误, 而是与已推荐方向 (BEAR/TFR) 高度冗余, 经评审建议合并或淘汰。部分方向还伴有核心数学错误, 进一步削弱独立价值。

### MEC-RF (Modified-Equation Compensated RF)

#### 核心设想

MEC-RF 提出: 用修正方程/后向误差分析理论, 正则化对流加速度 $a_\theta = (\nabla_x v_\theta) v_\theta$ (网络速度场沿自身的 Jacobian-向量积), 期望降低推理时数值轨迹与真实轨迹的偏差 (ISS 界)。理论根: 修正方程 (Hairer-Lubich-Wanner 2006)、JVP (前向模式自动微分)、Bartlett-Foster-Telgarsky 有效秩 (Rademacher 泛化界)。

#### 评审淘汰过程

**状态**: ⛔ 冗余淘汰 (未真实实验, 降为 BEAR fallback)
**R2 评分**: 6.5/10 (R1: 7.0, Reviewer B, 谨慎推荐-下降)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_MEC_RF.md` + `MEC_RF_REVIEW_R1.md` + `MEC_RF_R1_RESPONSE.md` + `MEC_RF_REVIEW_R2.md`)

#### 理论缺陷与冗余分析

1. **Theorem 2.3 Cauchy-Schwarz 方向反转错误 (核心数学错误)**: Theorem 2.3 声称 $\|\nabla_x v_\theta\|_{\text{op}} \leq M/V$ (JVP 控制完整算子范数), 但反例验证: $\text{diag}(1000, 0, 0, 0) + v_\theta = (0, 1, 0, 0)$ 时 $a_\theta = 0$ 但 $\|\nabla_x v_\theta\|_{\text{op}} = 1000$。JVP 仅控制 Jacobian 沿 $v_\theta$ 方向, 不控制完整算子范数。A 在 R2 中接受并删除 Theorem 2.3, 改为 Theorem 2.3' (仅下界 $\|\nabla_x v_\theta\|_{\text{op}} \geq M/V$)。

2. **Rademacher 路径完全删除**: 推论 2.2 的 Rademacher 泛化界因 Theorem 2.3 错误而失效, A 在 R2 中删除。MEC-RF 的精度保证从 "ISS + Rademacher 双路径" 缩减为 "仅 ISS 单路径", 小数据集 ($n=884$) 泛化论证实际失效 (ISS 仅保证推理精度, 非训练泛化)。

3. **A 自认 "BEAR 的高效近似"**: A 在 R2 中重新定位 MEC-RF 为 "BEAR 的高效近似" (1.3× JVP vs 3× 完整二阶)。MEC-RF 的正则目标 $a_\theta = (\nabla_x v_\theta) v_\theta$ 确实是 BEAR 正则目标 $\ddot{\hat{x}}_0$ 的一个子分量 (由引理 1.1)。独立价值仅在于计算效率 (1.3× vs 3×), 但若 BEAR 的 3× 开销可接受, 直接用 BEAR (完整二阶导数, 理论更严谨)。

4. **推论 1.1 "主导阶" 降级为 "显著分量"**: MEC-RF 选择正则对流加速度的理由从 "数学主导性" 变为 "计算可行性", 理论动机削弱。

5. **1/t² 数值风险升级**: 改进版 $t^2 \|a_\theta\|^2$ (消除 1/t²) 实际等价于 MDC-RF 的对流项, 失去 MEC-RF 独立性。

#### 状态

⛔ 冗余淘汰 (未实验, 降为 BEAR fallback)。R2 推荐: "作为 BEAR 开销不可接受时的 fallback, 不建议作为独立主推方向"。

### EXER-RF (Extrapolation Error Regularization)

#### 核心设想

EXER-RF 提出: 用 Newton 插值余项 + Lebesgue 常数理论, 正则化 $\hat{x}_0$ 沿轨迹的 $k$ 阶均差 $\|D_k\|^2$ ($k=1, 2$), 期望降低 DPM-Solver++ 的截断误差。核心创新: "原理性正则权重" $\lambda_{\text{ext}}^{(k)} = 2^{k+1} - 1$ (等距节点外插 Lebesgue 常数, $k=1: \lambda=3$, $k=2: \lambda=7$)。理论根: Newton 均差 (Davis 1975)、Lebesgue 常数 (Brutman 1978, 1982)、DPM-Solver++ 截断误差 (Lu et al. 2022)。

#### 评审淘汰过程

**状态**: ⛔ 冗余淘汰 (未实验, 被 BEAR 吸收)
**R1 评分**: 7.4/10 (Reviewer B, 谨慎推荐)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_EXER_RF.md` + `EXER_RF_REVIEW_R1.md`)

#### 理论缺陷与冗余分析

1. **与 BEAR (k=2) 高度冗余 — 正则同一对象**: EXER-RF (k=2) 正则化 $\|D_2\|^2$ (二阶均差), BEAR 也正则化 $\|D_2\|^2$ (二阶均差)。两者**正则化同一数学对象**, 仅权重不同:
   - EXER-RF: 权重为常数 $\lambda_{\text{ext}} = 7$ (节点几何决定)
   - BEAR: 权重为 $|\phi_2(t_a, t_b, t_c)|$ (时间步依赖, 求解器感知)
   BEAR 的权重设计更精细 (在高截断误差区域施加强正则)。

2. **Lebesgue 常数与 Newton 余项的耦合关系不严格**: 推论 2.4 声称 $\|\varepsilon_k\| \leq (2^{k+1}-1) \cdot \|D_{k+1}\| \cdot |\omega_{k+1}|$, 但 Newton 余项 $|P_k(t^*) - y(t^*)| = |D_{k+1}| \cdot |\omega_{k+1}(t^*)|$ **不含 Lebesgue 常数**。Lebesgue 常数衡量的是插值对**数据扰动**的敏感性 ($\|P_k - \tilde{P}_k\| \leq \Lambda_k \|y - \tilde{y}\|$), 非截断误差本身。将 $\lambda_{\text{ext}}$ 直接乘到截断误差界上缺乏严格推导。R1 标为 Blocking issue: "原理性正则权重" 应降级为 "启发式正则权重 (基于稳定性论证)"。

3. **两者权重均缺乏严格推导**: EXER-RF 的 $\lambda_{\text{ext}} = 7$ 来自 Lebesgue 常数 (但耦合关系不严格), BEAR 的 $|\phi_2|$ 来自修正方程缺陷项系数 (但 $\phi_2$ 与二阶方法的关联也有 gap)。两者权重设计的理论严格性均不足。

4. **EXER-RF (k=1) 与 MDC-RF 高度相似**: EXER-RF (k=1) 正则化 $\|D_1\|^2 \approx \|d\hat{x}_0/dt\|^2$ (全导数有限差分近似), 与 MDC-RF 的物质导数约束高度相似, 仅计算方式不同。

#### 状态

⛔ 冗余淘汰 (未实验, 被 BEAR 吸收)。R1 推荐: "保留 EXER-RF 的 k=1 模式 (与 TFR 关系需澄清), 淘汰 k=2 模式 (被 BEAR 覆盖)"。但 k=1 模式与 MDC-RF/TFR 高度相似, 独立价值有限, 最终整体淘汰。

### MDC-RF (Material Derivative Constraint)

#### 核心设想

MDC-RF 提出: 用连续介质力学物质导数 $D\hat{x}_0/Dt = \partial \hat{x}_0/\partial t + v_\theta \cdot \nabla_{x_t} \hat{x}_0$ 约束 $\hat{x}_0$ 沿轨迹的变化率, 期望从 Lagrangian 视角完整约束轨迹曲率。理论根: 物质导数 (Truesdell-Toupin 1960)、修正方程 (Hairer-Lubich-Wanner 2006)、DPM-Solver++ 截断误差 (Lu et al. 2022)。

#### 评审淘汰过程

**状态**: ⛔ 冗余淘汰 (未实验, 被 TFR+BEAR 覆盖)
**R1 评分**: 7.0/10 (Reviewer B, 谨慎推荐, 在 4 方向中排名最低)
**评审产物**: 已归档至 `docs/research/proposals/` 目录 (`V3_CONSERVATIVE_DESIGN_MDCRF.md` + `MDCRF_REVIEW_R1.md`)

#### 理论缺陷与冗余分析

1. **与 TFR 是包含关系 (MDC-RF = TFR + 对流项)**: MDC-RF 的正则目标展开为:
   $$\|D\hat{x}_0/Dt\|^2 = \underbrace{\|\partial_t \hat{x}_0\|^2}_{\text{TFR}} + 2\langle \partial_t \hat{x}_0, (\nabla_x \hat{x}_0) v_\theta \rangle + \underbrace{\|(\nabla_x \hat{x}_0) v_\theta\|^2}_{\text{纯对流项}}$$
   MDC-RF 严格包含 TFR 的正则目标, 是 TFR 的**严格推广** (subsumes), 非正交。文档 §5.4 和附录 B 声称的 "正交且可叠加" 是错误的。MDC-RF 单独使用时已包含 TFR 效果, 叠加 TFR 是冗余的。

2. **定理 2.2 链式法则代数错误 (核心数学错误)**: 定理 2.2 声称 $D^2 x_t / Dt^2 = D\hat{x}_0/Dt \cdot (-1/t) + \text{h.o.t.}$, 但正确展开为:
   $$\ddot{x}_t = \partial_t v_\theta + (\nabla_x v_\theta) v_\theta = -\frac{v_\theta}{t} - \frac{\partial_t \hat{x}_0}{t} + (\nabla_x v_\theta) v_\theta$$
   文档遗漏了 $v_\theta/t$ 和 $(\nabla_x v_\theta) v_\theta$ 项。R1 标为 Blocking issue。

3. **一阶导数正则 vs 二阶截断误差因果链断裂**: 定理 2.2 声称截断误差 $\propto \|D^2 \hat{x}_0/Dt^2\|$ (二阶), 但 MDC-RF 正则化 $\|D\hat{x}_0/Dt\|^2$ (一阶)。减小一阶导数**不严格蕴含**减小二阶导数。反例: 若 $D\hat{x}_0/Dt = c$ (常数非零), 则 $\|D\hat{x}_0/Dt\| = \|c\| > 0$ 但 $\|D^2 \hat{x}_0/Dt^2\| = 0$。此时 DPM-Solver++ 二阶方法**精确** ($\hat{x}_0$ 线性于 $t$), 但 MDC-RF 仍施加正则, 错误惩罚合法的线性变化。

4. **与 BEAR 设计哲学冲突**: MDC-RF 强制 $\hat{x}_0$ 常数 ($D\hat{x}_0/Dt = 0$), BEAR 允许 $\hat{x}_0$ 线性变化 ($D_2 = 0$ 但 $D_1 \neq 0$)。若 $\hat{x}_0$ 有合法的线性变化, MDC-RF 错误惩罚, BEAR 正确允许。BEAR 的设计哲学更精细。

5. **与 EXER-RF (k=1) 实质重复**: MDC-RF 正则化 $\|D\hat{x}_0/Dt\|^2$ (物质导数), EXER-RF (k=1) 正则化 $\|D_1\|^2 \approx \|d\hat{x}_0/dt\|^2$ (全导数有限差分近似), 两者在数学目标上高度相似, 仅计算方式不同。

#### 状态

⛔ 冗余淘汰 (未实验, 被 TFR+BEAR 覆盖)。R1 推荐: "在 4 个保守方向中冗余度最高, 独立价值最低, 其核心贡献可被 TFR + MEC-RF 的组合覆盖"。最终被 TFR (一阶偏导) + BEAR (二阶均差, 允许线性) 的组合完全覆盖。

### 三个方向的冗余关系总结

| 方向对 | 冗余度 | 关系 | 淘汰决策 |
|--------|--------|------|----------|
| MEC-RF ↔ BEAR | 高 | MEC-RF 是 BEAR 的子分量 ($a_\theta \subset \ddot{\hat{x}}_0$) | MEC-RF 降为 BEAR fallback |
| EXER-RF (k=2) ↔ BEAR | 极高 | 正则同一对象 $\|D_2\|^2$, 仅权重不同 | EXER-RF (k=2) 被 BEAR 吸收 |
| MDC-RF ↔ TFR | 严格包含 | MDC-RF = TFR + 对流项 | MDC-RF 被 TFR 覆盖 |
| MDC-RF ↔ BEAR | 设计冲突 | 强制常数 vs 允许线性 | BEAR 哲学更精细, MDC-RF 淘汰 |
| EXER-RF (k=1) ↔ MDC-RF | 高 | 正则同一数学对象 $\|d\hat{x}_0/dt\|^2$ | 两者均淘汰 |

### 教训

1. **冗余性检查必须在设计阶段进行**: MEC-RF/EXER-RF/MDC-RF 三个方向与 BEAR/TFR 的冗余性在 R1 评审中才被发现, 浪费了设计资源。后续方向设计时必须先绘制 "正则对象 vs 已有方向" 的覆盖图。
2. **"严格推广" 不等于 "独立方向"**: MDC-RF 严格包含 TFR, 但文档错误声称 "正交且可叠加"。后续方向必须严格验证与已有方向的关系 (正交/推广/重复), 不能凭直觉声称正交。
3. **核心数学错误会削弱独立价值**: MEC-RF (Theorem 2.3 Cauchy-Schwarz 反转) 和 MDC-RF (Theorem 2.2 链式法则错误) 的数学错误, 即使修订后仍削弱了独立价值, 使冗余性问题更突出。

---

## 二十三、SC-RF (自条件化 RF, 边际不采用)

### 核心设想

把上一步预测作为条件输入 (借鉴自条件化扩散模型思想), 期望模型利用预测历史改善去噪精度。

### 证伪证据

#### 实验证明目的

验证自条件化 RF 是否能超越 A4 baseline (0.863)。

- SC-RF (自条件化 RF, Dataset 2)
  -- 数据集: Dataset 2
  -- 结果: best mAP=0.860@ep82, **Δ = −0.003 vs A4 0.863** (在 noise 范围内但无增益)
  -- Early Stop @ ep112
  -- 自条件化机制实现正确 (50% 激活、零初始化过渡、校正幅度增长均正常)
  -- 数据源: ross `/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/sc_rf_24obj/`
  -- SwanLab: `ldmdet-breakthrough`
  -- 详细设计与复盘: [SC-RF_Self-Conditioned_Rectified_Flow.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/SC-RF_Self-Conditioned_Rectified_Flow.md)

### 失败原因分析

1. **边际结果, 无增益**: 自条件化机制实现正确但未带来 mAP 改善, 说明自条件化在低维 (d=4) RF 检测中价值有限。
2. **与 ScaleConditionedRF 形成对照**: ScaleConditionedRF (0.741 < 0.746, §一) 因训练-推理不一致而崩塌; SC-RF 至少无害 (未崩塌), 但也无增益, 证实自条件化本身非低维检测的关键瓶颈。

---

## 二十四、PD-RF (4→1 Progressive Distillation, 灾难性崩塌)

### 核心设想

将 A4 (4步 DPM-Solver++, mAP=0.862) 蒸馏到 1步 Euler, 期望 4× 推理加速 (4步→1步) 同时保持精度。采用直接 4→1 蒸馏 (非 Salimans 级联式渐进蒸馏)。

### 证伪证据

#### 实验证明目的

验证 4→1 直接蒸馏是否能将 4步 DPM-Solver++ 压缩到 1步 Euler 同时保持精度 ≥ A4 baseline。

- PD-RF v1-v4 (4→1 直接蒸馏, Dataset 2)
  -- 数据集: Dataset 2
  -- Teacher: A4 (4步 DPM-Solver++, mAP=0.862, 冻结)
  -- Student: 1步 Euler
  -- v1-v4 共 4 次迭代均失败
  -- best mAP=0.851@ep1 (即 A4 初始化点, 训练零增益)
  -- v4 最终 mAP 从 0.851 灾难性崩塌至 0.252@ep28, Early Stop @ ep31
  -- 归档时间: 2026-07-11
  -- 详细设计与复盘: [PD-RF_Progressive_Distillation.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/PD-RF_Progressive_Distillation.md)

### 失败原因分析

1. **1步 Euler 无法逼近 4步 DPM-Solver++ 预测**: teacher (4步 DPM++) 与 student (1步 Euler) 的预测 gap ~1.0, 蒸馏目标不可学, student 无法收敛。
2. **蒸馏梯度与检测梯度严重冲突**: grad_norm 持续 150-200, 蒸馏损失 (MSE 拟合 teacher) 与检测损失 (分类+回归 GT) 梯度方向冲突, 导致训练崩塌。
3. **直接 4→1 蒸馏跨度过大**: 不同于 Head Distillation (H=6→H=3, 同一 solver 内压缩, → LINEAGE §十五) 的成功, PD-RF 跨 solver (DPM++→Euler) 且跨步数 (4→1) 双重压缩, 蒸馏目标本身不可达。

### 教训

1. **蒸馏目标必须可达**: teacher 与 student 的预测 gap 过大时, 蒸馏目标不可学。后续蒸馏方向 (如 Head Distillation) 应先验证 teacher-student gap 在可学范围内。
2. **跨 solver 蒸馏比同 solver 蒸馏困难**: Head Distillation (同 DPM++ solver, H 压缩) 成功, PD-RF (DPM++→Euler + 步数压缩) 失败, 证实 solver 跨越是蒸馏的主要难点。

---

## 十二、证伪方向汇总与教训

### 证伪方向汇总表

| 方向 | mAP | Δ vs baseline | 数据集 | SwanLab project | 状态 |
|------|-----|---------------|--------|------------------|------|
| ScaleConditionedRF eps=2.0 (NEW) | 0.741 | −0.005 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| FBM Frozen | 0.737 | −0.009 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| FBM Enhanced | 0.703 | −0.043 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| FBM Frozen-Enhanced | 0.696 | −0.050 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| FBM Cross-Attention | 0.733 | −0.013 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| Flow Matching Detection | 0.823 | −0.033 | Dataset 2 | ldmdet-frontier-directions | ⛔ 证伪 |
| Cascade Head Count E2E | 0.684 | −0.172 | Dataset 2 | ldmdet-frontier-directions | ⛔ 证伪 |
| Adaptive Step | — | — | Dataset 2 | — | ⛔ 证伪 |
| Draft-Verify | — | — | Dataset 2 | — | ⛔ 证伪 |
| Head Early-Exit | — | — | Dataset 2 | — | ⛔ 证伪 |
| RoI Feature Cache | — | — | Dataset 2 | — | ⛔ 证伪 |
| Decoupled Head | 0.749 (best@ep74) | +0.003 (原记录 0.702/−0.044 错误) | Dataset 1 | ldmdet-ablation | ⚠ 证伪存疑 (单 seed + 训练中断, 待 3-seed 复核) |
| Morphology-Contrastive / Structured Prior / Deformable P1 / Class-Balanced | — | — | — | — | ⛔ 未启动废弃 |
| h_velocity_loss | — | — | Dataset 2 | ldmdet-frontier-directions | ⛔ CRASHED |
| FBM SimpleGate | 0.677 | — | Dataset 2 | few-shot-benchmark | ⛔ 失败 |
| Head Distillation 失败配置 (freeze_backbone=True) | 0.717 | −0.146 | Dataset 2 | ldmdet-head-distill | ⛔ 配置Bug (修复配置 0.860 有效, → LINEAGE) |
| ReFlow Standard MSE (方法本质失败, 重试确认) | 0.862@ep1(=A4) / 0.646(v1) | −0.001 / −0.217 | Dataset 2 | ldmdet-reflow | ⛔ 方法本质失败 (重试确认) |
| scale_aware_loss | 0.742 | −0.004 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| relative_l1_loss | 0.740 | −0.006 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| high_cls_weight | 0.739 | −0.007 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| high_giou_weight | 0.737 | −0.009 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| no_box_renewal | 0.730 | −0.016 | Dataset 1 | ldmdet-ablation | ⛔ 训练消融证伪 (⚠ 推理时切换 K≥200 安全, 见 LINEAGE §六) |
| class_balanced_sampling | — | — | Dataset 1 | ldmdet-ablation | ⛔ SIGKILL |
| proposals_100 | — | — | Dataset 1 | ldmdet-ablation | ⛔ SIGKILL |
| multi_seed/rf_heun_adaln (无 aug) | 0.712 | −0.034 | Dataset 1 | chromosome-kd | ⛔ 简化非标准 |
| multi_seed/rf_heun_adaln_bs4 | 0.702 | −0.044 | Dataset 1 | — | ⛔ 简化非标准 |
| multi_seed/hard_ot (无 aug) | 0.705 | −0.041 | Dataset 1 | chromosome-kd | ⛔ 简化非标准 |
| debug_train_final | 0.525 | −0.221 | Dataset 1 | — | ⛔ 调试 |
| stability/warm_restart_* | 0.714-0.728 | −0.018~-0.032 | Dataset 1 | chromosome-kd | ⛔ 证伪 |
| scheme_a_dinov2_s | 0.502-0.676 | −0.070~-0.244 | Dataset 1 | chromosome-kd | ⛔ 全失败 |
| scheme_E_bifpn | 0.744 | −0.002 | Dataset 1 | chromosome-kd | 🟠 持平, 未采用 |
| cspnext_l_rf_heun_adaln_stochot | 0.730 | −0.016 | Dataset 1 | chromosome-kd | ⛔ 证伪 |
| VLR (Velocity Lipschitz Regularization) | — | — | — | — | ⛔ 理论评审淘汰 (未实验, R1=3.0/10, 正则化目标与理想 RF 矛盾) |
| DSCR (Dahlquist-Stability Cascade Regularization) | — | — | — | — | ⛔ 理论评审淘汰 (未实验, R1=4.0/10, Dahlquist 范畴错误) |
| BDS-RF (Bounded-Differences Stability-Regularized RF) | — | — | — | — | ⛔ 理论评审淘汰 (未实验, R1=5.0/10, McDiarmid/Efron-Stein 适用对象错误) |
| PDR (Proposal Dependency Regularization) | — | — | — | — | ⛔ 理论评审淘汰 (未实验, R1=5.0/10, PAC-Bayes/HSIC/Janson 三大支柱均有问题) |
| LDCR (Large Deviation Cascade Regularization) | — | — | — | — | ⛔ 理论评审淘汰 (未实验, R1=5.6/10, K=6 不满足大偏差渐近性) |
| PCR-Matcher (Perturbation-Conditioned Robust Matcher) | — | — | — | — | ⛔ 理论评审淘汰 (未实验, R1=5.7/10, Renegar 条件数误用) |
| Traj-SAM (Trajectory-Integrated Sharpness-Aware Minimization) | — | — | — | — | ⛔ 理论评审淘汰 (未实验, R1=6.2/10, Cauchy-Schwarz 错误+训练翻倍) |
| MEC-RF (Modified-Equation Compensated RF) | — | — | — | — | ⛔ 冗余淘汰 (未实验, R2=6.5/10, 降为 BEAR fallback) |
| EXER-RF (Extrapolation Error Regularization) | — | — | — | — | ⛔ 冗余淘汰 (未实验, R1=7.4/10, 与 BEAR k=2 正则同一对象) |
| MDC-RF (Material Derivative Constraint) | — | — | — | — | ⛔ 冗余淘汰 (未实验, R1=7.0/10, 与 TFR 严格包含关系) |
| SC-RF (自条件化 RF) | 0.860 | −0.003 | Dataset 2 | ldmdet-breakthrough | 🟠 边际不采用 (自条件化在低维 RF 检测中价值有限, §二十三) |
| PD-RF (4→1 Progressive Distillation) | 0.851@ep1→0.252@ep28 | −0.011→−0.611 (崩塌) | Dataset 2 | — | ⛔ 灾难性崩塌 (1步 Euler 无法逼近 4步 DPM++, 蒸馏梯度冲突, §二十四) |

### 核心教训

1. **配置字段存在 ≠ 代码生效**: ScaleConditionedRF 的最大教训——dumped config 中含字段不代表 head.py 集成。后续所有"声称有效"的方向必须经代码级核查 (备份 head.py / commit diff / 单元测试)。
2. **推理时依赖 x0_pred 的决策都不可靠**: ScaleConditionedRF 缺陷 1 + Adaptive Step/Draft-Verify 证伪 + h_velocity_loss 崩溃, 三处独立证实 RF 在 t≈1 时模型输入近乎纯噪声, 任何基于 x0_pred 的决策 (尺度估计 / 提前终止 / draft-verify / v-prediction) 都不可靠。
3. **生成模型特征与检测任务空间不兼容**: FBM §二 + scheme_a_dinov2_s §十一 + FBM SimpleGate §九, 三处独立证实生成模型 (ChromoGen UNet / DINOv2) 的特征服务于像素级任务, 与 bbox 级检测任务特征空间不兼容, 简单注入反而有害。
4. **架构解耦方向边际收益不足**: Decoupled Head (数据更正: 真实 best 0.749, Δ=+0.003 在 noise 内, 原证伪结论存疑, 单 seed + 训练中断待 3-seed 复核) + Cascade Head Count E2E (−0.172 严重退化) + Head Early-Exit (退出率 0%), 后两者仍证实 cascade head 的共享特征路径和横向精化不可解耦; S1 理论分析 (theory_analysis_RF_DPM.md §2) 预测并解释了该现象。
5. **Dataset 1 架构天花板约 0.75**: Bottleneck 系列 §十 + Architecture Decoupling series §六/七 + 早期失败 §十一, 均未超越 0.746 baseline, 证实 Dataset 1 上架构天花板约 0.75。突破需换数据集 (Dataset 2, +Stoch. Coupling=0.859) 或换范式 (RF + DPM-Solver++), 这正是论文最终选择。
6. **box renewal 是推理时机制, 可完全移除 (K≥200)**: box_renewal 是原版 DiffusionDet 的推理时机制 (projects/DiffusionDet/diffusiondet/head.py), 不在 loss() 路径 (head.py:628 loss() vs :1186 predict()), 训练时 True/False 不影响模型权重。(1) no_box_renewal "训练消融" (Dataset 1, Heun, −0.016) 的退化根因是 **early stopping 选择偏差** (验证无 renewal → val mAP 波动 → 次优 checkpoint), 非 model 能力退化; (2) RoI Feature Cache 证伪 (box 位移 93-124 px/步) 证实 renewal 使 proposal 位置大幅变化; (3) D3 矛盾 (theory_analysis_RF_DPM.md §3) 证实 renewal 污染 η_str 诊断但不影响精度; (4) **推理时关闭 renewal 在 K≥200 下安全** (DPM-Solver++, 3-seed ΔmAP=−0.0003, LINEAGE §六 方案 B), DPM++ D1 校正自然连续, 不需要 D3 化解路径 A; K=100 推理关闭有 −0.031±0.012 退化 (3-seed, proposal 稀缺时回收机制价值凸显)。
7. **负面记录的价值**: 这些证伪方向证明最终设计 (RF + Heun/DPM-Solver++ + Stochastic Coupling + Top-K 剪枝 + 6 cascade head + box renewal) 的每个组件都经过充分验证, 排除了多个看似合理的替代方案, 支撑论文的方法选择合理性。
8. **理论评审可发现致命错误, 避免无效实验** (§十五~§二十一, 2026-07-28): 7 个方向 (VLR/DSCR/BDS-RF/PDR/LDCR/PCR-Matcher/Traj-SAM) 在 R1 A↔B 理论评审阶段即被淘汰, 未经真实实验。理论评审发现的错误类型包括: (a) 正则化目标与理想解矛盾 (VLR: $J_{v_\theta} \to 0$ vs 理想 $-I/t \neq 0$); (b) 理论范畴错误 (DSCR: Dahlquist 用于非 ODE 数值方法); (c) 适用对象错误 (BDS-RF: McDiarmid 用于确定性预测; PDR: PAC-Bayes $n$ 混淆训练图像数与 proposal 数); (d) 渐近性不满足 (LDCR: $K=6$ 不满足大偏差 $K \to \infty$); (e) 条件数误用 (PCR-Matcher: Renegar 混淆 A 与 c); (f) 核心推导错误 (Traj-SAM: Cauchy-Schwarz 多一个平方)。**理论评审在实验前过滤了 7 个方向, 节省了大量计算资源**。
9. **冗余性检查必须前置** (§二十二, 2026-07-28): MEC-RF/EXER-RF/MDC-RF 三个方向在 R1/R2 评审中才发现与已推荐方向 (BEAR/TFR) 高度冗余 — MEC-RF 是 BEAR 的子分量, EXER-RF (k=2) 与 BEAR 正则同一对象 $\|D_2\|^2$ 仅权重不同, MDC-RF 严格包含 TFR。后续方向设计时必须先绘制 "正则对象 vs 已有方向" 的覆盖图, 避免重复设计。
10. **理论包装与实际机制必须匹配** (§十五~§二十一): VLR 将真实速度 $J_{v^*}=0$ 错误迁移到网络速度 $J_{v_\theta}$; DSCR 将 Dahlquist ODE 稳定性用于学习映射; BDS-RF 将浓度不等式用于一致性正则化; PDR 将 PAC-Bayes 训练样本界用于 proposal 依赖。这些错误的共同特征是 **理论包装宏大但与实际机制不匹配**。后续方向必须验证理论前提与实际机制的对应关系, 不能仅凭文献引用包装。

---

<!-- 文档结束。本文档对应论文 Appendix B "被证伪的方向", 与 docs/EXPERIMENT_LINEAGE.md §十一 ScaleConditionedRF 证伪记录、docs/paper/theory_analysis_RF_DPM.md §1.6/§2.5/§4 理论分析交叉引用。 -->
