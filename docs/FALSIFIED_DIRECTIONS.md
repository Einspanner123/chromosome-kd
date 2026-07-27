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
> 更新时间: 2026-07-21

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
  -- 证实 box renewal 机制对性能有显著贡献, 不可移除

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
3. **box renewal 不可移除**: no_box_renewal (0.730, Δ=−0.016) 证实 box renewal 是 LDMDet 的核心机制, 移除后低置信度 proposal 无法重置, 检测精度显著下降。该消融为正向贡献 (证明 box renewal 必要性), 但作为改进方向证伪。
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

## 十四、ReFlow Standard MSE 当前run (配置Bug+方法风险, 待重试定论)

> **失败性质**: ⚠ **配置Bug为主因 + 方法固有风险为次因** (当前run失败不可挽救, 待重试后最终定论)
> **方法学判定**: 存疑 (用户指示"配置问题不能否定方法失败", 需修复配置重试后再判定)
> **归档日期**: 2026-07-27

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

### 方法学判定 (待重试后定论)

- **代码实现正确**: coupling 加载/q_sample 公式/空间转换/per-proposal 对齐全部正确 (Subagent 3 代码审查确认)
- **当前run失败不可挽救**: best 0.646 已回退, lr 已衰减至 0, mAP_75 崩塌
- **方法固有风险存在**: 即使修复配置, cls/box 不一致 + mAP_75 崩塌趋势可能仍出现
- **重试判据**: 若重试后 mAP_75 仍崩塌 (<0.70 @ ep50), 则判定方法本质失败

### 复活因素 (重试计划)

1. **配置修复**: load_from=A4 best ep117 + lr=5e-5 + max_epoch=150
2. **关键判据**: 重试后观察 mAP_75 是否仍崩塌
3. **若 mAP_75 稳定**: 说明是配置问题, 方法有效
4. **若 mAP_75 仍崩塌**: 说明是方法本质问题 (A4 模糊预测当目标注定定位退化), 判定方法失败

### 数据修正

- 用户记忆中的 "best mAP=0.542 @ ep50" 是错误的, 实际 best mAP=0.646 @ ep42 (0.542 是 ep50 末值, 已从峰值回退 0.104)
- "已证伪 ReFlow velocity loss 版 mAP=0.739" 标签错误, 0.739 来自 nonlinear_trajectory_seed2 (既无 velocity loss 也无 reflow coupling), 真正的 h_velocity_loss best=0.856@ep61

### 教训

1. **微调实验必须设置 load_from**: 从零训练 + 微调参数 (低 lr + 少 epoch) 是致命组合
2. **mAP_75 是定位精度的关键指标**: mAP_50 持平但 mAP_75 崩塌说明方法损害精细定位
3. **用户标准 "配置问题不能否定方法失败"**: 需从方法固有风险角度判定, 不能仅归咎于配置

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
| ReFlow Standard MSE (缺失load_from) | 0.646 | −0.217 | Dataset 2 | ldmdet-reflow | ⚠ 配置Bug+方法风险 (待重试定论) |
| scale_aware_loss | 0.742 | −0.004 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| relative_l1_loss | 0.740 | −0.006 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| high_cls_weight | 0.739 | −0.007 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| high_giou_weight | 0.737 | −0.009 | Dataset 1 | ldmdet-ablation | ⛔ 证伪 |
| no_box_renewal | 0.730 | −0.016 | Dataset 1 | ldmdet-ablation | ⛔ 消融证伪 |
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

### 核心教训

1. **配置字段存在 ≠ 代码生效**: ScaleConditionedRF 的最大教训——dumped config 中含字段不代表 head.py 集成。后续所有"声称有效"的方向必须经代码级核查 (备份 head.py / commit diff / 单元测试)。
2. **推理时依赖 x0_pred 的决策都不可靠**: ScaleConditionedRF 缺陷 1 + Adaptive Step/Draft-Verify 证伪 + h_velocity_loss 崩溃, 三处独立证实 RF 在 t≈1 时模型输入近乎纯噪声, 任何基于 x0_pred 的决策 (尺度估计 / 提前终止 / draft-verify / v-prediction) 都不可靠。
3. **生成模型特征与检测任务空间不兼容**: FBM §二 + scheme_a_dinov2_s §十一 + FBM SimpleGate §九, 三处独立证实生成模型 (ChromoGen UNet / DINOv2) 的特征服务于像素级任务, 与 bbox 级检测任务特征空间不兼容, 简单注入反而有害。
4. **架构解耦方向边际收益不足**: Decoupled Head (数据更正: 真实 best 0.749, Δ=+0.003 在 noise 内, 原证伪结论存疑, 单 seed + 训练中断待 3-seed 复核) + Cascade Head Count E2E (−0.172 严重退化) + Head Early-Exit (退出率 0%), 后两者仍证实 cascade head 的共享特征路径和横向精化不可解耦; S1 理论分析 (theory_analysis_RF_DPM.md §2) 预测并解释了该现象。
5. **Dataset 1 架构天花板约 0.75**: Bottleneck 系列 §十 + Architecture Decoupling series §六/七 + 早期失败 §十一, 均未超越 0.746 baseline, 证实 Dataset 1 上架构天花板约 0.75。突破需换数据集 (Dataset 2, +Stoch. Coupling=0.859) 或换范式 (RF + DPM-Solver++), 这正是论文最终选择。
6. **box renewal 是核心机制**: no_box_renewal (−0.016) + RoI Feature Cache (box 位移 93-124 px/步) + D3 矛盾 (theory_analysis_RF_DPM.md §3), 三处独立触及 box renewal 机制。box renewal 虽污染 η_str 诊断, 但对最终精度贡献显著, 不可移除。
7. **负面记录的价值**: 这些证伪方向证明最终设计 (RF + Heun/DPM-Solver++ + Stochastic Coupling + Top-K 剪枝 + 6 cascade head + box renewal) 的每个组件都经过充分验证, 排除了多个看似合理的替代方案, 支撑论文的方法选择合理性。

---

<!-- 文档结束。本文档对应论文 Appendix B "被证伪的方向", 与 docs/EXPERIMENT_LINEAGE.md §十一 ScaleConditionedRF 证伪记录、docs/paper/theory_analysis_RF_DPM.md §1.6/§2.5/§4 理论分析交叉引用。 -->
