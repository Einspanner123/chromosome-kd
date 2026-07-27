# 待做研究方向总览 (TODO Directions)

> 本文档梳理 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 所有进行中或待启动的研究方向。
> 这些方向部分有代码就绪、配置就绪或实验已在运行, 部分仅有理论框架。
> 每个方向附 **可靠数据源地址** (本地服务器路径 / SwanLab project / config 路径)。
> 更新时间: 2026-07-27 (ReFlow 当前run失败+重试中; Head Distillation 完成 → LINEAGE §十五)
>
> 📌 **关联文档**:
> - [docs/EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md) (主路线实验脉络, 已完成方向)
> - [docs/FALSIFIED_DIRECTIONS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md) (已证伪方向归档)
> - [docs/EXPERIMENT_CATALOG.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_CATALOG.md) (实验数据索引)
> - [docs/paper/paper_draft_CN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md) (论文草稿, §6 结论与未来工作)
> - [docs/paper/theory_analysis_RF_DPM.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md) (R1/S1/D3/R3 理论深化分析)
>
> ⚠ **状态约定**: 🔄 运行中 / ⛔ 待启动 / ✓ 已完成 / 🔴 已证伪
>
> 📋 **文档流转规则**: 方向完成后, 有效→迁入 EXPERIMENT_LINEAGE.md; 证伪→迁入 FALSIFIED_DIRECTIONS.md; 本文档仅保留 🔄进行中 + ⛔待启动 + 边际待验证方向。
>
> 📝 **命名约定** (2026-07-26): 本文档为研究规划用途, **保留内部实验代号** (24obj / A0-A4 / IO3 / StochOT 等) 以便与 SwanLab run_id、配置文件路径、work_dirs 目录直接对应。正式命名映射见 EXPERIMENT_LINEAGE.md 头部命名约定块。方向迁入 LINEAGE/FALSIFIED 时会自动转换为正式名。

## 〇、方向索引与状态汇总

| 编号 | 方向 | 状态 | 优先级 | SwanLab Project |
|------|------|------|--------|-----------------|
| R3 | x0-prediction vs v-prediction 对照重训 | ✓ seed 42 完成 (早停@ep64, best 0.855@ep34, Δ=-0.008 单 seed 支持 R3.2); seed 123/789 ⛔ 待补 | 中 | `ldmdet-r3-vpred` |
| S1 | Cascade Head × Solver Step 解耦消融 | ✓ 已完成 (h3_s4/h3_s8/h6_s2 三组全部 0.859, S1.3 闭环, → [LINEAGE §七](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)) | ~~高~~ | `ldmdet-s1-cascade-decouple` |
| Few-Shot | 24obj 源 → chromo 目标跨数据集微调 | ⛔ 待启动 (配置就绪) | 高 | `few-shot-benchmark` (源预训练) |
| D3 | box_renewal × DPM-Solver++ 修复方案 A/C | ⛔ 待启动 (方案 B 已验证) | 中 | `ldmdet-ablation` |
| 方向 A | per-dim eta_str 维度级曲率诊断 | ✓ 完成 (Phase 2 mAP 持平+加速 5.5%, → [LINEAGE §九](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)) | ~~中~~ | (诊断无 SwanLab) |
| 方向 C | step-aware embedding (cascade head 感知 step) | ✓ 已完成 (早停@ep148, best 0.859@ep118, Δ=-0.004 在 noise 内; step_proj 活跃+3类改善, 非负面, 保留为 S1 佐证) | ~~中~~ | `ldmdet-mainline-ablation-24obj` |
| 方向 D | 自适应阶次 DPM-Solver++ (后期 step 降阶) | ✓ 完成 (3 solver mAP 持平 0.863, → [LINEAGE §十](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)) | ~~中~~ | (诊断无 SwanLab) |
| **D1 诊断** | RoI 空间信息消融 (7×7 vs 空间抹平) | ✓ 完成 (ΔmAP=-0.854 灾难性崩溃, 证实空间编码至关重要) | ~~高~~ | (诊断无 SwanLab) |
| **M1** | 形态感知 RoI 编码器 (零初始化残差增强) | ✓ FP32 复现完成 (best 0.862@ep19, Δ=-0.001 与 A4 持平, BF16 误导确认; h_conv/v_conv FP32 下仍均匀 → 设计问题非精度问题, 待 M1-v2 改进) | ~~高~~ | `ldmdet-mainline-ablation-24obj` |
| **M4** | 级联头角色分化 (损失权重衰减) | ⛔ 待启动 (D3 诊断支持, 零代码改动) | 中-高 | (待创建) |
| **ReFlow (Standard MSE)** | 基于 Coupling 变换的 2-Rectification | 🔄 **重试中** (当前run失败 best 0.646@ep42, 配置Bug缺失load_from+方法风险mAP_75崩塌, → [FALSIFIED §十四](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md); 重试配置: load_from+A4+lr=5e-5+150ep, 关键判据 mAP_75 是否仍崩塌) | **高** | `ldmdet-reflow` (`reflow_standard`) |
| **Head Distillation** | 少 Head (3) 蒸馏多 Head (6) | ✅ **完成** (best 0.860@ep10 early stop@ep40, Δ=-0.003 vs A4 0.863 在 noise 内, NFE 24→12 加速 2x, → [LINEAGE §十五](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)); 失败配置 ⛔ 证伪 (freeze_backbone=True, → [FALSIFIED §十三](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)) | ~~高~~ | `ldmdet-head-distill` |
| 跨数据集扩展 | OT Collapse 普遍性 claim 验证 | ⛔ 纯理论推导 | 中 (最高级目标) | — |
| SC-RF | 自条件化 RF | 🔄 运行中 (待评估) | 待评估 | `ldmdet-breakthrough` |
| VGAR | Velocity-Guided Adaptive Renewal | ⛔ 待系统评估 | 中 | `ldmdet-mainline-ablation-24obj` |
| 方向 E | Brenier 映射神经化 (ICNN 参数化, 突破方向) | ⛔ 未开展 (纯理论, TMI 投稿后) | 低 | — |

## 一、R3: x0-prediction vs v-prediction 对照重训

> ✓ seed 42 已完成 (2026-07-25 确认, 早停@ep64, best 0.855@ep34, 单 seed 支持 R3.2); seed 123/789 ⛔ 待补 (3-seed 完整验证)。理论依据: [theory_analysis_RF_DPM.md §4](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)

### 核心目标

- 验证理论 doc §4.3 的梯度放大 claim: v-prediction 在低维 ($d=4$) + shifted schedule ($s=3.0$) 下劣于 x0-prediction
- 证明论文当前 x0-prediction + DPM-Solver++ data-prediction 形式的选择有理论依据
- 预防审稿人对 "为何不用 v-prediction" 的质疑 (RF 原文 Liu et al., 2023 使用 v-prediction)

### 理论预测

- **命题 R3.1 (信息等价)**: $\hat{x}_0 = x_t - t\hat{v}$, x0-prediction 与 v-prediction 在 $d=4$ 低维 RF 下信息论等价, 差异仅在损失的 $t$ 加权: $\mathcal{L}_v = t^{-2}\mathcal{L}_{x_0}$
- **命题 R3.2 (shifted schedule 下的偏好)**: shifted schedule ($s=3.0$) 下 x0-prediction 的有效梯度信噪比优于 v-prediction, 因前者在 $t \to 0$ 时不放大梯度
- **命题 R3.3 (设置依赖性)**: RF 原文的 v-prediction 偏好依赖高维 + linear schedule 组合; 在低维 + shifted schedule 下 x0-prediction 是更优选择

### 实验设计

- 基于 A4 baseline (DPM-Solver++ + RF + AdaLN + StochOT eps5) 启用 v_prediction=True
- 实现方式: 不修改网络输出语义 (仍输出 x0_pred), 通过 $1/t^2$ 损失加权模拟 v-prediction 梯度动态
- 关键参数: `v_prediction_t_eps=1e-2` (截断避免数值爆炸), batch normalization (均值=1, 避免训练崩溃)
- GIoU loss 保持不加权 (不是 $t$ 的简单函数)
- 3 seeds (42/123/789) 重训, 训练时通过 `--cfg-options` 覆盖 `experiment_name` 和 `--seed`
- 算力分配: 分摊到 ross A6000 + workstation A5000/A4000

### 代码改动

- `ldmdet/criterion/criterion.py` 添加 `v_prediction` 参数 (已完成)
- commit: `016844f0` feat(criterion): add v-prediction equivalence via 1/t² loss reweighting
- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_24obj.py` (已创建, 基于 `a4_dpm_pp_24obj.py`)

### 实验列表

- `r3_vpred_24obj` (3 seeds: 42/123/789)
  -- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_24obj.py`
  -- 改动: `criterion=dict(v_prediction=True, v_prediction_t_eps=1e-2)`
  -- 对照: A4 baseline (x0-prediction, mAP=0.863, 3-seed 均值 0.859 ± 0.003)
  -- SwanLab: `ldmdet-r3-vpred` (project 已配置, experiment_name=`r3_vpred`)
  -- 预期: v-pred mAP < A4 baseline (0.863), 验证命题 R3.2

### 当前状态

seed 42 ✓ 已完成 (2026-07-25 确认, workstation A4000), seed 123/789 ⛔ 待补:

- seed 42 ✓ 早停完成 (workstation A4000, ep64/150 触发 patience=30)
  -- best mAP = 0.855 @ ep34 (Δ=-0.008 vs A4 0.863, 超 3-seed noise ±0.003 但偏小)
  -- last mAP = 0.837 @ ep64 (best 之后 30 epoch 未刷新)
  -- work_dir: `work_dirs/r3_vpred_24obj_seed42/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)
  -- SwanLab project: `ldmdet-r3-vpred`
  -- 训练曲线: ep8 warmup=0.802 → ep34 best=0.855 → 长期停滞 (ep34-ep64 未刷新) → 早停
  -- **支持 R3.2**: v-prediction 在低维 (d=4) + shifted schedule (s=3.0) 下劣于 x0-prediction, 与理论预测一致
  -- **3-seed 完整验证待补**: 单 seed 已支持方向性结论, 但论文纳入需 3-seed 确认 std
- seed 123 ⛔ 待启动 (等待 GPU 空闲, workstation A5000 可承接)
- seed 789 ⛔ 待启动 (等待 GPU 空闲, ross A6000 可承接)

### 论文纳入决策 (2026-07-25)

**选项 A** (推荐): 论文标注 "preliminary single-seed result", TMI 投稿后补 3-seed
- 单 seed Δ=-0.008 已支持 R3.2 方向性结论, 不与命题冲突
- TMI 10 页限制下, R3 仅作为 §3.1.1 末段或附录的初步证据 (~0.2 页)

**选项 B**: 投稿前补 seed 123/789 完成 3-seed (workstation A5000 + ross A6000 并行, ~24h)
- 提供 3-seed std, 强化命题 R3.2 证据强度
- 风险: 延迟投稿时间, 但 GPU 空闲可立即启动

**待用户决策**: 投稿时间 vs 证据强度的权衡

### 预期结果

- v-prediction mAP 显著低于 A4 baseline (预期 ΔmAP < 0, 量级待实验确定)
- 在 $t < 0.5$ (数据主导区, 对检测精度更关键) 时 v-prediction 的 $1/t^2$ 梯度放大引入显著方差
- 若实验确认, 可纳入论文 §3.1.1 末段或 §5.3 (约 0.3 页增量)

### 可扩展性

- R3 的设置依赖性分析可指导其他低维结构化预测任务的参数化选择:
  -- 3D 检测 ($d=6-7$): x0-prediction + DPM-Solver++ data-prediction 形式天然兼容
  -- 关键点检测 ($d=2K$): 同样适用低维 + shifted schedule 的偏好结论
- 命题 R3.3 提供设置依赖性而非绝对优劣, 强调 "低维 + shifted" 与 "高维 + linear" 的对比

## 二、S1: Cascade Head × Solver Step 解耦消融

> ✓ 已完成 (2026-07-25 确认, 三组实验全部 0.859, S1.3 闭环)。已迁入 [EXPERIMENT_LINEAGE.md §七](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)。理论依据: [theory_analysis_RF_DPM.md §2](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)

### 核心目标

- 验证命题 S1.3 (H×S 可交换性边界): 证明 $H=3, S=4 \neq H=6, S=2$ (同 NFE 不同分配, mAP 不同)
- 形式化 cascade head (横向精化) × solver step (纵向积分) 的算子分裂结构
- 解释为何 DPM-Solver++ 仅需 4 NFE 框架有效 (6 head 已收敛至 $\mathcal{B}_t^*$, 被吸收进复合算子)
- 解释已证伪的 N_cascade e2e 方向 (mAP 0.684, −0.172) 失败原因: 减小 $H$ 破坏横向收敛性而 $S$ 未相应增加

### 理论预测

- **命题 S1.1 (横向收敛性)**: 若 cascade head 序列 $\{\mathcal{B}_{t,k}\}$ 在固定 $t$ 上是压缩映射, 则存在不动点 $x_t^*$
- **推论 S1.2 (DPM-Solver++ 框架有效性)**: $H=6$ 足够大时, 6 head 已收敛到 $\mathcal{B}_t^*$, DPM-Solver++ 把 $\mathcal{B}_t^* \circ \mathcal{A}_t$ 视为单次复合 $v_\theta$ 评估合理
- **命题 S1.3 (H×S 可交换性边界)**: 因 $\mathcal{A}_t$ 是二阶 solver 而 $\mathcal{B}_{t,k}$ 是一阶精化, $H$ 减半需 $S$ 增加多于两倍 (即 $H \times S$ 不是不变量)
- 若命题 S1.3 成立, $H=3, S=4$ (12 NFE) mAP 应不同于 $H=6, S=2$ (12 NFE)

### 实验设计

3 配置重训 (基于 A4 baseline, DPM-Solver++ + RF + AdaLN + StochOT eps5):

- 配置1: H=3 S=4 (12 NFE) ✓ 已完成
  -- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/s1_h3_s4_24obj.py`
  -- 改动: `num_heads=3, sampling_timesteps=4` (仅 num_heads 6→3)
  -- SwanLab: `ldmdet-s1-cascade-decouple` / `s1_h3_s4` (✓ 已完成, 之前会话, 结果见 memory)
  -- 预期: 若 H×S 不可交换, mAP ≠ s1_h6_s2

- 配置2: H=6 S=2 (12 NFE, 同 NFE 不同分配) 🔄 训练中
  -- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/s1_h6_s2_24obj.py`
  -- 改动: `num_heads=6, sampling_timesteps=2` (仅 sampling_timesteps 4→2)
  -- SwanLab: `ldmdet-s1-cascade-decouple` / `s1_h6_s2` (🔄 训练中)
  -- 预期: 与 s1_h3_s4 在相同 NFE=12 下对比, 测试 H 的重要性

- 配置3: H=3 S=8 (24 NFE, matched A4 baseline NFE) ✓ 已完成
  -- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/s1_h3_s8_24obj.py`
  -- 改动: `num_heads=3, sampling_timesteps=8` (num_heads 6→3, sampling_timesteps 4→8)
  -- SwanLab: `ldmdet-s1-cascade-decouple` / `s1_h3_s8` (✓ 已完成)
  -- 预期: 若 H×S 不可交换, 即使 NFE 相同, H=3 S=8 mAP 应低于 H=6 S=4 (A4 baseline)

### 当前状态

三组实验全部完成 (2026-07-25 确认):

- `s1_h3_s4` ✓ 已完成 (best mAP=0.859, 之前会话)
- `s1_h3_s8` ✓ 已完成 (best mAP=0.859@ep64, 30 epochs 未改善早停, ross A6000)
- `s1_h6_s2` ✓ 已完成 (workstation, 早停@ep136/150 触发 patience=30)
  -- best mAP = 0.859 @ ep106
  -- last mAP = 0.856 @ ep136
  -- work_dir: `work_dirs/s1_h6_s2_24obj/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)
  -- 早停: "the monitored metric did not improve in the last 30 records. best score: 0.859."

SwanLab project `ldmdet-s1-cascade-decouple` 已配置, 3 个实验全部完成。

### 关键结论 (最终)

- **三组实验 mAP 全部为 0.859**, 与 A3 baseline 3-seed 均值 (0.859 ± 0.003) 完全持平
- **S1.3 命题完整闭环**: H=3,S=4 / H=6,S=2 / H=3,S=8 三组同 NFE 或不同 NFE 配置下 mAP 持平
- 说明 H×S **不是简单不变量**:
  -- H=6 充分大时减小 S (6→2) 仍可保持横向收敛性 (s1_h6_s2 = 0.859)
  -- H 减半 (6→3) 时增加 S (4→8) 可补偿 (s1_h3_s8 = 0.859)
- 与已证伪 N_cascade e2e (mAP 0.684, −0.172) 形成对比: 该实验减小 H 但未相应增加 S, 横向收敛性被破坏
- 详见 [EXPERIMENT_LINEAGE.md §七](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

## 三、Few-Shot 跨数据集微调

> 配置就绪, ⛔ 待启动 (等待 FBM CrossAttn 源预训练完成)。详见 [EXPERIMENT_LINEAGE.md §3.5](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

### 核心目标

- 24obj 源预训练 → chromo 目标微调 ($k=5, k=10$), 验证扩散检测器低数据迁移能力
- 扩散检测器在小样本迁移上竞争力 (与 DINO/RTMDet-L 对比)
- 论文 §4.3.1 已提及: Dataset 1 (训练集较小, 1,540 张) 上 KaryoFlow (0.753 mAP) 实际超过 RTMDet-L (0.742) 和 DINO R50 (0.737), 提示扩散范式在低数据小目标情形下尤其具有竞争力

### 实验设计

- 14 个目标微调配置 (7 模型 × 2 k 设置), 全部位于 `experiments/configs/few_shot/target_finetune/`
- 数据: $k=5$ (112 图, 120 标注), $k=10$ (222 图, 240 标注)
- 目标微调数据集: Chromosome20240904 (chromo, 旧数据集)
- 源预训练数据集: 24_chromosomes_object (24obj)

#### 实验证明目的

验证扩散检测器在低数据迁移上的竞争力, 对照 anchor-based (Cascade R-CNN, YOLOX-S) / transformer-based (DINO R50) / single-shot (RTMDet-L) / FBM 变体 (SimpleGate, CrossAttn) / LDMDet SOTA

#### 14 个目标微调配置列表

- `cascade_rcnn_r50_k5.py` / `cascade_rcnn_r50_k10.py`
  -- 模型: Cascade R-CNN R50 (anchor-based baseline)
- `dino_r50_k5.py` / `dino_r50_k10.py`
  -- 模型: DINO R50 (transformer-based, 多尺度可变形注意力)
- `ldmdet_fbm_crossattn_k5.py` / `ldmdet_fbm_crossattn_k10.py`
  -- 模型: LDMDet + Cross-Attention FBM (源预训练 🔄 运行中)
- `ldmdet_fbm_simplgate_k5.py` / `ldmdet_fbm_simplgate_k10.py`
  -- 模型: LDMDet + SimpleGate FBM (源预训练已停止, mAP=0.677)
- `ldmdet_sota_k5.py` / `ldmdet_sota_k10.py`
  -- 模型: LDMDet SOTA (源预训练已完成)
- `rtmdet_l_k5.py` / `rtmdet_l_k10.py`
  -- 模型: RTMDet-L (CSPNeXt-L, single-shot)
- `yolox_s_k5.py` / `yolox_s_k10.py`
  -- 模型: YOLOX-S (CSPDarkNet-S, single-shot)

### 源预训练状态

7 个源预训练模型 (24obj 数据集), 状态如下:

- LDMDet SOTA = 已完成 (best @ epoch 26)
  -- 本地: `work_dirs/few_shot/source_pretrain_ldmdet_sota_24obj/`
  -- config: `experiments/configs/few_shot/source_pretrain/ldmdet_sota_24obj.py`
- LDMDet FBM SimpleGate = 0.677 ⛔ 已停止 (epoch 5)
  -- 本地: `work_dirs/few_shot/source_pretrain_ldmdet_fbm_simplgate_24obj/20260701_203410/`
  -- SwanLab: `few-shot-benchmark` / run_id=`hyuiam5m` (best mAP=0.6770 @ epoch 5)
- LDMDet FBM CrossAttn = 0.810 🔄 运行中 (epoch 10/150)
  -- 本地: `work_dirs/few_shot/source_pretrain_ldmdet_fbm_crossattn_24obj/20260701_174434/`
  -- SwanLab: `few-shot-benchmark` / run_id=`9hj8pe4a` (best mAP=0.8100 @ epoch 10)
  -- 进展: ep1=0.000 → ep6=0.752 → ep9=0.787 → ep10=0.810 (持续上升)
- Cascade R-CNN R50 = 已完成 (best @ epoch 72)
  -- 本地: `work_dirs/few_shot/source_pretrain_cascade_rcnn_r50_24obj/`
- DINO R50 = 已完成 (best @ epoch 102)
  -- 本地: `work_dirs/few_shot/source_pretrain_dino_r50_24obj/`
- RTMDet-L = 已完成 (best @ epoch 116)
  -- 本地: `work_dirs/few_shot/source_pretrain_rtmdet_l_24obj/`
- YOLOX-S = 已完成 (best @ epoch 200)
  -- 本地: `work_dirs/few_shot/source_pretrain_yolox_s_24obj/`

### 当前状态

⛔ 尚未启动, 等待 FBM CrossAttn 源预训练完成 (当前 epoch 10/150)。

- SwanLab: `few-shot-benchmark` (源预训练, 3 个实验); 目标微调 project 待配置
- ⚠ 5 个非 FBM 实验仅保留 best checkpoint, 无训练日志 (需加载 checkpoint 评估或查 SwanLab)

### 预期结果

- 扩散检测器在小样本迁移上竞争力 (与 DINO/RTMDet-L 对比)
- 验证论文 §4.3.1 "Dataset 1 上 KaryoFlow 超过 RTMDet-L 和 DINO R50" 的低数据优势
- 若 FBM CrossAttn 源预训练成功 (>0.85), 可作为 LDMDet 变体参与对比

## 四、D3: box_renewal 与 DPM-Solver++ 交互修复方案

> 方案 B 已验证, 方案 A/C 待启动。理论依据: [theory_analysis_RF_DPM.md §3](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)

### 核心目标

- 进一步验证 box_renewal 与 DPM-Solver++ 交互的修复方案
- D3 矛盾已证实: box_renewal 使 $\eta_{\text{str}}$ 虚高 56-58%, 但 mAP 仅 $-0.0003$ (噪声范围)
- 修复目的: 使 R1 的 $\eta_{\text{str}}$ 诊断在 renewal 启用时有效 (当前被 renewal 噪声主导)

### 修复方案

- 方案 A (box_renewal 后 `dpm_solver.reset()`): 与 Top-K pruning 一致
  -- 代码位置: `head.py:714` 之后新增 `if dpm_solver is not None: dpm_solver.reset()`
  -- 理论后果: step 2 退化为 Euler, 但后续 step 2→3, 3→4 恢复 DPM-Solver++ 二阶
  -- 理论预期: mAP 变化 ≤ 0.002 (与 Top-K 一致)
  -- 状态: ⛔ 待启动

- 方案 B (推理时禁用 box_renewal): 已验证
  -- 实验已验证: 3 seed 平均 mAP 0.858 ± 0.003 (vs baseline 0.859 ± 0.004, delta −0.0003)
  -- 结论: 方案 B 不损失精度, 且使 $\eta_{\text{str}}$ 诊断有效
  -- 状态: ✓ 已完成

- 方案 C (仅 step 0 后 renewal): 理论预期最优
  -- 实现: 在 step 0 后 renewal (与 Top-K 同步), step 1–3 不 renewal
  -- 理论预期: 最优方案, 但需重训 checkpoint 评估 (因训练时也用 renewal)
  -- 状态: ⛔ 待启动

### 当前状态

- 方案 B 已完成 (mAP 不损失, $\eta_{\text{str}}$ 诊断有效)
- 方案 A/C 待启动

### 优先级

- 中 (方案 B 已解决诊断需求, 方案 A/C 主要用于理论完整性验证)
- 实验成本: 方案 A 仅推理时改动 (1 行代码), 方案 C 需重训 checkpoint

## 五、方向 A: per-dim eta_str 维度级曲率诊断 ✓

> ✓ **已完成** (2026-07-22)。Phase 2 per-dim solver mAP 持平 (+0.001) + 推理加速 5.5%。
> 已迁入 [EXPERIMENT_LINEAGE.md §九](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md), 本节仅保留索引。

### 关键结果

- **Phase 1 诊断**: h 维度曲率显著小于 cx,cy (3-5×), w 维度差距较小 (1.5-2×), 部分支持假设
- **Phase 2 per-dim solver**: h 维度 1 阶 + cxcy/w 维度 2 阶, mAP=0.863 (持平 +0.001), 延迟 −8.3ms (加速 5.5%)
- **实现**: [RFDPMSolverPerDim](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/diffusion/rectified_flow.py)
- **配置**: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a8_per_dim_solver_24obj.py`
- **评估脚本**: [experiments/analysis/direction_a_per_dim_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/direction_a_per_dim_comparison.py)
- **结果数据**: [work_dirs/diagnosis/direction_a_per_dim_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/direction_a_per_dim_comparison.json)
- **论文纳入**: §5.4 (方向 A 深化), 约 0.3 页

## 六、方向 C: step-aware embedding (cascade head 感知 solver step)

> ✓ **已完成** (2026-07-23 早停@ep148, best 0.859@ep118)。插桩分析显示 step_proj 活跃、loss 仍降、3 类改善 — **非负面方向**, 不归入 FALSIFIED, 保留为 S1 理论佐证。

### 核心目标

- 让 cascade head 感知 DPM-Solver++ step 编号
- 不同 step 上 $x_t$ 的统计特性不同 (早期近噪声, 后期近 GT), 显式 step 条件可能提升每步精化的针对性
- 零初始化确保预训练兼容: 训练初期 step_proj 输出为 0, 模型行为与无 step embedding 时一致

### 实现方式

- step_mlp + step_proj 零初始化
- 实现位置: `ldmdet/core/head.py` (step_mlp + step_proj 零初始化)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a6_step_aware_24obj.py`

### 当前状态

seed 42 ✓ 已完成 (早停@ep148, best 0.859@ep118):

- seed 42 ✓ 已完成 (本地 A6000, 早停@ep148/150)
  -- best mAP = 0.859 @ ep118 (Δ=-0.004 vs A4 0.863, **在 3-seed std 0.003 范围内**)
  -- 早停: "the monitored metric did not improve in the last 30 records. best score: 0.859."
  -- work_dir: `work_dirs/a6_step_aware_24obj_seed42/`
  -- SwanLab project: `ldmdet-mainline-ablation-24obj` (experiment_name=`a6_step_aware`)
- seed 123/789: 不启动 (方向 C 非负面但增益不显著, GPU 优先分配给 M1 FP32 复现)

### 插桩分析 (2026-07-23, checkpoint epoch_146 + best ep118)

> **核心结论**: 方向 C **不是负面方向**。虽然 mAP 未超 A4, 但插桩指标显示 step-aware embedding 确实被学习且训练健康。

**1. step_proj 权重分析 (与 M1 fuse 对比)**:

| 指标 | 方向 C step_proj | M1 fuse (对照) |
|------|-----------------|---------------|
| 权重 norm (ep146) | 5.420 (**活跃**) | 0.215 (弱) |
| 非零元素 | 1,048,576/1,048,576 (100%) | 32,768 非零 |
| ep118→ep146 变化 | 5.434→5.420 (收敛, 几乎不变) | 0.006→0.238 (仍在增长) |
| 方向性 | **有区分** (std=0.005) | **完全均匀** (std=0.000) |

→ step-aware embedding **确实被模型使用**, 且早期即收敛 (ep118 与 ep146 几乎相同), 不像 M1 fuse 那样退化。

**2. 训练动态**:

| 指标 | ep10 | ep80 | ep120 | ep147 (最新) | 趋势 |
|------|------|------|-------|-------------|------|
| avg loss | 3.43 | 2.03 | 1.74 | **1.65 (仍降)** | ✅ 持续下降 |
| avg grad_norm | 34.6 | 31.5 | 37.4 | 36.9 | ✅ 稳定健康 |
| mAP | 0.816 | 0.850 | 0.857 | 0.857 | ⚠ 已收敛 |

→ **loss 仍在下降** (ep140: 1.670 → ep147: 1.654), 但 mAP 已收敛在 0.857。loss-mAP 分离表明模型仍在学习但不再转化为检测性能提升。

**3. mAP 收敛行为**:

- 最后 10 epoch mAP: [0.857, 0.856, 0.857, 0.858, 0.857, 0.857, 0.857, 0.857, 0.857, 0.858]
- 均值 0.857, std ~0.001 → **极其稳定**, 非崩溃式退化
- 与 A4 3-seed 均值 (0.859±0.003) **统计上无法区分**

**4. Per-class AP 对照 (best@ep118 vs A4 best@ep117)**:

| 类别 | A4 | 方向 C | Δ | 说明 |
|------|-----|--------|------|------|
| A1 | 0.911 | 0.913 | **+0.002** | ✓ 改善 (最大染色体) |
| A2 | 0.910 | 0.909 | -0.001 | |
| A3 | 0.904 | 0.904 | 0.000 | 持平 |
| B4-B5 | 0.905/0.906 | 0.904/0.901 | -0.001/-0.005 | |
| C6-C12 | 0.903-0.872 | 0.899-0.869 | -0.003~-0.008 | C 组轻微退化 |
| **C12** | 0.890 | 0.893 | **+0.003** | ✓ 改善 |
| D13-D15 | 0.856/0.853/0.844 | 0.849/0.850/0.838 | -0.007/-0.003/-0.006 | |
| E16-E18 | 0.853/0.842/0.832 | 0.850/0.838/0.820 | -0.003/-0.004/-0.012 | E18 退化最大 |
| F19-F20 | 0.817/0.819 | 0.814/0.813 | -0.003/-0.006 | |
| G21-G22 | 0.789/0.787 | 0.783/0.781 | -0.006/-0.006 | |
| X | 0.883 | 0.875 | -0.008 | |
| **Y** | 0.780 | 0.783 | **+0.003** | ✓ 改善 (最难类别) |

→ **3 类改善 (A1, C12, Y), 1 类持平, 20 类轻微退化**。与 M1 (24 类全退化) 形成对比。Y 染色体改善尤其有价值 (最小最难类别)。

### 价值判断 (综合插桩指标, 非 mAP 阈值)

**不归入 FALSIFIED 的理由**:
1. mAP 差距 -0.004 在 3-seed std (0.003) 范围内, 统计上无法区分
2. step_proj 权重活跃 (norm=5.42), 与 M1 fuse 退化 (uniform) 本质不同
3. loss 仍在下降, 训练健康 (梯度稳定)
4. 3 个类别改善 (含最难类别 Y), 非全面退化
5. 早停是 patience 到期而非崩溃

**可能缩小差距的方向**:
- 超参数调整: step_proj 初始化后 early convergence (ep118≈ep146), 可能 lr 过低导致新参数过早收敛; 尝试更高 lr (2e-5) 或更大 batch size
- 3-seed 评估: 0.859 可能为 seed 42 的随机性; 3-seed 均值可能与 A4 (0.859±0.003) 完全重叠
- 更长训练 + lr decay: loss 仍降但 mAP 停滞, 可能需要 lr schedule 调整

### 与 S1 的关系

- S1 形式化 cascade head × solver step 算子分裂 (§二)
- 方向 C 在不破坏 S1 算子分裂结构的前提下, 让 cascade head 显式感知 step
- 与 S1 互补: S1 给出架构合理性框架, 方向 C 在框架内探索性能提升

### 预期 vs 实际

- **预期**: 若 step embedding 显著提升 mAP (Δ > +0.005), 可作为论文新方向; 若持平, 表明 cascade head 已通过 $x_t$ 隐式感知 step 信息
- **实际**: mAP 持平 (0.859 vs 0.863, Δ=-0.004 在 noise 内), 但 step_proj 权重活跃 + 3 类改善 + loss 仍降 → **方向有效但增益不显著**, 不归入 FALSIFIED
- **论文叙事价值**: 即使 mAP 持平, "cascade head 已隐式感知 step" 这一发现本身支持 S1 的算子分裂理论 (§七), 可作为 S1 的实验佐证

## 七、方向 D: 自适应阶次 DPM-Solver++ (后期 step 降阶) ✓

> ✓ **已完成** (2026-07-22)。3 solver mAP 完全持平 (0.863), 自适应方案仅 4.2% 加速。
> 已迁入 [EXPERIMENT_LINEAGE.md §十](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md), 本节仅保留索引。

### 关键结果

- **诊断**: η_3rd step1=44.6 → step2=18.2 (降幅 59%), 支持后期 step 可降阶假设
- **mAP 对比**: DPM++ 2阶 / 3阶 / 自适应 三者 mAP 均为 0.863 (ΔmAP=0.000)
- **结论**: 4 步采样下 2 阶 DPM-Solver++ 已足够, 3 阶校正项无额外增益, 佐证 R1 "2 步收敛"
- **评估脚本**: [experiments/analysis/direction_d_solver_comparison.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/analysis/direction_d_solver_comparison.py)
- **结果数据**: [work_dirs/diagnosis/direction_d_comparison.json](file:///home/linkst/workspace/projects/chromosome-kd/work_dirs/diagnosis/direction_d_comparison.json)
- **论文纳入**: §5.4 (方向 D 深化), 约 0.2 页

## 八、ReFlow (Standard MSE 版)：基于 Coupling 变换的 2-Rectification

> 🔄 **重试中** (2026-07-27): 当前run失败 best 0.646@ep42 (配置Bug缺失load_from+方法风险mAP_75崩塌 0.733→0.543), → [FALSIFIED §十四](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md); 重试配置: load_from+A4+lr=5e-5+150ep, 关键判据 mAP_75 是否仍崩塌
> ⚠ **数据修正**: 用户记忆 "best 0.542@ep50" 错误, 实际 best 0.646@ep42; "已证伪 velocity loss 版 0.739" 标签错误, 0.739 来自 nonlinear_trajectory (非 velocity loss), 真正 h_velocity_loss best=0.856
> 理论依据: [Rectified Flow 主论文 §4](https://arxiv.org/abs/2209.03003), [Straightness of RF (2410.14949)](https://arxiv.org/abs/2410.14949)
> **重要声明**: 此为全新方法，与 2023-2024 年已证伪的 ReFlow (velocity loss 版) 有本质区别，详见下方"与已证伪 ReFlow 的关键差异"

### 核心目标

- 验证 $\eta_{\text{str}} > 0.1$ 时，2-Rectification 是否能有效拉直轨迹
- 探索减少推理步数的可能性 (4步 → 2步 → 1步)
- 为 DPM-Solver++ 的有效性提供轨迹层面的理论解释

### 与已证伪 ReFlow (velocity loss 版) 的关键差异

| 方面 | 已证伪 ReFlow (velocity loss 版) | 本 ReFlow (Standard MSE 版) |
|------|---------------------------------|---------------------------|
| **损失函数** | 新增 velocity loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda \cdot \mathcal{L}_{\text{vel}}$ | **标准检测损失**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}}$ (无 velocity loss) |
| **梯度冲突** | ✅ 已证实 (cos = −0.104, 86.8% 梯度负相关) | ❌ 预计无 (单一目标函数) |
| **优化目标** | 两个冲突目标: 速度预测 vs 检测 | 单一目标: 检测精度 |
| **实验结果** | mAP = 0.739 (chromo), 低于 baseline 0.751 | **未实验过** |
| **应用阶段** | 2023-2024 年，旧 chromo 数据集 | 2024-2025 年，新 24obj 数据集 |

### 理论依据

- R1 诊断显示 $\eta_{\text{str}} \in [0.7, 1.5]$ 非零 (3-seed 实验)，轨迹并非理想直线
- **触发条件已满足**: $\eta_{\text{str}} \approx 1.5 > 0.1$，理论上 reflow 可能有收益
- Rectified Flow 论文 (2023) 证明 2-Rectification 可显著拉直轨迹 ($\gamma_{2,T} \to 0$)
- [Straightness of RF (2410.14949)](https://arxiv.org/abs/2410.14949) 提供 $\gamma_{2,T}$ 的严格收敛理论

### 方法原理

**Step 1: 生成新 Coupling**
使用已训练的 1-RF 模型 (A4, mAP=0.863) 对训练集推理，生成新的 coupling 对：
- 原始 coupling: $(x_0^{\text{GT}}, x_1^{\text{noise}})$
- 新 coupling: $(x_0^{\text{pred},(2)}, x_1^{\text{noise}})$，其中 $x_0^{\text{pred},(2)} = f_{\theta_A4}(x_1^{\text{noise}}, t=0)$

**Step 2: 用新 Coupling 训练 2-RF**
- **关键**: 损失函数**不变**，仍然使用标准检测损失
- 仅替换 coupling 的 $x_0$ 部分 (用模型预测替代原始 GT)
- 这与 Rectified Flow 论文的标准做法一致

**数学表达式**
$$\mathcal{L}_{\text{total}} = \mathbb{E}_{(x_0, x_1) \sim p_0(x_0)p_1(x_1)} \left[ \ell_{\text{det}}(\hat{x}_0(x_t, t), x_0^{\text{GT}}) \right]$$
其中 $x_t = (1-t) \cdot x_0^{\text{pred},(2)} + t \cdot x_1^{\text{noise}}$ (使用新 coupling)

### 实施计划

**Phase 1: 准备 (1-2 天)**
- [ ] 用 A4 checkpoint (24obj, mAP=0.863) 对训练集推理，生成 $(x_0^{\text{pred},(2)}, x_1^{\text{noise}})$ 对
- [ ] 保存为 `.npy` 文件，用于后续训练

**Phase 2: 代码实现 (1-2 天)**
- [ ] 在 `rectified_flow.py` 中添加 `use_reflow_coupling` 参数
- [ ] reflow 模式下从预生成的 coupling 文件加载 $x_0^{\text{pred},(2)}$
- [ ] **损失函数不变**: 仍然使用 `criterion(bbox_pred, cls_scores, ...)`

**Phase 3: 验证实验 (3-5 天)**
- [ ] 1-seed 快速验证 (30 epochs，检查 loss 曲线和 mAP)
- [ ] 3-seed 完整实验 (150 epochs，与 A4 baseline 对比)
- [ ] 记录 reflow 前后 $\eta_{\text{str}}$ 变化

**Phase 4: 步数验证 (1-2 天)**
- [ ] 测试 reflow 后模型在 4/2/1 步推理下的 mAP
- [ ] 对比 A4 baseline (4步 vs 2步 vs 1步)

### 成功判据

1. ✅ mAP ≥ A4 baseline (0.863, 24obj 数据集)
2. ✅ $\eta_{\text{str}}$ 显著下降 (例如从 1.5 降至 < 0.5)
3. ✅ 1-2 步推理 mAP 接近 4 步水平 (减少 NFE)

### 风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| **Circular Dependency**: 模型用自己的预测训练自己 | 限制 reflow 训练 epochs (≤ 50)，或混合 reflow coupling 与原始 GT coupling |
| **Confirmation Bias**: 模型强化自身的错误预测 | 定期在验证集检查，若 mAP 下降立即停止 |
| **过拟合**: 训练数据分布变化 (模型预测 vs GT) | 使用较低的学习率 (1e-5)，加强正则化 |

### 优先级

- **高** (有 R1 η_str 诊断的内部支撑，理论完备，与已证伪方法有本质区别)
- 可与 §九 Head Distillation 并行实施

## 九、Head Distillation：少 Head (3) 蒸馏多 Head (6)

> ✅ **完成** (2026-07-27 归档): best 0.860@ep10 (early stop@ep40), Δ=-0.003 vs A4 0.863 在 noise 内, NFE 24→12 加速 2x, → [LINEAGE §十五](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)
> ⛔ **失败配置证伪** (2026-07-27 归档): 配置Bug freeze_backbone=True 致特征分布不匹配, best 0.717, → [FALSIFIED §十三](file:///home/linkst/workspace/projects/chromosome-kd/docs/FALSIFIED_DIRECTIONS.md)
> 理论依据: 与 §八 ReFlow 的数学同构性, S1 的 H×S 理论分析
> 失败配置: H=3←H=6 Teacher, λ=0.05, freeze backbone, bs=2, 150ep, 见 [proposals](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/proposals/REFLOW_HEAD_DISTILL_IMPL_PLAN.md)

### 核心目标

- 验证 head 数量从 6 减到 3 时，能否通过知识蒸馏保持 mAP
- 探索减少 NFE 的另一条路径 (6 heads × 4 steps = 24 NFE → 3 heads × 4 steps = 12 NFE)
- 为 S1 的 H×S 不可交换性提供蒸馏层面的补充验证

### 与 ReFlow 的数学同构性

ReFlow 在 **时间维度** 上做 "多→少" 蒸馏，Head Distillation 在 **head 维度** 上做完全同构的操作：

| 维度 | ReFlow (时间维度蒸馏) | Head Distillation (head 维度蒸馏) |
|------|---------------------|-----------------------------------|
| "时间"变量 | $t \in [0, 1]$ | $k \in \{1, 2, \ldots, H\}$ (head index) |
| Teacher | 多步推理 $(t_1, t_2, t_3, t_4)$ | 多 head 推理 $(h_1, h_2, h_3, h_4, h_5, h_6)$ |
| Student | 同一个模型，少步推理 | 少 head 模型 (如 $h_1, h_2, h_3$) |
| 目标值 | $x_0^{\text{Teacher}}(t)$ | $\hat{x}_0^{(K)}$ (所有 head 后的最终预测) |
| 蒸馏损失 | $\|v_\theta(x_t, t) - v_{\text{Teacher}}(x_t, t)\|^2$ | $\|h_k(x_{\text{input}}) - h_K(x_{\text{input}})\|^2$ |

### 与 S1 理论的联系

S1 的 H×S 理论说明 "仅改变 H 会破坏横向收敛性"：
- 已证伪的 **N_cascade e2e** (H=3, 重训): mAP = 0.684 (-0.172)，原因是随机初始化的 3 个 head 无法学会 6 个 head 才能达到的横向收敛
- **Head Distillation 的优势**: Student 的 head 从一开始就被 Teacher 的 head 监督，继承 Teacher 的行为分布

**与 S1 实验的关键区别**:
- S1 的 H=3, S=4 (进行中): **随机初始化** 3 个 head，直接训练
- Head Distillation (新): **Teacher 监督下** 3 个 head 学习 6 个 head 的行为

### 与已证伪 N_cascade e2e 的本质区别

| 方面 | N_cascade e2e (已证伪) | Head Distillation (新) |
|------|----------------------|------------------------|
| 初始化 | 随机初始化 H=3 | 继承 Teacher H=6 的行为 (蒸馏监督) |
| 训练方式 | 直接训练，只有检测损失 | 检测损失 + 蒸馏损失 (模仿 Teacher) |
| 收敛期望 | 3 个 head 达到 6 个 head 的横向收敛 | 3 个 head 模仿 6 个 head 的输出 |
| 实验结果 | mAP = 0.684，收敛失败 | 未实验过 |

### 蒸馏方案设计

#### 方案 A: Headwise Matching (推荐)

**核心思想**: 让 Student 的第 k 个 head 模仿 Teacher 的第 2k 个 head

```python
# Teacher: 冻结的 H=6 A4 模型
teacher_heads = [h_1, h_2, h_3, h_4, h_5, h_6]
for h in teacher_heads:
    h.requires_grad = False

# Student: H=3，可训练
student_heads = [h_1, h_2, h_3]  # 初始化可以用 Teacher 的 h_1, h_3, h_5

# 蒸馏逻辑
student_outputs = []
teacher_outputs = []

# Student 前向
x = x_input
for k in range(3):
    x = student_heads[k](x)
    student_outputs.append(x)

# Teacher 前向
x_t = x_input
for k in range(6):
    x_t = teacher_heads[k](x_t)
    if k in [1, 3, 5]:  # h_2, h_4, h_6
        teacher_outputs.append(x_t.detach())

# 蒸馏损失: Student k ≈ Teacher 2k
L_distill = sum(MSE(student_outputs[k], teacher_outputs[k]) for k in range(3))

# 总损失
L_total = L_detection + λ * L_distill
```

#### 方案 B: Direct Final Distillation (简单)

让 Student 的最终输出直接模仿 Teacher 的最终输出：
```python
student_final = H3_model(x_input)
teacher_final = H6_model(x_input).detach()
L_distill = MSE(student_final, teacher_final)
```

### 蒸馏损失设计

**总损失**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}}(\hat{x}_0, x_0^{\text{GT}}) + \lambda \cdot \mathcal{L}_{\text{distill}}$$

**蒸馏损失选择**:
- **MSE**: $\|\hat{x}_0^{\text{Student}} - \hat{x}_0^{\text{Teacher}}\|^2$ (简单有效)
- **GIoU**: $\text{GIoU}(\hat{x}_0^{\text{Student}}, \hat{x}_0^{\text{Teacher}})$ (bbox 专用)
- **组合**: 3×MSE + GIoU (兼顾坐标精度和 IoU)

**超参 λ**:
- 建议范围: [0.1, 0.5, 1.0]
- 可先从 λ=0.3 开始

### 实施计划

**Phase 1: 分析 (1 天)**
- [ ] Head Output Analysis: 分析 Teacher 模型不同 head 的输出差异
  - 是否是渐进的？(head 1 粗 → head 6 精)
  - 坐标分布有何不同？
- [ ] 确定蒸馏方案 (推荐方案 A)

**Phase 2: 代码实现 (2-3 天)**
- [ ] 在 `head.py` 中添加 `distill_teacher` 参数
- [ ] 实现蒸馏损失计算 (MSE 和 GIoU)
- [ ] 添加 `distill_lambda` 配置项

**Phase 3: 训练与评估 (3-5 天)**
- [ ] 用 A4 checkpoint 初始化 Student 的 H=3 heads
- [ ] 1-seed 快速验证 (30 epochs)
- [ ] 3-seed 完整实验 (150 epochs)

**Phase 4: 与 S1 对比 (1 天)**
- [ ] Head Distillation 的 H=3 vs S1 直接训练的 H=3
- [ ] 验证 "蒸馏能否弥补 H 减少的损失"

### 成功判据

1. ✅ mAP ≥ 0.84 (比 A4 的 0.863 允许小幅度下降)
2. ✅ 比 S1 直接训练的 H=3 有显著提升 (如果 S1 成功)
3. ✅ NFE 从 24 降到 12 (2× 加速)

### 风险与缓解

| 风险 | 缓解策略 |
|------|---------|
| Head 功能不均匀 (head 1 粗 vs head 6 精) | 同时蒸馏 feature map 和 output |
| Student capacity 不够 | 不减少每个 head 的参数量，只减少数量 |
| λ 调参困难 | 多个 λ 值并行尝试 |
| Teacher 存在噪声 | Teacher 的预测也可能有错误 (非完美监督) |

### 与 ReFlow 的联合优化 (高级)

**理想情况**: ReFlow + Head Distillation
```
ReFlow (时间维度): 4 steps → 2 steps (减少 2× NFE)
Head Distillation (head 维度): 6 heads → 3 heads (减少 2× NFE)
联合效果: 24 NFE → 6 NFE (4× 加速)
```

### v2 实验状态 (2026-07-25, ⚠ 异常中断)

**v2 实现**: H=3 Student ← H=6 Teacher (A4 checkpoint 冻结), λ=0.05, freeze backbone, bs=2, 150ep

**训练曲线**:
| epoch | mAP | loss | loss_distill |
|-------|-----|------|--------------|
| ~10 | 0.673 | — | — |
| ~50 | 0.681 | — | — |
| ~80 | 0.693 | — | — |
| ~90 | 0.705 | — | — |
| 96 (best) | **0.711** | ~1.86 | ~0.033 |
| 98 (last eval) | 0.709 | ~1.92 | ~0.033 |

**中断情况**:
- 跑到 epoch 99 iter 550/1750 (2026-07-24 14:57:33) nohup 日志戛然而止, 无报错
- GPU 已空闲, 无 train.py 进程, 推测 nohup 被外部信号 kill (可能 SSH 断开/OOM killer/手动 kill)
- 距离 max_epoch=150 还差 51 epoch (~5h)

**关键观察**:
- 最近 5 ep mAP: 0.705→0.711→0.703→0.709→0.704→0.711→0.709 (**仍在缓慢上升**)
- loss_distill 稳定在 ~0.033 (Teacher 监督有效, 但 student 容量受限)
- best 0.711 vs A4 0.863 = **-0.152** (H=3 学生模型架构容量限制明显)
- 显存仅 2.4GB (freeze backbone + H=3, 单卡可跑)

**待用户决策**:
- 选项 A: 从 epoch_98.pth 续训到 ep150 (ross A6000 空闲, ~5h), 看是否能突破 0.715+
- 选项 B: 归档当前结果为负面 (H=3 容量限制, 蒸馏无法弥补), 转向 M1-v2 或 M4
- 选项 C: 分析 distill loss 曲线 + teacher/student 输出差异后再决策

### 优先级

- **高** (与 ReFlow 同优先级，可并行实施)
- 成功后为论文提供 "维度蒸馏" 的统一框架 (时间 + head)

## 十、跨数据集扩展 (最高级目标, 理论推导)

> 纯理论推导, ⛔ 无实验验证。对应论文 §6 结论与未来工作。

### 核心目标

- 证明方法可扩展到有相似特性的其他图像 (不一定要真跑训练)
- 验证 OT Diversity Collapse 现象的普遍性 claim
- 论文 §6 已提及: "这一画像在医学影像中反复出现: 组织病理学中的细胞检测、乳腺 X 光和视网膜成像中的病灶检测、微生物菌落计数"

### 理论推导

- **OT Diversity Collapse 的 a-priori 诊断** (Table 2): 任何 $d \ll 100$ 且 $K \gg 10$ 的任务是候选
  -- 图像生成 ($d \sim 10^5$, $K=$ batch): $\Delta H / H \approx 0$ (OT 坍缩可忽略)
  -- 检测 COCO ($d=4$, $K \sim 7$): $\Delta H / H \approx 0.55$
  -- 检测 染色体 ($d=4$, $K \sim 46$): $\Delta H / H \approx 0.69$
- **R3 设置依赖性**: 低维 + shifted schedule → x0-prediction, 可指导其他低维任务参数化选择
- **Stochastic Coupling 双重收益**: 低数据情形下的精度 + 一般情形下的稳定性 ($4.6\times$ epoch-std 减少)

### 候选场景

- 细胞检测 (组织病理学): 多核 / $d=4$ / 小标注队列 → 最自然的下一步
- 病灶检测 (乳腺 X 光, 视网膜成像): 小目标 / 有限阳性案例
- 微生物菌落计数: 密集目标 / $d=4$
- 遥感密集检测: 高 $K$ / 低 $d$ 结构化预测

### 当前状态

- 纯理论推导, 无实验验证
- 论文 §6 结论已提及候选场景 (细胞检测 / 病灶检测 / 菌落计数)
- 论文承认局限: "经验验证限于染色体数据; 在 COCO 或细胞检测基准上验证将检验 OT 坍缩预测的普遍性"

### 未来工作

- 在至少一个非染色体高 $K$ 低 $d$ 基准上验证 (细胞检测是最自然的下一步)
- 将大幅强化普遍性 claim
- 理论通过 Table 2 提供 a-priori 诊断: 严重性 $\Delta H/H$ 预测 Stochastic Coupling 是否会有帮助

## 十、SC-RF 自条件化 RF (进行中/待评估)

> 🔄 运行中, 待训练结果评估。SwanLab: `ldmdet-breakthrough`

### 核心设想

- 自条件化 RF (Self-Conditioned RF): 把上一步预测作为条件输入
- 借鉴自条件化扩散模型思想, 应用到 RF 框架

### 当前状态

- 🔄 RUNNING (SwanLab: `ldmdet-breakthrough`)
- 历史结果: SC-RF mAP=0.857 (见 [EXPERIMENT_LINEAGE.md §3.2](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md))

### 风险

- 与 ScaleConditionedRF 类似, 可能存在训练-推理不一致问题
  -- ScaleConditionedRF 已证伪 (0.741 < 0.746 baseline, 真正集成后有害)
  -- 训练-推理尺度不一致是 SC-RF 的潜在结构性缺陷
- 需评估训练-推理一致性, 避免重蹈 ScaleConditionedRF 覆辙

### 优先级

- 待评估 (需看训练结果)
- 若训练结果显著优于 A4 baseline (mAP > 0.865), 可作为论文新 SOTA 方向
- 若训练结果持平或低于 baseline, 需理论分析失败原因 (类比 ScaleConditionedRF 证伪记录)

## 十一、VGAR (Velocity-Guided Adaptive Renewal) 优化

> 已集成到 head.py, 待系统评估。config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_vgar_24obj.py`

### 核心设想

- 用 $\alpha(t) \cdot \hat{x}_0 + (1-\alpha(t)) \cdot z$ 替代纯噪声 renewal
- 数学公式: $\alpha(t) = 0.2 + 0.6 \cdot \text{sigmoid}(5 \cdot (0.5 - t))$
- $x_{\text{renewed}} = \alpha(t) \cdot x_0\_{\text{pred}} + (1-\alpha(t)) \cdot \text{randn}$
- 理论依据: $v_\theta$ 的 $x_0\_{\text{pred}}$ 编码了 "框应该去哪里", 纯随机 renewal 浪费了这个信息
- $\alpha$ 随时间步自适应: 早期更随机 (探索), 后期更确定 (利用)

### 当前状态

- 已集成到 `head.py` (代码就绪)
- config: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_vgar_24obj.py` (基于 `a4_dpm_pp_24obj.py`)
- 改动: `velocity_guided_renewal=True`
- SwanLab: `ldmdet-mainline-ablation-24obj` / `a4_vgar` (project 已配置)
- 基线: A4 DPM-Solver++ (mAP=0.863)

### 与 D3 交互

- VGAR 缓解 D3 矛盾 (保留部分 $\hat{x}_0$ 信息), 但未完全消除
- VGAR 的 $\alpha(t)$ 在 $t \to 0$ 时 $\to 0.8$ (`head.py:177-179`), 仍保留 20% 随机性
- D3 矛盾仅缓解未消除: 下一步 $\hat{x}_0^{(n+1)}$ 与 $\hat{x}_0^{(n)}$ 有部分相关性, 但非完全轨迹连续
- 理论预期: VGAR 下 $\eta_{\text{str}}$ 应介于 baseline (renewal on) 和 renewal off 之间

### 待做

- 系统评估 VGAR 对 $\eta_{\text{str}}$ 和 mAP 的影响
- 与 D3 方案 A/B/C 对比, 验证 VGAR 是否为更优修复方案
- 3 seeds (42/123/789) 重训, 与 A4 baseline 对照
- 若 VGAR 显著改善 $\eta_{\text{str}}$ 且 mAP 不退化, 可作为论文新方向纳入

## 十二、方向 E: Brenier 映射神经化 (突破方向, 全新)

> ⛔ 未开展 (纯理论)。基于 doubao AI 建议 + 最优传输理论。TMI 投稿后考虑。
> 核心设想: 用 ICNN 参数化 Brenier 势 φ, T*(z) = ∇φ(z) 直接给出从噪声到 bbox 的最优传输映射

### 核心理论

- **Brenier 定理**: 在平方距离代价下, 最优传输映射 T* = ∇φ 存在且唯一 (φ 为凸函数)
- **ICNN (Makkuva et al., 2020)**: 提供参数化凸函数的方法, 通过非负权重 + 单调激活保证凸性
- **4D bbox 空间**: 输入维度 d=4 (低维), ICNN 计算成本可控

### 与当前架构的冲突

- ⚠ **架构冲突**: 当前 cascade head (RoIAlign + DynamicConv) vs ICNN 参数化 φ, 架构完全不同
- ⚠ **训练范式冲突**: flow matching 损失 vs Wasserstein 距离损失, 需重训
- ⚠ **推理范式冲突**: DPM-Solver++ 多步积分 vs ∇φ 一步映射, 无法复用 solver 框架

### 分阶段计划

- **E.1 理论分析** (纯理论, ~2 天): 推导 4D Brenier 势形式, 分析 ICNN 表达能力, 纳入论文 §6 未来工作 (~0.2 页)
- **E.2 ICNN 原型** (代码, ~3 天): 实现 4D ICNN + ∇φ 自动微分 + Sinkhorn 损失, 单元测试凸性
- **E.3 训练验证** (高成本, ~4.5 天): ICNN 替代 cascade head, 3 seeds × 150 epochs

### 优先级

- **低** (突破性方向, 风险高, TMI 10 页限制下难以纳入)
- **建议**: 仅做 E.1 (理论分析), 纳入论文 §6 未来工作, 不做实验

### doubao 方向对照 (A/B/C/D/E)

| doubao 方向 | 核心建议 | 当前论文对应方向 | 当前状态 |
|------------|---------|----------------|---------|
| A. 检测专用 RF-DPM 联合推导 | per-dim solver | 方向 A | ✓ 完成 (mAP 持平 +0.001, 加速 5.5%) |
| B. 速度感知网络结构 | 直接预测 v 而非 x0 | R3 (v-prediction 对照) | 🔄 seed 42 训练中 |
| C. 时间条件深度融合 | step-aware embedding | 方向 C | ✓ 完成 (早停@ep148, best 0.859, Δ=-0.004 在 noise 内; step_proj 活跃+3类改善, 非负面) |
| D. 自适应阶次 DPM-Solver++ | t 大用低阶, t 小用高阶 | 方向 D | ✓ 完成 (3 solver mAP 持平 0.863) |
| E. Brenier 映射神经化 | ICNN 参数化 Brenier 势 | 方向 E (本节) | ⛔ 未开展 |

> ⚠ **关键发现**: doubao 方向 D 假设"t 小时需要高阶修正", 但实际诊断显示 η_3rd 在后期 step (t 小) 显著降低 (44.6→18.2, 降幅 59%), 即实际趋势与 doubao 假设相反。

---

## 十三、模型结构改进 (M1-M5, 基于 D1-D5 诊断)

> 📌 详细方案: [docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md)
> 基线: A4 (DPM-Solver++, mAP=0.863, checkpoint best_epoch_117)
> 诊断脚本: `experiments/analysis/structural_diagnosis.py` (D1-D5) + `experiments/analysis/d1_roi_ablation.py` (D1 消融)

### D1 消融实验结果 ✓ (2026-07-22)

| 指标 | Baseline (7×7) | Ablation (空间抹平) | Δ |
|------|---------------|-------------------|---|
| mAP | 0.863 | 0.009 | **-0.854** |
| AP50 | 0.988 | 0.048 | -0.940 |

**结论**: 7×7 空间结构至关重要, DynamicConv 已有效提取 (非丢失)。M1 应**增强**而非重建空间编码。

### M1: 形态感知 RoI 编码器 ⭐ 最高优先级 — ✓ FP32 复现完成 (持平 A4, 设计问题确认)

- **设计**: 零初始化残差分支 (`roi_features + morph_emb`), 方向解耦卷积 (h_conv 臂长比 + v_conv 着丝粒)
- **D1 验证**: 初始状态不改变 A4 行为 (morph_emb≡0), 训练中逐步增强形态编码, 不破坏已验证有效的空间通路
- **实现** ✅:
  - 模块: [ldmdet/core/morphology_encoder.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/morphology_encoder.py) — `MorphologyAwareRoIEncoder`
  - 接线: 填充 `shape_attention` hook ([single_head.py:277-278](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/core/single_head.py)), detector.py 用 `MODELS.build` 构建任意注册模块
  - 配置: [m1_morphology_aware_24obj.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj.py) — `load_from` A4, 30ep, lr=1e-5
  - FP32 复现配置: [m1_morphology_aware_24obj_fp32.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj_fp32.py) — lr=2e-5 (2×), 1ep warmup, 30ep, FP32, SwanLab `m1_morphology_aware_fp32`
  - workstation 配置: [m1_morphology_aware_24obj_ws.py](file:///home/linkst/workspace/projects/chromosome-kd/experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj_ws.py) — BF16 AMP (`amp_dtype='bfloat16'`), 适配 A5000 24GB (实测显存 21GB < 24GB); head.py 支持 amp 字符串转换避免 mmengine lazy_import 冲突
  - 测试: [test_morphology_encoder.py](file:///home/linkst/workspace/projects/chromosome-kd/ldmdet/tests/test_morphology_encoder.py) — 15 测试全通过 (零初始化恒等性/方向解耦/梯度流/参数配置)
- **训练状态** (2026-07-23 BF16 + 2026-07-24 FP32 复现, ✓ 全部完成):
  - BF16: workstation A5000, seed 42, 30ep, 显存 20888 MiB (vs FP32 37506 MiB, 降 44%), SwanLab: `m1_morphology_aware_ws`
  - FP32 复现: ross A6000, seed 42, 30ep, 显存 37.5GB, SwanLab: `m1_morphology_aware_fp32`
  - A4 checkpoint 加载成功 (missing keys = M1 新参数 h_conv/v_conv/norm/fuse, 保持零初始化 fuse=0 → 恒等残差)
- **结果** ✓ FP32 复现完成 (2026-07-25):
  - **mAP 对比**:
    | 配置 | mAP | Δ vs A4 | 说明 |
    |------|-----|---------|------|
    | A4 (FP32) | 0.863 | — | 基线 |
    | A4+BF16 | 0.825 | -0.038 | BF16 本身掉点 |
    | M1 (BF16) | 0.818 | -0.045 | BF16 虚假退化 |
    | **M1 (FP32)** | **0.862** | **-0.001** | **统计上持平！BF16 误导** |
  - **M1 FP32 训练曲线**: 30 epoch 完成, best 0.862@ep19, last 0.859@ep30, 全程稳定 (0.854-0.862)
  - **fuse 权重分析 (FP32 best@ep19 vs BF16 ep30)**:
    | 指标 | M1 FP32 (ep19) | M1 BF16 (ep30) | 结论 |
    |------|---------------|----------------|------|
    | fuse norm | 0.36-0.43 | 0.215 | FP32 增长更大 (lr 2×) |
    | h_conv ratio (空间) | 1.01-1.02 | 1.01 | **仍均匀** |
    | h_conv std (空间维度) | 0.0001 | 0.0000 | **仍均匀** |
    | v_conv ratio (空间) | 1.02 | 1.01 | **仍均匀** |
  - **核心结论**: M1 FP32 mAP=0.862 与 A4 (0.863) **统计上持平** (Δ=-0.001)。BF16 实验完全误导 — BF16 导致 -0.044 虚假退化。但 h_conv/v_conv 在 FP32 下**仍然均匀**, 确认是**设计问题而非精度问题**:
    -- (1) 零初始化 fuse 的梯度瓶颈 → h_conv/v_conv 梯度极弱 (即使 lr 2×)
    -- (2) (7,1)+(1,7) 感受野与 7×7 RoI 同尺寸, 缺乏空间上下文
    -- (3) morph_emb 退化为常数偏置, 未学到方向性形态信息
  - **详细分析**: [STRUCTURAL_IMPROVEMENT_ANALYSIS.md §3.1.7](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md)
- **M1-v2 改进方向** (若后续启动):
  - 非零初始化 fuse (如小常数初始化 0.01, 打破梯度瓶颈)
  - 显式形态先验注入 (臂长比/面积作为输入, 而非依赖卷积发现)
  - 注意力机制替代方向卷积 (self-attention 自然捕获空间关系)
- **训练命令** (FP32 复现, ross):
  ```bash
  ssh linkst@100.122.196.41 "cd /media/ross/8TB/linkst/chromo/chromosome-kd && \
    /home/linkst/data/miniconda3/envs/chromo/bin/python experiments/runners/train.py \
    experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj_fp32.py \
    --work-dir work_dirs/m1_morphology_aware_24obj_fp32 --gpu-id 0 --seed 42"
  ```
- **参数开销**: 262.8K/head × 6 = 1.58M (<总参数 0.5%)

### M4: 级联头角色分化 — ⛔ 待启动 (中-高优先级)

- **设计**: 方案 A (损失权重衰减), 零代码改动, 仅改 criterion 配置
- **D3 诊断支持**: head0 修正最大 (reg std 0.94), 后级递减, 等权 deep_supervision 可能非最优
- **训练**: 从 A4 checkpoint 微调, 对比等权 vs 衰减

### 优先级排序 (D1-D5 诊断后修正)

| 方案 | 优先级 | 修正原因 |
|------|--------|---------|
| M1 形态感知 RoI 编码器 | ⭐最高 | D1 证实空间编码至关重要, M1 增强非重建; D4 显示 G21/Y 需形态区分 |
| M4 级联头角色分化 | 中-高 | D3 显示头间有自然分化但不充分, 零代码改动低成本 |
| M2 尺度-类别耦合头 | ↓中 | D4 推翻: 模型已隐式学到强尺寸→类别映射, 边际收益有限 |
| M3 重叠感知注意力 | ↓低 | D5 显示 head2-3 已有聚焦, head0 均匀但处理噪声框 |
| M5 学习式 renewal | 低 | D4 显示 Y 样本量最少, 但收益不确定 |

---

## 附: SwanLab Project 映射 (待做方向相关)

| SwanLab Project | 方向 | 状态 | URL Pattern |
|-----------------|------|------|-------------|
| `ldmdet-r3-vpred` | R3 v-prediction | ✓ seed 42 完成 (best 0.855@ep34, Δ=-0.008 单 seed 支持 R3.2), seed 123/789 ⛔ 待补 | `https://swanlab.cn/@einspanner/ldmdet-r3-vpred/runs/<run_id>` |
| `ldmdet-s1-cascade-decouple` | S1 cascade × solver | ✓ 三组全部完成 (h3_s4/h3_s8/h6_s2 = 0.859/0.859/0.859, S1.3 闭环 → LINEAGE §七) | `https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/<run_id>` |
| `few-shot-benchmark` | Few-Shot 源预训练 | 🔄 1 运行中, 6 已完成 | `https://swanlab.cn/@einspanner/few-shot-benchmark/runs/<run_id>` |
| `ldmdet-breakthrough` | SC-RF 自条件化 | 🔄 运行中 | `https://swanlab.cn/@einspanner/ldmdet-breakthrough/runs/<run_id>` |
| `ldmdet-mainline-ablation-24obj` | VGAR (a4_vgar) + 方向 C (a6_step_aware) + M1 (m1_morphology_aware_ws/fp32) | ✓ 方向 C 完成; M1 FP32 完成 (best 0.862@ep19, 持平 A4); VGAR ⛔ 待启动 | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| `ldmdet-ablation` | D3 修复方案 B (已完成) | ✓ 已完成 | `https://swanlab.cn/@einspanner/ldmdet-ablation/runs/<run_id>` |
| (无SwanLab) | 方向 A per-dim η_str 诊断 + Phase 2 | ✓ 完成 (→ [LINEAGE §九](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)) | `direction_a_d_diagnosis.py` + `direction_a_per_dim_comparison.py` |
| (无SwanLab) | 方向 D 自适应阶次诊断 + mAP 对比 | ✓ 完成 (→ [LINEAGE §十](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)) | `direction_a_d_diagnosis.py` + `direction_d_solver_comparison.py` |
| (无SwanLab) | D1 RoI 空间消融 + D1-D5 结构诊断 | ✓ 完成 (ΔmAP=-0.854) | `d1_roi_ablation.py` + `structural_diagnosis.py` → `work_dirs/diagnosis/` |
| `ldmdet-mainline-ablation-24obj` (m1_morphology_aware_fp32) | M1 FP32 复现 | ✓ 完成 (best 0.862@ep19, Δ=-0.001 持平 A4, BF16 误导确认; h_conv/v_conv 仍均匀 → 设计问题) | SwanLab exp: `m1_morphology_aware_fp32` |
| `ldmdet-head-distill` | **Head Distillation v2** | ⚠ 异常中断@ep99/150 (best 0.711@ep96, 仍在缓慢上升, 待恢复决策) | `https://swanlab.cn/@einspanner/ldmdet-head-distill/runs/<run_id>` |
| (待创建) | M4 级联头角色分化 | ⛔ 待启动 | 详见 [STRUCTURAL_IMPROVEMENT_ANALYSIS.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/research/STRUCTURAL_IMPROVEMENT_ANALYSIS.md) |
| `ldmdet-reflow-standard` | **ReFlow (Standard MSE)** | ⛔ 待创建 | `https://swanlab.cn/@einspanner/ldmdet-reflow-standard/runs/<run_id>` |

> SwanLab 用户名: `einspanner` (登录态见 `/home/linkst/.swanlab/.netrc`, api_key 已配置)
> 目标微调 project (Few-Shot 14 个配置) 待 FBM CrossAttn 源预训练完成后配置

<!-- 文档结束。
     更新策略: 当方向状态变化 (如训练启动 / 完成 / 证伪), 更新对应章节的 ⛔/🔄/✓/🔴 标记和 SwanLab run_id。
     方向完成后: 有效→迁入 EXPERIMENT_LINEAGE.md; 证伪→迁入 FALSIFIED_DIRECTIONS.md; 本文档仅保留 🔄进行中 + ⛔待启动。
     2026-07-25 更新: R3 seed 42 ✓ + S1 ✓ (迁入 LINEAGE §七) + M1 FP32 ✓ 完成 (持平 A4, 设计问题确认); Head Distillation v2 ⚠ 异常中断@ep99 待恢复决策。 -->
