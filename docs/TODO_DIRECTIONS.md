# 待做研究方向总览 (TODO Directions)

> 本文档梳理 KaryoFlow (染色体检测论文, 目标 TMI 期刊) 所有进行中或待启动的研究方向。
> 这些方向部分有代码就绪、配置就绪或实验已在运行, 部分仅有理论框架。
> 每个方向附 **可靠数据源地址** (本地服务器路径 / SwanLab project / config 路径)。
> 更新时间: 2026-07-21
>
> 📌 **关联文档**:
> - [docs/EXPERIMENT_LINEAGE.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md) (主路线实验脉络, A0-A4 主路线消融已完成)
> - [docs/paper/paper_draft_CN.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/paper_draft_CN.md) (论文草稿, §6 结论与未来工作)
> - [docs/paper/theory_analysis_RF_DPM.md](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md) (R1/S1/D3/R3 理论深化分析)
>
> ⚠ **状态约定**: 🔄 运行中 / ⛔ 待启动 / ✓ 已完成 / 🔴 已证伪

## 〇、方向索引与状态汇总

| 编号 | 方向 | 状态 | 优先级 | SwanLab Project |
|------|------|------|--------|-----------------|
| R3 | x0-prediction vs v-prediction 对照重训 | 🔄 进行中 (seed 42 训练中) | 中 | `ldmdet-r3-vpred` |
| S1 | Cascade Head × Solver Step 解耦消融 | 🔄 部分完成 (s1_h3_s4 ✓ / s1_h3_s8 ✓ / s1_h6_s2 🔄) | 高 | `ldmdet-s1-cascade-decouple` |
| Few-Shot | 24obj 源 → chromo 目标跨数据集微调 | ⛔ 待启动 (配置就绪) | 高 | `few-shot-benchmark` (源预训练) |
| D3 | box_renewal × DPM-Solver++ 修复方案 A/C | ⛔ 待启动 (方案 B 已验证) | 中 | `ldmdet-ablation` |
| 方向 A | per-dim eta_str 维度级曲率诊断 | ✓ 诊断完成 (部分支持) / Phase 2 ⛔ 待启动 | 中 | (诊断无 SwanLab) |
| 方向 C | step-aware embedding (cascade head 感知 step) | ⛔ 待启动 (代码就绪) | 中 | `ldmdet-mainline-ablation-24obj` |
| 方向 D | 自适应阶次 DPM-Solver++ (后期 step 降阶) | ✓ 诊断完成 (支持假设) / mAP 对比 ⛔ 待跑 | 中 | (诊断无 SwanLab) |
| 2-RectFlow | Reflow 进一步拉直轨迹 | ⛔ 纯理论推测 | 低 | — |
| 跨数据集扩展 | OT Collapse 普遍性 claim 验证 | ⛔ 纯理论推导 | 中 (最高级目标) | — |
| SC-RF | 自条件化 RF | 🔄 运行中 (待评估) | 待评估 | `ldmdet-breakthrough` |
| VGAR | Velocity-Guided Adaptive Renewal | ⛔ 待系统评估 | 中 | `ldmdet-mainline-ablation-24obj` |

## 一、R3: x0-prediction vs v-prediction 对照重训

> 🔄 进行中: seed 42 训练中 (workstation A4000), seed 123/789 ⛔ 待启动。理论依据: [theory_analysis_RF_DPM.md §4](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)

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

seed 42 🔄 训练中, seed 123/789 ⛔ 待启动:

- seed 42 🔄 训练中 (workstation A4000)
  -- 进度: epoch 8, best mAP=0.802 (warmup 阶段), ETA ~1.4 天
  -- SwanLab project: `ldmdet-r3-vpred`
- seed 123 ⛔ 待启动 (等待 GPU 空闲, 计划 ross A6000)
- seed 789 ⛔ 待启动 (等待 GPU 空闲, 计划 workstation A5000)

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

> 🔄 部分完成: s1_h3_s4 ✓ / s1_h3_s8 ✓ 已完成, s1_h6_s2 🔄 训练中 (epoch 123/150)。理论依据: [theory_analysis_RF_DPM.md §2](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)

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

s1_h3_s4 / s1_h3_s8 已完成, s1_h6_s2 训练中:

- `s1_h3_s4` ✓ 已完成 (之前会话, 结果见 memory)
- `s1_h3_s8` ✓ 已完成
  -- 进度: best mAP=0.859 (epoch 64), 30 epochs 未改善早停
  -- 算力: ross A6000 (PID 501721)
- `s1_h6_s2` 🔄 训练中
  -- 进度: epoch 123/150, best mAP=0.859 (epoch 106), ETA ~5h
  -- 算力: workstation A5000 (PID 1098621)

SwanLab project `ldmdet-s1-cascade-decouple` 已配置, 3 个实验状态如下: 2 已完成 + 1 训练中。

### 关键结论 (阶段性)

- s1_h6_s2 (H=6, S=2, NFE=12) 在 12 NFE 下 best mAP=0.859, **达到 A3 baseline 3-seed 均值水平 (0.859 ± 0.003)**
- 说明**减少 step 并保持 head 可在更少 NFE 下维持性能**, 与 S1 命题 S1.3 (H×S 可交换性边界) 对照: H=6 充分大时减小 S 仍可保持横向收敛性
- s1_h3_s8 (H=3, S=8, NFE=24) best mAP=0.859, 与 baseline 持平, 表明同等 NFE 下 H=3 S=8 可补偿 H 减半
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

## 五、方向 A: per-dim eta_str 维度级曲率诊断

> ✓ 诊断完成 (部分支持), Phase 2 (检测专用 solver) ⛔ 待启动。零成本诊断, 无 SwanLab。详见 [EXPERIMENT_LINEAGE.md §九](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

### 核心目标

- 在 A4 checkpoint 上零成本诊断检测空间 4 维 (cxcywh) 各维度的曲率差异
- 探究是否可设计 per-dim solver: w,h 维度用低阶 solver, cx,cy 用高阶
- 量化位置维度 (cx,cy) vs 尺度维度 (w,h) 的曲率差距

### 诊断方法

- 在 A4 checkpoint (best mAP=0.859, epoch 117) 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
- 计算 wh/cxcy 维度 eta_str 比值, 量化位置维度 vs 尺度维度的曲率差距
- 诊断脚本: `experiments/analysis/direction_a_d_diagnosis.py`
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py`
- checkpoint: `work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth`
- 结果 JSON: `work_dirs/diagnosis/dpm_pp_2nd.json`, `work_dirs/diagnosis/dpm_pp_3rd.json`, `work_dirs/diagnosis/dpm_pp_adaptive.json`

### 诊断结果

- **dpm_solver_pp (2 阶)**: wh/cxcy 比值 0.41-0.50
  -- h 维度 eta_str (4-11) 显著小于 cx,cy (17-50), h 维度曲率比 cx,cy 小 3-5×
- **dpm_solver_pp_3 (3 阶)**: wh/cxcy 比值 0.25-0.59
  -- step 0 有数值异常 (h=77.9, 待分析)
  -- w 维度差距较小 (1.5-2×)
- **结论**: 部分支持假设
  -- h 维度曲率显著小于 cx,cy (3-5× 差距), 支持原假设
  -- w 维度差距较小 (1.5-2×), 部分证伪 "w,h 都显著小于 cx,cy" 的强假设 (w 维度需修正假设)
- **不加入 FALSIFIED_DIRECTIONS.md**: 不是完全证伪, 仅需修正假设 (w 维度差距小于预期)

### Phase 2 (待启动)

- 设计 w,h 维度用低阶 solver、cx,cy 用高阶的混合方案 (检测专用 solver)
- 但 w 维度差距较小, 实际增益可能有限
- 状态: ⛔ 待启动 (优先级中)

### 与 R1 的关系

- R1: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 A: per-dim $\eta_{str}$ 量化各维度曲率差异
- 互补: R1 决定步数, 方向 A 决定 per-dim 阶数分配

## 六、方向 C: step-aware embedding (cascade head 感知 solver step)

> ⛔ 待启动 (代码就绪)。详见 [EXPERIMENT_LINEAGE.md §十一](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

### 核心目标

- 让 cascade head 感知 DPM-Solver++ step 编号
- 不同 step 上 $x_t$ 的统计特性不同 (早期近噪声, 后期近 GT), 显式 step 条件可能提升每步精化的针对性
- 零初始化确保预训练兼容: 训练初期 step_proj 输出为 0, 模型行为与无 step embedding 时一致

### 实现方式

- step_mlp + step_proj 零初始化
- 实现位置: `ldmdet/core/head.py` (step_mlp + step_proj 零初始化)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a6_step_aware_24obj.py`

### 当前状态

- ⛔ 待启动训练 (ross A6000 即将启动 seed 42)
- 3 seeds (42/123/789) 重训计划
- 对照: A4 baseline (3-seed 均值 0.859 ± 0.003)
- SwanLab project: `ldmdet-mainline-ablation-24obj` (experiment_name=`a6_step_aware`)

### 与 S1 的关系

- S1 形式化 cascade head × solver step 算子分裂 (§二)
- 方向 C 在不破坏 S1 算子分裂结构的前提下, 让 cascade head 显式感知 step
- 与 S1 互补: S1 给出架构合理性框架, 方向 C 在框架内探索性能提升

### 预期

- 若 step embedding 显著提升 mAP (Δ > +0.005), 可作为论文新方向
- 若持平, 表明 cascade head 已通过 $x_t$ 隐式感知 step 信息 (因 $x_t$ 在不同 step 上统计不同)

## 七、方向 D: 自适应阶次 DPM-Solver++ (后期 step 降阶)

> ✓ 诊断完成 (支持假设), mAP 对比实验 ⛔ 待跑。零成本诊断, 无 SwanLab。详见 [EXPERIMENT_LINEAGE.md §十](file:///home/linkst/workspace/projects/chromosome-kd/docs/EXPERIMENT_LINEAGE.md)

### 核心目标

- 基于 $\eta_{3rd}$ 趋势设计自适应降阶策略: 后期 step 的 3 阶校正项显著小于早期, 可降为 2 阶
- 验证 RF 轨迹在 $t \to 0$ 时趋于直线的假设 (3 阶校正项 $D_2$ 应小)
- 与 R1 整体 $\eta_{str}$ 互补: R1 决定步数, 方向 D 决定每步阶数

### 诊断方法

- 在 A4 checkpoint 上跑 50 张图 × 3 个 solver (dpm_solver_pp / dpm_solver_pp_3 / dpm_solver_pp_adaptive)
- 测量 $\eta_{3rd} = \|D_2\|/\|\hat{x}_0\|$ 随 step 的变化趋势
- 实现位置: `ldmdet/diffusion/rectified_flow.py` (`RFDPMSolverAdaptive`, static + eta_threshold 两种模式)
- 配置: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a7_dpm_pp_adaptive_24obj.py`
- 结果 JSON: `work_dirs/diagnosis/dpm_pp_adaptive.json`

### 诊断结果

- $\eta_{3rd}$ 趋势: step 1 = 44.6 → step 2 = 18.2 (decreasing, 降幅 59%)
- **结论**: ✓ 支持重构假设 — 后期 step 的 3 阶校正项显著小于早期, 可降为 2 阶
- 与 R1 整体 $\eta_{str}$ 单调下降 (3.43→2.45→1.68) 一致, 但方向 D 量化了 3 阶项的衰减

### 待跑实验

- mAP 对比实验: dpm_solver_pp (2 阶) vs dpm_solver_pp_3 (3 阶) vs dpm_solver_pp_adaptive (自适应)
- 零成本推理 (无需重训, 直接在 A4 checkpoint 上评估)
- 预期: adaptive 在保持 mAP 的同时减少后期 step 计算量
- 状态: ⛔ 待跑 (优先级中)

### 与 R1 的关系

- R1: 整体 $\eta_{str}$ 量化"2 步收敛"
- 方向 D: per-step 3 阶项 $\eta_{3rd}$ 量化"后期 step 可降阶"
- 互补: R1 决定步数, 方向 D 决定每步阶数

## 八、2-RectFlow (Reflow) 潜在方向

> 纯理论推测, ⛔ 未启动。理论依据: [theory_analysis_RF_DPM.md §1.4](file:///home/linkst/workspace/projects/chromosome-kd/docs/paper/theory_analysis_RF_DPM.md)

### 核心目标

- 若 $\eta_{\text{str}}$ 持续 > 0.1, 考虑 2-RectFlow (Liu et al., 2023) 进一步拉直轨迹
- 将 "RF 训练成功" 从经验观察提升为可量化结论 ($\eta_{\text{str}} \to 0$)

### 理论依据

- R1 诊断显示 $\eta_{\text{str}} \in [0.7, 1.5]$ 非零 (3-seed 实验), 轨迹并非理想直线
- baseline (renewal on): step 1 $\eta_{\text{str}}=3.43 \pm 0.36$, step 3 $=1.68 \pm 0.15$
- renewal off: step 1 $\eta_{\text{str}}=1.50 \pm 0.33$, step 3 $=0.70 \pm 0.09$
- 修正了论文 §4.5.2 "RF 轨迹接近直线" 的 claim: 更准确表述是 "轨迹曲率在 step 2 后足够小, 使 DPM-Solver++ 校正项对 mAP 的边际贡献 < 0.001"

### 触发条件

- 若 $\eta_{\text{str}} > 0.1$ 持续 (当前 $\eta_{\text{str}} \approx 1.5$, 理论上 reflow 可能有收益)
- 若 $\eta_{\text{str}} < 0.01$, reflow 收益有限 (无需启动)

### 风险

- reflow 需重训, 成本高 (3 seeds × 150 epochs)
- 当前 2 步已收敛 (mAP 0.863, DPM-Solver++ 2 步即收敛), reflow 收益可能有限
- 理论分析已预防审稿人质疑 (R1 已提供 $\eta_{\text{str}}$ 量化诊断)

### 优先级

- 低 (理论分析已预防审稿人质疑, 无需实验验证)
- 仅在审稿人强烈要求或 $\eta_{\text{str}}$ 显著恶化时启动

## 九、跨数据集扩展 (最高级目标, 理论推导)

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

## 附: SwanLab Project 映射 (待做方向相关)

| SwanLab Project | 方向 | 状态 | URL Pattern |
|-----------------|------|------|-------------|
| `ldmdet-r3-vpred` | R3 v-prediction | 🔄 seed 42 训练中, seed 123/789 ⛔ | `https://swanlab.cn/@einspanner/ldmdet-r3-vpred/runs/<run_id>` |
| `ldmdet-s1-cascade-decouple` | S1 cascade × solver | 🔄 2 已完成 + 1 训练中 (s1_h6_s2) | `https://swanlab.cn/@einspanner/ldmdet-s1-cascade-decouple/runs/<run_id>` |
| `few-shot-benchmark` | Few-Shot 源预训练 | 🔄 1 运行中, 6 已完成 | `https://swanlab.cn/@einspanner/few-shot-benchmark/runs/<run_id>` |
| `ldmdet-breakthrough` | SC-RF 自条件化 | 🔄 运行中 | `https://swanlab.cn/@einspanner/ldmdet-breakthrough/runs/<run_id>` |
| `ldmdet-mainline-ablation-24obj` | VGAR (a4_vgar) + 方向 C (a6_step_aware) | ⛔ 待启动 | `https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/<run_id>` |
| `ldmdet-ablation` | D3 修复方案 B (已完成) | ✓ 已完成 | `https://swanlab.cn/@einspanner/ldmdet-ablation/runs/<run_id>` |
| (无 SwanLab) | 方向 A per-dim η_str 诊断 | ✓ 诊断完成 (部分支持) | 本地脚本 `experiments/analysis/direction_a_d_diagnosis.py` |
| (无 SwanLab) | 方向 D 自适应阶次诊断 | ✓ 诊断完成 (支持假设) | 本地脚本 `experiments/analysis/direction_a_d_diagnosis.py` |

> SwanLab 用户名: `einspanner` (登录态见 `/home/linkst/.swanlab/.netrc`, api_key 已配置)
> 目标微调 project (Few-Shot 14 个配置) 待 FBM CrossAttn 源预训练完成后配置

<!-- 文档结束。
     更新策略: 当方向状态变化 (如训练启动 / 完成 / 证伪), 更新对应章节的 ⛔/🔄/✓/🔴 标记和 SwanLab run_id。
     R3/S1/Few-Shot/D3/方向 C 方向均有代码或配置就绪, 方向 A/D 已完成零成本诊断, 2-RectFlow/跨数据集扩展仍为纯理论。 -->
