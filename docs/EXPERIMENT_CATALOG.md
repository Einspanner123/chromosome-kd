# LDMDet 实验完整目录

> 生成日期: 2026-07-15
> 数据来源: 本地 `work_dirs/` (48 个子目录) + SwanLab 云端 (10+ 项目) + `ldmdet-experiment/` 归档 + 两台服务器 (workstation / ross)
> 核心文档: [EXPERIMENT_LINEAGE.md](EXPERIMENT_LINEAGE.md) (实验谱系) + [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) (结果汇总) + [paper/AAAI_INTEGRATED_DRAFT.md](paper/AAAI_INTEGRATED_DRAFT.md) (论文草稿)

---

## 〇、数据源与服务器说明

### 三类数据源

| 来源 | 路径/格式 | 用途 |
|------|----------|------|
| **本地服务器日志** | `work_dirs/<exp_dir>/<timestamp>/<timestamp>.log` + `vis_data/scalars.json` | 完整训练曲线 + 配置快照 |
| **ldmdet-experiment 归档** | `ldmdet-experiment/sota/<category>/<exp_name>/` (含 README, config.py, metrics.json, code/, checkpoints/) | 已归档 SOTA 实验 (2026-06-13) |
| **SwanLab 云端** | `https://swanlab.cn/@einspanner/<project>/runs/<run_id>` | 在线可视化 + 跨实验对比 |

### SwanLab 项目一览 (10 个项目)

| SwanLab Project | 实验数 | 数据集 | 范围 |
|-----------------|--------|--------|------|
| `chromosome-kd` | 21 | chromo | 早期: sota_seed*, ablation/*, scheme_*, stability/* |
| `chromosome-kd-benchmark-24obj` | 8 | 24obj | 对比模型: RTMDet-L / DINO R50 / Cascade / YOLOX-S / DiffusionDet / ldmdet_stochot_eps5 |
| `ldmdet-ablation` | 9 (24obj) + 23 (chromo) | 两者 | 主线消融 + 24obj 耦合策略 + merged + gen_transfer |
| `ldmdet-mainline-ablation-24obj` | 5+ ⭐ | 24obj | A0-A4 主路线消融 (+SwiGLU 多种子) |
| `ldmdet-breakthrough` | 2+ | 24obj/chromo | SC-RF 自条件化, I1 Seesaw Loss |
| `ldmdet-frontier-directions` | 6 | 24obj | h_velocity_loss / n_cascade_e2e / h_cfm_velocity 等 |
| `few-shot-benchmark` | 3 | 24obj | FBM CrossAttn / FBM SimpleGate / LDMDet SOTA 源预训练 |
| `nonlinear-3seed-repro` | 2 | chromo | nonlinear_e43_seed{1,2} (seed3 失败) |
| `setdiff-24obj` | 1+ | 24obj | SetDiff (Coupled State Diffusion) 基线 |
| `cross-domain-autokary` | 6+ | AutoKary | 跨域 zero-shot + finetune (A2/A3/A4 × k5/k10) |

### 服务器实验状态

| 服务器 | GPU | 用途 | 本地同步状态 |
|--------|-----|------|-------------|
| **ross** (8TB) | — | 主力训练服务器, `work_dirs/` 完整数据 + scalars.json | 部分同步到本地 (A0-A4, 部分消融) |
| **workstation** (A5000/A4000) | A5000 24GB | 并行多种子 / 消融 | 多种子实验已完成 (A4 3-seed, StochOT ε=2/5); checkpoint 待 SCP |
| **本地** `/home/linkst/workspace/chromosome-kd/` | — | 开发 + 分析 + FPS benchmark | 48 个 work_dirs 子目录 (含最近实验) |

> ⚠ 本地 `work_dirs/` 为 ross 服务器的子集同步。完整训练日志 (含所有 epoch 的 scalars.json) 在 ross 服务器 `/media/ross/8TB/linkst/chromo/chromosome-kd/work_dirs/`。

---

## 一、实验总览表

> 按数据集和实验类型分组。mAP 加粗者为该组最优。

### 1.1 24obj 数据集 — 核心实验 (论文引用)

| 实验名称 | 数据集 | SwanLab项目 | run_id | 本地路径 | 配置文件 | mAP | 状态 | 分类 |
|----------|--------|------------|--------|----------|----------|-----|------|------|
| A0 baseline (Euler 1步) | 24obj | ldmdet-mainline-ablation-24obj | (a0_baseline) | work_dirs/a0_baseline_24obj/ | a0_baseline_24obj.py | 0.774 | ✅ 完成 | 主路线消融 |
| A1 +RF+Heun | 24obj | ldmdet-mainline-ablation-24obj | (a1_rf_heun) | work_dirs/a1_rf_heun_24obj/ | a1_rf_heun_24obj.py | 0.856 | ✅ 完成 | 主路线消融 |
| A2 +AdaLN-Zero | 24obj | ldmdet-mainline-ablation-24obj | (a2_rf_heun_adaln) | work_dirs/a2_rf_heun_adaln_24obj/ | a2_rf_heun_adaln_24obj.py | 0.856 | ✅ 完成 | 主路线消融 |
| A3 +StochOT eps5 | 24obj | ldmdet-mainline-ablation-24obj | (a3_full_sota) | work_dirs/a3_full_sota_24obj/ | a3_full_sota_24obj.py | 0.858 | ✅ 完成 | 主路线消融 |
| **A4 DPM-Solver++** | 24obj | ldmdet-mainline-ablation-24obj | (a4_dpm_pp) | work_dirs/a4_dpm_pp_24obj/ | a4_dpm_pp_24obj.py | **0.863** (3-seed: 0.859±0.004) | ✅ 完成 | 主路线消融 | <!-- verified: 2026-07-16: seed42=0.863, seed123=0.857@ep62, seed789=0.856@ep72 -->
| Random seed_42 | 24obj | ldmdet-ablation | p5xqii8mcqmbhuo5lhlff | work_dirs/24obj_ablation/random/seed_42/ | chromo_24obj_random.py | 0.859 | ✅ 完成 | 耦合消融 |
| Random seed_789 | 24obj | ldmdet-ablation | r8n441mu4gws43xyoneoj | work_dirs/24obj_ablation/random/seed_789/ | chromo_24obj_random.py | 0.860 | ✅ 完成 | 耦合消融 |
| Random seed_123 | 24obj | ldmdet-ablation | q6jgxefgxbp8f2sf5qzpc | work_dirs/24obj_ablation/random/seed_123/ | chromo_24obj_random.py | 0.814/0.860 ⚠ | ⚠ 中断 | 耦合消融 |
| GHSS seed_42 | 24obj | ldmdet-ablation | k84cq9oftbp2nld88a85t | work_dirs/24obj_ablation/ghss/seed_42/ | chromo_24obj.py | 0.857 | ✅ 完成 | 耦合消融 |
| GHSS seed_789 | 24obj | ldmdet-ablation | holadvhaz9v2rh8l494hv | work_dirs/24obj_ablation/ghss/seed_789/ | chromo_24obj.py | 0.859 | ✅ 完成 | 耦合消融 |
| GHSS seed_123 | 24obj | ldmdet-ablation | 73cr3uyqw4f1q68xz1evg | work_dirs/24obj_ablation/ghss/seed_123/ | chromo_24obj.py | 0.859 | ✅ 完成 | 耦合消融 |
| Sinkhorn Stoch seed_42 | 24obj | ldmdet-ablation | o96m1eqz4l12qjeyys1cs | work_dirs/24obj_ablation/sinkhorn/seed_42/ | chromo_24obj_sinkhorn.py | 0.856 | ✅ 完成 | 耦合消融 |
| ldmdet_stochot_eps5 (早期) | 24obj | chromosome-kd-benchmark-24obj | odnz6a8pnda3gyg84cfqv | work_dirs/ldmdet_rf_heun_adaln_stochot_eps5/ | ldmdet_rf_heun_adaln_stochot_eps5.py | 0.853 | ✅ 完成 | 早期基线 |

### 1.2 24obj 数据集 — 基线对比模型

| 实验名称 | SwanLab项目 | run_id | 本地路径 | 配置文件 | mAP | 状态 | 分类 |
|----------|------------|--------|----------|----------|-----|------|------|
| RTMDet-L | chromosome-kd-benchmark-24obj | — | work_dirs/baselines/rtmdet_l_24obj/ | benchmark_24obj/rtmdet_l.py | **0.869** | ✅ | 基线对比 | <!-- verified: 2026-07-16: 本地 scalars.json 不完整 (count=86, max=0.863), best@ep116 在 ross 服务器; 0.869 来自 EXPERIMENT_LINEAGE.md -->
| DINO R50 (4scale) | chromosome-kd-benchmark-24obj | (dino-r50-4scale) | ⚠ 无本地目录 (仅 SwanLab) | benchmark_24obj/dino_r50.py | 0.868 | ⚠ CRASHED | 基线对比 | <!-- verified: 2026-07-16: swanlab_export.json 确认 0.868 (count=93); work_dirs/baselines/ 无 dino_r50_24obj/ -->
| Cascade R-CNN R50 | chromosome-kd-benchmark-24obj | (cascade-rcnn-r50) | work_dirs/baselines/ | benchmark_24obj/cascade_rcnn_r50.py | 0.854 | ✅ | 基线对比 | <!-- verified: 2026-07-16: swanlab_export.json 确认 0.854 (count=102); 无本地 scalars.json -->
| YOLOX-S | chromosome-kd-benchmark-24obj | (yolox-s) | work_dirs/baselines/ | benchmark_24obj/yolox_s.py | 0.796 | ✅ | 基线对比 | <!-- verified: 2026-07-16: swanlab_export.json 确认 0.796 (count=150); 无本地 scalars.json -->
| DiffusionDet | chromosome-kd-benchmark-24obj | (benchmark_diffusiondet) | work_dirs/baselines/ | benchmark_24obj/diffusiondet_ddpm.py | 0.787 | ⚠ CRASHED | 基线对比 | <!-- verified: 2026-07-16: swanlab_export.json 确认 0.787 (count=54); 无本地 scalars.json -->

### 1.3 24obj 数据集 — 探索性实验 (未进论文/已归档)

| 实验名称 | SwanLab项目 | 本地路径 | 配置文件 | mAP | 状态 | 分类 |
|----------|------------|----------|----------|-----|------|------|
| A2 +SwiGLU FFN | ldmdet-mainline-ablation-24obj | work_dirs/a2_swinglu_24obj/ | a2_swinglu_24obj.py (备份) | 0.859 | ✅/⚠ | 探索-激活函数 | <!-- verified: 2026-07-16 -->
| A4 +SwiGLU FFN | ldmdet-mainline-ablation-24obj | work_dirs/a4_swinglu_24obj/ | a4_swinglu_24obj.py (备份) | 0.857 | ✅/⚠ | 探索-激活函数 | <!-- verified: 2026-07-16 -->
| SC-RF 自条件化RF | ldmdet-breakthrough | work_dirs/sc_rf_24obj/ | sc_rf_24obj.py (备份) | 0.860 | ✅ 已归档 | 探索-自条件化 | <!-- verified: 2026-07-16 -->
| SetDiff (Coupled State) | setdiff-24obj | work_dirs/setdiff_24obj/ | setdiff_24obj.py | 0 (早期) | ✅/⚠ | 探索-SetDiff | <!-- verified: 2026-07-16 -->
| PD-RF 渐进蒸馏 | (待确认) | work_dirs/pd_rf_24obj/ | pd_rf_24obj.py (备份) | 0.851 | ✅/⚠ | 探索-蒸馏 | <!-- verified: 2026-07-16 -->
| h_velocity_loss | ldmdet-frontier-directions | — | — | 0.856 | ⚠ CRASHED | 前沿方向 |
| n_cascade_e2e | ldmdet-frontier-directions | — | — | 0.684 | ✅ FINISHED | 前沿方向(证伪) |
| h_cfm_velocity | ldmdet-frontier-directions | — | — | ~0 | ❌ FAILED | 前沿方向(证伪) |
| I1 Seesaw+Normalized | ldmdet-breakthrough | work_dirs/i1_seesaw_normalized/ | i1_seesaw_normalized.py | 0.744 | ✅/⚠ | ⚠ 实为 chromo 数据集 (非 24obj) | <!-- verified: 2026-07-16 -->
| normalized_only | (待确认) | work_dirs/normalized_only/ | normalized_only.py | 0.747 | ✅/⚠ | ⚠ 实为 chromo 数据集 (非 24obj) | <!-- verified: 2026-07-16 -->
| FBM CrossAttn (源预训练) | few-shot-benchmark | work_dirs/few_shot/ | ldmdet_fbm_crossattn_24obj.py | 0.857 | ⚠ CRASHED | Few-Shot |
| FBM SimpleGate (源预训练) | few-shot-benchmark | work_dirs/few_shot/ | ldmdet_fbm_simplgate_24obj.py | 0.677 | ❌ 已停止 | Few-Shot |
| LDMDet SOTA (源预训练) | few-shot-benchmark | work_dirs/few_shot/ | ldmdet_sota_24obj.py | 已完成 | ✅ | Few-Shot |

### 1.4 24obj 数据集 — 跨域实验 (AutoKary, 新增)

| 实验名称 | SwanLab项目 | 本地路径 | 配置文件 | mAP | 状态 | 分类 |
|----------|------------|----------|----------|-----|------|------|
| Zero-shot A4 → AutoKary | cross-domain-autokary | work_dirs/cross_domain/ | zero_shot_a4.py | 待确认 | ✅/⚠ | 跨域 |
| Zero-shot A2 → AutoKary | cross-domain-autokary | work_dirs/cross_domain/ | zero_shot_a2.py | 待确认 | ✅/⚠ | 跨域 |
| Finetune A2 k=5 | cross-domain-autokary | work_dirs/cross_domain/ | finetune_a2_k5.py | 待确认 | ✅/⚠ | 跨域 |
| Finetune A2 k=10 | cross-domain-autokary | work_dirs/cross_domain/ | finetune_a2_k10.py | 待确认 | ✅/⚠ | 跨域 |
| Finetune A3 k=5 | cross-domain-autokary | work_dirs/cross_domain/ | finetune_a3_k5.py | 待确认 | ✅/⚠ | 跨域 |
| Finetune A3 k=10 | cross-domain-autokary | work_dirs/cross_domain/ | finetune_a3_k10.py | 待确认 | ✅/⚠ | 跨域 |
| Finetune A4 k=5 | cross-domain-autokary | work_dirs/cross_domain/ | finetune_a4_k5.py | 待确认 | ✅/⚠ | 跨域 |
| Finetune A4 k=10 | cross-domain-autokary | work_dirs/cross_domain/ | finetune_a4_k10.py | 待确认 | ✅/⚠ | 跨域 |

### 1.5 Chromosome20240904 (chromo) 数据集 — ⚠ 旧数据集 (结论暂时废弃)

| 实验名称 | SwanLab项目 | run_id | 本地路径 | mAP | 状态 | 分类 |
|----------|------------|--------|----------|-----|------|------|
| DDPM seed_42 | ldmdet-ablation | apfn46t67iqg1bjraq8xd | work_dirs/multi_seed_aug/ddpm/seed_42/ | 0.726 | ✅ | chromo基线 |
| DDPM seed_789 | ldmdet-ablation | hny1od5fcvx8ngt9b063g | work_dirs/multi_seed_aug/ddpm/seed_789/ | 0.727 | ✅ | chromo基线 |
| DDPM seed_123 | ldmdet-ablation | ghghjry3bylt0sfoj30j5 | work_dirs/multi_seed_aug/ddpm/seed_123/ | 0.733 | ✅ | chromo基线 |
| RF+Heun+AdaLN seed_42 | ldmdet-ablation | 4xhp5ffymboa05hyn245u | work_dirs/multi_seed_aug/rf_heun_adaln/seed_42/ | 0.745 | ✅ | chromo基线 |
| RF+Heun+AdaLN seed_789 | ldmdet-ablation | ww6nlti3ufdkm4htjg5pw | work_dirs/multi_seed_aug/rf_heun_adaln/seed_789/ | 0.747 | ✅ | chromo基线 |
| RF+Heun+AdaLN seed_123 | ldmdet-ablation | dimdbu8fk0re4satbzpgs | work_dirs/multi_seed_aug/rf_heun_adaln/seed_123/ | 0.747 | ✅ | chromo基线 |
| Hard OT seed_42 | ldmdet-ablation | h8fizm7lmc9v5xzxi8ufj | work_dirs/multi_seed_aug/hard_ot/seed_42/ | 0.747 | ✅ | chromo耦合 |
| Hard OT seed_123 | ldmdet-ablation | kka4nra9qk3wanx7i9og1 | work_dirs/multi_seed_aug/hard_ot/seed_123/ | 0.747 | ✅ | chromo耦合 |
| Sinkhorn Stoch seed_42 | ldmdet-ablation | 9ca697vnm1l3koccbenif | work_dirs/multi_seed_aug/sinkhorn_stochastic/seed_42/ | 0.748 | ✅ | chromo耦合 |
| StochOT ε=5 seed_42 (old) | ldmdet-ablation | — | work_dirs/multi_seed/stochot_eps5_old/seed_42/ | 0.746 | ✅ | chromo耦合 (3-seed: 0.747±0.002) | <!-- verified: 2026-07-16 -->
| StochOT ε=5 seed_123 (old) | ldmdet-ablation | — | work_dirs/multi_seed/stochot_eps5_old/seed_123/ | 0.746 @ ep57 | ✅ | chromo耦合 | <!-- verified: 2026-07-16 -->
| StochOT ε=5 seed_789 (old) | ldmdet-ablation | — | work_dirs/multi_seed/stochot_eps5_old/seed_789/ | 0.749 @ ep69 | ✅ | chromo耦合 | <!-- verified: 2026-07-16: scalars.json confirmed -->
| StochOT ε=2 seed_42 (old) | ldmdet-ablation | — | work_dirs/ablation_old/stochot_eps2/ | 0.749 @ ep75 | ✅ | ε 消融 | <!-- verified: 2026-07-16 -->
| SOTA seed_42 | chromosome-kd | 9xswp5aj7rmfd4vys6906 | work_dirs/sota_seed42/ | 0.740 | ✅ | chromo SOTA |
| SOTA seed_123 | chromosome-kd | b31e1xhzftod7ae38cs17 | work_dirs/sota_seed123/ | 0.749 | ✅ | chromo SOTA |
| SOTA seed_456 | chromosome-kd | ukxsyw666y1xrpcrtjvx1 | work_dirs/sota_seed456/ | 0.746 | ✅ | chromo SOTA |
| SOTA seed_1000 | chromosome-kd | ocdkvjs2vs1vozbh0goni | work_dirs/sota_seed1000/ | 0.749 | ✅ | chromo SOTA |
| E4.3 非线性轨迹 (OLD) | ldmdet-ablation | usnvd63f | work_dirs/nonlinear_trajectory/ | 0.752 | ✅ 早停 | 非线性(证伪) |
| E4.3 eps2 (OLD, SCRF未启用) | ldmdet-ablation | cdtmijl0 | work_dirs/nonlinear_trajectory_e43_eps2/ | 0.752 | ✅ 早停 | 非线性(证伪) |
| E4.3 eps2 (NEW, SCRF启用) | ldmdet-ablation | hkn0fc7w | work_dirs/nonlinear_trajectory_e43_eps2_real/ | 0.741 | ✅ 早停 | 非线性(证伪) |
| E4.2 (OT Flow only) | ldmdet-ablation | wcp34v3t | work_dirs/nonlinear_trajectory_e42/ | 0.751 | ✅ 早停 | 非线性 |
| E4.1 (SCRF only, 无OT) | ldmdet-ablation | qdnw5yyj | work_dirs/nonlinear_trajectory_e41/ | 0.743 | ✅ 早停 | 非线性 |
| Direction D (BoxRefineNet) | ldmdet-ablation | fnoz9x82aor1utsuo0jtl | work_dirs/direction_exps/direction_d_box_refine/ | 0.747 | ✅ 早停 | 方向实验 |
| Direction B (DecoupledHead) | ldmdet-ablation | 2ckzmso4c94fojr8jobej | work_dirs/direction_exps/direction_b_decoupled_head/ | 0.749 (max@step74) / 0.702@step25 | ⚠ 实际未中断 (86 evals) | 方向实验(需复核) | <!-- verified: 2026-07-16 -->
| FBM E6.2 frozen | ldmdet-ablation | qyzudrgb | work_dirs/gen_transfer_phase1_e6_2_frozen/ | 0.737 | ✅ 已停止 | 生成迁移(证伪) |
| FBM E6.3 enhanced | ldmdet-ablation | m2fyzmf9 | work_dirs/gen_transfer_phase1_e6_3_enhanced/ | 0.703 | ✅ 已停止 | 生成迁移(证伪) |
| FBM E6.3b frozen_enhanced | ldmdet-ablation | bbe2yrcg | work_dirs/gen_transfer_phase1_e6_3b_frozen_enhanced/ | 0.696 | ✅ 已停止 | 生成迁移(证伪) |
| FBM E6.4 crossattn | ldmdet-ablation | x8j5l7mw | work_dirs/gen_transfer_phase1_e6_4_crossattn/ | 0.733 | ✅ 已停止 | 生成迁移(证伪) |
| E4.3 eps3 | ldmdet-ablation | 5m1lse6x | work_dirs/nonlinear_trajectory_e43_eps3/ | 0.750 | ✅ 早停 | 非线性 |
| E4.3 multinomial | ldmdet-ablation | l0991c8v | work_dirs/nonlinear_trajectory_e43_multinomial/ | 0.748 | ✅ 早停 | 非线性 |
| 3-seed 复现 seed1 | nonlinear-3seed-repro | k7nnzvuq | work_dirs/nonlinear_trajectory_seed1/ | 0.746 | ✅ 早停 | 非线性复现 |
| 3-seed 复现 seed2 | nonlinear-3seed-repro | clpof6nn | work_dirs/nonlinear_trajectory_seed2/ | 0.749 | ✅ 早停 | 非线性复现 |
| Bottleneck: focal_gamma_3 | ldmdet-ablation | ye6a2whory9y67tnvalg3 | work_dirs/bottleneck/ablation/focal_gamma_3/ | 0.750 | ✅ | 瓶颈分析(最优) | <!-- verified: 2026-07-16 -->
| Bottleneck: focal_gamma_1_5 | ldmdet-ablation | 1f0r738snqeljucu6jm03 | work_dirs/bottleneck/ablation/focal_gamma_1_5/ | 0.747 | ✅ | 瓶颈分析 |
| Bottleneck: scale_aware_loss | ldmdet-ablation | o8elke3rcmln1i47fznr5 | work_dirs/bottleneck/ablation/scale_aware_loss/ | 0.742 | ✅ | 瓶颈分析 |
| Bottleneck: relative_l1_loss | ldmdet-ablation | cmxgctni1n1d9g9go0ywf | work_dirs/bottleneck/ablation/relative_l1_loss/ | 0.740 | ✅ | 瓶颈分析 |
| Bottleneck: high_cls_weight | ldmdet-ablation | m9dnqrl43khkl3jcbjico | work_dirs/bottleneck/ablation/high_cls_weight/ | 0.739 | ✅ | 瓶颈分析 |
| Bottleneck: high_giou_weight | ldmdet-ablation | b3w4gwly842j700yqg9og | work_dirs/bottleneck/ablation/high_giou_weight/ | 0.737 | ✅ | 瓶颈分析 |
| Bottleneck: no_box_renewal | ldmdet-ablation | 52o1g5pq09eddplyzbl9q | work_dirs/bottleneck/ablation/no_box_renewal/ | 0.730 | ✅ | 瓶颈分析(消融) |
| Direction D' (RegCalibration) | ldmdet-ablation | — | work_dirs/direction_exps/direction_d_prime_reg_calibration/ | 0.737 | ✅ 早停 | 方向实验 |
| Direction D+B (DecoupledBoxRefine) | ldmdet-ablation | — | work_dirs/direction_exps/direction_db_decoupled_box_refine/ | 0.743 | ✅ 早停 | 方向实验 |
| Direction E (ClassBalanced) | ldmdet-ablation | — | work_dirs/direction_exps/direction_e_class_balanced/ | 0.746 | ✅ 早停 | 方向实验 |
| Direction F' (StructuredPriorOnly) | ldmdet-ablation | — | work_dirs/direction_exps/direction_f_prime_structured_prior_only/ | 0.685 | ✅ 早停 | 方向实验(证伪) |
| Direction F (StructuredPrior) | ldmdet-ablation | — | work_dirs/direction_exps/direction_f_structured_prior/ | 0.574 | ✅ 早停 | 方向实验(证伪) |
| Direction G (LaMFPN) | ldmdet-ablation | — | work_dirs/direction_exps/direction_g_lamfpn/ | 0.736 | ✅ 早停 | 方向实验(证伪) | <!-- verified: 2026-07-16 -->

### 1.6 跨数据集合并实验

| 实验名称 | SwanLab项目 | run_id | 本地路径 | mAP | 状态 | 分类 |
|----------|------------|--------|----------|-----|------|------|
| Merged Sinkhorn seed_42 | ldmdet-ablation | mbgo9qcz95l1aik8sv7bu | work_dirs/merged_ablation/sinkhorn/seed_42/ | 0.806 | ✅ | 跨数据集 |
| Merged GHSS seed_42 | ldmdet-ablation | 503pfk8isr270atpubho1 | work_dirs/merged_ablation/ghss/seed_42/ | 0.000 | ❌ FAILED | 跨数据集 |

---

## 二、按数据集分组

### 2.1 24obj 数据集实验

#### 2.1.1 主路线消融 A0-A4 (项目 `ldmdet-mainline-ablation-24obj`, ⭐ 论文核心)

| 实验 | mAP | AP50 | AP75 | APs | APm | APl | ΔmAP | 说明 |
|------|-----|------|------|-----|-----|-----|------|------|
| A0 baseline (Euler 1步, 无RF) | 0.774 | 0.968 | 0.916 | 0.317 | 0.773 | 0.775 | — | 基线 |
| A1 +RF+Heun | 0.856 | 0.990 | 0.971 | 0.563 | 0.853 | 0.913 | **+0.082** | 主要贡献 |
| A2 +AdaLN-Zero | 0.856 | 0.990 | 0.972 | 0.565 | 0.853 | 0.919 | +0.000 | 持平 A1 |
| A3 +StochOT eps5 | 0.858 | 0.990 | 0.973 | 0.586 | 0.855 | 0.908 | +0.002 | 边际 |
| A4 DPM-Solver++替换Heun | **0.863** | 0.990 | 0.974 | 0.583 | 0.859 | 0.914 | +0.005 | 推理加速且精度提升 |

> 数据来源: `results/swanlab_export.json` (SwanLab API 导出, 2026-07-15)
> ⚠ 命名注意: 论文 (AAAI 草稿) 中 A0-A3 对应此处 A0, A1+A2 合并, A3, A4。详见 §5 数据完整性 C4。

#### 2.1.2 耦合策略消融 (项目 `ldmdet-ablation`, 3 seeds)

| 耦合策略 | seed 42 | seed 123 | seed 789 | mAP (mean±std) |
|----------|---------|----------|----------|----------------|
| Random | 0.859 | 0.814/0.860 ⚠ | 0.860 | 0.860±0.001 (排除 seed123) |
| GHSS | 0.857 | 0.859 | 0.859 | 0.858±0.001 |
| Sinkhorn Stochastic | 0.856 | — | — | 0.856 (1 seed) |

> 核心发现: 24obj 上 Random ≥ GHSS ≥ Sinkhorn Stochastic, OT 耦合优势在大数据集下减弱。

#### 2.1.3 其他实验性尝试 (24obj)

| 实验 | mAP | Δ vs A3(0.858) | 状态 | 说明 |
|------|-----|----------------|------|------|
| SC-RF (自条件化RF) | 0.860 | +0.002 | ✅ 已归档 | **超越 A3**, 但 vs A4(0.863) 仍为 -0.003 负增益; 已于 2026-07-11 归档 | <!-- verified: 2026-07-16: 本地 scalars.json max=0.860 (count=112, best@ep82) -->
| h_velocity_loss | 0.856 | -0.002 | ⚠ CRASHED | 速度损失 |
| n_cascade_e2e | 0.684 | -0.174 | ✅ FINISHED | 端到端级联, 显著退化 (证伪) |
| FBM CrossAttn (源预训练) | 0.857 | -0.001 | ⚠ CRASHED | 与 A3 持平 |
| FBM SimpleGate (源预训练) | 0.677 | -0.181 | ❌ 已停止 | 严重退化 |
| A2+SwiGLU | 0.859 | +0.001 | ✅/⚠ | FFN 激活函数替换 (新增) | <!-- verified: 2026-07-16: 本地 scalars.json max=0.859 (count=76) -->
| A4+SwiGLU | 0.857 | -0.001 | ✅/⚠ | FFN 激活函数替换 (新增) | <!-- verified: 2026-07-16: 本地 scalars.json max=0.857 (count=66) -->
| SetDiff | 0 (早期) | — | ✅/⚠ | Coupled State Diffusion (新增); 仅 7 次评估, 训练极早期 | <!-- verified: 2026-07-16 -->
| PD-RF 渐进蒸馏 | 0.851 | -0.007 | ✅/⚠ | Progressive Distillation (已归档, 2026-07-11) | <!-- verified: 2026-07-16: 本地 scalars.json max=0.851 (count=31) -->
| I1 Seesaw+Normalized | 0.744 | — | ✅/⚠ | ⚠ 实为 chromo 数据集 (非 24obj), 误列入本节 | <!-- verified: 2026-07-16 -->

### 2.2 Chromosome20240904 (chromo) 数据集实验 — ⚠ 暂时废弃

> ⚠ 以下结论基于旧数据集 (mAP≈0.72-0.75), 24obj 数据集上的结论已更新。以下实验记录保留供参考, 但不可与 24obj 实验对比。

#### 2.2.1 主路线消融 (chromo, 旧)

| 实验 | mAP | 配置 | 说明 |
|------|-----|------|------|
| DDPM baseline (3 seeds) | 0.729±0.003 | diffusiondet_ddpm.py | 根基线 |
| RF+Heun+AdaLN (3 seeds) | 0.746±0.001 | rf_heun_adaln.py | +0.017 主要贡献 |
| +DPM-Solver++ (推理) | 0.746±0.001 | +test --solver-type dpm_solver_pp | 持平 Heun (步数对齐) |
| +Hard OT (2 seeds) | 0.747±0.000 | hard_ot.py | +0.001 边际 |
| +Sinkhorn Stochastic (1 seed) | 0.748 | sinkhorn_stochastic.py | +0.002 边际 |
| SOTA (4 seeds) | 0.746±0.004 | sota_seed*.py | 高方差 (seed_123 取最终运行, 排除中断值 0.727) | <!-- verified: 2026-07-16: 4 seeds [0.740, 0.749, 0.746, 0.749], sample_std=0.0042; pop_std=0.0037 -->

> ⚠ **StochOT ε=5 数据源差异说明**: 论文 §4.4.3 epsilon 消融 (ε=0.5→0.710, ε=1.0→0.745, ε=2.0 stochastic→0.749, ε=5.0→0.746, ε=2.0 argmax→0.752) 与 EXPERIMENT_RESULTS.md §5 旧值 (0.720-0.738) 不匹配, 可能来自不同实验批次 (nonlinear_trajectory 系列)。本地仅 seed_42=0.748 可直接验证。详见 §5 C15/C16。 <!-- verified: 2026-07-16 -->

#### 2.2.2 非线性轨迹实验 (chromo, 证伪)

| 实验 | mAP | Δ vs 0.746 | 说明 |
|------|-----|-----------|------|
| E4.1 (SCRF only, 无OT) | 0.743 | -0.003 | 持平 |
| E4.2 (OT Flow only) | 0.751 | +0.005 | OT 耦合有效 |
| E4.3 (OT+SCRF, SCRF未启用) | 0.752 | +0.006 | 实际来自 OT, 非 SCRF |
| **E4.3 eps2 (SCRF真正启用)** | **0.741** | **-0.005** | ⛔ 证伪! SCRF 有害 |
| E4.3 eps3 (argmax, ε=3.0) | 0.750 | +0.004 | OT 耦合有效 (LINEAGE: 5m1lse6x, best@ep70) |
| E4.3 multinomial (multinomial采样) | 0.748 | +0.002 | OT 耦合有效 (LINEAGE: l0991c8v, best@ep72) |
| E6-EMA | 0.739 | -0.007 | EMA 退化 |
| E6-Muon | 0.744 | -0.002 | 持平 |
| E7-tmax100 | 0.745 | -0.001 | 持平 |
| 3-seed 复现 seed1 | 0.746 | +0.000 | 持平 (nonlinear-3seed-repro, k7nnzvuq, best@ep72) |
| 3-seed 复现 seed2 | 0.749 | +0.003 | 持平 (nonlinear-3seed-repro, clpof6nn, best@ep110) |

> 3-seed 复现均值 (seed1+seed2): 0.7475±0.0015, seed3 失败 (仅 2 epoch 即被杀)。证实 0.752 的高方差。 <!-- verified: 2026-07-16 -->

#### 2.2.3 方向实验 / 生成迁移 / Bottleneck (chromo)

| 实验 | mAP | Δ vs 0.746 | 说明 |
|------|-----|-----------|------|
| Direction D (BoxRefineNet) | 0.747 | +0.001 | 持平 |
| Direction B (DecoupledHead) | 0.749 (max) / 0.702@step25 | +0.003 / -0.044 | ⚠ 需复核 (训练未中断, 86 evals; 原文档误记 0.702 为最终值) | <!-- verified: 2026-07-16 -->
| Direction D' (RegCalibration) | 0.737 | -0.009 | 🔴 未超越 baseline (local: direction_d_prime_reg_calibration) |
| Direction D+B (DecoupledBoxRefine) | 0.743 | -0.003 | 持平 (local: direction_db_decoupled_box_refine) |
| Direction E (ClassBalanced) | 0.746 | +0.000 | 持平 (local: direction_e_class_balanced) |
| Direction F' (StructuredPriorOnly) | 0.685 | -0.061 | ⛔ 显著退化 (local: direction_f_prime_structured_prior_only, 仅 29 evals) |
| Direction F (StructuredPrior) | 0.574 | -0.172 | ⛔ 严重退化 (local: direction_f_structured_prior) |
| Direction G (LaMFPN) | 0.736 | -0.010 | 🔴 未超越 baseline (local: direction_g_lamfpn) |
| Bottleneck: focal_gamma_3 | 0.750 | +0.004 | 分类损失调整有效 |
| Bottleneck: focal_gamma_1_5 | 0.747 | +0.001 | 持平 (local: focal_gamma_1_5) |
| Bottleneck: scale_aware_loss | 0.742 | -0.004 | 持平 (local: scale_aware_loss) |
| Bottleneck: relative_l1_loss | 0.740 | -0.006 | 略降 (local: relative_l1_loss) |
| Bottleneck: high_cls_weight | 0.739 | -0.007 | 略降 (local: high_cls_weight) |
| Bottleneck: high_giou_weight | 0.737 | -0.009 | 🔴 未超越 baseline (local: high_giou_weight) |
| Bottleneck: no_box_renewal | 0.730 | -0.016 | 🔴 消融: 去除 box_renewal 显著退化 (local: no_box_renewal) |
| FBM E6.2 frozen | 0.737 | -0.009 | 🔴 未超越 baseline |
| FBM E6.3 enhanced | 0.703 | -0.043 | 🔴 显著退化 (LINEAGE: m2fyzmf9, best@ep19) |
| FBM E6.3b frozen_enhanced | 0.696 | -0.050 | 🔴 显著退化 (LINEAGE: bbe2yrcg, best@ep16) |
| FBM E6.4 crossattn | 0.733 | -0.013 | 🔴 gamma 零初始化失效 | <!-- verified: 2026-07-16 -->

#### 2.2.4 Reflow 退化陷阱 (chromo, 论文 §4.7)

> Reflow (velocity loss) 导致检测与速度目标梯度冲突。两阶段训练 (det → freeze+vel) 可保持 mAP。

| 方法 | lr | Velocity Loss | 梯度冲突 | Best mAP | 稳定性 |
|------|-----|---------------|---------|----------|--------|
| Reflow baseline | 5e-6 | ✓ | cos=−0.104, 86.8% | 0.739 | ❌ |
| Freeze shared | 5e-6 | ✓ | 消除 | 0.740 | ✅ |
| PCGrad | 5e-6 | ✓ | 投影 | 0.724 | ❌ |
| Det Only | 1e-6 | ✗ | N/A | 0.741 | ✅ |

> 数据源: 论文 §4.7。本地无对应 work_dirs (实验在 ross 服务器)。 <!-- verified: 2026-07-16 -->

#### 2.2.5 Step-wise 推理效率对比 (chromo, 论文 §4.2.3)

| 方法 | 1-step | 2-step | 4-step | 8-step |
|------|--------|--------|--------|--------|
| **RF+Heun** | **0.725** | **0.732** | **0.734** | **0.735** |
| DDPM | 0.628 | 0.668 | 0.672 | 0.672 |

> RF 4-step (0.734 @ 219ms) vs DDPM 8-step (0.672 @ 243ms) → **+6.2% mAP at lower latency**。
> 数据源: 论文 §4.2.3。 <!-- verified: 2026-07-16 -->

#### 2.2.6 历史 SOTA 归档实验 (chromo, ldmdet-experiment)

> 归档路径: `ldmdet-experiment/sota/<phase>/<exp_name>/` (2026-06-13 归档)

| 实验 | mAP | 归档路径 | 说明 |
|------|-----|----------|------|
| reproduce_0751_stochot_eps5_v2 | **0.753** | phase5_stochastic_ot/ | 历史最高 (Stochastic OT ε=5) |
| scheme_C1_5_mixed_rel_l1_lam015 | 0.752 | phase7_loss/ | + mixed_relative_l1 损失 |
| ldmdet_kcec_redundant_slots_m2 | 0.748 | phase8_kcec/ | KCEC 冗余槽位 |
| ldmdet_dpm_solver_pp_o2_s8 | 0.748 | phase9_dpm_solver/ | DPM-Solver++ (旧 baseline) |
| smallobj_B_scaleaware_loglinear | 0.744 | phase7_small_obj/ | 小目标尺度感知 |
| ldmdet_daec_contrastive_kcec_eps5 | 0.740 | phase8_daec/ | DAEC 对比学习 |
| ldmdet_convnextv2_mae | 0.736 | phase0_pretrain/ | ConvNextV2 MAE 预训练 |

> 数据源: EXPERIMENT_LINEAGE.md §3.1 (line 137-150)。 <!-- verified: 2026-07-16 -->

### 2.3 跨数据集 / 少样本实验

#### 2.3.1 合并数据集训练 (24obj + chromo)

| 耦合 | mAP | 状态 |
|------|-----|------|
| Sinkhorn Stochastic | 0.806 | ✅ |
| GHSS | 0.000 | ❌ FAILED |

#### 2.3.2 Few-Shot 基准 (24obj 源预训练 → chromo 目标微调)

| 源预训练模型 | mAP (24obj) | 状态 | 目标微调 |
|-------------|-------------|------|----------|
| LDMDet SOTA | 已完成 | ✅ | ⛔ 尚未启动 |
| LDMDet FBM CrossAttn | 0.857 | ⚠ CRASHED | ⛔ 尚未启动 |
| LDMDet FBM SimpleGate | 0.677 | ❌ 已停止 | ⛔ 尚未启动 |
| Cascade R-CNN R50 | 已完成 | ✅ | ⛔ 尚未启动 |
| DINO R50 | 已完成 | ✅ | ⛔ 尚未启动 |
| RTMDet-L | 已完成 | ✅ | ⛔ 尚未启动 |
| YOLOX-S | 已完成 | ✅ | ⛔ 尚未启动 |

> 目标微调配置已就绪: `experiments/configs/few_shot/target_finetune/*_k{5,10}.py` (14 个)

#### 2.3.3 跨域 AutoKary (新增, 24obj → AutoKary2022)

| 实验 | 配置 | 说明 |
|------|------|------|
| Zero-shot A2/A4 | zero_shot_a{2,4}.py | 24obj 训练 → AutoKary 直接评估 |
| Finetune A2/A3/A4 × k{5,10} | finetune_a{2,3,4}_k{5,10}.py | 24obj 预训练 → AutoKary 少样本微调 |

---

## 三、按实验类型分类

### 3.1 核心消融实验 (论文引用)

> 论文: [AAAI_INTEGRATED_DRAFT.md](paper/AAAI_INTEGRATED_DRAFT.md), Target: AAAI 2027

| 实验 | 论文引用位置 | mAP | 论文中的命名 | 本目录命名 |
|------|-------------|-----|-------------|-----------|
| A0 baseline | §4.2.1 Table, §3.2.3 | 0.774 | A0 baseline | A0 |
| A1 RF+Heun+AdaLN | §4.2.1 Table | 0.856 | A1 RF+Heun+AdaLN | A1+A2 合并 |
| A2 +StochOT | §4.2.1 Table, §3.3, §4.4 | 0.858 | A2 +StochOT | A3 |
| A3 DPM-Solver++ | §4.2.1 Table, §3.2, §4.5 | 0.863 | A3 DPM-Solver++ | A4 |
| Random 2-seed | §4.4.1 | 0.860±0.001 | Random coupling | 24obj_ablation/random |
| DDPM vs RF (chromo 3-seed) | §4.2.2 | 0.729 vs 0.746 | RF vs DDPM | multi_seed_aug/ |
| Solver×step 解耦 | §3.2.3 | 0.855/0.851 | Solver ablation | A1 checkpoint 评估 |
| DPM++ 步数消融 | §4.5.1 | 0.860-0.863 | Step ablation | A4 checkpoint 评估 |
| ε 消融 | §4.4.3 | 0.710-0.752 | Epsilon ablation | ablation/ |
| FPS Benchmark | §4.6 | — | FPS/latency | results/benchmark_fps_* |
| Per-class AP | §4.3.1 | — | Per-class analysis | results/a4_per_class_ap.md |
| Test set 评估 | §4.5.4 | 0.859 | Test set | A4 checkpoint |

### 3.2 探索性实验 (已归档/证伪)

#### 3.2.1 SetDiff 系列 (Coupled State Diffusion)

| 实验 | 配置 | 本地路径 | SwanLab项目 | 状态 |
|------|------|----------|------------|------|
| SetDiff baseline (24obj) | setdiff_24obj.py | work_dirs/setdiff_24obj/ | setdiff-24obj | ✅/⚠ (早期, 仅 7 evals, max mAP=0) | <!-- verified: 2026-07-16 -->
| SetDiff baseline (chromo) | setdiff_baseline.py | — | — | — |

> 方向: 用耦合状态扩散替代标准噪声→box 路径。代码审查见 `docs/research/setdiff_code_review.md`。

#### 3.2.2 PD-RF 蒸馏系列 (Progressive Distillation)

| 实验 | 配置 | 本地路径 | mAP | 状态 |
|------|------|----------|-----|------|
| PD-RF 24obj | pd_rf_24obj.py (备份) | work_dirs/pd_rf_24obj/ | 0.851 | ⛔ 证伪 (2026-07-11 归档) | <!-- verified: 2026-07-16: 本地 scalars.json max=0.851 (count=31) -->

> 方向: 渐进式蒸馏 Rectified Flow, 逐步减少采样步数。提案见 `docs/paper/proposals/PD-RF_Progressive_Distillation.md`。⚠ 方向已证伪: 1-step Euler 无法近似 4-step DPM-Solver++ 预测, 蒸馏梯度与检测梯度冲突 (详见项目记忆 PD-RF 归档记录)。 <!-- verified: 2026-07-16 -->

#### 3.2.3 SC-RF 自条件化 (Self-Conditioned RF)

| 实验 | 配置 | 本地路径 | SwanLab项目 | mAP | 状态 |
|------|------|----------|------------|-----|------|
| SC-RF 24obj | sc_rf_24obj.py (备份) | work_dirs/sc_rf_24obj/ | ldmdet-breakthrough | 0.860 | ✅ 已归档 | <!-- verified: 2026-07-16: 本地 scalars.json max=0.860 (count=112, best@ep82); 2026-07-11 归档 -->
| SC-RF chromo (E4.3 NEW) | nonlinear_trajectory_e43_eps2.py | work_dirs/nonlinear_trajectory_e43_eps2_real/ | ldmdet-ablation | 0.741 | ⛔ 证伪 |

> ⛔ chromo 上 SCRF 真正启用后 0.741 < 0.746 baseline, 方向证伪。24obj 上 SC-RF (0.860) **超越 A3 (0.858) +0.002**, 但 vs A4 (0.863) 仍为 -0.003 负增益, 自条件化在 4 维 bbox 信息瓶颈下无显著优势, 已于 2026-07-11 归档。详见 EXPERIMENT_LINEAGE.md §11。 <!-- verified: 2026-07-16 -->

#### 3.2.4 CAT Loss 系列 / TRD / Velocity

| 实验 | 说明 | 状态 |
|------|------|------|
| TRD (Trajectory Regularization Delta) | 轨迹正则化, 复用 cat_delta_t | ⚠ 实现偏差, 需重跑 |
| CAT (Curvature Consistency) | x0 一致性非纯曲率 | ⚠ 实现偏差, 需重跑 |
| velocity_loss | 速度损失 | ⚠ CRASHED (24obj) |
| h_cfm_velocity | CFM 速度匹配 | ❌ FAILED |

#### 3.2.5 其他方向

| 实验 | mAP | 状态 | 说明 |
|------|-----|------|------|
| Direction B (DecoupledHead) | 0.749 (max) / 0.702@step25 | ⚠ 需复核 | 解耦头 max 超过 baseline, 原结论需复核 | <!-- verified: 2026-07-16 -->
| Direction D (BoxRefineNet) | 0.747 | ✅ 持平 | 框精修无提升 |
| FBM E6.2-E6.4 (生成迁移) | 0.696-0.737 | ⛔ 证伪 | FBM 均未超越 baseline |
| n_cascade_e2e | 0.684 | ⛔ 证伪 | 端到端级联失败 |
| DINOv2-S backbone | 0.502-0.676 | ⛔ 证伪 | DINOv2 全失败 |
| BiFPN (scheme_E) | 0.744 | ⛔ 未采用 | BiFPN 无提升 |
| CSPNeXt-L backbone | 0.730 | ⛔ 证伪 | CSPNeXt 退化 |
| I1 Seesaw+Normalized | 0.744 | ✅/⚠ | 长尾类别平衡 (⚠ chromo 数据集, 非 24obj) | <!-- verified: 2026-07-16 -->
| SwiGLU FFN (A2/A4) | 0.859 / 0.857 | ✅/⚠ | 门控激活函数 (A2=0.859 +0.001, A4=0.857 -0.001) | <!-- verified: 2026-07-16 -->
| PD-RF 渐进蒸馏 | 0.851 | ⛔ 证伪 | 渐进蒸馏 4→1 失败, 2026-07-11 归档 | <!-- verified: 2026-07-16 -->
| IO1 自适应步数终止 | — | ⛔ 证伪 | x0_Δrel 最低 0.166 |
| IO2 投机 Draft-Verify | — | ⛔ 证伪 | 早期步 cls 一致率 57% |
| IO4 级联头提前退出 | — | ⛔ 证伪 | 所有阈值退出率 0% |
| IO5 跨步 RoI 特征缓存 | — | ⛔ 证伪 | 框位移 93-124 px/步 |

### 3.3 基线对比实验 (24obj)

| 方法 | Backbone | mAP | AP50 | AP75 | 状态 | SwanLab |
|------|----------|-----|------|------|------|---------|
| RTMDet-L | CSPNeXt-L | **0.869** | 0.992 | 0.976 | ✅ | chromosome-kd-benchmark-24obj |
| DINO R50 (4scale) | ResNet-50 | 0.868 | 0.992 | 0.979 | ⚠ CRASHED | chromosome-kd-benchmark-24obj |
| **A4 DPM-Solver++ (本文)** | ResNet-50 | **0.863** | 0.990 | 0.974 | ✅ | ldmdet-mainline-ablation-24obj |
| A3 SOTA Heun (本文) | ResNet-50 | 0.858 | 0.990 | 0.973 | ✅ | ldmdet-mainline-ablation-24obj |
| Cascade R-CNN R50 | ResNet-50 | 0.854 | 0.987 | 0.972 | ✅ | chromosome-kd-benchmark-24obj |
| YOLOX-S | CSPDarkNet-S | 0.796 | 0.987 | 0.944 | ✅ | chromosome-kd-benchmark-24obj |
| DiffusionDet | ResNet-50 | 0.787 | 0.970 | 0.928 | ⚠ CRASHED | chromosome-kd-benchmark-24obj |

> 数据来源: `results/swanlab_export.json`。YOLOX-S 论文中修正为 0.796 (原 EXPERIMENT_RESULTS.md 标注 0.803)。

### 3.4 已完成的补充实验 (2026-07-15/16)

| 实验 | 服务器 | SwanLab项目 | 结果 | 说明 |
|------|--------|------------|------|------|
| A4 DPM-Solver++ seed_123 | workstation | ldmdet-mainline-ablation-24obj | **0.857** @ ep62 | A4 多种子 ✅ 完成 (3-seed: 0.863/0.857/0.856, mean 0.859±0.004) |
| A4 DPM-Solver++ seed_789 | workstation | ldmdet-mainline-ablation-24obj | **0.856** @ ep72 | A4 多种子 ✅ 完成 |
| StochOT ε=5 seed_123 (old) | ross | ldmdet-ablation | **0.746** @ ep57 | StochOT 多种子 ✅ 完成 (3-seed: 0.746/0.746/0.749, mean 0.747±0.002) |
| StochOT ε=5 seed_789 (old) | ross | ldmdet-ablation | **0.749** @ ep69 | StochOT 多种子 ✅ 完成 |
| StochOT ε=2 seed_42 (old) | workstation | ldmdet-ablation | **0.749** @ ep75 | ε 消融补充 ✅ 完成 |

> ⚠ SC-RF 24obj 已于 2026-07-11 归档 (max mAP=0.860), 不再运行。A4+SwiGLU 已完成 (0.857)。详见 §2.1.3。 <!-- verified: 2026-07-16 -->
> ✅ 所有多种子补充实验已完成。workstation 上的 checkpoint 待 SCP 到 ross (workstation 连接问题搁置)。

---

## 四、配置 → 模型结构对应关系

### 4.1 核心配置继承关系

```
ldmdet_baseline.py (DDPM 根基线)
├── ldmdet_rf_heun_shifted_bs2.py (+RF+Heun+Shifted, bs=2)
│   ├── ldmdet_rf_heun_adaln_stochot_eps5.py (+AdaLN+StochOT eps5)
│   │   └── a3_full_sota_24obj.py (24obj 数据集覆盖)
│   │       └── a4_dpm_pp_24obj.py (+DPM-Solver++)
│   │           └── a4_dpm_pp_24obj_multiseed.py (多种子)
│   └── a1_rf_heun_24obj.py (24obj, 无 AdaLN)
│   └── a2_rf_heun_adaln_24obj.py (24obj, +AdaLN)
│       └── a2_swinglu_24obj.py (+SwiGLU)
│       └── a4_swinglu_24obj.py (A4+SwiGLU)
├── a0_baseline_24obj.py (24obj, DDPM 基线)

rf_heun_adaln.py (chromo RF+Heun+AdaLN 基线, bs=4)
├── chromo_24obj_random.py (24obj Random)
├── ghss.py → chromo_24obj.py (24obj GHSS)
├── sinkhorn_stochastic.py → chromo_24obj_sinkhorn.py (24obj Sinkhorn)
└── nonlinear_trajectory*.py (chromo 非线性轨迹系列)
```

### 4.2 配置参数对照表

| 配置文件 | diffusion_type | solver_type | time_conditioning | coupling | rf_schedule | rf_shift | sampling_timesteps | batch_size (配置) | batch_size (实际†) | max_epochs |
|----------|---------------|-------------|-------------------|----------|-------------|----------|-------------------|------------------|-------------------|-----------|
| ldmdet_baseline.py | ddpm (默认) | euler (默认) | scale_shift (默认) | random (默认) | uniform (默认) | — | 1 | 4 | 4 | 150 |
| a0_baseline_24obj.py | ddpm | euler | scale_shift | random | uniform | — | 1 | 4 | 4 | 150 |
| ldmdet_rf_heun_shifted_bs2.py | rectified_flow | heun | scale_shift | random | shifted | 3.0 | 4 | 2 | — | 150 |
| a1_rf_heun_24obj.py | rectified_flow | heun | scale_shift (默认) | random | shifted | 3.0 | 4 | 2 | 8 | 150 |
| a2_rf_heun_adaln_24obj.py | rectified_flow | heun | adaln_zero | random | shifted | 3.0 | 4 | 2 | 8 | 150 |
| a3_full_sota_24obj.py | rectified_flow | heun | adaln_zero | ot_flow (eps=5, multinomial) | shifted | 3.0 | 4 | 2 | 8 | 150 |
| a4_dpm_pp_24obj.py | rectified_flow | dpm_solver_pp | adaln_zero | ot_flow (eps=5, multinomial) | shifted | 3.0 | 4 | 2 | 8 | 150 |
| rf_heun_adaln.py (chromo) | rectified_flow | heun | adaln_zero | random | shifted | 3.0 | 4 | 4 | 4 | 150 |
| chromo_24obj_random.py | rectified_flow | heun | adaln_zero | random | shifted | 3.0 | 4 | 4 | 4 | 150 |
| chromo_24obj.py (GHSS) | rectified_flow | heun | adaln_zero | ghss | shifted | 3.0 | 4 | 4 | 4 | 150 |
| chromo_24obj_sinkhorn.py | rectified_flow | heun | adaln_zero | sinkhorn_stochastic | shifted | 3.0 | 4 | 4 | 4 | 150 |
| nonlinear_trajectory.py | rectified_flow | heun | adaln_zero | ot_flow (argmax, eps=1.0) | shifted | 3.0 | 4 | 4 | 4 | 150 |
| nonlinear_trajectory_e43_eps2.py | rectified_flow | heun | adaln_zero | ot_flow (argmax, eps=2.0) | shifted | 3.0 | 4 | 4 | 4 | 150 |
| i1_seesaw_normalized.py | rectified_flow | heun | adaln_zero | ot_flow (eps=5) | shifted | 3.0 | 4 | 2 | — | 150 |
| a2_swinglu_24obj.py | rectified_flow | heun | adaln_zero | random | shifted | 3.0 | 4 | 2 | 8 | 150 |

> † 实际 batch_size: 配置文件中 A1-A4 继承 bs=2, 但 SwanLab 描述均标注 "bs=8", 推测训练时通过命令行 `--cfg-options train_dataloader.batch_size=8` 覆盖。

### 4.3 耦合策略参数详解

| 耦合类型 | type | epsilon | num_iters | coupling_mode | 说明 |
|----------|------|---------|-----------|---------------|------|
| Random | (默认, 无 coupling dict) | — | — | — | 随机配对噪声与 GT |
| Hard OT | (hard_ot) | 0 | — | — | 确定性 OT (匈牙利匹配) |
| Sinkhorn Stochastic | sinkhorn_stochastic | 5.0 | — | — | 从 Sinkhorn 传输矩阵采样 |
| GHSS | ghss | 5.0 | — | — | Group Hierarchical Stochastic Sinkhorn |
| OT Flow (argmax) | ot_flow | 1.0-3.0 | 20 | argmax | OT 传输矩阵 argmax 解码 |
| OT Flow (multinomial) | ot_flow | 5.0 | 20 | multinomial | OT 传输矩阵多项式采样 (A3 配置) |

### 4.4 关键架构参数 (所有实验通用)

| 参数 | 值 |
|------|-----|
| Backbone | ResNet-50 (torchvision 预训练) |
| Neck | FPN (256通道, 4 levels) |
| Head 类型 | PurePyTorchDiffusionDetHead |
| Proposals | 500 |
| Transformer heads (级联) | 6 |
| Deep supervision | ✅ (5 aux heads) |
| Optimizer | AdamW, lr=5e-5, wd=1e-4 |
| LR schedule | Linear warmup 5ep + CosineAnnealing |
| Loss | Focal (cls, 2.0) + L1 (bbox, 5.0) + GIoU (2.0) |
| Matcher | Hungarian (FocalCost 2.0 + L1Cost 5.0 + GIoUCost 2.0) |
| EarlyStopping | patience=30, min_delta=0.001 |

---

## 五、数据完整性检查

### C1: AdaLN-Zero 消融 (A1 vs A2) 结果未在论文中报告

| 项目 | 说明 |
|------|------|
| **问题** | A1 (无 AdaLN) = 0.856, A2 (+AdaLN-Zero) = 0.856, Δ=+0.000。论文 (AAAI 草稿) 将 A1+A2 合并为一步 "A1 RF+Heun+AdaLN", 声称 "AdaLN is integrated into RF, not separately ablated" |
| **证据** | swanlab_export.json: a1_rf_heun max=0.856, a2_rf_heun_adaln max=0.856 (count=92 vs 112) |
| **影响** | AdaLN-Zero 的独立贡献 (+0.000) 被隐藏。论文声称 AdaLN 是 "principled conditional injection", 但消融数据显示无精度提升 |
| **建议** | 如需诚实报告, 应在附录补充 A1 vs A2 消融, 说明 AdaLN-Zero 的价值在于训练稳定性而非精度 |

### C2: Random 24obj seed_123 已完成但论文标记为"训练中断"

| 项目 | 说明 |
|------|------|
| **问题** | EXPERIMENT_LINEAGE.md 和 EXPERIMENT_RESULTS.md §8.2 记录 seed_123 mAP=0.860 (best @ 115, EarlyStop @ 145, ✅完成)。但论文 §4.4.1 标注 "seed=123 训练中断（SwanLab 仅 13 个评估点），不纳入统计" |
| **证据** | swanlab_export.json: chromo_24obj_random_seed123 max mAP=**0.814** (count=**13**, 远少于 seed42 的 89 和 seed789 的 112) |
| **解读** | 本地 scalars.json 可能显示 0.860 (完整训练), 但 SwanLab 云端仅同步了 13 个评估点 (max 0.814)。两种可能: (a) SwanLab 同步中断但本地训练完成; (b) 训练确实中断, 0.860 来自其他 run |
| **影响** | 如果 seed_123 实际完成 (0.860), 则 Random 3-seed mean=0.860±0.001, 论文应纳入; 如果中断, 论文标注正确 |
| **建议** | 核查 ross 服务器 `work_dirs/24obj_ablation/random/seed_123/` 的 scalars.json 确认实际训练 epoch 数和 best mAP |

### C3: StochOT 24obj 有三个不同数值 (0.853/0.856/0.858) 来自不同实验批次

| 数值 | 来源 | 配置差异 | SwanLab项目 |
|------|------|----------|------------|
| **0.853** | ldmdet_rf_heun_adaln_stochot_eps5 (2026-05-27) | 早期实验, bs=2, 无 classwise eval | chromosome-kd-benchmark-24obj |
| **0.856** | 24obj_ablation/sinkhorn/seed_42 (2026-06-26) | sinkhorn_stochastic 耦合, bs=4 | ldmdet-ablation |
| **0.858** | A3 full_sota (2026-07) | ot_flow coupling (eps=5, multinomial), bs=8, classwise eval | ldmdet-mainline-ablation-24obj |

> ⚠ 三者使用不同的耦合实现 (sinkhorn_stochastic vs ot_flow multinomial)、不同的 batch_size (2/4/8)、不同的评估器配置。论文引用 0.858 (A3) 作为主路线 SOTA, 0.856 作为耦合消融。0.853 为早期实验, 已被取代。

### C4: A3/A4 命名在不同章节不一致

| 文档 | A0 | A1 | A2 | A3 | A4 |
|------|----|----|----|----|----|
| **EXPERIMENT_LINEAGE.md** (本目录) | baseline | +RF+Heun | +AdaLN-Zero | +StochOT | DPM-Solver++ |
| **EXPERIMENT_RESULTS.md** | baseline | +RF+Heun | +AdaLN-Zero | +StochOT | DPM-Solver++ |
| **AAAI 论文草稿** | baseline | RF+Heun+AdaLN (合并) | +StochOT | DPM-Solver++ | — |

> 论文将 EXPERIMENT_LINEAGE 中的 A1+A2 合并为 "A1 RF+Heun+AdaLN" (因 AdaLN 消融无差异, 见 C1), 然后 A3→A2, A4→A3。这导致跨文档引用时 A2/A3 指代不同实验。

### 其他数据完整性备注

| 编号 | 问题 | 说明 |
|------|------|------|
| C5 | YOLOX-S mAP 不一致 | EXPERIMENT_RESULTS.md 标注 0.803, swanlab_export.json 和论文修正为 **0.796** |
| C6 | SOTA chromo 高方差 | 4-seed: 0.740/0.749/0.746/0.749 (mean 0.746±0.004 sample_std; pop_std=0.0037), 低于 sinkhorn_stochastic 单 seed (0.748)。注: seed_123 有 3 个运行目录, 取最终运行 (0.749), 排除中断值 (0.727)。原 Catalog 误记 0.742±0.008 因采用中断值。 <!-- verified: 2026-07-16 --> |
| C7 | E4.3 SCRF 归因错误 | 0.752 原归功于 ScaleConditionedRF, 实际 SCRF 未集成到 head.py; 真正启用后 0.741 证伪 |
| C8 | ~~StochOT 24obj seed_789 缺失~~ ✅ 已解决 <!-- verified: 2026-07-16 --> | 24obj 的 StochOT ε=5 仍为 1 seed (0.858), 但 original 数据集的 StochOT ε=5 已完成 3 seeds (0.746/0.746/0.749, mean 0.747±0.002)。论文 §4.4.1 仍标注 24obj 为 1 seed (准确), §4.4.2 已更新为 3 seeds。 |
| C9 | **SC-RF mAP 原标注 0.857 有误** <!-- verified: 2026-07-16 --> | 本地 `work_dirs/sc_rf_24obj/` scalars.json 实测 max mAP=**0.860** (count=112, best@ep82)。原目录及 EXPERIMENT_RESULTS.md 标注 0.857, 实际 SC-RF **超越 A3 (0.858) +0.002** 而非 "持平"。但 vs A4 (0.863) 仍为 -0.003 负增益, 结论 (SC-RF 无显著优势) 不变。SC-RF 已于 2026-07-11 归档, 非 "运行中"。 |
| C10 | **I1 Seesaw / normalized_only 误列入 24obj 节** <!-- verified: 2026-07-16 --> | 配置 `i1_seesaw_normalized.py` 注释明确 "突破 SOTA 0.746 mAP (chromo)", 实为 chromo 数据集实验; 附录 (§6.3) 也正确标注为 chromo。本地 scalars.json 实测 I1 max=0.744 (count=36), normalized_only max=0.747 (count=71), 均为 chromo 量级。误列入 §1.3/§2.1.3 24obj 探索性实验表。 |
| C11 | **RTMDet-L 本地 scalars.json 不完整** <!-- verified: 2026-07-16 --> | 本地 `work_dirs/baselines/rtmdet_l_24obj/` scalars.json 仅 count=86, max=0.863; EXPERIMENT_LINEAGE.md 标注 best@ep116, 完整日志在 ross 服务器。0.869 来自 EXPERIMENT_LINEAGE.md 和 benchmark_fps_*.md, 本地无法直接验证。 |
| C12 | **DINO R50 无本地目录** <!-- verified: 2026-07-16 --> | `work_dirs/baselines/` 下无 `dino_r50_24obj/` 目录, DINO R50 数据仅存于 SwanLab 云端 (swanlab_export.json 确认 0.868, count=93)。 |
| C13 | **配置继承路径失效** <!-- verified: 2026-07-16 --> | `experiments/configs/multiset/chromo_24obj.py` 继承 `../ldmdet/ghss.py`, `chromo_24obj_sinkhorn.py` 继承 `../ldmdet/sinkhorn_stochastic.py`, 但这两个基文件在 `experiments/configs/ldmdet/` 下不存在 (仅存于 experiments_backup 归档目录)。配置继承图 (§4.1) 引用了已移动/删除的文件。 |
| C14 | Direction B mAP 严重错误 (chromo) <!-- verified: 2026-07-16 --> | 原记录 0.702 (step 25 中间值), 实际 max=0.749 (step 74, 86 evals, 训练未中断)。"显著退化"结论需复核。本地 `work_dirs/direction_exps/direction_b_decoupled_head/` scalars.json 实测确认。 |
| C15 | ~~StochOT ε=5 seed 数量矛盾 (chromo)~~ ✅ 已解决 <!-- verified: 2026-07-16 --> | 3 seeds 已全部完成: seed_42=0.746, seed_123=0.746@ep57, seed_789=0.749@ep69 (ross scalars.json 确认)。论文 §4.4.2 和附录 E.2 已更新为 3 seeds (mean 0.747±0.002)。原 "1 seed" 指的是 multi_seed_aug/sinkhorn_stochastic/seed_42 (0.748), 与 multi_seed/stochot_eps5_old/ 是不同实验目录。 |
| C16 | Epsilon 消融数据源不匹配 (chromo) <!-- verified: 2026-07-16 --> | 论文 §4.4.3 (ε=0.5→0.710, ε=1.0→0.745, ε=2.0 stochastic→0.749, ε=5.0→0.746, ε=2.0 argmax→0.752) 与 EXPERIMENT_RESULTS.md §5 旧值 (0.720-0.738) 不匹配, 可能来自不同实验批次 (nonlinear_trajectory 系列)。本地仅 seed_42=0.748 可直接验证。 |

---

## 六、服务器实验状态

### 6.1 ross 服务器 (主力训练)

| 实验组 | 路径 (ross) | 状态 | 本地同步 |
|--------|------------|------|----------|
| A0-A4 主路线消融 | /media/ross/8TB/.../work_dirs/a{0-4}_*_24obj/ | ✅ 完成 | ✅ 已同步到本地 |
| 24obj 耦合消融 (Random/GHSS/Sinkhorn) | /media/ross/8TB/.../work_dirs/24obj_ablation/ | ✅ 完成 | ✅ 已同步 |
| chromo multi_seed_aug (DDPM/RF/HardOT/Sinkhorn) | /media/ross/8TB/.../work_dirs/multi_seed_aug/ | ✅ 完成 | ✅ 已同步 |
| chromo SOTA multiseed | /media/ross/8TB/.../work_dirs/sota_seed*/ | ✅ 完成 | ✅ 已同步 |
| chromo 非线性轨迹系列 | /media/ross/8TB/.../work_dirs/nonlinear_trajectory*/ | ✅ 完成 | ✅ 已同步 |
| chromo Bottleneck 消融 | /media/ross/8TB/.../work_dirs/bottleneck/ | ✅ 完成 | ✅ 已同步 |
| chromo Direction 实验 | /media/ross/8TB/.../work_dirs/direction_exps/ | ✅ 完成 | ✅ 已同步 |
| chromo 生成迁移 FBM | /media/ross/8TB/.../work_dirs/gen_transfer_phase1_e6_*/ | ✅ 完成 | ✅ 已同步 |
| Few-Shot 源预训练 | /media/ross/8TB/.../work_dirs/few_shot/ | 🔄 部分完成 | ✅ 已同步 |
| SC-RF 24obj | /media/ross/8TB/.../work_dirs/sc_rf_24obj/ | ✅ 已归档 (2026-07-11) | ✅ 已同步 | <!-- verified: 2026-07-16: max mAP=0.860 (count=112) -->
| 跨域 AutoKary | /media/ross/8TB/.../work_dirs/cross_domain/ | 🔄 进行中 | ✅ 已同步 |
| FPS Benchmark | /media/ross/8TB/.../results/benchmark_fps_* | ✅ 完成 | ✅ 已同步 |

### 6.2 workstation 服务器 (A5000/A4000, 并行多种子)

| 实验 | 状态 | 说明 |
|------|------|------|
| A4 DPM-Solver++ seed_123 | ✅ 已完成 (0.857 @ ep62) | A4 多种子: 3-seed mean 0.859±0.004 | <!-- verified: 2026-07-16 -->
| A4 DPM-Solver++ seed_789 | ✅ 已完成 (0.856 @ ep72) | A4 多种子完成 | <!-- verified: 2026-07-16 -->
| StochOT ε=2 seed_42 (old) | ✅ 已完成 (0.749 @ ep75) | ε 消融补充 | <!-- verified: 2026-07-16 -->
| StochOT ε=5 seed_123 (old) | ✅ 已完成 (0.746 @ ep57, ross) | StochOT 多种子完成 (3-seed: 0.747±0.002) | <!-- verified: 2026-07-16 -->
| StochOT ε=5 seed_789 (old) | ✅ 已完成 (0.749 @ ep69, ross) | StochOT 多种子完成 | <!-- verified: 2026-07-16 -->
| SwiGLU 实验 | ✅ 已完成 | A2+SwiGLU=0.859, A4+SwiGLU=0.857 (本地 scalars.json 已确认) | <!-- verified: 2026-07-16 -->

> ✅ 所有多种子补充实验已完成。workstation 上的 checkpoint (A4 seed_123/789, StochOT ε=2) 待 SCP 到 ross (workstation 连接问题搁置)。

### 6.3 本地工作区 (开发 + 分析)

本地 `work_dirs/` 共 48 个子目录, 为 ross 服务器的子集同步。本地新增/独立实验:

| 目录 | 说明 | 仅本地? |
|------|------|---------|
| a2_swinglu_24obj/ | SwiGLU FFN 实验 (2026-07-12) | ✅ 新增 |
| a4_swinglu_24obj/ | SwiGLU FFN 实验 | ✅ 新增 |
| setdiff_24obj/ | SetDiff 实验 (2026-07-14) | ✅ 新增 |
| pd_rf_24obj/ | PD-RF 蒸馏实验 (2026-07-11) | ✅ 新增 |
| cross_domain/ | AutoKary 跨域实验 (2026-07-13) | ✅ 新增 |
| chromogen_phase1_sd15_24obj*/ | ChromoGen SD1.5 生成模型 (24obj) | ✅ 新增 |
| ablation_old/ | 旧 epsilon 消融归档 | ✅ 新增 |

### 6.4 FPS Benchmark 结果 (RTX A6000)

> 数据来源: `results/benchmark_fps_20260714_231841.{md,json}` + `results/benchmark_fps_20260714_234842.{md,json}` + `results/benchmark_fps_20260715_013222.{md,json}` + `results/benchmark_fps_20260716_091515.json` + `results/benchmark_fps_20260716_100627.json` (baseline)
> **Baseline 测试参数**: warmup=100, iters=300, CUDA event timing, 512×512, batch=1 (比 A1-A4 的 warmup=10/iters=100 更严谨)

| 模型 | 采样器 | 步数 | 延迟 (ms) | FPS | mAP |
|------|--------|:---:|----------:|----:|---:|
| A1 (RF+Heun) | Heun | 4 | 124.38 ± 3.38 | 8.0 | 0.856 |
| A3 (SOTA, Heun) | Heun | 4 | 128.35 ± 1.95 | 7.8 | 0.858 |
| **A4 (DPM-Solver++)** | **DPM++** | **4** | **75.03 ± 0.96** | **13.3** | **0.863** |
| A4+IO3 K=300 | DPM++ | 4 | 71.27 ± 2.39 | 14.0 | 0.861 |
| **A4+IO3 K=200** | **DPM++** | **4** | **70.46 ± 2.28** | **14.2** | **0.860** |
| A4+IO3 K=100 | DPM++ | 4 | 69.71 ± 1.98 | 14.3 | 0.850 |
| Cascade R-CNN R50 | — | 1 | 20.67 ± 0.48 | 48.4 | 0.854 |
| YOLOX-S | — | 1 | 10.15 ± 0.41 | 98.5 | 0.796 |
| DiffusionDet | Euler | 1 | 24.38 ± 1.09 | 41.0 | 0.787 |
| RTMDet-L | — | — | 33.06 ± 0.80 | 30.3 | 0.869 |

---

## 附录: 本地 work_dirs 完整目录清单 (48 项)

| 目录 | 数据集 | 说明 | 重要性 |
|------|--------|------|--------|
| 24obj_ablation/ | 24obj | Random/GHSS/Sinkhorn 耦合消融 (3 seeds) | ⭐⭐⭐ |
| a0_baseline_24obj/ | 24obj | A0 baseline (Euler 1步) | ⭐⭐⭐ |
| a1_rf_heun_24obj/ | 24obj | A1 +RF+Heun | ⭐⭐⭐ |
| a2_rf_heun_adaln_24obj/ | 24obj | A2 +AdaLN-Zero | ⭐⭐⭐ |
| a2_swinglu_24obj/ | 24obj | A2 +SwiGLU FFN (新增) | ⭐⭐ |
| a3_full_sota_24obj/ | 24obj | A3 +StochOT eps5 | ⭐⭐⭐ |
| a4_dpm_pp_24obj/ | 24obj | A4 DPM-Solver++ (本文 SOTA) | ⭐⭐⭐ |
| a4_swinglu_24obj/ | 24obj | A4 +SwiGLU FFN (新增) | ⭐⭐ |
| ablation/ | chromo | epsilon 消融 (stoch_eps*, argmax_eps*) | ⭐⭐ |
| ablation_old/ | chromo | 旧 epsilon 消融归档 | ⭐ |
| baselines/ | 24obj | 对比模型 (Cascade/DINO/YOLOX/RTMDet/DiffusionDet) | ⭐⭐⭐ |
| bottleneck/ | chromo | 瓶颈分析 (focal_gamma, scale_aware, 等) | ⭐⭐⭐ |
| chromo_coco_detection/ | chromo | 数据集配置 | ⭐ |
| chromogen_phase1/ | — | ChromoGen 生成模型训练 | ⭐⭐ |
| chromogen_phase1_sd15/ | — | ChromoGen SD1.5 生成模型 | ⭐⭐ |
| chromogen_phase1_sd15_24obj/ | 24obj | ChromoGen SD1.5 (24obj) | ⭐⭐ |
| chromogen_phase1_sd15_24obj_v2/ | 24obj | ChromoGen SD1.5 v2 | ⭐⭐ |
| cross_domain/ | AutoKary | 跨域 zero-shot + finetune (新增) | ⭐⭐⭐ |
| cspnext_l_rf_heun_adaln_stochot/ | chromo | CSPNeXt-L backbone (证伪) | ⭐ |
| direction_exps/ | chromo | Direction B/D 实验 | ⭐⭐ |
| frontier_directions/ | 24obj | h_velocity_loss, n_cascade_e2e 等 | ⭐⭐ |
| i1_seesaw_normalized/ | chromo | Seesaw Loss + Normalized Classifier | ⭐⭐ |
| instrumentation*/ | chromo | 插桩分析 | ⭐ |
| ldmdet_rf_heun_adaln_stochot_eps5/ | 24obj | 早期 24obj StochOT 实验 (0.853) | ⭐⭐ |
| ldmdet_rf_heun_shifted_bs8_aug_v1/v2/v3/ | chromo | 冗余 checkpoint (v3=156G) | ⭐ |
| merged_ablation/ | merged | 合并数据集训练 | ⭐⭐ |
| multi_dataset/ | — | 多数据集实验 | ⭐ |
| multi_seed/ | chromo | 无 aug 旧 baseline (非标准) | ⭐ |
| multi_seed_aug/ | chromo | DDPM/RF/HardOT/Sinkhorn 多种子 (标准 baseline) | ⭐⭐⭐ |
| normalized_only/ | chromo | Normalized Classifier only | ⭐⭐ |
| optim_test*/ | — | 优化器调试 | ⭐ |
| pd_rf_24obj/ | 24obj | PD-RF 渐进蒸馏 (新增) | ⭐⭐ |
| scheme_a_dinov2_s/ | chromo | DINOv2-S backbone (证伪) | ⭐ |
| scheme_E_bifpn/ | chromo | BiFPN (未采用) | ⭐ |
| sc_rf_24obj/ | 24obj | SC-RF 自条件化 RF (✅ 已归档, mAP=0.860) | ⭐⭐ | <!-- verified: 2026-07-16 -->
| setdiff_24obj/ | 24obj | SetDiff Coupled State Diffusion (新增) | ⭐⭐ |
| sota_seed{42,123,456,1000}/ | chromo | SOTA 多种子 (4 seeds) | ⭐⭐⭐ |
| stability/ | chromo | warm_restart 稳定性实验 | ⭐ |

---

## 七、C 类推理任务归档（2026-07-18）

> 本章节为 C 类推理任务（阶段 4）最终归档，数据来源：
> - `/home/linkst/workspace/projects/chromosome-kd/docs/paper/C_CLASS_TASK_PLAN.md` §10 执行进度
> - `/tmp/c_class_results/c2_coupling_analysis.md` (C2 详细统计)
> - `/tmp/c_class_results/c3_statistical_tests.md` (C3 详细统计)
> - `/tmp/c_class_results/c6_perclass_stability.md` (C6 per-class 稳定性)
>
> 合并日期: 2026-07-18 | 合并日志: `/tmp/c_class_results/merge_log.md`

### 7.1 C 类任务总览

| 任务 | 优先级 | 目的 | 状态 | 关键结论 |
|------|--------|------|------|---------|
| C1: A4 多种子推理 | P0 | 验证主结果 0.863 mAP 跨种子稳定性 | ✅ 完成 | 3-seed mean 0.859±0.003，与 seed 42 单值一致 |
| C2: 耦合多种子推理 | P0 | 验证 Stochastic Coupling 稳定性声明 | ✅ 完成 | Dataset 1 上 StochOT vs Random +0.0308 mAP (p<1e-120) |
| C3: 配对显著性检验 | P0 | 关键 mAP 差异的统计检验 | ✅ 完成 | A4 vs A3: Δ=+0.0056, Wilcoxon p=2.54e-07 *** |
| C4: 测试集评估 | P1 | test split 泛化性 | ✅ 完成 | A4 seed42 test mAP=0.859，与 val 一致 |
| C5: Dataset 1 SOTA 数值核实 | ✅ 已解决 | 论文 A1 段落三数值核实 | ✅ 完成 | 三个数值 (0.753/0.742/0.737) 均有实验依据 |
| C6: Per-class 多种子稳定性 | P2 | per-class AP 跨种子验证 | ✅ 完成 | Y AP 0.771±0.006，最大 std 0.0060 (Y) |
| C7: Shift 消融独立推理 | P1 | SG6 shift 消融数据验证 | ✅ 完成 | Δ=−0.001 mAP，与训练 log 一致 |
| C8: FPS 基准复测 | P2 | 验证 FPS 表 latency 数值 | ⏸️ 跳过 | 现有 FPS 数据方法论足够严谨，无需复测 |

### 7.2 C1: A4 (DPM-Solver++) 多种子推理结果

> 数据源: C_CLASS_TASK_PLAN.md §10 / C1
> 配置: Dataset 2 (24obj) val, 500 images, 推理 seed=42, DPM-Solver++ 4-step

| Seed | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|------|------|------|------|------|------|------|
| 42 | 0.863 | 0.989 | 0.972 | 0.499 | 0.860 | 0.901 |
| 123 | 0.857 | 0.988 | 0.969 | 0.521 | 0.855 | 0.901 |
| 789 | 0.856 | 0.988 | 0.964 | 0.527 | 0.853 | 0.868 |
| **mean** | **0.859** | **0.988** | **0.968** | **0.516** | **0.856** | **0.890** |
| **std** | **0.003** | **0.0005** | **0.003** | **0.012** | **0.003** | **0.016** |

**关键发现**: 3-seed 主结果稳定，mAP std=0.003，与 seed 42 单值 0.863 在误差范围内一致。AP_S 跨种子波动较大 (std=0.012)，但 mean 0.516 与训练 scalars 记录一致。

> ⚠️ 数据完整性问题（已在 §5 数据完整性检查 C1-C16 中记录）：论文 SOTA 表 A3/A4 标签串列。两个子代理独立验证（scalars.json + swanlab_export.json）确认论文 "A3" 行数值 0.863/0.583 实际来自 A4。

### 7.3 C2: 耦合策略多种子推理（Dataset 1）

> 数据源: `/tmp/c_class_results/c2_coupling_analysis.md`
> 配置: Dataset 1 (Chromosome20240904) val, 440 images, 推理 seed=42
> 训练种子: [42, 123, 789]

#### 7.3.1 聚合指标 — per-image mean (mean±std across 3 seeds)

| 策略 | mAP@0.5:0.95 | AP50 | AP75 | AP_S | AP_M | AP_L |
|------|-------------|------|------|------|------|------|
| **Random** | 0.7658 ± 0.0036 | 0.9268 ± 0.0015 | 0.8404 ± 0.0015 | 0.6874 ± 0.0043 | 0.7846 ± 0.0043 | 0.7655 ± 0.0379 |
| **Hard OT** | 0.7597 ± 0.0018 | 0.9181 ± 0.0016 | 0.8352 ± 0.0013 | 0.6823 ± 0.0073 | 0.7777 ± 0.0009 | 0.7697 ± 0.0031 |
| **StochOT** | 0.7966 ± 0.0016 | 0.9566 ± 0.0008 | 0.8692 ± 0.0013 | 0.7324 ± 0.0062 | 0.8119 ± 0.0010 | 0.7910 ± 0.0042 |

> 注: per-image mean 是 440 张图的 per-image AP 算术平均，与 COCO aggregate mAP (IoU+area 加权) 略有差异，但用于跨策略对比足够稳定。

#### 7.3.2 每种子 per-image mean (debug)

| 策略 | seed 42 | seed 123 | seed 789 | mean | std |
|------|---------|----------|----------|------|-----|
| **Random** | 0.7648 | 0.7698 | 0.7629 | **0.7658** | ±0.0036 |
| **Hard OT** | 0.7597 | 0.7580 | 0.7616 | **0.7597** | ±0.0018 |
| **StochOT** | 0.7956 | 0.7958 | 0.7985 | **0.7966** | ±0.0016 |

#### 7.3.2b COCO 聚合 mAP — per-seed + mean±std (Dataset 1 val, 440 images)

> 数据源: main.tex `tab:per-seed-coupling` (L1206-1218) + C_CLASS_TASK_PLAN.md §10
> 注: COCO 聚合 mAP 使用 IoU+area 加权，不同于 §7.3.1 的 per-image 算术平均。此表与论文 `tab:per-seed-coupling` 完全一致。

| 策略 | ε | Seed | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|------|---|------|------|------|------|------|------|------|
| Hard OT | 0 | 42 | 0.705 | 0.894 | 0.792 | 0.409 | 0.702 | 0.543 |
| Hard OT | 0 | 123 | 0.703 | 0.897 | 0.792 | 0.412 | 0.700 | 0.548 |
| Hard OT | 0 | 789 | 0.707 | 0.897 | 0.794 | 0.425 | 0.702 | 0.566 |
| Random | ∞ | 42 | 0.713 | 0.909 | 0.800 | 0.421 | 0.711 | 0.626 |
| Random | ∞ | 123 | 0.718 | 0.909 | 0.804 | 0.442 | 0.716 | 0.629 |
| Random | ∞ | 789 | 0.708 | 0.904 | 0.796 | 0.423 | 0.702 | 0.527 |
| StochOT | 5 | 42 | 0.745 | 0.941 | 0.832 | 0.510 | 0.738 | 0.638 |
| StochOT | 5 | 123 | 0.745 | 0.942 | 0.834 | 0.505 | 0.739 | 0.636 |
| StochOT | 5 | 789 | 0.750 | 0.943 | 0.837 | 0.519 | 0.743 | 0.620 |
| **Hard OT (mean±std)** | — | — | 0.705±0.002 | 0.896 | 0.793 | 0.415 | 0.701 | 0.552 |
| **Random (mean±std)** | — | — | 0.713±0.005 | 0.907 | 0.800 | 0.429 | 0.710 | 0.594 |
| **StochOT (mean±std)** | — | — | **0.747±0.003** | **0.942** | **0.834** | **0.511** | **0.740** | **0.631** |

#### 7.3.3 配对统计检验 — pooled (跨种子, n=440×3=1320)

| 对比 (metric) | Δ均值 | Wilcoxon p | paired t p | n | 显著性 |
|------|-------|-----------|-----------|---|--------|
| StochOT vs Random — mAP@0.5:0.95 | +0.0308 | 9.03e-126 | 9.88e-130 | 1320 | *** |
| Hard OT vs Random — mAP@0.5:0.95 | -0.0061 | 1.20e-08 | 1.59e-09 | 1320 | *** |
| StochOT vs Hard OT — mAP@0.5:0.95 | +0.0369 | 8.98e-155 | 2.81e-156 | 1320 | *** |
| StochOT vs Random — AP@IoU=0.5 | +0.0298 | 1.61e-128 | 9.69e-126 | 1320 | *** |
| Hard OT vs Random — AP@IoU=0.5 | -0.0087 | 4.51e-17 | 1.30e-18 | 1320 | *** |
| StochOT vs Hard OT — AP@IoU=0.5 | +0.0385 | 5.91e-155 | 3.44e-145 | 1320 | *** |
| StochOT vs Random — 小目标 AP_S | +0.0450 | 3.89e-68 | 2.50e-75 | 1314 | *** |
| Hard OT vs Random — 小目标 AP_S | -0.0051 | 1.70e-02 | 2.21e-02 | 1314 | * |
| StochOT vs Hard OT — 小目标 AP_S | +0.0501 | 1.88e-83 | 5.93e-85 | 1314 | *** |

> 显著性: `*` p<0.05, `**` p<0.01, `***` p<0.001 (Wilcoxon). Pooled 检验将 3 个种子的 per-image AP 拼接 (n=1320)，提供单一 p 值。

#### 7.3.4 关键发现 — Dataset-dependent 结论

> ⚠️ C2 的结论与论文 §4.6 的 "StochOT 仅稳定性，无 mAP 增益" 声明在 Dataset 1 上**冲突**，需修订论文表述。

| 对比 | Dataset 1 (Chromosome20240904) | Dataset 2 (24obj) | 论文当前声明 |
|------|-------------------------------|-------------------|------------|
| StochOT vs Random | **+0.034 mAP, p<1e-120** | +0.0001 mAP, p=0.80 | "无 mAP 增益"（仅适用 Dataset 2） |
| Hard OT vs Random | **-0.008 mAP**（更差） | — | "Hard OT 应劣于 Random/StochOT" |
| StochOT vs Hard OT | +0.037 mAP, p<1e-150 | — | 一致 |

**结论**: StochOT 增益是 dataset-dependent 的 — 在小数据集 (Dataset 1, 1540 images) 上高度显著，在大数据集 (Dataset 2, 5000 images) 上统计上等价。论文需修订为 "低数据 regime 高度显著" 而非笼统的 "无 mAP 增益"。

### 7.4 C3: 配对统计显著性检验（Dataset 2）

> 数据源: `/tmp/c_class_results/c3_statistical_tests.md`
> 配置: Dataset 2 (24obj) val, 500 images, 推理 seed=42
> 检验方法: Wilcoxon signed-rank + paired t-test (两者都报告)

#### 7.4.1 各模型 per-image 均值

| 模型 | mAP@0.5:0.95 | AP50 | AP_S (有效图数) |
|------|-------------|------|-----------------|
| A2 (Heun) | 0.8943 ± 0.0655 | 0.9936 ± 0.0157 | 0.6321 ± 0.2491 (60) |
| A3 (Heun+StochOT) | 0.8944 ± 0.0646 | 0.9944 ± 0.0138 | 0.6333 ± 0.2352 (60) |
| A4 (DPM-Solver++) | 0.9000 ± 0.0645 | 0.9947 ± 0.0133 | 0.6302 ± 0.2514 (60) |

#### 7.4.2 mAP@0.5:0.95 配对检验

| 对比 | Δ均值 | Wilcoxon W | Wilcoxon p | t-statistic | paired t p | n | 显著性 |
|------|-------|-----------|-----------|-------------|-----------|---|--------|
| A3 vs A2 (StochOT 效应) | +0.0001 | 61544.5 | 7.97e-01 | 0.071 | 9.44e-01 | 500 |  |
| **A4 vs A3 (DPM-Solver++ 效应)** | **+0.0056** | **45165.0** | **2.54e-07** | **4.988** | **8.44e-07** | **500** | **\*\*\*** |
| A4 vs A2 (联合效应) | +0.0057 | 50858.0 | 4.53e-04 | 4.097 | 4.88e-05 | 500 | *** |

#### 7.4.3 AP@IoU=0.5 配对检验

| 对比 | Δ均值 | Wilcoxon W | Wilcoxon p | t-statistic | paired t p | n | 显著性 |
|------|-------|-----------|-----------|-------------|-----------|---|--------|
| A3 vs A2 (StochOT 效应) | +0.0009 | 2941.5 | 1.26e-01 | 1.869 | 6.22e-02 | 500 |  |
| A4 vs A3 (DPM-Solver++ 效应) | +0.0002 | 2276.0 | 4.87e-01 | 0.710 | 4.78e-01 | 500 |  |
| A4 vs A2 (联合效应) | +0.0011 | 2789.0 | 2.76e-02 | 2.393 | 1.71e-02 | 500 | * |

#### 7.4.4 小目标 AP_S 配对检验

| 对比 | Δ均值 | Wilcoxon W | Wilcoxon p | t-statistic | paired t p | n | 显著性 |
|------|-------|-----------|-----------|-------------|-----------|---|--------|
| A3 vs A2 (StochOT 效应) | +0.0012 | 351.5 | 7.83e-01 | 0.067 | 9.47e-01 | 60 |  |
| A4 vs A3 (DPM-Solver++ 效应) | -0.0031 | 437.0 | 8.55e-01 | -0.183 | 8.55e-01 | 60 |  |
| A4 vs A2 (联合效应) | -0.0019 | 380.5 | 6.91e-01 | -0.128 | 8.98e-01 | 60 |  |

> 显著性: `*` p<0.05, `**` p<0.01, `***` p<0.001 (Wilcoxon). `†` p<0.05 (paired t-test, 但 Wilcoxon 不显著). AP_S 仅包含两模型都有有效值 (≥0) 的图像。

#### 7.4.5 关键发现 — DPM-Solver++ 显著性

> ⚠️ C3 的结论与论文 §4.4 的 "DPM-Solver++ 纯计算优势，无精度增益" 声明**冲突**，需修订论文表述。

| 对比 | Δ mAP | Wilcoxon p | 显著性 | 论文当前声明 | 修订建议 |
|------|-------|-----------|--------|------------|---------|
| A4 vs A3 (DPM-Solver++) | **+0.0056** | **2.54e-07** | **\*\*\*** | "纯计算优势，无精度增益" | "小幅但统计显著的精度优势 (Δ=+0.0056, p=2.54e-07)" |
| A3 vs A2 (StochOT) | +0.0001 | 0.80 | n.s. | "无 mAP 增益"（适用 Dataset 2） | 一致 — Dataset 2 上统计等价 |
| A4 vs A2 (联合) | +0.0057 | 4.53e-04 | *** | — | 联合改进显著 |

**结论**: 在 Dataset 2 (500 images) 上，DPM-Solver++ 替换 Heun 采样器带来 +0.0056 mAP 的精度提升，Wilcoxon p=2.54e-07（远低于 0.001 阈值），具有强统计显著性。论文需修订为 "小幅但显著的精度优势"。

### 7.5 C4: 测试集评估

> 数据源: C_CLASS_TASK_PLAN.md §10 / C4
> 配置: A4 (DPM-Solver++ 4-step), seed 42, Dataset 2 test split

| Split | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|-------|------|------|------|------|------|------|
| val (500 images) | 0.863 | 0.989 | 0.972 | 0.499 | 0.860 | 0.901 |
| **test** | **0.859** | 0.988 | 0.971 | **0.577** | 0.856 | 0.914 |

**结论**: test split mAP=0.859 与 val mean (0.859±0.003) 完全一致，无过拟合迹象。AP_S point estimate 从 val 的 0.499 摆动到 test 的 0.577（与 main.tex L861-865 一致），反映了小目标 AP_S 的高方差特性（C1 三种子 AP_S std=0.012）。

### 7.6 C7: Shift 消融独立推理验证

> 数据源: C_CLASS_TASK_PLAN.md §10 / C7
> 配置: Dataset 2 val, 推理 seed=42

| 配置 | shift | best mAP (训练 log) | 独立推理 mAP | AP_S | AP_M | AP_L | Δ mAP | Δ AP_S |
|------|-------|---------------------|-------------|------|------|------|-------|--------|
| Linear (A1 unshifted) | 0 | 0.857 @ ep89 | 0.857 | 0.528 | 0.853 | 0.907 | — | — |
| Shifted (A1 主实验) | 3.0 | 0.856 | 0.856 | 0.542 | 0.853 | 0.897 | -0.001 | +0.014 |

**结论**: Shift 消融 Δ=−0.001 mAP，与训练 log 一致，shift 不带来 mAP 提升（仅用于训练稳定性）。AP_S 上 shifted (+0.542) 略高于 linear (+0.528)，Δ=+0.014；AP_L 上 shifted (0.897) 略低于 linear (0.907)，Δ=−0.010。两指标差异方向相反且均在 C1 显示的高方差范围内（AP_S std=0.012），不改变 shift 对精度影响可忽略的结论。支撑新增 Appendix 数据可信度。

### 7.7 C8: FPS 基准复测 — 决策记录

> 数据源: 内部决策 (2026-07-18)

**决策**: ⏸️ **跳过** — 现有 FPS 数据方法论足够严谨，无需复测。

**理由**:
1. FPS 表已包含 latency ± std（基于多轮 benchmark_fps.py 测量）
2. benchmark_fps.py 已使用 MODEL_REGISTRY 和 KNOWN_MAP 进行配置化测量
3. A3+IO3 K=200 的 14.2 FPS 声明已通过 benchmark 验证
4. 复测不会带来新信息，且消耗 GPU 时间

### 7.8 C6: Per-class AP 多种子稳定性

> 数据源: `/tmp/c_class_results/c6_perclass_stability.md`
> 配置: A4 (DPM-Solver++ 4-step), Dataset 2 val, 3 training seeds [42, 123, 789]
> Metric: per-class mAP@0.5:0.95 (mmengine classwise 'precision' field)

#### 7.8.1 24 类 per-class mAP@0.5:0.95 across 3 seeds

| Class | Seed 42 | Seed 123 | Seed 789 | Mean | Std | Range |
|-------|---------|----------|----------|------|-----|-------|
| A1 | 0.9140 | 0.9100 | 0.9080 | **0.9107** | 0.0025 | 0.0060 |
| A2 | 0.9100 | 0.9040 | 0.9000 | **0.9047** | 0.0041 | 0.0100 |
| A3 | 0.9060 | 0.8940 | 0.9000 | **0.9000** | 0.0049 | 0.0120 |
| B4 | 0.9070 | 0.9010 | 0.8970 | **0.9017** | 0.0041 | 0.0100 |
| B5 | 0.9070 | 0.9020 | 0.8980 | **0.9023** | 0.0037 | 0.0090 |
| C6 | 0.9060 | 0.8950 | 0.8930 | **0.8980** | 0.0057 | 0.0130 |
| C7 | 0.8960 | 0.8890 | 0.8900 | **0.8917** | 0.0031 | 0.0070 |
| C8 | 0.8880 | 0.8820 | 0.8830 | **0.8843** | 0.0026 | 0.0060 |
| C9 | 0.8800 | 0.8800 | 0.8780 | **0.8793** | 0.0009 | 0.0020 |
| C10 | 0.8790 | 0.8790 | 0.8760 | **0.8780** | 0.0014 | 0.0030 |
| C11 | 0.8740 | 0.8640 | 0.8660 | **0.8680** | 0.0043 | 0.0100 |
| C12 | 0.8910 | 0.8880 | 0.8830 | **0.8873** | 0.0033 | 0.0080 |
| D13 | 0.8560 | 0.8520 | 0.8530 | **0.8537** | 0.0017 | 0.0040 |
| D14 | 0.8540 | 0.8540 | 0.8520 | **0.8533** | 0.0009 | 0.0020 |
| D15 | 0.8430 | 0.8390 | 0.8340 | **0.8387** | 0.0037 | 0.0090 |
| E16 | 0.8540 | 0.8500 | 0.8460 | **0.8500** | 0.0033 | 0.0080 |
| E17 | 0.8420 | 0.8430 | 0.8380 | **0.8410** | 0.0022 | 0.0050 |
| E18 | 0.8350 | 0.8260 | 0.8260 | **0.8290** | 0.0042 | 0.0090 |
| F19 | 0.8180 | 0.8120 | 0.8150 | **0.8150** | 0.0024 | 0.0060 |
| F20 | 0.8200 | 0.8160 | 0.8140 | **0.8167** | 0.0025 | 0.0060 |
| G21 | 0.7910 | 0.7770 | 0.7820 | **0.7833** | 0.0058 | 0.0140 |
| G22 | 0.7880 | 0.7760 | 0.7790 | **0.7810** | 0.0051 | 0.0120 |
| X | 0.8840 | 0.8750 | 0.8700 | **0.8763** | 0.0058 | 0.0140 |
| **Y** | 0.7790 | 0.7650 | 0.7680 | **0.7707** | **0.0060** | **0.0140** |

#### 7.8.2 Size group summary (mean of per-class AP within group)

| Group | Classes | Seed 42 | Seed 123 | Seed 789 | Mean | Std |
|-------|---------|---------|----------|----------|------|-----|
| Large | 5 | 0.9088 | 0.9022 | 0.9006 | **0.9039** | 0.0035 |
| Medium | 11 | 0.8774 | 0.8725 | 0.8707 | **0.8735** | 0.0028 |
| Small | 8 | 0.8159 | 0.8081 | 0.8085 | **0.8108** | 0.0036 |

#### 7.8.3 C-group (C6-C12) intra-group spread per seed

| Seed | C-group Mean | C-group Std | C-group Range (max-min) |
|------|-------------|-------------|------------------------|
| 42 | 0.8877 | 0.0103 | 0.0320 |
| 123 | 0.8824 | 0.0092 | 0.0310 |
| 789 | 0.8813 | 0.0084 | 0.0270 |

#### 7.8.4 Y chromosome stability (key concern)

- Y AP values: [0.779, 0.765, 0.768]
- Y AP mean: **0.7707 ± 0.0060**
- Y AP range: [0.7650, 0.7790] (spread 0.0140)
- 论文当前报告: Y AP = 0.779 (seed 42)，建议修订为 mean ± std: 0.771 ± 0.006

#### 7.8.5 Overall stability assessment

- Mean per-class std across 3 seeds: **0.0035**
- Max per-class std: **0.0060 (Y)**
- Mean per-class range: 0.0083
- Max per-class range: 0.0140 (G21)

### 7.9 论文修订影响汇总

#### 7.9.1 6 项关键发现（影响论文表述）

| # | 发现 | 来源 | 当前论文表述 | 修订建议 |
|---|------|------|------------|---------|
| 1 | DPM-Solver++ 在 Dataset 2 上有 +0.0056 mAP 显著增益 | C3 | "纯计算优势，无精度增益" | "小幅但统计显著的精度优势 (p=2.54e-07)" |
| 2 | StochOT 在 Dataset 1 上 +0.034 mAP 高度显著 | C2 | "无 mAP 增益"（笼统） | "低数据 regime 高度显著，大数据集统计等价" |
| 3 | Hard OT 在 Dataset 1 上比 Random 差 -0.008 mAP | C2 | "Hard OT 应劣于 Random/StochOT" | 一致 — Dataset 1 验证 |
| 4 | A4 3-seed mean 0.859±0.003 | C1 | 仅报告 seed 42 单值 0.863 | 报告 mean ± std: 0.859 ± 0.003 |
| 5 | test split mAP=0.859 与 val 一致 | C4 | 未报告 test | 添加 test split 评估 |
| 6 | Y AP 跨种子 mean 0.771±0.006 | C6 | 仅报告 seed 42 单值 0.779 | 报告 mean ± std: 0.771 ± 0.006 |

#### 7.9.2 20 项论文修订清单

> 详细修订清单见 `/home/linkst/workspace/projects/chromosome-kd/docs/paper/C_CLASS_TASK_PLAN.md` §10 论文修订清单。本归档仅记录数据层修订影响，具体文字修订在 AAAI_INTEGRATED_DRAFT.md 中执行。

---

*本文档基于 EXPERIMENT_LINEAGE.md, EXPERIMENT_RESULTS.md, AAAI_INTEGRATED_DRAFT.md, swanlab_export.json, PUBLICATION_EVALUATION.md, 以及本地 work_dirs/ 和 experiments/configs/ 目录的综合整理。*
*2026-07-18 追加 §七 C 类推理任务归档（来源: C_CLASS_TASK_PLAN.md + /tmp/c_class_results/）。*
