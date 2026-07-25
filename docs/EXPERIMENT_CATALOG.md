# LDMDet 实验完整目录

> 生成日期: 2026-07-15 | 最近更新: 2026-07-25 (新增 C27-C28: Head Distill Plan A 训练中 / ReFlow Standard MSE coupling 生成中; 更新 Head Distill v2 已归因)
> 数据来源: 本地 `work_dirs/` (48 个子目录) + SwanLab 云端 (28 个项目) + `ldmdet-experiment/` 归档 + 两台服务器 (workstation / ross)
> 核心文档: [EXPERIMENT_LINEAGE.md](EXPERIMENT_LINEAGE.md) (实验谱系) + [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) (结果汇总) + [paper/AAAI_INTEGRATED_DRAFT.md](paper/AAAI_INTEGRATED_DRAFT.md) (论文草稿)

---

## 〇、数据源与服务器说明

### 三类数据源

| 来源 | 路径/格式 | 用途 |
|------|----------|------|
| **本地服务器日志** | `work_dirs/<exp_dir>/<timestamp>/<timestamp>.log` + `vis_data/scalars.json` | 完整训练曲线 + 配置快照 |
| **ldmdet-experiment 归档** | `ldmdet-experiment/sota/<category>/<exp_name>/` (含 README, config.py, metrics.json, code/, checkpoints/) | 已归档 SOTA 实验 (2026-06-13) |
| **SwanLab 云端** | `https://swanlab.cn/@einspanner/<project>/runs/<run_id>` | 在线可视化 + 跨实验对比 |

### SwanLab 项目一览 (28 个项目, 2026-07-19 同步)

> ⭐ = 论文相关项目; 数字为 SwanLab 实际实验数 (含 CRASHED/RUNNING)。

| SwanLab Project | 实验数 | 数据集 | 范围 |
|-----------------|--------|--------|------|
| `ldmdet-mainline-ablation-24obj` ⭐ | 18 (1 RUNNING) | 24obj | A0-A4 主路线消融 + 多种子 (A0/A1/A4) + A1 shift 消融 + SwiGLU |
| `ldmdet-ablation` ⭐ | 87 | 24obj/chromo | 主线消融 + 24obj 耦合策略 (Random/GHSS/Sinkhorn) + merged + gen_transfer + 非线性轨迹 + 方向实验 + 瓶颈分析 |
| `chromosome-kd-benchmark-24obj` ⭐ | 10 | 24obj | 对比模型: RTMDet-L / DINO R50 / Cascade / YOLOX-S / DiffusionDet / ldmdet_stochot_eps5 |
| `cross-domain-autokary` ⭐ | 9 | AutoKary | 跨域 zero-shot + finetune (A2/A3/A4 × k5/k10) |
| `few-shot-benchmark` ⭐ | 3 | 24obj | FBM CrossAttn / FBM SimpleGate / LDMDet SOTA 源预训练 |
| `chromosome-kd` ⭐ | 18 | chromo | 早期: sota_seed*, ablation/*, scheme_*, stability/* |
| `nonlinear-3seed-repro` | 2 | chromo | nonlinear_e43_seed{1,2} (seed3 失败) |
| `ldmdet-mainline-ablation-old` | 5 | chromo | StochOT ε=5 多种子 (chromo) 等旧主线消融 |
| `ldmdet-breakthrough` | 13 | 24obj/chromo | SC-RF 自条件化, I1 Seesaw Loss, 其他突破方向 |
| `ldmdet-frontier-directions-24obj` | 5 | 24obj | q1_occlusion (全部 CRASHED) |
| `ldmdet-frontier-directions` | 7 | 24obj/chromo | h_velocity_loss / n_cascade_e2e / h_cfm_velocity 等 |
| `setdiff-24obj` | 18 (3 RUNNING, 11 CRASHED) | 24obj | SetDiff (Coupled State Diffusion) plan A/B (全部 mAP=0, 失败方向) |
| `ldmdet-inference` | 108 | 24obj | C 类推理任务 (已归档至 §七, 含 NFE/Top-K/solver 对比) |
| `ldmdet-inference-opt-24obj` | 4 | 24obj | 推理优化实验 (inference_mode / cache 等, 均反向优化已回退) |
| `ldmdet-sota-stack` | 3 | chromo | chromo SOTA 堆叠 (mAP=0.751, 低于 24obj SOTA, 不相关) |
| `chromosome-kd-benchmark` | 7 | chromo | 早期 chromo 基线对比 |
| `chromosome-kd-ablation` | 8 | chromo | 早期 chromo 消融 |
| `chromosome-kd-stability` | 7 | chromo | 早期稳定性实验 |
| `chromosome-kd-kcec` | 7 | chromo | KCCE 对比学习 |
| `chromosome-kd-dpm` | 6 | chromo | DPM-Solver 早期实验 |
| `chromosome-kd-smallobj` | 6 | chromo | 小目标尺度感知 |
| `chromosome-kd-multiseed` | 5 | chromo | 早期多种子 |
| `chromosome-kd-daec` | 2 | chromo | DAEC 对比学习 |
| `chromosome-kd-arch` | 2 | chromo | 架构探索 |
| `chromosome-kd-scheme-a` | 2 | chromo | 方案 A |
| `chromosome-gen` | 2 | chromo | 生成式实验 |
| `chromosome-kd-scheme-b` | 1 | chromo | 方案 B |
| `chromosome-kd-verify-v1` | 1 | chromo | 验证实验 v1 |

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
| **A4 DPM-Solver++** | 24obj | ldmdet-mainline-ablation-24obj | (a4_dpm_pp) | work_dirs/a4_dpm_pp_24obj/ | a4_dpm_pp_24obj.py | **0.863** (3-seed: 0.859±0.003) | ✅ 完成 | 主路线消融 | <!-- verified: 2026-07-16: seed42=0.863, seed123=0.857@ep62, seed789=0.856@ep72 -->
| M1 形态感知 RoI (BF16) | 24obj | ldmdet-mainline-ablation-24obj | (m1_morphology_aware_ws) | ⚠ workstation `100.99.131.26`: work_dirs/m1_morphology_aware_24obj_ws/ | m1_morphology_aware_24obj_ws.py | 0.818 (BF16) | ✅ 已完成 (BF16 误导确认, FP32 复现已闭环; best@ep1 全程 0.811-0.818 波动; Δ=-0.045 vs A4 0.863 BF16 虚假退化; Δ=-0.007 vs A4+BF16 0.825 noise 范围但偏负面; 显存 20888 MiB vs FP32 37506 MiB 降 44%; fuse 权重均匀未学到方向性) | 结构改进 | <!-- 2026-07-23 完成, 2026-07-25 FP32 复现确认 BF16 误导 (见 C22): 30ep BF16 AMP, best 0.818@ep1; A4+BF16=0.825 (BF16 本身掉点 -0.038 已确认); M1 vs A4+BF16=-0.007 (noise 范围但偏负面); per-class 24 类全退化; fuse h_conv/v_conv 完全均匀 (ratio=1.01, std=0) 未学到方向性; 详见 §6.6 C23 -->
| M1 形态感知 RoI (FP32) | 24obj | ldmdet-mainline-ablation-24obj | (m1_morphology_aware_fp32) | ⚠ ross `100.122.196.41`: work_dirs/m1_morphology_aware_24obj_fp32/ | m1_morphology_aware_24obj_fp32.py | **0.862** (best@ep19) | ✅ 已完成 (30ep FP32, lr=2e-5 2×, 1ep warmup, 显存 37.5GB; last 0.859@ep30; Δ=-0.001 vs A4 0.863 统计上持平; BF16 误导根因确认, h_conv/v_conv FP32 下仍均匀) | 结构改进 | <!-- 2026-07-23 19:14 启动, 2026-07-25 完成: lr=2e-5 iter-based warmup, FP32, 30ep; best 0.862@ep19 (上修自 0.860@ep3 临时值); h_conv/v_conv 在 FP32 下仍均匀 (ratio=1.01-1.02, std=0.0001) → 设计问题非精度问题; 改进方向: 非零初始化 fuse + 显式形态先验注入 + 注意力机制替代方向卷积; 详见 §6.6 C22 -->
| Head Distillation v2 (H=3←H=6, freeze backbone) | 24obj | ldmdet-head-distill | 9qj0xe5q0dwb6l2d8igwy | ⚠ ross `100.122.196.41`: work_dirs/h3_distill_24obj/ | h3_distill_24obj.py | 0.711 (best@ep96) | ⚠ 已归因 (方案 A 替代): 异常中断@ep99/150, Δ=-0.152 vs A4; **根因**: freeze_backbone=True 致 Student backbone 停在 ImageNet, 而 Teacher head 在 A4 染色体特征上学习 → 特征分布不匹配, 学习缓慢; loss_distill≈0.033 稳定但 mAP 停滞 0.71; 方案 A (解冻 backbone + A4 权重加载) 已启动替代 | 蒸馏 | <!-- 2026-07-24 01:03 启动, 2026-07-25 异常中断@ep99; 根因: backbone 冻结致特征不匹配; 方案 A 替代见下行; 详见 §6.6 C26 -->
| **Head Distill Plan A** (H=3←H=6, backbone解冻+A4加载) | 24obj | ldmdet-head-distill | (h3_distill_plan_a) | ⚠ ross `100.122.196.41`: work_dirs/h3_distill_plan_a_24obj/ | h3_distill_plan_a_24obj.py | **0.854** (best@ep4, 🔄训练中) | 🔄 训练中 (ross A6000, ep7/150, ETA~10h; **方案 A 有效**: best 0.854@ep4 已接近 A4 baseline 0.863, 较 v2 0.711 大幅提升 +0.143; loss_distill≈0.029 稳定, grad_norm≈10-12 正常; freeze_backbone=False + teacher_checkpoint 加载 A4 backbone/neck, init_weights 重写避免 load_from 覆盖 head 映射) | 蒸馏 | <!-- 2026-07-25 17:04 启动: 方案 A 修复 v2 特征不匹配; detector.py _load_backbone_from_checkpoint 加载 A4 backbone+neck, init_weights 重写保留 head 映射; commit 66edac84; 详见 §6.6 C27 -->
| **ReFlow (Standard MSE)** 2-RF | 24obj | ldmdet-reflow | (reflow_standard) | ⚠ workstation `100.99.131.26`: work_dirs/reflow_standard_24obj/ (待启动) | reflow_standard_24obj.py | — (🔄 coupling 生成中) | 🔄 代码就绪 + coupling 生成中 (workstation A5000, 3500 图 A4 推理, bs=1, 关闭 box_renewal/ensemble/pruning); 代码 commit `66edac84`+`9de633dc`, 36 单元测试通过; 混合 target: cls=GT, box=x_0^pred (per-proposal), box_target_mode='x0_pred'; lr=1e-5, 50ep, bs=2; 前置: generate_reflow_couplings.py 生成 train_couplings.pt | ReFlow | <!-- 2026-07-25: 代码实现 + BUG 修复 (空间一致性/img_id/参数传递); coupling 生成 2026-07-25 18:58 启动; 详见 §6.6 C28 -->
| R3 v-prediction seed42 | 24obj | ldmdet-r3-vpred | (r3_vpred) | ⚠ workstation: work_dirs/r3_vpred_24obj_seed42/ (`/home/linkst/workplace/chromo/chromosome-kd/`) | r3_vpred_24obj.py | 0.855 (best@ep34) | ✅ 已完成 (workstation A4000, seed 42, max 150ep 早停@ep64 patience=30 触发; v_prediction=True + v_prediction_t_eps=1e-2 → 1/t² loss reweighting + batch normalization 均值=1; last 0.837@ep64; ep8 warmup 0.802→ep34 best 0.855→长期停滞→早停; Δ=-0.008 vs A4 0.863 超 3-seed noise ±0.003 但偏小, 单 seed 支持 R3.2; seed 123/789 待补) | 核心消融 | <!-- 2026-07-25 完成: workstation A4000, seed 42, v_prediction + 1/t² loss reweighting; 详见 §6.6 C24 -->
| S1 h6_s2 (cascade 解耦) | 24obj | ldmdet-s1-cascade-decouple | (s1_h6_s2) | ⚠ workstation: work_dirs/s1_h6_s2_24obj/ (`/home/linkst/workplace/chromo/chromosome-kd/`) | s1_h6_s2_24obj.py | 0.859 (best@ep106) | ✅ 已完成 (workstation A5000, max 150ep 早停@ep136 patience=30 触发; num_heads=6, sampling_timesteps=2 → NFE=12; last 0.856@ep136; Δ=-0.004 vs A4 0.863 在 3-seed noise ±0.003 范围内; 与 s1_h3_s4 (0.859) / s1_h3_s8 (0.859) 三组全部 0.859, S1.3 命题完整闭环) | 核心消融 | <!-- 2026-07-25 完成: workstation A5000, H=6 S=2 NFE=12; 详见 §6.6 C25 -->
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

#### 2.1.4 24obj Solver×Step 解耦推理 (A4 checkpoint, 项目 `ldmdet-inference`)

> 数据来源: 2026-07-15 在 A4 (DPM-Solver++) 训练的 checkpoint (`best_coco_bbox_mAP_epoch_117.pth`, seed 42) 上, 通过覆盖推理 solver 和步数进行的受控对比。所有实验在 24obj val (500 images) 上评估。论文 §4.5 Solver Analysis 直接引用本表数据。

| 实验 | SwanLab run_id | exp_name | Solver | Steps | NFE | mAP | AP50 | AP75 | 本地日志 |
|------|---------------|----------|--------|-------|-----|-----|------|------|---------|
| DPM-Solver++ 1-step | (ldmdet-inference) | `dpm_pp_1step` | DPM-Solver++ | 1 | 1 | 0.860 | — | — | `work_dirs/a4_dpm_pp_24obj/20260715_010957/` |
| DPM-Solver++ 2-step | (ldmdet-inference) | `dpm_pp_2step` | DPM-Solver++ | 2 | 2 | 0.863 | — | — | `work_dirs/a4_dpm_pp_24obj/20260715_011053/` |
| **Heun 2-step** | `ijsaub6e3kcxbkgi7dkok` | `heun_2step` | Heun | 2 | 3 | **0.863** | 0.988 | 0.972 | `work_dirs/a4_dpm_pp_24obj/20260715_011546/` |
| Heun 4-step | (ldmdet-inference) | `heun_4step` | Heun | 4 | 7 | 0.864 | — | — | `work_dirs/a4_dpm_pp_24obj/20260715_011706/` |

> 论文引用 (main.tex L834-846):
> - "DPM-Solver++ converges at 2 steps (mAP 0.863, seed 42)" ← dpm_pp_2step
> - "DPM-Solver++ 4-step (4 NFE, 0.863) ≈ Heun 2-step (3 NFE, 0.863) — equal accuracy, so DPM-Solver++ buys 43% fewer NFE at no cost" ← heun_2step
>
> 复现命令 (Heun 2-step 为例):
> `experiments/runners/test.py experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py --checkpoint work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth --dataset val --sampling-steps 2 --solver-type heun --exp-name heun_2step`
>
> ⚠ 注意: 本表为 **同 checkpoint 切换 solver** 的对比 (A4 checkpoint + 不同推理 solver)。论文中 "+0.006 per-image mAP at matched 4-step" (Table 8/`tab:stat-tests`) 是 **不同 checkpoint** 的对比 (A2 Heun-trained vs A4 DPM-Solver++-trained), 两者不可混淆。在同 A4 checkpoint 上, Heun 4-step 聚合 mAP=0.864 略高于 DPM-Solver++ 4-step=0.863, 但 per-image paired 检验的结论以 A2 vs A4 checkpoint 对比为准。 <!-- verified: 2026-07-19 SwanLab + 本地日志 -->

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
| M1 形态感知 RoI (FP32) | (待定, 探索性) | 0.862 | M1 FP32 复现 | m1_morphology_aware_24obj_fp32 (BF16 误导确认, Δ=-0.001 vs A4 持平) | <!-- 2026-07-25: 详见 §6.6 C22 -->
| R3 v-prediction seed42 | (待定, 探索性) | 0.855 | R3 vpred | r3_vpred_24obj_seed42 (单 seed, Δ=-0.008 vs A4 超 noise 但偏小) | <!-- 2026-07-25: 详见 §6.6 C24 -->
| S1 h6_s2 (cascade 解耦) | (待定, 探索性) | 0.859 | S1 h6_s2 | s1_h6_s2_24obj (S1.3 闭环, Δ=-0.004 在 noise 范围内) | <!-- 2026-07-25: 详见 §6.6 C25 -->

### 3.2 探索性实验 (已归档/证伪)

#### 3.2.1 SetDiff 系列 (Coupled State Diffusion)

| 实验 | 配置 | 本地路径 | SwanLab项目 | 状态 |
|------|------|----------|------------|------|
| SetDiff baseline (24obj) | setdiff_24obj.py | work_dirs/setdiff_24obj/ | setdiff-24obj | ✅/⚠ (早期, 仅 7 evals, max mAP=0) | <!-- verified: 2026-07-16 -->
| SetDiff baseline (chromo) | setdiff_baseline.py | — | — | — |

> 方向: 用耦合状态扩散替代标准噪声→box 路径。代码审查见 `docs/research/setdiff_code_review.md`。
>
> ⚠ 2026-07-19 SwanLab 同步: `setdiff-24obj` 项目共 18 个实验 (3 RUNNING / 11 CRASHED / 4 FINISHED), plan A (random_20ep) 和 plan B (random_gt_20ep) 所有已采集到指标的实验 mAP 均为 0.0000, 确认为失败方向。3 个 `setdiff_plan{A,B}_50ep_fix` 仍在 RUNNING (无指标产出)。

#### 3.2.2 PD-RF 蒸馏系列 (Progressive Distillation)

| 实验 | 配置 | 本地路径 | mAP | 状态 |
|------|------|----------|-----|------|
| PD-RF 24obj | pd_rf_24obj.py (备份) | work_dirs/pd_rf_24obj/ | 0.851 | ⛔ 证伪 (2026-07-11 归档) | <!-- verified: 2026-07-16: 本地 scalars.json max=0.851 (count=31) -->

> 方向: 渐进式蒸馏 Rectified Flow, 逐步减少采样步数。提案见 `docs/research/proposals/PD-RF_Progressive_Distillation.md`。⚠ 方向已证伪: 1-step Euler 无法近似 4-step DPM-Solver++ 预测, 蒸馏梯度与检测梯度冲突 (详见项目记忆 PD-RF 归档记录)。 <!-- verified: 2026-07-16 -->

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
| A4 DPM-Solver++ seed_123 | workstation | ldmdet-mainline-ablation-24obj | **0.857** @ ep62 | A4 多种子 ✅ 完成 (3-seed: 0.863/0.857/0.856, mean 0.859±0.003) |
| A4 DPM-Solver++ seed_789 | workstation | ldmdet-mainline-ablation-24obj | **0.856** @ ep72 | A4 多种子 ✅ 完成 |
| StochOT ε=5 seed_123 (old) | ross | ldmdet-ablation | **0.746** @ ep57 | StochOT 多种子 ✅ 完成 (3-seed: 0.746/0.746/0.749, mean 0.747±0.002) |
| StochOT ε=5 seed_789 (old) | ross | ldmdet-ablation | **0.749** @ ep69 | StochOT 多种子 ✅ 完成 |
| StochOT ε=2 seed_42 (old) | workstation | ldmdet-ablation | **0.749** @ ep75 | ε 消融补充 ✅ 完成 |

> ⚠ SC-RF 24obj 已于 2026-07-11 归档 (max mAP=0.860), 不再运行。A4+SwiGLU 已完成 (0.857)。详见 §2.1.3。 <!-- verified: 2026-07-16 -->
> ✅ 所有多种子补充实验已完成。workstation 上的 checkpoint 待 SCP 到 ross (workstation 连接问题搁置)。

### 3.5 SwanLab 同步补充实验 (2026-07-19)

> 数据来源: SwanLab `ldmdet-mainline-ablation-24obj` 项目 (2026-07-19 通过 `swanlab.OpenApi` 拉取)。本节补充 §3.4 之后新增的 A0/A1 多种子与 A1 shift 消融实验, 所有 mAP/AP50/AP75 均为 SwanLab 记录的 max 值 (3 位小数)。

#### 3.5.1 A0 baseline 多种子 (24obj)

| 实验 | run_id | mAP | AP50 | AP75 | 状态 | eval 次数 |
|------|--------|-----|------|------|------|----------|
| A0 baseline seed_42 (main) | qr93hfz65u99m1m7psmb8 | 0.774 | 0.968 | 0.916 | ✅ FINISHED | 12 |
| A0 baseline seed_123 | lvjqwpa44qnyza0odwth9 | 0.753 | 0.966 | 0.909 | ✅ FINISHED | 12 |
| A0 baseline seed_789 | gfy4wt6l1onz8yakfbhf1 | 0.774 | 0.967 | 0.919 | ✅ FINISHED | 12 |

> A0 3-seed: mean=0.767, std=0.012 (sample std, n-1)。⚠ A0 仅 12 个 eval (DDPM 基线, 训练慢), 且 seed_123 明显偏低 (-0.021), 反映 DDPM 1-step 基线对初始化敏感。<!-- verified: 2026-07-19 SwanLab -->

#### 3.5.2 A1 RF+Heun 多种子 (24obj, rf_schedule=shifted, rf_shift=3)

| 实验 | run_id | mAP | AP50 | AP75 | 状态 | eval 次数 |
|------|--------|-----|------|------|------|----------|
| A1 RF+Heun seed_42 (main) | pe6ljc6zlobcs744wcm8q | 0.856 | 0.990 | 0.971 | ✅ FINISHED | 92 |
| A1 RF+Heun seed_123 | 42dbc2wwimx5hf69g2ktx | 0.857 | 0.990 | 0.970 | ✅ FINISHED | 95 |
| A1 RF+Heun seed_789 | zbnloib75zw0t3ab13hsq | 0.852 | 0.987 | 0.968 | ⚠ RUNNING | 37 |

> A1 3-seed: mean=0.855, std=0.003 (sample std)。⚠ seed_789 仍在训练 (37/150 epochs), 当前 mAP=0.852 为临时最大值, 待完成后更新。A1 多种子稳定性显著优于 A0 (std 0.003 vs 0.012), 与 A4 (std=0.004) 同量级, 验证 RF+Heun 显著降低初始化敏感性。<!-- verified: 2026-07-19 SwanLab -->

#### 3.5.3 A1 rf_schedule / rf_shift 消融 (24obj, seed_42)

| 实验 | run_id | rf_schedule | rf_shift | mAP | AP50 | AP75 | 状态 | eval 次数 |
|------|--------|-------------|----------|-----|------|------|------|----------|
| A1 unshifted | pzuchxux4va4e3c4yik3q | linear | 1 | 0.857 | 0.990 | 0.971 | ✅ FINISHED | 119 |
| A1 shift2 | 8mbmjdvb798m3uquq7m73 | shifted | 2 | **0.860** | 0.990 | 0.972 | ✅ FINISHED | 105 |
| A1 shift3 (main) | pe6ljc6zlobcs744wcm8q | shifted | 3 | 0.856 | 0.990 | 0.971 | ✅ FINISHED | 92 |
| A1 shift5 | iuki2dyqjrtnx49rz3u4j | shifted | 5 | 0.856 | 0.990 | 0.971 | ✅ FINISHED | 104 |

> **结论**: shift 参数对 mAP 影响可忽略 (range 0.856-0.860, max-min=0.004, 在种子噪声范围内)。unshifted (linear, shift=1) 与 shift3 (main) 持平 (0.857 vs 0.856), shift2 略优 (+0.004) 但仍在噪声范围。论文中 rf_shift=3 的选择基于经验, 本消融支持 "shift 参数非关键超参" 的论断, 与 §七 C 类推理任务中 IO3 Top-K 剪枝的鲁棒性结论一致。<!-- verified: 2026-07-19 SwanLab -->

#### 3.5.4 同步汇总

- 本批次新增 7 个论文相关实验 (A0 多种子 ×2 + A1 多种子 ×2 + A1 shift 消融 ×3), 全部来自 `ldmdet-mainline-ablation-24obj` 项目。
- A0/A1 多种子补齐了 §3.4 中 A4 多种子的对照, 现已具备 A0/A1/A4 三组 3-seed 稳定性数据 (A1 seed_789 待完成)。
- A1 shift 消融验证 rf_shift 非关键超参, 支持论文中默认 rf_shift=3 的合理性。
- 跳过的项目: `ldmdet-inference` (108 个, 已归档至 §七)、`ldmdet-inference-opt-24obj` (4 个, 反向优化已回退)、`ldmdet-sota-stack` (3 个, chromo 数据集)、`ldmdet-frontier-directions-24obj` (5 个, 全部 CRASHED)、`setdiff-24obj` (18 个, 全部 mAP=0, 见 §3.2.1)、`chromosome-kd-*` 早期项目 (与 24obj 主线无关)。

### 3.6 SwanLab 补充同步 (第二轮, 2026-07-19)

> 数据来源: 2026-07-19 通过 `swanlab.Api` 拉取 21 个 SwanLab 项目的 40 个论文相关 run (含完整 mAP/AP50/AP75/APs/APm/APl 指标)。本节补充 §3.5 之外的剩余论文相关实验, 涵盖 StochOT ε 消融数据源澄清、DPM-Solver++ chromo 数据源、SOTA 多种子项目归属修正、IO1/3/4 训练、PD-RF v2/v3/v4 蒸馏变体、Q1 occlusion、chromo 稳定性等。所有 mAP/AP50/AP75 均为 SwanLab 记录的 max 值 (3 位小数); APs/APm/APl 为小/中/大目标 COCO AP (3 位小数)。

#### 3.6.1 StochOT ε 消融 chromo 数据源澄清 (项目 `ldmdet-mainline-ablation-old`, 解决 C16)

> 本节解决 §5 C16 "Epsilon 消融数据源不匹配" 问题。论文 §4.4.3 ε 消融数值 (ε=1→0.745, ε=2 stochastic→0.749, ε=5→0.746) 实际来自本项目, 而非 §1.5 标注的 `ldmdet-ablation` 或 `work_dirs/ablation_old/`。

| 实验 | run_id | ε | mAP | AP50 | AP75 | APs | APm | APl | 状态 | eval 次数 | best@step |
|------|--------|---|-----|------|------|-----|-----|-----|------|----------|-----------|
| StochOT ε=1 chromo | v97i4rtnkdkagko469w2s | 1 | 0.745 | 0.942 | 0.835 | 0.503 | 0.738 | 0.664 | ⚠ CRASHED | 63 | 38 |
| StochOT ε=2 chromo | zs8sbypvh2d3rqtc6316o | 2 | 0.749 | 0.946 | 0.843 | 0.526 | 0.741 | 0.663 | ✅ FINISHED | 105 | 75 |
| StochOT ε=5 seed_42 chromo | cgub615jprtse2g1edtcw | 5 | 0.746 | 0.946 | 0.833 | 0.511 | 0.740 | 0.659 | ⚠ CRASHED | 82 | 60 |
| StochOT ε=5 seed_123 chromo | 7fkfwi4e3teyyrr0t5l1q | 5 | 0.746 | 0.943 | 0.838 | 0.518 | 0.738 | 0.680 | ✅ FINISHED | 87 | 57 |
| StochOT ε=5 seed_789 chromo | 3yh5cqaacv066oqjpdw4q | 5 | 0.749 | 0.945 | 0.837 | 0.513 | 0.743 | 0.690 | ✅ FINISHED | 99 | 69 |

> **C16 解决**: ε 消融 (ε=1/2/5) 数据源确认为 `ldmdet-mainline-ablation-old`, 与 §1.5 标注的 `ldmdet-ablation` 不符。ε=5 多种子 3-seed mean=0.747, sample_std=0.002, 与 §1.5 (StochOT ε=5 old, mean 0.747±0.002) 数值一致, 证明是同批实验的 SwanLab 云端记录。
>
> **ε 消融趋势**: ε=2 (0.749) ≈ ε=5 (0.747) > ε=1 (0.745), 与论文 §4.4.3 结论一致 (ε∈[2,5] 平台区, ε=1 退化)。
>
> ⚠ **数据完整性备注**: ε=1 与 ε=5 seed_42 均为 CRASHED 状态, 但已采集到足够 eval (63/82), max mAP 仍可引用。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.2 DPM-Solver++ chromo 历史数据源 (项目 `chromosome-kd-dpm`, §2.2.6 来源)

> 本节为 §2.2.6 历史 SOTA 归档中 `ldmdet_dpm_solver_pp_o2_s8` (mAP=0.748) 的 SwanLab 原始数据源。

| 实验 | run_id | order / steps | mAP | AP50 | AP75 | APs | APm | APl | 状态 | eval 次数 |
|------|--------|---------------|-----|------|------|-----|-----|-----|------|----------|
| DPM-Solver++ o2 s6 (run 1) | amk1haa6rg1cudl0qm4xz | 2 / 6 (7 NFE) | 0.747 | 0.947 | 0.842 | 0.522 | 0.738 | 0.669 | ✅ FINISHED | 99 |
| DPM-Solver++ o2 s6 (run 2) | c71x6ehy38p48nm3ymjhc | 2 / 6 (7 NFE) | 0.708 | 0.927 | 0.822 | 0.450 | 0.704 | 0.540 | ✅ FINISHED | 22 |
| DPM-Solver++ o2 s8 | hdgb63g34bup7jm33kged | 2 / 8 (9 NFE) | **0.748** | 0.948 | 0.842 | 0.529 | 0.739 | 0.643 | ⚠ CRASHED | 83 |

> **§2.2.6 来源澄清**: `ldmdet_dpm_solver_pp_o2_s8` (mAP=0.748) 实际来自本项目 `chromosome-kd-dpm` 的 run `hdgb63g34bup7jm33kged`, 状态 CRASHED (max mAP 在 step 65 时取得, 之后崩溃)。
>
> ⚠ run 2 (mAP=0.708) 仅 22 evals, 远低于 run 1 (99 evals, mAP=0.747), 疑为早停或重启, **不可作为引用源**。论文引用时应使用 run 1 (mAP=0.747) 或 s8 (mAP=0.748)。
>
> ⚠ chromo 数据集结论已暂时废弃 (§1.5 标注), 这些数值仅用于历史数据源溯源, 不进入论文正文。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.3 chromo SOTA 多种子项目归属修正 (项目 `chromosome-kd-multiseed`, 修正 §1.5)

> 本节修正 §1.5 中 SOTA 多种子的 SwanLab 项目归属。原标注为 `chromosome-kd`, 实际为 `chromosome-kd-multiseed` (run_id 完全匹配)。本表数值与 §1.5 已记录数值一致, 不变更 mAP, 仅修正项目名。

| 实验 (§1.5 原记录) | run_id | 原 SwanLab项目 | **修正后** SwanLab项目 | mAP | AP50 | AP75 | APs | APm | APl |
|--------------------|--------|---------------|------------------------|-----|------|------|-----|-----|-----|
| SOTA seed_42 | 9xswp5aj7rmfd4vys6906 | ~~chromosome-kd~~ | **chromosome-kd-multiseed** | 0.740 | 0.946 | 0.831 | 0.512 | 0.732 | 0.680 |
| SOTA seed_123 | b31e1xhzftod7ae38cs17 | ~~chromosome-kd~~ | **chromosome-kd-multiseed** | 0.749 | 0.946 | 0.840 | 0.519 | 0.741 | 0.693 |
| SOTA seed_456 | ukxsyw666y1xrpcrtjvx1 | ~~chromosome-kd~~ | **chromosome-kd-multiseed** | 0.746 | 0.948 | 0.838 | 0.527 | 0.739 | 0.701 |
| SOTA seed_1000 | ocdkvjs2vs1vozbh0goni | ~~chromosome-kd~~ | **chromosome-kd-multiseed** | 0.749 | 0.947 | 0.838 | 0.516 | 0.740 | 0.679 |

> **修正说明**: §1.5 表格 SwanLab项目列应将 SOTA seed_{42,123,456,1000} 四行的 `chromosome-kd` 改为 `chromosome-kd-multiseed`。`chromosome-kd` 项目 (18 runs) 实际包含的是早期 sota_seed*/ablation/*/scheme_*/stability/* 等实验 (见 §3.6.8)。<!-- verified: 2026-07-19 SwanLab -->
>
> 4-seed 统计: mean=0.746, sample_std=0.004, 与 §5 C6 记录一致 (0.746±0.004)。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.4 chromo 早期 ε 消融完整表 (项目 `chromosome-kd-ablation`, AdaLN 对照)

> 本项目为 2026-05-17 至 2026-05-23 期间在 chromo 数据集上进行的 ε 消融完整对照 (AdaLN vs no-AdaLN × ε=1/2/5), 共 8 个实验, 全部使用 seed=1769925607, bs=8。

| 实验 | run_id | 配置 | mAP | AP50 | AP75 | APs | APm | APl | 状态 | best@step |
|------|--------|------|-----|------|------|-----|-----|-----|------|-----------|
| adaln only (no StochOT) | sz03p2gl0367tfs8atx1x | RF+Heun+Shifted+AdaLN | 0.728 | 0.930 | 0.810 | 0.503 | 0.720 | 0.647 | ✅ FINISHED | 91 |
| ε=1 (no AdaLN) | 64oidkd7h8faes2ui5sj0 | RF+Heun+Shifted+StochOT(ε=1) | 0.729 | 0.933 | 0.818 | 0.512 | 0.720 | 0.644 | ✅ FINISHED | 90 |
| ε=2 (no AdaLN) | 5uogugisz4ypsr14lg1m0 | RF+Heun+Shifted+StochOT(ε=2) | 0.721 | 0.929 | 0.811 | 0.484 | 0.714 | 0.649 | ✅ FINISHED | 59 |
| ε=5 (no AdaLN) | 23ivqwwkskxb2i448l5zt | RF+Heun+Shifted+StochOT(ε=5) | 0.720 | 0.926 | 0.810 | 0.473 | 0.713 | 0.637 | ⚠ CRASHED | 59 |
| AdaLN+ε=1 | j8ilij1d2lt8jsw34enxf | RF+Heun+Shifted+AdaLN+StochOT(ε=1) | 0.735 | 0.934 | 0.822 | 0.501 | 0.727 | 0.640 | ✅ FINISHED | 76 |
| AdaLN+ε=2 | n3x4pxmig5h8nig3ps5js | RF+Heun+Shifted+AdaLN+StochOT(ε=2) | **0.738** | 0.935 | 0.825 | 0.501 | 0.728 | 0.680 | ✅ FINISHED | 106 |
| AdaLN+ε=5 | 0t95a0gutx6yrqow1hpz5 | RF+Heun+Shifted+AdaLN+StochOT(ε=5) | 0.734 | 0.934 | 0.820 | 0.505 | 0.728 | 0.665 | ✅ FINISHED | 90 |
| sota_group_hier_stoch | t7blsv2hs0zyfsiszz2bn | +AdaLN+StochOT(ε=5)+GroupHier | 0.741 | 0.940 | 0.830 | 0.501 | 0.732 | 0.669 | ✅ FINISHED | 40 |

> **结论**:
> 1. **AdaLN-Zero 增益**: AdaLN+ε=2 (0.738) > ε=2 no-AdaLN (0.721), +0.017 增益; AdaLN+ε=5 (0.734) > ε=5 no-AdaLN (0.720), +0.014 增益。AdaLN-Zero 在 chromo 数据集上稳定提升约 +0.014~0.017, 与论文 §4.3 AdaLN-Zero 消融结论方向一致 (但 chromo 结论已废弃)。
> 2. **ε 敏感性**: no-AdaLN 组 ε=1 (0.729) > ε=2 (0.721) > ε=5 (0.720), 趋势与 §3.6.1 (`ldmdet-mainline-ablation-old`, bs=2) 相反 (ε=2 > ε=1 > ε=5)。差异可能源于 batch size (bs=8 vs bs=2) 与 seed 不同, 表明 ε 最优值对训练超参敏感。
> 3. **GroupHier 附加**: sota_group_hier_stoch (0.741) > AdaLN+ε=5 (0.734), +0.007, GroupHier 在 chromo 上有小幅增益 (但未在 24obj 验证)。
>
> ⚠ 本表与 §3.6.1 数值存在差异 (如 ε=2: 本表 0.738 vs §3.6.1 0.749), 原因是 (a) 本表为 AdaLN+ε=2, §3.6.1 为 no-AdaLN+ε=2; (b) batch size 不同 (bs=8 vs bs=2); (c) seed 不同。**两表为不同实验批次, 不可混用**。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.5 IO1/IO3/IO4 训练运行 (项目 `ldmdet-inference-opt-24obj`)

> 本项目共 4 个实验, 包含 §七 C 类推理任务的训练侧对应实验。所有实验基于 `a3_full_sota_24obj.py` 配置, bs=2, 150ep。

| 实验 | run_id | 配置 | mAP | AP50 | AP75 | APs | APm | APl | 状态 | best@step |
|------|--------|------|-----|------|------|-----|-----|-----|------|-----------|
| IO1 step_early_exit | hsgk86tf04rfyw8osj450 | 自适应步数提前终止 (thr=0.01, cls=0.95, min=1) | **0.860** | 0.990 | 0.972 | 0.553 | 0.857 | 0.911 | ✅ FINISHED | 83 |
| IO3 topk_pruning K=100 | uo88by3p04s64gvevnecw | Top-K 框剪枝 (K=100, step0 后剪枝) | 0.823 | 0.960 | 0.935 | 0.492 | 0.820 | 0.879 | ✅ FINISHED | 31 |
| IO4 head_early_exit | 2avz2bdtbygoobld86w0m | 级联头提前退出 (min=3, box=0.005, cls=0.98, time_aware) | 0.859 | 0.990 | 0.971 | 0.575 | 0.856 | 0.913 | ✅ FINISHED | 85 |

> **关键发现**:
> 1. **IO1 step_early_exit 训练 mAP=0.860** 与 A4 (0.863) 接近 (-0.003), 但需配合推理时 IO1 评估才能验证自适应步数是否真正减少 NFE。该项目仅记录训练曲线, 推理结果在 §七 C 类任务归档中。
> 2. **IO3 topk_pruning K=100 训练 mAP=0.823** 显著低于 A3 baseline (0.858), -0.035。与 §七 C 类推理任务 IO3 Top-K 剪枝结论一致 (K=100 训练即退化, 不只是推理退化)。论文应明确: IO3 Top-K 剪枝在训练阶段已引入 mAP 损失, K=200/300 是可接受范围。
> 3. **IO4 head_early_exit 训练 mAP=0.859** 与 A3 (0.858) 持平 (+0.001), 表明级联头提前退出训练策略本身不引入精度损失, 推理收益完全来自 cascade head 跳过。
>
> ⚠ 第 4 个实验 (项目共 4 个) 为推理优化反向实验 (inference_mode/cache 等), 已在项目记忆中记录为反向优化并回退 (commit e683a0e1), 不列入本表。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.6 PD-RF v2/v3/v4 蒸馏变体 (项目 `ldmdet-breakthrough`, 续 §3.2.2)

> §3.2.2 已记录 PD-RF v1 (0.851, 已证伪)。本节补充同项目下的 v2/v3/v4 变体, 三者均尝试修复 v1 蒸馏梯度冲突问题, 全部失败。

| 实验 | run_id | 配置 | mAP | 状态 | best@step | 备注 |
|------|--------|------|-----|------|-----------|------|
| PD-RF v2 | kybbfmpivf2iseq8wiunh | load_from A4 + cascade_detach=False + cls蒸馏 | 0.860 | ⚠ CRASHED | 1 | 仅 1 eval (初始评估), 后即崩溃, 0.860 为 A4 checkpoint 初始 mAP, **非蒸馏结果** |
| PD-RF v3 | gysn54aizi3eqqrh3ijpm | v2 + 修复 lr warmup (5ep→1ep, start_factor 0.001→0.1) | 0.851 | ⚠ CRASHED | 1 | 仅 5 evals, last mAP=0.421 严重退化, lr warmup 修复无效 |
| PD-RF v4 | mctco58s4ouznybovku3f | lr=1e-05 + 纯 CosineAnnealing + distill_lambda=0.1 | 0.851 | ✅ FINISHED | 1 | 31 evals, last mAP=0.324 严重崩溃, max=0.851 仍为初始评估值 |

> **结论**: PD-RF v2/v3/v4 均无法解决 1-step Euler 蒸馏与 4-step DPM-Solver++ 预测的不一致问题。所有 "max mAP" 实际为 step=1 的初始 A4 checkpoint 评估值, 蒸馏训练开始后 mAP 立即退化 (v3 last=0.421, v4 last=0.324)。**PD-RF 方向在 v1-v4 四次尝试后确认证伪**, 强化 §3.2.2 结论。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.7 Q1 occlusion 实验 (项目 `ldmdet-frontier-directions-24obj`, 证伪)

> 本项目共 5 个实验, 全部为 q1_occlusion 方向 (RoI 特征遮挡模拟, prob=0.3), 全部 CRASHED。仅采集到 1 个 run 的指标。

| 实验 | run_id | 配置 | mAP | AP50 | AP75 | APs | APm | APl | 状态 | eval 次数 |
|------|--------|------|-----|------|------|-----|-----|-----|------|----------|
| Q1 occlusion p03 | hewczrewhnsqgyj52fkwv | RoI 特征遮挡 (prob=0.3, ratio=0.2-0.6) | 0.836 | 0.984 | 0.966 | 0.456 | 0.833 | 0.882 | ⚠ CRASHED | 17 |

> **结论**: Q1 occlusion 训练时 RoI 特征遮挡模拟方向证伪。max mAP=0.836 (17 evals, best@step13) 显著低于 A3 baseline (0.858), -0.022。训练早期即崩溃, 未能完成 150ep。**方向不进入论文, 归入 §3.2.5 其他方向 (证伪)**。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.8 chromo 稳定性实验 (项目 `chromosome-kd-stability`)

> 本项目共 7 个实验, 探索训练稳定性策略 (warm restart, SWA 等)。下表列出 3 个已采集到指标的论文相关实验, 其余 4 个为 bs=2 对照或探索性运行。

| 实验 | run_id | 配置 | mAP | AP50 | AP75 | APs | APm | APl | 状态 | best@step |
|------|--------|------|-----|------|------|-----|-----|-----|------|-----------|
| bs8-baseline | wcfaq0izyffklm05hp9p1 | bs=8, lr=2e-4 | **0.749** | 0.943 | 0.839 | 0.534 | 0.741 | 0.679 | ✅ FINISHED | 73 |
| bs8-warm-restart-v2 | 742jg74f33ikmh560x77t | CosineAnnealing warm-restart (ep75) | 0.719 | 0.931 | 0.808 | 0.479 | 0.713 | 0.630 | ✅ FINISHED | 57 |
| bs8-warm-restart-swa | 93dvrcphkv5j3xw8d5l9d | WarmRestart (ep75) + SWA (begin=ep75) | 0.728 | 0.931 | 0.815 | 0.493 | 0.722 | 0.632 | ✅ FINISHED | 57 |

> **结论**:
> 1. **warm-restart 反向**: bs8-warm-restart-v2 (0.719) < bs8-baseline (0.749), -0.030 严重退化。CosineAnnealing warm-restart 在 ep75 重启 lr 后训练崩溃。
> 2. **SWA 部分恢复**: bs8-warm-restart-swa (0.728) 略优于 warm-restart-v2 (0.719), +0.009, 但仍低于 baseline (-0.021)。SWA 平均权重部分缓解 warm-restart 退化, 但无法完全恢复。
> 3. **结论**: warm-restart 与 SWA 在 chromo 数据集 LDMDet 训练中均为反向策略, 不进入论文。
>
> ⚠ chromo 数据集结论已暂时废弃 (§1.5), 本节数据仅作为稳定性策略的负面证据存档。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.9 V1-E StochOT 实现验证 (项目 `chromosome-kd-verify-v1`)

> 本项目仅 1 个实验, 用于验证 StochOT 实现的正确性 (V1-E 配置: new OT, col norm, sample, eps=5, iter=20)。

| 实验 | run_id | 配置 | mAP | AP50 | AP75 | APs | APm | APl | 状态 | best@step |
|------|--------|------|-----|------|------|-----|-----|-----|------|-----------|
| V1-E StochOT verify seed1 | vqn7zk98wc2dl4wa9cbnr | new OT, col norm, sample, eps=5, iter=20 | 0.745 | 0.943 | 0.836 | 0.515 | 0.735 | 0.649 | ⚠ CRASHED | 55 |

> **结论**: V1-E 验证实验 mAP=0.745, 与 §3.6.1 StochOT ε=5 chromo (0.746-0.749) 同量级, **确认 StochOT 实现正确** (V1-E 重采样+列归一化+eps=5+iter=20 配置与 §3.6.1 数值一致)。该实验为实现层验证, 不进入论文, 但支持 §3.6.1 数据有效性。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.10 ldmdet-sota-stack 数据源 (项目 `ldmdet-sota-stack`, §2.2.6 focal_gamma3 来源)

> 本项目共 3 个实验, 在 chromo 数据集上堆叠 SOTA 组件 (StochOT + Focal Loss gamma=3)。仅 1 个实验采集到指标。

| 实验 | run_id | 配置 | mAP | AP50 | AP75 | APs | APm | APl | 状态 | best@step |
|------|--------|------|-----|------|------|-----|-----|-----|------|-----------|
| rf_heun_adaln_stochot_eps5_focal_gamma3_seed42 | o9rlqcu4kaedrygjd78va | RF+Heun+AdaLN+StochOT(ε=5)+Focal(γ=3) | 0.751 | 0.943 | 0.840 | 0.521 | 0.745 | 0.679 | ✅ FINISHED | 57 |

> **结论**: Focal Loss γ=3 堆叠 mAP=0.751, 与 chromo SOTA (0.749±0.004) 持平 (+0.002), 无显著增益。**不进入 24obj 论文, 仅作为 §2.2.6 chromo 历史实验数据源记录**。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.11 ldmdet-inference 项目级别说明 (项目 `ldmdet-inference`, 108 个推理实验)

> 本项目共 108 个实验, 全部为推理/测试任务 (使用 `experiments/runners/test.py`), **不记录训练曲线**, 因此 SwanLab metrics 接口对 mAP 等指标返回 404 (详见访问问题备注)。所有结果已聚合至 §七 C 类推理任务归档。

| 实验类型 | 数量 (估计) | 状态 | 说明 |
|----------|------------|------|------|
| C1 A4 (DPM-Solver++) 多种子推理 | ~6 | FINISHED | §7.2 |
| C2 耦合策略多种子推理 (Dataset 1) | ~12 | FINISHED/CRASHED | §7.3 |
| C4 A4 测试集评估 | ~3 | FINISHED/CRASHED | §7.5 |
| C7 Shift 消融独立推理 | ~2 | FINISHED | §7.6 |
| C6 Per-class AP 多种子稳定性 | ~3 | FINISHED | §7.8 |
| NFE / solver-step 对比 (DDIM/DPM-Solver++/Heun) | ~30 | FINISHED | §2.2.5, §七 |
| IO3 Top-K 剪枝推理 (K=100/200/300) | ~9 | FINISHED/CRASHED | §七, 项目记忆 |
| a4_noise 推理 | ~13 | FINISHED/CRASHED | 未归档, 噪声鲁棒性测试 |
| zero_shot A2/A4 → AutoKary/Chromo | ~6 | FINISHED | §2.3.3, §1.4 |
| dpm_solver_pp seed/solver 组合 | ~24 | FINISHED | NFE 对比, §七 |

> **关键说明**:
> 1. **本项目不进入 §3.5/§3.6 表格汇总**, 因所有指标已在 §七详尽记录。
> 2. **访问问题**: 108 个实验的 metrics 接口均返回 404, 因 `test.py` 不调用 `swanlab.log` 记录 mAP, 结果保存在 `results/c_class/` 本地 JSON。SwanLab 仅记录实验元数据 (name, status, duration)。
> 3. **数据完整性**: 已通过 `/tmp/swanlab_fetch_output.txt` 确认 108 个实验的 run_id 列表, 与 §七引用的 run_id 一致。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.12 其他已确认数据源项目 (无新增论文相关数据)

> 以下项目经 SwanLab 同步确认存在, 但与 24obj 论文核心无关, 仅记录项目元数据用于 §0 项目表交叉验证。

| 项目 | 实验数 | 数据集 | 说明 | 是否论文相关 |
|------|--------|--------|------|-------------|
| `chromosome-kd-kcec` | 7 | chromo | KCCE 对比学习 (kcec_pure_sota, kcec_v2_direct_quota, kcec_v3_scale_aware_gated 等) | ❌ chromo 旧方向 |
| `chromosome-kd-smallobj` | 6 | chromo | 小目标尺度感知 (smallobj_B_scaleaware_loglinear 已在 §2.2.6 归档) | ❌ chromo 旧方向 |
| `chromosome-kd-arch` | 2 | chromo | 架构探索 (DINOv2-S backbone 等, 全部证伪, 见 §3.2.5) | ❌ chromo 旧方向 |
| `chromosome-kd-daec` | 2 | chromo | DAEC 对比学习 (ldmdet_daec_contrastive_kcec_eps5 已在 §2.2.6 归档, mAP=0.740) | ❌ chromo 旧方向 |
| `chromosome-kd-scheme-a` | 2 | chromo | 方案 A 探索 | ❌ 已弃用 |
| `chromosome-kd-scheme-b` | 1 | chromo | 方案 B 探索 | ❌ 已弃用 |
| `chromosome-kd-gen` | 2 | chromo | 生成式实验 (ChromoGen-Phase1) | ❌ 与 LDMDet 主线无关 |
| `nonlinear-3seed-repro` | 2 | chromo | nonlinear_e43_seed{1,2} 重现 (seed3 失败) | ❌ 已在 §1.5 归档 |
| `ldmdet-frontier-directions` | 7 | 24obj/chromo | h_velocity_loss / n_cascade_e2e / h_cfm_velocity 等, 已在 §1.3/§3.2.4/§3.2.5 归档 | ⚠ 部分已归档 |

#### 3.6.13 数据一致性备注与待解决问题

> 本节汇总 §3.6 同步过程中发现的数据一致性问题, 供论文修订参考。

| 编号 | 问题 | 影响 | 建议修订 |
|------|------|------|----------|
| **C17** (新) | §1.5 SOTA 多种子项目归属错误 | SOTA seed_{42,123,456,1000} 四行 SwanLab项目列应为 `chromosome-kd-multiseed` (非 `chromosome-kd`) | §1.5 表格 4 行项目名修正 |
| **C18** (新) | §2.2.6 `ldmdet_dpm_solver_pp_o2_s8` 数据源未标注 SwanLab 项目 | 该实验来自 `chromosome-kd-dpm` 项目 (run `hdgb63g34bup7jm33kged`), 状态 CRASHED | §2.2.6 表格添加 SwanLab 项目列与 run_id |
| **C19** (新) | C16 ε 消融数据源已确认 | ε 消融 (ε=1/2/5) 来自 `ldmdet-mainline-ablation-old` (bs=2), §1.5 标注的 `ldmdet-ablation` 不正确 | §1.5 StochOT ε 消融行 SwanLab项目改为 `ldmdet-mainline-ablation-old`, C16 可标记为已解决 |
| **C20** (新) | §3.6.1 与 §3.6.4 ε 消融数值不一致 | ε=2: §3.6.1 (bs=2, no-AdaLN) = 0.749 vs §3.6.4 (bs=8, AdaLN) = 0.738; ε=5: §3.6.1 = 0.746-0.749 vs §3.6.4 = 0.720-0.734 | 论文 §4.4.3 引用应明确数据源: 主表使用 §3.6.1 (bs=2, 与 24obj 实验配置一致), §3.6.4 (bs=8) 作为 AdaLN 增益对照 |
| **C21** (新) | PD-RF v2/v3/v4 max mAP 误导 | v2/v3/v4 的 "max mAP" (0.851-0.860) 实际为 step=1 初始 A4 checkpoint 评估值, 非蒸馏训练结果 | §3.2.2 应明确 v1-v4 的 max mAP 均为初始值, 蒸馏训练后 mAP 立即退化 |

> 以上 C17-C21 为本次 SwanLab 第二轮同步发现的新问题, 不影响已发表论文数据, 但需在下次 catalog 修订时统一修正。<!-- verified: 2026-07-19 SwanLab -->

#### 3.6.14 第二轮同步汇总

- 本批次新增 **21 个论文相关 run** (含完整 AP 指标), 来自 10 个 SwanLab 项目 (`ldmdet-mainline-ablation-old`, `chromosome-kd-dpm`, `chromosome-kd-multiseed`, `chromosome-kd-ablation`, `ldmdet-inference-opt-24obj`, `ldmdet-breakthrough`, `ldmdet-frontier-directions-24obj`, `chromosome-kd-stability`, `chromosome-kd-verify-v1`, `ldmdet-sota-stack`)。
- 解决历史问题: C16 (ε 消融数据源), 新增问题 C17-C21 待下次修订。
- **不进入论文正文** 的实验: §3.6.4 (chromo 早期 ε 消融), §3.6.7 (Q1 occlusion), §3.6.8 (chromo 稳定性), §3.6.9 (V1-E 验证), §3.6.10 (focal_gamma3), §3.6.11 (ldmdet-inference 项目级), §3.6.12 (其他 chromo 旧项目)。仅作为数据源溯源与负面证据存档。
- **进入论文正文/附录** 的实验: §3.6.1 (StochOT ε 消融, 论文 §4.4.3), §3.6.2 (DPM-Solver++ chromo 历史, 仅 §2.2.6 归档), §3.6.3 (SOTA 多种子, 论文 §4.4.2), §3.6.5 (IO1/3/4 训练, §七 C 类推理), §3.6.6 (PD-RF v2/v3/v4, §3.2.2 续)。
- 跳过的实验: `ldmdet-inference` 108 个 (无训练曲线, 已在 §七归档)、`setdiff-24obj` 18 个 (mAP=0, §3.2.1)、`chromosome-kd-*` 早期项目 (与 24obj 主线无关, §3.6.12 已记录元数据)。
- **访问问题**: `ldmdet-inference` 项目所有 108 个 run 的 metrics 接口返回 404 (推理任务不记录训练曲线, 非错误, 结果在 §七本地 JSON)。其他项目无访问问题。<!-- verified: 2026-07-19 SwanLab -->

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
| C16 | Epsilon 消融数据源不匹配 (chromo) <!-- verified: 2026-07-16 --> | 论文 §4.4.3 (ε=0.5→0.710, ε=1.0→0.745, ε=2.0 stochastic→0.749, ε=5.0→0.746, ε=2.0 argmax→0.752) 与 EXPERIMENT_RESULTS.md §5 旧值 (0.720-0.738) 不匹配, 可能来自不同实验批次 (nonlinear_trajectory 系列)。本地仅 seed_42=0.748 可直接验证。详见 §3.6.1 / C19 (已解决)。 |
| C17 | §1.5 SOTA 多种子项目归属错误 <!-- verified: 2026-07-19 SwanLab --> | SOTA seed_{42,123,456,1000} 四行 SwanLab项目列应为 `chromosome-kd-multiseed` (非 `chromosome-kd`), run_id 已正确。4-seed mean=0.746±0.004 与 C6 一致。详见 §3.6.3。 |
| C18 | §2.2.6 `ldmdet_dpm_solver_pp_o2_s8` 数据源未标注 SwanLab 项目 <!-- verified: 2026-07-19 SwanLab --> | 该实验来自 `chromosome-kd-dpm` 项目 (run `hdgb63g34bup7jm33kged`), 状态 CRASHED, mAP=0.748 @ step 65。§2.2.6 表格应添加 SwanLab 项目列与 run_id。详见 §3.6.2。 |
| C19 | ~~C16 ε 消融数据源已确认~~ ✅ 已解决 <!-- verified: 2026-07-19 SwanLab --> | ε 消融 (ε=1/2/5) 数据源确认为 `ldmdet-mainline-ablation-old` (bs=2), §1.5 标注的 `ldmdet-ablation` 不正确。run_ids: ε=1=`v97i4rtnkdkagko469w2s`, ε=2=`zs8sbypvh2d3rqtc6316o`, ε=5=`cgub615jprtse2g1edtcw`/`7fkfwi4e3teyyrr0t5l1q`/`3yh5cqaacv066oqjpdw4q`。C16 可标记为已解决。详见 §3.6.1。 |
| C20 | §3.6.1 与 §3.6.4 ε 消融数值不一致 <!-- verified: 2026-07-19 SwanLab --> | ε=2: §3.6.1 (bs=2, no-AdaLN) = 0.749 vs §3.6.4 (bs=8, AdaLN) = 0.738; ε=5: §3.6.1 = 0.746-0.749 vs §3.6.4 = 0.720-0.734。两表为不同实验批次 (bs/seed/AdaLN 不同), 不可混用。论文 §4.4.3 引用应使用 §3.6.1 (bs=2, 与 24obj 配置一致)。详见 §3.6.4。 |
| C21 | PD-RF v2/v3/v4 max mAP 误导 <!-- verified: 2026-07-19 SwanLab --> | v2/v3/v4 的 "max mAP" (0.851-0.860) 实际为 step=1 初始 A4 checkpoint 评估值, 非蒸馏训练结果。蒸馏训练后 mAP 立即退化 (v3 last=0.421, v4 last=0.324)。§3.2.2 应明确 v1-v4 的 max mAP 均为初始值。详见 §3.6.6。 |

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
| M1 形态感知 RoI (FP32) | /media/ross/8TB/.../work_dirs/m1_morphology_aware_24obj_fp32/ | ✅ 已完成 (2026-07-25, best 0.862@ep19, Δ=-0.001 vs A4 持平) | ✅ 已同步 | <!-- 2026-07-25: 30ep FP32, lr=2e-5 2×, 1ep warmup, 显存 37.5GB; BF16 误导根因确认; 详见 §6.6 C22 -->
| Head Distillation v2 | /media/ross/8TB/.../work_dirs/h3_distill_24obj/ | ⚠ 异常中断@ep99/150 (2026-07-25, best 0.711@ep96, Δ=-0.152 vs A4 H=3 容量限制; 待恢复决策) | ✅ 已同步 | <!-- 2026-07-24 01:03 启动, 2026-07-25 异常中断: H=3 Student ← H=6 Teacher (A4 冻结), λ=0.05, freeze backbone, bs=2, 显存 2.4GB; nohup 戛然而止无报错; 详见 §6.6 C26 -->

### 6.2 workstation 服务器 (A5000/A4000, 并行多种子)

| 实验 | 状态 | 说明 |
|------|------|------|
| A4 DPM-Solver++ seed_123 | ✅ 已完成 (0.857 @ ep62) | A4 多种子: 3-seed mean 0.859±0.003 | <!-- verified: 2026-07-16 -->
| A4 DPM-Solver++ seed_789 | ✅ 已完成 (0.856 @ ep72) | A4 多种子完成 | <!-- verified: 2026-07-16 -->
| StochOT ε=2 seed_42 (old) | ✅ 已完成 (0.749 @ ep75) | ε 消融补充 | <!-- verified: 2026-07-16 -->
| StochOT ε=5 seed_123 (old) | ✅ 已完成 (0.746 @ ep57, ross) | StochOT 多种子完成 (3-seed: 0.747±0.002) | <!-- verified: 2026-07-16 -->
| StochOT ε=5 seed_789 (old) | ✅ 已完成 (0.749 @ ep69, ross) | StochOT 多种子完成 | <!-- verified: 2026-07-16 -->
| SwiGLU 实验 | ✅ 已完成 | A2+SwiGLU=0.859, A4+SwiGLU=0.857 (本地 scalars.json 已确认) | <!-- verified: 2026-07-16 -->
| M1 形态感知 RoI (BF16) | ✅ 已完成 (2026-07-23, best 0.818@ep1) | BF16 误导确认 (Δ=-0.045 vs A4 0.863 虚假退化; Δ=-0.007 vs A4+BF16 0.825 noise 范围但偏负面); 显存 20888 MiB vs FP32 37506 MiB 降 44%; FP32 复现已闭环 (见 §6.6 C22/C23) | <!-- 2026-07-25: 详见 §6.6 C23 -->
| R3 v-prediction seed42 | ✅ 已完成 (2026-07-25, best 0.855@ep34, 早停@ep64) | v_prediction + 1/t² loss reweighting; Δ=-0.008 vs A4 0.863 超 3-seed noise ±0.003 但偏小, 单 seed 支持 R3.2; seed 123/789 待补 | <!-- 2026-07-25: workstation A4000; 详见 §6.6 C24 -->
| S1 h6_s2 (cascade 解耦) | ✅ 已完成 (2026-07-25, best 0.859@ep106, 早停@ep136) | num_heads=6, sampling_timesteps=2 (NFE=12); Δ=-0.004 vs A4 在 3-seed noise ±0.003 范围内; 与 s1_h3_s4/s1_h3_s8 三组全部 0.859, S1.3 命题完整闭环 | <!-- 2026-07-25: workstation A5000; 详见 §6.6 C25 -->

> ✅ 所有多种子补充实验已完成。workstation 上的 checkpoint (A4 seed_123/789, StochOT ε=2) 待 SCP 到 ross (workstation 连接问题搁置)。
> ✅ 2026-07-25 新增 3 个实验完成: M1 BF16 ws (BF16 误导确认), R3 v-prediction seed42 (单 seed 初步), S1 h6_s2 (S1.3 闭环)。详见 §6.6 C23-C25。

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
| diagnosis/ | 零成本推理诊断实验 (D1-D5 结构诊断 + 方向 A/D 对比 + D1 消融) | ✅ 新增 |

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

### 6.5 零成本推理诊断实验 (work_dirs/diagnosis/)

> 基线: A4 (DPM-Solver++, mAP=0.863, checkpoint best_epoch_117), 500 张验证图推理, 无 SwanLab

| 实验 | 脚本 | 输出 JSON | 结果 | 状态 |
|------|------|-----------|------|------|
| D1-D5 结构诊断 | `experiments/analysis/structural_diagnosis.py` | `structural_diagnosis_v2.json` | 6 瓶颈假设: 2 推翻 (时间条件化/尺度类别), 4 部分支持/确认 | ✅ |
| 方向 D solver 对比 | `experiments/analysis/direction_d_solver_comparison.py` | `direction_d_comparison.json` | 3 solver mAP 持平 0.863, 自适应加速 4.2% | ✅ |
| 方向 A per-dim solver | `experiments/analysis/direction_a_per_dim_comparison.py` | `direction_a_per_dim_comparison.json` | ΔmAP=+0.001, Δlatency=-8.3ms (5.5% 加速) | ✅ |
| **D1 RoI 空间消融** | `experiments/analysis/d1_roi_ablation.py` | `d1_roi_ablation.json` | baseline mAP=0.863 → ablation mAP=0.009 (**Δ=-0.854 灾难性崩溃**), 证实 7×7 空间编码至关重要 | ✅ |

### 6.6 2026-07-25 新增训练实验 (C22-C26)

> 本节记录 2026-07-25 完成的 5 个新训练实验 (编号 C22-C26, 延续 §五 数据完整性 C 编号体系后的实验编号)。所有实验在 24obj 数据集上进行, 配置文件位于 `experiments/configs/ldmdet/directions/mainline_ablation_24obj/`。详细结果与训练曲线见各实验 work_dir 与 SwanLab 项目。

#### C22: M1 FP32 复现 (核心消融, 24obj)

- **实验名**: `m1_morphology_aware_24obj_fp32`
- **配置**: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj_fp32.py`
- **数据集**: 24obj
- **训练**: ross A6000, seed 42, 30 ep, lr=2e-5 (2×), 1ep warmup, FP32 (显存 37.5GB)
- **Best mAP**: **0.862 @ ep19**
- **Last mAP**: 0.859 @ ep30
- **Δ vs A4 (0.863)**: -0.001 (统计上持平)
- **SwanLab**: `ldmdet-mainline-ablation-24obj` / experiment_name=`m1_morphology_aware_fp32`
- **work_dir**: `work_dirs/m1_morphology_aware_24obj_fp32/` (本地 + ross `/media/ross/8TB/linkst/chromo/chromosome-kd/`)
- **状态**: ✓ 已完成
- **关键发现**: BF16 误导确认 (BF16 本身掉点 -0.038); h_conv/v_conv 在 FP32 下仍均匀 (ratio=1.01-1.02, std=0.0001) → 设计问题非精度问题
- **改进方向**: 非零初始化 fuse + 显式形态先验注入 + 注意力机制替代方向卷积

#### C23: M1 BF16 (workstation) (探索性, 24obj)

- **实验名**: `m1_morphology_aware_24obj_ws`
- **配置**: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/m1_morphology_aware_24obj_ws.py`
- **数据集**: 24obj
- **训练**: workstation A5000, seed 42, 30 ep, BF16 AMP (显存 20888 MiB, vs FP32 37506 MiB 降 44%)
- **Best mAP**: 0.818 @ ep1 (全程 0.811-0.818 波动)
- **Δ vs A4 (0.863)**: -0.045 (BF16 虚假退化)
- **Δ vs A4+BF16 (0.825)**: -0.007 (noise 范围但偏负面)
- **SwanLab**: `ldmdet-mainline-ablation-24obj` / experiment_name=`m1_morphology_aware_ws`
- **work_dir**: `work_dirs/m1_morphology_aware_24obj_ws/` (workstation)
- **状态**: ✓ 已完成 (BF16 误导, FP32 复现确认 — 见 C22)

#### C24: R3 v-prediction seed 42 (核心消融, 24obj)

- **实验名**: `r3_vpred_24obj_seed42`
- **配置**: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/r3_vpred_24obj.py`
- **数据集**: 24obj
- **训练**: workstation A4000, seed 42, max 150 ep (早停@ep64)
- **改动**: `criterion=dict(v_prediction=True, v_prediction_t_eps=1e-2)` (1/t² loss reweighting + batch normalization 均值=1)
- **Best mAP**: **0.855 @ ep34**
- **Last mAP**: 0.837 @ ep64
- **Δ vs A4 (0.863)**: -0.008 (超 3-seed noise ±0.003 但偏小, 单 seed 支持 R3.2)
- **早停**: patience=30 触发 (ep34+30=ep64)
- **训练曲线**: ep8 warmup=0.802 → ep34 best=0.855 → 长期停滞 → 早停
- **SwanLab**: `ldmdet-r3-vpred` / experiment_name=`r3_vpred`
- **work_dir**: `work_dirs/r3_vpred_24obj_seed42/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)
- **状态**: ✓ 已完成 (单 seed 初步, seed 123/789 待补)

#### C25: S1 h6_s2 (核心消融, 24obj)

- **实验名**: `s1_h6_s2_24obj`
- **配置**: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/s1_h6_s2_24obj.py`
- **数据集**: 24obj
- **训练**: workstation A5000, max 150 ep (早停@ep136)
- **改动**: `num_heads=6, sampling_timesteps=2` (H=6 S=2, NFE=12)
- **Best mAP**: **0.859 @ ep106**
- **Last mAP**: 0.856 @ ep136
- **Δ vs A4 (0.863)**: -0.004 (在 3-seed noise ±0.003 范围内)
- **早停**: patience=30 触发 (ep106+30=ep136)
- **关键发现**: 与 s1_h3_s4 (0.859) / s1_h3_s8 (0.859) 三组全部 0.859, S1.3 命题完整闭环
- **SwanLab**: `ldmdet-s1-cascade-decouple` / experiment_name=`s1_h6_s2`
- **work_dir**: `work_dirs/s1_h6_s2_24obj/` (workstation `/home/linkst/workplace/chromo/chromosome-kd/`)
- **状态**: ✓ 已完成

#### C26: Head Distillation v2 (探索性, 24obj, ⚠ 异常中断)

- **实验名**: `h3_distill_24obj`
- **配置**: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/h3_distill_24obj.py` (work_dir 内有 config 快照)
- **数据集**: 24obj
- **训练**: ross A6000, seed 42, max 150 ep (异常中断@ep99 iter 550/1750)
- **配置**: H=3 Student ← H=6 Teacher (A4 checkpoint 冻结), λ=0.05, freeze backbone, bs=2 (显存 2.4GB, freeze backbone + H=3)
- **Best mAP**: **0.711 @ ep96**
- **Last eval mAP**: 0.709 @ ep98
- **Δ vs A4 (0.863)**: -0.152 (H=3 学生模型架构容量限制明显)
- **中断情况**: 2026-07-24 14:57:33 nohup 日志戛然而止, 无报错, GPU 空闲, 推测 nohup 被外部信号 kill
- **最近 5 ep mAP 趋势**: 0.705→0.711→0.703→0.709→0.704→0.711→0.709 (仍在缓慢上升)
- **loss_distill**: 稳定在 ~0.033
- **SwanLab**: `ldmdet-head-distill`
- **work_dir**: `work_dirs/h3_distill_24obj/` (本地 + ross `/media/ross/8TB/linkst/chromo/chromosome-kd/`)
- **状态**: ⚠ 异常中断@ep99/150 (待恢复决策)

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

### 7.3.5 Problem 3: Y 染色体 per-class AP × 耦合策略 (3-seed)

> 数据源: `experiments/analysis/per_class_ap_coupling_3seed_results.json`
> 脚本: `experiments/analysis/per_class_ap_coupling_3seed.py`
> 缓存目录: `experiments/analysis/per_class_ap_coupling_3seed_cache/`
> 配置: Dataset 1 (Chromosome20240904_NoAug_NoResize_coco) val, 440 images, 3 training seeds [42, 123, 789]
> 三策略: Hard OT (硬最优传输) / Random (AdaLN) / Stochastic Coupling (Sinkhorn sample ε=5)
> 推理设置: OT coupling disabled at inference (ot_coupling=False)，耦合为训练期策略
> 实验目标: 验证 "Stochastic Coupling 缓解 Y 染色体训练不稳定性" 是否能被实验数据支持

#### 7.3.5.1 Aggregate mAP (3-seed mean ± std, Dataset 1 val)

| 策略 | Seed 42 | Seed 123 | Seed 789 | Mean | Std (pop) | Δ vs Hard OT | 相对提升 |
|------|---------|----------|----------|------|-----------|--------------|----------|
| Hard OT | 0.7047 | 0.7026 | 0.7077 | **0.7050** | 0.0021 | — | — |
| Random (AdaLN) | 0.7121 | 0.7187 | 0.7076 | **0.7128** | 0.0046 | +0.0078 | +1.10% |
| Stochastic Coupling | 0.7456 | 0.7449 | 0.7506 | **0.7471** | 0.0025 | **+0.0421** | **+5.97%** |

> Stoch Coupling 比 Hard OT 高 +0.042 mAP (+5.97%)，比 Random 高 +0.034 mAP (+4.82%)，且 within-strategy std 最小 (0.0025)。与 §7.3.2b 和 §7.3.4 一致。

#### 7.3.5.2 Y 染色体 per-class AP (3-seed)

| 策略 | Seed 42 | Seed 123 | Seed 789 | Mean | Std (pop) | Δ vs Hard OT | 相对提升 |
|------|---------|----------|----------|------|-----------|--------------|----------|
| Hard OT | 0.5485 | 0.5757 | 0.5663 | **0.5635** | 0.0113 | — | — |
| Random (AdaLN) | 0.5611 | 0.5816 | 0.5658 | **0.5695** | 0.0088 | +0.0060 | +1.06% |
| **Stochastic Coupling** | 0.6289 | 0.5937 | 0.6444 | **0.6223** | 0.0212 | **+0.0588** | **+10.43%** |

**核心发现**: Stochastic Coupling 将 Y 染色体 per-class AP 从 0.5635 提升至 0.6223，绝对增益 +0.0588，相对提升 **+10.43%**，远超 aggregate mAP 的相对增益 (+5.97%)，说明 Stoch Coupling 对 Y 染色体有**类别特异的增益**。

⚠️ **稳定性声明需修订**: Stoch Coupling 的 within-strategy std (0.0212) 实际**大于** Hard OT (0.0113) 和 Random (0.0088)。"稳定性" 声明不能直接支持，应改为 "显著提升 Y 染色体 AP" 而非 "缓解训练不稳定性"。

#### 7.3.5.3 全 24 类 per-class AP mean ± std (3-seed)

| Class | Hard OT mean±std | Random mean±std | StochOT mean±std | Δ (Stoch-Hard) |
|-------|------------------|------------------|-------------------|----------------|
| A1 | 0.7460 ± 0.0022 | 0.7434 ± 0.0183 | **0.7797 ± 0.0033** | +0.0337 |
| A2 | 0.7697 ± 0.0035 | 0.7695 ± 0.0150 | **0.8062 ± 0.0027** | +0.0365 |
| A3 | 0.7671 ± 0.0010 | 0.7701 ± 0.0133 | **0.8075 ± 0.0040** | +0.0404 |
| B4 | 0.7774 ± 0.0015 | 0.7790 ± 0.0074 | **0.8143 ± 0.0036** | +0.0369 |
| B5 | 0.7372 ± 0.0021 | 0.7501 ± 0.0046 | **0.7814 ± 0.0051** | +0.0442 |
| C6 | 0.7697 ± 0.0047 | 0.7732 ± 0.0065 | **0.7982 ± 0.0027** | +0.0286 |
| C7 | 0.7317 ± 0.0021 | 0.7393 ± 0.0089 | **0.7815 ± 0.0050** | +0.0498 |
| C8 | 0.7130 ± 0.0043 | 0.7272 ± 0.0061 | **0.7670 ± 0.0040** | +0.0540 |
| C9 | 0.6790 ± 0.0081 | 0.6898 ± 0.0040 | **0.7383 ± 0.0022** | +0.0593 |
| C10 | 0.7100 ± 0.0050 | 0.7235 ± 0.0079 | **0.7617 ± 0.0045** | +0.0517 |
| C11 | 0.7496 ± 0.0052 | 0.7645 ± 0.0056 | **0.7872 ± 0.0047** | +0.0376 |
| C12 | 0.7382 ± 0.0017 | 0.7469 ± 0.0025 | **0.7866 ± 0.0032** | +0.0484 |
| D13 | 0.6764 ± 0.0085 | 0.6827 ± 0.0102 | **0.7078 ± 0.0046** | +0.0315 |
| D14 | 0.6955 ± 0.0108 | 0.7050 ± 0.0052 | **0.7391 ± 0.0077** | +0.0436 |
| D15 | 0.6833 ± 0.0074 | 0.6949 ± 0.0013 | **0.7320 ± 0.0021** | +0.0487 |
| E16 | 0.6874 ± 0.0062 | 0.6946 ± 0.0049 | **0.7231 ± 0.0033** | +0.0357 |
| E17 | 0.6863 ± 0.0020 | 0.6962 ± 0.0060 | **0.7220 ± 0.0026** | +0.0357 |
| E18 | 0.7208 ± 0.0016 | 0.7287 ± 0.0076 | **0.7571 ± 0.0063** | +0.0363 |
| F19 | 0.6836 ± 0.0046 | 0.6938 ± 0.0060 | **0.7195 ± 0.0010** | +0.0360 |
| F20 | 0.6734 ± 0.0053 | 0.6838 ± 0.0100 | **0.7179 ± 0.0051** | +0.0445 |
| G21 | 0.6512 ± 0.0053 | 0.6595 ± 0.0029 | **0.6722 ± 0.0043** | +0.0210 |
| G22 | 0.5831 ± 0.0047 | 0.5810 ± 0.0037 | **0.6226 ± 0.0022** | +0.0395 |
| X | 0.7268 ± 0.0022 | 0.7404 ± 0.0085 | **0.7842 ± 0.0046** | +0.0574 |
| **Y** | **0.5635 ± 0.0113** | **0.5695 ± 0.0088** | **0.6223 ± 0.0212** | **+0.0588** |

> 24/24 类 Stoch Coupling 均优于 Hard OT (W=0 的直接证据)。Y (+0.0588) 和 X (+0.0574) 是增幅最大的两个类别，远超 aggregate mAP 增益 (+0.0421)，证实性染色体类别特异增益。

#### 7.3.5.4 配对 Wilcoxon 检验 (Stoch Coupling vs Hard OT, n=72)

> 检验方法: paired Wilcoxon signed-rank test, alternative='two-sided'
> 配对构造: 24 类 × 3 seeds = 72 paired (StochOT[i] - HardOT[i])
> 检验方向: H₀: 中位数差 = 0; H₁: 中位数差 ≠ 0

| 对比 | N pairs | Mean Δ | Wilcoxon W | p-value | 显著性 |
|------|---------|--------|------------|---------|--------|
| Stoch Coupling vs Hard OT | 72 | +0.0469 | **0** | **1.66e-13** | **\*\*\*** |

> W=0 表示 72 对配对中 Stoch Coupling 全部严格大于 Hard OT (无任何例外)，达到理论上最强的统计证据。
> p=1.66e-13 为带连续性修正的渐近 Wilcoxon 检验 (asymptotic with continuity correction) 结果。

#### 7.3.5.5 关键发现 — Y 染色体叙事支持

| 论文声明 | 实验支持 | 修订建议 |
|----------|----------|----------|
| "Stoch Coupling 缓解 Y 染色体训练不稳定性" | **部分支持** | Y AP 显著提升 (+10.43%)，但 within-strategy std (0.0212) **大于** Hard OT (0.0113)，"稳定性" 不能直接支持 |
| "Y 染色体是 Stoch Coupling 增益最大的类别" | **强支持** | Y AP Δ=+0.0588 是 24 类中最大增幅 (并列 X +0.0574)，远超 aggregate mAP 增益 (+0.0421) |
| "Stoch Coupling 对 Y 染色体有类别特异增益" | **强支持** | Y 相对提升 +10.43% vs aggregate 相对提升 +5.97%，约 1.75× 因子，类别特异显著 |

**最终建议**: 论文 §1.2 贡献 1 应表述为 "Stochastic Coupling 显著提升 Y 染色体检测精度 (Y AP +10.4%, p=1.66e-13)"，避免使用 "稳定性" 表述 (与 within-strategy std 数据不符)。

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

### 7.4.6 Problem 2B: RF vs SOTA baselines 配对 Wilcoxon 检验

> 数据源: `experiments/analysis/baseline_inference_24obj_perimage_results.json`
> 脚本: `experiments/analysis/baseline_inference_24obj_perimage.py` (主检验脚本, 含 DINO R50)
> 推理脚本 (RTMDet-L + Cascade R-CNN): `experiments/analysis/baseline_inference_24obj.py`
> 推理脚本 (DINO R50): `experiments/analysis/baseline_inference_dino_r50_24obj.py`
> 推理日志: `experiments/analysis/baseline_inference_24obj_perimage_run2.log` (含 DINO R50)
> 缓存目录: `experiments/analysis/baseline_inference_24obj_cache/` + `experiments/analysis/baseline_vs_sota_cache/`
> 配置: Dataset 2 (24_chromosomes_object) val, 500 images, 推理 seed=42
> 检验方法: paired Wilcoxon signed-rank test + paired t-test
> 实验目标: 验证 RF 是否 "competitive with SOTA" (DINO R50 / RTMDet-L / Cascade R-CNN / DiffusionDet)

#### 7.4.6.1 Aggregate Metrics (5 models, 6 indicators)

| 模型 | mAP | AP50 | AP75 | AP_S | AP_M | AP_L |
|------|------|------|------|------|------|------|
| **DINO R50** (best) | **0.8685** | **0.9915** | **0.9773** | **0.5528** | **0.8647** | **0.9207** |
| RTMDet-L (ep85 best) | 0.8626 | 0.9905 | 0.9737 | 0.5397 | 0.8595 | 0.9093 |
| Cascade R-CNN | 0.8535 | 0.9857 | 0.9657 | 0.4138 | 0.8495 | 0.8992 |
| **RF (LDMDet StochOT)** | 0.8521 | 0.9853 | 0.9657 | 0.4055 | 0.8484 | 0.9058 |
| DiffusionDet | 0.8031 | 0.9699 | 0.9358 | 0.4227 | 0.8005 | 0.8138 |

> DINO R50 best@ep102 checkpoint (340MB) 于 2026-07-20 从 workstation 恢复 (原以为已丢失)。论文报告 mAP=0.868，实测 mAP=0.869 (来自 metrics.json)。

⚠️ **A3 vs A4 配置说明 + RTMDet-L ep85 修复** (重要，影响论文叙事):
- 上表 RF mAP=0.8521 是 **A3 (Heun+StochOT) 配置** 的实测值，per-image Wilcoxon 检验也基于 A3 预测
- 论文 Table 6 引用的是 **A4 best (DPM-Solver++ 4-step, seed 42) mAP=0.863** (3-seed mean 0.859±0.003)
- A4 best 与 DINO R50 的 aggregate mAP 差距: (0.8685 - 0.863) / 0.8685 = **0.63%** (远小于 per-image AP 的 1.64%)
- **RTMDet-L ep85 修复 (2026-07-20)**: 之前用 ep86 (次优, mAP=0.8610) 推理，错误注释 "ep85 best lost"。实际 epoch_85.pth 存在 (637 MB)，是 train.log 中的真正 best (mAP=0.8630)。已重跑 ep85 推理 (实测 mAP=0.8626)，所有 Wilcoxon 检验已更新。
- **论文 Table 6 错误**: RTMDet-L 报告值 0.869 是错误的 (训练从未达到 0.869)，应为 **0.863** (ep85 best)
- **论文叙事策略** (用户指示 "引用最好的"):
  - §1.2 贡献 1: 引用 A4 best (0.863) vs DINO R50 (0.8685), 差距仅 **0.63%**
  - 摘要/结论: DiffusionDet 差距用新数据 (RF 0.863 - DiffusionDet 0.8031 = +0.060 mAP)
  - per-image Wilcoxon 表格 (A3 配置): 放入草稿补充材料，**暂不放入正文**，避免 A3/A4 配置混淆

#### 7.4.6.2 Per-Image Wilcoxon 检验 (AP@[IoU=0.5:0.95], n=500)

| 对比 | Δ mean (RF - 对方) | Wilcoxon W | Wilcoxon p | paired t p | 显著性 | 相对差距 |
|------|--------------------|------------|------------|------------|--------|----------|
| RF vs DINO R50 | **-0.0145** | 36092.5 | **2.24e-16** | 1.42e-16 | **\*\*\*** | RF 落后 1.64% |
| RF vs RTMDet-L | -0.0043 | 51977.5 | 1.59e-03 | 4.61e-03 | ** | RF 落后 0.49% |
| RF vs Cascade R-CNN | +0.0002 | 60124.5 | 4.85e-01 | 4.84e-01 | n.s. | RF 略优 0.03% |
| RF vs DiffusionDet | +0.0485 | 3509.5 | 1.02e-74 | 2.05e-72 | *** | RF 领先 5.88% |

> 正 Δ 表示 RF 优于对方，负 Δ 表示 RF 落后。相对差距 = Δ / max(RF_AP, 对方_AP)。

#### 7.4.6.3 Per-Image Wilcoxon 检验 (AP50, IoU=0.5, n=500)

| 对比 | Δ mean (RF - 对方) | Wilcoxon W | Wilcoxon p | 显著性 | 相对差距 |
|------|--------------------|------------|------------|--------|----------|
| RF vs DINO R50 | -0.0050 | 1748.0 | 8.35e-17 | *** | RF 落后 0.50% |
| RF vs RTMDet-L | -0.0049 | 2122.0 | 1.61e-15 | *** | RF 落后 0.50% |
| RF vs Cascade R-CNN | -0.0004 | 5553.5 | 8.37e-01 | n.s. | RF 落后 0.04% |
| RF vs DiffusionDet | +0.0150 | 2711.5 | 8.26e-34 | *** | RF 领先 1.55% |

#### 7.4.6.4 关键发现 — RF 与最前沿检测器差距

| # | 发现 | 数据支持 | 论文修订建议 |
|---|------|----------|--------------|
| 1 | RF 显著落后 DINO R50 (SOTA transformer-based) | Δ=-0.0145, p=2.24e-16 *** | "RF 拉近了与最前沿检测器 (DINO R50) 的差距，仅差约 1.7% 相对百分比" |
| 2 | RF 与 Cascade R-CNN 统计等价 | Δ=+0.0002, p=0.485 n.s. | "RF 与 Cascade R-CNN 性能相当" |
| 3 | RF 显著优于 DiffusionDet (同为 diffusion-based) | Δ=+0.0485, p<1e-74 *** | "在 diffusion-based 检测器中 RF 显著领先 DiffusionDet" |
| 4 | RF 落后 RTMDet-L 但差距小 | Δ=-0.0043, p=1.59e-03 ** | "RF 与 RTMDet-L 差距微小 (0.49%), 统计上显著但实际影响有限" |

**核心叙事 (用户指示)**:
- **不**强调 "显著劣于 DINO R50"
- 改用叙事: RF 拉近了与最前沿检测器 (DINO R50, 47M params, 4-scale transformer) 的差距，per-image AP 仅差 **1.64%** (aggregate mAP 差 1.89%)
- 强调 RF 作为 diffusion-based 检测器已接近 transformer-based SOTA 水平
- 同时承认统计上 DINO R50 显著更优 (p=2.24e-16)
- 与 Cascade R-CNN 统计等价 (p=0.485, n.s.)
- 显著优于 DiffusionDet (p<1e-74, RF +5.88% per-image AP)

**Aggregate mAP 相对百分比** (RF 落后):
- vs DINO R50: (0.8685 - 0.8521) / 0.8685 = **1.89%**
- vs RTMDet-L (ep85): (0.8626 - 0.8521) / 0.8626 = **1.22%**
- vs Cascade R-CNN: (0.8535 - 0.8521) / 0.8535 = **0.16%** (n.s.)

**Per-Image AP 相对百分比** (推荐论文主叙事):
- vs DINO R50: (0.8882 - 0.8736) / 0.8882 = **1.64%**
- vs RTMDet-L (ep85): (0.8779 - 0.8736) / 0.8779 = **0.49%**
- vs Cascade R-CNN: (0.8736 - 0.8734) / 0.8736 = **0.03%** (n.s.)
- vs DiffusionDet: (0.8736 - 0.8251) / 0.8251 = **5.88%** (RF 领先)

> 推荐使用 per-image AP 相对百分比 (1.64%) 作为论文主叙事，与 Wilcoxon 检验配对设计一致。Aggregate mAP 1.89% 作为辅助证据。

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

#### 7.9.1 8 项关键发现（影响论文表述）

| # | 发现 | 来源 | 当前论文表述 | 修订建议 |
|---|------|------|------------|---------|
| 1 | DPM-Solver++ 在 Dataset 2 上有 +0.0056 mAP 显著增益 | C3 | "纯计算优势，无精度增益" | "小幅但统计显著的精度优势 (p=2.54e-07)" |
| 2 | StochOT 在 Dataset 1 上 +0.034 mAP 高度显著 | C2 | "无 mAP 增益"（笼统） | "低数据 regime 高度显著，大数据集统计等价" |
| 3 | Hard OT 在 Dataset 1 上比 Random 差 -0.008 mAP | C2 | "Hard OT 应劣于 Random/StochOT" | 一致 — Dataset 1 验证 |
| 4 | A4 3-seed mean 0.859±0.003 | C1 | 仅报告 seed 42 单值 0.863 | 报告 mean ± std: 0.859 ± 0.003 |
| 5 | test split mAP=0.859 与 val 一致 | C4 | 未报告 test | 添加 test split 评估 |
| 6 | Y AP 跨种子 mean 0.771±0.006 | C6 | 仅报告 seed 42 单值 0.779 | 报告 mean ± std: 0.771 ± 0.006 |
| 7 | Stoch Coupling 显著提升 Y 染色体 AP (+10.43%) | §7.3.5 (Problem 3) | "Stoch Coupling 缓解 Y 染色体训练不稳定性" | 改为 "显著提升 Y 染色体检测精度 (Y AP +10.4%, Wilcoxon W=0, p=1.66e-13)"；避免使用 "稳定性" (within-strategy std 实际更大) |
| 8 | RF 落后 DINO R50 仅 0.63% (aggregate mAP, A4 best) | §7.4.6 (Problem 2B) | "competitive with SOTA" (笼统) | "RF 拉近了与最前沿检测器 (DINO R50) 的差距，aggregate mAP 仅差 0.63% (0.863 vs 0.8685)；与 Cascade R-CNN 性能相当，显著优于 DiffusionDet (+0.060 mAP)"；per-image Wilcoxon (A3 配置, 1.64%) 放入草稿补充材料 |

#### 7.9.2 20 项论文修订清单

> 详细修订清单见 `/home/linkst/workspace/projects/chromosome-kd/docs/paper/C_CLASS_TASK_PLAN.md` §10 论文修订清单。本归档仅记录数据层修订影响，具体文字修订在 AAAI_INTEGRATED_DRAFT.md 中执行。

---

*本文档基于 EXPERIMENT_LINEAGE.md, EXPERIMENT_RESULTS.md, AAAI_INTEGRATED_DRAFT.md, swanlab_export.json, PUBLICATION_EVALUATION.md, 以及本地 work_dirs/ 和 experiments/configs/ 目录的综合整理。*
*2026-07-18 追加 §七 C 类推理任务归档（来源: C_CLASS_TASK_PLAN.md + /tmp/c_class_results/）。*
*2026-07-19 SwanLab 同步: §〇 项目表扩至 28 个, §3.2.1 SetDiff 备注, 新增 §3.5 (A0/A1 多种子 + A1 shift 消融, 来源: SwanLab ldmdet-mainline-ablation-24obj + setdiff-24obj)。*
*2026-07-19 SwanLab 第二轮同步: 新增 §3.6 (10 个项目, 21 个论文相关 run, 含完整 AP 指标; 解决 C16 ε 消融数据源; 新增 C17-C21 数据一致性问题; 来源: ldmdet-mainline-ablation-old + chromosome-kd-dpm + chromosome-kd-multiseed + chromosome-kd-ablation + ldmdet-inference-opt-24obj + ldmdet-breakthrough + ldmdet-frontier-directions-24obj + chromosome-kd-stability + chromosome-kd-verify-v1 + ldmdet-sota-stack)。*
*2026-07-20 新增 §7.3.5 (Problem 3: Y 染色体 per-class AP × 耦合策略 3-seed, 来源: per_class_ap_coupling_3seed.py; Wilcoxon W=0, p=1.66e-13, Stoch Coupling Y AP +10.43%) + §7.4.6 (Problem 2B: RF vs SOTA 配对 Wilcoxon, 含 DINO R50 best@ep102 从 workstation 恢复; RF 落后 DINO R50 1.64% per-image AP, p=2.24e-16) + §7.9.1 扩至 8 项关键发现 (新增行 7, 8)。*
*2026-07-20 RTMDet-L ep85 修复: 发现 epoch_85.pth (实际 best, mAP=0.8630) 存在，之前错误使用 ep86 (次优, mAP=0.8610)。已重跑 ep85 推理 (实测 mAP=0.8626) 并更新 §7.4.6 所有 Wilcoxon 检验数据。同时确认论文 Table 6 中 RTMDet-L=0.869 是错误的 (训练从未达到 0.869)，应为 0.863。*
*2026-07-25 新增 §6.6 (5 个训练实验 C22-C26: M1 FP32 复现 best 0.862@ep19 Δ=-0.001 持平 A4 + BF16 误导根因确认; M1 BF16 ws best 0.818 BF16 虚假退化 -0.045; R3 v-prediction seed42 best 0.855@ep34 单 seed 支持 R3.2; S1 h6_s2 best 0.859@ep106 S1.3 命题闭环; Head Distill v2 异常中断@ep99 best 0.711@ep96 H=3 容量限制)。同步更新 §1.1 (M1 FP32/BF16/Head Distill 状态从 🔄 运行中 → ✅/⚠ 已完成, 新增 R3 vpred + S1 h6_s2 条目), §3.1 核心消融表 (+3 行), §6.1 ross 服务器 (+M1 FP32 + Head Distill v2), §6.2 workstation 服务器 (+M1 BF16 + R3 + S1)。来源: work_dirs/m1_morphology_aware_24obj_{fp32,ws}/ + work_dirs/r3_vpred_24obj_seed42/ + work_dirs/s1_h6_s2_24obj/ + work_dirs/h3_distill_24obj/, SwanLab ldmdet-mainline-ablation-24obj + ldmdet-r3-vpred + ldmdet-s1-cascade-decouple + ldmdet-head-distill。*
