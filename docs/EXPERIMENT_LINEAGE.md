# 实验脉络关系总览

> 本文档梳理所有实验的递进关系,明确真正的 baseline,识别废弃/错误实验。
> 每条实验记录附 **可靠数据源地址** (本地服务器路径 / ldmdet-experiment 归档 / SwanLab run_id)
> 更新时间: 2026-07-02 (最近一次刷新: 补充非线性轨迹/生成式迁移/Few-Shot 实验系列, 修正 eps2 旧实验归因)

## 〇、数据源说明

所有实验数据来源 **三类**:

| 来源 | 路径/格式 | 用途 |
|------|----------|------|
| **本地服务器日志** | `work_dirs/<exp_dir>/<timestamp>/<timestamp>.log` + `vis_data/scalars.json` | 完整训练曲线 + 配置快照 |
| **ldmdet-experiment 归档** | `ldmdet-experiment/sota/<category>/<exp_name>/` (含 README, config.py, metrics.json, code/, checkpoints/) | 已归档 SOTA 实验 (2026-06-13) |
| **SwanLab 云端** | `https://swanlab.cn/@einspanner/<project>/runs/<run_id>` (project ∈ {chromosome-kd, chromosome-kd-benchmark-24obj, ldmdet-ablation}) | 在线可视化 + 跨实验对比 |

> SwanLab 用户名: `einspanner` (登录态见 `/home/linkst/.swanlab/.netrc`, api_key 已配置)。
> 已知 project: `chromosome-kd` (早期 21 个), `chromosome-kd-benchmark-24obj` (24obj 数据集 1 个), `ldmdet-ablation` (主线 32+ 个, 含 chromo/24obj/merged 数据集实验)。
> URL 拼接示例: `https://swanlab.cn/@einspanner/ldmdet-ablation/runs/<run_id>`

## 一、aug 策略统一基准 (关键修正)

**统计 73 个实验的 train_pipeline**:

| Aug 类型 | 实验数 | 代表 | 评价 |
|---------|-------|------|------|
| **multi_scale + RandomCrop (DiffusionDet 默认 aug)** | **56** | multi_seed_aug/*, sota_*, bottleneck/*, direction_exps/* | ✅ 统一基准 |
| simple_resize (仅 Resize + RandomFlip,无 multi-scale/crop) | 9 | multi_seed/* (mAP=0.712) | ❌ 非标准简化 |
| multi_scale_only (无 RandomCrop) | 8 | scheme_a_dinov2_s (失败系列) | ❌ |

**结论**: 多数实验 (56/73) 使用 DiffusionDet 默认 aug。
- `multi_seed/rf_heun_adaln` (0.712, **无 aug**) 是简化非标准实验,**不可作为 baseline**
- `multi_seed_aug/rf_heun_adaln` (0.746, **DiffusionDet 默认 aug**) 是标准 baseline
- 用户判断正确: aug 是 DiffusionDet 默认,不是额外增强

## 二、数据集分组 (不可跨数据集对比!)

| 数据集 | 路径 | 实验简称 | 实验组 | mAP 量级 | SwanLab project |
|--------|------|----------|--------|---------|------------------|
| **Chromosome20240904** | `data/Chromosome20240904_NoAug_NoResize_coco/` | chromo | multi_seed_aug/*, sota_*, bottleneck/*, direction_exps/*, ddpm | 0.72-0.75 | ldmdet-ablation, chromosome-kd |
| **24_chromosomes_object** | `data/24_chromosomes_object/coco/` | 24obj | 24obj_ablation/*, ldmdet_rf_heun_adaln_stochot_eps5 | 0.81-0.86 | ldmdet-ablation, chromosome-kd-benchmark-24obj |

> ⚠️ 跨数据集对比错误示例: ghss@24obj (0.857) vs rf_heun_adaln@chromo (0.746) — 不可对比
>
> 补充: merged_ablation 使用两数据集合并训练 (24obj + Chromosome20240904), 属跨数据集实验, 单独记录于递进树末尾

## 三、实验递进树

### 3.1 Chromosome20240904 数据集 (简称 chromo)

```
DiffusionDet DDPM (根 baseline, 默认 aug)
│  config: experiments/configs/baselines/diffusiondet_ddpm.py
│  result: 0.729 ± 0.003 (3 seeds)
│  本地: work_dirs/multi_seed_aug/ddpm/seed_{42,789,123}/
│  SwanLab (project=ldmdet-ablation):
│    diffusiondet_ddpm_seed42  run_id=apfn46t67iqg1bjraq8xd  mAP=0.7260
│    diffusiondet_ddpm_seed789 run_id=hny1od5fcvx8ngt9b063g  mAP=0.7270
│    diffusiondet_ddpm_seed123 run_id=ghghjry3bylt0sfoj30j5  mAP=0.7330
│
│  步数对齐验证 (DDIM 多步推理, project=ldmdet-inference):
│    DDIM 4-step:  0.729 ± 0.004  (seed42=0.727, seed123=0.734, seed789=0.726)
│    DDIM 8-step:  0.729 ± 0.003  (seed42=0.728, seed123=0.733, seed789=0.726)
│    → DDPM 加步数不提升, 1/4/8 步均为 0.729; +0.017 为纯算法贡献
│
├─→ + RF + Heun + Shifted Schedule + AdaLN-Zero  (DDPM → Rectified Flow)
│      config: experiments/configs/ldmdet/rf_heun_adaln.py
│      result: 0.746 ± 0.001 (3 seeds)  [+0.017, 主贡献]
│      本地: work_dirs/multi_seed_aug/rf_heun_adaln/seed_{42,789,123}/
│      SwanLab (project=ldmdet-ablation):
│        rf_heun_adaln_seed42  run_id=4xhp5ffymboa05hyn245u  mAP=0.7450
│        rf_heun_adaln_seed789 run_id=ww6nlti3ufdkm4htjg5pw  mAP=0.7470
│        rf_heun_adaln_seed123 run_id=dimdbu8fk0re4satbzpgs  mAP=0.7470
│      关键改动: diffusion_type=rectified_flow, solver=heun, rf_schedule=shifted, time_conditioning=adaln_zero
│      │
│      ├─→ + DPM-Solver++ 推理加速 (推理时改采样器, 不重训)
│      │      baseline ckpt: work_dirs/multi_seed_aug/rf_heun_adaln/seed_{42,789,123}/best_coco_bbox_mAP_epoch_*.pth
│      │      config: experiments/configs/ldmdet/rf_heun_adaln.py + test.py --solver-type dpm_solver_pp[_3]
│      │      ┌─ 步数对齐 (4 steps, 公平步数对比):
│      │      │  o2 (≈5 NFE): 0.746 ± 0.001  [Δ=+0.000, 同等步数下 NFE 降 37%]
│      │      │    seed42=0.7450  seed789=0.7470  seed123=0.7470
│      │      │  o3 (≈6 NFE): 0.746 ± 0.001  [Δ=+0.000, 同等步数下 NFE 降 25%]
│      │      │    seed42=0.7450  seed789=0.7470  seed123=0.7470
│      │      └─ NFE 对齐 (6 steps, 公平计算量对比, ≈7-8 NFE):
│      │         o2 (≈7 NFE): 0.747 ± 0.001  [Δ=+0.001, 边际, NFE 略低于 Heun 8]
│      │           seed42=0.7460  seed789=0.7480  seed123=0.7480
│      │         o3 (8 NFE): 0.747 ± 0.001  [Δ=+0.001, 边际, 同等 NFE]
│      │           seed42=0.7460  seed789=0.7470  seed123=0.7480
│      │      SwanLab (project=ldmdet-inference): 12 runs (dpm_solver_pp[_3]_s{4,6}_seed{42,789,123})
│      │        s6 o2: 5yvbs46dy657tbh7nu01z / pnbojx58ha5diof8czp96 / la6gtelnq4iifjw3l40ax
│      │        s6 o3: i70i148vlesr1ww79lu52 / nf7i2l4zizaj3fqds5zbh / 9oplynuqswxo4i383l09o
│      │      ⚠ 注: 旧档 ldmdet_dpm_solver_pp_o2_s8=0.748 基于简化 aug 旧 baseline(0.728), 不可与新 baseline(0.746) 直接对比
│      │      理论: docs/notes/dpm_solver_plus_plus_rf_derivation.md (t 空间多步法, 半线性精确积分)
│      │      结论: 步数对齐时 DPM-Solver++ 持平 Heun 但 NFE 降 25-37%; NFE 对齐时边际 +0.001
│      │
│      ├─→ + Hard OT Coupling (边际)
│      │      result: 0.747 ± 0.000 (2 seeds)  [+0.001]
│      │      本地: work_dirs/multi_seed_aug/hard_ot/seed_{42,123}/
│      │      SwanLab: hard_ot_seed42 (h8fizm7lmc9v5xzxi8ufj), hard_ot_seed123 (kka4nra9qk3wanx7i9og1)
│      │
│      ├─→ + Sinkhorn Stochastic OT (边际)
│      │      result: 0.748 (1 seed)  [+0.002]
│      │      本地: work_dirs/multi_seed_aug/sinkhorn_stochastic/seed_42/
│      │      SwanLab: sinkhorn_stochastic_seed42 run_id=9ca697vnm1l3koccbenif  mAP=0.7480
│      │
│      ├─→ + GHSS Coupling (未完成)
│      │      本地: work_dirs/multi_seed_aug/ghss/seed_42/ (mAP=0,未跑完)
│      │      SwanLab: ghss_seed42 run_id=t44ol7fdplhjbzs4sd8uy
│      │
│      └─→ SOTA (Sinkhorn Stochastic + ot_coupling=True)
│             config: work_dirs/sota_seed{42,123,456,1000}/sota_seed*.py (动态生成,无固定源)
│             result: 0.742 ± 0.008 (4 seeds, best 0.749)  [+0.003, 高方差]
│             本地: work_dirs/sota_seed{42,123,456,1000}/
│             SwanLab (project=chromosome-kd):
│               sota_seed42  run_id=9xswp5aj7rmfd4vys6906  mAP=0.7400
│               sota_seed123 run_id=b31e1xhzftod7ae38cs17  mAP=0.7490
│               sota_seed456 run_id=ukxsyw666y1xrpcrtjvx1  mAP=0.7460
│               sota_seed1000 run_id=ocdkvjs2vs1vozbh0goni  mAP=0.7490
│             ⚠ SOTA 平均(0.742)低于 sinkhorn_stochastic(0.748),高方差(0.727-0.749)
│
├─→ 历史 SOTA 归档 (ldmdet-experiment, 2026-06-13)
│      归档: ldmdet-experiment/sota/<phase>/<exp_name>/
│      index: ldmdet-experiment/index.json
│      README: ldmdet-experiment/README.md
│      │
│      ├─→ phase5_stochastic_ot: reproduce_0751_stochot_eps5_v2 = 0.753  [历史 SOTA]
│      │      归档: ldmdet-experiment/sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/
│      ├─→ phase7_loss: scheme_C1_5_mixed_rel_l1_lam015 = 0.752
│      ├─→ phase8_kcec: ldmdet_kcec_redundant_slots_m2 = 0.748
│      ├─→ phase9_dpm_solver: ldmdet_dpm_solver_pp_o2_s8 = 0.748
│      ├─→ phase7_small_obj: smallobj_B_scaleaware_loglinear = 0.744
│      ├─→ phase8_daec: ldmdet_daec_contrastive_kcec_eps5 = 0.740
│      ├─→ phase0_pretrain: ldmdet_convnextv2_mae = 0.736
│      └─→ baselines (非 LDMDet): dino_r50 = 0.869
│
└─→ Bottleneck 瓶颈分析 (基于 SOTA 模型)
       config: experiments/configs/bottleneck/*.py
       报告: work_dirs/bottleneck/bottleneck_report.json
       分析: experiments/configs/bottleneck/EXPERIMENT_ANALYSIS.md
       │
       ├─→ focal_gamma_3 = 0.750  [+0.004 vs SOTA 0.746]
       │      本地: work_dirs/bottleneck/ablation/focal_gamma_3/20260628_013823/
       │      SwanLab: focal_gamma_3_seed42 run_id=ye6a2whory9y67tnvalg3
       ├─→ focal_gamma_1_5 = 0.747
       │      本地: work_dirs/bottleneck/ablation/focal_gamma_1_5/20260628_100256/
       │      SwanLab: focal_gamma_1_5_seed42 run_id=1f0r738snqeljucu6jm03
       ├─→ scale_aware_loss = 0.742
       │      本地: work_dirs/bottleneck/ablation/scale_aware_loss/20260624_112630/
       │      SwanLab: scale_aware_loss run_id=o8elke3rcmln1i47fznr5
       ├─→ relative_l1_loss = 0.740
       │      本地: work_dirs/bottleneck/ablation/relative_l1_loss/20260626_093627/
       │      SwanLab: relative_l1_loss run_id=cmxgctni1n1d9g9go0ywf
       ├─→ high_cls_weight = 0.739
       │      本地: work_dirs/bottleneck/ablation/high_cls_weight/20260628_180423/
       │      SwanLab: high_cls_weight_seed42 run_id=m9dnqrl43khkl3jcbjico
       ├─→ high_giou_weight = 0.737
       │      本地: work_dirs/bottleneck/ablation/high_giou_weight/20260625_230646/
       │      SwanLab: high_giou_weight run_id=b3w4gwly842j700yqg9og
       ├─→ no_box_renewal = 0.730 (消融)
       │      本地: work_dirs/bottleneck/ablation/no_box_renewal/20260623_211757/
       │      SwanLab: no_box_renewal run_id=52o1g5pq09eddplyzbl9q
       ├─→ class_balanced_sampling = FAILED (SIGKILL)
       │      本地: work_dirs/bottleneck/ablation/class_balanced_sampling/20260629_013833/ (无 scalars.json)
       │      SwanLab: class_balanced_sampling_seed42 run_id=vsq3xd3bw51iss1n1at15
       └─→ proposals_100 = FAILED (SIGKILL)
              本地: work_dirs/bottleneck/ablation/proposals_100/20260624_021155/ (无 scalars.json)
              SwanLab: proposals_100 (无 run_id)

Direction 方向实验 A-F (基于 rf_heun_adaln, 默认 aug, 应与 0.746 对照)
       config: experiments/configs/ldmdet/direction_*.py
       │
       ├─→ D (BoxRefineNet) = 0.747  [+0.001 vs 0.746, 持平]  ✓ 完成 (early stop @ epoch 85)
       │      本地: work_dirs/direction_exps/direction_d_box_refine/20260629_091843/
       │      SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/fnoz9x82aor1utsuo0jtl
       │              run_id=fnoz9x82aor1utsuo0jtl  mAP=0.7470  (best @ epoch 55)
       ├─→ B (DecoupledHead) = 0.702  [-0.044, 显著低于 baseline]  ⛔ 已停止
       │      本地: work_dirs/direction_exps/direction_b_decoupled_head/20260629_152911/
       │      SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/2ckzmso4c94fojr8jobej
       │              run_id=2ckzmso4c94fojr8jobej  best mAP=0.7020  (best @ epoch 25)
       ├─→ C (Morphology+Contrastive) = ⛔ 已废弃 (未启动)
       ├─→ F (StructuredPrior+100 proposals) = ⛔ 已废弃 (未启动)
       ├─→ A (P1+Deformable) = ⛔ 已废弃 (未启动)
       └─→ E (ClassBalanced) = ⛔ 已废弃 (未启动)

       ⚠ Direction 系列已停止推进, 仅 D 持平 baseline, B 显著退化。
       后续非线性轨迹实验 (见下) 接替该方向继续探索。

非线性轨迹实验 (基于 rf_heun_adaln + ScaleConditionedRF + OTFlowCoupling, chromo 数据集)
       config: experiments/configs/ldmdet/nonlinear_trajectory*.py
       核心: ScaleConditionedRF (尺度调制噪声调度 κ(s)) + OTFlowCoupling (Sinkhorn OT 传输矩阵耦合)
       baseline 对照: rf_heun_adaln = 0.746
       │
       ├─→ E4.1 (ScaleConditionedRF only, 无 OT) = 0.743  [-0.003, 持平]  ✓ 早停
       │      config: experiments/configs/ldmdet/nonlinear_trajectory_e41.py
       │      改动: coupling=random (去 OT), 保留 scale_conditioned_rf
       │      本地: work_dirs/nonlinear_trajectory_e41/
       │      SwanLab: run_id=qdnw5yyj  best mAP=0.7430  (best @ epoch 74)
       │
       ├─→ E4.2 (OT Flow only, 无 ScaleConditionedRF) = 0.751  [+0.005]  ✓ 早停
       │      config: experiments/configs/ldmdet/nonlinear_trajectory_e42.py
       │      改动: lambda_mod=0.0 (关闭尺度条件), 保留 OT Flow
       │      本地: work_dirs/nonlinear_trajectory_e42/
       │      SwanLab: run_id=wcp34v3t  best mAP=0.7510  (best @ epoch 81)
       │
       ├─→ E4.3 (ScaleConditionedRF + OTFlowCoupling argmax eps=1.0) = 0.752  [+0.006]  ✓ 早停
       │      config: experiments/configs/ldmdet/nonlinear_trajectory.py
       │      本地: work_dirs/nonlinear_trajectory/
       │      SwanLab: run_id=usnvd63f  best mAP=0.7520  (best @ epoch 94)
       │      ⚠ 注意: 此实验的 ScaleConditionedRF 当时未真正集成到 head.py,
       │        0.752 主要来自 OTFlowCoupling + 种子方差
       │      │
       │      ├─→ E4.3-tune eps=2.0 (OLD, ScaleConditionedRF 未启用) = 0.752  [+0.006]  ✓ 早停
       │      │      config: experiments/configs/ldmdet/nonlinear_trajectory_e43_eps2.py
       │      │      改动: coupling.epsilon=1.0→2.0 (更平滑传输矩阵)
       │      │      本地: work_dirs/nonlinear_trajectory_e43_eps2/
       │      │      SwanLab: run_id=cdtmijl0  best mAP=0.7520  (best @ epoch 100)
       │      │      ⚠⚠ 关键修正: 配置 dump 虽含 scale_conditioned_rf 字段,
       │      │        但当时 head.py 未集成, ScaleConditionedRF 完全未生效!
       │      │        0.752 实际来自 OTFlowCoupling(eps=2.0,argmax) + Heun + 种子方差
       │      │
       │      ├─→ E4.3-tune eps=2.0 (NEW, ScaleConditionedRF 真正启用) = **0.741** [−0.005] ⛔ 证伪
      │      │      config: experiments/configs/ldmdet/nonlinear_trajectory_e43_eps2.py (同上)
      │      │      本地: work_dirs/nonlinear_trajectory_e43_eps2_real/
      │      │      SwanLab: run_id=hkn0fc7w  (best @ epoch 69, 训练至 ep93 平台化)
      │      │      进展: ep1=0.000 → ep10=0.564 → ep55=0.737 → ep69=**0.741** → ep93 平台
      │      │      ✓ TDD 红绿重构后, head.py 4 处真正集成 ScaleConditionedRF:
      │      │        1. _forward_diffusion (前向加噪, scales 参数)
      │      │        2. _build_training_targets (从 GT 计算 scales 经 matched_idx 映射)
      │      │        3. predict Euler/Heun 路径 (推理时从 x0_pred 计算 scales)
      │      │        4. _compute_inference_scales (raw→normalized cxcywh→sqrt(w*h))
      │      │      48 单元测试全部通过 (test_nonlinear_trajectory.py)
      │      │      commit: e757b856 feat(scale-conditioned-rf): integrate ...
      │      │      ⛔⛔ **关键结论: ScaleConditionedRF 真正启用后性能下降 (0.741 < 0.746 baseline)**
      │      │        理论缺陷详见 [第十一节: ScaleConditionedRF 证伪记录](#十一scaleconditionedrf-证伪记录)
       │      │
       │      ├─→ E4.3-tune eps=3.0 = 0.750  [+0.004]  ✓ 早停
       │      │      config: experiments/configs/ldmdet/nonlinear_trajectory_e43_eps3.py
       │      │      本地: work_dirs/nonlinear_trajectory_e43_eps3/
       │      │      SwanLab: run_id=5m1lse6x  best mAP=0.7500  (best @ epoch 70)
       │      │
       │      └─→ E4.3 multinomial = 0.748  [+0.002]  ✓ 早停
       │             config: experiments/configs/ldmdet/nonlinear_trajectory_e43_multinomial.py
       │             改动: coupling_mode=argmax→multinomial
       │             本地: work_dirs/nonlinear_trajectory_e43_multinomial/
       │             SwanLab: run_id=l0991c8v  best mAP=0.7480  (best @ epoch 72)
       │
       ├─→ E6-EMA (E4.3 + EMA Hook + weight_decay=5e-4) = 0.739  [-0.007]  ✓ 早停
       │      config: experiments/configs/ldmdet/nonlinear_trajectory_e6_ema.py
       │      本地: work_dirs/nonlinear_trajectory_e6_ema/
       │      SwanLab: run_id=q2168hth  best mAP=0.7390  (best @ epoch 92)
       │
       ├─→ E6-Muon (E4.3 + MuonHybrid 优化器) = 0.744  [-0.002, 持平]  ✓ 早停
       │      config: experiments/configs/ldmdet/nonlinear_trajectory_e6_muon.py
       │      改动: MuonHybridConstructor, batch_size=2, DynamicConv 大矩阵走 AdamW
       │      本地: work_dirs/nonlinear_trajectory_e6_muon/
       │      SwanLab: run_id=640t9s93  best mAP=0.7440  (best @ epoch 68)
       │
       └─→ E7-smax100 (E4.3 + T_max=100, 余弦退火对齐) = 0.745  [-0.001, 持平]  ✓ 自然结束
              config: experiments/configs/ldmdet/nonlinear_trajectory_e7_tmax100.py
              改动: max_epochs 150→100, T_max 150→100 (LR 完全退火基线)
              本地: work_dirs/nonlinear_trajectory_e7_tmax100/
              SwanLab: run_id=pte9vv1a  best mAP=0.7450  (best @ epoch 82)

3-seed 复现实验 (E4.3 eps=2.0 配置, 验证可复现性, SwanLab 项目=nonlinear-3seed-repro)
       config: experiments/configs/ldmdet/nonlinear_trajectory.py (seeds 1,2,3)
       │
       ├─→ seed 1 = 0.746  ✓ 早停
       │      本地: work_dirs/nonlinear_trajectory_seed1/
       │      SwanLab: project=nonlinear-3seed-repro  name=nonlinear_e43_seed1
       │              run_id=k7nnzvuq  best mAP=0.7460  (best @ epoch 72)
       ├─→ seed 2 = 0.749  ✓ 早停
       │      本地: work_dirs/nonlinear_trajectory_seed2/
       │      SwanLab: project=nonlinear-3seed-repro  name=nonlinear_e43_seed2
       │              run_id=clpof6nn  best mAP=0.7490  (best @ epoch 110)
       └─→ seed 3 = ⛔ 失败 (误启动, 仅 2 epoch 即被杀)
              本地: work_dirs/nonlinear_trajectory_seed3/ (无效)
              SwanLab: run_id=kaz1tog7  mAP=0.0000

       初步均值 (seed1+seed2): 0.7475 ± 0.0015, 落在 ±0.018 容差内
       ⚠ seed3 需重跑才能得到完整 3-seed 方差统计

═══════════════════════════════════════════════════════════════════════════
═══ 24_chromosomes_object 数据集 (简称 24obj) ═══
═══════════════════════════════════════════════════════════════════════════
⚠ 不可与 chromo 数据集实验对比! 数据集不同, mAP 量级不同 (0.85+ vs 0.72-0.75)

### 3.2 24_chromosomes_object 数据集 (简称 24obj)

24obj 数据集递进树:
DiffusionDet DDPM (根 baseline, 24obj 数据集)
│  数据集: data/24_chromosomes_object/coco/
│  mAP 量级: 0.85+
│
├─→ ldmdet_rf_heun_adaln_stochot_eps5 (RF+Heun+AdaLN+Sinkhorn Stochastic OT, eps=5)
│      mAP: 0.853
│      本地: work_dirs/ldmdet_rf_heun_adaln_stochot_eps5/20260527_141432/
│      SwanLab: https://swanlab.cn/@einspanner/chromosome-kd-benchmark-24obj/runs/odnz6a8pnda3gyg84cfqv
│              project=chromosome-kd-benchmark-24obj  experiment_name=ldmdet-rf-adaln-stochot-eps5
│              run_id=odnz6a8pnda3gyg84cfqv  mAP=0.8530  (best @ epoch 89, last @ epoch 119)
│
├─→ 24obj_ablation (3 种耦合策略 × 多种子, 2026-06)
│      config: experiments/configs/multiset/chromo_24obj_*.py
│      │
│      ├─→ Random Coupling (3 seeds)
│      │      平均: 0.860 ± 0.001
│      │      ├─ seed_42:  mAP=0.859 (best @ 59)
│      │      │   本地: work_dirs/24obj_ablation/random/seed_42/20260623_084555/
│      │      │   SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/p5xqii8mcqmbhuo5lhlff
│      │      │           experiment_name=chromo_24obj_random_seed42  run_id=p5xqii8mcqmbhuo5lhlff
│      │      ├─ seed_789: mAP=0.860 (best @ 82)
│      │      │   本地: work_dirs/24obj_ablation/random/seed_789/20260625_022750/
│      │      │   SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/r8n441mu4gws43xyoneoj
│      │      │           experiment_name=chromo_24obj_random_seed789  run_id=r8n441mu4gws43xyoneoj
│      │      └─ seed_123: mAP=0.860 (best @ 115)
│      │          本地: work_dirs/24obj_ablation/random/seed_123/20260624_021224/
│      │          SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/q6jgxefgxbp8f2sf5qzpc
│      │                   experiment_name=chromo_24obj_random_seed123  run_id=q6jgxefgxbp8f2sf5qzpc
│      │
│      ├─→ Sinkhorn Stochastic OT (1 seed)
│      │      mAP: 0.856 (best @ 53)
│      │      本地: work_dirs/24obj_ablation/sinkhorn/seed_42/20260626_095551/
│      │      SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/o96m1eqz4l12qjeyys1cs
│      │              experiment_name=chromo_24obj_sinkhorn_seed42  run_id=o96m1eqz4l12qjeyys1cs
│      │
│      └─→ GHSS Coupling (3 seeds)
│             平均: 0.858 ± 0.001
│             ├─ seed_42:  mAP=0.857 (best @ 83)
│             │   本地: work_dirs/24obj_ablation/ghss/seed_42/20260621_020046/
│             │   SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/k84cq9oftbp2nld88a85t
│             │            experiment_name=chromo_24obj_seed42  run_id=k84cq9oftbp2nld88a85t
│             ├─ seed_789: mAP=0.859 (best @ 75)
│             │   本地: work_dirs/24obj_ablation/ghss/seed_789/20260622_144356/
│             │   SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/holadvhaz9v2rh8l494hv
│             │            experiment_name=chromo_24obj_seed789  run_id=holadvhaz9v2rh8l494hv
│             └─ seed_123: mAP=0.859 (best @ 102)
│                 本地: work_dirs/24obj_ablation/ghss/seed_123/20260621_190440/
│                 SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/73cr3uyqw4f1q68xz1evg
│                          experiment_name=chromo_24obj_seed123  run_id=73cr3uyqw4f1q68xz1evg

═══════════════════════════════════════════════════════════════════════════
═══ 跨数据集合并实验 (24obj + Chromosome20240904) ═══
═══════════════════════════════════════════════════════════════════════════

### 3.3 跨数据集合并实验 (24obj + Chromosome20240904)

├─→ merged_ablation (24obj + Chromosome20240904 合并训练)
│      数据集: merged (24_chromosomes_object + Chromosome20240904_NoAug_NoResize)
│      config: experiments/configs/multiset/chromo_merged.py
│      │
│      ├─→ Sinkhorn Stochastic OT (1 seed)
│      │      mAP: 0.806 (best @ 77)
│      │      本地: work_dirs/merged_ablation/sinkhorn/seed_42/20260627_012848/
│      │      SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/mbgo9qcz95l1aik8sv7bu
│      │              experiment_name=chromo_merged_seed42  run_id=mbgo9qcz95l1aik8sv7bu
│      │
│      └─→ GHSS Coupling (1 seed) — FAILED
│             mAP: 0.000 (训练失败, 未产生有效指标)
│             本地: work_dirs/merged_ablation/ghss/seed_42/20260627_012525/
│             SwanLab: https://swanlab.cn/@einspanner/ldmdet-ablation/runs/503pfk8isr270atpubho1
│                     experiment_name=chromo_merged_seed42 (同名冲突)  run_id=503pfk8isr270atpubho1
```

### 3.4 生成式迁移实验 (ChromoGen → LDMDet, chromo 数据集)

```
生成式迁移 Phase1: 用 ChromoGen 生成模型特征增强 LDMDet (Feature Bridge Module, FBM)
基于 rf_heun_adaln + ChromoGen UNet 特征注入, baseline 对照 = 0.746
config: experiments/configs/ldmdet/gen_transfer_phase1_e6_*.py
│
├─→ E6.2 frozen (FBM alpha 可学习 + UNet 全冻结) = 0.737  [-0.009]  ✓ 已停止
│      本地: work_dirs/gen_transfer_phase1_e6_2_frozen/
│      SwanLab: run_id=qyzudrgb  best mAP=0.7370  (best @ epoch 66)
│
├─→ E6.3 enhanced (per-channel gate + GroupNorm + UNet 部分解冻) = 0.703  [-0.043]  ✓ 已停止
│      本地: work_dirs/gen_transfer_phase1_e6_3_enhanced/
│      SwanLab: run_id=m2fyzmf9  best mAP=0.7030  (best @ epoch 19)
│
├─→ E6.3b frozen_enhanced (同 E6.3 但 UNet 全冻结, 隔离 FBM 架构效果) = 0.696  [-0.050]  ✓ 已停止
│      本地: work_dirs/gen_transfer_phase1_e6_3b_frozen_enhanced/
│      SwanLab: run_id=bbe2yrcg  best mAP=0.6960  (best @ epoch 16)
│
└─→ E6.4 crossattn (Cross-Attention FBM + UNet 部分解冻 + zero-init gamma) = 0.733  [-0.013]  ✓ 已停止
       本地: work_dirs/gen_transfer_phase1_e6_4_crossattn/
       SwanLab: run_id=x8j5l7mw  best mAP=0.7330  (best @ epoch 35, 停止 @ epoch 48)
       ⚠ gamma 零初始化导致梯度信号微弱, cross-attention 路径未激活, 退化为 simple gate
       后续改进建议: 非零 gamma 初始化 (如 0.1) 或移除 gamma

结论: FBM 系列均未超越 baseline (0.746), 简单 frozen (E6.2) 优于复杂增强 (E6.3/E6.4)。
```

### 3.5 Few-Shot 跨数据集基准实验 (24obj 源 → chromo 目标)

```
Few-Shot Benchmark: 24obj 数据集源预训练 → chromo 数据集目标微调 (k=5, k=10)
config: experiments/configs/few_shot/source_pretrain/*.py (源预训练)
        experiments/configs/few_shot/target_finetune/*.py (目标微调, 14 个配置已就绪)
baseline 源预训练数据集: 24_chromosomes_object (24obj)
│
├─→ source_pretrain (7 模型, 24obj 数据集)
│      │
│      ├─→ LDMDet SOTA = 已完成 (best @ epoch 26)
│      │      本地: work_dirs/few_shot/source_pretrain_ldmdet_sota_24obj/
│      │      config: experiments/configs/few_shot/source_pretrain/ldmdet_sota_24obj.py
│      │
│      ├─→ LDMDet FBM SimpleGate = 0.677  ⛔ 已停止 (epoch 5)
│      │      本地: work_dirs/few_shot/source_pretrain_ldmdet_fbm_simplgate_24obj/20260701_203410/
│      │      SwanLab: run_id=hyuiam5m  best mAP=0.6770  (best @ epoch 5)
│      │      config: experiments/configs/few_shot/source_pretrain/ldmdet_fbm_simplgate_24obj.py
│      │
│      ├─→ LDMDet FBM CrossAttn = 0.810  🔄 运行中 (epoch 10/150)
│      │      本地: work_dirs/few_shot/source_pretrain_ldmdet_fbm_crossattn_24obj/20260701_174434/
│      │      SwanLab: run_id=9hj8pe4a  best mAP=0.8100  (best @ epoch 10)
│      │      config: experiments/configs/few_shot/source_pretrain/ldmdet_fbm_crossattn_24obj.py
│      │      进展: ep1=0.000 → ep6=0.752 → ep9=0.787 → ep10=0.810 (持续上升)
│      │
│      ├─→ Cascade R-CNN R50 = 已完成 (best @ epoch 72)
│      │      本地: work_dirs/few_shot/source_pretrain_cascade_rcnn_r50_24obj/
│      │      config: projects/LDMDet/configs/benchmark_24obj/cascade_rcnn_r50.py
│      │
│      ├─→ DINO R50 = 已完成 (best @ epoch 102)
│      │      本地: work_dirs/few_shot/source_pretrain_dino_r50_24obj/
│      │      config: projects/LDMDet/configs/benchmark_24obj/dino_r50.py
│      │
│      ├─→ RTMDet-L = 已完成 (best @ epoch 116)
│      │      本地: work_dirs/few_shot/source_pretrain_rtmdet_l_24obj/
│      │      config: projects/LDMDet/configs/benchmark_24obj/rtmdet_l.py
│      │
│      └─→ YOLOX-S = 已完成 (best @ epoch 200)
│             本地: work_dirs/few_shot/source_pretrain_yolox_s_24obj/
│             config: projects/LDMDet/configs/benchmark_24obj/yolox_s.py
│
└─→ target_finetune (k=5, k=10, chromo 数据集)
       config: experiments/configs/few_shot/target_finetune/*_k{5,10}.py (14 个)
       数据: k=5 (112 图, 120 标注), k=10 (222 图, 240 标注)
       状态: ⛔ 尚未启动 (等待 FBM CrossAttn 源预训练完成)

⚠ 5 个非 FBM 实验仅保留 best checkpoint, 无训练日志 (需加载 checkpoint 评估或查 SwanLab)
```

## 四、关键结论 (修正)

### 1. 真正的 Baseline (DiffusionDet 默认 aug)

| Baseline | mAP | 数据源 |
|----------|-----|--------|
| **DiffusionDet DDPM (根)** | **0.729 ± 0.003** | 本地 `work_dirs/multi_seed_aug/ddpm/` + SwanLab `ldmdet-ablation/diffusiondet_ddpm_seed*` |
| **RF+Heun+AdaLN (改进 baseline)** | **0.746 ± 0.001** | 本地 `work_dirs/multi_seed_aug/rf_heun_adaln/` + SwanLab `ldmdet-ablation/rf_heun_adaln_seed*` |
| **历史 SOTA (stochot_eps5_v2)** | **0.753** | ldmdet-experiment `sota/phase5_stochastic_ot/reproduce_0751_stochot_eps5_v2/` |

### 2. 各部件贡献 (基于默认 aug)
| 改进 | ΔmAP | 评价 |
|------|------|------|
| DDPM → RF+Heun+Shifted+AdaLN | **+0.017** | 🟢 主要贡献 |
| + DPM-Solver++ 推理加速 (o2/o3, 6步 NFE对齐) | +0.001 | 🟠 边际收益 (步数对齐 4步时 Δ=+0.000 但 NFE 降 25-37%) |
| + Hard OT Coupling | +0.001 | 🟠 边际收益 |
| + Sinkhorn Stochastic OT | +0.002 | 🟠 边际收益 |
| + SOTA (ot_coupling=True) | +0.003 (高方差) | 🟠 边际但高方差 |
| Bottleneck: focal_gamma_3 | +0.004 | 🟢 分类损失调整 |
| Direction D: BoxRefineNet | +0.001 | 🟠 持平 baseline |
| 非线性轨迹 E4.2 (OT Flow only) | +0.005 | 🟠 OT Flow 耦合有效 |
| 非线性轨迹 E4.3 (OT+SCRF, SCRF 未启用) | +0.006 | 🟠 主要来自 OT,非 ScaleConditionedRF |
| 非线性轨迹 E4.3 eps=2.0 (SCRF 真正启用) | **−0.005** | � **证伪! 0.741 < 0.746 baseline, SCRF 有害** |
| 生成式迁移 E6.2 (FBM frozen) | -0.009 | 🔴 FBM 未超越 baseline |
| 生成式迁移 E6.4 (CrossAttn FBM) | -0.013 | 🔴 gamma 零初始化致失效 |

> 📊 **算法示意图**: [experiment_lineage_schematics.png](figures/experiment_lineage_schematics.png) | [中文版](figures/experiment_lineage_schematics_zh.png)
> 7 个面板 (DDPM → RF+Heun → DPM-Solver++ → Hard OT → Sinkhorn OT → Focal γ=3 → OT Flow) 对应上表每次改进的底层算法可视化; BoxRefineNet 因 ΔmAP≈0 已移除; 生成脚本: `docs/figures/generate_algorithm_schematics.py`

### 3. 之前错误对照 (已修正)
| 错误 | 原因 |
|------|------|
| rf_heun_adaln multi_seed (0.712) | ❌ 简化 aug (无 multi-scale/crop),不是 DiffusionDet 默认 |
| ghss (0.857) | ❌ 24obj 数据集实验,与 chromo 数据集不可对照 |
| E4.3 eps2 OLD (0.752 归因 ScaleConditionedRF) | ❌ ScaleConditionedRF 未集成到 head.py, 0.752 来自 OTFlowCoupling+种子方差 |
| E6.4 CrossAttn (0.733 归因 cross-attention) | ❌ gamma 零初始化致 attention 路径未激活, 实际退化为 simple gate |

## 五、SwanLab 项目映射

| SwanLab Project | 实验数 | 范围 | 数据源 URL Pattern |
|-----------------|--------|------|---------------------|
| `chromosome-kd` | 21 | 早期: sota_seed*, ablation/* (无 aug), scheme_*, stability/* | `https://swanlab.cn/@einspanner/chromosome-kd/runs/<run_id>` |
| `chromosome-kd-benchmark-24obj` | 1 | 24obj 数据集早期: ldmdet-rf-adaln-stochot-eps5 | `https://swanlab.cn/@einspanner/chromosome-kd-benchmark-24obj/runs/<run_id>` |
| `ldmdet-ablation` | 32+ | 主线: chromo 数据集 (multi_seed_aug/*, bottleneck/*, direction_exps/*, nonlinear_trajectory*) + 24obj/merged + gen_transfer | `https://swanlab.cn/@einspanner/ldmdet-ablation/runs/<run_id>` |
| `nonlinear-3seed-repro` | 2 | 3-seed 复现实验: nonlinear_e43_seed{1,2} (seed3 失败) | `https://swanlab.cn/@einspanner/nonlinear-3seed-repro/runs/<run_id>` |

> 用户登录态见 `/home/linkst/.swanlab/.netrc` (api_key 已配置)
> 每个 experiment 的 run_id 见上文递进树,替换 URL 中的 `<run_id>` 即可直接访问

## 六、废弃/错误实验清理方案

### A. 24obj 数据集 + 跨数据集合并实验 (不可与 chromo 数据集主线对比, 已记录于上文递进树)
| 目录 | 大小 | 数据集 | mAP | SwanLab | 清理 |
|------|------|--------|-----|---------|------|
| `work_dirs/24obj_ablation/` | 23G | 24_chromosomes_object | 0.856-0.860 | ldmdet-ablation/chromo_24obj_* (7 runs) | 🗑️ 删 checkpoint (保留 metrics) |
| `work_dirs/merged_ablation/` | 3.3G | 合并 (24obj+chromo) | 0.806 (sinkhorn) / 0.000 (ghss failed) | ldmdet-ablation/chromo_merged_seed42 (2 runs) | 🗑️ 删 checkpoint |
| `work_dirs/ldmdet_rf_heun_adaln_stochot_eps5/` | 1.9G | 24_chromosomes_object | 0.853 | chromosome-kd-benchmark-24obj/ldmdet-rf-adaln-stochot-eps5 | 🗑️ 删 checkpoint |

> 完整数据源 (本地路径 + SwanLab URL + run_id) 见上文 "三、实验递进树" 3.2/3.3 节

### B. 无 aug 旧 baseline (非标准,被 multi_seed_aug 取代)
| 目录 | 大小 | mAP | SwanLab run_id | 清理 |
|------|------|-----|---------------|------|
| `work_dirs/multi_seed/rf_heun_adaln/` | ~5G | 0.712 | rf_heun_adaln_seed* (同名冲突) | 🗑️ 删 |
| `work_dirs/multi_seed/rf_heun_adaln_bs4/` | ~5G | 0.702 | - | 🗑️ 删 |
| `work_dirs/multi_seed/hard_ot/` | ~5G | 0.705 | hard_ot_seed789 | 🗑️ 删 |

### C. 旧 ablation (无 aug, 被 multi_seed_aug 取代)
| 目录 | 大小 | mAP | SwanLab run_id | 清理 |
|------|------|-----|---------------|------|
| `work_dirs/ablation/adaln*` `stochot*` | 13G | 0.720-0.738 | adaln, adaln-stochot-eps*, stochot-eps* | 🗑️ 删 |

### D. 失败/调试实验
| 目录 | 大小 | mAP | SwanLab | 清理 |
|------|------|-----|---------|------|
| `work_dirs/debug_train_final/` | 1.9G | 0.525 | - | 🗑️ 删 |
| `work_dirs/debug_train_timm/` | 8.1M | - | - | 🗑️ 删 |
| `work_dirs/optim_test_v2/` | 1.8G | - | - | 🗑️ 删 |
| `work_dirs/bottleneck/ablation/high_giou_weight/20260624_234713/` | - | 0.523 | - | 🗑️ 删 (failed) |
| `work_dirs/bottleneck/ablation/no_box_renewal/20260623_183224/` | - | 0.635 | - | 🗑️ 删 (failed early) |

### E. 被取代的方向实验
| 目录 | 大小 | mAP | SwanLab run_id | 清理 |
|------|------|-----|---------------|------|
| `work_dirs/scheme_a_dinov2_s/` | 3.9G | 0.502-0.676 | dinov2_s-rf-heun-adaln-stochot | 🗑️ 删 (DINOv2 全失败) |
| `work_dirs/scheme_E_bifpn/` | 1.9G | 0.744 | arch_E_bifpn | 🗑️ 删 (BiFPN 未采用) |
| `work_dirs/stability/warm_restart_*` | 6.7G | 0.714-0.728 | bs8-warm-restart-* | 🗑️ 删 |
| `work_dirs/cspnext_l_rf_heun_adaln_stochot/` | 2.2G | 0.730 | cspnext-l-rf-heun-adaln-stochot | 🗑️ 删 |

### F. 冗余 checkpoint (保留 best, 删除中间 epoch)
| 目录 | 大小 | 问题 | 清理 |
|------|------|------|------|
| `work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v3/` | **156G** | 保留所有 epoch_*.pth | 🗑️ 删非 best |
| `work_dirs/ldmdet_rf_heun_shifted_bs8_aug_v2/` | **96G** | 同上 | 🗑️ 删非 best |
| `work_dirs/chromogen_phase1/` | **103G** | 生成模型 checkpoint | ⚠️ 确认后清 |

### 清理预估空间释放
- A. 24obj 数据集 + 跨数据集合并: ~28G
- B. 无 aug 旧 baseline: ~15G
- C. 旧 ablation: ~13G
- D. 失败实验: ~4G
- E. 被取代方向: ~15G
- F. 冗余 checkpoint: **~250G+**
- **总计可释放: ~325G**

## 七、保留的核心实验 (不可清理)

| 目录 | 说明 | 重要性 |
|------|------|--------|
| `work_dirs/multi_seed_aug/ddpm/` | DDPM 根 baseline (默认 aug) | ⭐⭐⭐ |
| `work_dirs/multi_seed_aug/rf_heun_adaln/` | RF 改进 baseline (默认 aug) | ⭐⭐⭐ |
| `work_dirs/multi_seed_aug/hard_ot/` | OT 消融 | ⭐⭐ |
| `work_dirs/multi_seed_aug/sinkhorn_stochastic/` | OT 消融 | ⭐⭐ |
| `work_dirs/sota_seed*/` | SOTA 多种子 | ⭐⭐⭐ |
| `work_dirs/bottleneck/` | 瓶颈分析 (保留 best + report) | ⭐⭐⭐ |
| `work_dirs/direction_exps/` | 方向实验 (D=0.747, B=0.702, 其余废弃) | ⭐⭐ |
| `work_dirs/nonlinear_trajectory*/` | 非线性轨迹系列 (E4.x, E6, E7, 3-seed) | ⭐⭐⭐ |
| `work_dirs/nonlinear_trajectory_e43_eps2_real/` | 首次真正启用 ScaleConditionedRF (运行中) | ⭐⭐⭐ |
| `work_dirs/gen_transfer_phase1_e6_*/` | 生成式迁移 FBM 实验 (E6.2-E6.4) | ⭐⭐ |
| `work_dirs/few_shot/` | Few-Shot 基准 (源预训练 + 待启动微调) | ⭐⭐⭐ |
| `ldmdet-experiment/sota/` | 归档 SOTA (含完整代码备份) | ⭐⭐⭐ |

## 八、Direction 实验的正确对照 (修正)

Direction 实验基于 `rf_heun_adaln.py` (chromo 数据集, 默认 aug):
- ✅ 正确对照: multi_seed_aug/rf_heun_adaln = **0.746**
- ❌ 错误对照: multi_seed/rf_heun_adaln = 0.712 (简化 aug)
- ❌ 错误对照: ghss@24obj 数据集 = 0.857 (跨数据集不可对照)

| 方向 | mAP | Δ vs 0.746 | 价值 |
|------|-----|-----------|------|
| D (BoxRefineNet) | 0.747 | +0.001 | 🟠 持平 baseline,early stop @ epoch 85,best @ epoch 55 |
| B (DecoupledHead) | 0.702 | -0.044 | 🔴 显著低于 baseline,已停止 |
| C/F/A/E | — | — | ⛔ 未启动,已废弃 |

> Direction 系列已停止推进,后续由非线性轨迹实验 (见 3.4 节) 接替。
> 若要追求 SOTA,应将有效方向叠加到 SOTA config (含 OT) 上。

## 九、每步改进详解 (结构化)

> 本节以标准化格式梳理每个关键改进:动机 / 改动 / 实验 / 结论。
> 配套图见 [第十节:图索引](#十图索引)。

### 9.1 DDPM → Rectified Flow + Heun + Shifted + AdaLN-Zero

| 字段 | 内容 |
|------|------|
| **动机** | DDPM 采样需 1000 步,推理慢;标准 RF 线性轨迹可一步直达但精度受限 |
| **改动** | `diffusion_type=rectified_flow`, `solver_type=heun` (二阶), `rf_schedule=shifted` (rf_shift=3.0), `time_conditioning=adaln_zero` (零初始化保证训练稳定) |
| **实验** | config: `experiments/configs/ldmdet/rf_heun_adaln.py`; 3 seeds; work_dir: `work_dirs/multi_seed_aug/rf_heun_adaln/` |
| **结果** | 0.729 → **0.746** (+0.017), 3-seed 方差 ±0.001 |
| **结论** | 🟢 **主要贡献**,后续所有实验均基于此 baseline |
| **关键代码** | [head.py](file:///home/linkst/workplace/chromo/chromosome-kd/ldmdet/core/head.py) DiffusionDetHead, [rectified_flow.py](file:///home/linkst/workplace/chromo/chromosome-kd/ldmdet/diffusion/rectified_flow.py), [single_head.py](file:///home/linkst/workplace/chromo/chromosome-kd/ldmdet/core/single_head.py) AdaLN-Zero |
| **图** | [fig1_overall_architecture.png](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig1_overall_architecture.png) — 整体架构 |

### 9.2 Sinkhorn OT 耦合 (替代随机配对)

| 字段 | 内容 |
|------|------|
| **动机** | 随机耦合将 500 个噪声 proposal 与 M 个 GT 随机配对,引入不必要的传输成本;OT 配对可找到最小传输成本的匹配 |
| **改动** | `coupling=OTFlowCoupling`, Sinkhorn 算法 (log-domain, 10 iters), cost=cdist L2, 列归一化后 argmax 解码 |
| **实验** | E4.2 (OT only) vs baseline; work_dir: `work_dirs/nonlinear_trajectory_e42/` |
| **结果** | 0.746 → **0.751** (+0.005); E4.3 (OT+SCRF未启用) = 0.752 |
| **结论** | 🟠 **边际有效**,OT 耦合降低传输成本,但提升幅度有限 (架构天花板 ~0.75) |
| **关键代码** | [ot_flow_coupling.py](file:///home/linkst/workplace/chromo/chromosome-kd/ldmdet/coupling/ot_flow_coupling.py), [_sinkhorn_ops.py](file:///home/linkst/workplace/chromo/chromosome-kd/ldmdet/coupling/_sinkhorn_ops.py) |
| **图** | [fig3_ot_coupling.png](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig3_ot_coupling.png) — OT 配对示意 |

### 9.3 Scale-Conditioned RF (非线性轨迹)

| 字段 | 内容 |
|------|------|
| **动机** | 标准 RF 对所有尺度目标用相同时间调度 κ=1;小目标 (w·h 小) 信噪比低,应在更早的 t 去噪 |
| **改动** | `ScaleConditionedRF`: κ(s)=1+λ(s_max−s)/s_max, t_eff=t^(1/κ(s)); λ=0.5, s_max=0.15; 训练从 GT 算 s,推理从 x0_pred 算 s |
| **实验** | E4.3-tune eps=2.0 (NEW, 真正启用); work_dir: `work_dirs/nonlinear_trajectory_e43_eps2_real/` |
| **状态** | 🔄 运行中 (Epoch 11/150, ep10 mAP=0.564, 持续上升) |
| **历史教训** | E4.3 OLD (0.752) 的 ScaleConditionedRF 未集成到 head.py,0.752 全部来自 OT+种子方差 |
| **TDD** | 48 单元测试覆盖: heun_step 形状/退化等价/Euler 差异, head 4 处集成, 3 种向后兼容 |
| **关键代码** | [scale_conditioned_rf.py](file:///home/linkst/workplace/chromo/chromosome-kd/ldmdet/diffusion/scale_conditioned_rf.py), [head.py L375-410](file:///home/linkst/workplace/chromo/chromosome-kd/ldmdet/core/head.py#L375-L410) |
| **图** | [fig2_scale_conditioned_rf.png](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig2_scale_conditioned_rf.png) — 模块图, [fig4_trajectory_comparison.png](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig4_trajectory_comparison.png) — 轨迹对比 |

### 9.4 Bottleneck: Focal Loss γ=3

| 字段 | 内容 |
|------|------|
| **动机** | 瓶颈分析显示分类损失是主要瓶颈,默认 γ=2 对困难样本权重不足 |
| **改动** | `focal_gamma=3.0` |
| **实验** | work_dir: `work_dirs/bottleneck/ablation/focal_gamma_3/` |
| **结果** | 0.746 → **0.750** (+0.004) |
| **结论** | 🟢 分类损失调整有效,但未叠加到 SOTA |

### 9.5 Direction D: BoxRefineNet

| 字段 | 内容 |
|------|------|
| **动机** | 在 head 输出后增加框精修网络,二次校准边界框 |
| **改动** | 新增 BoxRefineNet 模块 |
| **实验** | work_dir: `work_dirs/direction_exps/direction_d_box_refine/` |
| **结果** | 0.746 → **0.747** (+0.001, 持平) |
| **结论** | 🟠 持平 baseline,精修网络未带来显著提升 |

### 9.6 Direction B: DecoupledHead (失败)

| 字段 | 内容 |
|------|------|
| **动机** | 解耦分类与回归路径,减少多任务干扰 |
| **改动** | DecoupledHead 替代共享特征路径 |
| **实验** | work_dir: `work_dirs/direction_exps/direction_b_decoupled_head/` |
| **结果** | 0.746 → **0.702** (−0.044, 显著退化) |
| **结论** | 🔴 解耦导致特征共享信息丢失,已停止 |

### 9.7 生成式迁移 FBM (E6.2-E6.4, 均失败)

| 字段 | 内容 |
|------|------|
| **动机** | 用 ChromoGen 生成模型的特征通过 Feature Bridge Module (FBM) 增强 LDMDet 检测器 |
| **改动** | E6.2: frozen + 可学习 alpha; E6.3: per-channel gate + UNet 部分解冻; E6.4: cross-attention + zero-init gamma |
| **实验** | work_dirs: `work_dirs/gen_transfer_phase1_e6_*/` |
| **结果** | E6.2=0.737 (−0.009), E6.3=0.703 (−0.043), E6.4=0.733 (−0.013) |
| **结论** | 🔴 FBM 系列均未超越 baseline; E6.4 的 gamma 零初始化致 attention 路径未激活,实际退化为 simple gate |

## 十、图索引

| 图 | 文件 | 说明 | 脚本 |
|----|------|------|------|
| 图1 | [fig1_overall_architecture](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig1_overall_architecture.pdf) | 整体架构: backbone → FPN → 6 级联 head → 训练/推理双路径 | [fig1_overall_architecture.py](file:///home/linkst/workplace/chromo/chromosome-kd/scripts/figs/fig1_overall_architecture.py) |
| 图2 | [fig2_scale_conditioned_rf](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig2_scale_conditioned_rf.pdf) | ScaleConditionedRF 模块: κ(s) 调制 + t_eff 变换 + 训练/推理路径 | [fig2_scale_conditioned_rf.py](file:///home/linkst/workplace/chromo/chromosome-kd/scripts/figs/fig2_scale_conditioned_rf.py) |
| 图3 | [fig3_ot_coupling](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig3_ot_coupling.pdf) | OT Flow Coupling: 噪声-GT 空间分布 + Sinkhorn 传输矩阵 + 配对结果 | [fig3_ot_coupling.py](file:///home/linkst/workplace/chromo/chromosome-kd/scripts/figs/fig3_ot_coupling.py) |
| 图4 | [fig4_trajectory_comparison](file:///home/linkst/workplace/chromo/chromosome-kd/docs/figs/fig4_trajectory_comparison.pdf) | 轨迹对比: t_eff vs t 曲线 + α(t,s) 权重 + 2D 轨迹位置 | [fig4_trajectory_comparison.py](file:///home/linkst/workplace/chromo/chromosome-kd/scripts/figs/fig4_trajectory_comparison.py) |

> 画图公共样式: [_style.py](file:///home/linkst/workplace/chromo/chromosome-kd/scripts/figs/_style.py) (Liberation Serif / CVPR 风格 / 300dpi PDF+PNG)
>
> 重新生成所有图: `for f in scripts/figs/fig*.py; do python "$f"; done`

## 十一、ScaleConditionedRF 证伪记录

> 本节记录 ScaleConditionedRF 方向的完整证伪过程:从"声称 0.752"到"真正集成后 0.741 证伪",以及 0.753/0.752 的真实改进谱系。

### 11.1 0.753 与 0.752 的真实改进谱系

经对归档代码、备份 head.py、dumped config 的全面核查,所有 0.753/0.752 实验的改进来源如下:

| 实验 | mAP | Coupling | ε | 架构 | bs | SCRF 配置 | SCRF 实际生效 | 真实改进来源 |
|------|-----|----------|---|------|----|-----------|--------------|-------------|
| reproduce_0751_stochot_eps5_v2 | **0.753** | sinkhorn_stochastic | 5.0 | PurePyTorch | 2 | 无 | — | Stochastic OT ε=5 |
| scheme_C1_5_mixed_rel_l1_lam015 | **0.752** | sinkhorn_stochastic | 5.0 | PurePyTorch | 2 | 无 | — | + mixed_relative_l1 损失 |
| nonlinear_trajectory E4.3 (usnvd63f) | **0.752** | ot_flow (argmax) | 1.0 | mmdet_bridge | 4 | 有 | ❌ **未集成** | ot_flow coupling |
| nonlinear_trajectory_e43_eps2 OLD (cdtmijl0) | **0.752** | ot_flow (argmax) | 2.0 | mmdet_bridge | 4 | 有 | ❌ **未集成** | ot_flow coupling ε=2.0 |
| **nonlinear_trajectory_e43_eps2 NEW** (hkn0fc7w) | **0.741** | ot_flow (argmax) | 2.0 | mmdet_bridge | 4 | 有 | ✅ **真正集成** | SCRF 导致 −0.005 |

**谱系图**:
```
0.746 baseline (random coupling, RF+Heun+AdaLN-Zero)
   │
   ├──→ 0.753  Stochastic OT ε=5  (sinkhorn_stochastic, PurePyTorch, bs=2)
   │    │      ← 历史最高, 改进来源: Stochastic OT (从传输矩阵采样而非 argmax)
   │    │
   │    └──→ 0.752  + mixed_relative_l1 λ=0.15  (phase7_loss, 损失优化未超越 0.753)
   │
   ├──→ 0.752  ot_flow ε=1.0 argmax  (E4.3, mmdet_bridge, bs=4)
   │    │      ← 改进来源: ot_flow coupling, **非 ScaleConditionedRF** (config 有但未集成)
   │    │
   │    ├──→ 0.752  ot_flow ε=2.0 argmax  (E4.3 eps2 OLD, 同上, ε 调优)
   │    │
   │    └──→ 0.741  ot_flow ε=2.0 + SCRF 真正集成  (E4.3 eps2 NEW)
   │             ← ⛔ 证伪! SCRF 导致性能下降 0.005
   │
   └──→ 0.746-0.749  3-seed 复现  (nonlinear_trajectory.py, SCRF 未集成)
            ← 证实 0.752 的高方差, 均值 0.7475 ± 0.0015
```

**核心结论**: 0.753 和 0.752 的改进 **全部来自 OT coupling 策略** (stochastic sinkhorn 或 ot_flow),ScaleConditionedRF 从未生效。真正集成后反而有害。

### 11.2 证伪实验详情

| 字段 | 内容 |
|------|------|
| **实验** | nonlinear_trajectory_e43_eps2_real (commit e757b856) |
| **config** | `experiments/configs/ldmdet/nonlinear_trajectory_e43_eps2.py` |
| **work_dir** | `work_dirs/nonlinear_trajectory_e43_eps2_real/` |
| **SCRF 集成** | ✅ head.py 4 处: `_forward_diffusion` / `_build_training_targets` / `predict` (Euler+Heun) / `_compute_inference_scales` |
| **测试** | 48 单元测试全部通过 (TDD 红绿重构) |
| **Best mAP** | **0.741** @ Epoch 69 |
| **mAP 曲线** | ep1=0.000 → ep10=0.564 → ep55=0.737 → ep69=**0.741** → ep93 平台化 (0.69-0.74 振荡) |
| **Δ vs baseline** | **−0.005** (0.741 < 0.746) |
| **Δ vs OLD** | **−0.011** (0.741 < 0.752, OLD 的 SCRF 未生效) |
| **结论** | ⛔ **ScaleConditionedRF 方向证伪,应终止** |

### 11.3 理论缺陷分析

ScaleConditionedRF 真正启用后性能下降,根本原因有四:

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

### 11.4 历史归因修正记录

| 时间 | 旧认知 | 新认知 | 证据 |
|------|--------|--------|------|
| 2026-06-24 | E4.3 (0.752) 归功于 ScaleConditionedRF + OT | 0.752 全部来自 OTFlowCoupling | 备份 head.py 确认 SCRF 未集成, 仅用于诊断回调 |
| 2026-06-27 | E4.3 eps2 (0.752) 进一步验证 SCRF 有效 | 同上, SCRF 仍未集成 | 备份 head.py 确认 |
| 2026-07-02 | — | SCRF 真正集成后 0.741 < 0.746, **证伪** | TDD 集成 + 真正训练实验 |

### 11.5 最终结论

1. **ScaleConditionedRF 方向证伪**: 真正集成后性能下降 0.005, 不应继续推进
2. **0.753 是历史最高**: 来自 Stochastic OT ε=5 (sinkhorn_stochastic, 从传输矩阵采样)
3. **0.752 的所有变体**: 均来自 OT coupling (stochastic sinkhorn 或 ot_flow argmax), 与 SCRF 无关
4. **OT coupling 是唯一有效改进**: 但提升幅度有限 (+0.005~0.007), 架构天花板约 0.75
5. **代码保留**: SCRF 代码和测试保留在仓库中 (commit e757b856), 作为证伪记录供后续研究参考
