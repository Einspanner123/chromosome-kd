# KaryoFlow双数据集实验—配置—证据总账

> **已弃用：** 本文档是 2026-08-14 v1 历史快照，包含旧配置路径和已经失效的运行状态。
> 当前唯一权威入口为 `experiments/configs/v2/manifests/paper_experiment_route_matrix.yaml`，
> 可读版本为 `docs/experiments/PAPER_EXPERIMENT_ROUTE_MATRIX.md`。请勿再从本文档启动实验或校对论文数字。

> 本文档仅保留历史审计用途；其中数值证据仍可通过 `tools/experiment_db/experiments.db` 追溯。

## 技术摘要

- 权威清单 SHA-256：`db8372d26cd958fe2d3de3beab956aa85b4ac37ed32bef2b750b6057b62bfce4`。
- 纯自建数据集当前有 18 个实验族、展开 122 个训练/推理/分析运行；正式 KaryoFlow 三训练seed正在运行。
- Dataset 2 当前有 27 个实验族、展开 138 个运行。KaryoFlow/LQCR已有三独立训练配对；五个传统基线仍主要是单训练权重点估计。
- 当前主模型队列不等于完整消融。严格消融还包括 G0→G1→G2→G3 的三seed训练、同checkpoint solver/steps、Top-K/renewal、LQCR验证选择、困难子集、蒸馏和统一A6000效率复评。

## 数据集与统一协议

| Dataset ID | Train / Val / Test | Annotation SHA | 用途与限制 |
|---|---:|---|---|
| `D1_INHOUSE1700_V1` | 1190 / 170 / 340 | train `318120af…`; val `57466f1f…`; test `52e8868d…` | 纯自建；1350个stem组；group-disjoint 70/10/20；无patient ID |
| `D2` | 3500 / 500 / 1000 | train `218ae0…`; val `bcf0f9…`; test `110fd280…` | 公开台中队列；与D1I同为70/10/20 |
| `D1_COMPOSITE2200_LEGACY` | 1540 / 440 / 220 | 历史哈希见数据库 | 含500张D2派生图；仅保留历史审计，不与新双数据集主实验混合 |

统一精度协议：训练seed为独立初始化单位；validation seed固定42；每个训练run按validation规则选一个checkpoint；test inference seed固定42；必须记录mAP/AP50/AP75/AP_S/AP_M/AP_L。效率仅在独占Ross A6000、batch=1、固定precision/warm-up/timed iterations/计时边界下比较。

## 状态与证据类型

- `COMPLETED_VERIFIED`：三独立训练或配对证据已验证，可进入主表。
- `COMPLETED_FIXED_CHECKPOINT`：只证明给定权重下的推理行为，不能称训练复现。
- `COMPLETED_POINT_ESTIMATE`：一个训练checkpoint，仅能报点估计。
- `COMPLETED_DESCRIPTIVE/DIAGNOSTIC`：历史或机制证据，不用于主SOTA因果结论。
- `PLANNED/BLOCKED_*`：尚未获得合格最终证据。

## 纯自建数据集：完整模型、消融与部署计划

### Detector comparison

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D1I.SOTA.karyoflow` | karyoflow | full_train / 3 | train=42,123,789; infer=42 | **RUNNING** | `experiments/configs/self1700/karyoflow.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/karyoflow/trainseed_{training_seed}` | `d1-inhouse1700-v1__karyoflow__trainseed-42__a752e7be1f0d;d1-inhouse1700-v1__karyoflow__trainseed-123__a752e7be1f0d;d1-inhouse1700-v1__karyoflow__trainseed-789__a752e7be1f0d` |
| `D1I.SOTA.diffusiondet` | diffusiondet | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/diffusiondet.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/diffusiondet/trainseed_{training_seed}` | `d1-inhouse1700-v1__diffusiondet__trainseed-42__0362fd3e571b;d1-inhouse1700-v1__diffusiondet__trainseed-123__0362fd3e571b;d1-inhouse1700-v1__diffusiondet__trainseed-789__0362fd3e571b` |
| `D1I.SOTA.dino_r50` | dino_r50 | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/dino_r50.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/dino_r50/trainseed_{training_seed}` | `d1-inhouse1700-v1__dino_r50__trainseed-42__02519716b443;d1-inhouse1700-v1__dino_r50__trainseed-123__02519716b443;d1-inhouse1700-v1__dino_r50__trainseed-789__02519716b443` |
| `D1I.SOTA.rtmdet_l` | rtmdet_l | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/rtmdet_l.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/rtmdet_l/trainseed_{training_seed}` | `d1-inhouse1700-v1__rtmdet_l__trainseed-42__c65dc5e0e455;d1-inhouse1700-v1__rtmdet_l__trainseed-123__c65dc5e0e455;d1-inhouse1700-v1__rtmdet_l__trainseed-789__c65dc5e0e455` |
| `D1I.SOTA.cascade_rcnn_r50` | cascade_rcnn_r50 | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/cascade_rcnn_r50.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/cascade_rcnn_r50/trainseed_{training_seed}` | `d1-inhouse1700-v1__cascade_rcnn_r50__trainseed-42__ff33c17c2649;d1-inhouse1700-v1__cascade_rcnn_r50__trainseed-123__ff33c17c2649;d1-inhouse1700-v1__cascade_rcnn_r50__trainseed-789__ff33c17c2649` |
| `D1I.SOTA.yolox_s` | yolox_s | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/yolox_s.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/yolox_s/trainseed_{training_seed}` | `d1-inhouse1700-v1__yolox_s__trainseed-42__df342a7930d9;d1-inhouse1700-v1__yolox_s__trainseed-123__df342a7930d9;d1-inhouse1700-v1__yolox_s__trainseed-789__df342a7930d9` |

- `D1I.SOTA.karyoflow` 参数：RF; shifted t; AdaLN-Zero; DPM++ 4-step; K=500; 6 heads; batch=2 验收：3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics 论文用途：Main two-cohort detector table 备注：Live registry: seed42=running@ross:a6000:0, run_id=n8hxeoo6pcyy0ivbdejbb; seed123=running@workstation:a5000:0, run_id=kgkw6702; seed789=running@workstation:a4000:1, run_id=wf92rzup
- `D1I.SOTA.diffusiondet` 参数：DDPM; DDIM 1-step; linear t; scale-shift; K=500; 6 heads; batch=4 (external baseline) 验收：3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics 论文用途：Main two-cohort detector table 备注：Live registry: seed42=planned@workstation:a5000:0, run_id=pending; seed123=planned@workstation:a4000:1, run_id=pending; seed789=planned@ross:a6000:0, run_id=pending
- `D1I.SOTA.dino_r50` 参数：R50; 900 queries; 6 encoder/decoder layers; batch=2 验收：3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics 论文用途：Main two-cohort detector table 备注：Live registry: seed42=planned@workstation:a4000:1, run_id=pending; seed123=planned@ross:a6000:0, run_id=pending; seed789=planned@workstation:a5000:0, run_id=pending
- `D1I.SOTA.rtmdet_l` 参数：RTMDet-L; inherited benchmark optimizer; batch=2 验收：3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics 论文用途：Main two-cohort detector table 备注：Live registry: seed42=planned@ross:a6000:0, run_id=pending; seed123=planned@workstation:a5000:0, run_id=pending; seed789=planned@workstation:a4000:1, run_id=pending
- `D1I.SOTA.cascade_rcnn_r50` 参数：Cascade R-CNN R50; 3 stages; batch=4 验收：3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics 论文用途：Main two-cohort detector table 备注：Live registry: seed42=planned@workstation:a5000:0, run_id=pending; seed123=planned@workstation:a4000:1, run_id=pending; seed789=planned@ross:a6000:0, run_id=pending
- `D1I.SOTA.yolox_s` 参数：YOLOX-S; 300 epochs; batch=8 验收：3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics 论文用途：Main two-cohort detector table 备注：Live registry: seed42=planned@workstation:a4000:1, run_id=pending; seed123=planned@ross:a6000:0, run_id=pending; seed789=planned@workstation:a5000:0, run_id=pending

### Decision

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D1I.SOTA.karyoflow_lqcr` | karyoflow_lqcr | short_train / 3 | train=42,123,789; infer=42 | **BLOCKED_PARENT** | `experiments/configs/self1700/lqcr.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/karyoflow_lqcr/trainseed_{training_seed}` | `d1-inhouse1700-v1__karyoflow_lqcr__trainseed-42__7266ebe11ac0;d1-inhouse1700-v1__karyoflow_lqcr__trainseed-123__7266ebe11ac0;d1-inhouse1700-v1__karyoflow_lqcr__trainseed-789__7266ebe11ac0` |
| `D1I.DEC.beta_val` | LQCR_beta_0_0.25_0.5_1_2 | inference / 15 | train=42,123,789; infer=42 | **BLOCKED_PARENT** | `tools/experiment_db/test_inference_ablations.py` (parameterize_for_d1i) | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/self1700/lqcr_beta_val/{parent}/beta_{beta}` | `lqcr_beta_d1i_val` |
| `D1I.DEC.strict_subsets` | AP90_AP95_scale_overlap_quality | analysis / 1 | train=42,123,789; infer=42 | **BLOCKED_PREDICTIONS** | `tools/experiment_db/analysis/difficult_subset_analysis.py` (adapt_from_d2) | CPU after test predictions | `results/self1700/analysis/lqcr_difficult_subsets` | `conditional_difficult_subset_d1i` |

- `D1I.SOTA.karyoflow_lqcr` 参数：parent-matched final-only quality head; beta=2 selected on validation; 12 epochs 验收：parent checkpoint identity; 590 shared tensors unchanged; paired test delta 论文用途：Independent decision contribution 备注：Live registry: seed42=planned@ross:a6000:0, run_id=pending; seed123=planned@workstation:a5000:0, run_id=pending; seed789=planned@workstation:a4000:1, run_id=pending
- `D1I.DEC.beta_val` 参数：3 LQCR parents x beta={0,0.25,0.5,1,2}; validation only 验收：select beta before any new test evaluation 论文用途：Validation selection; not a test claim
- `D1I.DEC.strict_subsets` 参数：AP90/AP95; AP_S/M/L; overlap strata; size quartiles; quality-IoU Spearman; paired image bootstrap 验收：image-level pairing; diagnostics not paper SOTA rows 论文用途：Mechanism evidence and limitation

### Generation

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D1I.ABL.G0` | DDPM_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/ablations/g0_ddpm_batch2.py` (to_create) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/ablations/g0/trainseed_{training_seed}` | `generation_train_ablation_d1i_test` |
| `D1I.ABL.G1` | RF_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/ablations/g1_rf_linear_scaleshift.py` (to_create) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/ablations/g1/trainseed_{training_seed}` | `generation_train_ablation_d1i_test` |
| `D1I.ABL.G2` | RF_shifted_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/ablations/g2_rf_shifted_scaleshift.py` (to_create) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/ablations/g2/trainseed_{training_seed}` | `generation_train_ablation_d1i_test` |
| `D1I.ABL.G3` | RF_shifted_AdaLNZero | full_train / 3 | train=42,123,789; infer=42 | **RUNNING_REUSED** | `experiments/configs/self1700/karyoflow.py` (exists) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/ablations/g3/trainseed_{training_seed}` | `generation_train_ablation_d1i_test` |
| `D1I.INF.solver_steps` | Euler_Heun_DPMpp_x_steps1to4 | inference / 36 | train=42,123,789; infer=42 | **PLANNED** | `tools/experiment_db/test_inference_ablations.py` (parameterize_for_d1i) | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/self1700/inference/solver/{parent}/{solver}_{steps}` | `solver_d1i_test` |
| `D1I.INF.topk_renewal` | TopK_x_renewal | inference / 30 | train=42,123,789; infer=42 | **PLANNED** | `tools/experiment_db/test_inference_ablations.py` (parameterize_for_d1i) | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/self1700/inference/topk_renewal/{parent}/k{K}_{renewal}` | `topk_renewal_d1i_test` |

- `D1I.ABL.G0` 参数：DDPM; DDIM1; linear t; scale-shift; batch=2 验收：only named factor changes; batch/optimizer/augmentation/selection fixed 论文用途：Strict incremental generation ablation
- `D1I.ABL.G1` 参数：RF; Euler/locked validation protocol; linear t; scale-shift; batch=2 验收：only named factor changes; batch/optimizer/augmentation/selection fixed 论文用途：Strict incremental generation ablation
- `D1I.ABL.G2` 参数：RF; shifted t (shift=3); scale-shift; batch=2 验收：only named factor changes; batch/optimizer/augmentation/selection fixed 论文用途：Strict incremental generation ablation
- `D1I.ABL.G3` 参数：RF; shifted t; AdaLN-Zero; DPM++4 validation; batch=2 验收：only named factor changes; batch/optimizer/augmentation/selection fixed 论文用途：Strict incremental generation ablation
- `D1I.INF.solver_steps` 参数：3 parents x {Euler,Heun,DPM++} x {1,2,3,4}; report steps and NFE 验收：same checkpoint per parent; common test; no cross-checkpoint causal claim 论文用途：Solver/step accuracy-efficiency ablation
- `D1I.INF.topk_renewal` 参数：3 parents x K={100,150,200,300,500} x renewal={off,on} 验收：same checkpoint/seed; renewal-off candidate identity audit 论文用途：Identity/renewal and candidate-budget analysis

### Deployment

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D1I.DEP.distill_h3` | H6_teacher_to_H3_student | short_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/deployment/h3_distill.py` (to_create) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/self1700/deployment/h3/trainseed_{training_seed}` | `head_distillation_d1i_test` |
| `D1I.DEP.GACS` | GACS | inference / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/self1700/deployment/gacs.py` (to_create_or_port) | accuracy:any; latency=ross:A6000:0 | `results/self1700/deployment/gacs/{parent}` | `gacs_d1i_test` |
| `D1I.DEP.speed` | all_operating_points | benchmark / 1 | train=42,123,789; infer=none | **BLOCKED_CHECKPOINTS** | `tools/benchmark_fps.py` (exists_requires_protocol_wrapper) | ross:A6000:0 exclusive | `results/self1700/benchmark/a6000/{variant}` | `speed_accuracy_d1i` |

- `D1I.DEP.distill_h3` 参数：teacher heads=6; student heads=3; mapping 0<-0,1<-2,2<-5; fixed loss/protocol 验收：three parent-matched students; test mAP; A6000 latency only 论文用途：Deployment extension
- `D1I.DEP.GACS` 参数：prespecified dynamic policy; accuracy and latency; no precision-module claim 验收：held-out test on both cohorts; same A6000 timing protocol 论文用途：Optional dynamic deployment extension
- `D1I.DEP.speed` 参数：batch=1; same input; precision; warmup; timed iterations; generation/teacher/student/LQCR/GACS 验收：exclusive A6000; no foreign process; accuracy linked to verified test record 论文用途：Speed-accuracy figure/table

## Dataset 2：已完成证据与严格补跑缺口

### Detector comparison

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D2.SOTA.karyoflow_train3` | karyoflow | full_train / 3 | train=335778785,790448076,1342286018; infer=42 | **COMPLETED_VERIFIED** | `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py` (exists_plus_dumped_configs) | historical ross/workstation | `work_dirs/{a4_dpm_pp_24obj|multi_seed/a4_dpm_pp_24obj/seed_*}` | `paired_lqcr_d2_train3;sota_d2_test; artifact=d2-paired-lqcr-train3-4152cdbd5100` |
| `D2.SOTA.diffusiondet.existing` | diffusiondet | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | `experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py` (exists) | historical | `work_dirs/baselines/diffusiondet_24obj` | `sota_d2_test; artifact=d2-test-diffusiondet-b819d77e81b3` |
| `D2.SOTA.diffusiondet.missing_train2` | diffusiondet | full_train / 2 | train=123,789; infer=42 | **PLANNED** | `experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py` (exists) | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/d2_train3_completion/diffusiondet/trainseed_{training_seed}` | `sota_d2_train3_completion` |
| `D2.SOTA.dino_r50.existing` | dino_r50 | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | `experiments/configs/baselines/benchmark_24obj/dino_r50.py` (exists) | historical | `work_dirs/baselines/dino_r50_24obj` | `sota_d2_test; artifact=d2-test-dino_r50-789860257a37` |
| `D2.SOTA.dino_r50.missing_train2` | dino_r50 | full_train / 2 | train=123,789; infer=42 | **PLANNED** | `experiments/configs/baselines/benchmark_24obj/dino_r50.py` (exists) | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/d2_train3_completion/dino_r50/trainseed_{training_seed}` | `sota_d2_train3_completion` |
| `D2.SOTA.rtmdet_l.existing` | rtmdet_l | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | `experiments/configs/baselines/benchmark_24obj/rtmdet_l.py` (exists) | historical | `work_dirs/baselines/rtmdet_l_24obj` | `sota_d2_test; artifact=d2-test-rtmdet_l-feca98add171` |
| `D2.SOTA.rtmdet_l.missing_train2` | rtmdet_l | full_train / 2 | train=123,789; infer=42 | **PLANNED** | `experiments/configs/baselines/benchmark_24obj/rtmdet_l.py` (exists) | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/d2_train3_completion/rtmdet_l/trainseed_{training_seed}` | `sota_d2_train3_completion` |
| `D2.SOTA.cascade_rcnn_r50.existing` | cascade_rcnn_r50 | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | `experiments/configs/baselines/benchmark_24obj/cascade_rcnn_r50.py` (exists) | historical | `work_dirs/baselines/cascade_rcnn_r50_24obj` | `sota_d2_test; artifact=d2-test-cascade_rcnn-25e7f63f7418` |
| `D2.SOTA.cascade_rcnn_r50.missing_train2` | cascade_rcnn_r50 | full_train / 2 | train=123,789; infer=42 | **PLANNED** | `experiments/configs/baselines/benchmark_24obj/cascade_rcnn_r50.py` (exists) | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/d2_train3_completion/cascade_rcnn_r50/trainseed_{training_seed}` | `sota_d2_train3_completion` |
| `D2.SOTA.yolox_s.existing` | yolox_s | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | `experiments/configs/baselines/benchmark_24obj/yolox_s.py` (exists) | historical | `work_dirs/baselines/yolox_s` | `sota_d2_test; artifact=d2-test-yolox_s-ea70f679365b` |
| `D2.SOTA.yolox_s.missing_train2` | yolox_s | full_train / 2 | train=123,789; infer=42 | **PLANNED** | `experiments/configs/baselines/benchmark_24obj/yolox_s.py` (exists) | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/d2_train3_completion/yolox_s/trainseed_{training_seed}` | `sota_d2_train3_completion` |

- `D2.SOTA.karyoflow_train3` 参数：RF; shifted t; AdaLN-Zero; DPM++4; K=500; 6 heads 验收：already verified: 3 checkpoint SHA; 1000-image test 论文用途：Main D2 KaryoFlow mean
- `D2.SOTA.diffusiondet.existing` 参数：published table checkpoint; six test metrics 验收：must not be labeled three-training-seed 论文用途：D2 point estimate only
- `D2.SOTA.diffusiondet.missing_train2` 参数：same configuration and selection rule as existing checkpoint 验收：actual framework seed recorded; distinct checkpoint SHA; 1000-image test 论文用途：Upgrade D2 baseline to three independent trainings
- `D2.SOTA.dino_r50.existing` 参数：published table checkpoint; six test metrics 验收：must not be labeled three-training-seed 论文用途：D2 point estimate only
- `D2.SOTA.dino_r50.missing_train2` 参数：same configuration and selection rule as existing checkpoint 验收：actual framework seed recorded; distinct checkpoint SHA; 1000-image test 论文用途：Upgrade D2 baseline to three independent trainings
- `D2.SOTA.rtmdet_l.existing` 参数：published table checkpoint; six test metrics 验收：must not be labeled three-training-seed 论文用途：D2 point estimate only
- `D2.SOTA.rtmdet_l.missing_train2` 参数：same configuration and selection rule as existing checkpoint 验收：actual framework seed recorded; distinct checkpoint SHA; 1000-image test 论文用途：Upgrade D2 baseline to three independent trainings
- `D2.SOTA.cascade_rcnn_r50.existing` 参数：published table checkpoint; six test metrics 验收：must not be labeled three-training-seed 论文用途：D2 point estimate only
- `D2.SOTA.cascade_rcnn_r50.missing_train2` 参数：same configuration and selection rule as existing checkpoint 验收：actual framework seed recorded; distinct checkpoint SHA; 1000-image test 论文用途：Upgrade D2 baseline to three independent trainings
- `D2.SOTA.yolox_s.existing` 参数：published table checkpoint; six test metrics 验收：must not be labeled three-training-seed 论文用途：D2 point estimate only
- `D2.SOTA.yolox_s.missing_train2` 参数：same configuration and selection rule as existing checkpoint 验收：actual framework seed recorded; distinct checkpoint SHA; 1000-image test 论文用途：Upgrade D2 baseline to three independent trainings

### Decision

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D2.SOTA.karyoflow_lqcr_train3` | karyoflow_lqcr | short_train / 3 | train=335778785,790448076,1342286018; infer=42 | **COMPLETED_VERIFIED** | `experiments/configs/ldmdet/directions/capr/paper_train3_lqcr_run*.py` (exists) | historical ross/workstation | `work_dirs/{capr_quality_only_24obj|paper_d2_lqcr_trainrun_*}` | `paired_lqcr_d2_train3;sota_d2_test; artifact=d2-paired-lqcr-train3-4152cdbd5100` |
| `D2.DEC.beta_test.fixed1` | LQCR_beta_0_0.25_0.5_1_2 | inference / 5 | train=335778785; infer=42 | **COMPLETED_DIAGNOSTIC** | `tools/experiment_db/test_inference_ablations.py` (exists) | historical | `results/d2_test_inference_ablations/lqcr_beta/{beta}` | `lqcr_beta_d2_test; artifact=d2-test-inference-ablations-d9fd109403da` |
| `D2.DEC.strict_subsets` | AP90_AP95_scale_overlap_bootstrap | analysis / 2 | train=335778785,790448076,1342286018; infer=42 | **COMPLETED_DIAGNOSTIC** | `tools/experiment_db/analysis/d2_difficult_subset_analysis.py` (exists) | CPU | `tools/experiment_db/evidence_sources/d2_{difficult_subsets|lqcr_vs_dino}_*.json` | `conditional_difficult_subset_d2;conditional_image_bootstrap_d2; artifact=d2-difficult-subsets-paired-train3-725aab1ab102;d2-lqcr-vs-dino-image-bootstrap-1000-seed20260812` |

- `D2.SOTA.karyoflow_lqcr_train3` 参数：parent-matched final-only quality head; beta=2 验收：already verified: shared detector weights unchanged 论文用途：Main D2 LQCR paired effect
- `D2.DEC.beta_test.fixed1` 参数：test beta sweep after beta=2 validation selection; transparency only 验收：caption must state beta=2 selected on validation 论文用途：Post-selection sensitivity, not selection evidence
- `D2.DEC.strict_subsets` 参数：overlap/size strict recall plus 1000-replicate LQCR-vs-DINO image bootstrap 验收：diagnostic paper_eligible=0 论文用途：Mechanism/uncertainty; not SOTA proof

### Generation

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D2.ABL.historical_chain` | DDPM_RF_Heun_AdaLN_DPMpp | mixed / 4 | train=mixed; infer=mixed | **COMPLETED_DESCRIPTIVE** | `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a{0,1,2,4}_*.py` (exists) | historical ross/workstation | `work_dirs/{a0_baseline_24obj|a1_rf_heun_24obj|a2_rf_heun_adaln_24obj|a4_dpm_pp_24obj}` | `legacy_protocol` |
| `D2.ABL.strict.G0` | DDPM_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/d2_strict_ablations/g0_ddpm_batch2.py` (to_create) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/d2_strict_ablations/g0/trainseed_{training_seed}` | `generation_train_ablation_d2_test` |
| `D2.ABL.strict.G1` | RF_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/d2_strict_ablations/g1_rf_linear_scaleshift.py` (to_create) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/d2_strict_ablations/g1/trainseed_{training_seed}` | `generation_train_ablation_d2_test` |
| `D2.ABL.strict.G2` | RF_shifted_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | `experiments/configs/d2_strict_ablations/g2_rf_shifted_scaleshift.py` (to_create) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/d2_strict_ablations/g2/trainseed_{training_seed}` | `generation_train_ablation_d2_test` |
| `D2.ABL.strict.G3` | RF_shifted_AdaLNZero | full_train / 3 | train=335778785,790448076,1342286018; infer=42 | **COMPLETED_VERIFIED_REUSED** | `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py` (exists_plus_dumped_configs) | historical ross/workstation | `work_dirs/{a4_dpm_pp_24obj|multi_seed/a4_dpm_pp_24obj/seed_*}` | `paired_lqcr_d2_train3; artifact=d2-paired-lqcr-train3-4152cdbd5100` |
| `D2.INF.solver_steps.fixed1` | Euler_Heun_DPMpp_x_steps1to4 | inference / 12 | train=335778785; infer=42 | **COMPLETED_FIXED_CHECKPOINT** | `tools/experiment_db/test_inference_ablations.py` (exists) | historical | `results/d2_test_inference_ablations/solver/{solver}_{steps}` | `solver_d2_test; artifact=d2-test-inference-ablations-b81d6a3d498a` |
| `D2.INF.solver_steps.train3` | Euler_Heun_DPMpp_x_steps1to4_train3 | inference / 36 | train=335778785,790448076,1342286018; infer=42 | **PLANNED** | `tools/experiment_db/test_inference_ablations.py` (exists_parameterize_parent) | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/d2_train3_inference/solver/{parent}/{solver}_{steps}` | `solver_d2_train3_test` |
| `D2.INF.topk_renewal.fixed1` | TopK_x_renewal | inference / 10 | train=335778785; infer=42 | **COMPLETED_FIXED_CHECKPOINT** | `tools/experiment_db/test_inference_ablations.py` (exists) | historical | `results/d2_test_inference_ablations/topk_renewal/{variant}` | `topk_renewal_d2_test; artifact=d2-test-inference-ablations-6dd1dcb1cc91` |
| `D2.INF.topk_renewal.train3` | TopK_x_renewal_train3 | inference / 30 | train=335778785,790448076,1342286018; infer=42 | **PLANNED** | `tools/experiment_db/test_inference_ablations.py` (exists_parameterize_parent) | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/d2_train3_inference/topk_renewal/{parent}/{variant}` | `topk_renewal_d2_train3_test` |

- `D2.ABL.historical_chain` 参数：historical A0/A1/A2/A4; configs/checkpoints and training seeds not fully paired 验收：never report adjacent deltas as isolated effects 论文用途：Context only; not cumulative causal ablation
- `D2.ABL.strict.G0` 参数：DDPM; DDIM1; linear t; scale-shift; batch=2 验收：only named factor changes; same batch/optimizer/selection; six test metrics 论文用途：Cross-cohort strict ablation replication
- `D2.ABL.strict.G1` 参数：RF; linear t; scale-shift; batch=2 验收：only named factor changes; same batch/optimizer/selection; six test metrics 论文用途：Cross-cohort strict ablation replication
- `D2.ABL.strict.G2` 参数：RF; shifted t; scale-shift; batch=2 验收：only named factor changes; same batch/optimizer/selection; six test metrics 论文用途：Cross-cohort strict ablation replication
- `D2.ABL.strict.G3` 参数：RF; shifted t; AdaLN-Zero; DPM++4 验收：confirm optimizer/batch/selection compatibility with new G0-G2 论文用途：Strict chain endpoint; reuse only after config compatibility audit
- `D2.INF.solver_steps.fixed1` 参数：one fixed KaryoFlow checkpoint x 3 solvers x 4 steps 验收：label fixed-checkpoint; not training replication 论文用途：Conditional solver mechanism evidence
- `D2.INF.solver_steps.train3` 参数：3 independent parents x 3 solvers x 4 steps 验收：aggregate one value per training parent 论文用途：Training-robust solver ablation
- `D2.INF.topk_renewal.fixed1` 参数：K={100,150,200,300,500} x renewal={off,on} 验收：label fixed-checkpoint 论文用途：Conditional identity/candidate-budget evidence
- `D2.INF.topk_renewal.train3` 参数：3 parents x 5 K x renewal off/on 验收：aggregate by training parent 论文用途：Training-robust identity analysis

### Deployment

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `D2.DEP.distill_h3.existing` | H6_to_H3_single_parent | short_train / 1 | train=one; infer=42 | **COMPLETED_SINGLE_PARENT** | `experiments/configs/ldmdet/directions/mainline_ablation_24obj/h3_distill_plan_a_24obj.py` (exists) | historical ross/workstation | `work_dirs/h3_distill_plan_a_24obj` | `head_distillation` |
| `D2.DEP.distill_h3.train3` | H6_to_H3_parent_matched_train3 | short_train / 3 | train=335778785,790448076,1342286018; infer=42 | **PLANNED** | `experiments/configs/d2_deployment/h3_distill_parent.py` (to_create_or_parameterize) | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/d2_deployment/h3/{parent}` | `head_distillation_d2_train3` |
| `D2.DEP.GACS` | GACS | inference / 1 | train=one/unknown; infer=42 | **PARTIAL_LEGACY** | `experiments/configs/ldmdet/directions/inference_opt/gacs*.py` (locate_and_standardize) | accuracy:any; latency=ross:A6000:0 | `results/d2_deployment/gacs/{parent}` | `GACS` |
| `D2.DEP.speed` | all_operating_points | benchmark / 1 | train=mixed; infer=none | **COMPLETED_PARTIAL** | `tools/benchmark_fps.py` (exists) | ross:A6000:0 exclusive | `results/benchmark/a6000/{variant}` | `speed_accuracy` |

- `D2.DEP.distill_h3.existing` 参数：H6 teacher -> H3 student; mapping 0,2,5 验收：do not call three-training-seed 论文用途：Conditional deployment point
- `D2.DEP.distill_h3.train3` 参数：one H3 student per KaryoFlow parent; identical mapping/loss 验收：three parent-matched students; A6000 latency only 论文用途：Training-robust deployment evidence
- `D2.DEP.GACS` 参数：dynamic deployment policy; historical D1 loss ~0.0007 mAP; D2 held-out test missing 验收：D2 test plus A6000 speed before paper use 论文用途：Optional deployment extension, not main precision module
- `D2.DEP.speed` 参数：A6000: RF-Heun; DPM++ K=100/200/300/500; LQCR; H3; repeat after final checkpoints 验收：repeat final selected points with exclusive GPU and frozen timing boundary 论文用途：Speed-accuracy figure/table

## 历史复合Dataset 1：非对照归档

### Archive

| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |
|---|---|---:|---|---|---|---|---|---|
| `LEGACY.D1.composite` | all_historical_results | mixed / 1 | train=mixed; infer=mixed | **ARCHIVED_NONCOMPARABLE** | `tools/experiment_db/experiments.db` (exists) | historical ross/workstation | `work_dirs/*; results/*` | `all D1 families except D1_INHOUSE1700_V1; artifact=multiple` |

- `LEGACY.D1.composite` 参数：old 2200-image composite contains 500 D2-derived images and old split; results remain auditable 验收：must be labeled legacy and non-comparable 论文用途：Provenance/history only; never pool with new D1I or D2

## 两数据集逐项对照与最小投稿闭环

| 论文问题 | D1I要求 | D2要求 | 当前闭环 |
|---|---|---|---|
| KaryoFlow是否优于检测基线 | 6模型×3训练seed | KaryoFlow已有3；五基线补2seed | 未闭合 |
| RF本身是否有效 | G0/G1三seed严格配对 | G0/G1严格重跑 | 未闭合 |
| shifted schedule是否有效 | G1/G2三seed | G1/G2三seed | 未闭合 |
| AdaLN-Zero是否有效 | G2/G3三seed | G2/G3三seed | 未闭合 |
| DPM++是否有效 | G3同权重Euler/Heun/DPM++ | 已有fixed1；补train3 | 条件性闭合 |
| LQCR是否有效 | 3 parent-child配对 | 已完成3配对 | D1I进行后闭合 |
| 小/重叠目标机制 | D1I预测后分析 | D2已完成诊断 | 单数据集条件性 |
| 部署收益 | H3×3 + A6000测速 | H3目前单parent；补train3 | 未闭合 |
| GACS | D1I test+速度 | D2 test+速度 | 未闭合；不得作主精度模块 |

## 运行和登记流程

1. 先在本总账和 `experiment_ledger` 中存在唯一 `ledger_id`。
2. 训练任务再创建 `train_run_registry.train_run_id`，固定dataset manifest SHA、config SHA、training seed、executor与work directory。
3. SwanLab项目按数据集计划登记；D1I统一使用 `KaryoFlow-Self1700`，run名称 `{variant}_seed{training_seed}`。
4. checkpoint只能由validation选择；test不能改变配置、beta、epoch或阈值。
5. test保存原始预测、日志、resolved config与SHA；独立pycocotools复算六指标。
6. 先写immutable run-evidence JSON，再幂等导入数据库；审计PASS后才能设paper eligible。
7. 论文表格/摘要数字必须反向链接controlled result或本总账中的artifact ID。

## 已知限制与下一步

- 新D1I的三条KaryoFlow是正式SwanLab跟踪run；先前未跟踪尝试已归档为invalid。
- D1I DiffusionDet外部基线继承batch=4；严格G0另设batch=2，二者不可混为同一因果消融。
- D2历史A0→A4链跨checkpoint且seed口径混合，只能作为描述性背景；严格链需要重跑G0-G2。
- 固定checkpoint三推理seed不得替代三训练seed。
- 未取得patient/specimen ID，因此D1I只能称acquisition-stem group-disjoint，不能称patient-disjoint。
