# KaryoFlow v2 完整实验路线矩阵

> 这是双数据集实验调度、配置解析、数据库登记和论文数字校对的唯一权威入口。
> YAML 是源文件；本 Markdown、CSV 与 SQLite 表均由生成器派生，禁止手工修改派生文件。

## 权威身份

- YAML：`experiments/manifests/paper_experiment_route_matrix.yaml`
- YAML SHA-256：`e04843969ea41095b447ec16ac279f7489cb89357f2c9d98e0a482dd9ac6c21a`
- 数据库 artifact：`paper-route-matrix-v2-e04843969ea4`
- 论文 claim manifest：`experiments/manifests/paper_claim_manifest.yaml`（SHA-256 `12dacfbf137e3973829f6925a4f279f6acc9d5bf982ec3e0ce792092155865b5`）
- 实验组：53；展开运行：287。
- 状态分布：ARCHIVED_NONCOMPARABLE=1, BLOCKED_CHECKPOINTS=1, BLOCKED_PARENT=3, BLOCKED_PREDICTIONS=5, COMPLETED_CENTRAL_TEST_VERIFIED=2, COMPLETED_DESCRIPTIVE=1, COMPLETED_DIAGNOSTIC=2, COMPLETED_EVIDENCE_ONLY=1, COMPLETED_FIXED_CHECKPOINT=2, COMPLETED_PARTIAL=1, COMPLETED_POINT_ESTIMATE=5, COMPLETED_VERIFIED=2, DEFERRED_UNTIL_D1_COMPLETE=12, PARTIAL_LEGACY=1, PLANNED=6, QUEUED_SERIAL_A4000=2, RECOVERY_RUNNING_2_OF_3_COMPLETE=1, REVIEW_EXISTING_VALIDATION_SELECTION=1, TEST_QUEUED_1_OF_3_COMPLETE=2, TRAIN3_COMPLETE_TEST_QUEUED=2。

## 数据与统计口径

- `D1_INHOUSE1700_V2`：1190/170/340，纯自建、D2 类别 ID 对齐、group-disjoint 70/10/20 划分。
- `D2`：3500/500/1000，使用作者发布的原始公开划分，以保持与既有 benchmark 和历史权重直接可比。
- repaired-split 不再作为论文数据口径；相关数据、配置、路线和数据库发布记录均不保留。
- 训练复现以不同训练 checkpoint 为统计单位；固定 checkpoint 的推理 seed 不得冒充训练 seed。
- 主精度只允许 held-out test；validation 只用于 checkpoint/超参数选择。
- 效率只允许按 A6000 严格协议比较；服务器名称只存在内部路线矩阵，不进入论文。

## 路线矩阵

### D1_INHOUSE1700_V2

#### Data

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D1I.DATA.test_characterization` | test_characterization | analysis / 1 | train=none; infer=none | **PLANNED** | PROTOCOL_TO_IMPLEMENT | `—` | CPU | `results/v2/d1/analysis/test_characterization` | `route:D1I.DATA.test_characterization; dataset_characterization_d1i` |

- `D1I.DATA.test_characterization`：科学因素=D1 test population, scale CDF, overlap and size bins；验收=versioned analysis JSON bound to final test annotation/prediction SHA；论文用途=Dataset table and scale figure；备注=Required before the manuscript is refreshed.
#### Detector comparison

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D1I.SOTA.karyoflow` | karyoflow | full_train / 3 | train=42,123,789; infer=42 | **COMPLETED_CENTRAL_TEST_VERIFIED** | READY | `experiments/configs/methods/karyoflow.py` | 42=workstation:A5000:0;123=workstation:A4000:1;789=ross:A6000:0 | `work_dirs/v2/d1/karyoflow_r50/trainseed_{training_seed}` | `route:D1I.SOTA.karyoflow; sota_d1i_test` |
| `D1I.SOTA.diffusiondet` | diffusiondet | full_train / 3 | train=42,123,789; infer=42 | **TEST_QUEUED_1_OF_3_COMPLETE** | READY | `experiments/configs/methods/diffusiondet_ddpm.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d1/diffusiondet_ddpm_r50/trainseed_{training_seed}` | `route:D1I.SOTA.diffusiondet; sota_d1i_test` |
| `D1I.SOTA.dino_r50` | dino_r50 | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | READY | `experiments/configs/methods/dino_r50.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d1/dino_r50/trainseed_{training_seed}` | `route:D1I.SOTA.dino_r50; sota_d1i_test` |
| `D1I.SOTA.rtmdet_l` | rtmdet_l | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | READY | `experiments/configs/methods/rtmdet_l.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d1/rtmdet_l/trainseed_{training_seed}` | `route:D1I.SOTA.rtmdet_l; sota_d1i_test` |
| `D1I.SOTA.cascade_rcnn_r50` | cascade_rcnn_r50 | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | READY | `experiments/configs/methods/cascade_rcnn_r50.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d1/cascade_rcnn_r50/trainseed_{training_seed}` | `route:D1I.SOTA.cascade_rcnn_r50; sota_d1i_test` |
| `D1I.SOTA.yolox_s` | yolox_s | full_train / 3 | train=42,123,789; infer=42 | **PLANNED** | READY | `experiments/configs/methods/yolox_s.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d1/yolox_s/trainseed_{training_seed}` | `route:D1I.SOTA.yolox_s; sota_d1i_test` |

- `D1I.SOTA.karyoflow`：科学因素=RF; shifted t; AdaLN-Zero; DPM++ 4-step; K=500; 6 heads；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d1.karyoflow_r50.train；验收=3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics；论文用途=Main two-cohort detector table；备注=Canonical v2 seeds 42, 123, and 789 completed independent training and central held-out test evaluation (340 images; inference seed 42); all three immutable config digests were preserved during evaluation.
- `D1I.SOTA.diffusiondet`：科学因素=DDPM; DDIM 1-step; linear t; scale-shift; K=500; 6 heads；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d1.diffusiondet_ddpm_r50.train；验收=3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics；论文用途=Main two-cohort detector table；备注=No active v2 run is currently registered.
- `D1I.SOTA.dino_r50`：科学因素=R50; 900 queries; 6 encoder/decoder layers；解析配置=batch 2, 150 epochs, AdamW, lr=2.5e-05, config_id=v2.d1.dino_r50.train；验收=3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics；论文用途=Main two-cohort detector table；备注=No active v2 run is currently registered.
- `D1I.SOTA.rtmdet_l`：科学因素=RTMDet-L; inherited benchmark optimizer；解析配置=batch 2, 150 epochs, AdamW, lr=0.0001, config_id=v2.d1.rtmdet_l.train；验收=3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics；论文用途=Main two-cohort detector table；备注=No active v2 run is currently registered.
- `D1I.SOTA.cascade_rcnn_r50`：科学因素=Cascade R-CNN R50; 3 stages；解析配置=batch 2, 150 epochs, AdamW, lr=0.0001, config_id=v2.d1.cascade_rcnn_r50.train；验收=3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics；论文用途=Main two-cohort detector table；备注=No active v2 run is currently registered.
- `D1I.SOTA.yolox_s`：科学因素=YOLOX-S；解析配置=batch 2, 200 epochs, AdamW, lr=0.0001, config_id=v2.d1.yolox_s.train；验收=3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics；论文用途=Main two-cohort detector table；备注=No active v2 run is currently registered.
#### Generation

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D1I.ABL.G0` | DDPM_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **TEST_QUEUED_1_OF_3_COMPLETE** | READY | `experiments/configs/methods/strict_g0_ddpm_linear_scaleshift.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d1_generation/strict_g0_ddpm_linear_scaleshift_r50/trainseed_{training_seed}` | `route:D1I.ABL.G0; generation_train_ablation_d1i_test` |
| `D1I.ABL.G1` | RF_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **TRAIN3_COMPLETE_TEST_QUEUED** | READY | `experiments/configs/methods/strict_g1_rf_linear_scaleshift.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A5000:0 | `work_dirs/v2/d1_generation/strict_g1_rf_linear_scaleshift_r50/trainseed_{training_seed}` | `route:D1I.ABL.G1; generation_train_ablation_d1i_test` |
| `D1I.ABL.G2` | RF_shifted_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **TRAIN3_COMPLETE_TEST_QUEUED** | READY | `experiments/configs/methods/strict_g2_rf_shifted_scaleshift.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A5000:0 | `work_dirs/v2/d1_generation/strict_g2_rf_shifted_scaleshift_r50/trainseed_{training_seed}` | `route:D1I.ABL.G2; generation_train_ablation_d1i_test` |
| `D1I.ABL.G3` | RF_shifted_AdaLNZero | full_train / 3 | train=42,123,789; infer=42 | **RECOVERY_RUNNING_2_OF_3_COMPLETE** | READY | `experiments/configs/methods/strict_g3_rf_shifted_adaln_zero.py` | 42=ross:A6000:0;123=ross:A6000:0;789=ross:A6000:0 (recovery) | `work_dirs/v2/d1_generation/strict_g3_rf_shifted_adaln_zero_r50/trainseed_{training_seed}` | `route:D1I.ABL.G3; generation_train_ablation_d1i_test` |
| `D1I.INF.solver_steps` | Euler_Heun_DPMpp_x_steps1to4 | inference / 36 | train=42,123,789; infer=42 | **QUEUED_SERIAL_A4000** | PROTOCOL_READY | `experiments/configs/methods/karyoflow.py` | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/v2/d1_inhouse1700/inference/solver/{parent}/{solver}_{steps}` | `route:D1I.INF.solver_steps; solver_d1i_test` |
| `D1I.INF.topk_renewal` | TopK_x_renewal | inference / 30 | train=42,123,789; infer=42 | **QUEUED_SERIAL_A4000** | PROTOCOL_READY | `experiments/configs/methods/karyoflow.py` | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/v2/d1_inhouse1700/inference/topk_renewal/{parent}/k{K}_{renewal}` | `route:D1I.INF.topk_renewal; topk_renewal_d1i_test` |

- `D1I.ABL.G0`：科学因素=DDPM; DDIM1; linear t; scale-shift；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d1_generation.strict_g0_ddpm_linear_scaleshift_r50.train；验收=only named factor changes; batch/optimizer/augmentation/selection fixed；论文用途=Strict incremental generation ablation；备注=All three independent G0 runs completed with distinct validation-best checkpoint SHA values. Seed 789 has verified test evidence; seeds 42 and 123 are queued for the same held-out-test protocol.
- `D1I.ABL.G1`：科学因素=RF; Euler/locked validation protocol; linear t; scale-shift；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d1_generation.strict_g1_rf_linear_scaleshift_r50.train；验收=only named factor changes; batch/optimizer/augmentation/selection fixed；论文用途=Strict incremental generation ablation；备注=All three independent G1 runs completed training and are queued for serial held-out-test evaluation on the accuracy worker.
- `D1I.ABL.G2`：科学因素=RF; shifted t (shift=3); scale-shift；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d1_generation.strict_g2_rf_shifted_scaleshift_r50.train；验收=only named factor changes; batch/optimizer/augmentation/selection fixed；论文用途=Strict incremental generation ablation；备注=All three G2 training processes completed; final evidence processing and the three serial held-out-test evaluations are queued.
- `D1I.ABL.G3`：科学因素=RF; shifted t; AdaLN-Zero; Euler 1-step validation；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d1_generation.strict_g3_rf_shifted_adaln_zero_r50.train；验收=only time conditioning changes from G2; common validation protocol；论文用途=Strict one-factor generation ablation；备注=Seeds 42 and 123 completed training. After the workstation became unreachable, seed 789 was reassigned to an isolated Ross recovery directory and is running on the A6000; the unreachable remote attempt is excluded from evidence import. All three held-out tests remain preregistered; canonical DPM++ is a separate inference comparison.
- `D1I.INF.solver_steps`：科学因素=3 parents x {Euler,Heun,DPM++} x {1,2,3,4}; report steps and NFE；验收=same checkpoint per parent; common test; no cross-checkpoint causal claim；论文用途=Solver/step accuracy-efficiency ablation；备注=All 36 fixed-parent test evaluations are queued serially: three independent KaryoFlow parents by Euler/Heun/DPM++ and one to four steps.
- `D1I.INF.topk_renewal`：科学因素=3 parents x K={100,150,200,300,500} x renewal={off,on}；验收=same checkpoint/seed; renewal-off candidate identity audit；论文用途=Identity/renewal and candidate-budget analysis；备注=All 30 fixed-parent test evaluations are queued serially: three independent KaryoFlow parents by five candidate budgets and renewal on/off.
#### Decision

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D1I.SOTA.karyoflow_lqcr` | karyoflow_lqcr | short_train / 3 | train=42,123,789; infer=42 | **COMPLETED_CENTRAL_TEST_VERIFIED** | READY | `experiments/configs/methods/karyoflow_lqcr.py` | 42=workstation:A5000:0;123=workstation:A4000:1;789=ross:A6000:0 | `work_dirs/v2/d1/karyoflow_lqcr_r50/trainseed_{training_seed}` | `route:D1I.SOTA.karyoflow_lqcr; paired_lqcr_d1i_train3` |
| `D1I.DEC.beta_val` | LQCR_beta_0_0.25_0.5_1_2 | inference / 15 | train=42,123,789; infer=42 | **BLOCKED_PARENT** | PROTOCOL_READY_PARENT_PENDING | `experiments/configs/methods/karyoflow_lqcr.py` | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/v2/d1_inhouse1700/lqcr_beta_val/{parent}/beta_{beta}` | `route:D1I.DEC.beta_val; lqcr_beta_d1i_val` |
| `D1I.DEC.strict_subsets` | AP90_AP95_scale_overlap_quality | analysis / 1 | train=42,123,789; infer=42 | **BLOCKED_PREDICTIONS** | DATASET_ADAPTATION_REQUIRED | `tools/experiment_db/protocols/d2_difficulty_strata.json` | CPU after test predictions | `results/v2/d1_inhouse1700/analysis/lqcr_difficult_subsets` | `route:D1I.DEC.strict_subsets; conditional_difficult_subset_d1i` |
| `D1I.DEC.quality_validity` | quality_iou_validity | analysis / 1 | train=42,123,789; infer=42 | **BLOCKED_PREDICTIONS** | PROTOCOL_TO_IMPLEMENT | `—` | CPU | `results/v2/d1/analysis/quality_iou_validity` | `route:D1I.DEC.quality_validity; lqcr_quality_validity_d1i` |
| `D1I.ANALYSIS.per_class` | per_class_error | analysis / 1 | train=42,123,789; infer=42 | **BLOCKED_PREDICTIONS** | PROTOCOL_TO_IMPLEMENT | `—` | CPU | `results/v2/d1/analysis/per_class_error` | `route:D1I.ANALYSIS.per_class; per_class_d1i_test` |

- `D1I.SOTA.karyoflow_lqcr`：科学因素=parent-matched final-only quality head; beta=2 selected on validation；解析配置=batch 2, 12 epochs, AdamW, lr=0.001, config_id=v2.d1.karyoflow_lqcr_r50.train；验收=parent checkpoint identity; 590 shared tensors unchanged; paired test delta；论文用途=Independent decision contribution；备注=All three parent-matched quality-head runs completed training, passed the final-only tensor audit (590 shared tensors unchanged; five quality-head tensors added), and completed central held-out test evaluation with immutable prediction and metric evidence.
- `D1I.DEC.beta_val`：科学因素=3 LQCR parents x beta={0,0.25,0.5,1,2}; validation only；验收=select beta before any new test evaluation；论文用途=Validation selection; not a test claim；备注=No active v2 run is currently registered.
- `D1I.DEC.strict_subsets`：科学因素=AP90/AP95; AP_S/M/L; overlap strata; size quartiles; quality-IoU Spearman; paired image bootstrap；验收=image-level pairing; diagnostics not paper SOTA rows；论文用途=Mechanism evidence and limitation；备注=No active v2 run is currently registered.
- `D1I.DEC.quality_validity`：科学因素=quality-IoU Spearman, MAE/RMSE and reliability bins；验收=versioned analysis JSON bound to final test annotation/prediction SHA；论文用途=Mechanism/error analysis；备注=Required before the manuscript is refreshed.
- `D1I.ANALYSIS.per_class`：科学因素=24-class AP and morphology-group error analysis；验收=versioned analysis JSON bound to final test annotation/prediction SHA；论文用途=Mechanism/error analysis；备注=Required before the manuscript is refreshed.
#### Deployment

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D1I.DEP.distill_h3` | H6_teacher_to_H3_student | short_train / 3 | train=42,123,789; infer=42 | **BLOCKED_PARENT** | READY_PARENT_PENDING | `experiments/configs/methods/karyoflow_h3_distill.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d1_distill/karyoflow_h3_distill_r50/trainseed_{training_seed}` | `route:D1I.DEP.distill_h3; head_distillation_d1i_test` |
| `D1I.DEP.GACS` | GACS | inference / 3 | train=42,123,789; infer=42 | **BLOCKED_PARENT** | PROTOCOL_READY_PARENT_PENDING | `experiments/configs/methods/karyoflow.py` | accuracy:any; latency=ross:A6000:0 | `results/v2/d1_inhouse1700/deployment/gacs/{parent}` | `route:D1I.DEP.GACS; gacs_d1i_test` |
| `D1I.DEP.speed` | all_operating_points | benchmark / 1 | train=42,123,789; infer=none | **BLOCKED_CHECKPOINTS** | READY | `experiments/configs/deployment/a6000_speed_protocol.yaml` | ross:A6000:0 exclusive | `results/v2/d1_inhouse1700/benchmark/a6000/{variant}` | `route:D1I.DEP.speed; speed_accuracy_d1i` |

- `D1I.DEP.distill_h3`：科学因素=teacher heads=6; student heads=3; mapping 0<-0,1<-2,2<-5; fixed loss/protocol；解析配置=batch 2, 50 epochs, AdamW, lr=1e-05, config_id=v2.d1_distill.karyoflow_h3_distill_r50.train；验收=three parent-matched students; test mAP; A6000 latency only；论文用途=Deployment extension；备注=No active v2 run is currently registered.
- `D1I.DEP.GACS`：科学因素=prespecified dynamic policy; accuracy and latency; no precision-module claim；验收=held-out test on both cohorts; same A6000 timing protocol；论文用途=Optional dynamic deployment extension；备注=No active v2 run is currently registered.
- `D1I.DEP.speed`：科学因素=batch=1; same input; precision; warmup; timed iterations; generation/teacher/student/LQCR/GACS；验收=exclusive A6000; no foreign process; accuracy linked to verified test record；论文用途=Speed-accuracy figure/table；备注=No active v2 run is currently registered.

### D2

#### Data

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D2.DATA.test_characterization` | test_characterization | analysis / 1 | train=none; infer=none | **PLANNED** | PROTOCOL_TO_IMPLEMENT | `—` | CPU | `results/v2/d2/analysis/test_characterization` | `route:D2.DATA.test_characterization; dataset_characterization_d2` |

- `D2.DATA.test_characterization`：科学因素=D2 test population, scale CDF, overlap and size bins；验收=versioned analysis JSON bound to final test annotation/prediction SHA；论文用途=Dataset table and scale figure；备注=Required before the manuscript is refreshed.
#### Detector comparison

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D2.SOTA.karyoflow_train3` | karyoflow | full_train / 3 | train=335778785,790448076,1342286018; infer=42 | **COMPLETED_VERIFIED** | EXACT | `experiments/configs/methods/karyoflow_ot_legacy.py` | historical ross/workstation | `work_dirs/{a4_dpm_pp_24obj|multi_seed/a4_dpm_pp_24obj/seed_*}` | `route:D2.SOTA.karyoflow_train3; paired_lqcr_d2_train3;sota_d2_test` |
| `D2.SOTA.diffusiondet.existing` | diffusiondet | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | STRUCTURAL_ONLY | `experiments/configs/methods/diffusiondet_ddpm_legacy.py` | historical | `work_dirs/baselines/diffusiondet_24obj` | `route:D2.SOTA.diffusiondet.existing; sota_d2_test` |
| `D2.SOTA.diffusiondet.train3_supplement` | diffusiondet | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/diffusiondet_ddpm.py` | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/v2/d2_sota_completion/diffusiondet_ddpm_r50/trainseed_{training_seed}` | `route:D2.SOTA.diffusiondet.train3_supplement; sota_d2_train3_completion` |
| `D2.SOTA.dino_r50.existing` | dino_r50 | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | EXACT | `experiments/configs/methods/dino_r50_legacy.py` | historical | `work_dirs/baselines/dino_r50_24obj` | `route:D2.SOTA.dino_r50.existing; sota_d2_test` |
| `D2.SOTA.dino_r50.train3_supplement` | dino_r50 | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/dino_r50.py` | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/v2/d2_sota_completion/dino_r50/trainseed_{training_seed}` | `route:D2.SOTA.dino_r50.train3_supplement; sota_d2_train3_completion` |
| `D2.SOTA.rtmdet_l.existing` | rtmdet_l | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | EXACT | `experiments/configs/methods/rtmdet_l_legacy.py` | historical | `work_dirs/baselines/rtmdet_l_24obj` | `route:D2.SOTA.rtmdet_l.existing; sota_d2_test` |
| `D2.SOTA.rtmdet_l.train3_supplement` | rtmdet_l | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/rtmdet_l.py` | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/v2/d2_sota_completion/rtmdet_l/trainseed_{training_seed}` | `route:D2.SOTA.rtmdet_l.train3_supplement; sota_d2_train3_completion` |
| `D2.SOTA.cascade_rcnn_r50.existing` | cascade_rcnn_r50 | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | EXACT | `experiments/configs/methods/cascade_rcnn_r50_legacy.py` | historical | `work_dirs/baselines/cascade_rcnn_r50_24obj` | `route:D2.SOTA.cascade_rcnn_r50.existing; sota_d2_test` |
| `D2.SOTA.cascade_rcnn_r50.train3_supplement` | cascade_rcnn_r50 | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/cascade_rcnn_r50.py` | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/v2/d2_sota_completion/cascade_rcnn_r50/trainseed_{training_seed}` | `route:D2.SOTA.cascade_rcnn_r50.train3_supplement; sota_d2_train3_completion` |
| `D2.SOTA.yolox_s.existing` | yolox_s | evaluation / 1 | train=unknown/one; infer=42 | **COMPLETED_POINT_ESTIMATE** | EXACT | `experiments/configs/methods/yolox_s_legacy.py` | historical | `work_dirs/baselines/yolox_s` | `route:D2.SOTA.yolox_s.existing; sota_d2_test` |
| `D2.SOTA.yolox_s.train3_supplement` | yolox_s | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/yolox_s.py` | 123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits) | `work_dirs/v2/d2_sota_completion/yolox_s/trainseed_{training_seed}` | `route:D2.SOTA.yolox_s.train3_supplement; sota_d2_train3_completion` |

- `D2.SOTA.karyoflow_train3`：科学因素=RF; shifted t; AdaLN-Zero; DPM++4; K=500; 6 heads；验收=already verified: 3 checkpoint SHA; 1000-image test；论文用途=Original-split D2 main detector；备注=Verified three-independent-training-run evidence on the publisher-provided original D2 split.
- `D2.SOTA.diffusiondet.existing`：科学因素=published table checkpoint; six test metrics；验收=must not be labeled three-training-seed；论文用途=Original-split D2 benchmark point estimate；备注=Recorded test result on the publisher-provided original D2 split.
- `D2.SOTA.diffusiondet.train3_supplement`：科学因素=same configuration and selection rule as existing checkpoint；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d2_sota_completion.diffusiondet_ddpm_r50.train；验收=three independently trained models; six held-out-test metrics；论文用途=Optional D2 three-training-run robustness supplement；备注=Do not enqueue until every required D1 experiment and evaluation is complete.
- `D2.SOTA.dino_r50.existing`：科学因素=published table checkpoint; six test metrics；验收=must not be labeled three-training-seed；论文用途=Original-split D2 benchmark point estimate；备注=Recorded test result on the publisher-provided original D2 split.
- `D2.SOTA.dino_r50.train3_supplement`：科学因素=same configuration and selection rule as existing checkpoint；解析配置=batch 2, 150 epochs, AdamW, lr=2.5e-05, config_id=v2.d2_sota_completion.dino_r50.train；验收=three independently trained models; six held-out-test metrics；论文用途=Optional D2 three-training-run robustness supplement；备注=Do not enqueue until every required D1 experiment and evaluation is complete.
- `D2.SOTA.rtmdet_l.existing`：科学因素=published table checkpoint; six test metrics；验收=must not be labeled three-training-seed；论文用途=Original-split D2 benchmark point estimate；备注=Recorded test result on the publisher-provided original D2 split.
- `D2.SOTA.rtmdet_l.train3_supplement`：科学因素=same configuration and selection rule as existing checkpoint；解析配置=batch 2, 150 epochs, AdamW, lr=0.0001, config_id=v2.d2_sota_completion.rtmdet_l.train；验收=three independently trained models; six held-out-test metrics；论文用途=Optional D2 three-training-run robustness supplement；备注=Do not enqueue until every required D1 experiment and evaluation is complete.
- `D2.SOTA.cascade_rcnn_r50.existing`：科学因素=published table checkpoint; six test metrics；验收=must not be labeled three-training-seed；论文用途=Original-split D2 benchmark point estimate；备注=Recorded test result on the publisher-provided original D2 split.
- `D2.SOTA.cascade_rcnn_r50.train3_supplement`：科学因素=same configuration and selection rule as existing checkpoint；解析配置=batch 2, 150 epochs, AdamW, lr=0.0001, config_id=v2.d2_sota_completion.cascade_rcnn_r50.train；验收=three independently trained models; six held-out-test metrics；论文用途=Optional D2 three-training-run robustness supplement；备注=Do not enqueue until every required D1 experiment and evaluation is complete.
- `D2.SOTA.yolox_s.existing`：科学因素=published table checkpoint; six test metrics；验收=must not be labeled three-training-seed；论文用途=Original-split D2 benchmark point estimate；备注=Recorded test result on the publisher-provided original D2 split.
- `D2.SOTA.yolox_s.train3_supplement`：科学因素=same configuration and selection rule as existing checkpoint；解析配置=batch 2, 200 epochs, AdamW, lr=0.0001, config_id=v2.d2_sota_completion.yolox_s.train；验收=three independently trained models; six held-out-test metrics；论文用途=Optional D2 three-training-run robustness supplement；备注=Do not enqueue until every required D1 experiment and evaluation is complete.
#### Generation

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D2.ABL.historical_chain` | DDPM_RF_Heun_AdaLN_DPMpp | mixed / 4 | train=mixed; infer=mixed | **COMPLETED_DESCRIPTIVE** | MIGRATED_MIXED_HISTORY | `experiments/manifests/d2_paper_experiment_inventory.yaml` | historical ross/workstation | `work_dirs/{a0_baseline_24obj|a1_rf_heun_24obj|a2_rf_heun_adaln_24obj|a4_dpm_pp_24obj}` | `route:D2.ABL.historical_chain; legacy_protocol` |
| `D2.ABL.strict.G0` | DDPM_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/strict_g0_ddpm_linear_scaleshift.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d2_generation/strict_g0_ddpm_linear_scaleshift_r50/trainseed_{training_seed}` | `route:D2.ABL.strict.G0; generation_train_ablation_d2_test` |
| `D2.ABL.strict.G1` | RF_linear_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/strict_g1_rf_linear_scaleshift.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d2_generation/strict_g1_rf_linear_scaleshift_r50/trainseed_{training_seed}` | `route:D2.ABL.strict.G1; generation_train_ablation_d2_test` |
| `D2.ABL.strict.G2` | RF_shifted_scaleshift | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/strict_g2_rf_shifted_scaleshift.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d2_generation/strict_g2_rf_shifted_scaleshift_r50/trainseed_{training_seed}` | `route:D2.ABL.strict.G2; generation_train_ablation_d2_test` |
| `D2.ABL.strict.G3` | RF_shifted_AdaLNZero | full_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/strict_g3_rf_shifted_adaln_zero.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d2_generation/strict_g3_rf_shifted_adaln_zero_r50/trainseed_{training_seed}` | `route:D2.ABL.strict.G3; strict_generation_d2_test` |
| `D2.INF.solver_steps.fixed1` | Euler_Heun_DPMpp_x_steps1to4 | inference / 12 | train=335778785; infer=42 | **COMPLETED_FIXED_CHECKPOINT** | PROTOCOL_READY | `experiments/configs/methods/karyoflow_ot_legacy.py` | historical | `results/d2_test_inference_ablations/solver/{solver}_{steps}` | `route:D2.INF.solver_steps.fixed1; solver_d2_test` |
| `D2.INF.solver_steps.train3` | Euler_Heun_DPMpp_x_steps1to4_train3 | inference / 36 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | PROTOCOL_READY | `experiments/configs/methods/karyoflow.py` | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/v2/d2_taichung/inference/solver/{parent}/{solver}_{steps}` | `route:D2.INF.solver_steps.train3; solver_d2_train3_test` |
| `D2.INF.topk_renewal.fixed1` | TopK_x_renewal | inference / 10 | train=335778785; infer=42 | **COMPLETED_FIXED_CHECKPOINT** | PROTOCOL_READY | `experiments/configs/methods/karyoflow_ot_legacy.py` | historical | `results/d2_test_inference_ablations/topk_renewal/{variant}` | `route:D2.INF.topk_renewal.fixed1; topk_renewal_d2_test` |
| `D2.INF.topk_renewal.train3` | TopK_x_renewal_train3 | inference / 30 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | PROTOCOL_READY | `experiments/configs/methods/karyoflow.py` | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/v2/d2_taichung/inference/topk_renewal/{parent}/{variant}` | `route:D2.INF.topk_renewal.train3; topk_renewal_d2_train3_test` |

- `D2.ABL.historical_chain`：科学因素=historical A0/A1/A2/A4; configs/checkpoints and training seeds not fully paired；验收=never report adjacent deltas as isolated effects；论文用途=Context only; not cumulative causal ablation；备注=Historical foundation comparison only; never interpret adjacent rows as isolated cumulative effects.
- `D2.ABL.strict.G0`：科学因素=DDPM; DDIM1; linear t; scale-shift；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d2_generation.strict_g0_ddpm_linear_scaleshift_r50.train；验收=only named factor changes; same batch/optimizer/selection; six test metrics；论文用途=Cross-cohort strict ablation replication；备注=D2 three-training-run supplement; do not enqueue until all required D1 work is complete.
- `D2.ABL.strict.G1`：科学因素=RF; linear t; scale-shift；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d2_generation.strict_g1_rf_linear_scaleshift_r50.train；验收=only named factor changes; same batch/optimizer/selection; six test metrics；论文用途=Cross-cohort strict ablation replication；备注=D2 three-training-run supplement; do not enqueue until all required D1 work is complete.
- `D2.ABL.strict.G2`：科学因素=RF; shifted t; scale-shift；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d2_generation.strict_g2_rf_shifted_scaleshift_r50.train；验收=only named factor changes; same batch/optimizer/selection; six test metrics；论文用途=Cross-cohort strict ablation replication；备注=D2 three-training-run supplement; do not enqueue until all required D1 work is complete.
- `D2.ABL.strict.G3`：科学因素=RF; shifted t; AdaLN-Zero; Euler 1-step validation；解析配置=batch 2, 150 epochs, AdamW, lr=5e-05, config_id=v2.d2_generation.strict_g3_rf_shifted_adaln_zero_r50.train；验收=only time conditioning changes from G2; common validation protocol；论文用途=Cross-cohort strict one-factor generation ablation；备注=D2 three-training-run supplement; do not enqueue until all required D1 work is complete.
- `D2.INF.solver_steps.fixed1`：科学因素=one fixed KaryoFlow checkpoint x 3 solvers x 4 steps；验收=label fixed-checkpoint; not training replication；论文用途=Conditional solver mechanism evidence；备注=
- `D2.INF.solver_steps.train3`：科学因素=3 independent parents x 3 solvers x 4 steps；验收=aggregate one value per training parent；论文用途=Training-robust solver ablation；备注=D2 three-training-run supplement; do not enqueue until all required D1 work is complete.
- `D2.INF.topk_renewal.fixed1`：科学因素=K={100,150,200,300,500} x renewal={off,on}；验收=label fixed-checkpoint；论文用途=Conditional identity/candidate-budget evidence；备注=
- `D2.INF.topk_renewal.train3`：科学因素=3 parents x 5 K x renewal off/on；验收=aggregate by training parent；论文用途=Training-robust identity analysis；备注=D2 three-training-run supplement; do not enqueue until all required D1 work is complete.
#### Decision

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D2.SOTA.karyoflow_lqcr_train3` | karyoflow_lqcr | short_train / 3 | train=335778785,790448076,1342286018; infer=42 | **COMPLETED_VERIFIED** | EXACT | `experiments/configs/methods/karyoflow_ot_lqcr_legacy.py` | historical ross/workstation | `work_dirs/{capr_quality_only_24obj|paper_d2_lqcr_trainrun_*}` | `route:D2.SOTA.karyoflow_lqcr_train3; paired_lqcr_d2_train3;sota_d2_test` |
| `D2.DEC.beta_test.fixed1` | LQCR_beta_0_0.25_0.5_1_2 | inference / 5 | train=335778785; infer=42 | **COMPLETED_DIAGNOSTIC** | PROTOCOL_READY | `experiments/configs/methods/karyoflow_ot_legacy.py` | historical | `results/d2_test_inference_ablations/lqcr_beta/{beta}` | `route:D2.DEC.beta_test.fixed1; lqcr_beta_d2_test` |
| `D2.DEC.strict_subsets` | AP90_AP95_scale_overlap_bootstrap | analysis / 2 | train=335778785,790448076,1342286018; infer=42 | **COMPLETED_DIAGNOSTIC** | PROTOCOL_READY | `experiments/configs/methods/karyoflow_ot_legacy.py` | CPU | `tools/experiment_db/evidence_sources/d2_{difficult_subsets|lqcr_vs_dino}_*.json` | `route:D2.DEC.strict_subsets; conditional_difficult_subset_d2;conditional_image_bootstrap_d2` |
| `D2.DEC.quality_validity` | quality_iou_validity | analysis / 1 | train=42,123,789; infer=42 | **BLOCKED_PREDICTIONS** | PROTOCOL_TO_IMPLEMENT | `—` | CPU | `results/v2/d2/analysis/quality_iou_validity` | `route:D2.DEC.quality_validity; lqcr_quality_validity_d2_canonical` |
| `D2.ANALYSIS.per_class` | per_class_error | analysis / 1 | train=42,123,789; infer=42 | **BLOCKED_PREDICTIONS** | PROTOCOL_TO_IMPLEMENT | `—` | CPU | `results/v2/d2/analysis/per_class_error` | `route:D2.ANALYSIS.per_class; per_class_d2_canonical_test` |
| `D2.DEC.beta_val` | LQCR_beta_0_0.25_0.5_1_2 | inference / 15 | train=42,123,789; infer=42 | **REVIEW_EXISTING_VALIDATION_SELECTION** | PROTOCOL_READY_PARENT_PENDING | `experiments/configs/methods/karyoflow_lqcr.py` | accuracy:any idle GPU; efficiency=ross:A6000:0 only | `results/v2/d2/lqcr_beta_val/{parent}/beta_{beta}` | `route:D2.DEC.beta_val; lqcr_beta_d2_canonical_val` |

- `D2.SOTA.karyoflow_lqcr_train3`：科学因素=parent-matched final-only quality head; beta=2；验收=already verified: shared detector weights unchanged；论文用途=Original-split D2 paired LQCR effect；备注=Verified parent-matched three-training-run evidence on the publisher-provided original D2 split.
- `D2.DEC.beta_test.fixed1`：科学因素=test beta sweep after beta=2 validation selection; transparency only；验收=caption must state beta=2 selected on validation；论文用途=Post-selection sensitivity, not selection evidence；备注=
- `D2.DEC.strict_subsets`：科学因素=overlap/size strict recall plus 1000-replicate LQCR-vs-DINO image bootstrap；验收=diagnostic paper_eligible=0；论文用途=Mechanism/uncertainty; not SOTA proof；备注=
- `D2.DEC.quality_validity`：科学因素=quality-IoU Spearman, MAE/RMSE and reliability bins；验收=versioned analysis JSON bound to final test annotation/prediction SHA；论文用途=Mechanism/error analysis；备注=Required before the manuscript is refreshed.
- `D2.ANALYSIS.per_class`：科学因素=24-class AP and morphology-group error analysis；验收=versioned analysis JSON bound to final test annotation/prediction SHA；论文用途=Mechanism/error analysis；备注=Required before the manuscript is refreshed.
- `D2.DEC.beta_val`：科学因素=3 LQCR parents x beta={0,0.25,0.5,1,2}; validation only；验收=select beta before any new test evaluation；论文用途=Original-split D2 validation selection; never a test claim；备注=Recover and bind existing validation-selection evidence; no retraining is required.
#### Deployment

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `D2.DEP.distill_h3.existing` | H6_to_H3_single_parent | short_train / 1 | train=one; infer=42 | **COMPLETED_EVIDENCE_ONLY** | EXACT_INFERENCE_ARCHIVED_TRAINING | `experiments/configs/methods/karyoflow_ot_h3_student_legacy.py` | historical ross/workstation | `work_dirs/h3_distill_plan_a_24obj` | `route:D2.DEP.distill_h3.existing; head_distillation` |
| `D2.DEP.distill_h3.train3` | H6_to_H3_parent_matched_train3 | short_train / 3 | train=42,123,789; infer=42 | **DEFERRED_UNTIL_D1_COMPLETE** | READY | `experiments/configs/methods/karyoflow_h3_distill.py` | 42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1 | `work_dirs/v2/d2_distill_canonical/karyoflow_h3_distill_r50/trainseed_{training_seed}` | `route:D2.DEP.distill_h3.train3; head_distillation_d2_train3` |
| `D2.DEP.GACS` | GACS | inference / 1 | train=one/unknown; infer=42 | **PARTIAL_LEGACY** | PROTOCOL_READY | `experiments/configs/methods/karyoflow_ot_legacy.py` | accuracy:any; latency=ross:A6000:0 | `results/v2/d2_taichung/deployment/gacs/{parent}` | `route:D2.DEP.GACS; GACS` |
| `D2.DEP.speed` | all_operating_points | benchmark / 1 | train=mixed; infer=none | **COMPLETED_PARTIAL** | PROTOCOL_READY | `experiments/configs/methods/karyoflow_ot_legacy.py` | ross:A6000:0 exclusive | `results/benchmark/a6000/{variant}` | `route:D2.DEP.speed; speed_accuracy` |

- `D2.DEP.distill_h3.existing`：科学因素=H6 teacher -> H3 student; mapping 0,2,5；验收=do not call three-training-seed；论文用途=Conditional deployment point；备注=Inference identity is EXACT; the archived historical distillation training implementation is not executable in cleaned ldmdet.
- `D2.DEP.distill_h3.train3`：科学因素=one H3 student per KaryoFlow parent; identical mapping/loss；解析配置=batch 2, 50 epochs, AdamW, lr=1e-05, config_id=v2.d2_distill_canonical.karyoflow_h3_distill_r50.train；验收=three parent-matched students; A6000 latency only；论文用途=Training-robust deployment evidence；备注=D2 three-training-run supplement; do not enqueue until all required D1 work is complete.
- `D2.DEP.GACS`：科学因素=dynamic deployment policy; historical D1 loss ~0.0007 mAP; D2 held-out test missing；验收=D2 test plus A6000 speed before paper use；论文用途=Optional deployment extension, not main precision module；备注=
- `D2.DEP.speed`：科学因素=A6000: RF-Heun; DPM++ K=100/200/300/500; LQCR; H3; repeat after final checkpoints；验收=repeat final selected points with exclusive GPU and frozen timing boundary；论文用途=Original-split D2 speed-accuracy evidence；备注=Use original-split accuracy and the unified A6000 latency protocol; no repaired-split rerun.

### D1_COMPOSITE2200_LEGACY

#### Archive

| ID | Variant | Kind / runs | Seeds | Status | Config state | v2 method / protocol | Server | Output | DB/evidence |
|---|---|---:|---|---|---|---|---|---|---|
| `LEGACY.D1.composite` | all_historical_results | mixed / 1 | train=mixed; infer=mixed | **ARCHIVED_NONCOMPARABLE** | ARCHIVED | `tools/experiment_db/experiments.db` | historical ross/workstation | `work_dirs/*; results/*` | `route:LEGACY.D1.composite; all D1 families except D1_INHOUSE1700_V2` |

- `LEGACY.D1.composite`：科学因素=old 2200-image composite contains 500 D2-derived images and old split; results remain auditable；验收=must be labeled legacy and non-comparable；论文用途=Provenance/history only; never pool with new D1I or D2；备注=

## 当前执行结论

- D1_INHOUSE1700_V2 的三条 canonical KaryoFlow 独立训练及 held-out test 已完成；三条 parent-matched LQCR 已完成训练，等待 held-out test 和 final-only tensor audit。
- D2 使用作者原始公开划分；已完成证据直接作为主结果，确需补充的三训练种子实验统一延后到全部 D1 工作完成后，且未进入当前调度队列。
- D1 严格 G0/G1 三训练种子已完成；G2 已完成两条且一条运行中；G3 已完成一条且两条依赖排队。
- H3 推理身份可精确复现，但历史蒸馏训练实现仍需恢复；GACS 保持可选部署扩展。
