# Paper evidence register

> Auto-generated from `tools/experiment_db/experiments.db`. Do not edit
> numerical values here; update the manifest/database and regenerate.

## Paper-eligible controlled results

| ID | Family / variant | Data | Metric | Value | Delta | Source |
|---|---|---|---:|---:|---:|---|
| `gacs-d1-adaptive-map` | GACS / adaptive_1_or_2_renewal_on | D1 val, mean(42,123,789) | mAP (absolute) | 0.7383 | -0.0007 | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `gacs-d1-fixed2-map` | GACS / fixed2_renewal_on | D1 val, mean(42,123,789) | mAP (absolute) | 0.739 |  | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `gacs-d1-fourstep-reference` | GACS / fixed4_reference | D1 val, mean(42,123,789) | mAP (absolute) | 0.747 |  | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `gacs-d1-latency-reduction` | GACS / adaptive_1_or_2_renewal_on | D1 benchmark, mean(42,123,789) | latency_reduction (percent) | 13.79 |  | `ross:work_dirs/diagnosis/gacs_chr2024_summary.json` |
| `lqcr-d1-test-aps` | LQCR / final_only_clean | D1 test, mean(42,123,789) | AP_S (absolute) | 0.5147 | 0.0057 | `ross+workstation:tools/experiment_db/evidence_sources/lqcr_cross_dataset_20260811.json` |
| `lqcr-d1-test-map` | LQCR / final_only_clean | D1 test, mean(42,123,789) | mAP (absolute) | 0.748 | 0.0067 | `ross+workstation:tools/experiment_db/evidence_sources/lqcr_cross_dataset_20260811.json` |
| `lqcr-d1-val-aps` | LQCR / final_only_clean | D1 val, mean(42,123,789) | AP_S (absolute) | 0.5187 |  | `ross+workstation:tools/experiment_db/evidence_sources/lqcr_cross_dataset_20260811.json` |
| `lqcr-d1-val-map` | LQCR / final_only_clean | D1 val, mean(42,123,789) | mAP (absolute) | 0.755 | 0.008 | `ross+workstation:tools/experiment_db/evidence_sources/lqcr_cross_dataset_20260811.json` |
| `lqcr-d2-ap90-gain` | LQCR / final_only_beta2 | D2 val, 42 | AP90_delta (absolute_delta) | 0.03041 |  | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `lqcr-d2-ap95-gain` | LQCR / final_only_beta2 | D2 val, 42 | AP95_delta (absolute_delta) | 0.03251 |  | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `lqcr-d2-baseline-map` | LQCR / disabled | D2 val, 42 | mAP (absolute) | 0.8630128 |  | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `lqcr-d2-beta0-map` | LQCR / learned_q_beta0 | D2 val, 42 | mAP (absolute) | 0.8630128 |  | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta025-map` | LQCR / learned_q_beta0.25 | D2 val, 42 | mAP (absolute) | 0.8657292 | 0.002716436 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta05-map` | LQCR / learned_q_beta0.5 | D2 val, 42 | mAP (absolute) | 0.8673472 | 0.004334369 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta1-map` | LQCR / learned_q_beta1 | D2 val, 42 | mAP (absolute) | 0.8689229 | 0.005910119 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-beta2-map` | LQCR / learned_q_beta2 | D2 val, 42 | mAP (absolute) | 0.8704443 | 0.007431458 | `ross:tools/experiment_db/evidence_sources/lqcr_learned_beta_sweep_seed42.json` |
| `lqcr-d2-enabled-map` | LQCR / final_only_beta2 | D2 val, 42 | mAP (absolute) | 0.8704443 | 0.0074315 | `ross:docs/paper/latex/figures/v2/data/source_lqcr_final_only_seed42.json` |
| `annotation-d2-sigma0px_flip0-ap50` | annotation_perturbation / sigma0px_flip0 | D2 test, 42 | AP50 (absolute) | 0.988 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma0px_flip0-ap75` | annotation_perturbation / sigma0px_flip0 | D2 test, 42 | AP75 (absolute) | 0.971 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma0px_flip0-apl` | annotation_perturbation / sigma0px_flip0 | D2 test, 42 | APl (absolute) | 0.914 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma0px_flip0-apm` | annotation_perturbation / sigma0px_flip0 | D2 test, 42 | APm (absolute) | 0.856 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma0px_flip0-aps` | annotation_perturbation / sigma0px_flip0 | D2 test, 42 | APs (absolute) | 0.576 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma0px_flip0-map` | annotation_perturbation / sigma0px_flip0 | D2 test, 42 | mAP (absolute) | 0.859 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.05-ap50` | annotation_perturbation / sigma10px_flip0.05 | D2 test, 42 | AP50 (absolute) | 0.529 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.05-ap75` | annotation_perturbation / sigma10px_flip0.05 | D2 test, 42 | AP75 (absolute) | 0.081 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.05-apl` | annotation_perturbation / sigma10px_flip0.05 | D2 test, 42 | APl (absolute) | 0.199 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.05-apm` | annotation_perturbation / sigma10px_flip0.05 | D2 test, 42 | APm (absolute) | 0.163 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.05-aps` | annotation_perturbation / sigma10px_flip0.05 | D2 test, 42 | APs (absolute) | 0.026 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.05-map` | annotation_perturbation / sigma10px_flip0.05 | D2 test, 42 | mAP (absolute) | 0.179 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.1-ap50` | annotation_perturbation / sigma10px_flip0.1 | D2 test, 42 | AP50 (absolute) | 0.477 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.1-ap75` | annotation_perturbation / sigma10px_flip0.1 | D2 test, 42 | AP75 (absolute) | 0.073 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.1-apl` | annotation_perturbation / sigma10px_flip0.1 | D2 test, 42 | APl (absolute) | 0.17 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.1-apm` | annotation_perturbation / sigma10px_flip0.1 | D2 test, 42 | APm (absolute) | 0.146 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.1-aps` | annotation_perturbation / sigma10px_flip0.1 | D2 test, 42 | APs (absolute) | 0.016 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.1-map` | annotation_perturbation / sigma10px_flip0.1 | D2 test, 42 | mAP (absolute) | 0.162 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.2-ap50` | annotation_perturbation / sigma10px_flip0.2 | D2 test, 42 | AP50 (absolute) | 0.382 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.2-ap75` | annotation_perturbation / sigma10px_flip0.2 | D2 test, 42 | AP75 (absolute) | 0.059 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.2-apl` | annotation_perturbation / sigma10px_flip0.2 | D2 test, 42 | APl (absolute) | 0.133 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.2-apm` | annotation_perturbation / sigma10px_flip0.2 | D2 test, 42 | APm (absolute) | 0.115 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.2-aps` | annotation_perturbation / sigma10px_flip0.2 | D2 test, 42 | APs (absolute) | 0.014 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma10px_flip0.2-map` | annotation_perturbation / sigma10px_flip0.2 | D2 test, 42 | mAP (absolute) | 0.129 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.05-ap50` | annotation_perturbation / sigma2px_flip0.05 | D2 test, 42 | AP50 (absolute) | 0.894 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.05-ap75` | annotation_perturbation / sigma2px_flip0.05 | D2 test, 42 | AP75 (absolute) | 0.812 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.05-apl` | annotation_perturbation / sigma2px_flip0.05 | D2 test, 42 | APl (absolute) | 0.401 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.05-apm` | annotation_perturbation / sigma2px_flip0.05 | D2 test, 42 | APm (absolute) | 0.657 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.05-aps` | annotation_perturbation / sigma2px_flip0.05 | D2 test, 42 | APs (absolute) | 0.206 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.05-map` | annotation_perturbation / sigma2px_flip0.05 | D2 test, 42 | mAP (absolute) | 0.666 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.1-ap50` | annotation_perturbation / sigma2px_flip0.1 | D2 test, 42 | AP50 (absolute) | 0.804 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.1-ap75` | annotation_perturbation / sigma2px_flip0.1 | D2 test, 42 | AP75 (absolute) | 0.73 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.1-apl` | annotation_perturbation / sigma2px_flip0.1 | D2 test, 42 | APl (absolute) | 0.339 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.1-apm` | annotation_perturbation / sigma2px_flip0.1 | D2 test, 42 | APm (absolute) | 0.589 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.1-aps` | annotation_perturbation / sigma2px_flip0.1 | D2 test, 42 | APs (absolute) | 0.127 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.1-map` | annotation_perturbation / sigma2px_flip0.1 | D2 test, 42 | mAP (absolute) | 0.599 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.2-ap50` | annotation_perturbation / sigma2px_flip0.2 | D2 test, 42 | AP50 (absolute) | 0.641 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.2-ap75` | annotation_perturbation / sigma2px_flip0.2 | D2 test, 42 | AP75 (absolute) | 0.58 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.2-apl` | annotation_perturbation / sigma2px_flip0.2 | D2 test, 42 | APl (absolute) | 0.266 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.2-apm` | annotation_perturbation / sigma2px_flip0.2 | D2 test, 42 | APm (absolute) | 0.465 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.2-aps` | annotation_perturbation / sigma2px_flip0.2 | D2 test, 42 | APs (absolute) | 0.088 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma2px_flip0.2-map` | annotation_perturbation / sigma2px_flip0.2 | D2 test, 42 | mAP (absolute) | 0.477 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.05-ap50` | annotation_perturbation / sigma5px_flip0.05 | D2 test, 42 | AP50 (absolute) | 0.847 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.05-ap75` | annotation_perturbation / sigma5px_flip0.05 | D2 test, 42 | AP75 (absolute) | 0.375 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.05-apl` | annotation_perturbation / sigma5px_flip0.05 | D2 test, 42 | APl (absolute) | 0.321 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.05-apm` | annotation_perturbation / sigma5px_flip0.05 | D2 test, 42 | APm (absolute) | 0.414 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.05-aps` | annotation_perturbation / sigma5px_flip0.05 | D2 test, 42 | APs (absolute) | 0.095 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.05-map` | annotation_perturbation / sigma5px_flip0.05 | D2 test, 42 | mAP (absolute) | 0.428 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.1-ap50` | annotation_perturbation / sigma5px_flip0.1 | D2 test, 42 | AP50 (absolute) | 0.763 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.1-ap75` | annotation_perturbation / sigma5px_flip0.1 | D2 test, 42 | AP75 (absolute) | 0.338 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.1-apl` | annotation_perturbation / sigma5px_flip0.1 | D2 test, 42 | APl (absolute) | 0.272 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.1-apm` | annotation_perturbation / sigma5px_flip0.1 | D2 test, 42 | APm (absolute) | 0.371 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.1-aps` | annotation_perturbation / sigma5px_flip0.1 | D2 test, 42 | APs (absolute) | 0.062 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.1-map` | annotation_perturbation / sigma5px_flip0.1 | D2 test, 42 | mAP (absolute) | 0.386 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.2-ap50` | annotation_perturbation / sigma5px_flip0.2 | D2 test, 42 | AP50 (absolute) | 0.608 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.2-ap75` | annotation_perturbation / sigma5px_flip0.2 | D2 test, 42 | AP75 (absolute) | 0.27 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.2-apl` | annotation_perturbation / sigma5px_flip0.2 | D2 test, 42 | APl (absolute) | 0.213 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.2-apm` | annotation_perturbation / sigma5px_flip0.2 | D2 test, 42 | APm (absolute) | 0.293 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.2-aps` | annotation_perturbation / sigma5px_flip0.2 | D2 test, 42 | APs (absolute) | 0.043 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `annotation-d2-sigma5px_flip0.2-map` | annotation_perturbation / sigma5px_flip0.2 | D2 test, 42 | mAP (absolute) | 0.308 |  | `ross:tools/experiment_db/evidence_sources/annotation_perturbation_d2_test.json` |
| `clean-a4-d1-test-aps` | clean_random_mainline / RF_DPMpp_random | D1 test, mean(42,123,789) | AP_S (absolute) | 0.509 |  | `ross:tools/experiment_db/evidence_sources/scale_and_clean_test_20260811.json` |
| `clean-a4-d1-test-map` | clean_random_mainline / RF_DPMpp_random | D1 test, mean(42,123,789) | mAP (absolute) | 0.7413 |  | `ross:tools/experiment_db/evidence_sources/scale_and_clean_test_20260811.json` |
| `dataset-d1-test-annotations` | dataset_inventory / annotation_file | D1 test, none | annotations (count) | 10262 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-test-classes` | dataset_inventory / annotation_file | D1 test, none | classes (count) | 24 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-test-images` | dataset_inventory / annotation_file | D1 test, none | images (count) | 220 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-train-annotations` | dataset_inventory / annotation_file | D1 train, none | annotations (count) | 72023 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-train-classes` | dataset_inventory / annotation_file | D1 train, none | classes (count) | 24 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-train-images` | dataset_inventory / annotation_file | D1 train, none | images (count) | 1540 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-val-annotations` | dataset_inventory / annotation_file | D1 val, none | annotations (count) | 20574 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-val-classes` | dataset_inventory / annotation_file | D1 val, none | classes (count) | 24 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d1-val-images` | dataset_inventory / annotation_file | D1 val, none | images (count) | 440 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-test-annotations` | dataset_inventory / annotation_file | D2 test, none | annotations (count) | 45980 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-test-classes` | dataset_inventory / annotation_file | D2 test, none | classes (count) | 24 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-test-images` | dataset_inventory / annotation_file | D2 test, none | images (count) | 1000 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-train-annotations` | dataset_inventory / annotation_file | D2 train, none | annotations (count) | 160888 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-train-classes` | dataset_inventory / annotation_file | D2 train, none | classes (count) | 24 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-train-images` | dataset_inventory / annotation_file | D2 train, none | images (count) | 3500 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-val-annotations` | dataset_inventory / annotation_file | D2 val, none | annotations (count) | 22984 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-val-classes` | dataset_inventory / annotation_file | D2 val, none | classes (count) | 24 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `dataset-d2-val-images` | dataset_inventory / annotation_file | D2 val, none | images (count) | 500 |  | `ross:tools/experiment_db/evidence_sources/dataset_splits_20260812.json` |
| `scale-d1-small-instance` | dataset_scale / train | D1 train, none | small_instance_percent (percent) | 23.0329 |  | `ross:tools/experiment_db/evidence_sources/scale_and_clean_test_20260811.json` |
| `scale-d2-small-instance` | dataset_scale / train | D2 train, none | small_instance_percent (percent) | 0.3748 |  | `ross:tools/experiment_db/evidence_sources/scale_and_clean_test_20260811.json` |
| `head-h3s4-d1-map` | head_compression_control / H3S4_no_distill | D1 val, 42 | mAP (absolute) | 0.746 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `distill-d2-student-latency` | head_distillation / H3_student | D2 benchmark, 42 | latency (ms) | 44.0959 | -31.1011 | `ross:results/audit_20260811/h3_same_protocol/benchmark_fps_20260811_164416.json` |
| `distill-d2-student-map` | head_distillation / H3_student | D2 val, 42 | mAP (absolute) | 0.859 | -0.004 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `distill-d2-teacher-latency` | head_distillation / H6_teacher | D2 benchmark, 42 | latency (ms) | 75.197 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `distill-d2-teacher-map` | head_distillation / H6_teacher | D2 val, 42 | mAP (absolute) | 0.863 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `d2-test-lqcr-beta-beta_0p0-ap50-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | AP50 (absolute) | 0.9881545 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p0-ap75-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | AP75 (absolute) | 0.9708508 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p0-ap90-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | AP90 (absolute) | 0.6787767 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p0-ap95-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | AP95 (absolute) | 0.2045899 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p0-ap_l-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | AP_L (absolute) | 0.8772192 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p0-ap_m-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | AP_M (absolute) | 0.8569901 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p0-ap_s-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | AP_S (absolute) | 0.5722398 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p0-map-21e73b4b6a56` | lqcr_beta_d2_test / LQCR beta=0 | D2 test, 42 | mAP (absolute) | 0.8595844 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-ap50-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | AP50 (absolute) | 0.9881505 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-ap75-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | AP75 (absolute) | 0.9710211 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-ap90-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | AP90 (absolute) | 0.687833 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-ap95-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | AP95 (absolute) | 0.2149043 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-ap_l-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | AP_L (absolute) | 0.8855084 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-ap_m-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | AP_M (absolute) | 0.8593057 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-ap_s-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | AP_S (absolute) | 0.5716585 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p25-map-e2bb3c31e79e` | lqcr_beta_d2_test / LQCR beta=0.25 | D2 test, 42 | mAP (absolute) | 0.8620698 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-ap50-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | AP50 (absolute) | 0.9881511 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-ap75-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | AP75 (absolute) | 0.9713009 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-ap90-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | AP90 (absolute) | 0.6939249 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-ap95-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | AP95 (absolute) | 0.2215359 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-ap_l-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | AP_L (absolute) | 0.8894409 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-ap_m-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | AP_M (absolute) | 0.8608903 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-ap_s-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | AP_S (absolute) | 0.5731709 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_0p5-map-66591b3de63d` | lqcr_beta_d2_test / LQCR beta=0.5 | D2 test, 42 | mAP (absolute) | 0.8636428 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-ap50-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | AP50 (absolute) | 0.9881976 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-ap75-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | AP75 (absolute) | 0.9710454 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-ap90-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | AP90 (absolute) | 0.6998607 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-ap95-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | AP95 (absolute) | 0.2283046 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-ap_l-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | AP_L (absolute) | 0.8866602 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-ap_m-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | AP_M (absolute) | 0.8622232 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-ap_s-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | AP_S (absolute) | 0.5667435 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_1p0-map-e8be07b11a20` | lqcr_beta_d2_test / LQCR beta=1 | D2 test, 42 | mAP (absolute) | 0.8650795 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-ap50-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | AP50 (absolute) | 0.9880983 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-ap75-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | AP75 (absolute) | 0.9709532 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-ap90-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | AP90 (absolute) | 0.7045906 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-ap95-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | AP95 (absolute) | 0.2340442 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-ap_l-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | AP_L (absolute) | 0.8868893 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-ap_m-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | AP_M (absolute) | 0.8635979 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-ap_s-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | AP_S (absolute) | 0.5690838 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `d2-test-lqcr-beta-beta_2p0-map-a36b680708c5` | lqcr_beta_d2_test / LQCR beta=2 | D2 test, 42 | mAP (absolute) | 0.8663381 |  | `ross:tools/experiment_db/evidence_sources/d2_test_inference_ablations_d9fd109403da.json` |
| `vpred-d1-val` | prediction_parameterization / v | D1 val, mean(42,123,789) | mAP (absolute) | 0.745 | -0.002 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `x0-d1-val` | prediction_parameterization / x0 | D1 val, mean(42,123,789) | mAP (absolute) | 0.747 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `vpred-d2-val` | prediction_parameterization / v | D2 val, mean(42,123,789) | mAP (absolute) | 0.857 | -0.002 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `x0-d2-val` | prediction_parameterization / x0 | D2 val, mean(42,123,789) | mAP (absolute) | 0.859 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-d1-off-map` | renewal_consistency / renewal_off_K500 | D1 val, mean(42,123,789) | mAP (absolute) | 0.7433333 | -0.003 | `ross:work_dirs/diagnosis/d1_topk_validation_seed42.json` |
| `renew-d1-on-map` | renewal_consistency / renewal_on_K500 | D1 val, mean(42,123,789) | mAP (absolute) | 0.7463333 |  | `ross:work_dirs/diagnosis/d1_topk_validation_seed42.json` |
| `renew-d2-off-map` | renewal_consistency / renewal_off_K500 | D2 val, mean(42,123,789) | mAP (absolute) | 0.8583333 | -0.0003334 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-d2-on-map` | renewal_consistency / renewal_on_K500 | D2 val, mean(42,123,789) | mAP (absolute) | 0.8586667 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-k100-off-map` | renewal_consistency / renewal_off_K100 | D2 val, mean(42,123,789) | mAP (absolute) | 0.808 | -0.0313 | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-k100-on-map` | renewal_consistency / renewal_on_K100 | D2 val, mean(42,123,789) | mAP (absolute) | 0.8393 |  | `ross:docs/EXPERIMENT_LINEAGE.md` |
| `renew-latency-saving` | renewal_consistency / renewal_off_K500 | D2 benchmark, mean(42,123,789) | latency_reduction (percent) | 2.427621 |  | `ross:work_dirs/diagnosis/renewal_latency_3seed.json` |
| `d1-rf-ddpm-ddpm-mean-map` | rf_ddpm_d1_test / DDPM | D1 test, mean(42,123,789) | mAP (absolute) | 0.7186667 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d1-rf-ddpm-ddpm-seed123-map` | rf_ddpm_d1_test / DDPM | D1 test, 123 | mAP (absolute) | 0.722 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d1-rf-ddpm-ddpm-seed42-map` | rf_ddpm_d1_test / DDPM | D1 test, 42 | mAP (absolute) | 0.716 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d1-rf-ddpm-ddpm-seed789-map` | rf_ddpm_d1_test / DDPM | D1 test, 789 | mAP (absolute) | 0.718 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d1-rf-ddpm-rf_heun-mean-map` | rf_ddpm_d1_test / RF_Heun | D1 test, mean(42,123,789) | mAP (absolute) | 0.7366667 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d1-rf-ddpm-rf_heun-seed123-map` | rf_ddpm_d1_test / RF_Heun | D1 test, 123 | mAP (absolute) | 0.735 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d1-rf-ddpm-rf_heun-seed42-map` | rf_ddpm_d1_test / RF_Heun | D1 test, 42 | mAP (absolute) | 0.737 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d1-rf-ddpm-rf_heun-seed789-map` | rf_ddpm_d1_test / RF_Heun | D1 test, 789 | mAP (absolute) | 0.738 |  | `ross:tools/experiment_db/evidence_sources/d1_rf_ddpm_test_20260730.json` |
| `d2-test-solver-dpm_solver_pp_1-ap50-a999c9bf1f51` | solver_d2_test / Dpm Solver Pp / 1 step | D2 test, 42 | AP50 (absolute) | 0.984712 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_1-ap75-a999c9bf1f51` | solver_d2_test / Dpm Solver Pp / 1 step | D2 test, 42 | AP75 (absolute) | 0.9633645 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_1-ap_l-a999c9bf1f51` | solver_d2_test / Dpm Solver Pp / 1 step | D2 test, 42 | AP_L (absolute) | 0.8419959 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_1-ap_m-a999c9bf1f51` | solver_d2_test / Dpm Solver Pp / 1 step | D2 test, 42 | AP_M (absolute) | 0.8486545 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_1-ap_s-a999c9bf1f51` | solver_d2_test / Dpm Solver Pp / 1 step | D2 test, 42 | AP_S (absolute) | 0.584927 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_1-map-a999c9bf1f51` | solver_d2_test / Dpm Solver Pp / 1 step | D2 test, 42 | mAP (absolute) | 0.8513168 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_2-ap50-1b381fa35049` | solver_d2_test / Dpm Solver Pp / 2 step | D2 test, 42 | AP50 (absolute) | 0.9876872 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_2-ap75-1b381fa35049` | solver_d2_test / Dpm Solver Pp / 2 step | D2 test, 42 | AP75 (absolute) | 0.9676527 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_2-ap_l-1b381fa35049` | solver_d2_test / Dpm Solver Pp / 2 step | D2 test, 42 | AP_L (absolute) | 0.847822 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_2-ap_m-1b381fa35049` | solver_d2_test / Dpm Solver Pp / 2 step | D2 test, 42 | AP_M (absolute) | 0.8533123 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_2-ap_s-1b381fa35049` | solver_d2_test / Dpm Solver Pp / 2 step | D2 test, 42 | AP_S (absolute) | 0.5814524 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_2-map-1b381fa35049` | solver_d2_test / Dpm Solver Pp / 2 step | D2 test, 42 | mAP (absolute) | 0.8555733 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_3-ap50-77d2752edea3` | solver_d2_test / Dpm Solver Pp / 3 step | D2 test, 42 | AP50 (absolute) | 0.987776 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_3-ap75-77d2752edea3` | solver_d2_test / Dpm Solver Pp / 3 step | D2 test, 42 | AP75 (absolute) | 0.9689515 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_3-ap_l-77d2752edea3` | solver_d2_test / Dpm Solver Pp / 3 step | D2 test, 42 | AP_L (absolute) | 0.851275 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_3-ap_m-77d2752edea3` | solver_d2_test / Dpm Solver Pp / 3 step | D2 test, 42 | AP_M (absolute) | 0.8546984 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_3-ap_s-77d2752edea3` | solver_d2_test / Dpm Solver Pp / 3 step | D2 test, 42 | AP_S (absolute) | 0.5754845 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_3-map-77d2752edea3` | solver_d2_test / Dpm Solver Pp / 3 step | D2 test, 42 | mAP (absolute) | 0.8571874 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_4-ap50-72fb4e071281` | solver_d2_test / Dpm Solver Pp / 4 step | D2 test, 42 | AP50 (absolute) | 0.9878089 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_4-ap75-72fb4e071281` | solver_d2_test / Dpm Solver Pp / 4 step | D2 test, 42 | AP75 (absolute) | 0.9705674 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_4-ap_l-72fb4e071281` | solver_d2_test / Dpm Solver Pp / 4 step | D2 test, 42 | AP_L (absolute) | 0.8474451 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_4-ap_m-72fb4e071281` | solver_d2_test / Dpm Solver Pp / 4 step | D2 test, 42 | AP_M (absolute) | 0.8544972 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_4-ap_s-72fb4e071281` | solver_d2_test / Dpm Solver Pp / 4 step | D2 test, 42 | AP_S (absolute) | 0.5843465 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-dpm_solver_pp_4-map-72fb4e071281` | solver_d2_test / Dpm Solver Pp / 4 step | D2 test, 42 | mAP (absolute) | 0.8570178 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_1-ap50-5d5a1e24b003` | solver_d2_test / Euler / 1 step | D2 test, 42 | AP50 (absolute) | 0.9847892 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_1-ap75-5d5a1e24b003` | solver_d2_test / Euler / 1 step | D2 test, 42 | AP75 (absolute) | 0.9633627 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_1-ap_l-5d5a1e24b003` | solver_d2_test / Euler / 1 step | D2 test, 42 | AP_L (absolute) | 0.8419911 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_1-ap_m-5d5a1e24b003` | solver_d2_test / Euler / 1 step | D2 test, 42 | AP_M (absolute) | 0.8486891 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_1-ap_s-5d5a1e24b003` | solver_d2_test / Euler / 1 step | D2 test, 42 | AP_S (absolute) | 0.5849267 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_1-map-5d5a1e24b003` | solver_d2_test / Euler / 1 step | D2 test, 42 | mAP (absolute) | 0.851351 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_2-ap50-5eef491db129` | solver_d2_test / Euler / 2 step | D2 test, 42 | AP50 (absolute) | 0.9876268 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_2-ap75-5eef491db129` | solver_d2_test / Euler / 2 step | D2 test, 42 | AP75 (absolute) | 0.9676195 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_2-ap_l-5eef491db129` | solver_d2_test / Euler / 2 step | D2 test, 42 | AP_L (absolute) | 0.8541813 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_2-ap_m-5eef491db129` | solver_d2_test / Euler / 2 step | D2 test, 42 | AP_M (absolute) | 0.85248 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_2-ap_s-5eef491db129` | solver_d2_test / Euler / 2 step | D2 test, 42 | AP_S (absolute) | 0.5811545 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_2-map-5eef491db129` | solver_d2_test / Euler / 2 step | D2 test, 42 | mAP (absolute) | 0.8553129 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_3-ap50-bb70c2ec616d` | solver_d2_test / Euler / 3 step | D2 test, 42 | AP50 (absolute) | 0.9876829 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_3-ap75-bb70c2ec616d` | solver_d2_test / Euler / 3 step | D2 test, 42 | AP75 (absolute) | 0.9691407 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_3-ap_l-bb70c2ec616d` | solver_d2_test / Euler / 3 step | D2 test, 42 | AP_L (absolute) | 0.8567937 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_3-ap_m-bb70c2ec616d` | solver_d2_test / Euler / 3 step | D2 test, 42 | AP_M (absolute) | 0.8540099 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_3-ap_s-bb70c2ec616d` | solver_d2_test / Euler / 3 step | D2 test, 42 | AP_S (absolute) | 0.5745903 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_3-map-bb70c2ec616d` | solver_d2_test / Euler / 3 step | D2 test, 42 | mAP (absolute) | 0.8567034 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_4-ap50-f050647969ac` | solver_d2_test / Euler / 4 step | D2 test, 42 | AP50 (absolute) | 0.987772 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_4-ap75-f050647969ac` | solver_d2_test / Euler / 4 step | D2 test, 42 | AP75 (absolute) | 0.9689509 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_4-ap_l-f050647969ac` | solver_d2_test / Euler / 4 step | D2 test, 42 | AP_L (absolute) | 0.8572918 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_4-ap_m-f050647969ac` | solver_d2_test / Euler / 4 step | D2 test, 42 | AP_M (absolute) | 0.8538728 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_4-ap_s-f050647969ac` | solver_d2_test / Euler / 4 step | D2 test, 42 | AP_S (absolute) | 0.5862603 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-euler_4-map-f050647969ac` | solver_d2_test / Euler / 4 step | D2 test, 42 | mAP (absolute) | 0.8564711 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_1-ap50-88ee026d5e2e` | solver_d2_test / Heun / 1 step | D2 test, 42 | AP50 (absolute) | 0.984712 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_1-ap75-88ee026d5e2e` | solver_d2_test / Heun / 1 step | D2 test, 42 | AP75 (absolute) | 0.9633576 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_1-ap_l-88ee026d5e2e` | solver_d2_test / Heun / 1 step | D2 test, 42 | AP_L (absolute) | 0.8420354 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_1-ap_m-88ee026d5e2e` | solver_d2_test / Heun / 1 step | D2 test, 42 | AP_M (absolute) | 0.8486532 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_1-ap_s-88ee026d5e2e` | solver_d2_test / Heun / 1 step | D2 test, 42 | AP_S (absolute) | 0.5849268 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_1-map-88ee026d5e2e` | solver_d2_test / Heun / 1 step | D2 test, 42 | mAP (absolute) | 0.85134 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_2-ap50-6990e32e05d7` | solver_d2_test / Heun / 2 step | D2 test, 42 | AP50 (absolute) | 0.9876689 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_2-ap75-6990e32e05d7` | solver_d2_test / Heun / 2 step | D2 test, 42 | AP75 (absolute) | 0.9674338 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_2-ap_l-6990e32e05d7` | solver_d2_test / Heun / 2 step | D2 test, 42 | AP_L (absolute) | 0.8485361 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_2-ap_m-6990e32e05d7` | solver_d2_test / Heun / 2 step | D2 test, 42 | AP_M (absolute) | 0.8525266 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_2-ap_s-6990e32e05d7` | solver_d2_test / Heun / 2 step | D2 test, 42 | AP_S (absolute) | 0.5859324 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_2-map-6990e32e05d7` | solver_d2_test / Heun / 2 step | D2 test, 42 | mAP (absolute) | 0.8552636 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_3-ap50-b4fde98d9011` | solver_d2_test / Heun / 3 step | D2 test, 42 | AP50 (absolute) | 0.9877157 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_3-ap75-b4fde98d9011` | solver_d2_test / Heun / 3 step | D2 test, 42 | AP75 (absolute) | 0.9696402 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_3-ap_l-b4fde98d9011` | solver_d2_test / Heun / 3 step | D2 test, 42 | AP_L (absolute) | 0.8556143 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_3-ap_m-b4fde98d9011` | solver_d2_test / Heun / 3 step | D2 test, 42 | AP_M (absolute) | 0.8537963 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_3-ap_s-b4fde98d9011` | solver_d2_test / Heun / 3 step | D2 test, 42 | AP_S (absolute) | 0.5759824 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_3-map-b4fde98d9011` | solver_d2_test / Heun / 3 step | D2 test, 42 | mAP (absolute) | 0.8563458 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_4-ap50-724befe74b35` | solver_d2_test / Heun / 4 step | D2 test, 42 | AP50 (absolute) | 0.9877898 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_4-ap75-724befe74b35` | solver_d2_test / Heun / 4 step | D2 test, 42 | AP75 (absolute) | 0.9690752 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_4-ap_l-724befe74b35` | solver_d2_test / Heun / 4 step | D2 test, 42 | AP_L (absolute) | 0.8489307 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_4-ap_m-724befe74b35` | solver_d2_test / Heun / 4 step | D2 test, 42 | AP_M (absolute) | 0.8540109 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_4-ap_s-724befe74b35` | solver_d2_test / Heun / 4 step | D2 test, 42 | AP_S (absolute) | 0.5839502 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d2-test-solver-heun_4-map-724befe74b35` | solver_d2_test / Heun / 4 step | D2 test, 42 | mAP (absolute) | 0.8565748 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_b81d6a3d498a.json` |
| `d1-test-cascade_rcnn-ap50-8f8d14d201de` | sota_d1_test / Cascade R-CNN | D1 test, 42 | AP50 (absolute) | 0.922127 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-cascade_rcnn-ap75-8f8d14d201de` | sota_d1_test / Cascade R-CNN | D1 test, 42 | AP75 (absolute) | 0.8307158 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-cascade_rcnn-ap_l-8f8d14d201de` | sota_d1_test / Cascade R-CNN | D1 test, 42 | AP_L (absolute) | 0.648822 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-cascade_rcnn-ap_m-8f8d14d201de` | sota_d1_test / Cascade R-CNN | D1 test, 42 | AP_M (absolute) | 0.7087721 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-cascade_rcnn-ap_s-8f8d14d201de` | sota_d1_test / Cascade R-CNN | D1 test, 42 | AP_S (absolute) | 0.500453 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-cascade_rcnn-map-8f8d14d201de` | sota_d1_test / Cascade R-CNN | D1 test, 42 | mAP (absolute) | 0.7235469 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-diffusiondet-ap50-0ec6c5e334d1` | sota_d1_test / DiffusionDet | D1 test, 42 | AP50 (absolute) | 0.9068407 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-diffusiondet-ap75-0ec6c5e334d1` | sota_d1_test / DiffusionDet | D1 test, 42 | AP75 (absolute) | 0.796471 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-diffusiondet-ap_l-0ec6c5e334d1` | sota_d1_test / DiffusionDet | D1 test, 42 | AP_L (absolute) | 0.6361783 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-diffusiondet-ap_m-0ec6c5e334d1` | sota_d1_test / DiffusionDet | D1 test, 42 | AP_M (absolute) | 0.7034057 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-diffusiondet-ap_s-0ec6c5e334d1` | sota_d1_test / DiffusionDet | D1 test, 42 | AP_S (absolute) | 0.4639875 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-diffusiondet-map-0ec6c5e334d1` | sota_d1_test / DiffusionDet | D1 test, 42 | mAP (absolute) | 0.7127698 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-dino_r50-ap50-9609255e14cb` | sota_d1_test / DINO R50 | D1 test, 42 | AP50 (absolute) | 0.9340033 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-dino_r50-ap75-9609255e14cb` | sota_d1_test / DINO R50 | D1 test, 42 | AP75 (absolute) | 0.8133487 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-dino_r50-ap_l-9609255e14cb` | sota_d1_test / DINO R50 | D1 test, 42 | AP_L (absolute) | 0.6251955 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-dino_r50-ap_m-9609255e14cb` | sota_d1_test / DINO R50 | D1 test, 42 | AP_M (absolute) | 0.711125 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-dino_r50-ap_s-9609255e14cb` | sota_d1_test / DINO R50 | D1 test, 42 | AP_S (absolute) | 0.4846417 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-dino_r50-map-9609255e14cb` | sota_d1_test / DINO R50 | D1 test, 42 | mAP (absolute) | 0.7248026 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow-ap50-63e5a02a9c13` | sota_d1_test / KaryoFlow | D1 test, 42 | AP50 (absolute) | 0.9308826 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow-ap75-63e5a02a9c13` | sota_d1_test / KaryoFlow | D1 test, 42 | AP75 (absolute) | 0.8229617 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow-ap_l-63e5a02a9c13` | sota_d1_test / KaryoFlow | D1 test, 42 | AP_L (absolute) | 0.6385985 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow-ap_m-63e5a02a9c13` | sota_d1_test / KaryoFlow | D1 test, 42 | AP_M (absolute) | 0.7255643 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow-ap_s-63e5a02a9c13` | sota_d1_test / KaryoFlow | D1 test, 42 | AP_S (absolute) | 0.5016113 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow-map-63e5a02a9c13` | sota_d1_test / KaryoFlow | D1 test, 42 | mAP (absolute) | 0.7401182 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow_k100-ap50-811ec7ce8089` | sota_d1_test / KaryoFlow K=100 | D1 test, 42 | AP50 (absolute) | 0.8831972 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k100-ap75-811ec7ce8089` | sota_d1_test / KaryoFlow K=100 | D1 test, 42 | AP75 (absolute) | 0.7864488 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k100-ap_l-811ec7ce8089` | sota_d1_test / KaryoFlow K=100 | D1 test, 42 | AP_L (absolute) | 0.5817735 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k100-ap_m-811ec7ce8089` | sota_d1_test / KaryoFlow K=100 | D1 test, 42 | AP_M (absolute) | 0.6962549 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k100-ap_s-811ec7ce8089` | sota_d1_test / KaryoFlow K=100 | D1 test, 42 | AP_S (absolute) | 0.435891 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k100-map-811ec7ce8089` | sota_d1_test / KaryoFlow K=100 | D1 test, 42 | mAP (absolute) | 0.7060669 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k200-ap50-e967f3e89b14` | sota_d1_test / KaryoFlow K=200 | D1 test, 42 | AP50 (absolute) | 0.9247837 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k200-ap75-e967f3e89b14` | sota_d1_test / KaryoFlow K=200 | D1 test, 42 | AP75 (absolute) | 0.8197426 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k200-ap_l-e967f3e89b14` | sota_d1_test / KaryoFlow K=200 | D1 test, 42 | AP_L (absolute) | 0.6200914 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k200-ap_m-e967f3e89b14` | sota_d1_test / KaryoFlow K=200 | D1 test, 42 | AP_M (absolute) | 0.7220441 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k200-ap_s-e967f3e89b14` | sota_d1_test / KaryoFlow K=200 | D1 test, 42 | AP_S (absolute) | 0.4887213 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_k200-map-e967f3e89b14` | sota_d1_test / KaryoFlow K=200 | D1 test, 42 | mAP (absolute) | 0.7368626 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_742d7d56fd05.json` |
| `d1-test-karyoflow_lqcr-ap50-93ad71bab76c` | sota_d1_test / KaryoFlow+LQCR | D1 test, 42 | AP50 (absolute) | 0.9308614 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow_lqcr-ap75-93ad71bab76c` | sota_d1_test / KaryoFlow+LQCR | D1 test, 42 | AP75 (absolute) | 0.8304128 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow_lqcr-ap_l-93ad71bab76c` | sota_d1_test / KaryoFlow+LQCR | D1 test, 42 | AP_L (absolute) | 0.6424099 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow_lqcr-ap_m-93ad71bab76c` | sota_d1_test / KaryoFlow+LQCR | D1 test, 42 | AP_M (absolute) | 0.7297576 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow_lqcr-ap_s-93ad71bab76c` | sota_d1_test / KaryoFlow+LQCR | D1 test, 42 | AP_S (absolute) | 0.5060658 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-karyoflow_lqcr-map-93ad71bab76c` | sota_d1_test / KaryoFlow+LQCR | D1 test, 42 | mAP (absolute) | 0.7468858 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-rtmdet_l-ap50-f58b44202cca` | sota_d1_test / RTMDet-L | D1 test, 42 | AP50 (absolute) | 0.9343724 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-rtmdet_l-ap75-f58b44202cca` | sota_d1_test / RTMDet-L | D1 test, 42 | AP75 (absolute) | 0.8387079 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-rtmdet_l-ap_l-f58b44202cca` | sota_d1_test / RTMDet-L | D1 test, 42 | AP_L (absolute) | 0.6221765 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-rtmdet_l-ap_m-f58b44202cca` | sota_d1_test / RTMDet-L | D1 test, 42 | AP_M (absolute) | 0.7253095 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-rtmdet_l-ap_s-f58b44202cca` | sota_d1_test / RTMDet-L | D1 test, 42 | AP_S (absolute) | 0.4802419 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-rtmdet_l-map-f58b44202cca` | sota_d1_test / RTMDet-L | D1 test, 42 | mAP (absolute) | 0.7318719 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-yolox_s-ap50-0a6df98ea434` | sota_d1_test / YOLOX-S | D1 test, 42 | AP50 (absolute) | 0.9177187 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-yolox_s-ap75-0a6df98ea434` | sota_d1_test / YOLOX-S | D1 test, 42 | AP75 (absolute) | 0.6919681 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-yolox_s-ap_l-0a6df98ea434` | sota_d1_test / YOLOX-S | D1 test, 42 | AP_L (absolute) | 0.5375828 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-yolox_s-ap_m-0a6df98ea434` | sota_d1_test / YOLOX-S | D1 test, 42 | AP_M (absolute) | 0.577646 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-yolox_s-ap_s-0a6df98ea434` | sota_d1_test / YOLOX-S | D1 test, 42 | AP_S (absolute) | 0.3298704 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d1-test-yolox_s-map-0a6df98ea434` | sota_d1_test / YOLOX-S | D1 test, 42 | mAP (absolute) | 0.5810722 |  | `ross:tools/experiment_db/evidence_sources/d1_test_unified_64973f1cb4e0.json` |
| `d2-test-cascade_rcnn-ap50-25e7f63f7418` | sota_d2_test / Cascade R-CNN | D2 test, 42 | AP50 (absolute) | 0.9850455 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-cascade_rcnn-ap75-25e7f63f7418` | sota_d2_test / Cascade R-CNN | D2 test, 42 | AP75 (absolute) | 0.9670476 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-cascade_rcnn-ap_l-25e7f63f7418` | sota_d2_test / Cascade R-CNN | D2 test, 42 | AP_L (absolute) | 0.8745239 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-cascade_rcnn-ap_m-25e7f63f7418` | sota_d2_test / Cascade R-CNN | D2 test, 42 | AP_M (absolute) | 0.8498566 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-cascade_rcnn-ap_s-25e7f63f7418` | sota_d2_test / Cascade R-CNN | D2 test, 42 | AP_S (absolute) | 0.5724726 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-cascade_rcnn-map-25e7f63f7418` | sota_d2_test / Cascade R-CNN | D2 test, 42 | mAP (absolute) | 0.8526474 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-diffusiondet-ap50-b819d77e81b3` | sota_d2_test / DiffusionDet | D2 test, 42 | AP50 (absolute) | 0.971417 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-diffusiondet-ap75-b819d77e81b3` | sota_d2_test / DiffusionDet | D2 test, 42 | AP75 (absolute) | 0.93832 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-diffusiondet-ap_l-b819d77e81b3` | sota_d2_test / DiffusionDet | D2 test, 42 | AP_L (absolute) | 0.8069923 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-diffusiondet-ap_m-b819d77e81b3` | sota_d2_test / DiffusionDet | D2 test, 42 | AP_M (absolute) | 0.8012791 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-diffusiondet-ap_s-b819d77e81b3` | sota_d2_test / DiffusionDet | D2 test, 42 | AP_S (absolute) | 0.5372902 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-diffusiondet-map-b819d77e81b3` | sota_d2_test / DiffusionDet | D2 test, 42 | mAP (absolute) | 0.8036371 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-dino_r50-ap50-789860257a37` | sota_d2_test / DINO R50 | D2 test, 42 | AP50 (absolute) | 0.9890508 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-dino_r50-ap75-789860257a37` | sota_d2_test / DINO R50 | D2 test, 42 | AP75 (absolute) | 0.9727885 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-dino_r50-ap_l-789860257a37` | sota_d2_test / DINO R50 | D2 test, 42 | AP_L (absolute) | 0.9195447 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-dino_r50-ap_m-789860257a37` | sota_d2_test / DINO R50 | D2 test, 42 | AP_M (absolute) | 0.8617116 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-dino_r50-ap_s-789860257a37` | sota_d2_test / DINO R50 | D2 test, 42 | AP_S (absolute) | 0.594855 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-dino_r50-map-789860257a37` | sota_d2_test / DINO R50 | D2 test, 42 | mAP (absolute) | 0.8649615 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow-ap50-8a787e978f71` | sota_d2_test / KaryoFlow | D2 test, 42 | AP50 (absolute) | 0.9881621 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow-ap75-8a787e978f71` | sota_d2_test / KaryoFlow | D2 test, 42 | AP75 (absolute) | 0.9703613 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow-ap_l-8a787e978f71` | sota_d2_test / KaryoFlow | D2 test, 42 | AP_L (absolute) | 0.87863 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow-ap_m-8a787e978f71` | sota_d2_test / KaryoFlow | D2 test, 42 | AP_M (absolute) | 0.8568456 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow-ap_s-8a787e978f71` | sota_d2_test / KaryoFlow | D2 test, 42 | AP_S (absolute) | 0.5728428 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow-map-8a787e978f71` | sota_d2_test / KaryoFlow | D2 test, 42 | mAP (absolute) | 0.8593603 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow_lqcr-ap50-076577b8ea96` | sota_d2_test / KaryoFlow+LQCR | D2 test, 42 | AP50 (absolute) | 0.9880828 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow_lqcr-ap75-076577b8ea96` | sota_d2_test / KaryoFlow+LQCR | D2 test, 42 | AP75 (absolute) | 0.9711153 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow_lqcr-ap_l-076577b8ea96` | sota_d2_test / KaryoFlow+LQCR | D2 test, 42 | AP_L (absolute) | 0.8868869 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow_lqcr-ap_m-076577b8ea96` | sota_d2_test / KaryoFlow+LQCR | D2 test, 42 | AP_M (absolute) | 0.8636624 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow_lqcr-ap_s-076577b8ea96` | sota_d2_test / KaryoFlow+LQCR | D2 test, 42 | AP_S (absolute) | 0.5660675 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-karyoflow_lqcr-map-076577b8ea96` | sota_d2_test / KaryoFlow+LQCR | D2 test, 42 | mAP (absolute) | 0.8662877 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-rtmdet_l-ap50-feca98add171` | sota_d2_test / RTMDet-L | D2 test, 42 | AP50 (absolute) | 0.9890287 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-rtmdet_l-ap75-feca98add171` | sota_d2_test / RTMDet-L | D2 test, 42 | AP75 (absolute) | 0.9705708 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-rtmdet_l-ap_l-feca98add171` | sota_d2_test / RTMDet-L | D2 test, 42 | AP_L (absolute) | 0.871243 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-rtmdet_l-ap_m-feca98add171` | sota_d2_test / RTMDet-L | D2 test, 42 | AP_M (absolute) | 0.8584423 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-rtmdet_l-ap_s-feca98add171` | sota_d2_test / RTMDet-L | D2 test, 42 | AP_S (absolute) | 0.5726758 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-rtmdet_l-map-feca98add171` | sota_d2_test / RTMDet-L | D2 test, 42 | mAP (absolute) | 0.8608965 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-yolox_s-ap50-ea70f679365b` | sota_d2_test / YOLOX-S | D2 test, 42 | AP50 (absolute) | 0.9854106 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-yolox_s-ap75-ea70f679365b` | sota_d2_test / YOLOX-S | D2 test, 42 | AP75 (absolute) | 0.9426535 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-yolox_s-ap_l-ea70f679365b` | sota_d2_test / YOLOX-S | D2 test, 42 | AP_L (absolute) | 0.7895607 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-yolox_s-ap_m-ea70f679365b` | sota_d2_test / YOLOX-S | D2 test, 42 | AP_M (absolute) | 0.7928075 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-yolox_s-ap_s-ea70f679365b` | sota_d2_test / YOLOX-S | D2 test, 42 | AP_S (absolute) | 0.4860796 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `d2-test-yolox_s-map-ea70f679365b` | sota_d2_test / YOLOX-S | D2 test, 42 | mAP (absolute) | 0.7951425 |  | `ross:tools/experiment_db/evidence_sources/d2_test_unified_f737cc0038ca.json` |
| `lqcr-d2-latency` | speed_accuracy / KaryoFlow_LQCR | D2 benchmark, 42 | latency (ms) | 77.68 |  | `ross+workstation:tools/experiment_db/evidence_sources/lqcr_cross_dataset_20260811.json` |
| `tradeoff-a1-latency` | speed_accuracy / RF_Heun | D2 benchmark, 42 | latency (ms) | 120.3407 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-a1-map` | speed_accuracy / RF_Heun | D2 test, 42 | mAP (absolute) | 0.8565748 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-a4-latency` | speed_accuracy / DPMpp_K500 | D2 benchmark, 42 | latency (ms) | 75.197 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-a4-map` | speed_accuracy / DPMpp_K500 | D2 test, 42 | mAP (absolute) | 0.8593603 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-cascade-latency` | speed_accuracy / Cascade_R50 | D2 benchmark, 42 | latency (ms) | 20.756 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-cascade-map` | speed_accuracy / Cascade_R50 | D2 test, 42 | mAP (absolute) | 0.8526474 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-diffdet-latency` | speed_accuracy / DiffusionDet | D2 benchmark, 42 | latency (ms) | 24.0855 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-diffdet-map` | speed_accuracy / DiffusionDet | D2 test, 42 | mAP (absolute) | 0.8036371 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-dino-latency` | speed_accuracy / DINO_R50 | D2 benchmark, 42 | latency (ms) | 34.3343 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_dino.json` |
| `tradeoff-dino-map` | speed_accuracy / DINO_R50 | D2 test, 42 | mAP (absolute) | 0.8649615 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_dino.json` |
| `tradeoff-k100-latency` | speed_accuracy / DPMpp_K100 | D2 benchmark, 42 | latency (ms) | 68.8989 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-k100-map` | speed_accuracy / DPMpp_K100 | D2 test, 42 | mAP (absolute) | 0.8476735 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-k200-latency` | speed_accuracy / DPMpp_K200 | D2 benchmark, 42 | latency (ms) | 70.5685 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-k200-map` | speed_accuracy / DPMpp_K200 | D2 test, 42 | mAP (absolute) | 0.8589109 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-k300-latency` | speed_accuracy / DPMpp_K300 | D2 benchmark, 42 | latency (ms) | 70.844 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-k300-map` | speed_accuracy / DPMpp_K300 | D2 test, 42 | mAP (absolute) | 0.8596773 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-rtmdet-latency` | speed_accuracy / RTMDet_L | D2 benchmark, 42 | latency (ms) | 33.829 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-rtmdet-map` | speed_accuracy / RTMDet_L | D2 test, 42 | mAP (absolute) | 0.8608965 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-yolox-latency` | speed_accuracy / YOLOX_S | D2 benchmark, 42 | latency (ms) | 10.035 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `tradeoff-yolox-map` | speed_accuracy / YOLOX_S | D2 test, 42 | mAP (absolute) | 0.7951425 |  | `ross:docs/paper/latex/figures/v2/data/source_fps_a6000_karyoflow.json` |
| `d2-test-topk-renewal-k100_renewal_off-ap50-761f6f275fc2` | topk_renewal_d2_test / K=100, renewal off | D2 test, 42 | AP50 (absolute) | 0.9578651 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_off-ap75-761f6f275fc2` | topk_renewal_d2_test / K=100, renewal off | D2 test, 42 | AP75 (absolute) | 0.935878 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_off-ap_l-761f6f275fc2` | topk_renewal_d2_test / K=100, renewal off | D2 test, 42 | AP_L (absolute) | 0.8212012 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_off-ap_m-761f6f275fc2` | topk_renewal_d2_test / K=100, renewal off | D2 test, 42 | AP_M (absolute) | 0.8307329 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_off-ap_s-761f6f275fc2` | topk_renewal_d2_test / K=100, renewal off | D2 test, 42 | AP_S (absolute) | 0.5009897 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_off-map-761f6f275fc2` | topk_renewal_d2_test / K=100, renewal off | D2 test, 42 | mAP (absolute) | 0.8307645 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_on-ap50-e7d706b58b5a` | topk_renewal_d2_test / K=100, renewal on | D2 test, 42 | AP50 (absolute) | 0.9761879 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_on-ap75-e7d706b58b5a` | topk_renewal_d2_test / K=100, renewal on | D2 test, 42 | AP75 (absolute) | 0.957272 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_on-ap_l-e7d706b58b5a` | topk_renewal_d2_test / K=100, renewal on | D2 test, 42 | AP_L (absolute) | 0.8455454 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_on-ap_m-e7d706b58b5a` | topk_renewal_d2_test / K=100, renewal on | D2 test, 42 | AP_M (absolute) | 0.8456667 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_on-ap_s-e7d706b58b5a` | topk_renewal_d2_test / K=100, renewal on | D2 test, 42 | AP_S (absolute) | 0.5390906 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k100_renewal_on-map-e7d706b58b5a` | topk_renewal_d2_test / K=100, renewal on | D2 test, 42 | mAP (absolute) | 0.8476735 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_off-ap50-72eb148531b7` | topk_renewal_d2_test / K=150, renewal off | D2 test, 42 | AP50 (absolute) | 0.9855476 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_off-ap75-72eb148531b7` | topk_renewal_d2_test / K=150, renewal off | D2 test, 42 | AP75 (absolute) | 0.9648349 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_off-ap_l-72eb148531b7` | topk_renewal_d2_test / K=150, renewal off | D2 test, 42 | AP_L (absolute) | 0.8635931 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_off-ap_m-72eb148531b7` | topk_renewal_d2_test / K=150, renewal off | D2 test, 42 | AP_M (absolute) | 0.8539304 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_off-ap_s-72eb148531b7` | topk_renewal_d2_test / K=150, renewal off | D2 test, 42 | AP_S (absolute) | 0.567223 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_off-map-72eb148531b7` | topk_renewal_d2_test / K=150, renewal off | D2 test, 42 | mAP (absolute) | 0.8558618 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_on-ap50-9b7cafc47cb4` | topk_renewal_d2_test / K=150, renewal on | D2 test, 42 | AP50 (absolute) | 0.9866572 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_on-ap75-9b7cafc47cb4` | topk_renewal_d2_test / K=150, renewal on | D2 test, 42 | AP75 (absolute) | 0.9668931 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_on-ap_l-9b7cafc47cb4` | topk_renewal_d2_test / K=150, renewal on | D2 test, 42 | AP_L (absolute) | 0.856021 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_on-ap_m-9b7cafc47cb4` | topk_renewal_d2_test / K=150, renewal on | D2 test, 42 | AP_M (absolute) | 0.8546706 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_on-ap_s-9b7cafc47cb4` | topk_renewal_d2_test / K=150, renewal on | D2 test, 42 | AP_S (absolute) | 0.560413 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k150_renewal_on-map-9b7cafc47cb4` | topk_renewal_d2_test / K=150, renewal on | D2 test, 42 | mAP (absolute) | 0.857068 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_off-ap50-1b4d311a8881` | topk_renewal_d2_test / K=200, renewal off | D2 test, 42 | AP50 (absolute) | 0.9880415 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_off-ap75-1b4d311a8881` | topk_renewal_d2_test / K=200, renewal off | D2 test, 42 | AP75 (absolute) | 0.9699219 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_off-ap_l-1b4d311a8881` | topk_renewal_d2_test / K=200, renewal off | D2 test, 42 | AP_L (absolute) | 0.8595456 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_off-ap_m-1b4d311a8881` | topk_renewal_d2_test / K=200, renewal off | D2 test, 42 | AP_M (absolute) | 0.8563802 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_off-ap_s-1b4d311a8881` | topk_renewal_d2_test / K=200, renewal off | D2 test, 42 | AP_S (absolute) | 0.5749795 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_off-map-1b4d311a8881` | topk_renewal_d2_test / K=200, renewal off | D2 test, 42 | mAP (absolute) | 0.8590443 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_on-ap50-b6d4dadb6643` | topk_renewal_d2_test / K=200, renewal on | D2 test, 42 | AP50 (absolute) | 0.9879767 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_on-ap75-b6d4dadb6643` | topk_renewal_d2_test / K=200, renewal on | D2 test, 42 | AP75 (absolute) | 0.9697318 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_on-ap_l-b6d4dadb6643` | topk_renewal_d2_test / K=200, renewal on | D2 test, 42 | AP_L (absolute) | 0.9066864 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_on-ap_m-b6d4dadb6643` | topk_renewal_d2_test / K=200, renewal on | D2 test, 42 | AP_M (absolute) | 0.85624 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_on-ap_s-b6d4dadb6643` | topk_renewal_d2_test / K=200, renewal on | D2 test, 42 | AP_S (absolute) | 0.5618472 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k200_renewal_on-map-b6d4dadb6643` | topk_renewal_d2_test / K=200, renewal on | D2 test, 42 | mAP (absolute) | 0.8589109 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_off-ap50-6e809442ebf5` | topk_renewal_d2_test / K=300, renewal off | D2 test, 42 | AP50 (absolute) | 0.9881919 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_off-ap75-6e809442ebf5` | topk_renewal_d2_test / K=300, renewal off | D2 test, 42 | AP75 (absolute) | 0.9698267 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_off-ap_l-6e809442ebf5` | topk_renewal_d2_test / K=300, renewal off | D2 test, 42 | AP_L (absolute) | 0.904995 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_off-ap_m-6e809442ebf5` | topk_renewal_d2_test / K=300, renewal off | D2 test, 42 | AP_M (absolute) | 0.8568063 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_off-ap_s-6e809442ebf5` | topk_renewal_d2_test / K=300, renewal off | D2 test, 42 | AP_S (absolute) | 0.5656429 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_off-map-6e809442ebf5` | topk_renewal_d2_test / K=300, renewal off | D2 test, 42 | mAP (absolute) | 0.8590595 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_on-ap50-4d88c2df2ab9` | topk_renewal_d2_test / K=300, renewal on | D2 test, 42 | AP50 (absolute) | 0.9883213 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_on-ap75-4d88c2df2ab9` | topk_renewal_d2_test / K=300, renewal on | D2 test, 42 | AP75 (absolute) | 0.9705832 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_on-ap_l-4d88c2df2ab9` | topk_renewal_d2_test / K=300, renewal on | D2 test, 42 | AP_L (absolute) | 0.9090044 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_on-ap_m-4d88c2df2ab9` | topk_renewal_d2_test / K=300, renewal on | D2 test, 42 | AP_M (absolute) | 0.8569603 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_on-ap_s-4d88c2df2ab9` | topk_renewal_d2_test / K=300, renewal on | D2 test, 42 | AP_S (absolute) | 0.5691925 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k300_renewal_on-map-4d88c2df2ab9` | topk_renewal_d2_test / K=300, renewal on | D2 test, 42 | mAP (absolute) | 0.8596773 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_off-ap50-0daaaa956319` | topk_renewal_d2_test / K=500, renewal off | D2 test, 42 | AP50 (absolute) | 0.988227 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_off-ap75-0daaaa956319` | topk_renewal_d2_test / K=500, renewal off | D2 test, 42 | AP75 (absolute) | 0.9708496 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_off-ap_l-0daaaa956319` | topk_renewal_d2_test / K=500, renewal off | D2 test, 42 | AP_L (absolute) | 0.8684567 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_off-ap_m-0daaaa956319` | topk_renewal_d2_test / K=500, renewal off | D2 test, 42 | AP_M (absolute) | 0.8567294 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_off-ap_s-0daaaa956319` | topk_renewal_d2_test / K=500, renewal off | D2 test, 42 | AP_S (absolute) | 0.5765257 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_off-map-0daaaa956319` | topk_renewal_d2_test / K=500, renewal off | D2 test, 42 | mAP (absolute) | 0.8592141 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_on-ap50-f12766631f38` | topk_renewal_d2_test / K=500, renewal on | D2 test, 42 | AP50 (absolute) | 0.9882093 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_on-ap75-f12766631f38` | topk_renewal_d2_test / K=500, renewal on | D2 test, 42 | AP75 (absolute) | 0.9705239 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_on-ap_l-f12766631f38` | topk_renewal_d2_test / K=500, renewal on | D2 test, 42 | AP_L (absolute) | 0.8879557 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_on-ap_m-f12766631f38` | topk_renewal_d2_test / K=500, renewal on | D2 test, 42 | AP_M (absolute) | 0.8568939 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_on-ap_s-f12766631f38` | topk_renewal_d2_test / K=500, renewal on | D2 test, 42 | AP_S (absolute) | 0.5766618 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |
| `d2-test-topk-renewal-k500_renewal_on-map-f12766631f38` | topk_renewal_d2_test / K=500, renewal on | D2 test, 42 | mAP (absolute) | 0.8594983 |  | `workstation:tools/experiment_db/evidence_sources/d2_test_inference_ablations_6dd1dcb1cc91.json` |

## Findings and claim boundaries

### F-COCO: General object-detection validation boundary [pending]

Detector-generic mechanisms have theoretical support but have not been empirically established on COCO.

- Scope: Current manuscript
- Generality basis: Proof assumptions are stated at proposal/solver level.
- Caveat: Only two distribution-shifted chromosome datasets are complete; mini-COCO is planned under resource constraints.

### F-DISTILL: Cascade-head distillation [supported]

A six-head teacher can supervise a three-head student while solver NFE remains four.

- Scope: D2 distillation; D1 architecture-only H3S4 control
- Generality basis: Repeated detector heads are compressible when intermediate behavior is supervised.
- Caveat: D1 result is not a distillation run; call 24-to-12 CHE, not NFE.
- Primary metric: mAP -0.0040
- Benefit: 1.71x speedup, 75.20 to 44.10 ms under the unified A6000 protocol

### F-GACS: Geometry-aware conditional stepping [supported]

Frozen geometry/confidence gates skip the second step for 38.64% of D1 images with near-neutral accuracy versus fixed two-step inference.

- Scope: D1, three checkpoint seeds
- Generality basis: Per-image computation can be allocated by proposal convergence statistics.
- Caveat: 0.7383 remains below the four-step reference 0.747; not a lossless production replacement.
- Primary metric: mAP -0.0007 vs fixed2
- Benefit: 13.79% latency reduction

### F-LQCR: Localization-quality calibrated final ranking [supported]

A final-only score p*q^2 improves high-IoU ranking without changing candidate coordinates, class logits, renewal, or solver states before score-dependent post-processing.

- Scope: D1 clean three-seed held-out test; D2 fixed-checkpoint seed42 held-out test
- Generality basis: The true-positive survival factorization applies to proposal detectors that rank localized candidates.
- Caveat: D2 remains a fixed-checkpoint intervention and no COCO empirical validation is complete.
- Primary metric: D1 +0.0067 mAP; D2 +0.0068 mAP on held-out test
- Benefit: D1 test AP_S +0.0057; D2 test AP90 +0.0258 and AP95 +0.0295; 77.68 ms same-hardware latency

### F-RENEW: History-consistent multistep inference [supported]

Disabling box renewal avoids mixing proposal identities in DPM++ history and removes renewal overhead.

- Scope: D1 and D2, K=500; D2 latency on RTX A6000
- Generality basis: Any multistep proposal solver using historical states requires state identity across steps.
- Caveat: Redundancy is required: at K=100, renewal removal loses 0.0313 mAP.
- Primary metric: D2 mAP -0.0003; D1 -0.0030
- Benefit: 2.43% mean latency reduction

### F-X0: Stable data prediction in low-dimensional shifted RF [supported]

x0 prediction is at least as accurate as v prediction and avoids inverse-t squared loss amplification.

- Scope: D1 and D2, three-seed validation
- Generality basis: The parameterization identity and loss weighting depend on path/schedule, not chromosome semantics.
- Caveat: Observed gains are small (+0.002 val on each dataset); treat as optimization evidence, not a universal ranking.
- Primary metric: +0.002 mAP on D1 and D2 val
- Benefit: more stable low-t optimization

## Theory statements

### T-LQCR

The ideal ranking score for IoU-thresholded detection factors class correctness and localization survival.

- Assumptions: A candidate is a true positive at threshold tau iff its class is correct and IoU>=tau; using conditional mean IoU as a cross-threshold statistic additionally assumes a family stochastically ordered by its mean.
- Derivation: P(TP_tau|h)=P(C|h) P(IoU>=tau|C,h). The implemented q is a practical surrogate trained toward E[M*IoU|h], with unmatched indicator M=0; p*q^beta is therefore empirically calibrated rather than asserted to equal the posterior.
- Prediction: Quality-aware re-ranking should mainly improve strict-IoU AP without changing candidate coordinates.
- Empirical status: Supported by D1 clean three-seed validation/test replication, a learned-quality beta sweep, and D2 AP90/AP95 gains.
- Source: `docs/paper/latex/main.tex`

### T-RENEW

A multistep DPM++ correction is inconsistent on rows whose proposal identity is replaced between steps.

- Assumptions: The update uses a finite-difference/history term across adjacent states; renewal replaces selected rows with independent proposals.
- Derivation: For D1=(x0_hat_n-x0_hat_{n-1})/(t_n-t_{n-1}), a renewed row compares predictions belonging to different proposal identities, so D1 is not a derivative estimate along one trajectory.
- Prediction: Disabling renewal should preserve or improve solver coherence when K provides enough redundant proposals, and should fail when K is too small.
- Empirical status: Supported at K>=200; low-K counterexample observed at K=100.
- Source: `docs/EXPERIMENT_LINEAGE.md`

### T-X0

x0 and velocity prediction are information-equivalent but not optimization-equivalent under the chosen objective.

- Assumptions: Rectified path x_t=(1-t)x0+t z and v=z-x0.
- Derivation: x0=x_t-t v; converting an x0 squared error to velocity error gives L_v=t^{-2} L_x0.
- Prediction: A shifted schedule emphasizing small t makes v loss amplify low-t gradients; direct x0 prediction should be no worse in four-dimensional box space.
- Empirical status: Directionally supported by D1 and D2 three-seed validation.
- Source: `docs/EXPERIMENT_LINEAGE.md`
