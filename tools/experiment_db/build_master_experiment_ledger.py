#!/usr/bin/env python3
"""Build and register the cross-dataset experiment/configuration ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_DEFAULT = ROOT / "tools/experiment_db/experiments.db"
DOC_DEFAULT = ROOT / "docs/experiments/MASTER_EXPERIMENT_LEDGER.md"
CSV_DEFAULT = ROOT / "tools/experiment_db/exports/master_experiment_ledger.csv"
MANIFEST_DEFAULT = ROOT / "tools/experiment_db/manifests/master_experiment_ledger_v1.json"

SEEDS = "42,123,789"
TRAIN_ASSIGN = "42=ross:A6000:0;123=workstation:A5000:0;789=workstation:A4000:1"
ACC_INFER_ASSIGN = "accuracy:any idle GPU; efficiency=ross:A6000:0 only"

D2_BASELINE_ARTIFACTS = {
    "diffusiondet": "d2-test-diffusiondet-b819d77e81b3",
    "dino_r50": "d2-test-dino_r50-789860257a37",
    "rtmdet_l": "d2-test-rtmdet_l-feca98add171",
    "cascade_rcnn_r50": "d2-test-cascade_rcnn-25e7f63f7418",
    "yolox_s": "d2-test-yolox_s-ea70f679365b",
}

DDL = """
CREATE TABLE IF NOT EXISTS experiment_ledger (
    ledger_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    family TEXT NOT NULL,
    variant TEXT NOT NULL,
    execution_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    replication_unit TEXT NOT NULL,
    training_seeds TEXT,
    inference_seeds TEXT,
    run_count INTEGER NOT NULL,
    config_path TEXT NOT NULL,
    config_status TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    executor_plan TEXT NOT NULL,
    work_dir_template TEXT NOT NULL,
    train_run_ids TEXT,
    result_family TEXT,
    evidence_artifact_ids TEXT,
    swanlab_project TEXT,
    swanlab_run_template TEXT,
    parent_ledger_id TEXT,
    paper_role TEXT NOT NULL,
    acceptance_gate TEXT NOT NULL,
    notes TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ledger_dataset ON experiment_ledger(dataset_id);
CREATE INDEX IF NOT EXISTS idx_ledger_status ON experiment_ledger(status);
CREATE INDEX IF NOT EXISTS idx_ledger_layer ON experiment_ledger(layer);
"""


def row(ledger_id, dataset_id, layer, family, variant, execution_kind, status,
        priority, replication_unit, training_seeds, inference_seeds, run_count,
        config_path, config_status, parameters, executor_plan, work_dir_template,
        result_family="", evidence_artifact_ids="", parent_ledger_id="",
        paper_role="", acceptance_gate="", notes="", swanlab=True):
    return dict(
        ledger_id=ledger_id, dataset_id=dataset_id, layer=layer, family=family,
        variant=variant, execution_kind=execution_kind, status=status,
        priority=priority, replication_unit=replication_unit,
        training_seeds=training_seeds, inference_seeds=inference_seeds,
        run_count=run_count, config_path=config_path, config_status=config_status,
        parameters=parameters, executor_plan=executor_plan,
        work_dir_template=work_dir_template, result_family=result_family,
        evidence_artifact_ids=evidence_artifact_ids,
        swanlab_project="KaryoFlow-Self1700" if swanlab and dataset_id == "D1_INHOUSE1700_V1" else "",
        swanlab_run_template="{variant}_seed{training_seed}" if swanlab and execution_kind in ("full_train", "short_train") else "",
        parent_ledger_id=parent_ledger_id, paper_role=paper_role,
        acceptance_gate=acceptance_gate, notes=notes, train_run_ids="",
    )


def build_rows():
    rows = []
    add = rows.append
    # Dataset 1: primary detector comparison.
    for method, config, status, params in [
        ("karyoflow", "experiments/configs/self1700/karyoflow.py", "RUNNING", "RF; shifted t; AdaLN-Zero; DPM++ 4-step; K=500; 6 heads; batch=2"),
        ("diffusiondet", "experiments/configs/self1700/diffusiondet.py", "PLANNED", "DDPM; DDIM 1-step; linear t; scale-shift; K=500; 6 heads; batch=4 (external baseline)"),
        ("dino_r50", "experiments/configs/self1700/dino_r50.py", "PLANNED", "R50; 900 queries; 6 encoder/decoder layers; batch=2"),
        ("rtmdet_l", "experiments/configs/self1700/rtmdet_l.py", "PLANNED", "RTMDet-L; inherited benchmark optimizer; batch=2"),
        ("cascade_rcnn_r50", "experiments/configs/self1700/cascade_rcnn_r50.py", "PLANNED", "Cascade R-CNN R50; 3 stages; batch=4"),
        ("yolox_s", "experiments/configs/self1700/yolox_s.py", "PLANNED", "YOLOX-S; 300 epochs; batch=8"),
    ]:
        add(row(f"D1I.SOTA.{method}", "D1_INHOUSE1700_V1", "Detector comparison", "SOTA", method,
                "full_train", status, "P0", "independent_training_seed", SEEDS, "42", 3,
                config, "exists", params, TRAIN_ASSIGN,
                f"work_dirs/self1700/{method}/trainseed_{{training_seed}}",
                result_family="sota_d1i_test", paper_role="Main two-cohort detector table",
                acceptance_gate="3 distinct checkpoint SHA; same manifest/config/protocol; six test metrics"))
    add(row("D1I.SOTA.karyoflow_lqcr", "D1_INHOUSE1700_V1", "Decision", "SOTA", "karyoflow_lqcr",
            "short_train", "BLOCKED_PARENT", "P0", "paired_final_stage_intervention", SEEDS, "42", 3,
            "experiments/configs/self1700/lqcr.py", "exists",
            "parent-matched final-only quality head; beta=2 selected on validation; 12 epochs",
            TRAIN_ASSIGN, "work_dirs/self1700/karyoflow_lqcr/trainseed_{training_seed}",
            result_family="paired_lqcr_d1i_train3", parent_ledger_id="D1I.SOTA.karyoflow",
            paper_role="Independent decision contribution",
            acceptance_gate="parent checkpoint identity; 590 shared tensors unchanged; paired test delta"))

    # Dataset 1: strict structural training ablation.
    for code, variant, config, config_status, status, params, parent in [
        ("G0", "DDPM_linear_scaleshift", "experiments/configs/self1700/ablations/g0_ddpm_batch2.py", "to_create", "PLANNED", "DDPM; DDIM1; linear t; scale-shift; batch=2", ""),
        ("G1", "RF_linear_scaleshift", "experiments/configs/self1700/ablations/g1_rf_linear_scaleshift.py", "to_create", "PLANNED", "RF; Euler/locked validation protocol; linear t; scale-shift; batch=2", "D1I.ABL.G0"),
        ("G2", "RF_shifted_scaleshift", "experiments/configs/self1700/ablations/g2_rf_shifted_scaleshift.py", "to_create", "PLANNED", "RF; shifted t (shift=3); scale-shift; batch=2", "D1I.ABL.G1"),
        ("G3", "RF_shifted_AdaLNZero", "experiments/configs/self1700/karyoflow.py", "exists", "RUNNING_REUSED", "RF; shifted t; AdaLN-Zero; DPM++4 validation; batch=2", "D1I.ABL.G2"),
    ]:
        add(row(f"D1I.ABL.{code}", "D1_INHOUSE1700_V1", "Generation", "Training ablation", variant,
                "full_train", status, "P0", "independent_training_seed", SEEDS, "42", 3,
                config, config_status, params, TRAIN_ASSIGN,
                f"work_dirs/self1700/ablations/{code.lower()}/trainseed_{{training_seed}}",
                result_family="generation_train_ablation_d1i_test", parent_ledger_id=parent,
                paper_role="Strict incremental generation ablation",
                acceptance_gate="only named factor changes; batch/optimizer/augmentation/selection fixed"))

    # Dataset 1: same-checkpoint inference and deployment analyses.
    add(row("D1I.INF.solver_steps", "D1_INHOUSE1700_V1", "Generation", "Inference ablation", "Euler_Heun_DPMpp_x_steps1to4",
            "inference", "PLANNED", "P0", "fixed_checkpoint_by_training_parent", SEEDS, "42", 36,
            "tools/experiment_db/test_inference_ablations.py", "parameterize_for_d1i",
            "3 parents x {Euler,Heun,DPM++} x {1,2,3,4}; report steps and NFE",
            ACC_INFER_ASSIGN, "results/self1700/inference/solver/{parent}/{solver}_{steps}",
            result_family="solver_d1i_test", parent_ledger_id="D1I.ABL.G3",
            paper_role="Solver/step accuracy-efficiency ablation",
            acceptance_gate="same checkpoint per parent; common test; no cross-checkpoint causal claim", swanlab=False))
    add(row("D1I.INF.topk_renewal", "D1_INHOUSE1700_V1", "Generation", "Inference ablation", "TopK_x_renewal",
            "inference", "PLANNED", "P1", "fixed_checkpoint_by_training_parent", SEEDS, "42", 30,
            "tools/experiment_db/test_inference_ablations.py", "parameterize_for_d1i",
            "3 parents x K={100,150,200,300,500} x renewal={off,on}", ACC_INFER_ASSIGN,
            "results/self1700/inference/topk_renewal/{parent}/k{K}_{renewal}",
            result_family="topk_renewal_d1i_test", parent_ledger_id="D1I.ABL.G3",
            paper_role="Identity/renewal and candidate-budget analysis",
            acceptance_gate="same checkpoint/seed; renewal-off candidate identity audit", swanlab=False))
    add(row("D1I.DEC.beta_val", "D1_INHOUSE1700_V1", "Decision", "Hyperparameter selection", "LQCR_beta_0_0.25_0.5_1_2",
            "inference", "BLOCKED_PARENT", "P0", "validation_selection_by_parent", SEEDS, "42", 15,
            "tools/experiment_db/test_inference_ablations.py", "parameterize_for_d1i",
            "3 LQCR parents x beta={0,0.25,0.5,1,2}; validation only",
            ACC_INFER_ASSIGN, "results/self1700/lqcr_beta_val/{parent}/beta_{beta}",
            result_family="lqcr_beta_d1i_val", parent_ledger_id="D1I.SOTA.karyoflow_lqcr",
            paper_role="Validation selection; not a test claim",
            acceptance_gate="select beta before any new test evaluation", swanlab=False))
    add(row("D1I.DEC.strict_subsets", "D1_INHOUSE1700_V1", "Decision", "Diagnostic", "AP90_AP95_scale_overlap_quality",
            "analysis", "BLOCKED_PREDICTIONS", "P1", "paired_prediction_analysis", SEEDS, "42", 1,
            "tools/experiment_db/analysis/difficult_subset_analysis.py", "adapt_from_d2",
            "AP90/AP95; AP_S/M/L; overlap strata; size quartiles; quality-IoU Spearman; paired image bootstrap",
            "CPU after test predictions", "results/self1700/analysis/lqcr_difficult_subsets",
            result_family="conditional_difficult_subset_d1i", parent_ledger_id="D1I.SOTA.karyoflow_lqcr",
            paper_role="Mechanism evidence and limitation",
            acceptance_gate="image-level pairing; diagnostics not paper SOTA rows", swanlab=False))
    add(row("D1I.DEP.distill_h3", "D1_INHOUSE1700_V1", "Deployment", "Head distillation", "H6_teacher_to_H3_student",
            "short_train", "PLANNED", "P1", "paired_student_by_parent", SEEDS, "42", 3,
            "experiments/configs/self1700/deployment/h3_distill.py", "to_create",
            "teacher heads=6; student heads=3; mapping 0<-0,1<-2,2<-5; fixed loss/protocol",
            TRAIN_ASSIGN, "work_dirs/self1700/deployment/h3/trainseed_{training_seed}",
            result_family="head_distillation_d1i_test", parent_ledger_id="D1I.ABL.G3",
            paper_role="Deployment extension",
            acceptance_gate="three parent-matched students; test mAP; A6000 latency only"))
    add(row("D1I.DEP.GACS", "D1_INHOUSE1700_V1", "Deployment", "Dynamic deployment", "GACS",
            "inference", "PLANNED", "P2", "fixed_checkpoint_by_training_parent", SEEDS, "42", 3,
            "experiments/configs/self1700/deployment/gacs.py", "to_create_or_port",
            "prespecified dynamic policy; accuracy and latency; no precision-module claim",
            "accuracy:any; latency=ross:A6000:0", "results/self1700/deployment/gacs/{parent}",
            result_family="gacs_d1i_test", parent_ledger_id="D1I.ABL.G3",
            paper_role="Optional dynamic deployment extension",
            acceptance_gate="held-out test on both cohorts; same A6000 timing protocol", swanlab=False))
    add(row("D1I.DEP.speed", "D1_INHOUSE1700_V1", "Deployment", "Efficiency", "all_operating_points",
            "benchmark", "BLOCKED_CHECKPOINTS", "P1", "hardware_efficiency_repeat", SEEDS, "none", 1,
            "tools/benchmark_fps.py", "exists_requires_protocol_wrapper",
            "batch=1; same input; precision; warmup; timed iterations; generation/teacher/student/LQCR/GACS",
            "ross:A6000:0 exclusive", "results/self1700/benchmark/a6000/{variant}",
            result_family="speed_accuracy_d1i", paper_role="Speed-accuracy figure/table",
            acceptance_gate="exclusive A6000; no foreign process; accuracy linked to verified test record", swanlab=False))

    # Dataset 2: completed primary KaryoFlow evidence and baseline gaps.
    add(row("D2.SOTA.karyoflow_train3", "D2", "Detector comparison", "SOTA", "karyoflow",
            "full_train", "COMPLETED_VERIFIED", "DONE", "independent_training_seed", "335778785,790448076,1342286018", "42", 3,
            "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py", "exists_plus_dumped_configs",
            "RF; shifted t; AdaLN-Zero; DPM++4; K=500; 6 heads",
            "historical ross/workstation", "work_dirs/{a4_dpm_pp_24obj|multi_seed/a4_dpm_pp_24obj/seed_*}",
            result_family="paired_lqcr_d2_train3;sota_d2_test",
            evidence_artifact_ids="d2-paired-lqcr-train3-4152cdbd5100",
            paper_role="Main D2 KaryoFlow mean", acceptance_gate="already verified: 3 checkpoint SHA; 1000-image test", swanlab=False))
    add(row("D2.SOTA.karyoflow_lqcr_train3", "D2", "Decision", "SOTA", "karyoflow_lqcr",
            "short_train", "COMPLETED_VERIFIED", "DONE", "paired_final_stage_intervention", "335778785,790448076,1342286018", "42", 3,
            "experiments/configs/ldmdet/directions/capr/paper_train3_lqcr_run*.py", "exists",
            "parent-matched final-only quality head; beta=2",
            "historical ross/workstation", "work_dirs/{capr_quality_only_24obj|paper_d2_lqcr_trainrun_*}",
            result_family="paired_lqcr_d2_train3;sota_d2_test",
            evidence_artifact_ids="d2-paired-lqcr-train3-4152cdbd5100", parent_ledger_id="D2.SOTA.karyoflow_train3",
            paper_role="Main D2 LQCR paired effect", acceptance_gate="already verified: shared detector weights unchanged", swanlab=False))
    for method, config, ckpt in [
        ("diffusiondet", "experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py", "work_dirs/baselines/diffusiondet_24obj"),
        ("dino_r50", "experiments/configs/baselines/benchmark_24obj/dino_r50.py", "work_dirs/baselines/dino_r50_24obj"),
        ("rtmdet_l", "experiments/configs/baselines/benchmark_24obj/rtmdet_l.py", "work_dirs/baselines/rtmdet_l_24obj"),
        ("cascade_rcnn_r50", "experiments/configs/baselines/benchmark_24obj/cascade_rcnn_r50.py", "work_dirs/baselines/cascade_rcnn_r50_24obj"),
        ("yolox_s", "experiments/configs/baselines/benchmark_24obj/yolox_s.py", "work_dirs/baselines/yolox_s"),
    ]:
        add(row(f"D2.SOTA.{method}.existing", "D2", "Detector comparison", "SOTA", method,
                "evaluation", "COMPLETED_POINT_ESTIMATE", "DONE_DESCRIPTIVE", "single_training_checkpoint", "unknown/one", "42", 1,
                config, "exists", "published table checkpoint; six test metrics",
                "historical", ckpt, result_family="sota_d2_test",
                evidence_artifact_ids=D2_BASELINE_ARTIFACTS[method], paper_role="D2 point estimate only",
                acceptance_gate="must not be labeled three-training-seed", swanlab=False))
        add(row(f"D2.SOTA.{method}.missing_train2", "D2", "Detector comparison", "SOTA completion", method,
                "full_train", "PLANNED", "P1", "independent_training_seed", "123,789", "42", 2,
                config, "exists", "same configuration and selection rule as existing checkpoint",
                "123=workstation:A5000:0;789=ross:A6000:0 (A4000 for lighter model if exact config fits)",
                f"work_dirs/d2_train3_completion/{method}/trainseed_{{training_seed}}",
                result_family="sota_d2_train3_completion", parent_ledger_id=f"D2.SOTA.{method}.existing",
                paper_role="Upgrade D2 baseline to three independent trainings",
                acceptance_gate="actual framework seed recorded; distinct checkpoint SHA; 1000-image test"))

    # Dataset 2: historical and strict generation ablations.
    add(row("D2.ABL.historical_chain", "D2", "Generation", "Historical ablation", "DDPM_RF_Heun_AdaLN_DPMpp",
            "mixed", "COMPLETED_DESCRIPTIVE", "DONE_DESCRIPTIVE", "mixed_cross_checkpoint", "mixed", "mixed", 4,
            "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a{0,1,2,4}_*.py", "exists",
            "historical A0/A1/A2/A4; configs/checkpoints and training seeds not fully paired",
            "historical ross/workstation", "work_dirs/{a0_baseline_24obj|a1_rf_heun_24obj|a2_rf_heun_adaln_24obj|a4_dpm_pp_24obj}",
            result_family="legacy_protocol", paper_role="Context only; not cumulative causal ablation",
            acceptance_gate="never report adjacent deltas as isolated effects", swanlab=False))
    for code, variant, config, params in [
        ("G0", "DDPM_linear_scaleshift", "experiments/configs/d2_strict_ablations/g0_ddpm_batch2.py", "DDPM; DDIM1; linear t; scale-shift; batch=2"),
        ("G1", "RF_linear_scaleshift", "experiments/configs/d2_strict_ablations/g1_rf_linear_scaleshift.py", "RF; linear t; scale-shift; batch=2"),
        ("G2", "RF_shifted_scaleshift", "experiments/configs/d2_strict_ablations/g2_rf_shifted_scaleshift.py", "RF; shifted t; scale-shift; batch=2"),
    ]:
        add(row(f"D2.ABL.strict.{code}", "D2", "Generation", "Training ablation", variant,
                "full_train", "PLANNED", "P2", "independent_training_seed", SEEDS, "42", 3,
                config, "to_create", params, TRAIN_ASSIGN,
                f"work_dirs/d2_strict_ablations/{code.lower()}/trainseed_{{training_seed}}",
                result_family="generation_train_ablation_d2_test", paper_role="Cross-cohort strict ablation replication",
                acceptance_gate="only named factor changes; same batch/optimizer/selection; six test metrics"))
    add(row("D2.ABL.strict.G3", "D2", "Generation", "Training ablation", "RF_shifted_AdaLNZero",
            "full_train", "COMPLETED_VERIFIED_REUSED", "DONE", "independent_training_seed",
            "335778785,790448076,1342286018", "42", 3,
            "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py", "exists_plus_dumped_configs",
            "RF; shifted t; AdaLN-Zero; DPM++4", "historical ross/workstation",
            "work_dirs/{a4_dpm_pp_24obj|multi_seed/a4_dpm_pp_24obj/seed_*}",
            result_family="paired_lqcr_d2_train3", evidence_artifact_ids="d2-paired-lqcr-train3-4152cdbd5100",
            paper_role="Strict chain endpoint; reuse only after config compatibility audit",
            acceptance_gate="confirm optimizer/batch/selection compatibility with new G0-G2", swanlab=False))

    add(row("D2.INF.solver_steps.fixed1", "D2", "Generation", "Inference ablation", "Euler_Heun_DPMpp_x_steps1to4",
            "inference", "COMPLETED_FIXED_CHECKPOINT", "DONE_CONDITIONAL", "fixed_checkpoint_inference", "335778785", "42", 12,
            "tools/experiment_db/test_inference_ablations.py", "exists",
            "one fixed KaryoFlow checkpoint x 3 solvers x 4 steps", "historical",
            "results/d2_test_inference_ablations/solver/{solver}_{steps}", result_family="solver_d2_test",
            evidence_artifact_ids="d2-test-inference-ablations-b81d6a3d498a", parent_ledger_id="D2.SOTA.karyoflow_train3",
            paper_role="Conditional solver mechanism evidence",
            acceptance_gate="label fixed-checkpoint; not training replication", swanlab=False))
    add(row("D2.INF.solver_steps.train3", "D2", "Generation", "Inference replication", "Euler_Heun_DPMpp_x_steps1to4_train3",
            "inference", "PLANNED", "P1", "fixed_checkpoint_by_training_parent", "335778785,790448076,1342286018", "42", 36,
            "tools/experiment_db/test_inference_ablations.py", "exists_parameterize_parent",
            "3 independent parents x 3 solvers x 4 steps", ACC_INFER_ASSIGN,
            "results/d2_train3_inference/solver/{parent}/{solver}_{steps}", result_family="solver_d2_train3_test",
            parent_ledger_id="D2.SOTA.karyoflow_train3", paper_role="Training-robust solver ablation",
            acceptance_gate="aggregate one value per training parent", swanlab=False))
    add(row("D2.INF.topk_renewal.fixed1", "D2", "Generation", "Inference ablation", "TopK_x_renewal",
            "inference", "COMPLETED_FIXED_CHECKPOINT", "DONE_CONDITIONAL", "fixed_checkpoint_inference", "335778785", "42", 10,
            "tools/experiment_db/test_inference_ablations.py", "exists",
            "K={100,150,200,300,500} x renewal={off,on}", "historical",
            "results/d2_test_inference_ablations/topk_renewal/{variant}", result_family="topk_renewal_d2_test",
            evidence_artifact_ids="d2-test-inference-ablations-6dd1dcb1cc91", parent_ledger_id="D2.SOTA.karyoflow_train3",
            paper_role="Conditional identity/candidate-budget evidence",
            acceptance_gate="label fixed-checkpoint", swanlab=False))
    add(row("D2.INF.topk_renewal.train3", "D2", "Generation", "Inference replication", "TopK_x_renewal_train3",
            "inference", "PLANNED", "P2", "fixed_checkpoint_by_training_parent", "335778785,790448076,1342286018", "42", 30,
            "tools/experiment_db/test_inference_ablations.py", "exists_parameterize_parent",
            "3 parents x 5 K x renewal off/on", ACC_INFER_ASSIGN,
            "results/d2_train3_inference/topk_renewal/{parent}/{variant}", result_family="topk_renewal_d2_train3_test",
            parent_ledger_id="D2.SOTA.karyoflow_train3", paper_role="Training-robust identity analysis",
            acceptance_gate="aggregate by training parent", swanlab=False))
    add(row("D2.DEC.beta_test.fixed1", "D2", "Decision", "Post-selection sensitivity", "LQCR_beta_0_0.25_0.5_1_2",
            "inference", "COMPLETED_DIAGNOSTIC", "DONE_DIAGNOSTIC", "fixed_checkpoint_inference", "335778785", "42", 5,
            "tools/experiment_db/test_inference_ablations.py", "exists",
            "test beta sweep after beta=2 validation selection; transparency only", "historical",
            "results/d2_test_inference_ablations/lqcr_beta/{beta}", result_family="lqcr_beta_d2_test",
            evidence_artifact_ids="d2-test-inference-ablations-d9fd109403da", parent_ledger_id="D2.SOTA.karyoflow_lqcr_train3",
            paper_role="Post-selection sensitivity, not selection evidence",
            acceptance_gate="caption must state beta=2 selected on validation", swanlab=False))
    add(row("D2.DEC.strict_subsets", "D2", "Decision", "Diagnostic", "AP90_AP95_scale_overlap_bootstrap",
            "analysis", "COMPLETED_DIAGNOSTIC", "DONE_DIAGNOSTIC", "paired_prediction_analysis",
            "335778785,790448076,1342286018", "42", 2,
            "tools/experiment_db/analysis/d2_difficult_subset_analysis.py", "exists",
            "overlap/size strict recall plus 1000-replicate LQCR-vs-DINO image bootstrap",
            "CPU", "tools/experiment_db/evidence_sources/d2_{difficult_subsets|lqcr_vs_dino}_*.json",
            result_family="conditional_difficult_subset_d2;conditional_image_bootstrap_d2",
            evidence_artifact_ids="d2-difficult-subsets-paired-train3-725aab1ab102;d2-lqcr-vs-dino-image-bootstrap-1000-seed20260812",
            parent_ledger_id="D2.SOTA.karyoflow_lqcr_train3", paper_role="Mechanism/uncertainty; not SOTA proof",
            acceptance_gate="diagnostic paper_eligible=0", swanlab=False))
    add(row("D2.DEP.distill_h3.existing", "D2", "Deployment", "Head distillation", "H6_to_H3_single_parent",
            "short_train", "COMPLETED_SINGLE_PARENT", "DONE_CONDITIONAL", "single_parent_student", "one", "42", 1,
            "experiments/configs/ldmdet/directions/mainline_ablation_24obj/h3_distill_plan_a_24obj.py", "exists",
            "H6 teacher -> H3 student; mapping 0,2,5", "historical ross/workstation",
            "work_dirs/h3_distill_plan_a_24obj", result_family="head_distillation",
            paper_role="Conditional deployment point", acceptance_gate="do not call three-training-seed", swanlab=False))
    add(row("D2.DEP.distill_h3.train3", "D2", "Deployment", "Head distillation", "H6_to_H3_parent_matched_train3",
            "short_train", "PLANNED", "P1", "paired_student_by_parent", "335778785,790448076,1342286018", "42", 3,
            "experiments/configs/d2_deployment/h3_distill_parent.py", "to_create_or_parameterize",
            "one H3 student per KaryoFlow parent; identical mapping/loss", TRAIN_ASSIGN,
            "work_dirs/d2_deployment/h3/{parent}", result_family="head_distillation_d2_train3",
            parent_ledger_id="D2.SOTA.karyoflow_train3", paper_role="Training-robust deployment evidence",
            acceptance_gate="three parent-matched students; A6000 latency only"))
    add(row("D2.DEP.GACS", "D2", "Deployment", "Dynamic deployment", "GACS",
            "inference", "PARTIAL_LEGACY", "P2", "fixed_checkpoint_by_training_parent", "one/unknown", "42", 1,
            "experiments/configs/ldmdet/directions/inference_opt/gacs*.py", "locate_and_standardize",
            "dynamic deployment policy; historical D1 loss ~0.0007 mAP; D2 held-out test missing",
            "accuracy:any; latency=ross:A6000:0", "results/d2_deployment/gacs/{parent}",
            result_family="GACS", parent_ledger_id="D2.SOTA.karyoflow_train3",
            paper_role="Optional deployment extension, not main precision module",
            acceptance_gate="D2 test plus A6000 speed before paper use", swanlab=False))
    add(row("D2.DEP.speed", "D2", "Deployment", "Efficiency", "all_operating_points",
            "benchmark", "COMPLETED_PARTIAL", "P1", "hardware_efficiency_repeat", "mixed", "none", 1,
            "tools/benchmark_fps.py", "exists",
            "A6000: RF-Heun; DPM++ K=100/200/300/500; LQCR; H3; repeat after final checkpoints",
            "ross:A6000:0 exclusive", "results/benchmark/a6000/{variant}", result_family="speed_accuracy",
            paper_role="Speed-accuracy figure/table",
            acceptance_gate="repeat final selected points with exclusive GPU and frozen timing boundary", swanlab=False))

    # Legacy composite Dataset 1 is retained only as an audit pointer.
    add(row("LEGACY.D1.composite", "D1_COMPOSITE2200_LEGACY", "Archive", "Legacy evidence", "all_historical_results",
            "mixed", "ARCHIVED_NONCOMPARABLE", "ARCHIVE", "mixed", "mixed", "mixed", 1,
            "tools/experiment_db/experiments.db", "exists",
            "old 2200-image composite contains 500 D2-derived images and old split; results remain auditable",
            "historical ross/workstation", "work_dirs/*; results/*",
            result_family="all D1 families except D1_INHOUSE1700_V1", evidence_artifact_ids="multiple",
            paper_role="Provenance/history only; never pool with new D1I or D2",
            acceptance_gate="must be labeled legacy and non-comparable", swanlab=False))
    return rows


def enrich_live_ids(conn, rows):
    for item in rows:
        if item["dataset_id"] != "D1_INHOUSE1700_V1" or item["family"] != "SOTA":
            continue
        method = item["variant"]
        db_method = "karyoflow_lqcr" if method == "karyoflow_lqcr" else method
        found = conn.execute(
            """SELECT train_run_id,training_seed,status,assigned_executor,work_dir,
                      tracker_project,tracker_run_name,tracker_run_id
               FROM train_run_registry WHERE dataset_id=? AND method=?
                 AND status NOT IN ('superseded','invalid') ORDER BY training_seed""",
            ("D1_INHOUSE1700_V1", db_method)).fetchall()
        if found:
            item["train_run_ids"] = ";".join(str(x[0]) for x in found)
            statuses = {str(x[2]).upper() for x in found}
            if statuses == {"RUNNING"}:
                item["status"] = "RUNNING"
            item["notes"] = (item["notes"] + " " if item["notes"] else "") + \
                "Live registry: " + "; ".join(
                    f"seed{x[1]}={x[2]}@{x[3]}, run_id={x[7] or 'pending'}" for x in found)


def upsert(conn, rows):
    columns = [
        "ledger_id", "dataset_id", "layer", "family", "variant", "execution_kind",
        "status", "priority", "replication_unit", "training_seeds", "inference_seeds",
        "run_count", "config_path", "config_status", "parameters_json", "executor_plan",
        "work_dir_template", "train_run_ids", "result_family", "evidence_artifact_ids",
        "swanlab_project", "swanlab_run_template", "parent_ledger_id", "paper_role",
        "acceptance_gate", "notes",
    ]
    conn.execute("DELETE FROM experiment_ledger")
    placeholders = ",".join("?" for _ in columns)
    for item in rows:
        values = []
        for key in columns:
            if key == "parameters_json":
                values.append(json.dumps({"summary": item["parameters"]}, ensure_ascii=False, sort_keys=True))
            else:
                values.append(item.get(key, ""))
        conn.execute(
            f"INSERT INTO experiment_ledger ({','.join(columns)}) VALUES ({placeholders})", values)
    conn.commit()


def write_manifest(rows, path):
    payload = {
        "schema_version": "1.0", "ledger_name": "KaryoFlow cross-dataset experiment ledger",
        "dataset_scope": ["D1_INHOUSE1700_V1", "D2", "D1_COMPOSITE2200_LEGACY"],
        "status_policy": {
            "COMPLETED_VERIFIED": "independent training or paired evidence fully verified",
            "COMPLETED_FIXED_CHECKPOINT": "conditional evidence from one fixed trained model",
            "COMPLETED_POINT_ESTIMATE": "one trained checkpoint; no training variance claim",
            "PLANNED": "predefined but not yet started", "RUNNING": "active formal run",
            "BLOCKED_PARENT": "waiting for registered parent checkpoint",
        },
        "experiments": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def markdown_table(items):
    lines = [
        "| Ledger ID | Variant | Kind / runs | Seeds | Status | Config | Executor | Work/results directory | DB/evidence IDs |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for x in items:
        ids = x["train_run_ids"] or x["result_family"] or "ledger row only"
        if x["evidence_artifact_ids"]:
            ids += f"; artifact={x['evidence_artifact_ids']}"
        lines.append(
            f"| `{x['ledger_id']}` | {x['variant']} | {x['execution_kind']} / {x['run_count']} | "
            f"train={x['training_seeds'] or '-'}; infer={x['inference_seeds'] or '-'} | **{x['status']}** | "
            f"`{x['config_path']}` ({x['config_status']}) | {x['executor_plan']} | "
            f"`{x['work_dir_template']}` | `{ids}` |"
        )
    return "\n".join(lines)


def write_doc(rows, manifest_sha, path):
    by_dataset = {}
    for item in rows:
        by_dataset.setdefault(item["dataset_id"], []).append(item)
    counts = {}
    for dataset, items in by_dataset.items():
        counts[dataset] = {
            "rows": len(items), "expanded_runs": sum(x["run_count"] for x in items),
            "running": sum(x["status"] == "RUNNING" for x in items),
            "pending_rows": sum(x["status"] in ("PLANNED", "BLOCKED_PARENT", "BLOCKED_PREDICTIONS", "BLOCKED_CHECKPOINTS") for x in items),
        }
    lines = [
        "# KaryoFlow双数据集实验—配置—证据总账",
        "",
        "> 本文档是后续实验调度、数据库登记和论文数字校对的唯一入口。数值结果仍以 `tools/experiment_db/experiments.db` 的受控证据为准。",
        "",
        "## 技术摘要",
        "",
        f"- 权威清单 SHA-256：`{manifest_sha}`。",
        f"- 纯自建数据集当前有 {counts['D1_INHOUSE1700_V1']['rows']} 个实验族、展开 {counts['D1_INHOUSE1700_V1']['expanded_runs']} 个训练/推理/分析运行；正式 KaryoFlow 三训练seed正在运行。",
        f"- Dataset 2 当前有 {counts['D2']['rows']} 个实验族、展开 {counts['D2']['expanded_runs']} 个运行。KaryoFlow/LQCR已有三独立训练配对；五个传统基线仍主要是单训练权重点估计。",
        "- 当前主模型队列不等于完整消融。严格消融还包括 G0→G1→G2→G3 的三seed训练、同checkpoint solver/steps、Top-K/renewal、LQCR验证选择、困难子集、蒸馏和统一A6000效率复评。",
        "",
        "## 数据集与统一协议",
        "",
        "| Dataset ID | Train / Val / Test | Annotation SHA | 用途与限制 |",
        "|---|---:|---|---|",
        "| `D1_INHOUSE1700_V1` | 1190 / 170 / 340 | train `318120af…`; val `57466f1f…`; test `52e8868d…` | 纯自建；1350个stem组；group-disjoint 70/10/20；无patient ID |",
        "| `D2` | 3500 / 500 / 1000 | train `218ae0…`; val `bcf0f9…`; test `110fd280…` | 公开台中队列；与D1I同为70/10/20 |",
        "| `D1_COMPOSITE2200_LEGACY` | 1540 / 440 / 220 | 历史哈希见数据库 | 含500张D2派生图；仅保留历史审计，不与新双数据集主实验混合 |",
        "",
        "统一精度协议：训练seed为独立初始化单位；validation seed固定42；每个训练run按validation规则选一个checkpoint；test inference seed固定42；必须记录mAP/AP50/AP75/AP_S/AP_M/AP_L。效率仅在独占Ross A6000、batch=1、固定precision/warm-up/timed iterations/计时边界下比较。",
        "",
        "## 状态与证据类型",
        "",
        "- `COMPLETED_VERIFIED`：三独立训练或配对证据已验证，可进入主表。",
        "- `COMPLETED_FIXED_CHECKPOINT`：只证明给定权重下的推理行为，不能称训练复现。",
        "- `COMPLETED_POINT_ESTIMATE`：一个训练checkpoint，仅能报点估计。",
        "- `COMPLETED_DESCRIPTIVE/DIAGNOSTIC`：历史或机制证据，不用于主SOTA因果结论。",
        "- `PLANNED/BLOCKED_*`：尚未获得合格最终证据。",
        "",
    ]
    sections = [
        ("D1_INHOUSE1700_V1", "纯自建数据集：完整模型、消融与部署计划"),
        ("D2", "Dataset 2：已完成证据与严格补跑缺口"),
        ("D1_COMPOSITE2200_LEGACY", "历史复合Dataset 1：非对照归档"),
    ]
    for dataset, title in sections:
        lines.extend([f"## {title}", ""])
        dataset_rows = by_dataset[dataset]
        for layer in dict.fromkeys(x["layer"] for x in dataset_rows):
            group = [x for x in dataset_rows if x["layer"] == layer]
            lines.extend([f"### {layer}", "", markdown_table(group), ""])
            for x in group:
                lines.append(f"- `{x['ledger_id']}` 参数：{x['parameters']} 验收：{x['acceptance_gate']} 论文用途：{x['paper_role']}" + (f" 备注：{x['notes']}" if x['notes'] else ""))
            lines.append("")
    lines.extend([
        "## 两数据集逐项对照与最小投稿闭环",
        "",
        "| 论文问题 | D1I要求 | D2要求 | 当前闭环 |",
        "|---|---|---|---|",
        "| KaryoFlow是否优于检测基线 | 6模型×3训练seed | KaryoFlow已有3；五基线补2seed | 未闭合 |",
        "| RF本身是否有效 | G0/G1三seed严格配对 | G0/G1严格重跑 | 未闭合 |",
        "| shifted schedule是否有效 | G1/G2三seed | G1/G2三seed | 未闭合 |",
        "| AdaLN-Zero是否有效 | G2/G3三seed | G2/G3三seed | 未闭合 |",
        "| DPM++是否有效 | G3同权重Euler/Heun/DPM++ | 已有fixed1；补train3 | 条件性闭合 |",
        "| LQCR是否有效 | 3 parent-child配对 | 已完成3配对 | D1I进行后闭合 |",
        "| 小/重叠目标机制 | D1I预测后分析 | D2已完成诊断 | 单数据集条件性 |",
        "| 部署收益 | H3×3 + A6000测速 | H3目前单parent；补train3 | 未闭合 |",
        "| GACS | D1I test+速度 | D2 test+速度 | 未闭合；不得作主精度模块 |",
        "",
        "## 运行和登记流程",
        "",
        "1. 先在本总账和 `experiment_ledger` 中存在唯一 `ledger_id`。",
        "2. 训练任务再创建 `train_run_registry.train_run_id`，固定dataset manifest SHA、config SHA、training seed、executor与work directory。",
        "3. SwanLab项目按数据集计划登记；D1I统一使用 `KaryoFlow-Self1700`，run名称 `{variant}_seed{training_seed}`。",
        "4. checkpoint只能由validation选择；test不能改变配置、beta、epoch或阈值。",
        "5. test保存原始预测、日志、resolved config与SHA；独立pycocotools复算六指标。",
        "6. 先写immutable run-evidence JSON，再幂等导入数据库；审计PASS后才能设paper eligible。",
        "7. 论文表格/摘要数字必须反向链接controlled result或本总账中的artifact ID。",
        "",
        "## 已知限制与下一步",
        "",
        "- 新D1I的三条KaryoFlow是正式SwanLab跟踪run；先前未跟踪尝试已归档为invalid。",
        "- D1I DiffusionDet外部基线继承batch=4；严格G0另设batch=2，二者不可混为同一因果消融。",
        "- D2历史A0→A4链跨checkpoint且seed口径混合，只能作为描述性背景；严格链需要重跑G0-G2。",
        "- 固定checkpoint三推理seed不得替代三训练seed。",
        "- 未取得patient/specimen ID，因此D1I只能称acquisition-stem group-disjoint，不能称patient-disjoint。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ledger_id", "dataset_id", "layer", "family", "variant", "execution_kind",
        "run_count", "training_seeds", "inference_seeds", "status", "priority",
        "replication_unit", "config_path", "config_status", "parameters", "executor_plan",
        "work_dir_template", "train_run_ids", "result_family", "evidence_artifact_ids",
        "swanlab_project", "swanlab_run_template", "parent_ledger_id", "paper_role",
        "acceptance_gate", "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, "") for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DB_DEFAULT)
    parser.add_argument("--doc", type=Path, default=DOC_DEFAULT)
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)
    rows = build_rows()
    enrich_live_ids(conn, rows)
    upsert(conn, rows)
    manifest_sha = write_manifest(rows, args.manifest)
    write_doc(rows, manifest_sha, args.doc)
    write_csv(rows, args.csv)
    print(json.dumps({
        "status": "PASS", "ledger_rows": len(rows),
        "expanded_runs": sum(x["run_count"] for x in rows),
        "manifest_sha256": manifest_sha,
        "doc": str(args.doc.relative_to(ROOT)), "csv": str(args.csv.relative_to(ROOT)),
        "manifest": str(args.manifest.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
