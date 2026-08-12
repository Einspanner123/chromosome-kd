#!/usr/bin/env python3
"""Unified, resumable chromosome test evaluation with immutable evidence.

The run identity covers the resolved configuration, checkpoint, test annotation,
inference code, seed, and evaluator protocol. A completed run is reused only
after every stored artifact hash has been verified. Metrics are recomputed from
the persisted COCO predictions and checked against MMDetection's output before
they are registered as paper-eligible controlled results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "tools/experiment_db/experiments.db"
RESULT_ROOT: Path
ANN_PATH: Path
DATA_ROOT: str
DATASET_ID: str
EXPECTED_IMAGES: int
EXPECTED_CATEGORIES: int
FAMILY: str
ARTIFACT_PREFIX: str
SEED = 42
PROTOCOL_VERSION: str

MODEL_SETS = {
    "d1": {
        "karyoflow": {
            "label": "KaryoFlow",
            "config": "work_dirs/paired_clean/a4_random_chr2024_seed42/a4_clean_test.py",
            "checkpoint": "work_dirs/paired_clean/a4_random_chr2024_seed42/best_coco_bbox_mAP_epoch_62.pth",
        },
        "karyoflow_k200": {
            "label": "KaryoFlow K=200",
            "config": "work_dirs/paired_clean/a4_random_chr2024_seed42/a4_clean_test.py",
            "checkpoint": "work_dirs/paired_clean/a4_random_chr2024_seed42/best_coco_bbox_mAP_epoch_62.pth",
            "overrides": {
                "model": {
                    "bbox_head": {
                        "topk_pruning_enabled": True,
                        "topk_k": 200,
                        "topk_pruning_step": 0,
                    }
                }
            },
        },
        "karyoflow_k100": {
            "label": "KaryoFlow K=100",
            "config": "work_dirs/paired_clean/a4_random_chr2024_seed42/a4_clean_test.py",
            "checkpoint": "work_dirs/paired_clean/a4_random_chr2024_seed42/best_coco_bbox_mAP_epoch_62.pth",
            "overrides": {
                "model": {
                    "bbox_head": {
                        "topk_pruning_enabled": True,
                        "topk_k": 100,
                        "topk_pruning_step": 0,
                    }
                }
            },
        },
        "karyoflow_lqcr": {
            "label": "KaryoFlow+LQCR",
            "config": "work_dirs/paired_clean/lqcr_final_only_chr2024_seed42/capr_quality_final_only_paired_clean_chr2024.py",
            "checkpoint": "work_dirs/paired_clean/lqcr_final_only_chr2024_seed42/best_coco_bbox_mAP_epoch_5.pth",
        },
        "dino_r50": {
            "label": "DINO R50",
            "config": "experiments/configs/baselines/benchmark/dino_r50.py",
            "checkpoint": "work_dirs/baselines/dino_r50_20240904/epoch_107.pth",
        },
        "rtmdet_l": {
            "label": "RTMDet-L",
            "config": "experiments/configs/baselines/benchmark/rtmdet_l.py",
            "checkpoint": "work_dirs/baselines/rtmdet_l_20240904/best_coco_bbox_mAP_epoch_52.pth",
        },
        "cascade_rcnn": {
            "label": "Cascade R-CNN",
            "config": "experiments/configs/baselines/benchmark/cascade_rcnn_r50.py",
            "checkpoint": "work_dirs/baselines/cascade_rcnn_r50_20240904/best_coco_bbox_mAP_epoch_86.pth",
        },
        "diffusiondet": {
            "label": "DiffusionDet",
            "config": "experiments/configs/baselines/diffusiondet_ddpm.py",
            "checkpoint": "work_dirs/multi_seed_aug/ddpm/seed_42/best_coco_bbox_mAP_epoch_79.pth",
        },
        "yolox_s": {
            "label": "YOLOX-S",
            "config": "experiments/configs/baselines/benchmark/yolox_s.py",
            "checkpoint": "work_dirs/baselines/yolox_s_20240904/best_coco_bbox_mAP_epoch_150.pth",
        },
    },
    "d2": {
    "karyoflow_trainrun_0": {
        "label": "KaryoFlow (train run 0)",
        "config": "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py",
        "checkpoint": "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth",
        "training_seed": 335778785,
        "replication_unit": "independent_training_seed",
    },
    "karyoflow_trainrun_1": {
        "label": "KaryoFlow (train run 1)",
        "config": "work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/a4_dpm_pp_24obj_multiseed.py",
        "checkpoint": "work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth",
        "training_seed": 790448076,
        "replication_unit": "independent_training_seed",
    },
    "karyoflow_trainrun_2": {
        "label": "KaryoFlow (train run 2)",
        "config": "work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/a4_dpm_pp_24obj_multiseed.py",
        "checkpoint": "work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth",
        "training_seed": 1342286018,
        "replication_unit": "independent_training_seed",
    },
    "karyoflow_lqcr_trainrun_0": {
        "label": "KaryoFlow+LQCR (train run 0)",
        "config": "experiments/configs/ldmdet/directions/capr/capr_quality_final_only_24obj.py",
        "checkpoint": "work_dirs/capr_quality_only_24obj/best_coco_bbox_mAP_epoch_2.pth",
        "training_seed": 335778785,
        "replication_unit": "paired_final_stage_intervention",
        "parent_model_id": "karyoflow_trainrun_0",
        "parent_checkpoint": "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth",
    },
    "karyoflow_lqcr_trainrun_1": {
        "label": "KaryoFlow+LQCR (train run 1)",
        "config": "experiments/configs/ldmdet/directions/capr/paper_train3_lqcr_run1_test.py",
        "checkpoint": "work_dirs/paper_d2_lqcr_trainrun_1/best_coco_bbox_mAP.pth",
        "training_seed": 790448076,
        "replication_unit": "paired_final_stage_intervention",
        "parent_model_id": "karyoflow_trainrun_1",
        "parent_checkpoint": "work_dirs/multi_seed/a4_dpm_pp_24obj/seed_123/best_coco_bbox_mAP_epoch_62.pth",
    },
    "karyoflow_lqcr_trainrun_2": {
        "label": "KaryoFlow+LQCR (train run 2)",
        "config": "experiments/configs/ldmdet/directions/capr/paper_train3_lqcr_run2_test.py",
        "checkpoint": "work_dirs/paper_d2_lqcr_trainrun_2/best_coco_bbox_mAP.pth",
        "training_seed": 1342286018,
        "replication_unit": "paired_final_stage_intervention",
        "parent_model_id": "karyoflow_trainrun_2",
        "parent_checkpoint": "work_dirs/multi_seed/a4_dpm_pp_24obj/seed_789/best_coco_bbox_mAP_epoch_72.pth",
    },
    "karyoflow": {
        "label": "KaryoFlow",
        "config": "experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_dpm_pp_24obj.py",
        "checkpoint": "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth",
    },
    "karyoflow_lqcr": {
        "label": "KaryoFlow+LQCR",
        "config": "experiments/configs/ldmdet/directions/capr/capr_quality_final_only_24obj.py",
        "checkpoint": "work_dirs/capr_quality_only_24obj/best_coco_bbox_mAP_epoch_2.pth",
    },
    "dino_r50": {
        "label": "DINO R50",
        "config": "experiments/configs/baselines/benchmark_24obj/dino_r50.py",
        "checkpoint": "work_dirs/baselines/dino_r50_24obj/best_coco_bbox_mAP_epoch_102.pth",
    },
    "rtmdet_l": {
        "label": "RTMDet-L",
        "config": "experiments/configs/baselines/benchmark_24obj/rtmdet_l.py",
        "checkpoint": "work_dirs/baselines/rtmdet_l_24obj/epoch_86.pth",
    },
    "cascade_rcnn": {
        "label": "Cascade R-CNN",
        "config": "experiments/configs/baselines/benchmark_24obj/cascade_rcnn_r50.py",
        "checkpoint": "work_dirs/baselines/cascade_rcnn_r50/best_coco_bbox_mAP_epoch_72.pth",
    },
    "diffusiondet": {
        "label": "DiffusionDet",
        "config": "experiments/configs/baselines/benchmark_24obj/diffusiondet_ddpm.py",
        "checkpoint": "work_dirs/baselines/diffusiondet_24obj/best_coco_bbox_mAP_epoch_26.pth",
    },
    "yolox_s": {
        "label": "YOLOX-S",
        "config": "experiments/configs/baselines/benchmark_24obj/yolox_s.py",
        "checkpoint": "work_dirs/baselines/yolox_s/best_coco_bbox_mAP_epoch_200.pth",
    },
    },
}

MODELS: dict = {}


def configure_dataset(dataset: str) -> None:
    global RESULT_ROOT, ANN_PATH, DATA_ROOT, DATASET_ID
    global EXPECTED_IMAGES, EXPECTED_CATEGORIES, FAMILY
    global ARTIFACT_PREFIX, PROTOCOL_VERSION, MODELS
    if dataset == "d1":
        DATASET_ID = "D1"
        DATA_ROOT = "data/Chromosome20240904_NoAug_NoResize_coco/"
        EXPECTED_IMAGES = 220
    else:
        DATASET_ID = "D2"
        DATA_ROOT = "data/24_chromosomes_object/coco/"
        EXPECTED_IMAGES = 1000
    EXPECTED_CATEGORIES = 24
    RESULT_ROOT = ROOT / f"results/{dataset}_test_unified"
    ANN_PATH = ROOT / DATA_ROOT / "test/_annotations.coco.json"
    FAMILY = f"sota_{dataset}_test"
    ARTIFACT_PREFIX = f"{dataset}-test"
    PROTOCOL_VERSION = f"{dataset}-test-coco-v1"
    MODELS = MODEL_SETS[dataset]

METRIC_KEYS = {
    "mAP": "coco/bbox_mAP",
    "AP50": "coco/bbox_mAP_50",
    "AP75": "coco/bbox_mAP_75",
    "AP_S": "coco/bbox_mAP_s",
    "AP_M": "coco/bbox_mAP_m",
    "AP_L": "coco/bbox_mAP_l",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def code_fingerprint() -> str:
    paths = [ROOT / "experiments/runners/test.py"]
    paths.extend(sorted((ROOT / "ldmdet").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def patch_dataset(dataset: object) -> None:
    if not isinstance(dataset, dict):
        return
    if "dataset" in dataset:
        patch_dataset(dataset["dataset"])
    if "datasets" in dataset and isinstance(dataset["datasets"], list):
        for item in dataset["datasets"]:
            patch_dataset(item)
    if "ann_file" in dataset:
        dataset["data_root"] = DATA_ROOT
        dataset["ann_file"] = "test/_annotations.coco.json"
        dataset["data_prefix"] = {"img": "test/"}
        dataset["test_mode"] = True


def patch_evaluator(evaluator: object, prediction_prefix: str) -> None:
    if isinstance(evaluator, list):
        for item in evaluator:
            patch_evaluator(item, prediction_prefix)
        return
    if not isinstance(evaluator, dict):
        return
    evaluator["ann_file"] = DATA_ROOT + "test/_annotations.coco.json"
    evaluator["metric"] = "bbox"
    evaluator["format_only"] = False
    evaluator["proposal_nums"] = (1, 10, 100)
    evaluator["outfile_prefix"] = prediction_prefix


def resolved_config(model: dict, prediction_prefix: str):
    from mmengine.config import Config

    cfg = Config.fromfile(str(ROOT / model["config"]))

    def merge_override(target: object, updates: dict) -> None:
        for key, value in updates.items():
            if isinstance(value, dict):
                merge_override(target[key], value)
            else:
                target[key] = value

    merge_override(cfg, model.get("overrides", {}))
    patch_dataset(cfg.test_dataloader.get("dataset", {}))
    cfg.test_dataloader["batch_size"] = 1
    cfg.test_dataloader["drop_last"] = False
    cfg.test_dataloader["persistent_workers"] = False
    cfg.test_dataloader["num_workers"] = min(int(cfg.test_dataloader.get("num_workers", 2)), 2)
    patch_evaluator(cfg.test_evaluator, prediction_prefix)
    # Keep evaluation offline and prevent external logging from changing evidence.
    cfg["vis_backends"] = [dict(type="LocalVisBackend")]
    cfg["visualizer"] = dict(
        type="DetLocalVisualizer",
        vis_backends=[dict(type="LocalVisBackend")],
        name="visualizer",
    )
    if "default_hooks" in cfg and "visualization" in cfg.default_hooks:
        cfg.default_hooks.visualization["draw"] = False
    # EMAHook is a training-time hook. Loading an already exported YOLOX best
    # checkpoint under TestLoop leaves the hook uninitialized and crashes in
    # after_load_checkpoint; removing it does not alter checkpoint weights.
    if "custom_hooks" in cfg:
        cfg.custom_hooks = [
            hook for hook in cfg.custom_hooks
            if not (isinstance(hook, dict) and hook.get("type") == "EMAHook")
        ]
    cfg.work_dir = f"__{DATASET_ID}_TEST_EVIDENCE_DIR__"
    return cfg


def load_annotation_profile() -> dict:
    source = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    image_ids = [int(row["id"]) for row in source["images"]]
    category_ids = [int(row["id"]) for row in source["categories"]]
    if len(image_ids) != EXPECTED_IMAGES or len(set(image_ids)) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"{DATASET_ID} test must contain {EXPECTED_IMAGES} unique images, "
            f"found {len(set(image_ids))}")
    if len(category_ids) != EXPECTED_CATEGORIES or len(set(category_ids)) != EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"{DATASET_ID} test must contain {EXPECTED_CATEGORIES} unique categories, "
            f"found {len(set(category_ids))}")
    missing = [row.get("file_name", "") for row in source["images"]
               if not (ANN_PATH.parent / row.get("file_name", "")).is_file()]
    if missing:
        raise RuntimeError(
            f"{DATASET_ID} test has {len(missing)} missing images; first={missing[0]!r}")
    return {
        "annotation": str(ANN_PATH.relative_to(ROOT)),
        "annotation_sha256": sha256_file(ANN_PATH),
        "images": len(image_ids),
        "annotations": len(source["annotations"]),
        "categories": len(category_ids),
        "image_ids": set(image_ids),
        "category_ids": set(category_ids),
    }


def protocol_for(name: str, model: dict, cfg_text: str, profile: dict,
                 code_sha: str) -> dict:
    checkpoint = ROOT / model["checkpoint"]
    config = ROOT / model["config"]
    protocol = {
        "version": PROTOCOL_VERSION,
        "model_id": name,
        "model_label": model["label"],
        "dataset": DATASET_ID,
        "split": "test",
        "seed": SEED,
        "inference_seed": SEED,
        "training_seed": model.get("training_seed"),
        "replication_unit": model.get(
            "replication_unit", "fixed_checkpoint_inference_seed"),
        "images": profile["images"],
        "annotations": profile["annotations"],
        "categories": profile["categories"],
        "annotation_path": profile["annotation"],
        "annotation_sha256": profile["annotation_sha256"],
        "config_path": model["config"],
        "config_source_sha256": sha256_file(config),
        "resolved_config_sha256": sha256_text(cfg_text),
        "checkpoint_path": model["checkpoint"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "inference_code_sha256": code_sha,
        "evaluator": "COCO bbox",
        "max_dets": 100,
        "test_batch_size": 1,
        "deterministic_seed": True,
    }
    if model.get("parent_checkpoint"):
        parent = ROOT / model["parent_checkpoint"]
        protocol.update({
            "parent_model_id": model["parent_model_id"],
            "parent_checkpoint_path": model["parent_checkpoint"],
            "parent_checkpoint_sha256": sha256_file(parent),
        })
    return protocol


def canonical_hash(data: dict) -> str:
    return sha256_text(json.dumps(data, sort_keys=True, separators=(",", ":")))


def parse_framework_metrics(log_text: str) -> dict:
    metrics = {}
    for metric, key in METRIC_KEYS.items():
        matches = re.findall(rf"^\s*{re.escape(key)}:\s*([0-9.]+)\s*$", log_text, re.MULTILINE)
        if matches:
            metrics[metric] = float(matches[-1])
    if set(metrics) != set(METRIC_KEYS):
        missing = sorted(set(METRIC_KEYS) - set(metrics))
        raise RuntimeError(f"framework metrics missing: {missing}")
    return metrics


def recompute_metrics(prediction_path: Path, profile: dict) -> tuple[dict, dict]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not predictions:
        raise RuntimeError("prediction file is empty")
    pred_image_ids = {int(row["image_id"]) for row in predictions}
    pred_category_ids = {int(row["category_id"]) for row in predictions}
    unknown_images = pred_image_ids - profile["image_ids"]
    unknown_categories = pred_category_ids - profile["category_ids"]
    if unknown_images or unknown_categories:
        raise RuntimeError(
            f"predictions contain unknown IDs: images={len(unknown_images)}, "
            f"categories={len(unknown_categories)}")

    coco_gt = COCO(str(ANN_PATH))
    coco_dt = coco_gt.loadRes(str(prediction_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    values = evaluator.stats[:6]
    metrics = dict(zip(METRIC_KEYS, (float(value) for value in values)))
    coverage = {
        "prediction_rows": len(predictions),
        "images_with_predictions": len(pred_image_ids),
        "test_images": profile["images"],
        "unknown_image_ids": len(unknown_images),
        "unknown_category_ids": len(unknown_categories),
    }
    return metrics, coverage


def validate_metrics(framework: dict, exact: dict, log_text: str, profile: dict) -> None:
    if f"[{profile['images']}/{profile['images']}]" not in log_text:
        raise RuntimeError(
            f"framework log does not confirm evaluation of all {profile['images']} test images")
    for metric in METRIC_KEYS:
        value = exact[metric]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"invalid {metric}={value}")
        # MMDetection reports three decimals. Exact COCO recomputation must round identically.
        if abs(framework[metric] - value) > 0.00051:
            raise RuntimeError(
                f"metric mismatch for {metric}: framework={framework[metric]}, exact={value}")


def verify_existing(summary_path: Path, protocol_hash: str) -> dict | None:
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "verified" or summary.get("protocol_hash") != protocol_hash:
        return None
    for key, digest in summary.get("artifact_sha256", {}).items():
        path = summary_path.parent / key
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"stored evidence hash mismatch: {path}")
    return summary


def register_database(summary_path: Path, summary: dict) -> str:
    artifact_sha = sha256_file(summary_path)
    short_key = summary["protocol_hash"][:12]
    model_id = summary["protocol"]["model_id"]
    artifact_id = f"{ARTIFACT_PREFIX}-{model_id}-{short_key}"
    relative_path = str(summary_path.relative_to(ROOT))
    timestamp = summary["completed_at"]
    protocol_json = json.dumps(summary["protocol"], sort_keys=True)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    backup = DB_PATH.with_name(
        f"experiments.db.bak_before_{DATASET_ID.lower()}test_{dt.datetime.now():%Y%m%d}")
    if not backup.exists():
        shutil.copy2(DB_PATH, backup)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT path, sha256 FROM evidence_artifact WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        if existing and existing != (relative_path, artifact_sha):
            raise RuntimeError(f"artifact ID collision for {artifact_id}")
        conn.execute(
            """INSERT OR IGNORE INTO evidence_artifact
               (artifact_id, server, path, sha256, kind, generated_at, status, notes)
               VALUES (?, 'ross', ?, ?, 'unified_test_evaluation', ?, 'verified', ?)""",
            (artifact_id, relative_path, artifact_sha, timestamp,
             "Exact COCO recomputation cross-checked against MMDetection output."),
        )
        for metric, value in summary["metrics_exact"].items():
            result_id = f"{ARTIFACT_PREFIX}-{model_id}-{metric.lower()}-{short_key}"
            row = conn.execute(
                "SELECT value, protocol_json, artifact_id FROM controlled_result WHERE result_id=?",
                (result_id,),
            ).fetchone()
            # A completed family aggregate intentionally repoints the row from
            # this per-run artifact to the immutable aggregate. Reuse remains
            # valid when the numerical value and full protocol are identical.
            if row and row[:2] != (value, protocol_json):
                raise RuntimeError(f"controlled-result collision for {result_id}")
            conn.execute(
                """INSERT OR IGNORE INTO controlled_result
                   (result_id, family, variant, dataset, split, seed, metric, value,
                    unit, baseline_result_id, delta, protocol_json, artifact_id,
                    evidence_level, paper_eligible, notes)
                   VALUES (?, ?, ?, ?, 'test', ?, ?, ?, 'absolute',
                           NULL, NULL, ?, ?, 'controlled', 1, ?)""",
                (result_id, FAMILY, summary["protocol"]["model_label"], DATASET_ID,
                 str(summary["protocol"].get("training_seed") or SEED), metric,
                 value, protocol_json, artifact_id,
                 f"Unified {EXPECTED_IMAGES}-image {DATASET_ID} test evaluation."),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return artifact_id


def finalize_aggregate(summaries: list[dict], profile: dict) -> tuple[Path, str]:
    """Create an immutable, Git-sized evidence source and repoint result rows."""
    records = []
    for summary in summaries:
        model_id = summary["protocol"]["model_id"]
        summary_path = (
            RESULT_ROOT / model_id / summary["protocol_hash"] / "summary.json"
        )
        records.append({
            "model_id": model_id,
            "model_label": summary["protocol"]["model_label"],
            "protocol_hash": summary["protocol_hash"],
            "protocol": summary["protocol"],
            "metrics": summary["metrics_exact"],
            "framework_metrics": summary["metrics_framework"],
            "coverage": summary["coverage"],
            "source_summary": str(summary_path.relative_to(ROOT)),
            "source_summary_sha256": sha256_file(summary_path),
            "source_artifacts_sha256": summary["artifact_sha256"],
        })
    records.sort(key=lambda row: row["model_id"])
    aggregate_kind = (
        "unified_test_aggregate" if len(records) == 7
        else "supplemental_test_aggregate"
    )
    aggregate = {
        "description": (
            f"Unified {DATASET_ID} test evaluation for Table VI detectors"
            if aggregate_kind == "unified_test_aggregate"
            else f"Supplemental {DATASET_ID} held-out test operating-point evaluation"
        ),
        "status": "verified",
        "protocol_version": PROTOCOL_VERSION,
        "completed_at": max(row["completed_at"] for row in summaries),
        "dataset": {
            key: value for key, value in profile.items()
            if key not in {"image_ids", "category_ids"}
        },
        "quality_controls": [
            f"{EXPECTED_IMAGES} unique test image IDs and "
            f"{EXPECTED_CATEGORIES} category IDs verified",
            "COCO maxDets explicitly fixed to [1, 10, 100]",
            "persisted predictions independently recomputed with pycocotools",
            "framework and independent metrics agree after framework rounding",
            "config, checkpoint, annotation, code, log, and prediction hashes retained",
        ],
        "results": records,
    }
    payload = (json.dumps(aggregate, indent=2, sort_keys=True) + "\n").encode("utf-8")
    aggregate_sha = hashlib.sha256(payload).hexdigest()
    relative_path = Path(
        f"tools/experiment_db/evidence_sources/{DATASET_ID.lower()}_test_unified_"
        f"{aggregate_sha[:12]}.json"
    )
    aggregate_path = ROOT / relative_path
    if aggregate_path.exists() and aggregate_path.read_bytes() != payload:
        raise RuntimeError(f"immutable aggregate collision: {aggregate_path}")
    aggregate_path.write_bytes(payload)
    artifact_id = f"{ARTIFACT_PREFIX}-unified-{aggregate_sha[:12]}"

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT path, sha256 FROM evidence_artifact WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        expected = (str(relative_path), aggregate_sha)
        if existing and existing != expected:
            raise RuntimeError(f"aggregate artifact collision for {artifact_id}")
        conn.execute(
            """INSERT OR IGNORE INTO evidence_artifact
               (artifact_id, server, path, sha256, kind, generated_at, status, notes)
               VALUES (?, 'ross', ?, ?, ?, ?, 'verified', ?)""",
            (artifact_id, str(relative_path), aggregate_sha, aggregate_kind,
             aggregate["completed_at"],
             f"Immutable compact evidence; raw predictions remain under "
             f"results/{DATASET_ID.lower()}_test_unified/"),
        )
        conn.execute(
            "UPDATE evidence_artifact SET kind=? WHERE artifact_id=?",
            (aggregate_kind, artifact_id),
        )
        result_ids = []
        for row in records:
            short_key = row["protocol_hash"][:12]
            result_ids.extend(
                f"{ARTIFACT_PREFIX}-{row['model_id']}-{metric.lower()}-{short_key}"
                for metric in METRIC_KEYS
            )
        cursor = conn.execute(
            f"""UPDATE controlled_result SET artifact_id=?
                WHERE family=?
                  AND result_id IN ({','.join('?' for _ in result_ids)})""",
            (artifact_id, FAMILY, *result_ids),
        )
        expected_rows = len(records) * len(METRIC_KEYS)
        if cursor.rowcount != expected_rows:
            raise RuntimeError(
                f"expected to repoint {expected_rows} controlled results, got {cursor.rowcount}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return aggregate_path, artifact_id


def run_model(name: str, model: dict, profile: dict, code_sha: str,
              gpu_id: int, plan_only: bool) -> dict:
    config_path = ROOT / model["config"]
    checkpoint_path = ROOT / model["checkpoint"]
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing config/checkpoint for {name}")

    provisional_cfg = resolved_config(model, "__PREDICTION_PREFIX__")
    cfg_text = provisional_cfg.pretty_text
    protocol = protocol_for(name, model, cfg_text, profile, code_sha)
    protocol_hash = canonical_hash(protocol)
    run_dir = RESULT_ROOT / name / protocol_hash
    summary_path = run_dir / "summary.json"
    prediction_prefix = run_dir / "predictions"
    prediction_path = run_dir / "predictions.bbox.json"
    config_output = run_dir / "resolved_test_config.py"
    log_path = run_dir / "evaluation.log"

    print(f"[{name}] protocol={protocol_hash[:12]} checkpoint={protocol['checkpoint_sha256'][:12]}")
    if plan_only:
        return {"model": name, "status": "planned", "protocol_hash": protocol_hash}

    run_dir.mkdir(parents=True, exist_ok=True)
    existing = verify_existing(summary_path, protocol_hash)
    if existing:
        artifact_id = register_database(summary_path, existing)
        print(f"[{name}] REUSED verified evidence; artifact={artifact_id}")
        return existing

    cfg = resolved_config(model, str(prediction_prefix.relative_to(ROOT)))
    cfg.work_dir = str(run_dir.relative_to(ROOT))
    cfg.dump(str(config_output))

    command = [
        sys.executable,
        "experiments/runners/test.py",
        str(config_output.relative_to(ROOT)),
        "--checkpoint", model["checkpoint"],
        "--dataset", "test",
        "--gpu-id", str(gpu_id),
        "--seed", str(SEED),
        "--exp-name", f"{DATASET_ID.lower()}_test_{name}_{protocol_hash[:12]}",
    ]
    env = os.environ.copy()
    env.update({
        "SWANLAB_MODE": "disabled",
        "WANDB_MODE": "disabled",
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "PYTHONHASHSEED": str(SEED),
    })
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            if "Epoch(test)" in line or "bbox_mAP:" in line:
                print(f"[{name}] {line.rstrip()}")
        return_code = process.wait()
    elapsed = time.time() - t0
    if return_code != 0:
        raise RuntimeError(f"{name} evaluation failed with return code {return_code}; see {log_path}")
    if not prediction_path.is_file():
        raise RuntimeError(f"{name} did not persist {prediction_path}")

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    framework_metrics = parse_framework_metrics(log_text)
    exact_metrics, coverage = recompute_metrics(prediction_path, profile)
    validate_metrics(framework_metrics, exact_metrics, log_text, profile)

    summary = {
        "status": "verified",
        "protocol_hash": protocol_hash,
        "protocol": protocol,
        "started_at": started,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "command": command,
        "metrics_framework": framework_metrics,
        "metrics_exact": exact_metrics,
        "coverage": coverage,
        "artifact_sha256": {
            config_output.name: sha256_file(config_output),
            log_path.name: sha256_file(log_path),
            prediction_path.name: sha256_file(prediction_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_id = register_database(summary_path, summary)
    print(
        f"[{name}] VERIFIED mAP={exact_metrics['mAP']:.6f} "
        f"AP_S={exact_metrics['AP_S']:.6f} elapsed={elapsed:.1f}s artifact={artifact_id}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["d1", "d2"], default="d2")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    configure_dataset(args.dataset)
    selected_models = args.models if args.models is not None else list(MODELS)
    unknown = sorted(set(selected_models) - set(MODELS))
    if unknown:
        parser.error(
            f"unknown models for {args.dataset}: {unknown}; "
            f"choose from {sorted(MODELS)}")

    os.chdir(ROOT)
    profile = load_annotation_profile()
    code_sha = code_fingerprint()
    print(
        f"{DATASET_ID} test: images={profile['images']} "
        f"annotations={profile['annotations']} "
        f"categories={profile['categories']} annotation_sha={profile['annotation_sha256']}"
    )
    print(f"Inference code fingerprint: {code_sha}")

    summaries = []
    for name in selected_models:
        summaries.append(run_model(name, MODELS[name], profile, code_sha, args.gpu_id, args.plan))

    if not args.plan:
        completed = [row for row in summaries if row.get("status") == "verified"]
        if len(completed) != len(selected_models):
            raise RuntimeError(
                f"only {len(completed)}/{len(selected_models)} models verified")
        print(f"\nUnified {DATASET_ID} test results:")
        for row in completed:
            metrics = row["metrics_exact"]
            print(
                f"  {row['protocol']['model_label']:<20} "
                f"mAP={metrics['mAP']:.6f} AP50={metrics['AP50']:.6f} "
                f"AP75={metrics['AP75']:.6f} AP_S={metrics['AP_S']:.6f}"
            )
        aggregate_path, artifact_id = finalize_aggregate(completed, profile)
        print(
            f"Aggregate evidence: {aggregate_path.relative_to(ROOT)} "
            f"artifact={artifact_id}"
        )


if __name__ == "__main__":
    main()
