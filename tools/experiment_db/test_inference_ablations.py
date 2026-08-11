#!/usr/bin/env python3
"""Run locked inference-only ablations on the Dataset-2 held-out test set.

The validation set remains the source of hyperparameter selection. This script
confirms the already selected solver, proposal-retention, renewal, and LQCR
settings on the held-out test set without changing checkpoint weights.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

import d2_test_unified as evaluator


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "tools/experiment_db/experiments.db"
BASE_A1_CONFIG = (
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/"
    "a1_rf_heun_24obj.py"
)
BASE_A1_CHECKPOINT = "work_dirs/a1_rf_heun_24obj/best_coco_bbox_mAP_epoch_62.pth"
BASE_A4_CONFIG = (
    "experiments/configs/ldmdet/directions/mainline_ablation_24obj/"
    "a4_dpm_pp_24obj.py"
)
BASE_A4_CHECKPOINT = "work_dirs/a4_dpm_pp_24obj/best_coco_bbox_mAP_epoch_117.pth"
LQCR_CONFIG = (
    "experiments/configs/ldmdet/directions/capr/"
    "capr_quality_final_only_24obj.py"
)
LQCR_CHECKPOINT = "work_dirs/capr_quality_only_24obj/best_coco_bbox_mAP_epoch_2.pth"


def model(label: str, config: str, checkpoint: str, **bbox_head: object) -> dict:
    return {
        "label": label,
        "config": config,
        "checkpoint": checkpoint,
        "overrides": {"model": {"bbox_head": bbox_head}},
    }


SUITES = {
    "solver": {
        f"{solver}_{steps}": model(
            f"{solver.replace('_', ' ').title()} / {steps} step",
            BASE_A1_CONFIG,
            BASE_A1_CHECKPOINT,
            solver_type=solver,
            sampling_timesteps=steps,
        )
        for solver in ("euler", "heun", "dpm_solver_pp")
        for steps in (1, 2, 3, 4)
    },
    "topk_renewal": {
        f"k{k}_renewal_{'on' if renewal else 'off'}": model(
            f"K={k}, renewal {'on' if renewal else 'off'}",
            BASE_A4_CONFIG,
            BASE_A4_CHECKPOINT,
            topk_pruning_enabled=(k < 500),
            topk_k=k,
            topk_pruning_step=0,
            box_renewal=renewal,
        )
        for k in (500, 300, 200, 150, 100)
        for renewal in (True, False)
    },
    "lqcr_beta": {
        f"beta_{str(beta).replace('.', 'p')}": model(
            f"LQCR beta={beta:g}",
            LQCR_CONFIG,
            LQCR_CHECKPOINT,
            quality_score_beta=beta,
            quality_calibration_mode="final_only",
        )
        for beta in (0.0, 0.25, 0.5, 1.0, 2.0)
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def high_iou_ap(prediction_path: Path, annotation_path: Path) -> dict[str, float]:
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(annotation_path))
    predictions = ground_truth.loadRes(str(prediction_path))
    evaluation = COCOeval(ground_truth, predictions, "bbox")
    evaluation.params.maxDets = [1, 10, 100]
    evaluation.evaluate()
    evaluation.accumulate()
    precision = evaluation.eval["precision"]
    output = {}
    for threshold in (0.90, 0.95):
        index = int(np.argmin(np.abs(evaluation.params.iouThrs - threshold)))
        values = precision[index, :, :, 0, 2]
        valid = values[values > -1]
        output[f"AP{int(threshold * 100)}"] = float(valid.mean())
    return output


def configure_suite(name: str) -> None:
    evaluator.configure_dataset("d2")
    evaluator.MODELS = SUITES[name]
    evaluator.RESULT_ROOT = ROOT / f"results/d2_test_inference_ablations/{name}"
    evaluator.FAMILY = f"{name}_d2_test"
    evaluator.ARTIFACT_PREFIX = f"d2-test-{name.replace('_', '-')}"
    evaluator.PROTOCOL_VERSION = f"d2-test-{name.replace('_', '-')}-v1"


def finalize(records: list[dict], groups: list[str], profile: dict) -> tuple[Path, str]:
    compact = []
    for item in records:
        summary = item["summary"]
        metrics = dict(summary["metrics_exact"])
        if item["group"] == "lqcr_beta":
            prediction = (
                evaluator.RESULT_ROOT.parent
                / item["group"]
                / summary["protocol"]["model_id"]
                / summary["protocol_hash"]
                / "predictions.bbox.json"
            )
            metrics.update(high_iou_ap(prediction, evaluator.ANN_PATH))
        compact.append({
            "group": item["group"],
            "model_id": summary["protocol"]["model_id"],
            "model_label": summary["protocol"]["model_label"],
            "protocol_hash": summary["protocol_hash"],
            "protocol": summary["protocol"],
            "metrics": metrics,
            "source_summary": item["source_summary"],
            "source_summary_sha256": sha256_file(ROOT / item["source_summary"]),
        })
    compact.sort(key=lambda row: (row["group"], row["model_id"]))
    aggregate = {
        "description": (
            "Locked inference-only confirmations on the Dataset-2 held-out test set; "
            "validation results selected the hyperparameters."
        ),
        "status": "verified",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {key: value for key, value in profile.items()
                    if key not in {"image_ids", "category_ids"}},
        "groups": groups,
        "seed": evaluator.SEED,
        "results": compact,
    }
    payload = (json.dumps(aggregate, indent=2, sort_keys=True) + "\n").encode()
    digest = sha256_bytes(payload)
    relative_path = Path(
        f"tools/experiment_db/evidence_sources/d2_test_inference_ablations_"
        f"{digest[:12]}.json"
    )
    output_path = ROOT / relative_path
    if output_path.exists() and output_path.read_bytes() != payload:
        raise RuntimeError(f"immutable aggregate collision: {output_path}")
    output_path.write_bytes(payload)
    artifact_id = f"d2-test-inference-ablations-{digest[:12]}"

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT OR IGNORE INTO evidence_artifact
               (artifact_id, server, path, sha256, kind, generated_at, status, notes)
               VALUES (?, 'ross', ?, ?, 'inference_ablation_test_aggregate', ?,
                       'verified', ?)""",
            (artifact_id, str(relative_path), digest, aggregate["completed_at"],
             "Accuracy-only evaluation; efficiency remains Ross-A6000 controlled."),
        )
        for row in compact:
            group_prefix = f"d2-test-{row['group'].replace('_', '-')}"
            short_hash = row["protocol_hash"][:12]
            for metric, value in row["metrics"].items():
                result_id = (
                    f"{group_prefix}-{row['model_id']}-{metric.lower()}-{short_hash}"
                )
                protocol_json = json.dumps(row["protocol"], sort_keys=True)
                existing = connection.execute(
                    "SELECT value, protocol_json FROM controlled_result WHERE result_id=?",
                    (result_id,),
                ).fetchone()
                if existing and existing != (value, protocol_json):
                    raise RuntimeError(f"controlled-result collision: {result_id}")
                if existing:
                    connection.execute(
                        "UPDATE controlled_result SET artifact_id=? WHERE result_id=?",
                        (artifact_id, result_id),
                    )
                else:
                    connection.execute(
                        """INSERT INTO controlled_result
                           (result_id, family, variant, dataset, split, seed, metric,
                            value, unit, baseline_result_id, delta, protocol_json,
                            artifact_id, evidence_level, paper_eligible, notes)
                           VALUES (?, ?, ?, 'D2', 'test', ?, ?, ?, 'absolute',
                                   NULL, NULL, ?, ?, 'controlled', 1, ?)""",
                        (result_id, f"{row['group']}_d2_test", row["model_label"],
                         str(evaluator.SEED), metric, value, protocol_json,
                         artifact_id,
                         "Locked test confirmation; not used for hyperparameter selection."),
                    )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return output_path, artifact_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups", nargs="+", choices=sorted(SUITES), default=sorted(SUITES)
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    evaluator.configure_dataset("d2")
    profile = evaluator.load_annotation_profile()
    code_sha = evaluator.code_fingerprint()
    all_records = []
    for group in args.groups:
        configure_suite(group)
        print(f"\n[{group}] {len(evaluator.MODELS)} locked test variants")
        for name, spec in evaluator.MODELS.items():
            summary = evaluator.run_model(
                name, spec, profile, code_sha, args.gpu_id, args.plan
            )
            if not args.plan:
                source = (
                    evaluator.RESULT_ROOT / name / summary["protocol_hash"] / "summary.json"
                )
                all_records.append({
                    "group": group,
                    "summary": summary,
                    "source_summary": str(source.relative_to(ROOT)),
                })
    if not args.plan:
        path, artifact = finalize(all_records, args.groups, profile)
        print(f"\nAggregate evidence: {path.relative_to(ROOT)} ({artifact})")


if __name__ == "__main__":
    main()
