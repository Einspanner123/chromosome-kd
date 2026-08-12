#!/usr/bin/env python3
"""Image-cluster paired bootstrap for D2 LQCR versus DINO COCO mAP.

The estimand is the mean mAP of three independently trained LQCR detectors
minus the mean mAP of three inference-seed evaluations of the fixed DINO
checkpoint used in the paper comparison.  The bootstrap resamples test images;
it does not estimate training-run uncertainty for DINO.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import multiprocessing
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ROOT = Path(__file__).resolve().parents[3]
ANN = ROOT / "data/24_chromosomes_object/coco/test/_annotations.coco.json"
LQCR_AGG = ROOT / "tools/experiment_db/evidence_sources/d2_paired_lqcr_train3_4152cdbd5100.json"
PAPER_AGG = ROOT / "tools/experiment_db/evidence_sources/paper_test_multiseed_92e803b9772e.json"

_WORKER_IMAGE_IDS: list[int] | None = None
_WORKER_EVAL_RUNS: list[dict] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_from_summary(relative: str) -> Path:
    summary_path = ROOT / relative
    summary = json.loads(summary_path.read_text())
    prediction = summary_path.parent / "predictions.bbox.json"
    expected = summary["artifact_sha256"][prediction.name]
    actual = sha256(prediction)
    if actual != expected:
        raise RuntimeError(f"prediction SHA mismatch: {prediction}")
    return prediction


def resolve_inputs() -> tuple[list[dict], list[dict]]:
    paired = json.loads(LQCR_AGG.read_text())
    lqcr = []
    for source in paired["source_artifacts"]:
        source_path = ROOT / source["path"]
        if sha256(source_path) != source["sha256"]:
            raise RuntimeError(f"aggregate source SHA mismatch: {source_path}")
        doc = json.loads(source_path.read_text())
        for row in doc["results"]:
            if "lqcr_trainrun" not in row["model_id"]:
                continue
            lqcr.append({
                "replication_unit": "independent_detector_training_run",
                "training_seed": row["protocol"]["training_seed"],
                "inference_seed": row["protocol"]["inference_seed"],
                "checkpoint_sha256": row["protocol"]["checkpoint_sha256"],
                "summary_path": row["source_summary"],
                "summary_sha256": row["source_summary_sha256"],
                "prediction_path": str(prediction_from_summary(row["source_summary"]).relative_to(ROOT)),
            })
    paper = json.loads(PAPER_AGG.read_text())
    dino_row = next(row for row in paper["results"] if row["paper_id"] == "d2_sota_dino_r50")
    dino = []
    for run in dino_row["runs"]:
        summary_path = ROOT / run["source_summary"]
        if sha256(summary_path) != run["source_summary_sha256"]:
            raise RuntimeError(f"DINO summary SHA mismatch: {summary_path}")
        summary = json.loads(summary_path.read_text())
        dino.append({
            "replication_unit": "inference_seed_fixed_checkpoint",
            "training_seed": None,
            "inference_seed": run["seed"],
            "checkpoint_sha256": dino_row["checkpoint_sha256"],
            "summary_path": run["source_summary"],
            "summary_sha256": run["source_summary_sha256"],
            "prediction_path": str(prediction_from_summary(run["source_summary"]).relative_to(ROOT)),
        })
    lqcr.sort(key=lambda row: row["training_seed"])
    dino.sort(key=lambda row: row["inference_seed"])
    if len(lqcr) != 3 or len(dino) != 3:
        raise RuntimeError(f"expected 3 LQCR and 3 DINO predictions, got {len(lqcr)}, {len(dino)}")
    return lqcr, dino


def group_by_image(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["image_id"])].append(row)
    return grouped


def bootstrap_ground_truth(gt: dict, gt_by_image: dict[int, list[dict]],
                           draws: list[int]) -> dict:
    images_by_id = {int(row["id"]): row for row in gt["images"]}
    images, annotations = [], []
    annotation_id = 1
    for new_image_id, old_image_id in enumerate(draws, 1):
        image = dict(images_by_id[old_image_id])
        image["id"] = new_image_id
        images.append(image)
        for source in gt_by_image[old_image_id]:
            row = dict(source)
            row["id"] = annotation_id
            row["image_id"] = new_image_id
            annotation_id += 1
            annotations.append(row)
    return {
        "images": images,
        "annotations": annotations,
        "categories": gt["categories"],
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
    }


def bootstrap_predictions(pred_by_image: dict[int, list[dict]], draws: list[int]) -> list[dict]:
    predictions = []
    for new_image_id, old_image_id in enumerate(draws, 1):
        for source in pred_by_image.get(old_image_id, ()):
            row = dict(source)
            row["image_id"] = new_image_id
            predictions.append(row)
    return predictions


def map_score(gt_doc: dict, predictions: list[dict]) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_doc
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[0])


def prepare_image_evaluations(gt_doc: dict, predictions: list[dict]) -> dict:
    """Cache image-local COCO matching; AP accumulation remains bootstrap-specific."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_doc
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [100]
        evaluator.params.areaRng = [[0**2, 1e5**2]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.evaluate()
    evaluated_params = evaluator._paramsEval
    if len(evaluated_params.areaRng) != 1 or len(evaluated_params.maxDets) != 1:
        raise RuntimeError("unexpected COCO evaluation dimensions")
    return {
        "eval_images": evaluator.evalImgs,
        "image_ids": list(evaluated_params.imgIds),
        "image_index": {int(image_id): index for index, image_id in enumerate(evaluated_params.imgIds)},
        "category_ids": list(evaluated_params.catIds),
        "iou_thresholds": np.asarray(evaluated_params.iouThrs),
        "recall_thresholds": np.asarray(evaluated_params.recThrs),
        "max_detections": 100,
    }


def accumulated_map(cache: dict, draws: list[int]) -> float:
    """COCO mAP from cached per-image matches, preserving official stable sorting."""
    eval_images = cache["eval_images"]
    image_count = len(cache["image_ids"])
    image_index = cache["image_index"]
    recall_thresholds = cache["recall_thresholds"]
    max_detections = cache["max_detections"]
    precision_sum = 0.0
    precision_count = 0
    for category_index, _ in enumerate(cache["category_ids"]):
        offset = category_index * image_count
        selected = [eval_images[offset + image_index[image_id]] for image_id in draws]
        selected = [row for row in selected if row is not None]
        if not selected:
            continue
        detection_scores = np.concatenate([
            row["dtScores"][:max_detections] for row in selected
        ])
        order = np.argsort(-detection_scores, kind="mergesort")
        detection_matches = np.concatenate([
            row["dtMatches"][:, :max_detections] for row in selected
        ], axis=1)[:, order]
        detection_ignore = np.concatenate([
            row["dtIgnore"][:, :max_detections] for row in selected
        ], axis=1)[:, order]
        ground_truth_ignore = np.concatenate([row["gtIgnore"] for row in selected])
        positive_ground_truth = np.count_nonzero(ground_truth_ignore == 0)
        if positive_ground_truth == 0:
            continue
        true_positives = np.logical_and(detection_matches, np.logical_not(detection_ignore))
        false_positives = np.logical_and(np.logical_not(detection_matches),
                                         np.logical_not(detection_ignore))
        true_positive_sum = np.cumsum(true_positives, axis=1, dtype=float)
        false_positive_sum = np.cumsum(false_positives, axis=1, dtype=float)
        for true_positive, false_positive in zip(true_positive_sum, false_positive_sum):
            recall = true_positive / positive_ground_truth
            precision = true_positive / (true_positive + false_positive + np.spacing(1))
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            sampled = np.zeros_like(recall_thresholds, dtype=float)
            indices = np.searchsorted(recall, recall_thresholds, side="left")
            valid = indices < precision.size
            sampled[valid] = precision[indices[valid]]
            precision_sum += float(sampled.sum())
            precision_count += int(sampled.size)
    if precision_count == 0:
        raise RuntimeError("no valid category precision values")
    return precision_sum / precision_count


def bootstrap_iteration(task: tuple[int, int]) -> tuple[int, float]:
    """Evaluate one deterministic bootstrap index, independent of scheduling."""
    index, base_seed = task
    if _WORKER_IMAGE_IDS is None or _WORKER_EVAL_RUNS is None:
        raise RuntimeError("bootstrap worker was not initialized")
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, index]))
    draws = rng.choice(_WORKER_IMAGE_IDS, size=len(_WORKER_IMAGE_IDS), replace=True).tolist()
    scores = [accumulated_map(cache, draws) for cache in _WORKER_EVAL_RUNS]
    return index, float(np.mean(scores[:3]) - np.mean(scores[3:]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validate-cache", action="store_true")
    args = parser.parse_args()
    if args.iterations < 100:
        raise ValueError("at least 100 bootstrap iterations are required")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    lqcr, dino = resolve_inputs()
    gt = json.loads(ANN.read_text())
    image_ids = sorted(int(row["id"]) for row in gt["images"])
    if len(image_ids) != 1000 or len(set(image_ids)) != 1000:
        raise RuntimeError("expected 1,000 unique D2 test images")
    gt_by_image = group_by_image(gt["annotations"])
    all_runs = lqcr + dino
    prediction_groups = []
    for run in all_runs:
        path = ROOT / run["prediction_path"]
        run["prediction_sha256"] = sha256(path)
        prediction_groups.append(group_by_image(json.loads(path.read_text())))

    global _WORKER_IMAGE_IDS, _WORKER_EVAL_RUNS
    _WORKER_IMAGE_IDS = image_ids
    _WORKER_EVAL_RUNS = [
        prepare_image_evaluations(gt, [row for rows in groups.values() for row in rows])
        for groups in prediction_groups
    ]
    full_scores = [accumulated_map(cache, image_ids) for cache in _WORKER_EVAL_RUNS]
    point_lqcr = float(np.mean(full_scores[:3]))
    point_dino = float(np.mean(full_scores[3:]))
    cache_validation = None
    if args.validate_cache:
        validation_rng = np.random.default_rng(
            np.random.SeedSequence([args.bootstrap_seed, 0]))
        validation_draws = validation_rng.choice(
            image_ids, size=len(image_ids), replace=True).tolist()
        cached_scores = [accumulated_map(cache, validation_draws) for cache in _WORKER_EVAL_RUNS]
        exact_scores = []
        boot_gt = bootstrap_ground_truth(gt, gt_by_image, validation_draws)
        for groups in prediction_groups:
            exact_scores.append(map_score(
                boot_gt, bootstrap_predictions(groups, validation_draws)))
        maximum_error = float(np.max(np.abs(np.asarray(cached_scores) - np.asarray(exact_scores))))
        if maximum_error > 1e-12:
            raise RuntimeError(f"cached accumulation mismatch: {maximum_error}")
        cache_validation = {
            "bootstrap_index": 0,
            "cached_scores": cached_scores,
            "exact_cloned_dataset_scores": exact_scores,
            "maximum_absolute_error": maximum_error,
            "tolerance": 1e-12,
        }
    tasks = [(index, args.bootstrap_seed) for index in range(args.iterations)]
    indexed_differences = []
    fork_context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=fork_context) as executor:
        futures = [executor.submit(bootstrap_iteration, task) for task in tasks]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            indexed_differences.append(future.result())
            if completed % 25 == 0 or completed == args.iterations:
                print(f"completed {completed}/{args.iterations}", flush=True)
    indexed_differences.sort(key=lambda item: item[0])
    differences = [difference for _, difference in indexed_differences]

    low, high = np.percentile(differences, [2.5, 97.5])
    evidence = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "image-level paired nonparametric bootstrap of COCO bbox mAP",
        "dataset": "D2",
        "split": "test",
        "images": 1000,
        "annotation_path": str(ANN.relative_to(ROOT)),
        "annotation_sha256": sha256(ANN),
        "estimand": "mean mAP over three independent LQCR detector training runs minus mean mAP over three inference seeds of the fixed DINO checkpoint",
        "sampling_unit": "test image; sampled with replacement and cloned to preserve repeated-image multiplicity",
        "important_boundary": "The interval quantifies test-image sampling uncertainty conditional on the evaluated checkpoints. DINO inference seeds are not training-seed replications, and the interval does not estimate DINO training uncertainty.",
        "bootstrap": {
            "iterations": args.iterations,
            "seed": args.bootstrap_seed,
            "seed_derivation": "numpy SeedSequence([bootstrap_seed, bootstrap_index])",
            "worker_count": args.workers,
            "merge_order": "ascending bootstrap_index",
            "worker_invariance": "Each index owns an independent RNG stream; worker count and scheduling do not alter replicate values.",
            "optimization": "COCO matching is cached per image; each replicate reruns the official global stable score ordering and AP accumulation over the resampled image records.",
            "cache_equivalence_validation": cache_validation,
            "interval": "two-sided percentile 95%",
        },
        "inputs": {"lqcr": lqcr, "dino": dino},
        "point_estimate": {
            "lqcr_mean_mAP": point_lqcr,
            "dino_mean_mAP": point_dino,
            "difference": point_lqcr - point_dino,
            "descriptive_lqcr_run_differences_against_dino_inference_mean": [
                float(score - point_dino) for score in full_scores[:3]
            ],
        },
        "bootstrap_result": {
            "difference_mean": float(np.mean(differences)),
            "difference_sample_std": float(np.std(differences, ddof=1)),
            "ci95_lower": float(low),
            "ci95_upper": float(high),
            "fraction_difference_gt_zero": float(np.mean(np.asarray(differences) > 0)),
        },
        "bootstrap_replicates": [
            {"index": index, "difference": difference}
            for index, difference in indexed_differences
        ],
    }
    replicate_payload = json.dumps(differences, separators=(",", ":")).encode()
    evidence["bootstrap"]["replicate_vector_sha256"] = hashlib.sha256(replicate_payload).hexdigest()
    payload = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    output = ROOT / "tools/experiment_db/evidence_sources" / f"d2_lqcr_vs_dino_image_bootstrap_{digest[:12]}.json"
    output.write_bytes(payload)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
