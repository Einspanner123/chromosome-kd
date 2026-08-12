#!/usr/bin/env python3
"""Compute manuscript dataset properties from held-out test annotations."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
ANNOTATIONS = {
    "D1": ROOT / "data/Chromosome20240904_NoAug_NoResize_coco/test/_annotations.coco.json",
    "D2": ROOT / "data/24_chromosomes_object/coco/test/_annotations.coco.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intersection_over_union(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def profile(path: Path) -> dict:
    payload = json.loads(path.read_text())
    images = {row["id"]: row for row in payload["images"]}
    by_image = {image_id: [] for image_id in images}
    for annotation in payload["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        by_image[annotation["image_id"]].append(annotation["bbox"])
    counts, relative_areas = [], []
    small_instances = 0
    images_with_small = 0
    overlap_images = 0
    total_instances = 0
    for image_id, boxes in by_image.items():
        image = images[image_id]
        counts.append(len(boxes))
        total_instances += len(boxes)
        has_small = False
        for _, _, width, height in boxes:
            relative_areas.append(width * height / (image["width"] * image["height"]))
            if width * height < 32 ** 2:
                small_instances += 1
                has_small = True
        images_with_small += int(has_small)
        overlap_images += int(any(
            intersection_over_union(boxes[i], boxes[j]) > 0.1
            for i in range(len(boxes)) for j in range(i + 1, len(boxes))))
    return {
        "images": len(images),
        "instances": total_instances,
        "median_instances": statistics.median(counts),
        "median_relative_area": statistics.median(relative_areas),
        "small_instance_fraction": small_instances / total_instances,
        "images_with_small_fraction": images_with_small / len(images),
        "overlap_image_fraction_iou_gt_0.1": overlap_images / len(images),
    }


def main() -> None:
    results = {dataset: profile(path) for dataset, path in ANNOTATIONS.items()}
    evidence = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split": "test",
        "definitions": {
            "small": "absolute bounding-box area < 32^2 pixels",
            "overlap": "image has at least one ground-truth pair with IoU > 0.1",
        },
        "annotations": {dataset: {"path": str(path.relative_to(ROOT)),
                                  "sha256": sha256(path)}
                        for dataset, path in ANNOTATIONS.items()},
        "results": results,
    }
    raw = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    relative = Path("tools/experiment_db/evidence_sources") / f"paper_test_dataset_properties_{digest[:12]}.json"
    (ROOT / relative).write_bytes(raw)
    artifact_id = f"paper-test-dataset-properties-{digest[:12]}"
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("""INSERT OR REPLACE INTO evidence_artifact
        (artifact_id,server,path,sha256,kind,generated_at,status,notes)
        VALUES (?,'ross',?,?,?,?,'verified',?)""",
        (artifact_id, str(relative), digest, "test_dataset_properties",
         evidence["generated_at"], "Deterministic descriptive statistics from held-out test annotations."))
    for dataset, metrics in results.items():
        for metric, value in metrics.items():
            conn.execute("""INSERT OR REPLACE INTO controlled_result
                (result_id,family,variant,dataset,split,seed,metric,value,unit,
                 baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
                 paper_eligible,notes) VALUES (?,?,?,?,'test','deterministic',?,?,?,
                 NULL,NULL,?,?,'controlled',1,?)""",
                (f"paper-test-property-{dataset.lower()}-{metric}",
                 "paper_test_dataset_properties", "held_out_test", dataset,
                 metric, float(value), "fraction" if "fraction" in metric else "absolute",
                 json.dumps(evidence["annotations"][dataset], sort_keys=True), artifact_id,
                 "Computed directly from complete held-out test annotations."))
    conn.commit()
    conn.close()
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(relative)


if __name__ == "__main__":
    main()
