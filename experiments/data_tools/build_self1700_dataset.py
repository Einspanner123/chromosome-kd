#!/usr/bin/env python3
"""Build the provenance-locked, group-split 1,700-image in-house dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSITE = ROOT / "data/Chromosome20240904_NoAug_NoResize_coco"
DEFAULT_PUBLIC = ROOT / "data/24_chromosomes_object/JEPG"
DEFAULT_OUTPUT = ROOT / "data/ChromosomeSelf1700_coco"
DEFAULT_EVIDENCE = ROOT / "tools/experiment_db/evidence_sources"
SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "valid", "test")
TARGET_IMAGES = {"train": 1190, "valid": 170, "test": 340}
SPLIT_SEED = 20260813
ZNCC_THRESHOLD = 0.99


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def normalized_stem(file_name: str) -> str:
    return re.sub(r"_(?:jpg|png)\.rf\.[0-9a-f]+$", "", Path(file_name).stem)


def decoded_pixel_sha(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    digest = hashlib.sha256()
    digest.update(f"{image.shape[1]}x{image.shape[0]}|BGR|".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def zncc_after_resize(source: Path, candidate: Path) -> float:
    left = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise RuntimeError(f"cannot decode comparison: {source}, {candidate}")
    right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left.ravel(), right.ravel()) / denom) if denom else 0.0


def load_source(root: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    categories = None
    for split in SOURCE_SPLITS:
        annotation_path = root / split / "_annotations.coco.json"
        document = json.loads(annotation_path.read_text())
        if categories is None:
            categories = document["categories"]
        elif categories != document["categories"]:
            raise RuntimeError("category definitions differ across source splits")
        by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in document["annotations"]:
            by_image[int(annotation["image_id"])].append(annotation)
        for image in document["images"]:
            image_id = int(image["id"])
            path = root / split / image["file_name"]
            records.append({
                "source_split": split,
                "source_image_id": image_id,
                "file_name": image["file_name"],
                "source_path": path,
                "image": image,
                "annotations": by_image[image_id],
                "group_id": normalized_stem(image["file_name"]),
            })
    return records, categories or []


def mark_provenance(records: list[dict], public_root: Path) -> None:
    public_by_stem = {
        path.stem: path for path in public_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    for record in records:
        public_path = public_by_stem.get(record["group_id"])
        # Quantize the audit score so manifests are byte-identical across
        # OpenCV builds while preserving a >0.16 safety margin at the threshold.
        score = round(zncc_after_resize(record["source_path"], public_path), 6) if public_path else None
        public_derived = score is not None and score >= ZNCC_THRESHOLD
        record["provenance"] = "public_d2_derived" if public_derived else "in_house"
        record["public_source_file"] = public_path.name if public_derived else None
        record["public_source_sha256"] = sha256_file(public_path) if public_derived else None
        record["public_match_zncc"] = score


def assign_group_splits(records: list[dict], category_ids: list[int]) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    if any(len(items) > 2 for items in groups.values()):
        raise RuntimeError("unexpected provenance group larger than two images")

    total_class = Counter()
    vectors = {}
    for group_id, items in groups.items():
        vector = Counter(
            int(annotation["category_id"])
            for item in items for annotation in item["annotations"]
        )
        vectors[group_id] = vector
        total_class.update(vector)

    ratios = {split: TARGET_IMAGES[split] / len(records) for split in OUTPUT_SPLITS}
    target_class = {
        split: {category: total_class[category] * ratios[split] for category in category_ids}
        for split in OUTPUT_SPLITS
    }
    assigned_images = Counter()
    assigned_class = {split: Counter() for split in OUTPUT_SPLITS}
    assignment: dict[str, str] = {}
    rng = random.Random(SPLIT_SEED)
    tie_break = {group_id: rng.random() for group_id in groups}

    # Place two-image groups first so the remaining singleton groups can satisfy
    # exact image capacities. Within each size, rare/class-heavy groups go first.
    def priority(group_id: str) -> tuple:
        vector = vectors[group_id]
        rarity = sum(value / max(total_class[category], 1) for category, value in vector.items())
        return (-len(groups[group_id]), -rarity, -sum(vector.values()), tie_break[group_id])

    for group_id in sorted(groups, key=priority):
        size = len(groups[group_id])
        feasible = [
            split for split in OUTPUT_SPLITS
            if assigned_images[split] + size <= TARGET_IMAGES[split]
        ]
        if not feasible:
            raise RuntimeError(f"no feasible split for group {group_id}")

        def benefit(split: str) -> tuple[float, float, str]:
            class_gain = sum(
                max(target_class[split][category] - assigned_class[split][category], 0.0)
                * count / max(target_class[split][category], 1.0)
                for category, count in vectors[group_id].items()
            )
            image_deficit = (TARGET_IMAGES[split] - assigned_images[split]) / TARGET_IMAGES[split]
            return (class_gain + size * image_deficit, image_deficit, split)

        split = max(feasible, key=benefit)
        assignment[group_id] = split
        assigned_images[split] += size
        assigned_class[split].update(vectors[group_id])

    if dict(assigned_images) != TARGET_IMAGES:
        raise RuntimeError(f"split capacity invariant failed: {dict(assigned_images)}")
    return assignment


def validate_annotations(document: dict) -> None:
    image_ids = {int(item["id"]) for item in document["images"]}
    category_ids = {int(item["id"]) for item in document["categories"]}
    for annotation in document["annotations"]:
        if int(annotation["image_id"]) not in image_ids:
            raise RuntimeError("orphan annotation")
        if int(annotation["category_id"]) not in category_ids:
            raise RuntimeError("unknown category")
        x, y, width, height = map(float, annotation["bbox"])
        if min(width, height) <= 0 or min(x, y) < 0:
            raise RuntimeError("invalid bbox")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite", type=Path, default=DEFAULT_COMPOSITE)
    parser.add_argument("--public-images", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records, categories = load_source(args.composite)
    mark_provenance(records, args.public_images)
    public_records = [row for row in records if row["provenance"] == "public_d2_derived"]
    in_house_records = [row for row in records if row["provenance"] == "in_house"]
    if (len(records), len(public_records), len(in_house_records)) != (2200, 500, 1700):
        raise RuntimeError("source/provenance count invariant failed")
    if len({row["public_source_file"] for row in public_records}) != 500:
        raise RuntimeError("public-derived mapping is not one-to-one")
    if min(row["public_match_zncc"] for row in public_records) < ZNCC_THRESHOLD:
        raise RuntimeError("public-derived match below threshold")
    if len({decoded_pixel_sha(row["source_path"]) for row in records}) != 2200:
        raise RuntimeError("duplicate decoded source images")

    assignment = assign_group_splits(in_house_records, [int(item["id"]) for item in categories])
    for record in records:
        record["output_split"] = assignment.get(record["group_id"])

    staging = args.output.with_name(args.output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {args.output}; use --force")
        shutil.rmtree(args.output)
    staging.mkdir(parents=True)

    split_evidence = {}
    for split in OUTPUT_SPLITS:
        split_dir = staging / split
        split_dir.mkdir()
        selected = sorted(
            (row for row in in_house_records if row["output_split"] == split),
            key=lambda row: (row["group_id"], row["file_name"]),
        )
        images, annotations = [], []
        annotation_id = 1
        for image_id, record in enumerate(selected, 1):
            shutil.copyfile(record["source_path"], split_dir / record["file_name"])
            image = dict(record["image"])
            image["id"] = image_id
            images.append(image)
            for source_annotation in record["annotations"]:
                annotation = dict(source_annotation)
                annotation["id"] = annotation_id
                annotation["image_id"] = image_id
                annotations.append(annotation)
                annotation_id += 1
        document = {"images": images, "annotations": annotations, "categories": categories}
        validate_annotations(document)
        annotation_path = split_dir / "_annotations.coco.json"
        annotation_path.write_bytes(canonical_json_bytes(document))
        split_evidence[split] = {
            "path": str((args.output / split / annotation_path.name).relative_to(ROOT)),
            "images": len(images),
            "annotations": len(annotations),
            "category_counts": dict(sorted(Counter(
                str(item["category_id"]) for item in annotations
            ).items())),
        }

    readme = {
        "dataset_id": "D1_INHOUSE1700_V1",
        "description": "Pure in-house chromosome detection dataset derived from the audited composite source.",
        "split_protocol": "Deterministic group-disjoint 70/10/20 split with class-aware greedy allocation.",
        "target_images": TARGET_IMAGES,
        "group_key": "filename stem after removing the Roboflow export suffix",
        "split_seed": SPLIT_SEED,
        "patient_id_status": "not available; conservative acquisition-stem proxy used",
        "public_exclusion_rule": "same normalized basename and grayscale resize ZNCC >= 0.99",
    }
    (staging / "README.provenance.json").write_bytes(canonical_json_bytes(readme))
    os.replace(staging, args.output)

    for split in OUTPUT_SPLITS:
        path = args.output / split / "_annotations.coco.json"
        split_evidence[split]["sha256"] = sha256_file(path)

    manifest_records = []
    for record in records:
        manifest_records.append({
            "source_split": record["source_split"],
            "output_split": record["output_split"],
            "source_image_id": record["source_image_id"],
            "file_name": record["file_name"],
            "group_id": record["group_id"],
            "file_sha256": sha256_file(record["source_path"]),
            "pixel_sha256": decoded_pixel_sha(record["source_path"]),
            "provenance": record["provenance"],
            "public_source_file": record["public_source_file"],
            "public_source_sha256": record["public_source_sha256"],
            "public_match_zncc": record["public_match_zncc"],
            "annotation_count": len(record["annotations"]),
        })
    manifest = {
        "schema_version": "2.0",
        "dataset_id": "D1_INHOUSE1700_V1",
        "source_composite": str(args.composite.relative_to(ROOT)),
        "public_reference": str(args.public_images.relative_to(ROOT)),
        "construction_protocol": readme,
        "counts": {"source": 2200, "excluded_public_derived": 500, "included_in_house": 1700},
        "splits": split_evidence,
        "records": sorted(manifest_records, key=lambda row: (row["source_split"], row["file_name"])),
    }
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.evidence_dir / f"dataset_self1700_v1_provenance_{digest[:12]}.json"
    manifest_path.write_bytes(payload)
    print(json.dumps({"dataset_root": str(args.output), "manifest": str(manifest_path),
                      "manifest_sha256": digest, "splits": TARGET_IMAGES}, indent=2))


if __name__ == "__main__":
    main()
