#!/usr/bin/env python3
"""Build a ratio-preserving, exact-duplicate-group-disjoint D2 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
SCHEMA = ROOT / "tools/experiment_db/schema.sql"
SOURCE_ROOT = ROOT / "data/24_chromosomes_object/coco"
OUTPUT_ROOT = ROOT / "data/TaichungLeakFree5000_coco"
DATASET_ID = "D2_TAICHUNG5000_V2"
EXPECTED = {
    "train": (3500, 160888, "218ae0ebb71ecfb186bdb0872101ac50179c22c69a586cad0366ef10a0d8d5f7"),
    "valid": (500, 22984, "bcf0f930dea7380a3d2d5e82b393b7576a5861d416c163ff2930985f9ee7ca10"),
    "test": (1000, 45980, "110fd2804f435b04a1eee969cb28666a0818b2886040a57cbae2dd05f2767495"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source() -> tuple[list[dict], list[dict]]:
    items, categories = [], None
    for split, (n_images, n_annotations, expected_sha) in EXPECTED.items():
        path = SOURCE_ROOT / split / "_annotations.coco.json"
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"{split}: source annotation SHA mismatch")
        document = json.loads(path.read_text(encoding="utf-8"))
        if (len(document["images"]), len(document["annotations"])) != (
            n_images,
            n_annotations,
        ):
            raise RuntimeError(f"{split}: source count mismatch")
        current_categories = [
            {"id": row["id"], "name": row["name"]}
            for row in document["categories"]
        ]
        if categories is None:
            categories = current_categories
        elif categories != current_categories:
            raise RuntimeError(f"{split}: category schema mismatch")
        annotations = defaultdict(list)
        for annotation in document["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
        for image in document["images"]:
            image_path = SOURCE_ROOT / split / image["file_name"]
            if not image_path.is_file():
                raise RuntimeError(f"missing source image: {image_path}")
            items.append(
                {
                    "source_split": split,
                    "source_image": image,
                    "source_path": image_path,
                    "annotations": annotations[image["id"]],
                    "file_sha256": sha256_file(image_path),
                    "assigned_split": split,
                    "move_reason": "publisher_split_unchanged",
                }
            )
    return items, categories


def feature(item: dict, category_ids: list[int]) -> np.ndarray:
    category_index = {value: index for index, value in enumerate(category_ids)}
    values = np.zeros(len(category_ids) + 3, dtype=np.int64)
    for annotation in item["annotations"]:
        values[category_index[annotation["category_id"]]] += 1
        area = float(
            annotation.get(
                "area", annotation["bbox"][2] * annotation["bbox"][3]
            )
        )
        size_index = 0 if area < 32**2 else 1 if area < 96**2 else 2
        values[len(category_ids) + size_index] += 1
    return values


def repair_assignment(items: list[dict], categories: list[dict]) -> dict:
    by_hash = defaultdict(list)
    for item in items:
        by_hash[item["file_sha256"]].append(item)
    cross = [
        group
        for group in by_hash.values()
        if len({item["source_split"] for item in group}) > 1
    ]
    if len(cross) != 2 or any(
        {item["source_split"] for item in group} != {"train", "valid"}
        for group in cross
    ):
        raise RuntimeError(
            f"unexpected source duplicate topology: {len(cross)} cross groups"
        )
    moved_from_valid = []
    for group in sorted(cross, key=lambda rows: rows[0]["file_sha256"]):
        for item in group:
            if item["source_split"] == "valid":
                item["assigned_split"] = "train"
                item["move_reason"] = "co_located_with_exact_train_duplicate"
                moved_from_valid.append(item)

    category_ids = [row["id"] for row in categories]
    target = sum(
        (feature(item, category_ids) for item in moved_from_valid),
        np.zeros(len(category_ids) + 3, dtype=np.int64),
    )
    candidates = [
        item
        for item in items
        if item["source_split"] == "train"
        and len(by_hash[item["file_sha256"]]) == 1
    ]
    candidates.sort(key=lambda item: item["source_image"]["file_name"])
    matrix = np.stack([feature(item, category_ids) for item in candidates])
    best = None
    for first in range(len(candidates) - 1):
        scores = np.abs(matrix[first] + matrix[first + 1 :] - target).sum(axis=1)
        second_offset = int(np.argmin(scores))
        score = int(scores[second_offset])
        second = first + 1 + second_offset
        key = (
            score,
            candidates[first]["source_image"]["file_name"],
            candidates[second]["source_image"]["file_name"],
        )
        if best is None or key < best[0]:
            best = (key, first, second)
    replacements = [candidates[best[1]], candidates[best[2]]]
    for item in replacements:
        item["assigned_split"] = "valid"
        item["move_reason"] = "ratio_preserving_histogram_matched_swap"

    counts = defaultdict(int)
    for item in items:
        counts[item["assigned_split"]] += 1
    if dict(counts) != {"train": 3500, "valid": 500, "test": 1000}:
        raise RuntimeError(f"repaired split count mismatch: {dict(counts)}")
    for group in by_hash.values():
        if len({item["assigned_split"] for item in group}) != 1:
            raise RuntimeError("exact duplicate group still crosses splits")
    return {
        "cross_split_groups_repaired": len(cross),
        "validation_duplicates_moved_to_train": [
            item["source_image"]["file_name"] for item in moved_from_valid
        ],
        "histogram_matched_train_images_moved_to_validation": [
            item["source_image"]["file_name"] for item in replacements
        ],
        "replacement_feature_l1": best[0][0],
    }


def materialize(
    items: list[dict], categories: list[dict], output_root: Path
) -> dict:
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing dataset: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=output_root.name + ".", dir=output_root.parent)
    )
    split_evidence, lineage = {}, []
    try:
        for split in ("train", "valid", "test"):
            split_dir = temporary / split
            split_dir.mkdir()
            assigned = [
                item for item in items if item["assigned_split"] == split
            ]
            assigned.sort(
                key=lambda item: (
                    item["source_split"],
                    item["source_image"]["file_name"],
                    item["source_image"]["id"],
                )
            )
            images, annotations = [], []
            next_annotation_id = 1
            for new_image_id, item in enumerate(assigned, start=1):
                source_image = item["source_image"]
                output_name = (
                    f"{item['source_split']}__{source_image['id']:06d}__"
                    f"{source_image['file_name']}"
                )
                destination = split_dir / output_name
                os.link(item["source_path"], destination)
                image = dict(source_image)
                image["id"] = new_image_id
                image["file_name"] = output_name
                images.append(image)
                for source_annotation in item["annotations"]:
                    annotation = dict(source_annotation)
                    annotation["id"] = next_annotation_id
                    annotation["image_id"] = new_image_id
                    annotations.append(annotation)
                    next_annotation_id += 1
                lineage.append(
                    {
                        "source_split": item["source_split"],
                        "source_image_id": source_image["id"],
                        "source_file_name": source_image["file_name"],
                        "assigned_split": split,
                        "output_file_name": output_name,
                        "file_sha256": item["file_sha256"],
                        "move_reason": item["move_reason"],
                    }
                )
            document = {
                "info": {"description": "D2 duplicate-group-disjoint V2"},
                "licenses": [],
                "categories": categories,
                "images": images,
                "annotations": annotations,
            }
            annotation_path = split_dir / "_annotations.coco.json"
            annotation_path.write_text(
                json.dumps(document, ensure_ascii=False, separators=(",", ":"))
                + "\n",
                encoding="utf-8",
            )
            split_evidence[split] = {
                "path": (
                    Path("data") / output_root.name / split / annotation_path.name
                ).as_posix(),
                "sha256": sha256_file(annotation_path),
                "images": len(images),
                "annotations": len(annotations),
            }
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"splits": split_evidence, "records": lineage}


def register(manifest_path: Path, manifest: dict, db: Path) -> str:
    manifest_sha = sha256_file(manifest_path)
    artifact_id = f"dataset-d2-v2-{manifest_sha[:12]}"
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA.read_text(encoding="utf-8"))
    try:
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO evidence_artifact "
                "(artifact_id,server,path,sha256,kind,status,notes) "
                "VALUES (?, 'repository', ?, ?, 'dataset_release_manifest', "
                "'verified', ?)",
                (
                    artifact_id,
                    manifest_path.relative_to(ROOT).as_posix(),
                    manifest_sha,
                    "D2 V2 ratio-preserving exact-duplicate split repair.",
                ),
            )
            artifact = connection.execute(
                "SELECT path,sha256,kind,status FROM evidence_artifact "
                "WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            expected_artifact = (
                manifest_path.relative_to(ROOT).as_posix(),
                manifest_sha,
                "dataset_release_manifest",
                "verified",
            )
            if tuple(artifact or ()) != expected_artifact:
                raise RuntimeError("conflicting D2 V2 evidence artifact")
            connection.execute(
                "INSERT OR IGNORE INTO dataset_release "
                "(dataset_id,display_name,root_path,version,image_count,"
                "category_count,construction_protocol_json,manifest_artifact_id,"
                "status,notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    DATASET_ID,
                    manifest["display_name"],
                    manifest["root_path"],
                    manifest_sha[:12],
                    5000,
                    24,
                    json.dumps(manifest["construction_protocol"], sort_keys=True),
                    artifact_id,
                    "verified",
                    "70/10/20 exact-duplicate-group-disjoint split; patient IDs unavailable.",
                ),
            )
            release = connection.execute(
                "SELECT version,manifest_artifact_id,status FROM dataset_release "
                "WHERE dataset_id=?",
                (DATASET_ID,),
            ).fetchone()
            if tuple(release or ()) != (manifest_sha[:12], artifact_id, "verified"):
                raise RuntimeError("conflicting D2 V2 dataset release")
            for split, evidence in manifest["splits"].items():
                normalized = "val" if split == "valid" else split
                values = (
                    DATASET_ID,
                    normalized,
                    evidence["images"],
                    evidence["annotations"],
                    evidence["path"],
                    evidence["sha256"],
                )
                old = connection.execute(
                    "SELECT dataset_id,split,image_count,annotation_count,"
                    "annotation_path,annotation_sha256 FROM dataset_split "
                    "WHERE dataset_id=? AND split=?",
                    (DATASET_ID, normalized),
                ).fetchone()
                if old is not None and tuple(old) != values:
                    raise RuntimeError(f"conflicting registered split: {normalized}")
                if old is None:
                    connection.execute(
                        "INSERT INTO dataset_split VALUES (?,?,?,?,?,?)", values
                    )
    finally:
        connection.close()
    return artifact_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "tools/experiment_db/evidence_sources/dataset_d2_v2_release.json",
    )
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    items, categories = load_source()
    repair = repair_assignment(items, categories)
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.manifest.is_file():
            raise RuntimeError("dataset exists without its release manifest")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != DATASET_ID:
            raise RuntimeError("existing manifest has the wrong dataset_id")
        if manifest.get("construction_protocol", {}).get(
            "histogram_matched_train_images_moved_to_validation"
        ) != repair["histogram_matched_train_images_moved_to_validation"]:
            raise RuntimeError("existing manifest disagrees with deterministic repair")
        for split, evidence in manifest["splits"].items():
            path = ROOT / evidence["path"]
            if sha256_file(path) != evidence["sha256"]:
                raise RuntimeError(f"existing {split} annotation SHA mismatch")
        for record in manifest["records"]:
            path = output_root / record["assigned_split"] / record["output_file_name"]
            if sha256_file(path) != record["file_sha256"]:
                raise RuntimeError(f"existing image SHA mismatch: {path}")
    else:
        materialized = materialize(items, categories, output_root)
        manifest = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "display_name": "Taichung chromosome cohort, leakage-repaired V2",
            "root_path": (Path("data") / args.output_root.name).as_posix(),
            "construction_protocol": {
                "source_release": "publisher-provided D2 3500/500/1000 split",
                "ratio": "70/10/20 (3500/500/1000)",
                "assignment": "minimal exact-duplicate group repair with deterministic histogram-matched swaps",
                "patient_level_guard": "unavailable: public files expose no patient/specimen identifier",
                **repair,
            },
            "categories": categories,
            **materialized,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.manifest)
    artifact_id = None
    if not args.no_register:
        artifact_id = register(args.manifest.resolve(), manifest, args.db)
    print(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "manifest": args.manifest.resolve().relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256_file(args.manifest),
                "artifact_id": artifact_id,
                "repair": repair,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
