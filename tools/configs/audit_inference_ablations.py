#!/usr/bin/env python3
"""Audit active inference protocols and any attached immutable evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
ABLATIONS = ROOT / "experiments/configs/ablations"
DB = ROOT / "tools/experiment_db/experiments.db"
REQUIRED_PROTOCOL = {
    "schema_version", "protocol_id", "protocol_type", "dataset_id", "split",
    "inference_seed", "selection", "base", "intervention",
}
REQUIRED_BASE = {"matrix", "method", "config_id"}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_exists(source: dict) -> bool:
    required = {"git_ref", "path", "blob_sha1"}
    if required - set(source or {}):
        return False
    spec = f'{source["git_ref"]}:{source["path"]}'
    resolved = subprocess.run(
        ["git", "rev-parse", spec], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    return resolved.returncode == 0 and resolved.stdout.strip() == source["blob_sha1"]


def variant_ids(data: dict) -> set[str]:
    intervention = data["intervention"]
    factors = intervention["factors"]
    names = list(factors)
    combinations = itertools.product(*(factors[name] for name in names))
    evidence = data.get("evidence")
    group = evidence.get("aggregate_group") if evidence else None
    output = set()
    for values in combinations:
        item = dict(zip(names, values))
        if group == "solver":
            variant = f"{item['solver_type']}_{item['sampling_timesteps']}"
        elif group == "topk_renewal":
            state = "on" if item["box_renewal"] else "off"
            variant = f"k{item['topk_k']}_renewal_{state}"
        elif group == "lqcr_beta":
            value = str(item["quality_score_beta"]).replace(".", "p")
            variant = f"beta_{value}"
        elif group is not None:
            raise ValueError(f"unsupported aggregate group: {group}")
        else:
            variant = "__".join(f"{name}={item[name]}" for name in names)
        if variant in output:
            raise ValueError(f"duplicate variant id: {variant}")
        output.add(variant)
    return output


def audit(path: Path) -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load(path.read_text())
    missing = REQUIRED_PROTOCOL - set(data or {})
    if missing:
        return [f"missing fields {sorted(missing)}"]
    if data["schema_version"] != 1:
        errors.append("unsupported schema_version")
    if data["protocol_type"] not in {
        "inference_ablation", "inference_replication", "validation_selection"
    }:
        errors.append("unsupported protocol_type")
    selection = data["selection"]
    if selection.get("split") != "val" or selection.get("test_tuned") is not False:
        errors.append("selection must be locked on val before test")

    base = data["base"]
    missing = REQUIRED_BASE - set(base)
    if missing:
        errors.append(f"base missing fields {sorted(missing)}")
    for field in ("matrix",):
        target = ROOT / base.get(field, "")
        if not target.is_file():
            errors.append(f"missing base {field}: {base.get(field)}")

    # Planned protocols intentionally have no evidence block yet. Their matrix,
    # selection boundary, intervention grid, and checkpoint registry structure
    # are still audited, but they are not treated as completed experiments.
    if "evidence" not in data:
        if "checkpoint_source" not in base and "checkpoints" not in base:
            errors.append("planned protocol lacks checkpoint source")
        return errors

    if data["dataset_id"] != "D2" or data["split"] != "test":
        errors.append("attached historical evidence must use D2/test")
    required_evidence_base = {
        "training_seed", "legacy_config", "legacy_source", "checkpoint",
        "checkpoint_sha256",
    }
    missing = required_evidence_base - set(base)
    if missing:
        errors.append(f"evidence base missing fields {sorted(missing)}")
        return errors
    legacy_config = ROOT / base["legacy_config"]
    if not legacy_config.is_file() and not git_blob_exists(base["legacy_source"]):
        errors.append("legacy config is absent and archived Git blob does not resolve")
    checkpoint = ROOT / base["checkpoint"]
    if not checkpoint.is_file():
        errors.append(f"missing base checkpoint: {base['checkpoint']}")
    elif file_sha256(checkpoint) != base["checkpoint_sha256"]:
        errors.append("checkpoint SHA-256 mismatch")

    evidence = data["evidence"]
    if evidence.get("replay_status") == "complete":
        artifact_fields = [
            name for name in (
                "raw_prediction_archive_artifact",
                "raw_prediction_manifest_artifact",
            ) if name in evidence
        ]
        if evidence["aggregate_group"] in ("solver", "topk_renewal") \
                and len(artifact_fields) != 2:
            errors.append("complete replay status requires raw archive artifacts")
        connection = sqlite3.connect(DB)
        for field in artifact_fields:
            artifact_id = evidence[field]
            row = connection.execute(
                "SELECT path,sha256,status FROM evidence_artifact WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                errors.append(f"unknown evidence artifact: {artifact_id}")
                continue
            artifact_path = ROOT / row[0]
            if row[2] != "verified" or not artifact_path.is_file():
                errors.append(f"unverified or missing artifact: {artifact_id}")
            elif file_sha256(artifact_path) != row[1]:
                errors.append(f"artifact SHA-256 mismatch: {artifact_id}")
        connection.close()
    producer = ROOT / evidence["producer"]
    if not producer.is_file():
        errors.append(f"missing producer: {evidence['producer']}")
    elif file_sha256(producer) != evidence["producer_sha256"]:
        errors.append("producer SHA-256 mismatch")
    aggregate_path = ROOT / evidence["aggregate"]
    if not aggregate_path.is_file():
        errors.append(f"missing aggregate: {evidence['aggregate']}")
        return errors
    if file_sha256(aggregate_path) != evidence["aggregate_sha256"]:
        errors.append("aggregate SHA-256 mismatch")
        return errors

    aggregate = json.loads(aggregate_path.read_text())
    if aggregate["dataset"].get("dataset", "D2") not in ("D2", None):
        errors.append("aggregate dataset mismatch")
    if aggregate.get("seed") != data["inference_seed"]:
        errors.append("aggregate inference seed mismatch")
    results = [r for r in aggregate["results"]
               if r["group"] == evidence["aggregate_group"]]
    expected = variant_ids(data)
    actual = {r[evidence["record_key"]] for r in results}
    if expected != actual:
        errors.append(
            f"variant coverage mismatch missing={sorted(expected-actual)} "
            f"extra={sorted(actual-expected)}")
    if len(results) != evidence["expected_records"]:
        errors.append("unexpected aggregate record count")

    required_metrics = set(evidence["required_metrics"])
    for row in results:
        model_id = row[evidence["record_key"]]
        protocol = row["protocol"]
        if protocol["config_path"] != base["legacy_config"]:
            errors.append(f"{model_id}: legacy config mismatch")
        if protocol["checkpoint_path"] != base["checkpoint"]:
            errors.append(f"{model_id}: checkpoint path mismatch")
        if protocol["checkpoint_sha256"] != base["checkpoint_sha256"]:
            errors.append(f"{model_id}: checkpoint SHA-256 mismatch")
        if protocol["split"] != data["split"] or protocol["seed"] != data["inference_seed"]:
            errors.append(f"{model_id}: split/seed mismatch")
        absent = required_metrics - set(row["metrics"])
        if absent:
            errors.append(f"{model_id}: missing metrics {sorted(absent)}")
        source = ROOT / row["source_summary"]
        if not source.is_file():
            errors.append(f"{model_id}: missing source summary")
        elif file_sha256(source) != row["source_summary_sha256"]:
            errors.append(f"{model_id}: source summary SHA-256 mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted(ABLATIONS.glob("*.yaml"))
    failures = []
    total = 0
    for path in paths:
        if not path.is_absolute():
            path = ROOT / path
        errors = audit(path)
        try:
            count = len(variant_ids(yaml.safe_load(path.read_text())))
        except Exception:
            count = 0
        total += count
        if errors:
            failures.extend(f"{path.relative_to(ROOT)}: {e}" for e in errors)
    if failures:
        print("INFERENCE ABLATION AUDIT: FAIL")
        print("\n".join(f"- {error}" for error in failures))
        return 1
    print("INFERENCE ABLATION AUDIT: PASS")
    print(f"manifests={len(paths)} variants={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
