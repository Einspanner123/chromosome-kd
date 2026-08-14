#!/usr/bin/env python3
"""Validate experiment evidence without optional third-party dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

METRICS = ("mAP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPLICATION_UNITS = {
    "independent_training_seed", "fixed_checkpoint_inference_seed",
    "paired_final_stage_intervention", "hardware_efficiency_repeat",
    "deterministic_evaluation",
}
FINAL_STATES = {"verified", "paper_eligible"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(record: dict, root: Path, verify_files: bool) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version", "record_id", "record_type", "dataset_id", "split",
        "annotation", "num_images", "training_seed", "replication_unit", "config",
        "checkpoint", "source_log", "code", "evaluation_protocol",
        "selection_source", "status", "metrics", "train_run_id",
        "protocol_sha256",
    }
    missing = sorted(required - record.keys())
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
        return errors
    if record["schema_version"] != "1.0":
        errors.append("schema_version must be 1.0")
    if record["replication_unit"] not in REPLICATION_UNITS:
        errors.append("unknown replication_unit")
    if not record.get("train_run_id"):
        errors.append("train_run_id must be non-empty")
    if not SHA256_RE.fullmatch(record.get("protocol_sha256") or ""):
        errors.append("invalid protocol_sha256")
    if not record.get("selection_source"):
        errors.append("selection_source must be non-empty")
    if record["status"] in FINAL_STATES and record["split"] != "test":
        errors.append("verified/paper-eligible accuracy evidence must use split=test")
    if record["status"] in FINAL_STATES and record.get("test_tuned", False):
        errors.append("test-tuned evidence cannot be verified/paper-eligible")
    protocol = record.get("evaluation_protocol", {})
    for field in ("metric_definition", "max_dets", "eval_code_version",
                  "candidate_count", "solver", "steps", "nfe"):
        if field not in protocol:
            errors.append(f"evaluation_protocol missing: {field}")
    if record["replication_unit"] == "paired_final_stage_intervention":
        if not record.get("parent_train_run_id"):
            errors.append("paired intervention requires parent_train_run_id")
        if not SHA256_RE.fullmatch(record.get("parent_checkpoint_sha256") or ""):
            errors.append("paired intervention requires parent_checkpoint_sha256")
    metrics = record.get("metrics", {})
    for metric in METRICS:
        value = metrics.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"missing/non-numeric metric: {metric}")
        elif not 0 <= value <= 1:
            errors.append(f"metric outside [0,1]: {metric}={value}")
    for name in ("annotation", "config", "checkpoint", "source_log"):
        artifact = record.get(name, {})
        digest = artifact.get("sha256", "") if isinstance(artifact, dict) else ""
        if not SHA256_RE.fullmatch(digest):
            errors.append(f"invalid SHA-256 for {name}")
            continue
        artifact_path = Path(artifact.get("path", ""))
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            errors.append(f"{name} path must be project-relative")
            continue
        if verify_files:
            path = root / artifact_path
            if not path.is_file():
                errors.append(f"missing artifact: {name}={path}")
            elif file_sha256(path) != digest:
                errors.append(f"SHA-256 mismatch: {name}={path}")
    code = record.get("code", {})
    if code.get("dirty") and not SHA256_RE.fullmatch(code.get("diff_sha256") or ""):
        errors.append("dirty code requires diff_sha256")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    failed = 0
    for path in args.records:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for index, record in enumerate(records):
            errors = validate(record, args.root, args.verify_files)
            label = f"{path}[{index}]"
            if errors:
                failed += 1
                print(f"FAIL {label}")
                for error in errors:
                    print(f"  - {error}")
            else:
                print(f"PASS {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
