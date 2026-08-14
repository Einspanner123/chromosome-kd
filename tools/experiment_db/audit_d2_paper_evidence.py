#!/usr/bin/env python3
"""Audit D2 analysis protocols and paper-cell provenance against experiments.db.

This script is read-only.  Exit code 0 means every declared result and artifact
exists and its eligibility/replication semantics agree with the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self, root: Path, db: Path) -> None:
        self.root = root
        self.db = sqlite3.connect(db)
        self.db.row_factory = sqlite3.Row
        self.errors: list[str] = []
        self.result_references = 0
        self.checked_result_ids: set[str] = set()
        self.checked_artifacts: set[str] = set()

    def error(self, message: str) -> None:
        self.errors.append(message)

    def check_artifact(self, label: str, spec: dict[str, Any]) -> None:
        artifact_id = spec.get("artifact_id")
        row = self.db.execute(
            "SELECT artifact_id, path, sha256, status FROM evidence_artifact WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            self.error(f"{label}: unknown artifact_id {artifact_id!r}")
            return
        self.checked_artifacts.add(str(artifact_id))
        for field in ("path", "sha256"):
            if spec.get(field) != row[field]:
                self.error(
                    f"{label}: artifact {artifact_id} {field} mismatch: "
                    f"manifest={spec.get(field)!r}, db={row[field]!r}"
                )
        if row["status"] != "verified":
            self.error(f"{label}: artifact {artifact_id} status is {row['status']!r}")
        path = self.root / str(row["path"])
        if not path.is_file():
            self.error(f"{label}: artifact file missing: {path}")
        elif sha256(path) != row["sha256"]:
            self.error(f"{label}: artifact file hash differs from DB: {path}")

    def result(self, result_id: str) -> sqlite3.Row | None:
        row = self.db.execute(
            "SELECT * FROM controlled_result WHERE result_id = ?", (result_id,)
        ).fetchone()
        if row is None:
            self.error(f"unknown controlled_result_id {result_id!r}")
            return None
        self.result_references += 1
        self.checked_result_ids.add(result_id)
        return row

    def check_role(
        self,
        where: str,
        row: sqlite3.Row,
        role: str,
        measurement_kind: str = "test_accuracy",
    ) -> None:
        expected_split = "benchmark" if measurement_kind == "deployment_latency" else "test"
        if row["dataset"] != "D2" or row["split"] != expected_split:
            self.error(
                f"{where}: expected D2/{expected_split}, got {row['dataset']}/{row['split']}"
            )
        expected = {
            "paper_eligible": (1, "controlled"),
            "diagnostic": (0, "diagnostic"),
        }.get(role)
        if expected is None:
            self.error(f"{where}: invalid evidence_role {role!r}")
            return
        if (row["paper_eligible"], row["evidence_level"]) != expected:
            self.error(
                f"{where}: role={role}, DB eligibility="
                f"{row['paper_eligible']}/{row['evidence_level']}"
            )
        if measurement_kind == "deployment_latency":
            if row["metric"] != "latency" or row["unit"] != "ms":
                self.error(
                    f"{where}: latency cell must use metric=latency, unit=ms; "
                    f"got {row['metric']!r}/{row['unit']!r}"
                )
        elif measurement_kind == "test_accuracy":
            if row["metric"] == "latency" or row["unit"] == "ms":
                self.error(f"{where}: latency result used as an accuracy cell")

    def check_semantics(self, where: str, row: sqlite3.Row, semantics: str | None) -> None:
        if semantics == "ours_sota":
            seed = str(row["seed"])
            if row["family"] != "paired_lqcr_d2_train3" or not seed.startswith(("mean(", "std(")):
                self.error(
                    f"{where}: ours SOTA must be a mean/SD over independent training runs; "
                    f"got family={row['family']!r}, seed={row['seed']!r}"
                )
        elif semantics == "baseline_sota":
            if row["family"] != "paper_three_inference_seed_test" or not str(row["seed"]).startswith("inference_mean("):
                self.error(
                    f"{where}: baseline SOTA must be a fixed-checkpoint inference-seed mean; "
                    f"got family={row['family']!r}, seed={row['seed']!r}"
                )
        elif semantics == "fixed_checkpoint_inference3":
            if row["family"] != "paper_three_inference_seed_test" or not str(row["seed"]).startswith("inference_mean("):
                self.error(
                    f"{where}: expected fixed-checkpoint three-inference-seed mean; "
                    f"got family={row['family']!r}, seed={row['seed']!r}"
                )
        elif semantics == "same_hardware_latency":
            if row["split"] != "benchmark" or row["metric"] != "latency" or row["unit"] != "ms":
                self.error(f"{where}: invalid same-hardware latency evidence")
        elif semantics == "conditional_diagnostic":
            if row["paper_eligible"] != 0 or row["evidence_level"] != "diagnostic":
                self.error(f"{where}: conditional diagnostic escaped the diagnostic gate")

    def audit_index(self, path: Path) -> None:
        doc = load_json(path)
        artifacts = doc.get("artifacts", {})
        if not isinstance(artifacts, dict):
            self.error(f"{path}: artifacts must be an object")
            return
        for name, spec in artifacts.items():
            self.check_artifact(f"{path}:artifacts.{name}", spec)

        keys: set[str] = set()
        for entry in doc.get("entries", []):
            key = entry.get("paper_key")
            if key in keys:
                self.error(f"{path}: duplicate paper_key {key!r}")
            keys.add(key)
            role = entry.get("evidence_role")
            artifact_ref = entry.get("artifact_ref")
            if artifact_ref not in artifacts:
                self.error(f"{key}: unknown artifact_ref {artifact_ref!r}")
                continue
            expected_artifact = artifacts[artifact_ref]["artifact_id"]
            semantics = entry.get("replication_semantics")
            measurement_kind = entry.get("measurement_kind", "test_accuracy")
            cells = entry.get("cells")
            if not isinstance(cells, dict) or not cells:
                self.error(f"{key}: cells must be a non-empty object")
                continue
            for cell, result_id in cells.items():
                where = f"{key}/{cell}"
                row = self.result(result_id)
                if row is None:
                    continue
                if row["artifact_id"] != expected_artifact:
                    self.error(
                        f"{where}: artifact mismatch: result={row['artifact_id']!r}, "
                        f"entry={expected_artifact!r}"
                    )
                self.check_role(where, row, role, measurement_kind)
                self.check_semantics(where, row, semantics)
            derived = entry.get("derived_cells", {})
            for cell, spec in derived.items():
                where = f"{key}/{cell}"
                operation = spec.get("operation")
                result_ids = spec.get("result_ids", [])
                if operation not in ("subtract", "reciprocal_ms_to_fps"):
                    self.error(f"{where}: unsupported derived operation {operation!r}")
                if operation == "subtract" and len(result_ids) != 2:
                    self.error(f"{where}: subtract requires exactly two result_ids")
                if operation == "reciprocal_ms_to_fps" and len(result_ids) != 1:
                    self.error(f"{where}: reciprocal requires exactly one result_id")
                for result_id in result_ids:
                    row = self.result(result_id)
                    if row is None:
                        continue
                    if row["artifact_id"] != expected_artifact:
                        self.error(
                            f"{where}: derived input artifact mismatch: "
                            f"result={row['artifact_id']!r}, entry={expected_artifact!r}"
                        )
                    self.check_role(where, row, role, measurement_kind)
                    self.check_semantics(where, row, semantics)

    def audit_protocol(self, path: Path) -> None:
        doc = load_json(path)
        role = doc.get("analysis_class")
        eligible = doc.get("paper_eligible")
        if role == "paper_eligible" and eligible is not True:
            self.error(f"{path}: paper_eligible protocol must set paper_eligible=true")
        if role == "diagnostic" and eligible is not False:
            self.error(f"{path}: diagnostic protocol must set paper_eligible=false")
        artifact = doc.get("required_artifact")
        if isinstance(artifact, dict):
            self.check_artifact(str(path), artifact)
            expected_artifact = artifact.get("artifact_id")
        elif isinstance(doc.get("required_artifacts"), list):
            for index, item in enumerate(doc["required_artifacts"]):
                self.check_artifact(f"{path}:required_artifacts[{index}]", item)
            expected_artifact = None
        else:
            self.error(f"{path}: required_artifact(s) is missing")
            return

        for result_id in doc.get("required_result_ids", []):
            row = self.result(result_id)
            if row is not None:
                if row["artifact_id"] != expected_artifact:
                    self.error(f"{path}: {result_id} does not belong to {expected_artifact}")
                if role in ("paper_eligible", "diagnostic"):
                    self.check_role(f"{path}:{result_id}", row, role)

        selector = doc.get("result_selector")
        if isinstance(selector, dict):
            clauses, values = [], []
            for field in ("family", "dataset", "split", "evidence_level", "paper_eligible"):
                if field in selector:
                    clauses.append(f"{field} = ?")
                    values.append(selector[field])
            query = "SELECT COUNT(*) FROM controlled_result WHERE " + " AND ".join(clauses)
            count = self.db.execute(query, values).fetchone()[0]
            if count == 0:
                self.error(f"{path}: result_selector matched no rows")
            artifact_count = self.db.execute(
                query + " AND artifact_id = ?", (*values, expected_artifact)
            ).fetchone()[0]
            if artifact_count != count:
                self.error(
                    f"{path}: selector spans artifacts ({artifact_count}/{count} on expected artifact)"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--db", type=Path, default=Path("tools/experiment_db/experiments.db"))
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("tools/experiment_db/manifests/d2_paper_evidence_index.json"),
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=Path("tools/experiment_db/protocols"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    audit = Audit(root, resolve(args.db))
    audit.audit_index(resolve(args.index))
    for protocol in sorted(resolve(args.protocol_dir).glob("*.json")):
        audit.audit_protocol(protocol)
    report = {
        "status": "PASS" if not audit.errors else "FAIL",
        "checked_results": len(audit.checked_result_ids),
        "result_references": audit.result_references,
        "checked_artifacts": len(audit.checked_artifacts),
        "errors": audit.errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not audit.errors else 1


if __name__ == "__main__":
    sys.exit(main())
