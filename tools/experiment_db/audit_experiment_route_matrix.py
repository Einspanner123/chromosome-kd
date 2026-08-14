#!/usr/bin/env python3
"""Audit the authoritative v2 paper experiment route matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "experiments/manifests/paper_experiment_route_matrix.yaml"
CSV_PATH = ROOT / "tools/experiment_db/exports/paper_experiment_route_matrix.csv"
DOC = ROOT / "docs/experiments/PAPER_EXPERIMENT_ROUTE_MATRIX.md"
DB = ROOT / "tools/experiment_db/experiments.db"
FORBIDDEN_ACTIVE_PREFIXES = (
    "projects/DiffusionDet", "experiments/configs/self1700/",
    "experiments/configs/ldmdet/", "experiments/configs/baselines/",
    "experiments/configs/d2_strict_ablations/", "experiments/configs/d2_deployment/",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    payload = yaml.safe_load(MATRIX.read_text())
    rows = payload["experiments"]
    ids = [r["ledger_id"] for r in rows]
    idset = set(ids)
    if payload.get("authoritative") is not True:
        errors.append("matrix is not marked authoritative")
    if len(rows) != 46 or len(idset) != 46:
        errors.append(f"expected 46 unique route groups, got rows={len(rows)} unique={len(idset)}")
    if sum(int(r["run_count"]) for r in rows) != 261:
        errors.append("expanded run count differs from the reviewed 261-run scope")
    if {r["dataset_id"] for r in rows} != {"D1_INHOUSE1700_V2", "D2", "D1_COMPOSITE2200_LEGACY"}:
        errors.append("dataset coverage mismatch")
    d1 = payload["dataset_scope"].get("D1_INHOUSE1700_V2", {})
    if d1.get("manifest_sha256") != "48d90fed63ecc107b374a316effc1e5ab0d63b7c4bd9110d33ddd16e1f43146c":
        errors.append("D1 V2 manifest SHA mismatch")
    if d1.get("annotation_sha256", {}).get("test") != "883696b8e60cc901cfe92b3f009d8c60e7b8cefbb5ba9ce3c742c3720343f08f":
        errors.append("D1 V2 test annotation SHA mismatch")

    con = sqlite3.connect(DB)
    artifacts = {r[0] for r in con.execute("SELECT artifact_id FROM evidence_artifact")}
    dataset_release = con.execute(
        "SELECT manifest_artifact_id,status FROM dataset_release WHERE dataset_id='D1_INHOUSE1700_V2'"
    ).fetchone()
    if dataset_release != ("dataset-self1700-48d90fed63ec", "verified"):
        errors.append("D1 V2 dataset_release is not bound to the verified manifest")
    active_runs = {
        r[0] for r in con.execute(
            "SELECT train_run_id FROM train_run_registry WHERE status IN ('running','queued','planned')"
        )
    }
    manifest_sha = sha256(MATRIX)
    artifact_id = f"paper-route-matrix-v2-{manifest_sha[:12]}"
    for row in rows:
        rid = row["ledger_id"]
        if row.get("parent_ledger_id") and row["parent_ledger_id"] not in idset:
            errors.append(f"{rid}: missing parent route")
        for required in ("status", "config_state", "parameters", "server_plan", "output_template", "database_ids", "paper_role", "acceptance_gate"):
            if required not in row or row[required] in (None, ""):
                errors.append(f"{rid}: missing {required}")
        for path_key in ("method_config", "matrix_config", "protocol_config"):
            path = row.get(path_key) or ""
            if path.startswith(FORBIDDEN_ACTIVE_PREFIXES):
                errors.append(f"{rid}: active {path_key} points to legacy tree: {path}")
        if row["status"].startswith("COMPLETED"):
            aids = row["database_ids"]["evidence_artifact_ids"]
            if not aids and not row.get("compatibility_report") and not row["database_ids"]["result_family"]:
                errors.append(f"{rid}: completed without evidence/result/compatibility reference")
            for aid in aids:
                if aid != "multiple" and aid not in artifacts:
                    errors.append(f"{rid}: missing evidence artifact {aid}")
        if row["status"] == "RUNNING":
            tids = set(row["database_ids"]["train_run_ids"])
            if not tids or not tids <= active_runs:
                errors.append(f"{rid}: RUNNING without matching active registry rows")
        if row["dataset_id"] == "D1_INHOUSE1700_V2" and row["database_ids"]["train_run_ids"]:
            errors.append(f"{rid}: stale D1 train run IDs retained")
        if row["config_state"] in {"READY", "EXACT", "STRUCTURAL_ONLY", "PROTOCOL_READY", "EXACT_INFERENCE_ARCHIVED_TRAINING"}:
            refs = [row.get(k) for k in ("method_config", "matrix_config", "protocol_config") if row.get(k)]
            for ref in refs:
                if not (ROOT / ref).exists():
                    errors.append(f"{rid}: referenced file missing: {ref}")

    db_rows = {
        r[0]: r[1] for r in con.execute(
            "SELECT route_id,source_manifest_sha256 FROM experiment_route_matrix_v2"
        )
    }
    if set(db_rows) != idset:
        errors.append("SQLite v2 route IDs differ from YAML")
    if any(value != manifest_sha for value in db_rows.values()):
        errors.append("SQLite route rows do not bind the current YAML hash")
    artifact = con.execute(
        "SELECT path,sha256,status FROM evidence_artifact WHERE artifact_id=?", (artifact_id,)
    ).fetchone()
    if artifact != (str(MATRIX.relative_to(ROOT)), manifest_sha, "verified"):
        errors.append("route matrix evidence artifact mismatch")
    route_artifacts = con.execute(
        "SELECT COUNT(*) FROM evidence_artifact WHERE artifact_id LIKE 'paper-route-matrix-v2-%'"
    ).fetchone()[0]
    if route_artifacts != 1:
        errors.append(f"expected one authoritative route artifact, found {route_artifacts}")
    con.close()

    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        csv_ids = [r["route_id"] for r in csv.DictReader(stream)]
    if csv_ids != ids:
        errors.append("CSV route order/IDs differ from YAML")
    doc = DOC.read_text(encoding="utf-8")
    if manifest_sha not in doc or artifact_id not in doc:
        errors.append("Markdown does not bind current YAML hash/artifact")
    for token in ("D1_INHOUSE1700_V2", "D2", "Generation", "Decision", "Deployment"):
        if token not in doc:
            errors.append(f"Markdown missing required token: {token}")

    status = "PASS" if not errors else "FAIL"
    print(json.dumps({
        "status": status, "groups": len(rows),
        "expanded_runs": sum(int(r["run_count"]) for r in rows),
        "manifest_sha256": manifest_sha, "artifact_id": artifact_id,
        "errors": errors,
    }, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
