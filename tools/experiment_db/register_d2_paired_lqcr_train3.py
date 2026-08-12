#!/usr/bin/env python3
"""Register the three-detector paired Dataset-2 LQCR test evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "tools/experiment_db/experiments.db"
SOURCE_FILES = (
    "tools/experiment_db/evidence_sources/d2_test_unified_5c4a95c61635.json",
    "tools/experiment_db/evidence_sources/d2_test_unified_16822b8d4654.json",
    "tools/experiment_db/evidence_sources/d2_test_unified_70df31c3d4d8.json",
)
METRICS = ("mAP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")
TENSOR_AUDIT = {
    "shared_tensor_count": 590,
    "changed_shared_tensor_count": 0,
    "child_only_tensor_count": 5,
    "child_only_scope": "final cascade localization-quality head",
    "run1_delta_sha256": "7d622010d3e27a7d5d8de407de716f96f08166bfe3f389367f82d143ecf9d23c",
    "run2_delta_sha256": "648b7c7a419e6fd82323bc0747966c4f8b3759864224984df83b29942ebe9976",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    docs = [json.loads((ROOT / path).read_text()) for path in SOURCE_FILES]
    base = sorted(docs[0]["results"], key=lambda row: row["model_id"])
    lqcr = sorted(docs[1]["results"] + docs[2]["results"],
                  key=lambda row: row["model_id"])
    if len(base) != 3 or len(lqcr) != 3:
        raise RuntimeError("expected three base and three LQCR runs")
    for index, (a, b) in enumerate(zip(base, lqcr)):
        if b["protocol"].get("parent_checkpoint_sha256") != a["protocol"]["checkpoint_sha256"]:
            raise RuntimeError(f"parent mismatch for pair {index}")
        if b["protocol"].get("training_seed") != a["protocol"].get("training_seed"):
            raise RuntimeError(f"training-seed mismatch for pair {index}")

    aggregates = {}
    for metric in METRICS:
        base_values = [row["metrics"][metric] for row in base]
        lqcr_values = [row["metrics"][metric] for row in lqcr]
        deltas = [right - left for left, right in zip(base_values, lqcr_values)]
        aggregates[metric] = {
            "baseline_values": base_values,
            "lqcr_values": lqcr_values,
            "paired_deltas": deltas,
            "baseline_mean": statistics.mean(base_values),
            "baseline_sample_std": statistics.stdev(base_values),
            "lqcr_mean": statistics.mean(lqcr_values),
            "lqcr_sample_std": statistics.stdev(lqcr_values),
            "paired_delta_mean": statistics.mean(deltas),
            "paired_delta_sample_std": statistics.stdev(deltas),
        }
    evidence = {
        "schema_version": "1.0",
        "description": "Dataset-2 paired LQCR evaluation over three independently trained detectors.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replication_unit": "independent_detector_training_run_with_paired_final_stage_intervention",
        "dataset": "D2", "split": "test", "images": 1000,
        "inference_seed": 42,
        "source_artifacts": [
            {"path": path, "sha256": sha256(ROOT / path)} for path in SOURCE_FILES
        ],
        "tensor_isolation_audit": TENSOR_AUDIT,
        "pairs": [
            {
                "training_seed": a["protocol"]["training_seed"],
                "parent_checkpoint_sha256": a["protocol"]["checkpoint_sha256"],
                "baseline_protocol_hash": a["protocol_hash"],
                "lqcr_protocol_hash": b["protocol_hash"],
            } for a, b in zip(base, lqcr)
        ],
        "aggregates": aggregates,
    }
    payload = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    relative = Path(f"tools/experiment_db/evidence_sources/d2_paired_lqcr_train3_{digest[:12]}.json")
    path = ROOT / relative
    path.write_bytes(payload)
    artifact_id = f"d2-paired-lqcr-train3-{digest[:12]}"
    protocol = json.dumps({
        "replication_unit": evidence["replication_unit"], "training_runs": 3,
        "training_seeds": [row["training_seed"] for row in evidence["pairs"]],
        "inference_seed": 42, "split": "test", "images": 1000,
        "tensor_isolation_audit": TENSOR_AUDIT,
    }, sort_keys=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute(
        """INSERT OR REPLACE INTO evidence_artifact
           (artifact_id,server,path,sha256,kind,generated_at,status,notes)
           VALUES (?, 'ross', ?, ?, 'paired_train3_test_aggregate', ?, 'verified', ?)""",
        (artifact_id, str(relative), digest, evidence["generated_at"],
         "Three independent detector runs; final-stage branch paired to each parent; shared tensors unchanged."),
    )
    for metric, values in aggregates.items():
        for variant, field, unit in (
            ("KaryoFlow", "baseline_mean", "absolute"),
            ("KaryoFlow+LQCR", "lqcr_mean", "absolute"),
        ):
            result_id = f"d2-paired-train3-{variant.lower().replace(' ','-').replace('+','plus')}-{metric.lower()}"
            baseline_id = (f"d2-paired-train3-karyoflow-{metric.lower()}"
                           if variant == "KaryoFlow+LQCR" else None)
            con.execute(
                """INSERT OR REPLACE INTO controlled_result
                   (result_id,family,variant,dataset,split,seed,metric,value,unit,
                    baseline_result_id,delta,protocol_json,artifact_id,evidence_level,
                    paper_eligible,notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result_id, "paired_lqcr_d2_train3", variant, "D2", "test",
                 "mean(335778785,790448076,1342286018)", metric,
                 values[field], unit, baseline_id,
                 values["paired_delta_mean"] if variant == "KaryoFlow+LQCR" else None,
                 protocol, artifact_id, "controlled", 1,
                 "Mean over three independent detector training runs; LQCR delta is paired within parent detector."),
            )
        for suffix, field in (("baseline_std", "baseline_sample_std"),
                              ("lqcr_std", "lqcr_sample_std"),
                              ("paired_delta_std", "paired_delta_sample_std")):
            con.execute(
                """INSERT OR REPLACE INTO controlled_result
                   (result_id,family,variant,dataset,split,seed,metric,value,unit,
                    protocol_json,artifact_id,evidence_level,paper_eligible,notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"d2-paired-train3-{suffix}-{metric.lower()}",
                 "paired_lqcr_d2_train3", suffix, "D2", "test",
                 "std(335778785,790448076,1342286018)", metric + "_std",
                 values[field], "sample_std", protocol, artifact_id,
                 "controlled", 1, "Sample SD across independent detector training runs."),
            )
    con.commit()
    print(relative, artifact_id)


if __name__ == "__main__":
    main()
