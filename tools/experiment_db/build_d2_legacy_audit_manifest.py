#!/usr/bin/env python3
"""Build a hash manifest for the one-time Dataset-2 legacy evidence audit."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "tools/experiment_db/evidence_sources/raw/d2_workstation_audit_20260812"
OUTPUT = ROOT / "tools/experiment_db/evidence_sources/d2_legacy_training_audit_20260812.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            files.append({
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "status": "archived_unpromoted",
            })
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "One-time recovery audit; these records are not paper-eligible test evidence.",
        "files": files,
        "findings": {
            "a3": {
                "independent_runs": [42, 123, 789],
                "test_status": "incomplete_uniform_test_evidence",
                "notes": "The 123 and 789 launch records explicitly set the seeds and start without a parent checkpoint; available metrics are validation results."
            },
            "a4": {
                "independent_run_count": 3,
                "claimed_seed_labels": [42, 123, 789],
                "effective_checkpoint_seeds": [335778785, 790448076, 1342286018],
                "test_status": "missing_for_two_independent_runs",
                "notes": "The launcher-level RANDOM_SEED labels were not consumed by the training framework. Treat these as independent random runs, not verified prescribed seeds 42/123/789."
            },
            "lqcr": {
                "verified_parent_links": 1,
                "test_status": "no_three_run_paired_test",
                "notes": "Only the recovered final-stage run has an explicit A4 parent checkpoint link; its available metrics are validation-only."
            }
        },
        "paper_eligibility": "blocked_until_standardized_held_out_test_and_parent_link_audit"
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(files)} files")


if __name__ == "__main__":
    main()
